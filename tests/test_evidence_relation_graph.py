from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medrag import evidence_relation_graph as g
from medrag.evidence_graph_runtime import DecisionCache
from scripts import pilot_rag2_kev_evidence_graphs as runner


SETTINGS = {"confidence": .7, "margin": .15, "pair_neighbors": 2,
            "entities_per_proposition": 2, "max_groups": 2, "group_size": 4}


def answered(job, overrides=None):
    answers = {}
    for key, q in job["body"]["questions"].items():
        winner = (overrides or {}).get(key, "NO")
        labels = list(q["criteria"])
        scores = {label: .98 if label == winner else .02/(len(labels)-1) for label in labels}
        answers[key] = {"probabilities": scores}
    return {"answers": answers}


def fixture():
    row = {"sample_id": "q1", "dataset": "medqa", "question": "What mechanism explains X?",
           "options": {"A": "SECRET_CHOICE_A", "B": "SECRET_CHOICE_B"},
           "documents": [
               {"document_id": "d1", "rank": 1, "propositions": [{"proposition_id": "p1", "claim": "X has mechanism Y."}]},
               {"document_id": "d2", "rank": 2, "propositions": [{"proposition_id": "p2", "claim": "X has property Z."}]},
               {"document_id": "d3", "rank": 3, "propositions": [{"proposition_id": "p3", "claim": "X has mechanism Y under condition K."}]}]}
    index = {"propositions": {p["proposition_id"]: {"claim": p["claim"], "document_id": d["document_id"],
             "entities": {"e1": ["X"]}} for d in row["documents"] for p in d["propositions"]},
             "entities": {"e1": {"name": "condition X", "mention_count": 5}}}
    return row, index


class GraphTests(unittest.TestCase):
    def test_sampling_proportions_no_label_dependency(self):
        rows = [{"sample_id": str(i), "dataset": "a" if i < 80 else "b", "gold": "A"} for i in range(100)]
        chosen = g.select_sample(rows, 10, 42)
        self.assertEqual(sum(r["dataset"] == "a" for r in chosen), 8)
        for r in rows:
            r["gold"] = "B"
        self.assertEqual([r["sample_id"] for r in chosen], [r["sample_id"] for r in g.select_sample(rows, 10, 42)])
        with self.assertRaises(ValueError):
            g.select_sample(rows + [rows[0]], 10, 42)

    def test_claim_direction_inconsistency_and_none(self):
        labels = {"A_ENTAILS_B": "YES", "B_ENTAILS_A": "NO", "INCOMPATIBLE": "NO"}
        edges, status = g.claim_edges("a", "b", labels)
        self.assertEqual(edges[0]["source"], "a")
        self.assertEqual(edges[0]["type"], "ENTAILS")
        labels["B_ENTAILS_A"] = "YES"
        self.assertEqual(g.claim_edges("a", "b", labels)[0][0]["type"], "EQUIVALENT")
        labels["INCOMPATIBLE"] = "YES"
        self.assertEqual(g.claim_edges("a", "b", labels), ([], "INCONSISTENT_DECISIONS"))
        self.assertEqual(g.claim_edges("a", "b", {k: "NO" for k in labels})[1], "NONE")
        self.assertEqual(g.claim_edges("a", "b", {k: "UNCERTAIN" for k in labels})[1], "UNCERTAIN")

    def test_confidence_is_not_automatic_acceptance(self):
        self.assertEqual(g.decision({"probabilities": {"YES": .51, "NO": .48, "UNCERTAIN": .01}}), "UNCERTAIN")
        with self.assertRaises(ValueError):
            g.decision({"probabilities": {"YES": float("nan"), "NO": .2}})

    def test_pipeline_structured_and_no_answer_options_in_experts(self):
        row, index = fixture()
        ac = g.annotation_requests(index, 2)
        pairs = [g.claim_request(a, b, index) for a, b in g.pair_candidates(row, index)]
        results = {j["id"]: answered(j, {"ABOUT": "YES", "CAUSE_MECHANISM": "YES"}) for j in ac}
        results.update({j["id"]: answered(j) for j in pairs})
        graph = g.graph_for_question(row, index, ac, pairs, results, SETTINGS)
        self.assertEqual(len(graph["A_topic"]), 3)
        self.assertEqual(len(graph["C_role"]), 3)
        self.assertFalse(graph["hierarchy_available"])
        query = g.query_requests(row, graph, index, 16)
        results.update({j["id"]: answered(j, {"RELEVANT": "YES", "CAUSE_MECHANISM": "YES"}) for j in query})
        groups = g.make_groups(row, graph, query, results, SETTINGS)
        self.assertEqual(len(groups), 1)
        jobs = [g.group_request(row, group, index) for group in groups]
        results.update({j["id"]: answered(j, {"USE": "JOINT"}) for j in jobs})
        notes = g.final_notes(row, graph, jobs, results, SETTINGS)
        self.assertEqual(notes["groups"][0]["use"], "JOINT")
        self.assertEqual(len(notes["groups"][0]["members"]), 3)
        for job in ac + pairs + query + jobs:
            self.assertNotIn("SECRET_CHOICE", json.dumps(job))
        self.assertNotIn("conclusion", json.dumps(notes))

    def test_no_role_without_about_and_no_graph_fallback(self):
        row, index = fixture()
        jobs = g.annotation_requests(index, 2)
        results = {j["id"]: answered(j, {"CAUSE_MECHANISM": "YES"}) for j in jobs}
        graph = g.graph_for_question(row, index, jobs, [], results, SETTINGS)
        self.assertEqual(graph["A_topic"], [])
        self.assertEqual(graph["C_role"], [])
        self.assertEqual(g.final_notes(row, graph, [], {}, SETTINGS), {})

    def test_other_role_cannot_coexist_with_named_roles(self):
        row, index = fixture()
        jobs = g.annotation_requests(index, 2)
        results = {j["id"]: answered(j, {"ABOUT": "YES", "CAUSE_MECHANISM": "YES", "OTHER": "YES"}) for j in jobs}
        graph = g.graph_for_question(row, index, jobs, [], results, SETTINGS)
        self.assertEqual({e["type"] for e in graph["C_role"]}, {"CAUSE_MECHANISM"})
        self.assertTrue(all("role_warning" in r for r in graph["assessments"]))

    def test_equivalence_not_transitive_group_collapse(self):
        row, index = fixture()
        graph = {"A_topic": [{"source": p, "target": "e1", "type": "ABOUT"} for p in index["propositions"]],
                 "B_claim": [{"source": "p1", "target": "p2", "type": "EQUIVALENT"},
                             {"source": "p2", "target": "p3", "type": "EQUIVALENT"}],
                 "C_role": [], "topic_nodes": []}
        jobs = g.query_requests(row, graph, index, 16)
        results = {j["id"]: answered(j, {"RELEVANT": "YES"}) for j in jobs}
        groups = g.make_groups(row, graph, jobs, results, SETTINGS)
        self.assertEqual(groups[0]["proposition_ids"], ["p1", "p3"])

    def test_verified_ontology_and_cycles(self):
        ontology = {"source": "official fixture", "version": "v1", "nodes": [{"id": "t1", "label": "child"}, {"id": "t2", "label": "parent"}],
                    "mappings": [{"entity_id": "e1", "topic_id": "t1"}], "broader": [{"child": "t1", "parent": "t2"}]}
        self.assertEqual(len(g.validate_ontology(ontology, {"e1"})["mappings"]), 1)
        ontology["broader"].append({"child": "t2", "parent": "t1"})
        with self.assertRaises(ValueError):
            g.validate_ontology(ontology, {"e1"})

    def test_exact_text_noop_and_json_parser(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)
        row, _ = fixture()
        before = copy.deepcopy(row)
        prompt = g.render_prompt(Tokenizer(), row)
        self.assertEqual(prompt, g.render_prompt(Tokenizer(), row, {}))
        for p in g.propositions(row):
            self.assertIn(p["claim"], prompt)
        self.assertEqual(row, before)
        self.assertEqual(g.parse_answer('{"rationale":"fact","answer":"B"}', row["options"])["answer"], "B")
        self.assertIsNotNone(g.parse_answer('{"rationale":"unfinished', row["options"])["parse_error"])
        self.assertIsNotNone(g.parse_answer('{"rationale":"fact","answer":"D"}', row["options"])["parse_error"])
        self.assertEqual(g.parse_answer('{"rationale":"fact","answer":"free text"}', {})["answer"], "free text")

    def test_cache_reuse_and_incompatible_manifest(self):
        row, index = fixture()
        job = g.annotation_requests(index, 2)[0]
        with tempfile.TemporaryDirectory() as root:
            cache = DecisionCache(root, "contract-a")
            self.assertIsNone(cache.get(job))
            cache.put(job, answered(job)["answers"])
            self.assertEqual(cache.get(job)["answers"], answered(job)["answers"])
            with self.assertRaises(RuntimeError):
                DecisionCache(root, "contract-b").get(job)

    def test_annotation_budget_never_prunes_evidence(self):
        from types import SimpleNamespace
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)
            def encode(self, text, **kwargs):
                return list(text)
        row, _ = fixture()
        before = copy.deepcopy(row)
        notes = {"groups": [{"text": "x"*900}], "claim_relations": [{"text": "y"*900}]}
        args = SimpleNamespace(annotation_tokens=50, max_new_tokens=64, max_model_len=9999)
        result, dropped = runner.bounded_notes(args, Tokenizer(), row, notes)
        self.assertEqual(result, {})
        self.assertEqual(dropped, 2)
        self.assertEqual(row, before)

    def test_mocked_worker_to_paired_report_without_gpu(self):
        """Exercise durable graph/expert artifacts and the actual evaluator."""
        from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest
        row, index = fixture()
        ac = g.annotation_requests(index, 2)
        pairs = [g.claim_request(a, b, index) for a, b in g.pair_candidates(row, index)]

        class FakeDecisions:
            parity, batch_disabled, oom_splits = {"passed": True}, False, 0
            def __init__(self, source, adapter, base, cache, batch_rows):
                self.cache = cache
            def score(self, jobs, stage, emit):
                results = {}
                for job in jobs:
                    result = answered(job, {"ABOUT": "YES", "CAUSE_MECHANISM": "YES", "RELEVANT": "YES", "USE": "JOINT"})
                    results[job["id"]] = self.cache.put(job, result["answers"])
                return results

        with tempfile.TemporaryDirectory() as root:
            a = runner.arguments(["--run", "--output-root", root])
            out = Path(root)
            gold = {row["sample_id"]: "B"}
            contract = {"settings": SETTINGS, "cohort": {"seed": 42},
                        "evaluation_labels_sha256": digest(gold), "graph_A_variant": "ABOUT_only_no_hierarchy"}
            atomic_json(out / "manifest.json", {"contract": contract, "contract_sha256": "fixture"})
            atomic_jsonl(out / "cohort.jsonl", [row])
            atomic_json(out / "common_graph.json", index)
            atomic_json(out / "evaluation_labels.json", gold)
            atomic_json(out / "ontology.json", g.validate_ontology(None, {"e1"}))
            atomic_jsonl(out / "requests_about_role.jsonl", ac)
            atomic_jsonl(out / "requests_claim.jsonl", pairs)
            with patch("medrag.evidence_graph_runtime.LocalDecisions", FakeDecisions), patch.object(runner, "emit"):
                runner.kev_worker(a)
            expert = json.loads(runner.item_path(out, "experts", "q1").read_text())
            self.assertEqual(expert["notes"]["groups"][0]["use"], "JOINT")
            for arm, answer in (("text", "A"), ("graphs", "B")):
                atomic_json(runner.item_path(out, "predictions/" + arm, "q1"), {
                    "contract_hash": "fixture", "sample_id": "q1", "answer": answer,
                    "parse_error": None, "output_tokens": 10, "notes": expert["notes"] if arm == "graphs" else {}})
            result = runner.report(a)
            self.assertEqual(result["overall"]["transitions"], {"W->C": 1})
            self.assertEqual(result["overall"]["delta_pp"], 100)
            self.assertFalse(result["scale_authorized"])
            self.assertTrue((out / "annotation_audit.jsonl").exists())
            gold["q1"] = "A"
            atomic_json(out / "evaluation_labels.json", gold)
            with self.assertRaisesRegex(RuntimeError, "labels changed"):
                runner.report(a)


if __name__ == "__main__":
    unittest.main()

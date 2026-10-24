from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from medrag import kev_abc_quality as q
from medrag.evidence_graph_runtime import DecisionCache
from medrag.training.graph_input_data import digest
from scripts import pilot_rag2_kev_abc_quality as runner


def fixture():
    pair={"pair_id":digest(["p1","p2"]),"case_id":"P01","split":"development",
          "P1":{"id":"p1","document_id":"d1","claim":"X acts before Y."},
          "P2":{"id":"p2","document_id":"d1","claim":"Y follows X."},
          "question":"SECRET_Q","answer":"SECRET_GOLD","stratum":"SECRET_ROUTE"}
    ref={"pair_id":pair["pair_id"],"A":"SAME_TOPIC","B":"EQUIVALENT","C_yes":[],"C_uncertain":[],
         "rationale":{g:"Whole-claim paraphrase; no additional C relation." for g in ("A","B","C")}}
    return pair,ref


class QualityTests(unittest.TestCase):
    def test_jobs_independent_and_no_reference_inputs(self):
        p,_=fixture()
        for v in q.VARIANTS:
            jobs={g:q.make_job(p,g,v) for g in ("A","B","C")}
            self.assertEqual([len(jobs[g]["body"]["questions"]) for g in ("A","B","C")],[1,3,18])
            for j in jobs.values():
                body=json.dumps(j["body"])
                self.assertNotIn("SECRET",body)
                self.assertNotIn("pair_id",body)
                self.assertNotIn("rationale",body)
            self.assertEqual(len({j["id"] for j in jobs.values()}),3)
        self.assertNotEqual(q.make_job(p,"A","concise")["id"],q.make_job(p,"A","explicit")["id"])

    def test_gate_and_direction(self):
        p,_=fixture()
        labels={k:"NO" for k in q.c_tasks()};labels["RESULT_OF:21"]="YES"
        edges,status=q.graph_edges(p,"C",labels)
        self.assertEqual(edges,[dict(graph="C",source="p2",target="p1",relation="RESULT_OF",directed=True)])
        self.assertEqual(q.hard_label({"probabilities":{"YES":.65,"NO":.30,"UNCERTAIN":.05}}),"UNCERTAIN")
        self.assertEqual(status,"ASSESSED")

    def test_inconsistent_B_is_not_an_edge(self):
        p,_=fixture()
        labels={"P1_ENTAILS_P2":"YES","P2_ENTAILS_P1":"NO","CONTRADICTS":"YES"}
        edges,status=q.graph_edges(p,"B",labels)
        self.assertEqual(edges,[]);self.assertEqual(status,"INCONSISTENT")

    def test_perfect_and_majority_negative_metrics(self):
        p,r=fixture()
        r["C_yes"]=["EXPLAINS:12"]
        preds={p["pair_id"]:q.reference_decisions(r)}
        perfect,errors=q.evaluate([p],{"P01":r},preds)
        self.assertFalse(errors)
        self.assertEqual(perfect["selection_score"],1.0)
        negative,_=q.evaluate([p],{"P01":r},runner.baseline_predictions([p]))
        self.assertEqual(negative["C"]["agreement"],17/18)
        self.assertEqual(negative["C"]["positive_relations"]["recall"],0)
        self.assertEqual(negative["C"]["quality_score"],0)

    def test_uncertain_reference_excluded_explicitly(self):
        p,r=fixture();r["B"]="UNCERTAIN";r["C_uncertain"]=["PRECEDES:12"]
        result,_=q.evaluate([p],{"P01":r},runner.baseline_predictions([p]))
        self.assertEqual(result["B"]["evaluated_decisions"],0)
        self.assertEqual(result["B"]["uncertain_reference_decisions"],3)
        self.assertEqual(result["C"]["evaluated_decisions"],17)

    def test_reference_doc_leakage(self):
        p,r=fixture();p2=copy.deepcopy(p);p2.update(case_id="P02",split="confirmation",pair_id="different")
        r2=copy.deepcopy(r);r2["pair_id"]="different"
        with self.assertRaises(ValueError): q.validate_references([p,p2],{"P01":r,"P02":r2})

    def test_cached_reporting_and_frozen_choice(self):
        p,r=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=DecisionCache(root/"cache","test")
            for v in q.VARIANTS:
                for g in ("A","B","C"):
                    job=q.make_job(p,g,v)
                    labels=q.reference_decisions(r)[g]
                    answers={}
                    for k,question in job["body"]["questions"].items():
                        options=list(question["criteria"])
                        winner=labels[k]
                        answers[k]={"choice":winner,"probabilities":{x:.98 if x==winner else .02/(len(options)-1) for x in options}}
                    cache.put(job,answers)
            a=SimpleNamespace(output_root=root,confidence=.7,margin=.15)
            report=runner.report(a,[p],{"P01":r},q.VARIANTS,cache,"development","test")
            self.assertEqual(report["selected_variant"],"concise")
            self.assertFalse(report["full_annotation_authorized"])
            self.assertEqual(json.loads((root/"selection.json").read_text())["contract_hash"],"test")
            runner.report(a,[p],{"P01":r},q.VARIANTS,cache,"development","test")
            self.assertTrue((root/"development_predictions.jsonl").exists())
            self.assertIn("RESULT_OF",report["C_relations_without_positive_examples"])

    def test_real_adjudicated_fixture_not_kevs_own_labels(self):
        root=runner.DEFAULT_ROOT
        if not (root/"cohort.json").exists(): self.skipTest("local derived pilot data not installed")
        cohort=json.loads((root/"cohort.json").read_text());reference=json.loads((root/"reference.json").read_text())
        self.assertFalse(reference["kev_predictions_seen"])
        q.validate_references(cohort["pairs"],reference["references"])
        jobs=[q.make_job(p,g,v) for p in cohort["pairs"] for g in ("A","B","C") for v in q.VARIANTS]
        self.assertEqual(digest(jobs),cohort["prompt_hash_before_reference"])


if __name__=="__main__": unittest.main()

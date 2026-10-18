import copy
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from medrag.training.graph_input_data import sha_file
from medrag.training.proposition_kg_attention_data import (
    build_rows, inspect_inputs, make_layout, render_prompt, shuffle_pairs,
)


class CharacterTokenizer:
    """Small deterministic tokenizer; each trailing ' A' is one answer token."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt
        return "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"

    def __call__(self, text, **kwargs):
        answer = len(text) >= 2 and text[-2] == " " and text[-1] in "ABCD"
        length = len(text) - 2 if answer else len(text)
        ids = [ord(c) for c in text[:length]]
        offsets = [(i, i + 1) for i in range(length)]
        if answer:
            ids.append(1000 + ord(text[-1]))
            offsets.append((length, len(text)))
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


def fixture_view():
    return {"sample_id": "medqa:fixture", "dataset": "medqa", "official_split": "train",
            "split": "train", "question": "Which approach is suitable?",
            "options": {"A": "First", "B": "Second"}, "document_ids": ["docB", "docA"],
            "usable_proposition_ids": ["p_a", "p_z", "p_b"],
            "binary_edge_ids": ["edge_a", "edge_z"]}


def fixture_props():
    return [{"proposition_id": "p_z", "document_id": "docA", "claim": "First fact in doc A."},
            {"proposition_id": "p_a", "document_id": "docA", "claim": "Second fact in doc A."},
            {"proposition_id": "p_b", "document_id": "docB", "claim": "Unconnected fact in doc B."}]


class DataLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_exact_text_stem_only_and_stable_answer_tokens(self):
        view, props = fixture_view(), fixture_props()
        row = make_layout(self.tokenizer, view, "B", props, {})
        text = "".join(chr(i) for i in row["input_ids"])
        for prop, (a, b) in zip(props, row["proposition_spans"]):
            self.assertEqual(text[a:b], prop["claim"])
        a, b = row["question_span"]
        self.assertEqual(text[a:b], view["question"])
        self.assertNotIn("Options:", text[a:b])
        self.assertEqual(row["choice_token_ids"], [1065, 1066])
        self.assertEqual(row["gold_index"], 1)
        self.assertTrue(text.endswith("<assistant>Answer:"))
        # Gold is never included; changing label changes only the loss index.
        other = make_layout(self.tokenizer, view, "A", props, {})
        for field in ("input_ids", "proposition_spans", "question_span", "pairs", "pair_roles"):
            self.assertEqual(other[field], row[field])

    def test_exact_endpoint_ids_and_receiver_sender_role_order(self):
        props = fixture_props()
        roles = {"p_z": {"UMLS:E": {0}, "UMLS:M": {1}},
                 "p_a": {"UMLS:M": {0}, "UMLS:N": {1}},
                 "p_b": {"MENTION:distinct_identity": {0}}}
        row = make_layout(self.tokenizer, fixture_view(), "A", props, roles)
        self.assertEqual(row["pairs"], [[1, 0]])
        self.assertEqual(row["pair_roles"], [[0.0, 1.0, 0.0, 0.0]])
        self.assertEqual(row["graph_stats"]["isolated_propositions"], 1)

    def test_whole_trailing_proposition_truncation_and_link_pruning(self):
        props, view = fixture_props(), fixture_view()
        short = make_layout(self.tokenizer, view, "A", props[:1], {})
        roles = {p["proposition_id"]: {"E": {0}} for p in props}
        row = make_layout(self.tokenizer, view, "A", props, roles,
                          max_prompt_tokens=len(short["input_ids"]))
        self.assertEqual(row["proposition_ids"], ["p_z"])
        self.assertEqual(row["pairs"], [])
        self.assertEqual(row["truncation"]["removed_propositions"], 2)
        self.assertEqual(row["question_span"], short["question_span"])

    def test_empty_evidence_retained(self):
        row = make_layout(self.tokenizer, fixture_view(), "A", [], {})
        self.assertEqual(row["proposition_spans"], [])
        self.assertEqual(row["pairs"], [])
        self.assertEqual(row["graph_stats"]["propositions"], 0)
        self.assertGreater(row["question_span"][1], row["question_span"][0])

    def test_stem_never_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "Stem/options alone exceed"):
            make_layout(self.tokenizer, fixture_view(), "A", [], {}, max_prompt_tokens=20)

    def test_option_changes_do_not_change_feature_tokens(self):
        view, props = fixture_view(), fixture_props()
        first = make_layout(self.tokenizer, view, "A", props, {})
        view["options"] = {"A": "Completely unrelated long alternative", "B": "Another alternative"}
        second = make_layout(self.tokenizer, view, "A", props, {})
        for span in first["proposition_spans"] + [first["question_span"]]:
            a, b = span
            self.assertEqual(first["input_ids"][a:b], second["input_ids"][a:b])

    def test_source_materialization_order_and_empty_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            calls = []
            rows, summary = build_rows(root, self.tokenizer,
                                       progress_callback=lambda *args: calls.append(args))
            self.assertEqual(rows[0]["proposition_ids"], ["p_b", "p_z", "p_a"])
            self.assertEqual(rows[0]["pairs"], [[2, 1]])
            self.assertEqual(rows[0]["pair_roles"], [[0.0, 1.0, 0.0, 0.0]])
            self.assertEqual(rows[1]["proposition_spans"], [])
            self.assertEqual(summary["split_counts"], {"train": 1, "validation": 1})
            self.assertEqual(summary["empty_evidence_questions"], 1)
            self.assertEqual(calls[-1], (2, 2, "tokenize"))

    def test_materialization_rejects_changed_question_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            with (root / "question_views.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_rows(root, self.tokenizer)

    def test_materialization_rejects_out_of_top8_graph_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            with sqlite3.connect(root / "kg/graph.sqlite3") as con:
                con.execute("UPDATE binary_relation SET document_id='outside' WHERE edge_id='edge_a'")
            with self.assertRaisesRegex(ValueError, "outside question evidence"):
                build_rows(root, self.tokenizer)

    def test_preflight_missing_sources_fails_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                inspect_inputs(directory, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    @staticmethod
    def _write_fixture(root):
        for name in ("cohort", "usable", "kg"):
            (root / name).mkdir()
        views = [fixture_view(), {**fixture_view(), "sample_id": "medqa:empty",
                 "split": "validation", "question": "Different empty-evidence question?",
                 "document_ids": [], "usable_proposition_ids": [], "binary_edge_ids": []}]
        def write_jsonl(path, values):
            path.write_text("".join(json.dumps(x) + "\n" for x in values))
        write_jsonl(root / "question_views.jsonl", views)
        write_jsonl(root / "cohort/labels.jsonl", [{"sample_id": x["sample_id"], "answer": "A"} for x in views])
        write_jsonl(root / "usable/usable_propositions.jsonl", fixture_props())
        (root / "COMPLETE.json").write_text(json.dumps({"questions": 2,
            "split_counts": {"train": 1, "validation": 1},
            "question_views_sha256": sha_file(root / "question_views.jsonl")}))
        with sqlite3.connect(root / "kg/graph.sqlite3") as con:
            con.execute("CREATE TABLE binary_relation(edge_id TEXT PRIMARY KEY, proposition_id TEXT, document_id TEXT, subject_entity_id TEXT, object_entity_id TEXT)")
            con.executemany("INSERT INTO binary_relation VALUES(?,?,?,?,?)", [
                ("edge_z", "p_z", "docA", "UMLS:E", "UMLS:M"),
                ("edge_a", "p_a", "docA", "UMLS:M", "UMLS:N")])


class ShuffleTests(unittest.TestCase):
    def test_shuffle_preserves_causal_directed_degrees_and_role_multiset(self):
        pairs = [[4, 0], [5, 1], [6, 2], [7, 3], [7, 0]]
        row = {"sample_id": "fixture", "pairs": pairs,
               "pair_roles": [[float(i % 4 == j) for j in range(4)] for i in range(len(pairs))],
               "input_ids": [1, 2, 3], "gold_index": 0}
        original = copy.deepcopy(row)
        shuffled = shuffle_pairs(row, 42)
        self.assertEqual(row, original)
        self.assertEqual(shuffled, shuffle_pairs(row, 42))
        self.assertEqual(Counter(a for a, _ in pairs), Counter(a for a, _ in shuffled["pairs"]))
        self.assertEqual(Counter(b for _, b in pairs), Counter(b for _, b in shuffled["pairs"]))
        self.assertTrue(all(a > b for a, b in shuffled["pairs"]))
        self.assertEqual(len(set(map(tuple, shuffled["pairs"]))), len(pairs))
        self.assertEqual(Counter(map(tuple, row["pair_roles"])), Counter(map(tuple, shuffled["pair_roles"])))
        self.assertEqual(shuffled["input_ids"], row["input_ids"])
        self.assertGreater(shuffled["shuffle_statistics"]["changed_fraction"], 0)

    def test_unshufflable_graph_explicitly_reported(self):
        row = {"sample_id": "one", "pairs": [[1, 0]], "pair_roles": [[1., 0., 0., 0.]]}
        result = shuffle_pairs(row, 42)
        self.assertEqual(result["pairs"], row["pairs"])
        self.assertTrue(result["shuffle_statistics"]["unshufflable_or_unchanged"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/extract_rag2_statement_graph_semantic_frames_v4.py"
spec = importlib.util.spec_from_file_location("semantic_frames_v4", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.path.insert(0, str(ROOT / "scripts"))
spec.loader.exec_module(module)


def sjogren_row():
    return {
        "proposition_id": "P1", "source": "textbooks",
        "claim": "The main symptoms of Sjogren syndrome are xerostomia.",
        "mentions": [
            {"mention_id": "N1", "surface": "main symptoms", "start": 4, "end": 17, "kind": "finding_or_outcome", "origin": "candidate"},
            {"mention_id": "N2", "surface": "Sjogren syndrome", "start": 21, "end": 37, "kind": "condition", "origin": "candidate"},
            {"mention_id": "N3", "surface": "xerostomia", "start": 42, "end": 52, "kind": "finding_or_outcome", "origin": "candidate"},
        ],
    }


class SemanticFramesV4Test(unittest.TestCase):
    def test_semantic_roles_can_ignore_grammatical_subject(self):
        raw = '''{"status":"frames","frames":[{"asserted_fact":"Sjogren syndrome has xerostomia as a symptom.","subject_mention_id":"N2","subject_text":"Sjogren syndrome","subject_kind":"condition","relation_type":"HAS_SIGN_OR_SYMPTOM","object_mention_id":"N3","object_text":"xerostomia","object_kind":"finding_or_outcome","qualifiers":[],"evidence_spans":["The main symptoms of Sjogren syndrome are xerostomia"],"negated":false,"uncertain":false,"modality":"asserted","frame_status":"grounded"}]}'''
        value = module.validate_output(sjogren_row(), raw)
        frame = value["semantic_frames"][0]
        self.assertTrue(frame["structurally_valid"])
        self.assertEqual(frame["subject_mention_id"], "N2")
        self.assertEqual(frame["object_mention_id"], "N3")
        self.assertEqual(frame["type_compatibility"], "compatible")

    def test_unseen_argument_can_be_added_only_from_exact_span(self):
        row = sjogren_row()
        mid, note = module.resolve_argument(row, None, "The main symptoms of Sjogren syndrome", "finding_or_outcome", [], "subject")
        self.assertIsNotNone(mid)
        self.assertEqual(note, "subject:mention_added_from_exact_text")

    def test_non_exact_evidence_rejects_frame(self):
        raw = '''{"status":"frames","frames":[{"asserted_fact":"Sjogren syndrome has xerostomia.","subject_mention_id":"N2","subject_text":"Sjogren syndrome","subject_kind":"condition","relation_type":"HAS_SIGN_OR_SYMPTOM","object_mention_id":"N3","object_text":"xerostomia","object_kind":"finding_or_outcome","qualifiers":[],"evidence_spans":["not in source"],"negated":false,"uncertain":false,"modality":"asserted","frame_status":"grounded"}]}'''
        value = module.validate_output(sjogren_row(), raw)
        self.assertFalse(value["semantic_frames"][0]["structurally_valid"])

    def test_review_selection_is_balanced_and_deterministic(self):
        rows = [{"proposition_id": f"{source}-{i}", "source": source} for source in ("cpg", "pmc", "pubmed", "textbooks") for i in range(10)]
        first = module.review_ids(rows, 3, 7)
        second = module.review_ids(rows, 3, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual({source: sum(x.startswith(source + "-") for x in first) for source in ("cpg", "pmc", "pubmed", "textbooks")}, {"cpg": 3, "pmc": 3, "pubmed": 3, "textbooks": 3})

    def test_prompt_does_not_expose_legacy_raw_relation(self):
        row = {**sjogren_row(), "statements": [{"predicate_text": "legacy wrong relation"}], "relation_raw_output": "legacy"}
        prompt = module.semantic_frame_prompt(row)
        self.assertNotIn("legacy wrong relation", prompt)
        self.assertNotIn("relation_raw_output", prompt)


if __name__ == "__main__":
    unittest.main()

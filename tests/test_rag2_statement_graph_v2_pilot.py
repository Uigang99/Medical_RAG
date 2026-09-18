import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rag2_statement_graph_v2_pilot.py"
spec = importlib.util.spec_from_file_location("statement_graph_v2", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.path.insert(0, str(ROOT / "scripts"))
spec.loader.exec_module(module)


class StatementGraphV2Test(unittest.TestCase):
    def test_contract_hash_ignores_git_commit_but_keeps_code_content(self):
        first = {"run_version": "v", "code": {"script": {"sha256": "abc"}, "git_commit": "one"}}
        second = {"run_version": "v", "code": {"script": {"sha256": "abc"}, "git_commit": "two"}}
        changed = {"run_version": "v", "code": {"script": {"sha256": "xyz"}, "git_commit": "two"}}
        self.assertEqual(module.contract_payload(first), module.contract_payload(second))
        self.assertNotEqual(module.contract_payload(first), module.contract_payload(changed))

    def test_preflight_only_legacy_manifest_is_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = {"run_version": "v", "code": {"script": {"mtime_ns": 1}, "git_commit": "one"}, "seed": 7}
            previous["contract_sha256"] = "legacy"
            (root / "manifest.json").write_text(json.dumps(previous), encoding="utf-8")
            current = {"run_version": "v", "code": {"script": {"sha256": "abc"}, "git_commit": "two"}, "seed": 7}
            current["contract_sha256"] = module.sha256_text(module.canonical_json(module.contract_payload(current)))
            module.ensure_manifest(SimpleNamespace(output_root=root), current)
            migrated = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(migrated["contract_sha256"], current["contract_sha256"])

    def test_clean_mentions_collapses_article_overlap(self):
        row = {
            "nodes": [
                {"node_id": "P::N1", "surface": "The pathway", "start": 0, "end": 11, "kind": "biomedical_concept", "umls_status": "linked_ambiguous"},
                {"node_id": "P::N2", "surface": "pathway", "start": 4, "end": 11, "kind": "biomedical_concept", "umls_status": "linked_ambiguous"},
            ]
        }
        value = module.clean_mentions(row)
        self.assertEqual(len(value["mentions"]), 1)
        self.assertEqual(value["node_cleaning"]["collapsed_nodes"], 1)

    def test_relation_validation_repairs_offsets_without_model_positions(self):
        row = {
            "proposition_id": "P", "claim": "Skeletal fluorosis causes joint stiffness.",
            "mentions": [
                {"mention_id": "P::N1", "surface": "Skeletal fluorosis", "kind": "condition"},
                {"mention_id": "P::N2", "surface": "joint stiffness", "kind": "finding_or_outcome"},
            ],
        }
        raw = '{"status":"extracted","statements":[{"subject_mention_id":"P::N1","subject_text":"Skeletal fluorosis","subject_kind":"condition","predicate_text":"causes","object_mention_id":"P::N2","object_text":"joint stiffness","object_kind":"finding_or_outcome","qualifiers":[],"negated":false,"uncertain":false,"modality":"asserted"}]}'
        value = module.validate_relation_output(row, raw)
        self.assertTrue(value["statements"][0]["structurally_valid"])
        self.assertEqual(value["statements"][0]["predicate_spans"][0]["start"], 19)

    def test_normalization_reverses_causal_direction(self):
        row = {
            "mentions": [],
            "statements": [{
                "statement_id": "P::S1", "raw_subject_mention_id": "A", "raw_object_mention_id": "B",
                "predicate_text": "is a consequence of", "predicate_spans": [], "qualifiers": [],
                "negated": False, "uncertain": False, "modality": "asserted",
                "structurally_valid": True, "validation_errors": [], "repairs": [],
            }],
        }
        raw = '{"mappings":[{"statement_id":"P::S1","relation_type":"CAUSES","direction":"reversed","mapping_status":"exact","confidence":1.0,"explanation":"cause to effect"}]}'
        value = module.validate_normalization_output(row, raw)
        statement = value["statements"][0]
        self.assertEqual(statement["normalized_subject_mention_id"], "B")
        self.assertEqual(statement["normalized_object_mention_id"], "A")

    def test_type_incompatible_umls_link_is_rejected(self):
        mention = {
            "kind": "procedure_or_intervention", "surface": "prevent", "umls_status": "linked_confident",
            "umls_cui": "C000", "umls_name": "PREVENT product", "umls_score": 0.99,
            "umls_candidates": [{"score": 0.99, "category": "DrugOrChemical"}],
        }
        args = SimpleNamespace(canonical_umls_threshold=0.9, canonical_umls_min_margin=0.05, canonical_lexical_overlap=0.6)
        self.assertFalse(module.umls_compatible(mention, args))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rag2_statement_graph_canonical_direction_v3.py"
spec = importlib.util.spec_from_file_location("canonical_direction_v3", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.path.insert(0, str(ROOT / "scripts"))
spec.loader.exec_module(module)


def base_row():
    return {
        "claim": "A causes B.",
        "mentions": [
            {"mention_id": "N1", "surface": "A", "kind": "condition"},
            {"mention_id": "N2", "surface": "B", "kind": "finding_or_outcome"},
        ],
        "statements": [{
            "statement_id": "S1", "raw_subject_mention_id": "N1", "raw_object_mention_id": "N2",
            "predicate_text": "causes", "structurally_valid": True, "negated": False,
            "uncertain": False, "modality": "asserted",
        }],
    }


class CanonicalDirectionV3Test(unittest.TestCase):
    def test_direct_canonical_ids_are_retained(self):
        raw = '{"mappings":[{"statement_id":"S1","relation_type":"CAUSES","canonical_subject_mention_id":"N1","canonical_object_mention_id":"N2","status":"normalized","confidence":1.0,"explanation":"active cause"}]}'
        value = module.validate_output(base_row(), raw)
        statement = value["statements"][0]
        self.assertEqual(statement["canonical_subject_mention_id_v3"], "N1")
        self.assertEqual(statement["canonical_object_mention_id_v3"], "N2")
        self.assertEqual(statement["orientation_status_v3"], "normalized")

    def test_non_raw_argument_is_rejected(self):
        raw = '{"mappings":[{"statement_id":"S1","relation_type":"CAUSES","canonical_subject_mention_id":"N3","canonical_object_mention_id":"N2","status":"normalized","confidence":1.0,"explanation":"bad"}]}'
        value = module.validate_output(base_row(), raw)
        self.assertEqual(value["statements"][0]["orientation_status_v3"], "uncertain")
        self.assertIn("mapping_0:canonical_arguments_not_raw_pair", value["v3_errors"])

    def test_other_is_never_a_resolved_directed_edge(self):
        raw = '{"mappings":[{"statement_id":"S1","relation_type":"OTHER","canonical_subject_mention_id":"N1","canonical_object_mention_id":"N2","status":"normalized","confidence":1.0,"explanation":"no ontology fit"}]}'
        value = module.validate_output(base_row(), raw)
        self.assertEqual(value["statements"][0]["orientation_status_v3"], "uncertain")

    def test_clear_direction_rules_distinguish_active_and_passive(self):
        self.assertEqual(module.clear_direction_rule("can cause"), ("active_cause", "kept"))
        self.assertEqual(module.clear_direction_rule("is caused by"), ("passive_cause", "swapped"))
        self.assertEqual(module.clear_direction_rule("results from"), ("passive_cause", "swapped"))


if __name__ == "__main__":
    unittest.main()

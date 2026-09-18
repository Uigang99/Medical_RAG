import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_rag2_statement_graph_relation_oracle.py"
spec = importlib.util.spec_from_file_location("oracle_analysis", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class RelationOracleAnalysisTest(unittest.TestCase):
    def test_auxiliary_relation_is_excluded_from_core_false_positives(self):
        rows = [{
            "candidate_pairs": [{
                "pair_id": "P1",
                "allowed_choices": ["NO_RELATION", "L2R:APPLIES_TO_POPULATION"],
                "gold_relations": [],
            }],
            "pair_decisions": [{
                "pair_id": "P1",
                "choice": "L2R:APPLIES_TO_POPULATION",
                "classification_valid": True,
            }],
        }]
        value = module.score(rows, choice_only=False, shared={"TREATS"})
        self.assertEqual(value["exact_core_relation"]["predicted"], 0)

    def test_choice_only_keeps_allowed_choice_despite_evidence_error(self):
        rows = [{
            "candidate_pairs": [{
                "pair_id": "P1",
                "allowed_choices": ["NO_RELATION", "L2R:TREATS"],
                "gold_relations": [{"choice": "L2R:TREATS", "relation_type": "TREATS"}],
            }],
            "pair_decisions": [{
                "pair_id": "P1",
                "choice": "L2R:TREATS",
                "classification_valid": False,
                "validation_errors": ["nonexact_evidence_span"],
            }],
        }]
        validated = module.score(rows, choice_only=False, shared={"TREATS"})
        choice_only = module.score(rows, choice_only=True, shared={"TREATS"})
        self.assertEqual(validated["exact_core_relation"]["true_positive"], 0)
        self.assertEqual(choice_only["exact_core_relation"]["true_positive"], 1)


if __name__ == "__main__":
    unittest.main()

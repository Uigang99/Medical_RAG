import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rag2_statement_graph_relation_oracle.py"
spec = importlib.util.spec_from_file_location("relation_oracle", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.path.insert(0, str(ROOT / "scripts"))
spec.loader.exec_module(module)


def entity(eid, surface, entity_type, start):
    return {
        "entity_id": eid,
        "surface": surface,
        "canonical_name": surface.casefold(),
        "concept_id": f"LOCAL::{entity_type}::{surface.casefold()}",
        "entity_type": entity_type,
        "start": start,
        "end": start + len(surface),
    }


class RelationOracleTest(unittest.TestCase):
    def test_gold_positive_and_type_compatible_negative_are_both_retained(self):
        rows = [{
            "proposition_id": "P1",
            "source": "textbooks",
            "claim": "Aspirin treats disease but is not related to fever.",
            "entities": [
                entity("E1", "Aspirin", "drug_or_chemical", 0),
                entity("E2", "disease", "condition", 15),
                entity("E3", "fever", "condition", 45),
            ],
            "relations": [{
                "relation_id": "R1",
                "subject_entity_id": "E1",
                "object_entity_id": "E2",
                "relation_type": "TREATS",
                "polarity": "asserted",
                "evidence": "Aspirin treats disease",
            }],
        }]
        values, summary = module.materialize_oracle_pairs(rows, {"P1": "development"})
        self.assertEqual(summary["positive_pairs"], 1)
        self.assertGreaterEqual(summary["hard_negative_pairs"], 1)
        positive = next(pair for pair in values[0]["candidate_pairs"] if pair["gold_positive"])
        self.assertEqual(positive["gold_relations"][0]["choice"], "L2R:TREATS")
        self.assertTrue(positive["gold_relations"][0]["schema_supported"])

    def test_schema_unsupported_gold_pair_is_not_silently_dropped(self):
        rows = [{
            "proposition_id": "P2",
            "source": "pmc",
            "claim": "Group A was compared with Group B.",
            "entities": [
                entity("E1", "Group A", "population", 0),
                entity("E2", "Group B", "population", 26),
            ],
            "relations": [{
                "relation_id": "R2",
                "subject_entity_id": "E1",
                "object_entity_id": "E2",
                "relation_type": "COMPARED_WITH",
                "polarity": "asserted",
                "evidence": "Group A was compared with Group B",
            }],
        }]
        values, summary = module.materialize_oracle_pairs(rows, {"P2": "validation"})
        self.assertEqual(len(values[0]["candidate_pairs"]), 1)
        self.assertEqual(summary["schema_unsupported_gold_relations"], 1)
        self.assertEqual(values[0]["candidate_pairs"][0]["allowed_choices"], ["NO_RELATION", "AMBIGUOUS"])

    def test_metrics_separate_negative_accuracy_from_exact_relation(self):
        records = [
            {
                "gold_relations": [{"relation_type": "TREATS"}],
                "gold_positive": True,
                "schema_supported_positive": True,
                "prediction": "L2R:TREATS",
                "predicted_positive": True,
                "prediction_valid": True,
                "exact_directed_relation": True,
                "supported_relation_type_correct": True,
                "polarity_correct_given_exact": True,
            },
            {
                "gold_relations": [],
                "gold_positive": False,
                "schema_supported_positive": False,
                "prediction": "NO_RELATION",
                "predicted_positive": False,
                "prediction_valid": True,
                "exact_directed_relation": False,
                "supported_relation_type_correct": False,
                "polarity_correct_given_exact": False,
            },
        ]
        value = module.metrics(records)
        self.assertEqual(value["exact_directed_relation"]["f1"], 1.0)
        self.assertEqual(value["hard_negative_no_relation_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

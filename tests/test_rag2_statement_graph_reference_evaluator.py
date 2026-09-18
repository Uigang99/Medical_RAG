import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rag2_statement_graph_against_reference.py"
spec = importlib.util.spec_from_file_location("statement_graph_reference_eval", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def mention(start: int, end: int, kind: str) -> dict:
    return {"start": start, "end": end, "kind": kind}


def gold_entity(entity_id: str, start: int, end: int, entity_type: str) -> dict:
    return {
        "entity_id": entity_id,
        "start": start,
        "end": end,
        "entity_type": entity_type,
    }


class StatementGraphReferenceEvaluatorTest(unittest.TestCase):
    def test_relaxed_entity_alignment_is_one_to_one_and_type_aware(self):
        gold = [
            gold_entity("g1", 0, 10, "condition"),
            gold_entity("g2", 20, 30, "drug_or_chemical"),
        ]
        pred = [
            mention(0, 8, "condition"),
            mention(0, 10, "finding_or_outcome"),
            mention(20, 30, "drug_or_chemical"),
        ]
        matches, missed_gold, unmatched_pred = module.align_entities(
            gold, pred, exact=False, require_type=True
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(missed_gold, [])
        self.assertEqual(len(unmatched_pred), 1)

    def test_directed_relation_does_not_accept_reversed_endpoints(self):
        entities = {
            "drug": gold_entity("drug", 0, 7, "drug_or_chemical"),
            "disease": gold_entity("disease", 15, 22, "condition"),
        }
        gold = {
            "subject_entity_id": "drug",
            "object_entity_id": "disease",
            "relation_type": "TREATS",
            "polarity": "asserted",
        }
        mentions = {
            "m_drug": mention(0, 7, "drug_or_chemical"),
            "m_disease": mention(15, 22, "condition"),
        }
        reversed_prediction = {
            "subject_mention_id": "m_disease",
            "object_mention_id": "m_drug",
            "relation_type": "TREATS",
        }
        self.assertFalse(module.relation_equal(
            gold,
            reversed_prediction,
            entities,
            mentions,
            exact_endpoint=True,
            typed_endpoint=True,
            require_relation=True,
            require_polarity=False,
            aliases=False,
        ))

    def test_symmetric_relation_accepts_reversed_endpoints(self):
        entities = {
            "a": gold_entity("a", 0, 4, "condition"),
            "b": gold_entity("b", 10, 14, "condition"),
        }
        gold = {
            "subject_entity_id": "a",
            "object_entity_id": "b",
            "relation_type": "ASSOCIATED_WITH",
            "polarity": "asserted",
        }
        mentions = {
            "ma": mention(0, 4, "condition"),
            "mb": mention(10, 14, "condition"),
        }
        reversed_prediction = {
            "subject_mention_id": "mb",
            "object_mention_id": "ma",
            "relation_type": "ASSOCIATED_WITH",
        }
        self.assertTrue(module.relation_equal(
            gold,
            reversed_prediction,
            entities,
            mentions,
            exact_endpoint=True,
            typed_endpoint=True,
            require_relation=True,
            require_polarity=False,
            aliases=False,
        ))


if __name__ == "__main__":
    unittest.main()

"""Targeted parser and comparison regression checks for relation pilot."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_rag2_train_relation_benchmark import match, score
from filter_rag2_kg_grounded_propositions import coverage
from run_rag2_gemma_relation_pilot import parse_entity, parse_relation


class RelationPilotTest(unittest.TestCase):
    def test_entity_requires_exact_specific_span(self):
        entities, errors = parse_entity(
            '{"entities":[{"span":"patient","type":"population"},'
            '{"span":"Aspirin","type":"drug_or_chemical"}]}',
            "Aspirin helped the patient.")
        self.assertEqual([e["span"] for e in entities], ["Aspirin"])
        self.assertIn("generic_span", errors)

    def test_relation_requires_existing_endpoints_and_predicate(self):
        claim = "Aspirin reduced fever."
        entities, _ = parse_entity(
            '{"entities":[{"span":"Aspirin","type":"drug_or_chemical"},'
            '{"span":"fever","type":"finding_or_outcome"}]}', claim)
        relations, missing, errors = parse_relation(
            '{"relations":[{"subject_id":"E1","predicate":"reduced",'
            '"object_id":"E2","negated":false,"uncertain":false,"qualifiers":[]},'
            '{"subject_id":"E3","predicate":"cured","object_id":"E2",'
            '"negated":false,"uncertain":false,"qualifiers":[]}],'
            '"missing_endpoints":[]}', claim, entities)
        self.assertEqual(len(relations), 1)
        self.assertEqual(missing, [])
        self.assertIn("invalid_endpoint", errors)

    def test_direction_sensitive_endpoint_pair(self):
        gold = {"entities": [
            {"entity_id": "G1", "surface": "Aspirin"},
            {"entity_id": "G2", "surface": "fever"}],
            "relations": [{"subject_entity_id": "G1", "object_entity_id": "G2"}]}
        row = {"models": {"model": {"entities": [
            {"id": "E1", "span": "Aspirin"}, {"id": "E2", "span": "fever"}],
            "relations": [{"subject_id": "E2", "object_id": "E1"}]}}}
        self.assertEqual(score(row, gold, "model", False)["matched_oriented_endpoint_pairs"], 0)
        self.assertTrue(match("heparin-induced thrombocytopenia", "Heparin Induced Thrombocytopenia"))

    def test_lexical_tier_excludes_unsupported_table_claim(self):
        score_value, count = coverage(
            "Certolizumab is a drug used in gout available under the name Cimzia.",
            "Certolizumab Cimzia")
        self.assertGreaterEqual(count, 3)
        self.assertLess(score_value, 0.8)


if __name__ == "__main__":
    unittest.main()

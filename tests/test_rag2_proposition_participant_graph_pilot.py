import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag2_proposition_participant_graph_pilot.py"
SPEC = importlib.util.spec_from_file_location("participant_graph", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParticipantGraphPilotTests(unittest.TestCase):
    def test_extractable_prediction(self):
        row = {"claim": "Aspirin reduces platelet aggregation.", "candidate_hints": [], "gold_entities": []}
        value = MODULE.parse_prediction(row, '{"u":"Y","e":["Aspirin","platelet aggregation"]}')
        self.assertTrue(value["predicted_use"])
        self.assertTrue(value["valid_output"])
        self.assertEqual([item["surface"] for item in value["predicted_entities"]], ["Aspirin", "platelet aggregation"])

    def test_nonextractable_span_is_rejected(self):
        row = {"claim": "Aspirin reduces platelet aggregation.", "candidate_hints": [], "gold_entities": []}
        value = MODULE.parse_prediction(row, '{"u":"Y","e":["warfarin"]}')
        self.assertFalse(value["valid_output"])
        self.assertIn("non_extractable_span", value["output_errors"])

    def test_boundary_tolerant_bipartite_match(self):
        predicted = [{"surface": "high-dose busulfan"}, {"surface": "cyclophosphamide"}]
        gold = [{"surface": "busulfan"}, {"surface": "cyclophosphamide"}]
        matched, false_positive, false_negative = MODULE.maximum_match(predicted, gold, relaxed=True)
        self.assertEqual(matched, 2)
        self.assertEqual(false_positive, [])
        self.assertEqual(false_negative, [])


if __name__ == "__main__":
    unittest.main()

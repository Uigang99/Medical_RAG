import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rag2_statement_graph_relation_learnability.py"
spec = importlib.util.spec_from_file_location("relation_learnability", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class RelationLearnabilityTest(unittest.TestCase):
    def test_mark_pair_inserts_distinct_markers(self):
        claim = "Aspirin treats fever."
        pair = {
            "left": {"start": 0, "end": 7, "surface": "Aspirin", "kind": "drug"},
            "right": {"start": 15, "end": 20, "surface": "fever", "kind": "condition"},
        }
        value = module.mark_pair(claim, pair)
        self.assertIn("[E1] Aspirin [/E1]", value)
        self.assertIn("[E2] fever [/E2]", value)

    def test_metrics_balanced_accuracy(self):
        value = module.metrics([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertEqual(value["recall"], 0.5)
        self.assertEqual(value["specificity"], 1.0)
        self.assertEqual(value["balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()

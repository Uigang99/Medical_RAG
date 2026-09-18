import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_rag2_context_graph_budgeted.py"
SPEC = importlib.util.spec_from_file_location("budget_context_graph", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BudgetContextGraphTests(unittest.TestCase):
    def test_surface_normalization(self):
        self.assertEqual(MODULE.normalized_surface("Crohn's  Disease"), "crohn s disease")

    def test_parse_keeps_exact_spans(self):
        row = {"proposition_id": "p1", "document_id": "d1", "source": "pmc", "origin_split": "test", "claim": "Aspirin reduces platelet aggregation.", "evidence": "Aspirin reduces platelet aggregation.", "text_sha256": "x"}
        value = MODULE.parse_output(row, '{"e":["Aspirin","platelet aggregation"]}')
        self.assertEqual([item["surface"] for item in value["participants"]], ["Aspirin", "platelet aggregation"])

    def test_target_cannot_drop_test_cohort(self):
        args = type("Args", (), {"target_propositions": 10})()
        with self.assertRaises(RuntimeError):
            MODULE.source_targets(args, {source: 3 for source in MODULE.SOURCES})


if __name__ == "__main__":
    unittest.main()

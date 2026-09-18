import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rag2_proposition_evidence_graph_pilot.py"
spec = importlib.util.spec_from_file_location("evidence_graph", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class EvidenceGraphTest(unittest.TestCase):
    def test_strict_link_requires_score_margin_and_category(self):
        mention = {
            "surface": "aspirin", "umls_status": "linked_confident", "umls_cui": "C1",
            "umls_candidates": [
                {"cui": "C1", "canonical_name": "Aspirin", "category": "DrugOrChemical", "score": 0.97},
                {"cui": "C2", "canonical_name": "Other", "category": "DrugOrChemical", "score": 0.80},
            ],
        }
        self.assertIsNotNone(module.strict_umls_link(mention, 0.90, 0.05))
        mention["umls_candidates"][1]["score"] = 0.95
        self.assertIsNone(module.strict_umls_link(mention, 0.90, 0.05))


if __name__ == "__main__":
    unittest.main()

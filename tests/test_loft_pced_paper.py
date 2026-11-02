import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_loft_pced_paper.py"
spec = importlib.util.spec_from_file_location("loft_pced_paper", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LoftPcedPaperTest(unittest.TestCase):
    def test_document_format_matches_official_converter(self):
        self.assertEqual(module.format_doc({"title_text": "T", "passage_text": "P"}),
                         "(T) CONTENT: P")

    def test_paper_subspan_em(self):
        self.assertEqual(module.subspan_em("Answer: The Toronto Coach Terminal.",
                                           ["Toronto Coach Terminal"]), 1)
        self.assertEqual(module.subspan_em("Answer: Seattle", ["Toronto"]), 0)

    def test_windows_and_max_aggregation(self):
        scores = module.length_safe_scores("q", ["a " * 1025, "b"],
            lambda pairs: [1.0 if pair[1].startswith("a") else 0.5 for pair in pairs])
        self.assertEqual(scores, [1.0, 0.5])

    def test_paper_prior_normalizes_dense_before_harmonic_mean(self):
        import torch
        values = {"bge_dense_pair_raw": [0.0, 0.5], "ce_normalized": [0.5, 0.5]}
        paper = module.prior_tensor(values, "paper", "cpu", torch.float32)
        script = module.prior_tensor(values, "released-script", "cpu", torch.float32)
        self.assertAlmostEqual(float(paper[0]), 0.5, places=4)
        self.assertEqual(float(script[0]), 0.0)


if __name__ == "__main__":
    unittest.main()

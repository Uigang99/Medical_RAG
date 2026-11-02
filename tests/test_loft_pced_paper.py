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

    def test_checked_embedding_preserves_order_and_exposes_runtime_error(self):
        import torch

        class Tokens(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return Tokens(input_ids=torch.tensor([[int(t)] for t in texts]),
                              attention_mask=torch.ones(len(texts), 1, dtype=torch.long))

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.fail = False

            def forward(self, inputs, **kwargs):
                if self.fail:
                    raise RuntimeError("actual model failure")
                return {"dense_vecs": inputs["input_ids"].float()}

        encoder = Encoder()
        model = type("BGE", (), {"model": encoder, "tokenizer": Tokenizer()})()
        self.assertEqual(module.encode_dense_checked(model, ["1", "2", "3"], 2).ravel().tolist(),
                         [1.0, 2.0, 3.0])
        encoder.fail = True
        with self.assertRaisesRegex(RuntimeError, "actual model failure"):
            module.encode_dense_checked(model, ["1"], 1)


if __name__ == "__main__":
    unittest.main()

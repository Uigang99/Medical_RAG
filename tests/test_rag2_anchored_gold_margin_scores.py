from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_rag2_anchored_gold_margin_scores import batch_choice_scores  # noqa: E402


class GoldMarginScoresTest(unittest.TestCase):
    def test_wrong_to_correct_crossing_and_utility(self) -> None:
        no_doc = torch.tensor([[1.2, 2.0, 0.3, -0.5]], dtype=torch.float32)
        with_doc = torch.tensor([[2.4, 2.1, 0.5, -0.2]], dtype=torch.float32)
        scores = batch_choice_scores(no_doc, with_doc, torch.tensor([0]), 1.0)

        self.assertAlmostEqual(float(scores["no_document_gold_margin"][0]), -0.8, places=6)
        self.assertAlmostEqual(float(scores["with_document_gold_margin"][0]), 0.3, places=6)
        self.assertAlmostEqual(float(scores["gold_margin_delta"][0]), 1.1, places=6)
        self.assertEqual(int(scores["no_document_prediction_index"][0]), 1)
        self.assertEqual(int(scores["with_document_prediction_index"][0]), 0)
        expected = 1 / (1 + math.exp(-0.3)) - 1 / (1 + math.exp(0.8))
        self.assertAlmostEqual(float(scores["boundary_probability_delta"][0]), expected, places=6)

    def test_gold_probability_can_rise_while_margin_worsens(self) -> None:
        # Gold A rises in the four-way softmax, but B rises even more relative
        # to A; the gold-vs-strongest-wrong margin therefore decreases.
        no_doc = torch.log(torch.tensor([[0.30, 0.40, 0.20, 0.10]]))
        with_doc = torch.log(torch.tensor([[0.34, 0.50, 0.10, 0.06]]))
        scores = batch_choice_scores(no_doc, with_doc, torch.tensor([0]), 1.0)

        self.assertGreater(float(scores["gold_choice_probability_delta"][0]), 0.0)
        self.assertLess(float(scores["gold_margin_delta"][0]), 0.0)
        self.assertLess(float(scores["boundary_probability_delta"][0]), 0.0)

    def test_best_wrong_is_recomputed_per_context(self) -> None:
        no_doc = torch.tensor([[3.0, 4.0, 1.0, 2.0]])
        with_doc = torch.tensor([[3.5, 2.0, 4.0, 1.0]])
        scores = batch_choice_scores(no_doc, with_doc, torch.tensor([0]), 1.0)

        self.assertEqual(int(scores["no_document_best_wrong_index"][0]), 1)
        self.assertEqual(int(scores["with_document_best_wrong_index"][0]), 2)
        self.assertAlmostEqual(float(scores["no_document_gold_margin"][0]), -1.0)
        self.assertAlmostEqual(float(scores["with_document_gold_margin"][0]), -0.5)

    def test_temperature_must_be_positive(self) -> None:
        logits = torch.zeros((1, 4))
        with self.assertRaises(ValueError):
            batch_choice_scores(logits, logits, torch.tensor([0]), 0.0)


if __name__ == "__main__":
    unittest.main()

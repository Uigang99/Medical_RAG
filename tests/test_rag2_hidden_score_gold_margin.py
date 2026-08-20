from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_rag2_hidden_score_gold_margin import (  # noqa: E402
    adaptive_exact_logits,
    adaptive_question_features,
    binary_metrics,
    choice_behavior,
    group_summary,
    quantile_table,
)


class HiddenScoreGoldMarginTests(unittest.TestCase):
    def test_adaptive_question_retry_releases_failed_batch_and_preserves_order(self) -> None:
        class FakeModel:
            def zero_grad(self, set_to_none: bool = True) -> None:
                self.set_to_none = set_to_none

        class FakeExtractor:
            def __init__(self) -> None:
                self.model = FakeModel()

            def no_document_features(self, sequences, gold_indices):
                if len(sequences) > 2:
                    raise torch.OutOfMemoryError("synthetic CUDA out of memory")
                values = torch.tensor([sequence[0] for sequence in sequences], dtype=torch.float32)
                return SimpleNamespace(
                    choice_logits=values[:, None].repeat(1, 4),
                    c_norm=values[:, None],
                )

        logits, norms = adaptive_question_features(
            FakeExtractor(),
            [[1], [2], [3], [4]],
            [0, 1, 2, 3],
            description="unit-test",
        )
        self.assertEqual(logits[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(norms.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_adaptive_exact_logit_retry_preserves_order(self) -> None:
        extractor = SimpleNamespace(model=SimpleNamespace(zero_grad=lambda **_: None))

        def fake_exact(_extractor, sequences):
            if len(sequences) > 2:
                raise torch.OutOfMemoryError("synthetic CUDA out of memory")
            values = torch.tensor([sequence[0] for sequence in sequences], dtype=torch.float32)
            return values[:, None].repeat(1, 4)

        with patch("analyze_rag2_hidden_score_gold_margin.exact_choice_logits", fake_exact):
            logits = adaptive_exact_logits(
                extractor,
                [[1], [2], [3], [4]],
                description="unit-test",
            )
        self.assertEqual(logits[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_choice_behavior_uses_gold_vs_strongest_competitor_margin(self) -> None:
        behavior = choice_behavior(np.asarray([1.0, 3.0, 2.0, -1.0]), gold_index=2)
        self.assertEqual(behavior["prediction"], "B")
        self.assertFalse(behavior["correct"])
        self.assertAlmostEqual(behavior["gold_margin"], -1.0)
        probabilities = np.exp(np.asarray([1.0, 3.0, 2.0, -1.0]))
        expected = np.log(probabilities[2] / probabilities.sum())
        self.assertAlmostEqual(behavior["gold_logprob"], expected)

    def test_binary_metrics_reports_both_positive_and_negative_recovery(self) -> None:
        predicted = pd.Series([True, True, False, False])
        actual = pd.Series([True, False, True, False])
        metrics = binary_metrics(predicted, actual)
        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)

    def test_group_summary_separates_logprob_and_margin_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_key": ["q1", "q1", "q2", "q2"],
                "projection_score": [-2.0, -1.0, 1.0, 2.0],
                "linearized_gold_logprob_delta": [-4.0, -2.0, 2.0, 4.0],
                "actual_gold_logprob_delta": [-4.1, -2.1, 2.1, 4.1],
                "actual_gold_margin_delta": [4.0, 2.0, -2.0, -4.0],
                "answer_transition": ["C->W", "C->C", "W->W", "W->C"],
            }
        )
        summary = group_summary(frame, thresholds=[0.0])
        self.assertGreater(summary["correlation"]["projection_vs_gold_logprob_delta"]["pearson"], 0.99)
        self.assertLess(summary["correlation"]["projection_vs_gold_margin_delta"]["pearson"], -0.99)
        logprob = summary["thresholds"]["0"]["vs_gold_logprob_improvement"]
        margin = summary["thresholds"]["0"]["vs_gold_margin_improvement"]
        self.assertEqual(logprob["accuracy"], 1.0)
        self.assertEqual(margin["accuracy"], 0.0)

    def test_quantile_table_retains_every_pair(self) -> None:
        frame = pd.DataFrame(
            {
                "pair_id": [f"p{index}" for index in range(10)],
                "projection_score": np.arange(10, dtype=float),
                "delta_h_norm": np.ones(10),
                "actual_gold_logprob_delta": np.arange(10, dtype=float),
                "actual_gold_margin_delta": np.arange(10, dtype=float),
                "with_document_correct": [index % 2 == 0 for index in range(10)],
            }
        )
        table = quantile_table(frame, "projection_score", bins=5)
        self.assertEqual(int(table["pairs"].sum()), 10)
        self.assertEqual(len(table), 5)


if __name__ == "__main__":
    unittest.main()

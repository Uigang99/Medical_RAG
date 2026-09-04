from __future__ import annotations

import unittest

from scripts.evaluate_rag2_direct_choice_document_attribution import (
    auc,
    comparison_metrics,
    pair_order_accuracy,
    spearman,
)


class DirectChoiceDocumentAttributionTest(unittest.TestCase):
    def test_relative_order_metrics(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(pair_order_accuracy([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(pair_order_accuracy([1, 2, 3], [30, 20, 10]), 0.0)

    def test_comparison_is_question_macro_and_reports_pooled_secondary(self) -> None:
        rows = [
            {"left": [1.0, 2.0, 3.0], "right": [0.1, 0.2, 0.3]},
            {"left": [3.0, 2.0, 1.0], "right": [0.3, 0.2, 0.1]},
        ]
        result = comparison_metrics(rows, "left", "right")
        self.assertAlmostEqual(result["mean_question_spearman"], 1.0)
        self.assertAlmostEqual(result["top1_overlap"], 1.0)
        self.assertAlmostEqual(result["mean_question_pair_order_accuracy"], 1.0)

    def test_auc(self) -> None:
        self.assertAlmostEqual(auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertIsNone(auc([False, False], [0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()

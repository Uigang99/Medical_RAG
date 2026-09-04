from __future__ import annotations

import unittest

from scripts.evaluate_rag2_document_attribution_faithfulness import (
    metric_block,
    rankdata,
    spearman,
    summarize,
    trace_category,
)


class DocumentAttributionFaithfulnessTest(unittest.TestCase):
    def test_semantic_cohort_requires_clean_mixed_or_all_non_support(self) -> None:
        self.assertEqual(trace_category(["direct_support", "no_evidence"]), "mixed")
        self.assertEqual(trace_category(["no_evidence", "misleading_evidence"]), "all_non_support")
        self.assertIsNone(trace_category(["direct_support", "supporting_evidence"]))
        self.assertIsNone(trace_category(["direct_support", "indeterminate_or_mixed"]))

    def test_rankdata_uses_average_tie_ranks(self) -> None:
        self.assertEqual(rankdata([3.0, 1.0, 1.0]).tolist(), [3.0, 1.5, 1.5])
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)

    @staticmethod
    def perfect_row(sample: str, correct: bool) -> dict:
        return {
            "sample_id": sample,
            "cohort": "mixed",
            "answer_correct": correct,
            "semantic_labels": ["direct_support", "no_evidence", "supporting_evidence", "misleading_evidence"],
            "attribution_abs": [4.0, 1.0, 3.0, 2.0],
            "removal_effect_abs": [0.4, 0.1, 0.3, 0.2],
            "document_token_counts": [40, 40, 40, 40],
            "rerank_ranks": [1, 2, 3, 4],
            "full_score": -0.5,
        }

    def test_perfect_attribution_metrics_and_success_rule(self) -> None:
        rows = [self.perfect_row("correct", True), self.perfect_row("wrong", False)]
        block = metric_block(rows)
        self.assertAlmostEqual(block["pooled_spearman_attribution_vs_removal"], 1.0)
        self.assertAlmostEqual(block["top1_overlap"], 1.0)
        self.assertEqual(block["questions"], 2)
        self.assertTrue(summarize(rows, replicates=20, seed=3)["pre_registered_success"]["passed"])


if __name__ == "__main__":
    unittest.main()

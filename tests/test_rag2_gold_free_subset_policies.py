from __future__ import annotations

import unittest

from scripts.materialize_rag2_gold_free_subset_policies import (
    choose_consensus_confidence,
    choose_max_confidence,
    choose_min_entropy,
    consensus_choice,
)


def subset(
    mask: int,
    ids: list[str],
    ranks: list[int],
    logits: list[float],
    probabilities: list[float],
) -> dict:
    return {
        "mask": mask,
        "selected_document_ids": ids,
        "selected_document_ranks": ranks,
        "choice_logits": logits,
        "choice_probabilities": probabilities,
    }


class GoldFreeSubsetPolicyTests(unittest.TestCase):
    def test_max_confidence_can_choose_empty_internal_knowledge(self) -> None:
        values = [
            subset(0, [], [], [5.0, 1.0, 0.0, -1.0], [0.97, 0.02, 0.008, 0.002]),
            subset(1, ["d1"], [1], [3.0, 1.0, 0.0, -1.0], [0.84, 0.11, 0.04, 0.01]),
        ]
        self.assertEqual(choose_max_confidence(values)["selected_document_ids"], [])

    def test_min_entropy_uses_full_distribution(self) -> None:
        diffuse_tail = subset(
            1,
            ["d1"],
            [1],
            [4.0, 2.0, 1.9, 1.8],
            [0.72, 0.10, 0.09, 0.09],
        )
        concentrated_tail = subset(
            2,
            ["d2"],
            [2],
            [3.8, 2.0, -4.0, -5.0],
            [0.858, 0.141, 0.0007, 0.0003],
        )
        self.assertEqual(
            choose_min_entropy([diffuse_tail, concentrated_tail])["selected_document_ids"],
            ["d2"],
        )

    def test_consensus_resists_single_confident_outlier(self) -> None:
        moderate_a = [
            subset(index, [f"a{index}"], [index], [2.0, 1.0, 0.0, -1.0], [0.60, 0.20, 0.12, 0.08])
            for index in (1, 2, 3)
        ]
        outlier_b = subset(4, ["b"], [4], [0.0, 8.0, -1.0, -2.0], [0.005, 0.99, 0.003, 0.002])
        values = [*moderate_a, outlier_b]
        self.assertEqual(consensus_choice(values), 0)
        self.assertEqual(choose_max_confidence(values)["selected_document_ids"], ["b"])
        self.assertNotEqual(
            choose_consensus_confidence(values)["selected_document_ids"],
            ["b"],
        )

    def test_confidence_tie_prefers_fewer_then_earlier_documents(self) -> None:
        values = [
            subset(3, ["d1", "d2"], [1, 2], [3.0, 1.0, 0.0, -1.0], [0.84, 0.11, 0.04, 0.01]),
            subset(2, ["d2"], [2], [3.0, 1.0, 0.0, -1.0], [0.84, 0.11, 0.04, 0.01]),
            subset(1, ["d1"], [1], [3.0, 1.0, 0.0, -1.0], [0.84, 0.11, 0.04, 0.01]),
        ]
        self.assertEqual(choose_max_confidence(values)["selected_document_ids"], ["d1"])


if __name__ == "__main__":
    unittest.main()

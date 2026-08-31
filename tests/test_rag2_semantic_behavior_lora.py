from __future__ import annotations

import math
import unittest

import torch

from medrag.training.semantic_behavior_lora import (
    choose_semantic_behavior_pair,
    gold_margins,
    jensen_shannon_divergence,
    semantic_behavior_losses,
    stratified_pair_limit,
)


def document(
    pair_id: str,
    label: str,
    *,
    delta: float,
    margin: float,
    js: float,
    rank: int,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "semantic_label": label,
        "gold_margin_delta": delta,
        "with_document_gold_margin": margin,
        "answer_js_divergence": js,
        "doc_rank": rank,
    }


class SemanticBehaviorPairTest(unittest.TestCase):
    def test_js_is_symmetric_and_zero_for_identical_distributions(self) -> None:
        left = [0.7, 0.1, 0.1, 0.1]
        right = [0.1, 0.7, 0.1, 0.1]
        self.assertAlmostEqual(jensen_shannon_divergence(left, left), 0.0, places=10)
        self.assertAlmostEqual(
            jensen_shannon_divergence(left, right),
            jensen_shannon_divergence(right, left),
            places=10,
        )
        self.assertGreater(jensen_shannon_divergence(left, right), 0.0)

    def test_pair_selection_uses_underused_direct_and_sensitive_negative(self) -> None:
        rows = [
            document("p1", "direct_support", delta=-1.0, margin=-0.4, js=0.02, rank=1),
            document("p2", "direct_support", delta=0.5, margin=1.1, js=0.04, rank=2),
            document("n1", "no_evidence", delta=0.8, margin=0.7, js=0.30, rank=3),
            document("n2", "misleading_evidence", delta=2.0, margin=1.2, js=0.10, rank=4),
        ]
        pair = choose_semantic_behavior_pair(rows)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair["positive"]["pair_id"], "p1")
        self.assertEqual(pair["negative"]["pair_id"], "n1")
        self.assertEqual(pair["pair_group"], "hard")
        self.assertAlmostEqual(pair["semantic_preference_violation"], 1.1)

    def test_stratified_limit_keeps_hard_and_aligned_examples(self) -> None:
        rows = []
        for index in range(20):
            rows.append(
                {
                    "sample_id": f"hard-{index}",
                    "pair_group": "hard",
                    "negative_semantic_label": (
                        "misleading_evidence" if index < 4 else "no_evidence"
                    ),
                }
            )
            rows.append(
                {
                    "sample_id": f"aligned-{index}",
                    "pair_group": "aligned",
                    "negative_semantic_label": (
                        "misleading_evidence" if index < 4 else "no_evidence"
                    ),
                }
            )
        selected = stratified_pair_limit(rows, limit=10, hard_fraction=0.7, seed=7)
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(row["pair_group"] == "hard" for row in selected), 7)
        self.assertEqual(sum(row["pair_group"] == "aligned" for row in selected), 3)
        self.assertGreaterEqual(
            sum(row["negative_semantic_label"] == "misleading_evidence" for row in selected),
            2,
        )

    def test_unlimited_selection_downsamples_to_requested_mixture(self) -> None:
        rows = [
            {
                "sample_id": f"hard-{index}",
                "pair_group": "hard",
                "negative_semantic_label": "no_evidence",
            }
            for index in range(4)
        ] + [
            {
                "sample_id": f"aligned-{index}",
                "pair_group": "aligned",
                "negative_semantic_label": "no_evidence",
            }
            for index in range(16)
        ]
        selected = stratified_pair_limit(rows, limit=0, hard_fraction=0.5, seed=3)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(row["pair_group"] == "hard" for row in selected), 4)


class SemanticBehaviorLossTest(unittest.TestCase):
    def test_gold_margin(self) -> None:
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0], [0.0, 2.0, 1.0, 3.0]])
        gold = torch.tensor([0, 1])
        self.assertTrue(torch.allclose(gold_margins(logits, gold), torch.tensor([1.0, -1.0])))

    def test_well_aligned_behavior_has_lower_loss(self) -> None:
        teacher = torch.tensor([[0.70, 0.10, 0.10, 0.10]])
        teacher_logits = teacher.log()
        gold = torch.tensor([0])
        good = semantic_behavior_losses(
            positive_logits=torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
            negative_logits=teacher_logits,
            no_rag_logits=teacher_logits,
            frozen_no_rag_probabilities=teacher,
            gold_indices=gold,
            preference_margin=0.5,
            positive_weight=1.0,
            preference_weight=1.0,
            negative_invariance_weight=0.1,
            no_rag_preservation_weight=0.1,
        )
        bad = semantic_behavior_losses(
            positive_logits=torch.tensor([[0.0, 5.0, 0.0, 0.0]]),
            negative_logits=torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
            no_rag_logits=torch.tensor([[0.0, 5.0, 0.0, 0.0]]),
            frozen_no_rag_probabilities=teacher,
            gold_indices=gold,
            preference_margin=0.5,
            positive_weight=1.0,
            preference_weight=1.0,
            negative_invariance_weight=0.1,
            no_rag_preservation_weight=0.1,
        )
        self.assertTrue(math.isfinite(float(good["loss"])))
        self.assertLess(float(good["loss"]), float(bad["loss"]))


if __name__ == "__main__":
    unittest.main()

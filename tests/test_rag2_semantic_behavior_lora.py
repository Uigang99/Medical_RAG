from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from medrag.training.semantic_behavior_lora import (
    choose_semantic_behavior_pair,
    gold_margins,
    jensen_shannon_divergence,
    natural_pair_limit,
    semantic_behavior_losses,
    stratified_pair_limit,
)
from scripts.train_rag2_semantic_behavior_lora import checkpoint_selection


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

    def test_natural_selection_keeps_all_rows_when_unlimited(self) -> None:
        rows = [{"sample_id": str(index)} for index in range(20)]
        selected = natural_pair_limit(rows, limit=0, seed=11)
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(
            {row["sample_id"] for row in selected},
            {row["sample_id"] for row in rows},
        )
        self.assertEqual(
            natural_pair_limit(rows, limit=5, seed=11),
            natural_pair_limit(rows, limit=5, seed=11),
        )
        self.assertEqual(len(natural_pair_limit(rows, limit=5, seed=11)), 5)


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

    def test_detached_preference_cannot_lower_negative_margin(self) -> None:
        positive = torch.tensor([[0.0, 2.0, 0.0, 0.0]], requires_grad=True)
        negative = torch.tensor([[2.0, 0.0, 0.0, 0.0]], requires_grad=True)
        no_rag = torch.tensor([[0.0, 2.0, 0.0, 0.0]], requires_grad=True)
        teacher = torch.softmax(no_rag.detach(), dim=-1)
        losses = semantic_behavior_losses(
            positive_logits=positive,
            negative_logits=negative,
            no_rag_logits=no_rag,
            frozen_no_rag_probabilities=teacher,
            negative_reference_probabilities=torch.softmax(no_rag.detach(), dim=-1),
            gold_indices=torch.tensor([0]),
            preference_margin=0.5,
            positive_weight=0.0,
            preference_weight=1.0,
            negative_invariance_weight=0.0,
            no_rag_preservation_weight=0.0,
            detach_negative_margin=True,
        )
        losses["loss"].backward()
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)
        self.assertEqual(float(negative.grad.abs().sum()), 0.0)

    def test_negative_can_follow_current_student_no_rag(self) -> None:
        frozen_teacher = torch.tensor([[0.70, 0.10, 0.10, 0.10]])
        current_no_rag = torch.tensor([[0.0, 3.0, 0.0, 0.0]])
        losses = semantic_behavior_losses(
            positive_logits=torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
            negative_logits=current_no_rag.clone(),
            no_rag_logits=current_no_rag,
            frozen_no_rag_probabilities=frozen_teacher,
            negative_reference_probabilities=torch.softmax(current_no_rag, dim=-1),
            gold_indices=torch.tensor([0]),
            preference_margin=0.5,
            positive_weight=0.0,
            preference_weight=0.0,
            negative_invariance_weight=1.0,
            no_rag_preservation_weight=0.0,
            detach_negative_margin=True,
        )
        self.assertAlmostEqual(float(losses["negative_invariance"]), 0.0, places=6)


class SemanticBehaviorCheckpointTest(unittest.TestCase):
    def test_preserved_selection_requires_all_constraints(self) -> None:
        args = SimpleNamespace(
            objective="proposed_preserved",
            max_negative_no_rag_answer_change_rate=0.10,
            max_negative_no_rag_js=0.05,
            max_negative_accuracy_drop=0.03,
            max_no_rag_accuracy_drop=0.01,
        )
        metrics = {
            "preference_accuracy": 0.80,
            "positive_accuracy": 0.85,
            "negative_vs_student_no_rag_answer_change_rate": 0.08,
            "mean_negative_js_from_student_no_rag": 0.04,
            "negative_accuracy_delta_vs_student_no_rag": -0.02,
            "no_rag_accuracy_delta_vs_frozen": 0.00,
        }
        feasible = checkpoint_selection(args, metrics)
        self.assertTrue(feasible["feasible"])
        metrics["negative_vs_student_no_rag_answer_change_rate"] = 0.14
        infeasible = checkpoint_selection(args, metrics)
        self.assertFalse(infeasible["feasible"])
        self.assertGreater(
            infeasible["violations"]["negative_no_rag_answer_change_rate"], 0.0
        )
        self.assertGreater(tuple(feasible["rank"]), tuple(infeasible["rank"]))


if __name__ == "__main__":
    unittest.main()

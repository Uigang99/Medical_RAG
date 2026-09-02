from __future__ import annotations

import unittest

import torch

from medrag.training.direct_semantic_contrastive import (
    balanced_epoch_samples,
    build_training_groups,
    semantic_contrastive_losses,
)


class DirectSemanticContrastiveTests(unittest.TestCase):
    @staticmethod
    def value(
        sample_id: str,
        semantic: str,
        *,
        document_correct: bool,
        no_rag_correct: bool,
        probabilities: list[float],
    ) -> dict[str, object]:
        return {
            "row": {
                "sample_id": sample_id,
                "semantic_label": semantic,
                "gold_answer": "A",
                "frozen_document_correct": document_correct,
                "frozen_no_rag_correct": no_rag_correct,
                "frozen_document_probabilities": probabilities,
            }
        }

    def test_no_evidence_wrong_is_a_pair_candidate_but_not_invariance_target(self) -> None:
        direct_wrong = self.value(
            "q1", "direct_support", document_correct=False, no_rag_correct=False,
            probabilities=[0.3, 0.4, 0.2, 0.1],
        )
        no_evidence_wrong = self.value(
            "q1", "no_evidence", document_correct=False, no_rag_correct=False,
            probabilities=[0.1, 0.7, 0.1, 0.1],
        )
        direct_correct = self.value(
            "q2", "direct_support", document_correct=True, no_rag_correct=True,
            probabilities=[0.8, 0.1, 0.05, 0.05],
        )
        no_evidence_safe = self.value(
            "q2", "no_evidence", document_correct=True, no_rag_correct=True,
            probabilities=[0.6, 0.2, 0.1, 0.1],
        )
        groups = build_training_groups(
            [direct_wrong, no_evidence_wrong, direct_correct, no_evidence_safe],
            min_pair_teacher_gap=0.5,
        )
        self.assertIn(no_evidence_safe, groups["no_evidence_invariance"])
        self.assertNotIn(no_evidence_wrong, groups["no_evidence_invariance"])
        self.assertEqual(groups["same_question_contrast"][0], (direct_wrong, no_evidence_wrong))

    def test_balanced_sampling_has_equal_group_counts(self) -> None:
        groups = {
            "direct_support_correction": [1, 2, 3],
            "direct_support_preservation": [4, 5],
            "no_evidence_invariance": [6, 7, 8, 9],
            "same_question_contrast": [(10, 11)],
        }
        target, selected = balanced_epoch_samples(groups, epoch=1, seed=42)
        self.assertEqual(target, 3)
        self.assertTrue(all(len(values) == 3 for values in selected.values()))

    def test_loss_is_finite_and_backpropagates(self) -> None:
        tensors = [torch.randn(3, 4, requires_grad=True) for _ in range(5)]
        teacher = torch.softmax(torch.randn(3, 4), dim=-1)
        losses = semantic_contrastive_losses(
            correction_logits=tensors[0],
            correction_gold=torch.tensor([0, 1, 2]),
            preservation_logits=tensors[1],
            preservation_teacher=teacher,
            invariance_logits=tensors[2],
            invariance_teacher=teacher,
            pair_direct_logits=tensors[3],
            pair_no_evidence_logits=tensors[4],
            pair_gold=torch.tensor([0, 1, 2]),
            boundary_margin=0.5,
            pair_margin=0.5,
        )
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        losses["loss"].backward()
        self.assertTrue(all(value.grad is not None for value in tensors))


if __name__ == "__main__":
    unittest.main()

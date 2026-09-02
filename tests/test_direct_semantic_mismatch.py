from __future__ import annotations

import unittest

import torch

from medrag.training.direct_semantic_mismatch import (
    TRAIN_CASES,
    gold_margins,
    semantic_mismatch_losses,
)


class DirectSemanticMismatchTests(unittest.TestCase):
    def test_gold_margin_uses_strongest_wrong_choice(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 2.0, 3.0]])
        gold = torch.tensor([1, 2])
        self.assertTrue(
            torch.allclose(gold_margins(logits, gold), torch.tensor([1.0, -3.0]))
        )

    def test_mismatch_loss_is_finite_and_backpropagates(self) -> None:
        document = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]] * len(TRAIN_CASES), requires_grad=True
        )
        question = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0]] * len(TRAIN_CASES), requires_grad=True
        )
        teacher_document = torch.softmax(document.detach(), dim=-1)
        teacher_question = torch.softmax(question.detach(), dim=-1)
        result = semantic_mismatch_losses(
            document_logits=document,
            no_rag_logits=question,
            frozen_document_probabilities=teacher_document,
            frozen_no_rag_probabilities=teacher_question,
            gold_indices=torch.tensor([0, 0, 1, 0, 2]),
            case_indices=torch.arange(len(TRAIN_CASES)),
            boundary_margin=0.0,
            gain_margin=0.5,
            case_weights={name: 1.0 for name in TRAIN_CASES},
            no_rag_preservation_weight=2.0,
            no_rag_row_weights=torch.ones(len(TRAIN_CASES)),
        )
        self.assertTrue(bool(torch.isfinite(result["loss"])))
        result["loss"].backward()
        self.assertIsNotNone(document.grad)
        self.assertIsNotNone(question.grad)


if __name__ == "__main__":
    unittest.main()

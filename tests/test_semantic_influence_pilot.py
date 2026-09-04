from __future__ import annotations

import argparse
import unittest

import torch

from scripts.train_rag2_semantic_influence_pilot import (
    build_train_batch,
    influence_losses,
    pair_win,
)


class SemanticInfluencePilotTest(unittest.TestCase):
    def test_train_batch_disables_adapter_only_for_frozen_reference(self) -> None:
        value = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),
            "token_document_ids": torch.tensor([-1, 0, 0, -1, 1, -1]),
            "document_mask": torch.tensor([False, True, True, False, True, False]),
        }
        batch = build_train_batch(value, 0, 1)
        self.assertEqual(tuple(batch["input_ids"].shape), (4, 6))
        self.assertFalse(bool(batch["adapter_document_mask"][0].any()))
        self.assertTrue(bool(batch["adapter_document_mask"][1:].any()))
        self.assertEqual(batch["blocked_document_ids"].tolist(), [-2, -2, 0, 1])

    def test_pair_win_is_within_question(self) -> None:
        self.assertEqual(pair_win([0.9, 0.1, 0.8], [0, 2], [1]), 1.0)
        self.assertEqual(pair_win([0.1, 0.9], [0], [1]), 0.0)

    def test_loss_prefers_larger_support_and_smaller_non_support_effect(self) -> None:
        args = argparse.Namespace(
            ranking_margin=0.02,
            ranking_weight=1.0,
            non_support_weight=0.5,
            support_floor_weight=1.0,
            full_preservation_weight=1.0,
        )
        base = torch.tensor([3.0, 0.0, -1.0])
        favorable = torch.stack([
            base,
            base,
            torch.tensor([0.0, 3.0, -1.0]),
            base,
        ]).requires_grad_(True)
        unfavorable = torch.stack([
            base,
            base,
            base,
            torch.tensor([0.0, 3.0, -1.0]),
        ]).requires_grad_(True)
        good = influence_losses(favorable, baseline_support_jsd=0.01, args=args)["loss"]
        bad = influence_losses(unfavorable, baseline_support_jsd=0.01, args=args)["loss"]
        self.assertLess(float(good.detach()), float(bad.detach()))
        good.backward()
        self.assertTrue(torch.isfinite(favorable.grad).all())


if __name__ == "__main__":
    unittest.main()

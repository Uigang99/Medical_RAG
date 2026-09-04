from __future__ import annotations

import unittest

import torch

from scripts.evaluate_rag2_direct_choice_document_mask_validity import (
    build_compact_mask_batch,
    build_physical_deletion_batch,
    build_token_document_ids,
    jsd_from_logits,
)


class DirectChoiceDocumentMaskValidityTest(unittest.TestCase):
    def test_jsd_identity_symmetry_and_separation(self) -> None:
        left = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        right = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        divergence = jsd_from_logits(left, right)
        self.assertAlmostEqual(float(divergence[0]), 0.0, places=7)
        self.assertGreater(float(divergence[1]), 0.0)
        reverse = jsd_from_logits(right, left)
        self.assertTrue(torch.allclose(divergence, reverse, atol=1e-7))
        self.assertTrue(bool((divergence <= 1.0).all()))

    def test_document_mapping_and_intervention_contract(self) -> None:
        sequence = {
            "input_ids": [10, 11, 12, 13, 14, 15],
            "document_token_indices": [[1, 2], [4]],
        }
        mapping = build_token_document_ids(sequence)
        self.assertEqual(mapping.tolist(), [-1, 0, 0, -1, 1, -1])
        ids = torch.tensor(sequence["input_ids"])
        physical = build_physical_deletion_batch(ids, mapping, pad_token_id=0)
        self.assertEqual(physical["input_ids"][0].tolist(), [0, 10, 13, 14, 15])
        self.assertEqual(physical["input_ids"][1].tolist(), [10, 11, 12, 13, 15])
        masked = build_compact_mask_batch(ids, mapping)
        self.assertEqual(masked["blocked_document_ids"].tolist(), [0, 1])
        self.assertEqual(masked["position_ids"][0].tolist(), [0, 0, 0, 1, 2, 3])
        self.assertEqual(masked["position_ids"][1].tolist(), [0, 1, 2, 3, 3, 4])


if __name__ == "__main__":
    unittest.main()

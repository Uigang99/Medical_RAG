from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_rag2_pcw_semantic_influence_pilot.py"
SPEC = importlib.util.spec_from_file_location("pcw_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def toy_layout() -> dict[str, object]:
    # Two shared-prefix tokens, two tokens per document, one output token.
    document_ids = torch.tensor([-1, -1, *sum(([index, index] for index in range(8)), []), -1])
    return {
        "sample_id": "toy",
        "document_ids": document_ids,
        "prefix_end": 2,
        "output_start": 18,
    }


class PCWSemanticInfluencePilotTest(unittest.TestCase):
    def test_pcw_mask_isolates_documents_and_exposes_all_to_output(self) -> None:
        layout = toy_layout()
        MODULE.validate_layout_mask(layout)
        mask = MODULE.pcw_additive_mask(layout, torch.device("cpu"), torch.float32)[0, 0]
        ids = layout["document_ids"]
        self.assertIsInstance(ids, torch.Tensor)
        self.assertTrue(bool(torch.all(mask[2:4, 4:18] < 0)))
        self.assertTrue(bool(torch.all(mask[-1, 2:18] == 0)))

    def test_no_block_sentinel_cannot_alias_non_document_tokens(self) -> None:
        self.assertLess(MODULE.NO_BLOCK_DOCUMENT_ID, -1)
        ids = toy_layout()["document_ids"]
        self.assertIsInstance(ids, torch.Tensor)
        self.assertFalse(bool(ids.eq(MODULE.NO_BLOCK_DOCUMENT_ID).any()))

    def test_set_router_is_permutation_equivariant(self) -> None:
        torch.manual_seed(7)
        router = MODULE.SetGateRouter(4, 16, 4, 2, 0.05, 1.5).eval()
        features = torch.randn(1, 8, 4)
        permutation = torch.tensor([7, 2, 5, 0, 6, 1, 4, 3])
        original = router(features)
        permuted = router(features[:, permutation])
        self.assertTrue(
            torch.allclose(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()

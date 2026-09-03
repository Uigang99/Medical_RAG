import unittest

import torch
from torch import nn

from medrag.training.document_path_lora import DocumentPathLoRALinear
from scripts.train_rag2_document_path_overfit import collate


class DocumentPathLoRATest(unittest.TestCase):
    def test_delta_is_zero_outside_document_positions(self):
        base = nn.Linear(4, 3, bias=False)
        module = DocumentPathLoRALinear(base, rank=2, alpha=2, dropout=0)
        nn.init.ones_(module.lora_a.weight)
        nn.init.ones_(module.lora_b.weight)
        values = torch.ones(1, 3, 4)
        frozen = base(values)
        module.set_document_mask(torch.tensor([[False, True, False]]))
        changed = module(values)
        torch.testing.assert_close(changed[:, 0], frozen[:, 0])
        torch.testing.assert_close(changed[:, 2], frozen[:, 2])
        self.assertFalse(torch.equal(changed[:, 1], frozen[:, 1]))
        self.assertEqual(module.last_non_document_delta_max, 0.0)

    def test_no_document_bypasses_adapter_exactly(self):
        base = nn.Linear(4, 3, bias=False)
        module = DocumentPathLoRALinear(base, rank=2, alpha=2, dropout=0)
        values = torch.randn(2, 3, 4)
        frozen = base(values)
        module.set_document_mask(torch.zeros(2, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(module(values), frozen))

    def test_collate_orders_conditions_by_block(self):
        def value(name, sizes):
            return {
                "row": {"sample_id": name},
                "conditions": {
                    key: {"input_ids": list(range(size)), "document_mask": [True] * size}
                    for key, size in zip(("positive", "negative", "swap"), sizes)
                },
            }

        batch = collate([value("a", (2, 3, 4)), value("b", (5, 6, 7))], 99)
        # The first B rows must be positive, next B negative, final B swap.
        lengths = batch["attention_mask"].sum(-1).tolist()
        self.assertEqual(lengths, [2, 5, 3, 6, 4, 7])


if __name__ == "__main__":
    unittest.main()

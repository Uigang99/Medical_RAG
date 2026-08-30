from __future__ import annotations

import math
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.semantic_attention import (
    register_semantic_attention,
    suppression_bias,
)


class SemanticAttentionTest(unittest.TestCase):
    def test_suppression_bias_is_monotonic_and_capped(self) -> None:
        self.assertEqual(suppression_bias(2.0, 1.0, 4.0), 0.0)
        self.assertEqual(suppression_bias(0.0, 1.0, 4.0), 0.0)
        self.assertAlmostEqual(suppression_bias(-0.5, 1.0, 4.0), -0.5)
        self.assertAlmostEqual(suppression_bias(-10.0, 1.0, 4.0), -math.log(4.0))

    def test_custom_attention_accepts_generation_tensors(self) -> None:
        attention_name = register_semantic_attention()
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            attn_implementation=attention_name,
        )
        model = LlamaForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        zero_bias = torch.zeros((1, 8), dtype=torch.float32)
        query_mask = torch.zeros((1, 8), dtype=torch.float32)
        query_mask[:, 4:] = 1.0
        suppressed = zero_bias.clone()
        suppressed[:, 1:3] = -math.log(4.0)
        with torch.inference_mode():
            baseline = model(
                input_ids=input_ids,
                semantic_token_bias=zero_bias,
                semantic_query_mask=query_mask,
                semantic_layer_start=0,
            ).logits
            changed = model(
                input_ids=input_ids,
                semantic_token_bias=suppressed,
                semantic_query_mask=query_mask,
                semantic_layer_start=0,
            ).logits
        self.assertTrue(torch.isfinite(changed).all())
        self.assertFalse(torch.equal(baseline[:, -1], changed[:, -1]))


if __name__ == "__main__":
    unittest.main()

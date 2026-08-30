from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.learned_semantic_attention import freeze_module_for_controller_training
from medrag.generation.semantic_attention import register_semantic_attention
from scripts.train_rag2_semantic_attention_controller import (
    collate_prefix_batch,
    final_choice_logits,
    normalize_accumulated_gradients,
)


class SemanticAttentionControllerTrainingTest(unittest.TestCase):
    def test_left_padding_does_not_change_choice_logits(self) -> None:
        attention_name = register_semantic_attention()
        torch.manual_seed(19)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            attn_implementation="sdpa",
        )
        model = freeze_module_for_controller_training(LlamaForCausalLM(config))
        payload = {
            "input_ids": [
                torch.tensor([1, 2, 3, 4, 5]),
                torch.tensor([1, 7, 8, 9, 10, 11, 12, 5]),
            ],
            "token_document_ids": [
                torch.tensor([-1, 0, 0, 1, -1]),
                torch.tensor([-1, 0, 0, 0, 1, 1, 1, -1]),
            ],
        }
        choice_ids = torch.tensor([20, 21, 22, 23])
        single = collate_prefix_batch(payload, [0], pad_token_id=0, device=torch.device("cpu"))
        paired = collate_prefix_batch(payload, [0, 1], pad_token_id=0, device=torch.device("cpu"))
        single_logits = final_choice_logits(
            model,
            attention_name,
            single,
            torch.zeros((1, 2)),
            choice_ids,
            0,
        )
        paired_logits = final_choice_logits(
            model,
            attention_name,
            paired,
            torch.zeros((2, 2)),
            choice_ids,
            0,
        )
        self.assertTrue(torch.allclose(single_logits[0], paired_logits[0], atol=1e-5, rtol=1e-5))

    def test_partial_accumulation_is_averaged_by_actual_count(self) -> None:
        module = torch.nn.Linear(2, 1, bias=False)
        module.weight.grad = torch.tensor([[6.0, 9.0]])
        normalize_accumulated_gradients(module, 3)
        self.assertTrue(torch.equal(module.weight.grad, torch.tensor([[2.0, 3.0]])))


if __name__ == "__main__":
    unittest.main()

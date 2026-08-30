from __future__ import annotations

import math
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.learned_semantic_attention import (
    SemanticResidualAttentionController,
    document_bias_to_token_bias,
    freeze_module_for_controller_training,
    group_robust_answer_loss,
    residual_anchor_loss,
    semantic_ordering_hinge_loss,
    single_query_document_attention_bias,
)
from medrag.generation.semantic_attention import register_semantic_attention


class LearnedSemanticAttentionTest(unittest.TestCase):
    def test_controller_sign_and_semantic_prior(self) -> None:
        controller = SemanticResidualAttentionController(dropout=0.0)
        features = torch.zeros((1, 3, 1024))
        margins = torch.tensor([[-2.0, 0.0, 2.0]])
        output = controller(features, margins)

        expected = torch.clamp(0.25 * margins, min=-math.log(4.0), max=0.0)
        minimum_gate = torch.tensor(0.25)
        relative_keep = ((expected.exp() - minimum_gate) / (1.0 - minimum_gate)).clamp(
            0.05, 0.95
        )
        expected_relaxed = torch.log(
            minimum_gate + (1.0 - minimum_gate) * relative_keep
        )
        self.assertTrue(torch.allclose(output.residual, torch.zeros_like(margins)))
        self.assertTrue(torch.allclose(output.prior_bias, expected))
        self.assertTrue(torch.allclose(output.document_bias, expected_relaxed))
        self.assertLess(output.document_bias[0, 0], output.document_bias[0, 1])
        # The validated prior is suppression-only: neutral and positive
        # semantic margins both begin at (approximately) zero bias.
        self.assertTrue(
            torch.allclose(output.document_bias[0, 1], output.document_bias[0, 2])
        )

    def test_controller_masks_padded_documents(self) -> None:
        controller = SemanticResidualAttentionController(dropout=0.0)
        output = controller(
            torch.zeros((1, 2, 1024)),
            torch.tensor([[1.0, -1.0]]),
            document_mask=torch.tensor([[True, False]]),
        )
        self.assertEqual(output.residual[0, 1].detach().item(), 0.0)
        self.assertEqual(output.prior_bias[0, 1].detach().item(), 0.0)
        self.assertEqual(output.combined_score[0, 1].detach().item(), 0.0)
        self.assertEqual(output.document_bias[0, 1].detach().item(), 0.0)

    def test_token_document_gather_is_differentiable(self) -> None:
        document_bias = torch.tensor([[-0.2, -1.0]], requires_grad=True)
        token_document_ids = torch.tensor([[-1, 0, 0, 1, -1]])
        token_bias = document_bias_to_token_bias(document_bias, token_document_ids)
        expected = torch.tensor([[0.0, -0.2, -0.2, -1.0, 0.0]])
        self.assertTrue(torch.allclose(token_bias, expected))
        expanded = single_query_document_attention_bias(document_bias, token_document_ids)
        self.assertEqual(tuple(expanded.shape), (1, 1, 1, 5))
        token_bias.sum().backward()
        self.assertTrue(torch.equal(document_bias.grad, torch.tensor([[2.0, 1.0]])))

    def test_auxiliary_losses_have_expected_semantics(self) -> None:
        biases = torch.tensor([[-0.1, -0.5, -0.4], [-0.7, -0.2, -0.1]])
        labels = torch.tensor([[1, 0, 1], [1, 0, 0]])
        # Q1 pair gaps are 0.4 and 0.1: hinge losses 0 and 0.1.
        # Q2 gaps are -0.5 and -0.6: hinge losses 0.7 and 0.8.
        expected_order = torch.tensor([(0.0 + 0.1) / 2, (0.7 + 0.8) / 2]).mean()
        actual_order = semantic_ordering_hinge_loss(biases, labels, margin=0.2)
        self.assertTrue(torch.allclose(actual_order, expected_order))

        residual = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [True, False]])
        self.assertTrue(torch.allclose(residual_anchor_loss(residual, mask), torch.tensor(5.0)))

    def test_group_robust_answer_loss_balances_question_groups(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 1.0],
                [0.0, 3.0],
                [2.0, 0.0],
            ]
        )
        gold = torch.tensor([0, 0, 1, 1])
        no_rag_correct = torch.tensor([False, False, False, True])
        per_example = torch.nn.functional.cross_entropy(logits, gold, reduction="none")
        balanced = 0.5 * (per_example[:3].mean() + per_example[3:].mean())
        expected = 0.5 * per_example.mean() + 0.5 * balanced
        actual = group_robust_answer_loss(logits, gold, no_rag_correct, balance_strength=0.5)
        self.assertTrue(torch.allclose(actual, expected))

    def test_frozen_llama_passes_answer_gradient_to_controller(self) -> None:
        attention_name = register_semantic_attention()
        torch.manual_seed(11)
        config = LlamaConfig(
            vocab_size=48,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            attn_implementation=attention_name,
        )
        llama = freeze_module_for_controller_training(LlamaForCausalLM(config))
        controller = SemanticResidualAttentionController(dropout=0.0)
        features = torch.randn((1, 2, 1024))
        margins = torch.tensor([[0.2, -0.3]])
        document_bias = controller(features, margins).document_bias
        # The frozen prefix can be cached without autograd.  Only the final
        # q_len=1 answer-decision query must retain the controller graph.
        with torch.no_grad():
            prefix = llama(
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                use_cache=True,
            )
        token_bias = document_bias_to_token_bias(
            document_bias,
            torch.tensor([[-1, 0, 0, 1, -1]]),
        )
        query_mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        outputs = llama(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            past_key_values=prefix.past_key_values,
            use_cache=True,
            semantic_token_bias=token_bias,
            semantic_query_mask=query_mask,
            semantic_layer_start=0,
        )
        loss = torch.nn.functional.cross_entropy(outputs.logits[:, -1], torch.tensor([6]))
        loss.backward()

        controller_grads = [
            parameter.grad
            for parameter in controller.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(controller_grads)
        self.assertTrue(any(bool((gradient != 0).any()) for gradient in controller_grads))
        self.assertTrue(all(parameter.grad is None for parameter in llama.parameters()))


if __name__ == "__main__":
    unittest.main()

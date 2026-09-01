from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.learned_semantic_attention import (
    SemanticResidualAttentionController,
    document_bias_to_token_bias,
)
from medrag.generation.semantic_attention import (
    DocumentAttentionCollector,
    register_semantic_attention,
)
from scripts.evaluate_rag2_all_layer_document_mask_contract import (
    build_exact_document_mask_batch,
    choice_logits_for_exact_document_masks,
    choice_logits_for_plain_batch,
)
from scripts.evaluate_rag2_semantic_gate_fidelity import (
    build_physical_loo_batch,
    evaluate_one,
    jensen_shannon_divergence,
    normalize_positive,
    pearson_correlation,
    rankdata,
    spearman_correlation,
)


class SemanticGateFidelityTest(unittest.TestCase):
    def test_physical_loo_removes_each_document_tokens(self) -> None:
        batch = build_physical_loo_batch(
            torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            torch.tensor([-1, 0, 0, 1, 1, -1, -1]),
            5,
            pad_token_id=0,
            attention_scope="rationale_wide",
            document_count=2,
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (3, 7))
        self.assertEqual(batch["variant_lengths"].tolist(), [7, 5, 5])
        self.assertFalse(bool(batch["token_document_ids"][1].eq(0).any()))
        self.assertFalse(bool(batch["token_document_ids"][2].eq(1).any()))
        self.assertEqual(int(batch["semantic_query_mask"][1].sum().item()), 2)
        self.assertEqual(int(batch["semantic_query_mask"][2].sum().item()), 2)

    def test_compact_exact_mask_matches_physical_deletion_on_tiny_llama(self) -> None:
        attention_name = register_semantic_attention()
        torch.manual_seed(17)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                pad_token_id=0,
                attention_dropout=0.0,
                attn_implementation=attention_name,
            )
        ).eval()
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        mapping = torch.tensor([-1, 0, 0, 1, 1, -1, -1], dtype=torch.long)
        physical = build_physical_loo_batch(
            input_ids,
            mapping,
            5,
            pad_token_id=0,
            attention_scope="rationale_wide",
            document_count=2,
        )
        choices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        physical_logits = choice_logits_for_plain_batch(
            model,
            physical,
            choices,
            torch.device("cpu"),
        )
        masked_logits = choice_logits_for_exact_document_masks(
            model,
            input_ids,
            mapping,
            choices,
            compact_positions=True,
            document_count=2,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(masked_logits[0], physical_logits[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(masked_logits[1], physical_logits[2], atol=1e-5, rtol=1e-5)

    def test_exact_mask_compacts_only_kept_token_positions(self) -> None:
        batch = build_exact_document_mask_batch(
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([-1, 0, 0, 1, -1, -1]),
            compact_positions=True,
            document_count=2,
        )
        self.assertEqual(batch["blocked_document_ids"].tolist(), [0, 1])
        self.assertEqual(batch["position_ids"][0].tolist(), [0, 0, 0, 1, 2, 3])
        self.assertEqual(batch["position_ids"][1].tolist(), [0, 1, 2, 2, 3, 4])

    def test_document_shares_sum_to_one(self) -> None:
        normalized = normalize_positive([0.25, 0.5, 0.25])
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertAlmostEqual(sum(normalized), 1.0)
        self.assertEqual(normalize_positive([0.0, 0.0], minimum_total=1e-6), None)

    def test_correlations_and_tie_ranks(self) -> None:
        self.assertEqual(rankdata([1.0, 1.0, 3.0]), [0.5, 0.5, 2.0])
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]) or 0.0, 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 3, 2], [2, 6, 4]) or 0.0, 1.0)

    def test_jsd_is_bounded_and_symmetric(self) -> None:
        left = torch.tensor([0.9, 0.1])
        right = torch.tensor([0.1, 0.9])
        first = jensen_shannon_divergence(left, right)
        second = jensen_shannon_divergence(right, left)
        self.assertGreater(first, 0.0)
        self.assertLessEqual(first, 1.0)
        self.assertAlmostEqual(first, second)
        self.assertAlmostEqual(jensen_shannon_divergence(left, left), 0.0)

    def test_attention_collector_sums_document_spans(self) -> None:
        collector = DocumentAttentionCollector(document_count=2, collect_value_norm=True)
        weights = torch.tensor(
            [[[[0.1, 0.3, 0.2, 0.4], [0.2, 0.2, 0.5, 0.1]]]],
            dtype=torch.float32,
        )
        collector.update(
            layer_index=3,
            attention_weights=weights,
            value_states=torch.tensor(
                [[[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            token_document_ids=torch.tensor([[-1, 0, 1, 1]]),
            active_query_mask=torch.tensor([[0.0, 1.0]]),
        )
        summary = collector.summarize()
        # Active second query: doc0=0.2, doc1=0.6, non-document=0.2.
        self.assertTrue(
            torch.allclose(summary["document_share"], torch.tensor([[0.25, 0.75]]))
        )
        self.assertTrue(
            torch.allclose(summary["document_attention_fraction"], torch.tensor([0.8]))
        )
        self.assertEqual(tuple(summary["document_value_share"].shape), (1, 2))
        self.assertAlmostEqual(float(summary["document_value_share"].sum()), 1.0, places=6)
        # doc0: 0.2*[2,0] -> norm .4; doc1: 0.5*[0,2] -> norm 1.0.
        torch.testing.assert_close(
            summary["document_value_share"],
            torch.tensor([[2.0 / 7.0, 5.0 / 7.0]]),
        )

    def test_tiny_llama_collects_attention_during_loo_batch(self) -> None:
        attention_name = register_semantic_attention()
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            attn_implementation=attention_name,
        )
        model = LlamaForCausalLM(config).eval()
        batch = build_physical_loo_batch(
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([-1, 0, 0, 1, -1, -1]),
            4,
            pad_token_id=0,
            attention_scope="rationale_wide",
            document_count=2,
        )
        document_bias = torch.tensor([[-0.1, -0.4]]).expand(3, -1)
        token_bias = document_bias_to_token_bias(
            document_bias,
            batch["token_document_ids"],
        )
        collector = DocumentAttentionCollector(document_count=2, collect_value_norm=True)
        with torch.inference_mode():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                logits_to_keep=1,
                semantic_token_bias=token_bias,
                semantic_query_mask=batch["semantic_query_mask"],
                semantic_layer_start=0,
                semantic_token_document_ids=batch["token_document_ids"],
                semantic_attention_collector=collector,
            )
        self.assertEqual(tuple(output.logits.shape), (3, 1, 64))
        summary = collector.summarize()
        self.assertEqual(tuple(summary["document_share"].shape), (1, 2))
        self.assertAlmostEqual(float(summary["document_share"].sum()), 1.0, places=5)
        self.assertEqual(tuple(summary["document_value_share"].shape), (1, 2))
        self.assertAlmostEqual(float(summary["document_value_share"].sum()), 1.0, places=5)

    def test_evaluate_one_unpacks_logits_and_collector_before_cpu_transfer(self) -> None:
        attention_name = register_semantic_attention()
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                pad_token_id=0,
                attn_implementation=attention_name,
            )
        ).eval()
        controller = SemanticResidualAttentionController(
            input_dim=4,
            hidden_dim=8,
            dropout=0.0,
        ).eval()
        payload = {
            "semantic_features": torch.zeros((1, 8, 4)),
            "semantic_margins": torch.linspace(-1.0, 1.0, 8).unsqueeze(0),
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            "token_document_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, -1, -1]]),
            "assistant_query_starts": torch.tensor([8]),
            "gold_options": torch.tensor([0]),
            "semantic_class_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "sample_ids": ["tiny-question"],
            "pair_ids": [[f"pair-{index}" for index in range(8)]],
        }
        row = evaluate_one(
            payload,
            0,
            controller,
            model,
            SimpleNamespace(pad_token_id=0),
            torch.tensor([11, 12, 13, 14]),
            {"attention_scope": "rationale_wide", "semantic_layer_start": 0},
            SimpleNamespace(
                device="cpu",
                minimum_total_jsd=0.0,
                dataset="medqa",
                split="test",
            ),
        )
        self.assertEqual(len(row["predicted_document_share"]), 8)
        self.assertEqual(len(row["loo_jsd"]), 8)


if __name__ == "__main__":
    unittest.main()

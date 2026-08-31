from __future__ import annotations

import unittest

import torch

from medrag.attribution.target_llm_predictor import (
    TargetLLMAttributionPredictor,
    attribution_loss,
    masked_document_distribution,
)
from scripts.prepare_rag2_target_llm_attribution import build_conditional_removal_batch
from scripts.train_rag2_target_llm_attribution import (
    permute_document_aligned_batch,
    selected_paths,
)


class TargetLLMAttributionPredictorTest(unittest.TestCase):
    def make_model(self) -> TargetLLMAttributionPredictor:
        torch.manual_seed(7)
        model = TargetLLMAttributionPredictor(
            target_hidden_size=8,
            selected_layer_count=2,
            model_dim=16,
            transformer_layers=2,
            attention_heads=4,
            feedforward_dim=32,
            dropout=0.0,
        )
        model.eval()
        return model

    def test_variable_k_padding_does_not_change_present_documents(self) -> None:
        model = self.make_model()
        documents = torch.randn(1, 2, 2, 8)
        global_features = torch.randn(1, 2, 8)
        output_two = model(
            documents,
            global_features,
            torch.tensor([[True, True]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.5, 0.7]]),
        )
        padded = torch.cat((documents, torch.randn(1, 2, 2, 8)), dim=1)
        output_four = model(
            padded,
            global_features,
            torch.tensor([[True, True, False, False]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.7, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            output_two.document_logits,
            output_four.document_logits[:, :2],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(output_two.log_total_loo, output_four.log_total_loo)
        torch.testing.assert_close(output_two.log_set_shift, output_four.log_set_shift)

    def test_document_permutation_with_rank_features_is_equivariant(self) -> None:
        model = self.make_model()
        documents = torch.randn(1, 3, 2, 8)
        global_features = torch.randn(1, 2, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        rank = torch.tensor([[0.0, 0.5, 1.0]])
        length = torch.tensor([[0.3, 0.6, 0.9]])
        original = model(documents, global_features, mask, rank, length)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            documents[:, permutation],
            global_features,
            mask[:, permutation],
            rank[:, permutation],
            length[:, permutation],
        )
        torch.testing.assert_close(
            permuted.document_logits,
            original.document_logits[:, permutation],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(permuted.log_total_loo, original.log_total_loo)

    def test_near_zero_signal_does_not_train_arbitrary_share(self) -> None:
        model = self.make_model()
        prediction = model(
            torch.randn(2, 3, 2, 8),
            torch.randn(2, 2, 8),
            torch.ones(2, 3, dtype=torch.bool),
            torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]),
            torch.ones(2, 3) * 0.5,
        )
        losses = attribution_loss(
            prediction,
            teacher_influence=torch.zeros(2, 3),
            teacher_total_loo=torch.zeros(2),
            teacher_set_shift=torch.tensor([1e-5, 1e-4]),
            document_mask=torch.ones(2, 3, dtype=torch.bool),
            minimum_total_for_share=1e-6,
        )
        self.assertEqual(int(losses["measurable_questions"].item()), 0)
        self.assertEqual(float(losses["share"].item()), 0.0)
        self.assertEqual(float(losses["rank"].item()), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_measurable_variable_k_batch_backpropagates(self) -> None:
        model = self.make_model()
        prediction = model(
            torch.randn(2, 4, 2, 8),
            torch.randn(2, 2, 8),
            torch.tensor([[True, True, False, False], [True, True, True, True]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.33, 0.67, 1.0]]),
            torch.ones(2, 4) * 0.5,
        )
        losses = attribution_loss(
            prediction,
            teacher_influence=torch.tensor(
                [[0.03, 0.01, 0.0, 0.0], [0.01, 0.08, 0.02, 0.04]]
            ),
            teacher_total_loo=torch.tensor([0.04, 0.15]),
            teacher_set_shift=torch.tensor([0.02, 0.10]),
            document_mask=torch.tensor(
                [[True, True, False, False], [True, True, True, True]]
            ),
        )
        losses["loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_relative_only_objective_excludes_set_level_losses(self) -> None:
        model = self.make_model()
        prediction = model(
            torch.randn(2, 3, 2, 8),
            torch.randn(2, 2, 8),
            torch.ones(2, 3, dtype=torch.bool),
            torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]),
            torch.ones(2, 3) * 0.5,
        )
        losses = attribution_loss(
            prediction,
            teacher_influence=torch.tensor([[0.08, 0.01, 0.03], [0.01, 0.07, 0.02]]),
            teacher_total_loo=torch.tensor([0.12, 0.10]),
            teacher_set_shift=torch.tensor([0.02, 0.04]),
            document_mask=torch.ones(2, 3, dtype=torch.bool),
            total_weight=0.0,
            share_weight=1.0,
            set_shift_weight=0.0,
            rank_weight=0.1,
        )
        torch.testing.assert_close(
            losses["loss"],
            losses["share"] + 0.1 * losses["rank"],
        )

    def test_document_order_augmentation_preserves_feature_target_alignment(self) -> None:
        batch = {
            "sample_ids": ["first", "second"],
            "document_features": torch.tensor(
                [[[[10.0]], [[20.0]], [[30.0]]], [[[40.0]], [[50.0]], [[0.0]]]]
            ),
            "document_mask": torch.tensor([[True, True, True], [True, True, False]]),
            "relative_rank": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]),
            "normalized_length": torch.tensor([[11.0, 21.0, 31.0], [41.0, 51.0, 0.0]]),
            "teacher_influence": torch.tensor([[12.0, 22.0, 32.0], [42.0, 52.0, 0.0]]),
            "teacher_total_loo": torch.tensor([66.0, 94.0]),
        }
        shuffled = permute_document_aligned_batch(batch, seed=123)
        self.assertFalse(
            torch.equal(shuffled["document_features"], batch["document_features"])
        )
        for row, count in ((0, 3), (1, 2)):
            feature_ids = shuffled["document_features"][row, :count, 0, 0]
            torch.testing.assert_close(
                shuffled["normalized_length"][row, :count], feature_ids + 1.0
            )
            torch.testing.assert_close(
                shuffled["teacher_influence"][row, :count], feature_ids + 2.0
            )
        torch.testing.assert_close(shuffled["teacher_total_loo"], batch["teacher_total_loo"])
        self.assertEqual(shuffled["sample_ids"], batch["sample_ids"])
        self.assertFalse(bool(shuffled["document_mask"][1, 2]))

    def test_training_sample_prefixes_are_nested(self) -> None:
        import hashlib
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_ids = [f"sample-{index:02d}" for index in range(16)]
            row_root = root / "rows" / "train"
            row_root.mkdir(parents=True)
            for sample_id in sample_ids:
                digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
                (row_root / f"{digest}.pt").touch()
            first = set(
                selected_paths(
                    root,
                    "train",
                    sample_ids=sample_ids,
                    maximum=4,
                    seed=42,
                )
            )
            second = set(
                selected_paths(
                    root,
                    "train",
                    sample_ids=sample_ids,
                    maximum=8,
                    seed=42,
                )
            )
            self.assertEqual(len(first), 4)
            self.assertEqual(len(second), 8)
            self.assertTrue(first < second)

    def test_masked_distribution_sums_only_present_documents(self) -> None:
        probabilities = masked_document_distribution(
            torch.tensor([[1.0, 2.0, 99.0]]),
            torch.tensor([[True, True, False]]),
        )
        self.assertAlmostEqual(float(probabilities.sum().item()), 1.0, places=6)
        self.assertEqual(float(probabilities[0, 2].item()), 0.0)

    def test_physical_batch_has_unbiased_full_repeat_empty_and_each_removal(self) -> None:
        ids = torch.arange(10)
        mapping = torch.tensor([-1, 0, 0, -1, 1, 1, -1, 2, 2, -1])
        batch = build_conditional_removal_batch(
            ids,
            mapping,
            document_count=3,
            pad_token_id=99,
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (6, 10))
        torch.testing.assert_close(batch["input_ids"][0], batch["input_ids"][1])
        self.assertEqual(int(batch["attention_mask"][2].sum().item()), 4)
        for offset, document_index in enumerate(range(3), start=3):
            present_mapping = batch["token_document_ids"][offset][
                batch["attention_mask"][offset].bool()
            ]
            self.assertFalse(bool(present_mapping.eq(document_index).any()))


if __name__ == "__main__":
    unittest.main()

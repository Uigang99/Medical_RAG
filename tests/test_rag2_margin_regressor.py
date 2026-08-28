from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from datasets import Dataset

from medrag.filtering.rag2_margin_regressor import (
    MarginRegressorConfig,
    SharedMarginRegressorConfig,
    SharedTextMarginRegressor,
    TextMarginRegressor,
)
from medrag.filtering.rag2_official import (
    build_answer_aware_filter_input,
    build_official_filter_input,
)
from scripts.train_rag2_margin_regressor import (
    NoRAGAnswerIndex,
    curriculum_direction_weights,
    select_curriculum_training_data,
)
from scripts.train_rag2_pairwise_utility_ranker import (
    pairwise_null_preference_loss,
    preference_metrics,
)


class FakeT5Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=4)
        self.embedding = nn.Embedding(16, 4)
        self.encoder = SimpleNamespace(
            block=nn.ModuleList([nn.Linear(4, 4) for _ in range(3)]),
            final_layer_norm=nn.LayerNorm(4),
        )
        self.add_module("blocks", self.encoder.block)
        self.add_module("final_layer_norm", self.encoder.final_layer_norm)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        hidden = self.embedding(input_ids)
        for block in self.encoder.block:
            hidden = torch.tanh(block(hidden))
        hidden = self.encoder.final_layer_norm(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class MarginRegressorTests(unittest.TestCase):
    def test_pairwise_null_loss_prefers_better_documents_and_anchors_zero(self) -> None:
        target = torch.tensor([0.6, 0.2, -0.5, 0.04])
        question_index = torch.tensor([0, 0, 0, 0])
        no_rag_correct = torch.tensor([False, False, False, False])
        good_prediction = torch.tensor([0.8, 0.3, -0.7, 0.0], requires_grad=True)
        bad_prediction = -good_prediction.detach()
        good_document, good_null, counters = pairwise_null_preference_loss(
            good_prediction,
            target,
            question_index,
            no_rag_correct,
            document_min_gap=0.1,
            null_min_gap=0.1,
            temperature=0.1,
        )
        bad_document, bad_null, _ = pairwise_null_preference_loss(
            bad_prediction,
            target,
            question_index,
            no_rag_correct,
            document_min_gap=0.1,
            null_min_gap=0.1,
            temperature=0.1,
        )
        self.assertLess(float(good_document.detach()), float(bad_document.detach()))
        self.assertLess(float(good_null.detach()), float(bad_null.detach()))
        self.assertEqual(counters["null_comparisons"], 3)
        (good_document + good_null).backward()
        self.assertIsNotNone(good_prediction.grad)

    def test_preference_metrics_include_document_null_and_no_rag_groups(self) -> None:
        metrics = preference_metrics(
            target=torch.tensor([0.6, -0.4, 0.3, -0.5]).numpy(),
            prediction=torch.tensor([0.8, -0.2, 0.1, -0.7]).numpy(),
            sample_ids=["q1", "q1", "q2", "q2"],
            no_rag_correct=torch.tensor([True, True, False, False]).numpy(),
            document_ranks=torch.tensor([1, 2, 1, 2]).numpy(),
            document_min_gap=0.1,
            null_min_gap=0.1,
        )
        self.assertEqual(metrics["document_pair"]["micro_accuracy"], 1.0)
        self.assertEqual(metrics["null_pair"]["micro_accuracy"], 1.0)
        self.assertEqual(metrics["combined_question_macro_accuracy"], 1.0)
        self.assertEqual(metrics["by_no_rag_state"]["no_rag_correct"]["questions"], 1)
        self.assertEqual(metrics["by_no_rag_state"]["no_rag_wrong"]["questions"], 1)

    def test_answer_aware_input_adds_prediction_without_gold_metadata(self) -> None:
        base = build_official_filter_input(
            question="Which treatment?",
            options="A) Alpha\nB) Beta",
            evidence="Evidence text.",
        )
        value = build_answer_aware_filter_input(base, "(B) Beta")
        self.assertIn("Initial answer generated without retrieved evidence: (B) Beta", value)
        self.assertIn("Evidence: Evidence text.", value)
        self.assertIn("Question: Which treatment?\nA) Alpha\nB) Beta", value)
        self.assertNotIn("gold", value.lower())

    def test_no_rag_answer_index_ignores_correctness_and_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "medqa" / "train"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(
                json.dumps({"dataset": "medqa", "split": "train", "rows": 2}),
                encoding="utf-8",
            )
            rows = [
                {
                    "sample_id": "q1",
                    "answer": "B",
                    "answer_text": "Beta",
                    "gold_answer": "A",
                    "answer_correct": False,
                },
                {
                    "sample_id": "q2",
                    "answer": "C",
                    "answer_text": "Gamma",
                    "gold_answer": "C",
                    "answer_correct": True,
                },
            ]
            (directory / "no_rag_generations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            index = NoRAGAnswerIndex(Path(temporary), "medqa", "train", False)
            self.assertEqual(index.answer_for("q1"), "(B) Beta")
            self.assertEqual(index.answer_for("q2"), "(C) Gamma")

    def test_extreme_curriculum_removes_neutral_and_balances_loss_mass(self) -> None:
        dataset = Dataset.from_dict(
            {
                "utility_target": [-0.5, -0.4, -0.1, 0.0, 0.1, 0.3],
                "sample_id": [f"q{index}" for index in range(6)],
            }
        )
        args = SimpleNamespace(
            curriculum_stage="extreme",
            extreme_threshold=0.2,
            calibration_extreme_fraction=0.5,
            seed=42,
        )
        selected, summary = select_curriculum_training_data(dataset, args)
        self.assertEqual(selected["utility_target"], [-0.5, -0.4, 0.3])
        self.assertEqual(summary["neutral"], 0)
        helpful_weight, harmful_weight = curriculum_direction_weights(selected, args)
        self.assertAlmostEqual(helpful_weight, 1.5)
        self.assertAlmostEqual(harmful_weight, 0.75)

    def test_calibration_curriculum_uses_half_extreme_and_stratified_neutral(self) -> None:
        values = [-0.4, -0.3, 0.3, 0.4, -0.19, -0.12, -0.08, -0.01, 0.01, 0.08, 0.12, 0.19]
        dataset = Dataset.from_dict(
            {
                "utility_target": values,
                "sample_id": [f"q{index}" for index in range(len(values))],
            }
        )
        args = SimpleNamespace(
            curriculum_stage="calibration",
            extreme_threshold=0.2,
            calibration_extreme_fraction=0.5,
            seed=42,
        )
        selected, summary = select_curriculum_training_data(dataset, args)
        self.assertEqual(len(selected), 8)
        self.assertEqual(summary["helpful_extreme"], 2)
        self.assertEqual(summary["harmful_extreme"], 2)
        self.assertEqual(summary["neutral"], 4)
        self.assertAlmostEqual(summary["selected_extreme_fraction"], 0.5)
        neutral = [value for value in selected["utility_target"] if abs(value) < 0.2]
        self.assertEqual(sum(-0.2 < value < -0.1 for value in neutral), 1)
        self.assertEqual(sum(-0.1 <= value < 0.0 for value in neutral), 1)
        self.assertEqual(sum(0.0 <= value < 0.1 for value in neutral), 1)
        self.assertEqual(sum(0.1 <= value < 0.2 for value in neutral), 1)

    @patch("medrag.filtering.rag2_margin_regressor.T5EncoderModel.from_pretrained")
    def test_forward_is_bounded_and_text_only(self, mocked: unittest.mock.Mock) -> None:
        mocked.return_value = FakeT5Encoder()
        model = TextMarginRegressor(
            MarginRegressorConfig(
                base_model_name_or_path="fake",
                hidden_size=8,
                trainable_encoder_layers=1,
            )
        )
        output = model(
            input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )
        self.assertEqual(tuple(output["utility_score"].shape), (2,))
        self.assertTrue(bool(torch.all(output["utility_score"].abs() <= 1.0)))
        trainable_blocks = [
            any(parameter.requires_grad for parameter in block.parameters())
            for block in model.encoder.encoder.block
        ]
        self.assertEqual(trainable_blocks, [False, False, True])

    @patch("medrag.filtering.rag2_margin_regressor.T5EncoderModel.from_pretrained")
    def test_trainable_round_trip(self, mocked: unittest.mock.Mock) -> None:
        mocked.side_effect = lambda *args, **kwargs: FakeT5Encoder()
        config = MarginRegressorConfig(
            base_model_name_or_path="fake",
            hidden_size=8,
            trainable_encoder_layers=1,
        )
        model = TextMarginRegressor(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model.save_trainable(path)
            loaded = TextMarginRegressor.from_trainable(path)
            self.assertEqual(
                json.loads((path / "margin_regressor_config.json").read_text())["model_version"],
                config.model_version,
            )
            expected = model.trainable_state_dict()
            actual = loaded.trainable_state_dict()
            self.assertEqual(set(expected), set(actual))
            for name in expected:
                self.assertTrue(torch.equal(expected[name], actual[name]), name)

    @patch("medrag.filtering.rag2_margin_regressor.T5EncoderModel.from_pretrained")
    def test_shared_forward_uses_one_unbounded_head_and_restores_margin_scale(
        self, mocked: unittest.mock.Mock
    ) -> None:
        mocked.return_value = FakeT5Encoder()
        model = SharedTextMarginRegressor(
            SharedMarginRegressorConfig(
                base_model_name_or_path="fake",
                hidden_size=8,
                trainable_encoder_layers=1,
                margin_scale=10.0,
            )
        )
        output = model(
            input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )
        self.assertEqual(tuple(output["margin"].shape), (2,))
        self.assertTrue(torch.allclose(output["margin"], output["scaled_margin"] * 10.0))

    @patch("medrag.filtering.rag2_margin_regressor.T5EncoderModel.from_pretrained")
    def test_shared_trainable_round_trip(self, mocked: unittest.mock.Mock) -> None:
        mocked.side_effect = lambda *args, **kwargs: FakeT5Encoder()
        config = SharedMarginRegressorConfig(
            base_model_name_or_path="fake",
            hidden_size=8,
            trainable_encoder_layers=1,
            margin_scale=10.0,
        )
        model = SharedTextMarginRegressor(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model.save_trainable(path)
            loaded = SharedTextMarginRegressor.from_trainable(path)
            expected = model.trainable_state_dict()
            actual = loaded.trainable_state_dict()
            self.assertEqual(set(expected), set(actual))
            for name in expected:
                self.assertTrue(torch.equal(expected[name], actual[name]), name)


if __name__ == "__main__":
    unittest.main()

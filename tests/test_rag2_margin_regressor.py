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
from scripts.train_rag2_margin_regressor import (
    curriculum_direction_weights,
    select_curriculum_training_data,
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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import T5Config

from medrag.filtering.rag2_two_head import Rag2TwoHeadFilterModel
from scripts.train_rag2_filter_model_two_head import (
    attach_sampling_weights,
    label_targets,
    select_thresholds,
    three_class_metrics,
)


class Rag2TwoHeadFilterTests(unittest.TestCase):
    @staticmethod
    def tiny_model() -> Rag2TwoHeadFilterModel:
        config = T5Config(
            vocab_size=64,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_heads=2,
            d_kv=8,
            dropout_rate=0.0,
        )
        config.rag2_two_head_dropout = 0.0
        return Rag2TwoHeadFilterModel(config)

    def test_label_targets_mask_discard_direction(self) -> None:
        self.assertEqual(label_targets("helpful", True), [1, 1, 1])
        self.assertEqual(label_targets("not helpful", False), [1, 0, 0])
        self.assertEqual(label_targets("discard", False), [0, -100, 0])

    def test_discard_loss_does_not_use_utility_head(self) -> None:
        torch.manual_seed(0)
        model = self.tiny_model().eval()
        input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[0, -100, 1], [0, -100, 0]])
        first = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        with torch.no_grad():
            model.utility_head.weight.fill_(100.0)
            model.utility_head.bias.fill_(-100.0)
        second = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        self.assertTrue(torch.allclose(first, second))

    def test_hierarchical_sampler_has_requested_class_mass(self) -> None:
        dataset = Dataset.from_dict({"class_id": [0] * 6 + [1] * 2 + [2] * 12})
        weighted, summary = attach_sampling_weights(dataset, "hierarchical_balanced")
        weights = np.asarray(weighted["sample_weight"])
        classes = np.asarray(weighted["class_id"])
        mass = {
            name: float(weights[classes == index].sum() / weights.sum())
            for name, index in (("helpful", 0), ("not helpful", 1), ("discard", 2))
        }
        self.assertEqual(summary["expected_sampled_ratios"], {"helpful": 0.25, "not helpful": 0.25, "discard": 0.5})
        self.assertAlmostEqual(mass["helpful"], 0.25)
        self.assertAlmostEqual(mass["not helpful"], 0.25)
        self.assertAlmostEqual(mass["discard"], 0.50)

    def test_metrics_apply_decisive_gate_before_helpful_head(self) -> None:
        # Rows: true H, NH, D.  The Discard row has a high Helpful head score,
        # but the low decisive score must still make it Discard.
        logits = np.asarray(
            [
                [-2.0, 2.0, -2.0, 2.0],
                [-2.0, 2.0, 2.0, -2.0],
                [2.0, -2.0, -2.0, 2.0],
            ]
        )
        labels = np.asarray([[1, 1, 1], [1, 0, 0], [0, -100, 0]])
        metrics = three_class_metrics(logits, labels, decisive_threshold=0.5, helpful_threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["discard_pass_rate"], 0.0)

    def test_threshold_selection_respects_discard_contamination_limit(self) -> None:
        logits = np.asarray(
            [
                [-3.0, 3.0, -3.0, 3.0],
                [-3.0, 3.0, -2.0, 2.0],
                [3.0, -3.0, -3.0, 3.0],
                [3.0, -3.0, 2.0, -2.0],
            ]
        )
        labels = np.asarray([[1, 1, 1], [1, 1, 0], [0, -100, 1], [0, -100, 0]])

        class Args:
            threshold_min = 0.25
            threshold_max = 0.75
            threshold_step = 0.25
            discard_contamination_limit = 0.10

        selection = select_thresholds(logits, labels, Args())
        self.assertLessEqual(
            selection["selected"]["discard_contamination_in_predicted_helpful"],
            0.10,
        )

    def test_save_and_reload_round_trip(self) -> None:
        torch.manual_seed(7)
        model = self.tiny_model().eval()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        expected = model(input_ids=input_ids, attention_mask=attention_mask).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_two_head_pretrained(Path(tmp), metadata={"theta_decisive": 0.6})
            loaded = Rag2TwoHeadFilterModel.from_two_head_checkpoint(tmp).eval()
            actual = loaded(input_ids=input_ids, attention_mask=attention_mask).logits
        self.assertTrue(torch.allclose(expected, actual))


if __name__ == "__main__":
    unittest.main()

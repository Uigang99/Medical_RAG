from __future__ import annotations

import unittest

import numpy as np
import torch
from datasets import Dataset

from scripts.train_rag2_margin_regressor import (
    ShardBatchSampler,
    ShardQuestionBatchSampler,
    action_metrics,
    utility_action_labels,
    within_question_pairwise_loss,
)


class UtilityActionMetricsTest(unittest.TestCase):
    def test_threshold_mapping(self) -> None:
        values = np.asarray([0.2, 0.1, 0.099, 0.0, -0.099, -0.1, -0.2])
        self.assertEqual(
            utility_action_labels(values, 0.1).tolist(),
            [0, 0, 1, 1, 1, 2, 2],
        )

    def test_perfect_predictions_have_perfect_action_metrics(self) -> None:
        target = np.asarray([0.4, 0.2, 0.0, 0.05, -0.2, -0.4])
        metrics = action_metrics(target, target.copy(), 0.1)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_shard_sampler_packs_all_remainders(self) -> None:
        dataset = Dataset.from_dict(
            {"trace_shard": ["a"] * 3 + ["b"] * 3 + ["c"] * 3}
        )
        sampler = ShardBatchSampler(dataset, batch_size=4, seed=7, shuffle=False)
        batches = list(sampler)
        self.assertEqual(len(sampler), 3)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(9)))

    def test_question_sampler_keeps_complete_questions(self) -> None:
        dataset = Dataset.from_dict(
            {
                "sample_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
                "trace_shard": ["a", "a", "a", "a", "b", "b"],
            }
        )
        sampler = ShardQuestionBatchSampler(dataset, questions_per_batch=2, seed=7, shuffle=False)
        batches = list(sampler)
        self.assertEqual(len(sampler), 2)
        self.assertEqual([len(batch) for batch in batches], [4, 2])
        for batch in batches:
            samples = [dataset[index]["sample_id"] for index in batch]
            for sample in set(samples):
                self.assertEqual(samples.count(sample), 2)

    def test_pairwise_loss_rewards_correct_order(self) -> None:
        target = torch.tensor([0.4, 0.0, -0.3, 0.2, -0.2])
        group = torch.tensor([0, 0, 0, 1, 1])
        ordered, questions, comparisons = within_question_pairwise_loss(
            target,
            target,
            group,
            min_target_gap=0.1,
            temperature=0.1,
        )
        reversed_loss, _, _ = within_question_pairwise_loss(
            -target,
            target,
            group,
            min_target_gap=0.1,
            temperature=0.1,
        )
        self.assertEqual(questions, 2)
        self.assertGreater(comparisons, 0)
        self.assertLess(float(ordered), float(reversed_loss))


if __name__ == "__main__":
    unittest.main()

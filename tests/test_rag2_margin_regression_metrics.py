from __future__ import annotations

import unittest

import numpy as np
from datasets import Dataset

from scripts.train_rag2_margin_regressor import (
    ShardBatchSampler,
    action_metrics,
    utility_action_labels,
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


if __name__ == "__main__":
    unittest.main()

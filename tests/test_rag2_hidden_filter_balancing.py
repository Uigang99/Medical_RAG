from __future__ import annotations

import unittest

from datasets import Dataset

from scripts.train_rag2_hidden_feature_filter import (
    add_no_rag_state_training_weights,
    make_no_rag_state_balanced_validation,
)


def balance_dataset(groups: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "balance_group": groups,
            "feature_shard_index": list(range(len(groups))),
        }
    )


class NoRagStateBalancingTests(unittest.TestCase):
    def test_loss_mass_is_equal_by_state_and_preserves_labels_within_state(self) -> None:
        dataset = balance_dataset(
            [
                "no_rag_correct__helpful",
                "no_rag_correct__helpful",
                "no_rag_correct__helpful",
                "no_rag_correct__not_helpful",
                "no_rag_wrong__helpful",
                "no_rag_wrong__not_helpful",
            ]
        )
        weighted, report = add_no_rag_state_training_weights(dataset)

        self.assertEqual(len(weighted), len(dataset))
        self.assertEqual(report["effective_state_fraction"], {
            "no_rag_correct": 0.5,
            "no_rag_wrong": 0.5,
        })
        self.assertEqual(
            report["natural_label_fraction_within_state"],
            report["effective_label_fraction_within_state"],
        )
        self.assertEqual(
            report["effective_label_fraction_within_state"]["no_rag_correct"],
            {"helpful": 0.75, "not_helpful": 0.25},
        )
        self.assertEqual(
            report["effective_label_fraction_within_state"]["no_rag_wrong"],
            {"helpful": 0.5, "not_helpful": 0.5},
        )
        self.assertEqual(weighted[0]["_sample_weight"], 0.75)
        self.assertEqual(weighted[-1]["_sample_weight"], 1.5)

    def test_validation_balances_states_without_label_stratification(self) -> None:
        dataset = balance_dataset(
            [
                "no_rag_correct__helpful",
                "no_rag_correct__helpful",
                "no_rag_correct__helpful",
                "no_rag_correct__not_helpful",
                "no_rag_wrong__helpful",
                "no_rag_wrong__not_helpful",
            ]
        )
        balanced, report = make_no_rag_state_balanced_validation(dataset, seed=42)

        self.assertEqual(len(balanced), 4)
        self.assertEqual(report["selected_per_state"], 2)
        selected_states = [value.split("__", 1)[0] for value in balanced["balance_group"]]
        self.assertEqual(selected_states.count("no_rag_correct"), 2)
        self.assertEqual(selected_states.count("no_rag_wrong"), 2)


if __name__ == "__main__":
    unittest.main()

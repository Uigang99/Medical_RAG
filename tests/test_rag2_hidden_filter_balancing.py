from __future__ import annotations

import unittest

import torch
from datasets import Dataset

from scripts.train_rag2_hidden_feature_filter import (
    QuestionGroupedBatchSampler,
    add_no_rag_state_training_weights,
    make_no_rag_state_balanced_validation,
    summarize_pairwise_supervision,
    within_question_pairwise_loss,
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


class PairwiseRankingTests(unittest.TestCase):
    def test_question_batch_sampler_never_splits_a_question(self) -> None:
        dataset = Dataset.from_dict(
            {
                "feature_shard_index": [0, 0, 0, 0, 0, 0, 1, 1],
                "feature_question_row": [10, 10, 11, 11, 11, 12, 20, 20],
            }
        )
        sampler = QuestionGroupedBatchSampler(dataset, batch_size=5, seed=42)
        batches = list(iter(sampler))
        index_to_batch = {
            index: batch_index
            for batch_index, batch in enumerate(batches)
            for index in batch
        }
        self.assertEqual(sorted(index_to_batch), list(range(len(dataset))))
        self.assertTrue(all(len(batch) <= 5 for batch in batches))
        for question_row in {10, 11, 12, 20}:
            indices = [
                index
                for index, value in enumerate(dataset["feature_question_row"])
                if value == question_row
            ]
            self.assertEqual(len({index_to_batch[index] for index in indices}), 1)

    def test_pairwise_summary_counts_mixed_and_single_label_questions(self) -> None:
        dataset = Dataset.from_dict(
            {
                "feature_shard_index": [0] * 8,
                "feature_question_row": [1, 1, 2, 2, 3, 3, 4, 4],
                "target_name": [
                    "helpful",
                    "not helpful",
                    "helpful",
                    "helpful",
                    "not helpful",
                    "not helpful",
                    "helpful",
                    "not helpful",
                ],
                "balance_group": [
                    "no_rag_correct__helpful",
                    "no_rag_correct__not_helpful",
                    "no_rag_correct__helpful",
                    "no_rag_correct__helpful",
                    "no_rag_wrong__not_helpful",
                    "no_rag_wrong__not_helpful",
                    "no_rag_wrong__helpful",
                    "no_rag_wrong__not_helpful",
                ],
            }
        )
        report = summarize_pairwise_supervision(dataset)
        self.assertEqual(
            report["question_label_configurations"],
            {"mixed": 2, "helpful_only": 1, "not_helpful_only": 1},
        )
        self.assertEqual(
            report["mixed_questions_by_no_rag_state"],
            {"no_rag_correct": 1, "no_rag_wrong": 1},
        )
        self.assertEqual(
            report["pairwise_no_rag_state_weights"],
            {"no_rag_correct": 1.0, "no_rag_wrong": 1.0},
        )

    def test_pairwise_loss_averages_each_question_before_state_weighting(self) -> None:
        utility = torch.tensor([2.0, 0.0, 3.0, 0.0, 5.0])
        groups = torch.tensor([0, 0, 1, 1, 2])
        targets = torch.tensor([1, 0, 1, 0, 1])
        states = torch.tensor([0, 0, 1, 1, 0])
        weights = torch.tensor([0.5, 1.5])
        loss, questions, comparisons = within_question_pairwise_loss(
            utility, groups, targets, states, weights
        )
        expected = (
            0.5 * torch.nn.functional.softplus(torch.tensor(-2.0))
            + 1.5 * torch.nn.functional.softplus(torch.tensor(-3.0))
        ) / 2.0
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(questions, 2)
        self.assertEqual(comparisons, 2)


if __name__ == "__main__":
    unittest.main()

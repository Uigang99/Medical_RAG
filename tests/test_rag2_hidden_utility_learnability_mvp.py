from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_rag2_hidden_utility_learnability_mvp import (  # noqa: E402
    SplitFeatures,
    feature_view,
    no_rag_state,
    no_rag_group_weights,
    permuted_document_features,
    shuffled_targets_within_question,
    within_question_ranking,
)


class HiddenUtilityLearnabilityMvpTests(unittest.TestCase):
    def test_no_rag_state_uses_answer_transition_prefix(self) -> None:
        self.assertEqual(no_rag_state({"answer_transition_audit_only": "C->C"}), 0)
        self.assertEqual(no_rag_state({"answer_transition_audit_only": "W->C"}), 1)
        with self.assertRaises(ValueError):
            no_rag_state({"answer_transition_audit_only": "unknown"})

    def test_feature_views_keep_the_declared_blocks(self) -> None:
        data = SplitFeatures(
            h0=np.asarray([[1.0, 2.0]], dtype=np.float32),
            delta=np.asarray([[3.0, 4.0]], dtype=np.float32),
            text=np.asarray([[5.0, 6.0, 7.0]], dtype=np.float32),
            score=np.asarray([0.5], dtype=np.float32),
            label=np.asarray([1], dtype=np.int64),
            state=np.asarray([0], dtype=np.int64),
            question=np.asarray([0], dtype=np.int64),
        )
        np.testing.assert_array_equal(feature_view(data, "h0_delta"), [[1, 2, 3, 4]])
        np.testing.assert_array_equal(
            feature_view(data, "text_h0_delta"), [[5, 6, 7, 1, 2, 3, 4]]
        )

    def test_within_question_shuffle_preserves_each_question_distribution(self) -> None:
        values = np.asarray([0, 1, 1, 10, 11, 12], dtype=np.int64)
        questions = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        shuffled = shuffled_targets_within_question(values, questions, seed=7)
        for question_id in (0, 1):
            mask = questions == question_id
            self.assertEqual(sorted(values[mask].tolist()), sorted(shuffled[mask].tolist()))

    def test_no_rag_group_weights_assign_equal_total_mass(self) -> None:
        state = np.asarray([0, 0, 0, 1], dtype=np.int64)
        weight = no_rag_group_weights(state)
        self.assertAlmostEqual(float(weight[state == 0].sum()), 2.0)
        self.assertAlmostEqual(float(weight[state == 1].sum()), 2.0)
        self.assertAlmostEqual(float(weight.mean()), 1.0)

    def test_document_permutation_never_moves_h0_block(self) -> None:
        # [h0(2), delta(2)] for two questions with three documents each.
        features = np.asarray(
            [
                [1, 2, 10, 11],
                [1, 2, 20, 21],
                [1, 2, 30, 31],
                [3, 4, 40, 41],
                [3, 4, 50, 51],
                [3, 4, 60, 61],
            ],
            dtype=np.float32,
        )
        questions = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        permuted = permuted_document_features(
            features,
            "h0_delta",
            questions,
            text_dim=0,
            hidden_dim=2,
            seed=13,
        )
        np.testing.assert_array_equal(permuted[:, :2], features[:, :2])
        for question_id in (0, 1):
            mask = questions == question_id
            self.assertEqual(
                sorted(map(tuple, features[mask, 2:].tolist())),
                sorted(map(tuple, permuted[mask, 2:].tolist())),
            )

    def test_within_question_ranking_excludes_single_class_questions(self) -> None:
        target = np.asarray([1, 0, 1, 1], dtype=np.int64)
        prediction = np.asarray([0.9, 0.1, 0.2, 0.3], dtype=np.float32)
        question = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = within_question_ranking(target, prediction, question)
        self.assertEqual(result["mixed_questions"], 1)
        self.assertEqual(result["comparisons"], 1)
        self.assertEqual(result["pair_accuracy_micro"], 1.0)


if __name__ == "__main__":
    unittest.main()

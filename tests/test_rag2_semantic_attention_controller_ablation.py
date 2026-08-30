from __future__ import annotations

import unittest

from scripts.evaluate_rag2_semantic_attention_controller_ablation import compare_conditions


class SemanticAttentionControllerAblationTest(unittest.TestCase):
    def test_condition_comparison_counts_answer_transitions(self) -> None:
        base = {
            "policy": "base",
            "predictions": [0, 1, 2, 3],
            "gold_options": [0, 0, 2, 0],
        }
        target = {
            "policy": "target",
            "predictions": [1, 0, 2, 0],
            "gold_options": [0, 0, 2, 0],
        }
        result = compare_conditions(base, target)
        self.assertEqual(result["wrong_to_correct"], 2)
        self.assertEqual(result["correct_to_wrong"], 1)
        self.assertEqual(result["net_answer_gain"], 1)
        self.assertEqual(result["changed_predictions"], 3)
        self.assertAlmostEqual(result["accuracy_delta"], 0.25)

    def test_condition_comparison_rejects_different_gold_order(self) -> None:
        with self.assertRaises(RuntimeError):
            compare_conditions(
                {"policy": "a", "predictions": [0], "gold_options": [0]},
                {"policy": "b", "predictions": [0], "gold_options": [1]},
            )


if __name__ == "__main__":
    unittest.main()

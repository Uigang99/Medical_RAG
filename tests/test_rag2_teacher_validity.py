from __future__ import annotations

import math
import unittest

from scripts.evaluate_rag2_teacher_validity import (
    comparison_metrics,
    e2e_metrics,
    logprobs_to_probabilities,
)


class TeacherValidityTest(unittest.TestCase):
    def test_choice_logprobs_are_renormalized_over_four_options(self) -> None:
        probabilities = logprobs_to_probabilities(
            {"A": -0.1, "B": -1.0, "C": -2.0, "D": -3.0}
        )
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(max(range(4), key=probabilities.__getitem__), 0)

    def test_end_to_end_metrics_detect_removal_flip_and_repeat_noise(self) -> None:
        def variant(name: str, logits: list[float]) -> dict[str, object]:
            return {
                "variant": name,
                "choice_logprobs": dict(zip(("A", "B", "C", "D"), logits, strict=True)),
            }

        full = [-0.1, -2.0, -3.0, -4.0]
        variants = [variant("full", full), variant("repeat", full)]
        variants.append(variant("remove_0", [-3.0, -0.1, -2.0, -4.0]))
        variants.extend(variant(f"remove_{index}", full) for index in range(1, 8))
        metrics = e2e_metrics({"sample_id": "sample", "variants": variants})
        self.assertEqual(metrics["flips"], [True, False, False, False, False, False, False, False])
        self.assertGreater(metrics["jsd"][0], 0.0)
        self.assertAlmostEqual(metrics["repeat_noise_jsd"], 0.0)

    def test_comparison_metrics_respect_reference_signal_threshold(self) -> None:
        rows = [
            {
                "candidate": [8.0, 1.0, 0.0],
                "reference": [0.8, 0.1, 0.0],
            },
            {
                "candidate": [0.0, 1.0, 2.0],
                "reference": [1e-8, 2e-8, 3e-8],
            },
        ]
        metrics = comparison_metrics(rows, "candidate", "reference", threshold=1e-4)
        self.assertEqual(metrics["questions"], 1)
        self.assertTrue(math.isclose(float(metrics["top1_agreement"]), 1.0))
        self.assertTrue(math.isclose(float(metrics["median_spearman"]), 1.0))


if __name__ == "__main__":
    unittest.main()

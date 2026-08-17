from __future__ import annotations

import unittest

from scripts.materialize_rag2_preanswer_hidden_labels import classify


class HiddenThresholdLabelTests(unittest.TestCase):
    def test_original_symmetric_neutral_mode_is_preserved(self) -> None:
        self.assertEqual(classify(0.5, 0.4), "Helpful")
        self.assertEqual(classify(-0.5, 0.4), "Not Helpful")
        self.assertEqual(classify(0.4, 0.4), "Neutral")
        self.assertEqual(classify(-0.4, 0.4), "Neutral")

    def test_positive_vs_rest_is_strict_binary_at_tau_point_four(self) -> None:
        self.assertEqual(classify(0.400001, 0.4, "positive_vs_rest"), "Helpful")
        self.assertEqual(classify(0.4, 0.4, "positive_vs_rest"), "Not Helpful")
        self.assertEqual(classify(0.0, 0.4, "positive_vs_rest"), "Not Helpful")
        self.assertEqual(classify(-1.0, 0.4, "positive_vs_rest"), "Not Helpful")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported label mode"):
            classify(0.5, 0.4, "unsupported")


if __name__ == "__main__":
    unittest.main()

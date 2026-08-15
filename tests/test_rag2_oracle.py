from __future__ import annotations

import unittest

from medrag.rag2_oracle import (
    canonicalize_rag2_labels,
    deterministic_question_sample,
    hidden_policy_name,
    oracle_document_is_helpful,
)


class Rag2OracleTests(unittest.TestCase):
    def test_deterministic_sample_is_input_order_independent(self) -> None:
        values = ["q4", "q1", "q3", "q2"]
        first = deterministic_question_sample(values, dataset="medqa", limit=2, seed=42)
        second = deterministic_question_sample(reversed(values), dataset="medqa", limit=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_rag2_duplicate_prefers_only_quality_passing_replacement(self) -> None:
        rows = [
            {
                "sample_id": "q1",
                "doc_rank": 1,
                "doc_stable_id": "d1",
                "pseudo_label": "Excluded",
                "quality_pass": False,
            },
            {
                "sample_id": "q1",
                "doc_rank": 1,
                "doc_stable_id": "d1",
                "pseudo_label": "Helpful",
                "quality_pass": True,
            },
        ]
        labels, audit = canonicalize_rag2_labels(
            rows, selected_sample_ids={"q1"}, max_rank=8
        )
        self.assertEqual(labels["q1"]["d1"], "Helpful")
        self.assertEqual(audit["duplicate_keys"], 1)
        self.assertEqual(audit["valid_replacements"], 1)

    def test_multiple_valid_duplicate_is_rejected(self) -> None:
        rows = [
            {
                "sample_id": "q1",
                "doc_rank": 1,
                "doc_stable_id": "d1",
                "pseudo_label": label,
                "quality_pass": True,
            }
            for label in ("Helpful", "Not Helpful")
        ]
        with self.assertRaisesRegex(ValueError, "Multiple quality-passing"):
            canonicalize_rag2_labels(rows, selected_sample_ids={"q1"}, max_rank=8)

    def test_oracle_policy_boundaries(self) -> None:
        self.assertTrue(
            oracle_document_is_helpful(
                policy="rag2", rag2_label="Helpful", hidden_projection=None, hidden_threshold=None
            )
        )
        self.assertFalse(
            oracle_document_is_helpful(
                policy="rag2", rag2_label="Discard", hidden_projection=None, hidden_threshold=None
            )
        )
        self.assertTrue(
            oracle_document_is_helpful(
                policy="hidden_tau_0p2",
                rag2_label=None,
                hidden_projection=0.2001,
                hidden_threshold=0.2,
            )
        )
        self.assertFalse(
            oracle_document_is_helpful(
                policy="hidden_tau_0p2",
                rag2_label=None,
                hidden_projection=0.2,
                hidden_threshold=0.2,
            )
        )
        self.assertEqual(hidden_policy_name(0.2), "hidden_tau_0p2")


if __name__ == "__main__":
    unittest.main()

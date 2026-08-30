from __future__ import annotations

import unittest

from medrag.filtering.rag2_official import build_official_filter_input
from medrag.filtering.semantic_features import official_evidence_span


class SemanticFeaturesTest(unittest.TestCase):
    def test_official_evidence_span_selects_only_evidence(self) -> None:
        prompt = build_official_filter_input(
            question="Which option is correct?",
            options="A) one\nB) two",
            evidence="alpha beta gamma",
        )
        start, end = official_evidence_span(prompt)
        self.assertEqual(prompt[start:end], "alpha beta gamma")

    def test_nonofficial_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            official_evidence_span("Evidence: incomplete")


if __name__ == "__main__":
    unittest.main()

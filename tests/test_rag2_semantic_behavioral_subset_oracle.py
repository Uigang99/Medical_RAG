from __future__ import annotations

import unittest

from scripts.materialize_rag2_semantic_behavioral_subset_oracle import (
    choose_best_subset,
    subset_document_ids,
)


class SemanticBehavioralSubsetOracleTests(unittest.TestCase):
    def test_subset_document_ids_preserves_rerank_order(self) -> None:
        candidates = [
            {"doc_stable_id": "d1"},
            {"doc_stable_id": "d2"},
            {"doc_stable_id": "d3"},
        ]
        self.assertEqual(subset_document_ids(candidates, 0b101), ["d1", "d3"])

    def test_best_direct_is_computed_separately_from_broad_optimum(self) -> None:
        subsets = [
            {
                "mask": 0,
                "selected_document_ids": [],
                "selected_document_ranks": [],
                "gold_margin": -1.0,
            },
            {
                "mask": 1,
                "selected_document_ids": ["direct"],
                "selected_document_ranks": [1],
                "gold_margin": 0.5,
            },
            {
                "mask": 2,
                "selected_document_ids": ["supporting"],
                "selected_document_ranks": [2],
                "gold_margin": 1.0,
            },
            {
                "mask": 3,
                "selected_document_ids": ["direct", "supporting"],
                "selected_document_ranks": [1, 2],
                "gold_margin": 1.5,
            },
        ]
        direct = choose_best_subset(subsets, {"direct"})
        broad = choose_best_subset(subsets, {"direct", "supporting"})
        self.assertEqual(direct["selected_document_ids"], ["direct"])
        self.assertEqual(broad["selected_document_ids"], ["direct", "supporting"])

    def test_empty_subset_allows_internal_knowledge_fallback(self) -> None:
        subsets = [
            {
                "mask": 0,
                "selected_document_ids": [],
                "selected_document_ranks": [],
                "gold_margin": 0.25,
            },
            {
                "mask": 1,
                "selected_document_ids": ["direct"],
                "selected_document_ranks": [1],
                "gold_margin": -0.5,
            },
        ]
        best = choose_best_subset(subsets, {"direct"})
        self.assertEqual(best["selected_document_ids"], [])

    def test_tie_break_prefers_fewer_then_earlier_documents(self) -> None:
        subsets = [
            {
                "mask": 3,
                "selected_document_ids": ["d1", "d2"],
                "selected_document_ranks": [1, 2],
                "gold_margin": 1.0,
            },
            {
                "mask": 2,
                "selected_document_ids": ["d2"],
                "selected_document_ranks": [2],
                "gold_margin": 1.0,
            },
            {
                "mask": 1,
                "selected_document_ids": ["d1"],
                "selected_document_ranks": [1],
                "gold_margin": 1.0,
            },
        ]
        best = choose_best_subset(subsets, {"d1", "d2"})
        self.assertEqual(best["selected_document_ids"], ["d1"])


if __name__ == "__main__":
    unittest.main()

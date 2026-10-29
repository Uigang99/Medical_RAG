from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from medrag.retrieval.inner_product_fourway import derive_candidate_ids, source_topk_inner_product
from scripts.build_question_only_fourway_retrieval import _ce_rank
from medrag.retrieval.corpus_filter import select_rows


class FakeFlatIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.ntotal = len(vectors)
        self.reconstructions = 0

    def reconstruct_n(self, start: int, size: int) -> np.ndarray:
        self.reconstructions += 1
        return self.vectors[start:start + size]


class FakeSource:
    def __init__(self, vectors: np.ndarray):
        self.source = "fake"
        self.rows = len(vectors)
        self.index = FakeFlatIndex(vectors)
        self._is_sharded = False
        self._shards = []
        self.cache_index = False


class InnerProductFourwayTests(unittest.TestCase):
    def test_three_source_quotas_and_global_refill(self):
        pools = {s: (np.arange(32, 0, -1, dtype=np.float32)[None] + shift,
                     np.arange(32, dtype=np.int64)[None])
                 for s, shift in (("pubmed", 100), ("cpg", 0), ("textbooks", -100))}
        balanced, global_hits = derive_candidate_ids(
            pools, 0, balanced_quotas={"pubmed": 11, "cpg": 11, "textbooks": 10})
        self.assertEqual(len(balanced), 32)
        self.assertEqual([sum(x[0] == s for x in balanced) for s in pools], [11, 11, 10])
        self.assertEqual([x[0] for x in global_hits], ["pubmed"] * 32)
        self.assertEqual([x[1] for x in global_hits], list(range(32)))
        with self.assertRaises(ValueError):
            derive_candidate_ids(pools, 0, balanced_quotas={"pubmed": 32})

    def test_filter_precedes_topk_and_preserves_original_ids(self):
        source = FakeSource(np.array([[100., 0.], [90., 0.], [3., 0.], [2., 0.], [1., 0.]], dtype=np.float32))
        keep = np.array([False, False, True, False, True])
        with tempfile.TemporaryDirectory() as directory:
            seen = []
            scores, ids = source_topk_inner_product(
                source, np.array([[1., 0.]], dtype=np.float32), k=2,
                vector_chunk_size=2, query_batch_size=1, device=torch.device("cpu"),
                checkpoint_path=Path(directory) / "filtered.npz", contract="filtered",
                eligibility=keep, on_vectors=seen.append)
            self.assertEqual(ids.tolist(), [[2, 4]])
            self.assertEqual(scores.tolist(), [[3., 1.]])
            self.assertEqual(sum(seen), 5)
            self.assertEqual(source.index.reconstructions, 2)

    def test_corpus_selection_uses_pmid_and_pubmed_only_title_rule(self):
        pmids = np.array([0, 10, 10, 20, 30], dtype=np.uint32)
        flags = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
        allowed = np.array([10, 20], dtype=np.uint32)
        self.assertEqual(select_rows("pubmed", pmids, flags, allowed).tolist(), [False, True, False, True, False])
        self.assertEqual(select_rows("pmc", pmids, flags, allowed).tolist(), [False, True, True, True, False])
        self.assertTrue(select_rows("cpg", pmids, flags, allowed).all())
        self.assertTrue(select_rows("textbooks", pmids, flags, allowed).all())

    def test_inner_product_preserves_norms_and_resumes(self):
        # The larger-norm second vector wins by inner product, unlike cosine.
        source = FakeSource(np.array([[1., 0.], [2., 10.], [0., 1.]], dtype=np.float32))
        queries = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.npz"
            scores, ids = source_topk_inner_product(
                source, queries, k=2, vector_chunk_size=2, query_batch_size=1,
                device=torch.device("cpu"), checkpoint_path=checkpoint, contract="test"
            )
            self.assertEqual(ids.tolist(), [[1, 0], [1, 2]])
            self.assertAlmostEqual(scores[0, 0], 2., places=5)
            calls = source.index.reconstructions
            again = source_topk_inner_product(
                source, queries, k=2, vector_chunk_size=2, query_batch_size=1,
                device=torch.device("cpu"), checkpoint_path=checkpoint, contract="test"
            )
            self.assertEqual(source.index.reconstructions, calls)
            np.testing.assert_array_equal(again[1], ids)

    def test_balanced_and_global_are_distinct_exact_views(self):
        per_source = {
            "a": (np.array([[.99, .97, .96, .95]], dtype=np.float32),
                  np.array([[0, 1, 2, 3]], dtype=np.int64)),
            "b": (np.array([[.90, .89, .88, .87]], dtype=np.float32),
                  np.array([[0, 1, 2, 3]], dtype=np.int64)),
        }
        balanced, global_hits = derive_candidate_ids(per_source, 0, balanced_per_source=2, global_k=4)
        self.assertEqual([hit[0] for hit in balanced], ["a", "a", "b", "b"])
        self.assertEqual([hit[0] for hit in global_hits], ["a", "a", "a", "a"])
        self.assertEqual([hit[3] for hit in global_hits], [1, 2, 3, 4])

    def test_ce_rank_preserves_dense_rank_and_score(self):
        dense = [{"source": "a", "local_id": 2, "dense_rank": 1, "inner_product_score": .9},
                 {"source": "b", "local_id": 3, "dense_rank": 2, "inner_product_score": .8}]
        reranked = _ce_rank(dense, {("a", 2): -.5, ("b", 3): 1.2})
        self.assertEqual([doc["local_id"] for doc in reranked], [3, 2])
        self.assertEqual([doc["dense_rank"] for doc in reranked], [2, 1])
        self.assertEqual([doc["ce_rank"] for doc in reranked], [1, 2])
        self.assertEqual([doc["rerank_score"] for doc in reranked], [1.2, -.5])


if __name__ == "__main__":
    unittest.main()

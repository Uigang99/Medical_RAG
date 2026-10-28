from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from medrag.retrieval.cosine_fourway import derive_candidate_ids, source_topk_cosine
from scripts.build_question_only_fourway_retrieval import _ce_rank


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


class CosineFourwayTests(unittest.TestCase):
    def test_cosine_normalizes_both_sides_and_resumes(self):
        # The large-norm second vector wins by inner product but loses by cosine.
        source = FakeSource(np.array([[1., 0.], [2., 10.], [0., 1.]], dtype=np.float32))
        queries = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.npz"
            scores, ids = source_topk_cosine(
                source, queries, k=2, vector_chunk_size=2, query_batch_size=1,
                device=torch.device("cpu"), checkpoint_path=checkpoint, contract="test"
            )
            self.assertEqual(ids.tolist(), [[0, 1], [2, 1]])
            self.assertAlmostEqual(scores[0, 0], 1., places=5)
            calls = source.index.reconstructions
            again = source_topk_cosine(
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
        dense = [{"source": "a", "local_id": 2, "dense_rank": 1, "cosine_score": .9},
                 {"source": "b", "local_id": 3, "dense_rank": 2, "cosine_score": .8}]
        reranked = _ce_rank(dense, {("a", 2): -.5, ("b", 3): 1.2})
        self.assertEqual([doc["local_id"] for doc in reranked], [3, 2])
        self.assertEqual([doc["dense_rank"] for doc in reranked], [2, 1])
        self.assertEqual([doc["ce_rank"] for doc in reranked], [1, 2])
        self.assertEqual([doc["rerank_score"] for doc in reranked], [1.2, -.5])


if __name__ == "__main__":
    unittest.main()

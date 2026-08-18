from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medrag.core import BenchmarkSample, RetrievedDocument
from medrag.filtering.rag2_preanswer_text_hidden import TextHiddenRag2Filter
from scripts.run_rag2_mcq_eval import (
    _load_preanswer_feature_shard,
    _preanswer_feature_expected_metadata,
    _write_preanswer_feature_shard,
    ensure_preanswer_hidden_feature_cache,
    preanswer_hidden_feature_cache_settings,
)


def sample(row: int, dataset: str = "medqa") -> BenchmarkSample:
    return BenchmarkSample(
        row_idx=row,
        id=f"sample-{row}",
        task="mcq",
        collection="unified",
        dataset=dataset,
        split="test",
        question="Question?",
        options={"A": "One", "B": "Two", "C": "Three", "D": "Four"},
        answer="A",
        answers=["A"],
        raw={},
    )


def document(local_id: int) -> RetrievedDocument:
    return RetrievedDocument(
        source="pubmed",
        local_id=local_id,
        db_id=f"pubmed:{local_id}",
        corpus_id=None,
        chunk_id=None,
        doc_id=None,
        title=None,
        text=f"Evidence {local_id}",
        retrieval_score=1.0,
    )


class PreAnswerHiddenCacheTests(unittest.TestCase):
    def test_feature_cache_settings_do_not_depend_on_filter_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_dir = root / "candidate"
            candidate_dir.mkdir()
            (candidate_dir / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
            state_model = root / "llama"
            state_model.mkdir()
            (state_model / "config.json").write_text("{}\n", encoding="utf-8")
            common = {
                "llm_model_path": state_model,
                "hidden_feature_layer": 28,
                "hidden_feature_max_input_tokens": 2048,
                "hidden_feature_dtype": "bfloat16",
                "hidden_feature_attn_implementation": "eager",
                "filter_max_doc_chars": 0,
                "hidden_filter_question_batch_size": 32,
            }
            first = SimpleNamespace(
                **common,
                medqa_filter_model_path=root / "filter-a",
                hidden_filter_helpful_threshold=0.5,
            )
            second = SimpleNamespace(
                **common,
                medqa_filter_model_path=root / "filter-b",
                hidden_filter_helpful_threshold=0.9,
            )
            self.assertEqual(
                preanswer_hidden_feature_cache_settings(first, candidate_dir),
                preanswer_hidden_feature_cache_settings(second, candidate_dir),
            )

    def test_feature_shard_round_trip_preserves_h0_hd_and_offsets(self) -> None:
        samples = [sample(0), sample(1)]
        documents = [[document(1), document(2)], [document(3)]]
        spec = {
            "name": "medqa_0000000",
            "route": "medqa",
            "indices": [0, 1],
            "questions": 2,
            "documents": 3,
        }
        expected = _preanswer_feature_expected_metadata(spec, samples, documents, "fingerprint")
        tensors = {
            "h0": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            "hD": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "document_offsets": torch.tensor([0, 2, 3], dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            _write_preanswer_feature_shard(cache_dir, expected, tensors)
            loaded = _load_preanswer_feature_shard(cache_dir, expected, hidden_size=4)
        for name, value in tensors.items():
            self.assertTrue(torch.equal(loaded[name], value))

    def test_classifier_can_score_cached_features_without_state_extractor(self) -> None:
        samples = [sample(0), sample(1)]
        documents = [[document(1), document(2)], [document(3)]]
        filterer = object.__new__(TextHiddenRag2Filter)
        filterer.hidden_size = 4
        filterer.hidden_layer = 28
        filterer.filter_batch_size = 2
        filterer.max_doc_chars = 0
        filterer.helpful_threshold = 0.5

        def fake_score_batch(self, batch_samples, evidences, h0, hD):
            return [
                {
                    "prediction": "helpful" if float(row.sum()) > 0 else "not helpful",
                    "margin": float(row.sum()),
                    "prob_helpful": 0.75 if float(row.sum()) > 0 else 0.25,
                }
                for row in hD - h0
            ]

        filterer._score_batch = MethodType(fake_score_batch, filterer)
        h0 = torch.zeros((2, 4), dtype=torch.bfloat16)
        hD = torch.tensor(
            [[1, 0, 0, 0], [-1, 0, 0, 0], [2, 0, 0, 0]], dtype=torch.bfloat16
        )
        completed: list[int] = []
        filterer.score_documents_from_features(
            samples,
            documents,
            h0,
            hD,
            torch.tensor([0, 2, 3], dtype=torch.int64),
            progress_callback=completed.append,
        )
        self.assertEqual([doc.filter_prediction for docs in documents for doc in docs], [
            "helpful",
            "not helpful",
            "helpful",
        ])
        self.assertEqual(sum(completed), 3)
        self.assertTrue(all(
            doc.metadata["preanswer_text_hidden_filter"]["hidden_layer"] == 28
            for docs in documents
            for doc in docs
        ))

    def test_completed_feature_cache_is_reused_without_loading_llama(self) -> None:
        class FakeExtractor:
            instances = 0

            def __init__(self, *args, **kwargs):
                type(self).instances += 1
                self.hidden_size = 4

            def states(self, samples, contexts):
                return torch.arange(len(samples) * 4, dtype=torch.float32).reshape(len(samples), 4)

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_dir = root / "candidate"
            candidate_dir.mkdir()
            (candidate_dir / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
            state_model = root / "llama"
            state_model.mkdir()
            (state_model / "config.json").write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                llm_model_path=state_model,
                hidden_feature_layer=28,
                hidden_feature_batch_size=2,
                hidden_feature_max_input_tokens=2048,
                hidden_feature_dtype="bfloat16",
                hidden_feature_attn_implementation="eager",
                hidden_filter_question_batch_size=2,
                filter_max_doc_chars=0,
                filter_device="cpu",
            )
            samples = [sample(0), sample(1)]
            documents = [[document(1), document(2)], [document(3)]]
            with patch(
                "scripts.run_rag2_mcq_eval.PreAnswerLayerExtractor", FakeExtractor
            ):
                first_dir, first_metadata, _ = ensure_preanswer_hidden_feature_cache(
                    args, samples, documents, candidate_dir
                )
                second_dir, second_metadata, _ = ensure_preanswer_hidden_feature_cache(
                    args, samples, documents, candidate_dir
                )
            self.assertEqual(FakeExtractor.instances, 1)
            self.assertEqual(first_dir, second_dir)
            self.assertEqual(first_metadata, second_metadata)
            self.assertTrue((first_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

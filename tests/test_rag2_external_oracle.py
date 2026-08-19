from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medrag.core import BenchmarkSample, RetrievedDocument
from scripts.run_rag2_mcq_eval import apply_oracle_labels
from scripts.materialize_rag2_external_hidden_oracle_labels import adaptive_question_feature_batches


def sample() -> BenchmarkSample:
    raw = {"question": "Q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A"}
    return BenchmarkSample(0, "id0", "mcq", "unified", "medqa", "test", "Q", raw["options"], "A", ["A"], raw)


def document(rank: int) -> RetrievedDocument:
    return RetrievedDocument("pubmed", rank, f"db{rank}", f"stable{rank}", None, None, None, "text", 1.0, rerank_rank=rank)


def write_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_apply_rag2_oracle_treats_only_quality_helpful_as_passing(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_rows(path, [
        {"sample_key": "medqa::test::id0::0", "doc_rank": 1, "doc_stable_id": "stable1", "pseudo_label": "Helpful", "quality_pass": True, "delta_ppl": 1.0},
        {"sample_key": "medqa::test::id0::0", "doc_rank": 2, "doc_stable_id": "stable2", "pseudo_label": "Helpful", "quality_pass": False, "delta_ppl": 2.0},
    ])
    docs = [[document(1), document(2)]]
    apply_oracle_labels(argparse.Namespace(oracle_labels_path=path, oracle_policy="rag2"), [sample()], docs)
    assert [value.filter_prediction for value in docs[0]] == ["helpful", "not helpful"]


def test_apply_hidden_oracle_uses_strict_threshold(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_rows(path, [
        {"sample_key": "medqa::test::id0::0", "doc_rank": 1, "doc_stable_id": "stable1", "projection_score": 0.4},
        {"sample_key": "medqa::test::id0::0", "doc_rank": 2, "doc_stable_id": "stable2", "projection_score": 0.401},
    ])
    docs = [[document(1), document(2)]]
    apply_oracle_labels(argparse.Namespace(oracle_labels_path=path, oracle_policy="hidden_tau_0p4"), [sample()], docs)
    assert [value.filter_prediction for value in docs[0]] == ["not helpful", "helpful"]


def test_adaptive_hidden_batch_splits_only_the_oom_batch():
    class FakeExtractor:
        def encode_questions(self, batch, documents):
            return [[1, 2, 3] for _ in batch], []

        def no_document_features(self, sequences, gold):
            if len(sequences) > 2:
                raise torch.OutOfMemoryError("synthetic")
            return SimpleNamespace(size=len(sequences))

    batch = [{"answer": "A"} for _ in range(5)]
    values = list(adaptive_question_feature_batches(FakeExtractor(), batch, absolute_start=7))
    assert [(start, len(rows), features.size) for start, rows, _, features in values] == [
        (7, 2, 2),
        (9, 1, 1),
        (10, 2, 2),
    ]

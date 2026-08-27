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
from scripts.materialize_rag2_external_hidden_oracle_labels import (
    adaptive_question_feature_batches,
    h0_exceeds_numerical_tolerance,
    validate_feature_cache_contract,
)
from scripts.materialize_rag2_margin_utility_oracle_labels import label_for


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


def test_apply_margin_oracle_passes_only_quality_helpful(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_rows(path, [
        {"sample_key": "medqa::test::id0::0", "doc_rank": 1, "doc_stable_id": "stable1", "pseudo_label": "Helpful", "quality_pass": True, "utility_score": 0.2},
        {"sample_key": "medqa::test::id0::0", "doc_rank": 2, "doc_stable_id": "stable2", "pseudo_label": "Neutral", "quality_pass": True, "utility_score": 0.05},
    ])
    docs = [[document(1), document(2)]]
    apply_oracle_labels(argparse.Namespace(oracle_labels_path=path, oracle_policy="margin_utility"), [sample()], docs)
    assert [value.filter_prediction for value in docs[0]] == ["helpful", "not helpful"]
    assert docs[0][0].filter_score == 0.2


def test_margin_utility_label_boundaries_are_inclusive():
    assert label_for(0.1, 0.1) == "Helpful"
    assert label_for(-0.1, 0.1) == "Harmful"
    assert label_for(0.099, 0.1) == "Neutral"


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


def test_h0_numerical_drift_is_separate_from_semantic_cache_validation():
    args = SimpleNamespace(
        h0_max_abs_tolerance=0.5,
        h0_max_relative_l2_tolerance=0.02,
        h0_min_cosine_similarity=0.999,
    )
    assert h0_exceeds_numerical_tolerance(
        max_abs=0.125,
        max_relative_l2=0.024579,
        min_cosine=0.99970049,
        args=args,
    )


def test_feature_cache_contract_validates_prompt_model_and_counts(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text("{}", encoding="utf-8")
    stat = config.stat()
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = {
        "type": "rag2_mcq_eval_preanswer_hidden_features",
        "version": "rag2_preanswer_hidden_states_v1",
        "questions": 1,
        "documents": 1,
        "shards": 1,
        "settings": {
            "prompt_version": "rag2_fixed_direct_choice_context_v1",
            "state_model": {
                "path": str(model),
                "files": [{"name": config.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}],
                "weight_shards": [],
            },
            "hidden_layer": 28,
            "hidden_max_input_tokens": 2048,
            "hidden_dtype": "bfloat16",
            "hidden_attn_implementation": "eager",
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        feature_cache_dir=cache,
        model_name_or_path=model,
        layer=28,
        max_input_tokens=2048,
        dtype="bfloat16",
        attn_implementation="eager",
    )
    candidates = [{"candidate_documents": [{"db_id": "x", "local_id": 1}]}]
    assert validate_feature_cache_contract(args, candidates) == manifest

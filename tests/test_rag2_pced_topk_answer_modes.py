from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_pced_topk", "prepare_rag2_pced_topk_candidates.py")
RATIONALE = load("pced_rationale", "evaluate_rag2_pced_rationale_answer.py")


def document(source: str, source_rank: int, rerank_score: float) -> dict:
    return {
        "source": source,
        "corpus_id": f"{source}:{source_rank}",
        "text": f"{source} document {source_rank}",
        "rerank_score": rerank_score,
        "metadata": {"retrieval_bucket": source},
    }


def test_dynamic_projection_uses_four_corpus_prefix_before_reranking() -> None:
    initial = []
    for rank in range(1, 4):
        for source in PREPARE.SOURCES:
            initial.append(document(source, rank, rerank_score=0.0))
    reranked = [dict(item) for item in reversed(initial)]
    for index, item in enumerate(reranked):
        item["rerank_score"] = float(len(reranked) - index)
    selected = PREPARE.project(
        {"key": "synthetic", "initial_documents": initial, "reranked_documents": reranked},
        top_k=2,
    )
    assert len(selected) == 2
    assert all(int(item["metadata"]["source_retrieval_rank"]) <= 2 for item in selected)
    assert [item["rerank_rank"] for item in selected] == [1, 2]


def test_jsd_beta_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert abs(RATIONALE.RationaleAnswerPcedGenerator.jsd_beta(logits)) < 1e-7


def test_jsd_beta_increases_when_expert_distribution_changes() -> None:
    identical = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    changed = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    assert RATIONALE.RationaleAnswerPcedGenerator.jsd_beta(changed) > RATIONALE.RationaleAnswerPcedGenerator.jsd_beta(identical)

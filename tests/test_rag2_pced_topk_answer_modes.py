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


def document(source: str, source_rank: int, rerank_score: float, dynamic_rank: int) -> dict:
    return {
        "source": source,
        "corpus_id": f"{source}:{source_rank}",
        "text": f"{source} document {source_rank}",
        "rerank_score": rerank_score,
        "oracle_union_rank": source_rank,
        "metadata": {
            "retrieval_bucket": source,
            "oracle_dynamic_top_k_membership": [2],
            "oracle_dynamic_rerank_rank_by_top_k": {"2": dynamic_rank},
        },
    }


def test_materialization_preserves_stored_semantic_candidate_ids_and_order() -> None:
    first = document("pubmed", 9, rerank_score=1.0, dynamic_rank=2)
    second = document("cpg", 4, rerank_score=99.0, dynamic_rank=1)
    selected = PREPARE.selected_from_union(
        {
            "key": "synthetic",
            "selected_document_ids_by_top_k": {"2": ["cpg:4", "pubmed:9"]},
            "candidate_documents": [first, second],
        },
        top_k=2,
    )
    assert [item["corpus_id"] for item in selected] == ["cpg:4", "pubmed:9"]
    assert [item["rerank_rank"] for item in selected] == [1, 2]


def test_jsd_beta_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert abs(RATIONALE.RationaleAnswerPcedGenerator.jsd_beta(logits)) < 1e-7


def test_jsd_beta_increases_when_expert_distribution_changes() -> None:
    identical = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    changed = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    assert RATIONALE.RationaleAnswerPcedGenerator.jsd_beta(changed) > RATIONALE.RationaleAnswerPcedGenerator.jsd_beta(identical)

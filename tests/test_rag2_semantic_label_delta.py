from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_rag2_semantic_label_delta.py"
SPEC = importlib.util.spec_from_file_location("semantic_delta", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def label(dataset: str, sample_id: str, stable: str, rank: int, semantic: str = "direct_support") -> dict:
    pid = f"{sample_id}::{rank}::{stable}"
    return {
        "id": pid,
        "pair_id": pid,
        "dataset": dataset,
        "sample_id": sample_id,
        "doc_rank": rank,
        "source": "pubmed",
        "doc_stable_id": stable,
        "title": "",
        "semantic_label": semantic,
        "topic_relation": None,
        "confidence": 0.9,
        "evidence_sentence_indices": [1],
        "short_reason": "test",
    }


def candidate(sample_id: str, documents: list[tuple[str, int]]) -> dict:
    return {
        "dataset": "medqa",
        "sample_id": sample_id,
        "row_idx": 0,
        "split": "train",
        "question": "Question?",
        "options": {"A": "one", "B": "two"},
        "answer": "A",
        "answers": ["A"],
        "candidate_documents": [
            {"stable_id": stable, "source": "pubmed", "rerank_rank": rank, "text": f"text {stable}"}
            for stable, rank in documents
        ],
    }


def test_prepare_reuses_by_stable_id_and_finalize_preserves_target_order(tmp_path: Path) -> None:
    candidates_path = tmp_path / "medqa" / "candidates.jsonl"
    labels_path = tmp_path / "old" / "medqa" / "labels.jsonl"
    candidates = [candidate("q1", [("B", 1), ("A", 2)]), candidate("q2", [("D", 1), ("E", 2)])]
    write_jsonl(candidates_path, candidates)
    write_jsonl(labels_path, [label("medqa", "q1", "A", 7)])

    delta = tmp_path / "delta"
    MODULE.prepare(
        Namespace(
            candidates_paths=[candidates_path],
            existing_labels_paths=[labels_path],
            delta_root=delta,
            docs_per_question=2,
        )
    )

    reused = list(MODULE.iter_jsonl(delta / "medqa" / "reused_labels.jsonl"))
    assert reused[0]["pair_id"] == "q1::2::A"
    pending_k1 = list(MODULE.iter_jsonl(delta / "medqa" / "pending_k1.jsonl"))
    pending_k2 = list(MODULE.iter_jsonl(delta / "medqa" / "pending_k2.jsonl"))
    assert [doc["stable_id"] for doc in pending_k1[0]["candidate_documents"]] == ["B"]
    assert [doc["stable_id"] for doc in pending_k2[0]["candidate_documents"]] == ["D", "E"]

    runs = tmp_path / "runs"
    write_jsonl(runs / "k1" / "medqa" / "codex_semantic_labels.jsonl", [label("medqa", "q1", "B", 1)])
    write_jsonl(
        runs / "k2" / "medqa" / "codex_semantic_labels.jsonl",
        [label("medqa", "q2", "D", 1, "no_evidence"), label("medqa", "q2", "E", 2, "misleading_evidence")],
    )
    final = tmp_path / "final"
    MODULE.finalize(
        Namespace(
            candidates_paths=[candidates_path],
            delta_root=delta,
            label_runs_root=runs,
            final_output_root=final,
            docs_per_question=2,
        )
    )
    rows = list(MODULE.iter_jsonl(final / "medqa" / "codex_semantic_labels.jsonl"))
    assert [row["pair_id"] for row in rows] == ["q1::1::B", "q1::2::A", "q2::1::D", "q2::2::E"]
    assert json.loads((final / "manifest.json").read_text())["datasets"]["medqa"]["pairs"] == 4

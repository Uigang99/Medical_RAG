#!/usr/bin/env python3
"""Materialize the exact document union needed by a dynamic RAG2 Top-k oracle.

The master cache stores dense Top-N from each logical corpus and MedCPT scores
for every pooled document.  For every requested k this script reconstructs the
paper contract ``four corpora x dense Top-k -> MedCPT Top-k`` and writes only
the union of documents that actually reach at least one final Top-k condition.
No retrieval, embedding, or cross-encoder inference is repeated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


DATASETS = (
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)
SOURCES = ("pubmed", "pmc", "cpg", "textbooks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-candidates-path", type=Path, required=True)
    parser.add_argument("--no-rag-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--master-per-source-top-k", type=int, default=32)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def no_rag_path(root: Path, dataset: str, split: str) -> Path:
    return root / "no_rag" / dataset / split / "no_rag_generations.jsonl"


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    if not value:
        value = f"{document.get('source')}:{document.get('local_id')}"
    return str(value)


def source_bucket(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    return str(document.get("retrieval_bucket") or metadata.get("retrieval_bucket") or document.get("source") or "")


def rerank_score(document: dict[str, Any]) -> float:
    value = document.get("rerank_score")
    return float(value) if value is not None else float("-inf")


def project_union(
    row: dict[str, Any],
    *,
    sources: list[str],
    top_k_values: list[int],
    master_per_source_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    initial = list(row.get("initial_documents") or [])
    reranked = list(row.get("reranked_documents") or [])
    expected_master = len(sources) * master_per_source_top_k
    if len(initial) != expected_master or len(reranked) != expected_master:
        raise ValueError(
            f"{row.get('key')}: master candidate size mismatch: "
            f"initial={len(initial)} reranked={len(reranked)} expected={expected_master}"
        )

    counts: Counter[str] = Counter()
    source_rank: dict[str, int] = {}
    for document in initial:
        bucket = source_bucket(document)
        counts[bucket] += 1
        identifier = stable_id(document)
        if identifier in source_rank:
            raise ValueError(f"{row.get('key')}: duplicate dense candidate {identifier}")
        source_rank[identifier] = counts[bucket]
    expected_counts = {source: master_per_source_top_k for source in sources}
    if dict(counts) != expected_counts:
        raise ValueError(f"{row.get('key')}: source-balanced master mismatch: {dict(counts)} != {expected_counts}")

    reranked_by_id: dict[str, dict[str, Any]] = {}
    for document in reranked:
        identifier = stable_id(document)
        if identifier in reranked_by_id or identifier not in source_rank:
            raise ValueError(f"{row.get('key')}: invalid reranked identity {identifier}")
        reranked_by_id[identifier] = document

    memberships: dict[str, list[int]] = defaultdict(list)
    selected_by_k: dict[str, list[str]] = {}
    rank_by_k: dict[tuple[str, int], int] = {}
    for top_k in top_k_values:
        eligible = [
            document
            for identifier, document in reranked_by_id.items()
            if source_rank[identifier] <= top_k
        ]
        if len(eligible) != len(sources) * top_k:
            raise ValueError(f"{row.get('key')}: projected 4k pool incomplete at k={top_k}")
        # Match ``project_paper_balanced_candidates`` exactly. Python's sort
        # is stable, so equal MedCPT scores retain their order in the fully
        # reranked master cache; introducing any secondary key changes the
        # evaluated document set on score ties.
        eligible.sort(key=rerank_score, reverse=True)
        selected = eligible[:top_k]
        selected_ids = [stable_id(document) for document in selected]
        selected_by_k[str(top_k)] = selected_ids
        for final_rank, identifier in enumerate(selected_ids, 1):
            memberships[identifier].append(top_k)
            rank_by_k[(identifier, top_k)] = final_rank

    union_ids = [identifier for identifier in reranked_by_id if identifier in memberships]
    union_documents: list[dict[str, Any]] = []
    for union_rank, identifier in enumerate(union_ids, 1):
        document = dict(reranked_by_id[identifier])
        metadata = dict(document.get("metadata") or {})
        bucket = source_bucket(document)
        metadata.update(
            {
                "retrieval_bucket": bucket,
                "source_retrieval_rank": source_rank[identifier],
                "oracle_dynamic_top_k_membership": memberships[identifier],
                "oracle_dynamic_rerank_rank_by_top_k": {
                    str(top_k): rank_by_k[(identifier, top_k)] for top_k in memberships[identifier]
                },
            }
        )
        document["metadata"] = metadata
        document["stable_id"] = identifier
        document["master_rerank_rank"] = document.get("rerank_rank")
        document["rerank_rank"] = union_rank
        document["oracle_union_rank"] = union_rank
        document["source_retrieval_rank"] = source_rank[identifier]
        document["retrieval_bucket"] = bucket
        union_documents.append(document)
    return union_documents, selected_by_k


def count_master_rows(path: Path) -> int:
    manifest = path.parent / "manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8")).get("rows")
        if value is not None and int(value) > 0:
            return int(value)
    return sum(1 for _ in iter_jsonl(path))


def reusable_output(args: argparse.Namespace, top_k_values: list[int], total_questions: int) -> bool:
    path = args.output_root / "manifest.json"
    if not args.resume or not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    source_stat = args.master_candidates_path.stat()
    expected = {
        "type": "rag2_paper_balanced_dynamic_oracle_candidate_union",
        "questions": total_questions,
        "master_candidates_path": str(args.master_candidates_path.resolve()),
        "master_candidates_size_bytes": source_stat.st_size,
        "master_candidates_mtime_ns": source_stat.st_mtime_ns,
        "no_rag_root": str(args.no_rag_root.resolve()),
        "dynamic_top_k_values": top_k_values,
        "sources": args.sources,
        "master_per_source_top_k": args.master_per_source_top_k,
    }
    if {key: manifest.get(key) for key in expected} != expected:
        return False
    files = manifest.get("candidate_files") or {}
    for dataset in args.datasets:
        item = files.get(dataset) or {}
        candidate_path = Path(str(item.get("path") or ""))
        candidate_manifest = candidate_path.parent / "candidate_manifest.json"
        if (
            not candidate_path.is_file()
            or not candidate_manifest.is_file()
            or candidate_path.stat().st_size != int(item.get("size_bytes", -1))
        ):
            return False
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if not args.master_candidates_path.is_file():
        raise FileNotFoundError(args.master_candidates_path)
    top_k_values = sorted(set(args.top_k_values))
    if not top_k_values or top_k_values[0] <= 0 or top_k_values[-1] > args.master_per_source_top_k:
        raise ValueError(f"Invalid --top-k-values: {top_k_values}")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.sources)) != len(args.sources):
        raise ValueError("Datasets and sources must not contain duplicates")

    total_questions = count_master_rows(args.master_candidates_path)
    if reusable_output(args, top_k_values, total_questions):
        logging.info("Complete dynamic Oracle candidate union already exists; reusing %s", args.output_root)
        return
    progress = PipelineProgress(overall_total=total_questions * 2, desc="RAG2DynamicOracleCandidates")
    no_rag: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        progress.set_stage("1/2 index anchored no-RAG traces", total=total_questions)
        for dataset in args.datasets:
            path = no_rag_path(args.no_rag_root, dataset, args.split)
            if not path.is_file():
                raise FileNotFoundError(path)
            for row in iter_jsonl(path):
                key = (dataset, str(row.get("sample_id") or ""))
                if not key[1] or key in no_rag:
                    raise ValueError(f"Invalid/duplicate no-RAG key: {key}")
                no_rag[key] = row
                progress.update()
        if len(no_rag) != total_questions:
            raise RuntimeError(f"No-RAG coverage mismatch: {len(no_rag)} != {total_questions}")

        args.output_root.mkdir(parents=True, exist_ok=True)
        partial_handles: dict[str, Any] = {}
        output_paths: dict[str, Path] = {}
        for dataset in args.datasets:
            directory = args.output_root / dataset / args.split
            directory.mkdir(parents=True, exist_ok=True)
            output_paths[dataset] = directory / "candidates_topk_union.jsonl"
            partial = output_paths[dataset].with_suffix(".jsonl.partial")
            partial_handles[dataset] = partial.open("w", encoding="utf-8", buffering=16 * 1024 * 1024)

        question_counts: Counter[str] = Counter()
        pair_counts: Counter[str] = Counter()
        source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        union_size_counts: Counter[int] = Counter()
        seen: set[tuple[str, str]] = set()
        progress.set_stage("2/2 reconstruct every 4k pool and write Top-k union", total=total_questions)
        try:
            for master_row in iter_jsonl(args.master_candidates_path):
                dataset = str(master_row.get("dataset") or "")
                sample_id = str(master_row.get("sample_id") or "")
                if dataset not in args.datasets:
                    raise ValueError(f"Unexpected dataset in master cache: {dataset}")
                key = (dataset, sample_id)
                baseline = no_rag.get(key)
                if baseline is None or key in seen:
                    raise ValueError(f"Missing no-RAG row or duplicate candidate key: {key}")
                seen.add(key)
                documents, selected_by_k = project_union(
                    master_row,
                    sources=args.sources,
                    top_k_values=top_k_values,
                    master_per_source_top_k=args.master_per_source_top_k,
                )
                output = {
                    "schema_version": 1,
                    "candidate_protocol": "rag2_paper_balanced_dynamic_topk_union_v1",
                    "key": master_row.get("key"),
                    "dataset": dataset,
                    "split": args.split,
                    "sample_id": sample_id,
                    "row_idx": int(master_row.get("row_idx", baseline.get("row_idx", 0))),
                    "question": baseline.get("question"),
                    "options": baseline.get("options"),
                    "answer": baseline.get("gold_answer"),
                    "answers": baseline.get("gold_answers"),
                    "subject": baseline.get("subject"),
                    "query_text": master_row.get("query_text"),
                    "retrieval_query_text": master_row.get("retrieval_query_text"),
                    "rerank_query_text": master_row.get("rerank_query_text"),
                    "dynamic_top_k_values": top_k_values,
                    "selected_document_ids_by_top_k": selected_by_k,
                    "candidate_documents": documents,
                }
                partial_handles[dataset].write(json.dumps(output, ensure_ascii=False) + "\n")
                question_counts[dataset] += 1
                pair_counts[dataset] += len(documents)
                union_size_counts[len(documents)] += 1
                source_counts[dataset].update(str(document.get("source") or "") for document in documents)
                progress.update()
        except Exception:
            for handle in partial_handles.values():
                handle.close()
            raise
        for dataset, handle in partial_handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(output_paths[dataset].with_suffix(".jsonl.partial"), output_paths[dataset])

        if len(seen) != total_questions:
            raise RuntimeError(f"Master candidate coverage mismatch: {len(seen)} != {total_questions}")
        for dataset in args.datasets:
            manifest = {
                "type": "rag2_filter_candidate_dataset",
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "dataset": dataset,
                "split": args.split,
                "candidate_layout": "source_balanced",
                "candidate_protocol": "rag2_paper_balanced_dynamic_topk_union_v1",
                "top_k": None,
                "variable_docs_per_question": True,
                "selected_question_count": question_counts[dataset],
                "selected_pair_count": pair_counts[dataset],
                "mean_documents_per_question": pair_counts[dataset] / question_counts[dataset],
                "per_source_top_k": args.master_per_source_top_k,
                "candidate_pool_top_k": args.master_per_source_top_k * len(args.sources),
                "dynamic_top_k_values": top_k_values,
                "sources": args.sources,
                "source_counts": dict(source_counts[dataset]),
                "candidate_path": str(output_paths[dataset].resolve()),
                "master_candidates_path": str(args.master_candidates_path.resolve()),
                "no_rag_root": str(args.no_rag_root.resolve()),
            }
            atomic_json(output_paths[dataset].parent / "candidate_manifest.json", manifest)
        atomic_json(
            args.output_root / "manifest.json",
            {
                "type": "rag2_paper_balanced_dynamic_oracle_candidate_union",
                "questions": sum(question_counts.values()),
                "pairs": sum(pair_counts.values()),
                "questions_by_dataset": dict(question_counts),
                "pairs_by_dataset": dict(pair_counts),
                "mean_documents_per_question": sum(pair_counts.values()) / sum(question_counts.values()),
                "union_size_distribution": {str(key): value for key, value in sorted(union_size_counts.items())},
                "dynamic_top_k_values": top_k_values,
                "sources": args.sources,
                "master_per_source_top_k": args.master_per_source_top_k,
                "master_candidates_path": str(args.master_candidates_path.resolve()),
                "master_candidates_size_bytes": args.master_candidates_path.stat().st_size,
                "master_candidates_mtime_ns": args.master_candidates_path.stat().st_mtime_ns,
                "no_rag_root": str(args.no_rag_root.resolve()),
                "candidate_files": {
                    dataset: {
                        "path": str(output_paths[dataset].resolve()),
                        "size_bytes": output_paths[dataset].stat().st_size,
                    }
                    for dataset in args.datasets
                },
            },
        )
    finally:
        progress.close()

    logging.info(
        "Dynamic Oracle candidates complete: questions=%s pairs=%s mean_docs=%.3f output=%s",
        sum(question_counts.values()),
        sum(pair_counts.values()),
        sum(pair_counts.values()) / sum(question_counts.values()),
        args.output_root,
    )


if __name__ == "__main__":
    main()

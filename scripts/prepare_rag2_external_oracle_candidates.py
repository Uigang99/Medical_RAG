#!/usr/bin/env python3
from __future__ import annotations

"""Join the frozen external-test rerank cache to benchmark and RAG2 label inputs."""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generate_rag2_document_traces import load_no_doc_cache
from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path
from medrag.io_utils import write_json, write_jsonl

DATASETS = (
    "medmcqa", "medqa", "mmlu_anatomy", "mmlu_clinical_knowledge",
    "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-cache-path", type=Path, required=True)
    p.add_argument("--no-rag-artifact-root", type=Path, required=True)
    p.add_argument("--quality-selection-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path, default=PROJECT_ROOT / "datasets/benchmark")
    p.add_argument("--collection", default="unified")
    p.add_argument("--split", default="test")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--docs-per-question", type=int, default=32)
    p.add_argument("--prompt-profile", choices=["paper_exact"], default="paper_exact")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL: {path}:{number}") from error


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    if not args.candidate_cache_path.is_file():
        raise FileNotFoundError(args.candidate_cache_path)
    candidates = {str(row["key"]): row for row in rows(args.candidate_cache_path)}
    if len(candidates) != sum(1 for _ in rows(args.candidate_cache_path)):
        raise ValueError("Duplicate candidate keys")

    summary: dict[str, Any] = {"datasets": {}, "candidate_cache_path": str(args.candidate_cache_path.resolve())}
    combined_all: list[dict[str, Any]] = []
    for dataset in args.datasets:
        benchmark_path = resolve_benchmark_path(args.benchmark_root, "mcq", args.collection, dataset, args.split)
        samples = load_benchmark_samples(benchmark_path, "mcq", args.collection, dataset, args.split)
        no_rag_path = args.no_rag_artifact_root / "no_rag" / dataset / args.split / "no_rag_generations.jsonl"
        cache_args = SimpleNamespace(
            quality_selection_path=args.quality_selection_root / dataset / args.split / "usable_rows.jsonl",
            prompt_profile=args.prompt_profile,
        )
        valid_no_rag = load_no_doc_cache(no_rag_path, cache_args)
        selected_ids = {
            str(row["sample_id"])
            for row in rows(cache_args.quality_selection_path)
            if str(row.get("sample_id") or "")
        }
        valid_no_rag = {key: value for key, value in valid_no_rag.items() if key in selected_ids}
        all_rows: list[dict[str, Any]] = []
        labelable_rows: list[dict[str, Any]] = []
        sources: Counter[str] = Counter()
        for sample in samples:
            key = f"{sample.dataset}::{sample.split}::{sample.id}::{sample.row_idx}"
            cached = candidates.get(key)
            if cached is None:
                raise KeyError(f"Missing frozen candidate row: {key}")
            documents = list(cached.get("reranked_documents") or [])[: args.docs_per_question]
            if len(documents) != args.docs_per_question:
                raise ValueError(f"{key}: expected {args.docs_per_question} documents, got {len(documents)}")
            ranks = [int(doc.get("rerank_rank") or 0) for doc in documents]
            if ranks != list(range(1, args.docs_per_question + 1)):
                raise ValueError(f"Noncanonical rerank prefix for {key}: {ranks}")
            sources.update(str(doc.get("source") or "") for doc in documents)
            row = {
                "schema_version": 1,
                "key": key,
                "sample_id": sample.id,
                "row_idx": sample.row_idx,
                "dataset": sample.dataset,
                "split": sample.split,
                "question": sample.question,
                "options": sample.options,
                "answer": sample.answer,
                "answers": sample.answers,
                "query_text": cached.get("query_text"),
                "retrieval_query_text": cached.get("retrieval_query_text"),
                "rerank_query_text": cached.get("rerank_query_text"),
                "candidate_documents": documents,
            }
            all_rows.append(row)
            combined_all.append(row)
            if sample.id in valid_no_rag:
                labelable_rows.append(row)
        destination = args.output_root / dataset / args.split
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / "candidates_top32_all.jsonl", all_rows)
        write_jsonl(destination / "candidates_top32_rag2_labelable.jsonl", labelable_rows)
        summary["datasets"][dataset] = {
            "questions": len(all_rows),
            "pairs": len(all_rows) * args.docs_per_question,
            "rag2_labelable_questions": len(labelable_rows),
            "rag2_labelable_pairs": len(labelable_rows) * args.docs_per_question,
            "invalid_no_rag_questions": len(all_rows) - len(labelable_rows),
            "sources": dict(sources),
            "benchmark_path": str(benchmark_path),
            "no_rag_path": str(no_rag_path),
            "quality_selection_path": str(cache_args.quality_selection_path),
        }
        logging.info("[%s] all=%s RAG2-labelable=%s", dataset, len(all_rows), len(labelable_rows))
    write_jsonl(args.output_root / "candidates_top32_all.jsonl", combined_all)
    summary["questions"] = len(combined_all)
    summary["pairs"] = len(combined_all) * args.docs_per_question
    write_json(args.output_root / "manifest.json", summary)


if __name__ == "__main__":
    main()

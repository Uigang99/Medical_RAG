#!/usr/bin/env python3
from __future__ import annotations

"""Build the paper-style all-MCQ table for a direct-choice Top-k sweep."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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
MMLU_DATASETS = tuple(value for value in DATASETS if value.startswith("mmlu_"))
TOP_K_VALUES = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=6545)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def latest_complete(case_root: Path, expected_questions: int) -> Path:
    if not case_root.is_dir():
        raise FileNotFoundError(case_root)
    for run_dir in sorted((path for path in case_root.iterdir() if path.is_dir()), reverse=True):
        result_path = run_dir / "results.jsonl"
        if not result_path.is_file():
            continue
        rows = sum(1 for line in result_path.open("r", encoding="utf-8") if line.strip())
        if rows == expected_questions:
            return run_dir
    raise RuntimeError(
        f"No complete {expected_questions}-question run found under {case_root}"
    )


def load_condition(case_root: Path, expected_questions: int) -> tuple[list[dict[str, Any]], Path]:
    run_dir = latest_complete(case_root, expected_questions)
    rows = list(iter_jsonl(run_dir / "results.jsonl"))
    sample_ids = [str((row.get("sample") or {}).get("id") or "") for row in rows]
    if len(set(sample_ids)) != expected_questions or any(not value for value in sample_ids):
        raise RuntimeError(f"Duplicate or missing sample ids in {run_dir}")
    protocols = {str(row.get("answer_decision_mode") or "") for row in rows}
    if protocols != {"constrained_choice"}:
        raise RuntimeError(f"Expected constrained_choice rows in {run_dir}, got {protocols}")
    parse_failures = sum(bool(row.get("parse_errors")) for row in rows)
    if parse_failures:
        raise RuntimeError(f"Unexpected direct-choice parse failures in {run_dir}: {parse_failures}")
    return rows, run_dir


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(bool((row.get("evaluation") or {}).get("correct")) for row in rows) / len(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str((row.get("sample") or {}).get("dataset") or "")
        if dataset not in DATASETS:
            raise RuntimeError(f"Unexpected dataset in results: {dataset!r}")
        grouped[dataset].append(row)
    if set(grouped) != set(DATASETS):
        raise RuntimeError(f"Dataset coverage mismatch: {sorted(grouped)}")
    mmlu_rows = [row for dataset in MMLU_DATASETS for row in grouped[dataset]]
    context_counts = [int(row.get("context_document_count") or 0) for row in rows]
    return {
        "questions": len(rows),
        "correct": sum(bool((row.get("evaluation") or {}).get("correct")) for row in rows),
        "mean_context_documents": sum(context_counts) / len(context_counts),
        "zero_context_questions": sum(value == 0 for value in context_counts),
        "medmcqa_accuracy": accuracy(grouped["medmcqa"]),
        "medqa_accuracy": accuracy(grouped["medqa"]),
        "mmlu_pooled_accuracy": accuracy(mmlu_rows),
        "micro_accuracy": accuracy(rows),
        "macro_accuracy": sum(accuracy(grouped[dataset]) for dataset in DATASETS) / len(DATASETS),
        "dataset_counts": {dataset: len(grouped[dataset]) for dataset in DATASETS},
    }


def condition_roots(results_root: Path):
    yield "No-RAG", None, results_root / "no_rag_reference/no_rag"
    for top_k in TOP_K_VALUES:
        yield "Without filter", top_k, results_root / f"unfiltered_rag/rerank_rag_top{top_k}"
        yield "RAG2", top_k, results_root / f"rag2_filter/filter_rag_top{top_k}"
        yield "Hidden State", top_k, results_root / f"hidden_state_filter/filter_rag_top{top_k}"


def render_table(summary: dict[str, Any]) -> str:
    lines = [
        "Direct-choice all-MCQ Top-k sweep",
        "",
        "Final answer protocol: fixed direct-choice prompt + constrained A/B/C/D one-token decoding",
        "Retrieval/reranking/filtering: identical to the paper-terminal rationale-answer sweep",
        "",
        "| Rerank Top-k | Filtering | # doc after filtering | MedMCQA | MedQA USMLE | MMLU pooled | Micro Avg | Macro Avg |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        top_k = "-" if row["top_k"] is None else str(row["top_k"])
        metrics = row["metrics"]
        lines.append(
            f"| {top_k} | {row['filtering']} | {metrics['mean_context_documents']:.2f} | "
            f"{metrics['medmcqa_accuracy'] * 100:.2f} | {metrics['medqa_accuracy'] * 100:.2f} | "
            f"{metrics['mmlu_pooled_accuracy'] * 100:.2f} | {metrics['micro_accuracy'] * 100:.2f} | "
            f"{metrics['macro_accuracy'] * 100:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_root
    rows: list[dict[str, Any]] = []
    for filtering, top_k, case_root in condition_roots(args.results_root):
        values, run_dir = load_condition(case_root, args.expected_questions)
        rows.append(
            {
                "filtering": filtering,
                "top_k": top_k,
                "run_dir": str(run_dir.resolve()),
                "metrics": summarize_rows(values),
            }
        )
    summary = {
        "version": "rag2_direct_choice_all_mcq_summary_v1",
        "answer_decision_mode": "constrained_choice",
        "expected_questions": args.expected_questions,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "direct_choice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = render_table(summary)
    (output_dir / "direct_choice_summary_table_pretty.txt").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

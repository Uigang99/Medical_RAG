#!/usr/bin/env python3
from __future__ import annotations

"""Summarize tau=0.4 Hidden-State MCQ sweeps under both answer protocols."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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
EXPECTED_DATASET_COUNTS = {
    "medmcqa": 4183,
    "medqa": 1273,
    "mmlu_anatomy": 135,
    "mmlu_clinical_knowledge": 265,
    "mmlu_college_biology": 144,
    "mmlu_college_medicine": 173,
    "mmlu_medical_genetics": 100,
    "mmlu_professional_medicine": 272,
}
MMLU_DATASETS = tuple(dataset for dataset in DATASETS if dataset.startswith("mmlu_"))
TOP_K_VALUES = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rationale-no-rag-root", type=Path, required=True)
    parser.add_argument("--rationale-rag2-results-root", type=Path, required=True)
    parser.add_argument("--rationale-results-root", type=Path, required=True)
    parser.add_argument("--direct-no-rag-root", type=Path, required=True)
    parser.add_argument("--direct-rag2-results-root", type=Path, required=True)
    parser.add_argument("--direct-results-root", type=Path, required=True)
    parser.add_argument("--rationale-medmcqa-rag2-filter-model-path", type=Path, required=True)
    parser.add_argument("--rationale-medqa-rag2-filter-model-path", type=Path, required=True)
    parser.add_argument("--direct-medmcqa-rag2-filter-model-path", type=Path, required=True)
    parser.add_argument("--direct-medqa-rag2-filter-model-path", type=Path, required=True)
    parser.add_argument("--medmcqa-filter-model-path", type=Path, required=True)
    parser.add_argument("--medqa-filter-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=6545)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
        results_path = run_dir / "results.jsonl"
        if not results_path.is_file():
            continue
        row_count = sum(1 for line in results_path.open("r", encoding="utf-8") if line.strip())
        if row_count == expected_questions:
            return run_dir
    raise RuntimeError(f"No complete {expected_questions}-question run found under {case_root}")


def load_condition(
    case_root: Path,
    expected_questions: int,
    expected_mode: str,
) -> tuple[list[dict[str, Any]], Path]:
    run_dir = latest_complete(case_root, expected_questions)
    rows = list(iter_jsonl(run_dir / "results.jsonl"))
    sample_ids = [str((row.get("sample") or {}).get("id") or "") for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise RuntimeError(f"Missing sample id in {run_dir}")
    if len(set(sample_ids)) != expected_questions:
        raise RuntimeError(f"Duplicate sample ids in {run_dir}")

    protocols = {str(row.get("answer_decision_mode") or "") for row in rows}
    nonempty_protocols = protocols - {""}
    run_config_path = run_dir / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file()
        else {}
    )
    configured_mode = str(run_config.get("answer_decision_mode") or "")
    if nonempty_protocols not in (set(), {expected_mode}) or configured_mode != expected_mode:
        raise RuntimeError(
            f"Expected {expected_mode} rows in {run_dir}, "
            f"got row_protocols={protocols}, configured_mode={configured_mode!r}"
        )
    parse_failures = sum(bool(row.get("parse_errors")) for row in rows)
    unevaluable = sum(not bool((row.get("evaluation") or {}).get("evaluable")) for row in rows)
    invalid_predictions = sum(
        str((row.get("evaluation") or {}).get("predicted_choice") or "") not in {"A", "B", "C", "D"}
        for row in rows
    )
    if parse_failures or unevaluable or invalid_predictions:
        raise RuntimeError(
            f"Invalid evaluation rows in {run_dir}: parse_failures={parse_failures}, "
            f"unevaluable={unevaluable}, invalid_predictions={invalid_predictions}"
        )
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
    dataset_counts = {dataset: len(grouped[dataset]) for dataset in DATASETS}
    if dataset_counts != EXPECTED_DATASET_COUNTS:
        raise RuntimeError(
            f"Dataset-count mismatch: expected={EXPECTED_DATASET_COUNTS}, actual={dataset_counts}"
        )

    mmlu_rows = [row for dataset in MMLU_DATASETS for row in grouped[dataset]]
    context_counts = [int(row.get("context_document_count") or 0) for row in rows]
    return {
        "questions": len(rows),
        "correct": sum(bool((row.get("evaluation") or {}).get("correct")) for row in rows),
        "mean_context_documents": sum(context_counts) / len(context_counts),
        "zero_context_questions": sum(value == 0 for value in context_counts),
        "context_document_count_distribution": dict(sorted(Counter(context_counts).items())),
        "medmcqa_accuracy": accuracy(grouped["medmcqa"]),
        "medqa_accuracy": accuracy(grouped["medqa"]),
        "mmlu_pooled_accuracy": accuracy(mmlu_rows),
        "micro_accuracy": accuracy(rows),
        "macro_accuracy": sum(accuracy(grouped[dataset]) for dataset in DATASETS) / len(DATASETS),
        "dataset_counts": dataset_counts,
        "dataset_accuracy": {dataset: accuracy(grouped[dataset]) for dataset in DATASETS},
    }


def validate_filter_run(
    run_dir: Path,
    medmcqa_filter_model_path: Path,
    medqa_filter_model_path: Path,
    expected_evidence_unit: str,
) -> None:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "medmcqa_filter_model_path": str(medmcqa_filter_model_path.resolve()),
        "medqa_filter_model_path": str(medqa_filter_model_path.resolve()),
        "filter_evidence_unit": expected_evidence_unit,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Filter contract mismatch in {run_dir}: {actual} != {expected}")


def collect_mode(
    label: str,
    expected_mode: str,
    no_rag_root: Path,
    rag2_results_root: Path,
    results_root: Path,
    expected_questions: int,
    medmcqa_rag2_filter_model_path: Path,
    medqa_rag2_filter_model_path: Path,
    medmcqa_filter_model_path: Path,
    medqa_filter_model_path: Path,
) -> dict[str, Any]:
    output_rows = []
    no_rag_rows, no_rag_run = load_condition(no_rag_root, expected_questions, expected_mode)
    output_rows.append(
        {
            "filtering": "No-RAG",
            "top_k": None,
            "run_dir": str(no_rag_run.resolve()),
            "metrics": summarize_rows(no_rag_rows),
        }
    )
    for top_k in TOP_K_VALUES:
        rag2_values, rag2_run_dir = load_condition(
            rag2_results_root / f"filter_rag_top{top_k}", expected_questions, expected_mode
        )
        validate_filter_run(
            rag2_run_dir,
            medmcqa_rag2_filter_model_path,
            medqa_rag2_filter_model_path,
            "document",
        )
        output_rows.append(
            {
                "filtering": "RAG2",
                "top_k": top_k,
                "run_dir": str(rag2_run_dir.resolve()),
                "metrics": summarize_rows(rag2_values),
            }
        )
        values, run_dir = load_condition(
            results_root / f"filter_rag_top{top_k}", expected_questions, expected_mode
        )
        validate_filter_run(
            run_dir, medmcqa_filter_model_path, medqa_filter_model_path, "preanswer_text_hidden"
        )
        output_rows.append(
            {
                "filtering": "Hidden State (tau=0.4)",
                "top_k": top_k,
                "run_dir": str(run_dir.resolve()),
                "metrics": summarize_rows(values),
            }
        )
    return {
        "label": label,
        "answer_decision_mode": expected_mode,
        "no_rag_root": str(no_rag_root.resolve()),
        "rag2_results_root": str(rag2_results_root.resolve()),
        "results_root": str(results_root.resolve()),
        "rows": output_rows,
    }


def render_mode_table(mode: dict[str, Any]) -> str:
    lines = [
        f"## {mode['label']}",
        "",
        "| Rerank Top-k | Filtering | # doc after filtering | MedMCQA | MedQA USMLE | MMLU pooled | Micro Avg | Macro Avg |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mode["rows"]:
        metrics = row["metrics"]
        top_k = "-" if row["top_k"] is None else str(row["top_k"])
        lines.append(
            f"| {top_k} | {row['filtering']} | {metrics['mean_context_documents']:.2f} | "
            f"{metrics['medmcqa_accuracy'] * 100:.2f} | {metrics['medqa_accuracy'] * 100:.2f} | "
            f"{metrics['mmlu_pooled_accuracy'] * 100:.2f} | {metrics['micro_accuracy'] * 100:.2f} | "
            f"{metrics['macro_accuracy'] * 100:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_summary(summary: dict[str, Any]) -> str:
    header = [
        "RAG2 vs Hidden-State tau=0.4 all-MCQ Top-k sweep",
        "",
        "Cohort: 6,545 questions (MedMCQA 4,183; MedQA 1,273; MMLU 1,089)",
        "Retrieval: stored no-RAG rationale query; source-balanced 8 x 4 = 32; MedCPT rerank Top-32",
        "Comparison: No-RAG vs RAG2 vs Hidden State tau=0.4 under the same final-answer protocol",
        "Filtering: k is the pre-filter rerank prefix; # doc is the mean number retained after filtering",
        "",
    ]
    return "\n".join(header + [render_mode_table(mode) for mode in summary["modes"]])


def main() -> None:
    args = parse_args()
    modes = [
        collect_mode(
            "Rationale + fixed terminal answer",
            "free_generation",
            args.rationale_no_rag_root,
            args.rationale_rag2_results_root,
            args.rationale_results_root,
            args.expected_questions,
            args.rationale_medmcqa_rag2_filter_model_path,
            args.rationale_medqa_rag2_filter_model_path,
            args.medmcqa_filter_model_path,
            args.medqa_filter_model_path,
        ),
        collect_mode(
            "Direct choice",
            "constrained_choice",
            args.direct_no_rag_root,
            args.direct_rag2_results_root,
            args.direct_results_root,
            args.expected_questions,
            args.direct_medmcqa_rag2_filter_model_path,
            args.direct_medqa_rag2_filter_model_path,
            args.medmcqa_filter_model_path,
            args.medqa_filter_model_path,
        ),
    ]
    summary = {
        "version": "rag2_hidden_tau0p4_answer_mode_summary_v1",
        "label_threshold_tau": 0.4,
        "filter_decision_threshold": 0.5,
        "expected_questions": args.expected_questions,
        "expected_dataset_counts": EXPECTED_DATASET_COUNTS,
        "filter_models": {
            "rag2": {
                "rationale_answer": {
                    "medmcqa_and_mmlu": str(
                        args.rationale_medmcqa_rag2_filter_model_path.resolve()
                    ),
                    "medqa": str(args.rationale_medqa_rag2_filter_model_path.resolve()),
                },
                "direct_choice": {
                    "medmcqa_and_mmlu": str(
                        args.direct_medmcqa_rag2_filter_model_path.resolve()
                    ),
                    "medqa": str(args.direct_medqa_rag2_filter_model_path.resolve()),
                },
            },
            "hidden_tau0p4": {
                "medmcqa_and_mmlu": str(args.medmcqa_filter_model_path.resolve()),
                "medqa": str(args.medqa_filter_model_path.resolve()),
            },
        },
        "modes": modes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hidden_tau0p4_answer_modes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = render_summary(summary)
    (args.output_dir / "hidden_tau0p4_answer_modes_summary_table_pretty.txt").write_text(
        table + "\n", encoding="utf-8"
    )
    print(table)


if __name__ == "__main__":
    main()

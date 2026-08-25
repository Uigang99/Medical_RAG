#!/usr/bin/env python3
"""Build one verified table for anchored no-RAG, unfiltered, and filtered sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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
DATASET_LABELS = {
    "medmcqa": "MedMCQA",
    "medqa": "MedQA USMLE",
    "mmlu_anatomy": "Anatomy",
    "mmlu_clinical_knowledge": "Clinical knowledge",
    "mmlu_college_biology": "College biology",
    "mmlu_college_medicine": "College medicine",
    "mmlu_medical_genetics": "Medical genetics",
    "mmlu_professional_medicine": "Professional medicine",
}
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
SUMMARY_VERSION = "rag2_anchored_paper_reproduction_sweep_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-prompt-profile", default="paper_compatible_three_anchor")
    parser.add_argument("--expected-answer-decision-mode", default="free_generation")
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


def latest_result_dir(case_root: Path) -> Path:
    if not case_root.is_dir():
        raise FileNotFoundError(case_root)
    candidates = sorted(
        (
            path
            for path in case_root.iterdir()
            if path.is_dir() and (path / "results.jsonl").is_file()
        ),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No completed result directory found under {case_root}")
    return candidates[0]


def validate_run_config(
    run_dir: Path,
    *,
    expected_case: str,
    expected_top_k: int | None,
    expected_prompt_profile: str,
    expected_answer_decision_mode: str,
) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "case": expected_case,
        "prompt_profile": expected_prompt_profile,
        "answer_decision_mode": expected_answer_decision_mode,
        "candidate_layout": "source_balanced",
        "per_source_top_k": 8,
        "candidate_pool_top_k": 32,
        "rerank_top_k": 32,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Run-contract mismatch in {run_dir}: {actual} != {expected}")
    if expected_case == "rerank_rag":
        actual_top_k = config.get("generation_top_k")
    elif expected_case == "filter_rag":
        actual_top_k = config.get("filter_rerank_top_k")
    else:
        actual_top_k = None
    if actual_top_k != expected_top_k:
        raise RuntimeError(
            f"Top-k mismatch in {run_dir}: actual={actual_top_k!r}, expected={expected_top_k!r}"
        )
    return config


def load_condition(
    case_root: Path,
    *,
    expected_case: str,
    expected_top_k: int | None,
    expected_prompt_profile: str,
    expected_answer_decision_mode: str,
    progress: PipelineProgress | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    run_dir = latest_result_dir(case_root)
    config = validate_run_config(
        run_dir,
        expected_case=expected_case,
        expected_top_k=expected_top_k,
        expected_prompt_profile=expected_prompt_profile,
        expected_answer_decision_mode=expected_answer_decision_mode,
    )
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    dataset_counts: Counter[str] = Counter()
    for row in iter_jsonl(run_dir / "results.jsonl"):
        sample = row.get("sample") or {}
        dataset = str(sample.get("dataset") or "")
        sample_id = str(sample.get("id") or "")
        if dataset not in DATASETS or not sample_id:
            raise RuntimeError(f"Invalid sample identity in {run_dir}: {(dataset, sample_id)!r}")
        key = (dataset, sample_id)
        if key in keys:
            raise RuntimeError(f"Duplicate result key in {run_dir}: {key}")
        keys.add(key)
        dataset_counts[dataset] += 1
        evaluation = row.get("evaluation") or {}
        if not bool(evaluation.get("evaluable")):
            raise RuntimeError(f"Unevaluable result in {run_dir}: {key}")
        if str(evaluation.get("predicted_choice") or "") not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"Invalid predicted choice in {run_dir}: {key}")
        rows.append(row)
        if progress is not None:
            progress.update(1)
    actual_counts = {dataset: dataset_counts[dataset] for dataset in DATASETS}
    if actual_counts != EXPECTED_DATASET_COUNTS:
        raise RuntimeError(
            f"Dataset-count mismatch in {run_dir}: expected={EXPECTED_DATASET_COUNTS}, actual={actual_counts}"
        )
    return rows, run_dir, config


def accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("Cannot compute accuracy for an empty group")
    return sum(bool((row.get("evaluation") or {}).get("correct")) for row in rows) / len(rows)


def sample_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (
            str((row.get("sample") or {}).get("dataset") or ""),
            str((row.get("sample") or {}).get("id") or ""),
        )
        for row in rows
    }


def cohort_fingerprint(keys: set[tuple[str, str]]) -> str:
    payload = "\n".join(f"{dataset}\t{sample_id}" for dataset, sample_id in sorted(keys))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    expected_counts = expected_counts or EXPECTED_DATASET_COUNTS
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str((row.get("sample") or {}).get("dataset") or "")
        grouped[dataset].append(row)
    actual_counts = {dataset: len(grouped[dataset]) for dataset in expected_counts}
    if actual_counts != expected_counts:
        raise RuntimeError(f"Dataset-count mismatch: expected={expected_counts}, actual={actual_counts}")

    dataset_accuracy = {dataset: accuracy(grouped[dataset]) for dataset in expected_counts}
    mmlu_rows = [row for dataset in MMLU_DATASETS if dataset in grouped for row in grouped[dataset]]
    if not mmlu_rows:
        raise RuntimeError("No MMLU rows available for pooled accuracy")
    mmlu_pooled_accuracy = accuracy(mmlu_rows)
    context_counts = [int(row.get("context_document_count") or 0) for row in rows]
    micro_accuracy = accuracy(rows)
    macro_8_accuracy = sum(dataset_accuracy.values()) / len(dataset_accuracy)
    macro_3_accuracy = (
        dataset_accuracy["medmcqa"]
        + dataset_accuracy["medqa"]
        + mmlu_pooled_accuracy
    ) / 3
    return {
        "questions": len(rows),
        "correct": sum(bool((row.get("evaluation") or {}).get("correct")) for row in rows),
        "dataset_counts": actual_counts,
        "dataset_accuracy": dataset_accuracy,
        "mmlu_pooled_accuracy": mmlu_pooled_accuracy,
        "micro_accuracy": micro_accuracy,
        "macro_8_accuracy": macro_8_accuracy,
        "macro_3_accuracy": macro_3_accuracy,
        "mean_context_documents": sum(context_counts) / len(context_counts),
        "zero_context_questions": sum(count == 0 for count in context_counts),
        "context_document_count_distribution": dict(sorted(Counter(context_counts).items())),
    }


def condition_specs(results_root: Path) -> list[dict[str, Any]]:
    specs = [
        {
            "top_k": None,
            "filtering": "No-RAG",
            "case": "no_rag",
            "case_root": results_root / "no_rag",
        }
    ]
    for top_k in TOP_K_VALUES:
        specs.extend(
            [
                {
                    "top_k": top_k,
                    "filtering": "No filtering",
                    "case": "rerank_rag",
                    "case_root": results_root / f"rerank_rag_top{top_k}",
                },
                {
                    "top_k": top_k,
                    "filtering": "RAG2 filtering",
                    "case": "filter_rag",
                    "case_root": results_root / f"filter_rag_top{top_k}",
                },
            ]
        )
    return specs


def render_table(summary: dict[str, Any]) -> str:
    dataset_headers = [DATASET_LABELS[dataset] for dataset in DATASETS]
    headers = [
        "Rerank Top-k",
        "Filtering",
        "Avg # docs",
        *dataset_headers,
        "MMLU pooled",
        "Micro Avg",
        "Macro Avg (8)",
        "Macro Avg (3 groups)",
    ]
    lines = [
        "# Anchored RAG2 paper-reproduction MCQ sweep",
        "",
        "Cohort: 6,545 questions (MedMCQA 4,183; MedQA 1,273; six MMLU subsets 1,089).",
        "Micro Avg = 6,545-question pooled accuracy; Macro Avg (8) = mean of eight dataset accuracies; "
        "Macro Avg (3 groups) = mean of MedMCQA, MedQA, and pooled MMLU accuracies.",
        "Top-k is the shared MedCPT reranked prefix. No filtering uses all k documents; RAG2 filtering uses "
        "only documents predicted Helpful inside the same prefix.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:", "---", "---:"] + ["---:"] * (len(headers) - 3)) + "|",
    ]
    for condition in summary["conditions"]:
        metrics = condition["metrics"]
        cells = [
            "-" if condition["top_k"] is None else str(condition["top_k"]),
            condition["filtering"],
            f"{metrics['mean_context_documents']:.2f}",
        ]
        cells.extend(
            f"{metrics['dataset_accuracy'][dataset] * 100:.2f}" for dataset in DATASETS
        )
        cells.extend(
            [
                f"{metrics['mmlu_pooled_accuracy'] * 100:.2f}",
                f"{metrics['micro_accuracy'] * 100:.2f}",
                f"{metrics['macro_8_accuracy'] * 100:.2f}",
                f"{metrics['macro_3_accuracy'] * 100:.2f}",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "rerank_top_k",
        "filtering",
        "mean_context_documents",
        *[f"{dataset}_accuracy" for dataset in DATASETS],
        "mmlu_pooled_accuracy",
        "micro_accuracy",
        "macro_8_accuracy",
        "macro_3_accuracy",
        "questions",
        "correct",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in summary["conditions"]:
            metrics = condition["metrics"]
            row = {
                "rerank_top_k": condition["top_k"],
                "filtering": condition["filtering"],
                "mean_context_documents": metrics["mean_context_documents"],
                "mmlu_pooled_accuracy": metrics["mmlu_pooled_accuracy"],
                "micro_accuracy": metrics["micro_accuracy"],
                "macro_8_accuracy": metrics["macro_8_accuracy"],
                "macro_3_accuracy": metrics["macro_3_accuracy"],
                "questions": metrics["questions"],
                "correct": metrics["correct"],
                "run_dir": condition["run_dir"],
            }
            row.update(
                {
                    f"{dataset}_accuracy": metrics["dataset_accuracy"][dataset]
                    for dataset in DATASETS
                }
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    specs = condition_specs(args.results_root)
    total_rows = sum(EXPECTED_DATASET_COUNTS.values()) * len(specs)
    progress = PipelineProgress(
        overall_total=total_rows + 3,
        desc="SweepSummary",
    )
    conditions: list[dict[str, Any]] = []
    reference_keys: set[tuple[str, str]] | None = None
    try:
        progress.set_stage("1/2 load, validate, and aggregate runs", total=total_rows)
        for spec in specs:
            progress.set_detail(
                f"condition={spec['filtering']} top_k={spec['top_k'] if spec['top_k'] is not None else '-'}"
            )
            rows, run_dir, config = load_condition(
                spec["case_root"],
                expected_case=spec["case"],
                expected_top_k=spec["top_k"],
                expected_prompt_profile=args.expected_prompt_profile,
                expected_answer_decision_mode=args.expected_answer_decision_mode,
                progress=progress,
            )
            keys = sample_keys(rows)
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                missing = sorted(reference_keys - keys)[:5]
                extra = sorted(keys - reference_keys)[:5]
                raise RuntimeError(
                    f"Cross-condition cohort mismatch in {run_dir}: "
                    f"missing_examples={missing}, extra_examples={extra}"
                )
            conditions.append(
                {
                    "top_k": spec["top_k"],
                    "filtering": spec["filtering"],
                    "case": spec["case"],
                    "run_dir": str(run_dir.resolve()),
                    "filter_models": {
                        "medmcqa_and_mmlu": config.get("medmcqa_filter_model_path"),
                        "medqa": config.get("medqa_filter_model_path"),
                    }
                    if spec["case"] == "filter_rag"
                    else None,
                    "metrics": summarize_rows(rows),
                }
            )

        output_dir = args.output_dir or (args.results_root / "combined_summary")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "version": SUMMARY_VERSION,
            "results_root": str(args.results_root.resolve()),
            "prompt_profile": args.expected_prompt_profile,
            "answer_decision_mode": args.expected_answer_decision_mode,
            "expected_dataset_counts": EXPECTED_DATASET_COUNTS,
            "cohort_sha256": cohort_fingerprint(reference_keys or set()),
            "metric_definitions": {
                "micro_accuracy": "accuracy pooled over all 6545 questions",
                "macro_8_accuracy": "unweighted mean of the eight dataset accuracies",
                "mmlu_pooled_accuracy": "accuracy pooled over the six MMLU subsets",
                "macro_3_accuracy": "unweighted mean of MedMCQA, MedQA, and pooled MMLU accuracies",
            },
            "conditions": conditions,
        }
        table = render_table(summary)
        progress.set_stage("2/2 write JSON, CSV, and table", total=3)
        (output_dir / "combined_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        progress.update(1)
        write_csv(output_dir / "combined_summary.csv", summary)
        progress.update(1)
        (output_dir / "summary_table_pretty.txt").write_text(table + "\n", encoding="utf-8")
        progress.update(1)
    finally:
        progress.close()
    print(table)
    print(f"Combined summary: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

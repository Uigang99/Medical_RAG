#!/usr/bin/env python3
"""Compare RAG2 and hidden-state Helpful-only gold oracles on the same MCQs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402
from summarize_rag2_anchored_paper_reproduction_sweep import (  # noqa: E402
    DATASETS,
    DATASET_LABELS,
    EXPECTED_DATASET_COUNTS,
    TOP_K_VALUES,
    load_condition,
    summarize_rows,
)


SUMMARY_VERSION = "rag2_anchored_rag2_vs_hidden_gold_oracle_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag2-oracle-results-root", type=Path, required=True)
    parser.add_argument("--hidden-oracle-results-root", type=Path, required=True)
    parser.add_argument("--hidden-label-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-prompt-profile", default="paper_compatible_three_anchor")
    parser.add_argument("--expected-answer-decision-mode", default="free_generation")
    parser.add_argument("--expected-per-source-top-k", type=int, default=32)
    parser.add_argument("--expected-candidate-pool-top-k", type=int, default=128)
    parser.add_argument("--expected-rerank-top-k", type=int, default=128)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def render(summary: dict[str, Any]) -> str:
    headers = [
        "Rerank Top-k",
        "Gold oracle",
        "Avg # docs",
        *[DATASET_LABELS[dataset] for dataset in DATASETS],
        "MMLU pooled",
        "Micro Avg",
        "Macro Avg (8)",
        "Macro Avg (3 groups)",
    ]
    lines = [
        "# RAG2 versus hidden-state Helpful-only gold oracle",
        "",
        "Both policies use the same 6,545 questions and the same paper-balanced 4k -> MedCPT Top-k documents. "
        "RAG2 passes only RAG2 Helpful; hidden-state passes only score > +tau and blocks Neutral/Harmful. "
        "A question with no passing document falls back to the identical cached no-RAG response.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:", "---", "---:"] + ["---:"] * (len(headers) - 3)) + "|",
    ]
    by_k: dict[int, dict[str, dict[str, Any]]] = {}
    for condition in summary["conditions"]:
        metrics = condition["metrics"]
        by_k.setdefault(condition["top_k"], {})[condition["policy"]] = metrics
        cells = [
            str(condition["top_k"]),
            condition["policy"],
            f"{metrics['mean_context_documents']:.2f}",
            *[f"{metrics['dataset_accuracy'][dataset] * 100:.2f}" for dataset in DATASETS],
            f"{metrics['mmlu_pooled_accuracy'] * 100:.2f}",
            f"{metrics['micro_accuracy'] * 100:.2f}",
            f"{metrics['macro_8_accuracy'] * 100:.2f}",
            f"{metrics['macro_3_accuracy'] * 100:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Hidden minus RAG2",
            "",
            "| Top-k | Avg docs | Micro | Macro (8) | Macro (3 groups) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for top_k in TOP_K_VALUES:
        rag2 = by_k[top_k]["RAG2 Helpful only"]
        hidden = by_k[top_k]["Hidden Helpful only"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(top_k),
                    f"{hidden['mean_context_documents'] - rag2['mean_context_documents']:+.2f}",
                    f"{(hidden['micro_accuracy'] - rag2['micro_accuracy']) * 100:+.2f}%p",
                    f"{(hidden['macro_8_accuracy'] - rag2['macro_8_accuracy']) * 100:+.2f}%p",
                    f"{(hidden['macro_3_accuracy'] - rag2['macro_3_accuracy']) * 100:+.2f}%p",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pair-label agreement", ""])
    cross = summary["hidden_label_manifest"].get("rag2_hidden_cross_counts") or {}
    total_cross: dict[str, int] = {}
    for values in cross.values():
        for key, count in values.items():
            total_cross[key] = total_cross.get(key, 0) + int(count)
    lines.extend(
        [
            "| RAG2 label | Hidden Helpful | Hidden Neutral | Hidden Harmful |",
            "|---|---:|---:|---:|",
        ]
    )
    for rag2_label in ("Helpful", "Discard", "Not Helpful", "Excluded"):
        values = [total_cross.get(f"{rag2_label}|{hidden}", 0) for hidden in ("Helpful", "Neutral", "Harmful")]
        if sum(values):
            lines.append(f"| {rag2_label} | {values[0]:,} | {values[1]:,} | {values[2]:,} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    label_manifest = json.loads(args.hidden_label_manifest.read_text(encoding="utf-8"))
    if int(label_manifest.get("questions", -1)) != sum(EXPECTED_DATASET_COUNTS.values()):
        raise RuntimeError("Hidden-label manifest does not cover the 6,545-question cohort")
    progress = PipelineProgress(
        overall_total=2 * len(TOP_K_VALUES) * sum(EXPECTED_DATASET_COUNTS.values()),
        desc="OracleComparisonSummary",
    )
    conditions: list[dict[str, Any]] = []
    try:
        stage = 0
        for top_k in TOP_K_VALUES:
            for policy, root, directory, expected_policy in (
                (
                    "RAG2 Helpful only",
                    args.rag2_oracle_results_root,
                    f"oracle_rag_rag2_top{top_k}",
                    "rag2",
                ),
                (
                    "Hidden Helpful only",
                    args.hidden_oracle_results_root,
                    f"oracle_rag_hidden_three_class_top{top_k}",
                    "hidden_three_class",
                ),
            ):
                stage += 1
                progress.set_stage(
                    f"{stage}/12 validate and summarize {policy} Top-{top_k}",
                    total=sum(EXPECTED_DATASET_COUNTS.values()),
                )
                rows, run_dir, config = load_condition(
                    root / directory,
                    expected_case="oracle_rag",
                    expected_top_k=top_k,
                    expected_prompt_profile=args.expected_prompt_profile,
                    expected_answer_decision_mode=args.expected_answer_decision_mode,
                    expected_per_source_top_k=args.expected_per_source_top_k,
                    expected_candidate_pool_top_k=args.expected_candidate_pool_top_k,
                    expected_rerank_top_k=args.expected_rerank_top_k,
                    expected_paper_balanced_projection=True,
                    progress=progress,
                )
                if config.get("oracle_policy") != expected_policy:
                    raise RuntimeError(
                        f"Oracle policy mismatch in {run_dir}: {config.get('oracle_policy')} != {expected_policy}"
                    )
                conditions.append(
                    {
                        "top_k": top_k,
                        "policy": policy,
                        "run_dir": str(run_dir.resolve()),
                        "metrics": summarize_rows(rows),
                    }
                )
    finally:
        progress.close()
    summary = {
        "summary_version": SUMMARY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
        "hidden_label_manifest": label_manifest,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_text(args.output_dir / "summary_table_pretty.txt", render(summary))
    csv_path = args.output_dir / "summary.csv"
    temporary = csv_path.with_name(csv_path.name + ".partial")
    fieldnames = [
        "top_k",
        "policy",
        "mean_context_documents",
        *[f"{dataset}_accuracy" for dataset in DATASETS],
        "mmlu_pooled_accuracy",
        "micro_accuracy",
        "macro_8_accuracy",
        "macro_3_accuracy",
        "run_dir",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in conditions:
            metrics = condition["metrics"]
            writer.writerow(
                {
                    "top_k": condition["top_k"],
                    "policy": condition["policy"],
                    "mean_context_documents": metrics["mean_context_documents"],
                    **{
                        f"{dataset}_accuracy": metrics["dataset_accuracy"][dataset]
                        for dataset in DATASETS
                    },
                    "mmlu_pooled_accuracy": metrics["mmlu_pooled_accuracy"],
                    "micro_accuracy": metrics["micro_accuracy"],
                    "macro_8_accuracy": metrics["macro_8_accuracy"],
                    "macro_3_accuracy": metrics["macro_3_accuracy"],
                    "run_dir": condition["run_dir"],
                }
            )
    os.replace(temporary, csv_path)
    print(render(summary))


if __name__ == "__main__":
    main()

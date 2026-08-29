#!/usr/bin/env python3
"""Validate and summarize direct-only versus broad semantic gold oracles."""

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
    sample_keys,
    summarize_rows,
)


POLICIES = (
    ("semantic_direct", "Direct support only"),
    ("semantic_direct_supporting", "Direct + supporting evidence"),
)
SUMMARY_VERSION = "rag2_external_semantic_oracle_topk_summary_v1"
EXPECTED_DYNAMIC_UNION_PAIRS = 211_875


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--semantic-label-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-prompt-profile", default="paper_compatible_three_anchor")
    parser.add_argument("--expected-answer-decision-mode", default="free_generation")
    parser.add_argument("--expected-per-source-top-k", type=int, default=32)
    parser.add_argument("--expected-candidate-pool-top-k", type=int, default=128)
    parser.add_argument("--expected-rerank-top-k", type=int, default=128)
    parser.add_argument(
        "--expected-paper-balanced-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require each k to use four source-local dense Top-k lists followed by MedCPT rerank Top-k.",
    )
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
        "Semantic gold oracle",
        "Avg # docs",
        *[DATASET_LABELS[dataset] for dataset in DATASETS],
        "MMLU pooled",
        "Micro Avg",
        "Macro Avg (8)",
        "Macro Avg (3 groups)",
    ]
    protocol_description = (
        "For each k, dense Top-k is selected independently from PubMed, PMC, CPG, and Textbooks "
        "(4k documents), then MedCPT reranks that exact pool to Top-k. There is no backfill: a question "
        "with no accepted semantic document receives an empty document context."
        if summary["paper_balanced_projection"]
        else "For each k, only the rank-1..k prefix of the cached global MedCPT rerank Top-32 is eligible. "
        "There is no backfill: a question with no accepted document receives an empty document context."
    )
    lines = [
        "# GPT-5.6-Terra semantic-label gold-oracle Top-k sweep",
        "",
        "Both policies use the same 6,545 questions and the same cached retrieval/reranking scores. "
        + protocol_description,
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
            condition["display_label"],
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
            "## Direct + supporting evidence minus direct support only",
            "",
            "| Top-k | Avg docs | Micro | Macro (8) | Macro (3 groups) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for top_k in TOP_K_VALUES:
        direct = by_k[top_k]["semantic_direct"]
        broad = by_k[top_k]["semantic_direct_supporting"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(top_k),
                    f"{broad['mean_context_documents'] - direct['mean_context_documents']:+.2f}",
                    f"{(broad['micro_accuracy'] - direct['micro_accuracy']) * 100:+.2f}%p",
                    f"{(broad['macro_8_accuracy'] - direct['macro_8_accuracy']) * 100:+.2f}%p",
                    f"{(broad['macro_3_accuracy'] - direct['macro_3_accuracy']) * 100:+.2f}%p",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, conditions: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    fieldnames = [
        "top_k",
        "policy",
        "display_label",
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
                    "display_label": condition["display_label"],
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
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    label_manifest = json.loads(args.semantic_label_manifest.read_text(encoding="utf-8"))
    expected_questions = sum(EXPECTED_DATASET_COUNTS.values())
    if label_manifest.get("status") != "complete":
        raise RuntimeError("Semantic oracle export manifest is not complete")
    if int(label_manifest.get("questions", -1)) != expected_questions:
        raise RuntimeError(
            f"Semantic oracle export covers {label_manifest.get('questions')} questions, expected {expected_questions}"
        )
    expected_label_pairs = (
        EXPECTED_DYNAMIC_UNION_PAIRS
        if args.expected_paper_balanced_projection
        else expected_questions * 32
    )
    if int(label_manifest.get("pairs", -1)) != expected_label_pairs:
        raise RuntimeError(
            f"Semantic oracle export pair mismatch: {label_manifest.get('pairs')} != {expected_label_pairs}"
        )
    if args.expected_paper_balanced_projection and label_manifest.get("dynamic_top_k_values") != list(TOP_K_VALUES):
        raise RuntimeError("Semantic oracle export omits the exact dynamic Top-k membership contract")

    condition_count = len(POLICIES) * len(TOP_K_VALUES)
    progress = PipelineProgress(
        overall_total=condition_count * expected_questions + 3,
        desc="SemanticOracleSummary",
    )
    conditions: list[dict[str, Any]] = []
    reference_keys: set[tuple[str, str]] | None = None
    try:
        stage_index = 0
        for top_k in TOP_K_VALUES:
            for policy, display_label in POLICIES:
                stage_index += 1
                progress.set_stage(
                    f"{stage_index}/{condition_count + 1} validate {display_label} Top-{top_k}",
                    total=expected_questions,
                )
                run_root = args.results_root / f"oracle_rag_{policy}_top{top_k}"
                rows, run_dir, config = load_condition(
                    run_root,
                    expected_case="oracle_rag",
                    expected_top_k=top_k,
                    expected_prompt_profile=args.expected_prompt_profile,
                    expected_answer_decision_mode=args.expected_answer_decision_mode,
                    expected_per_source_top_k=args.expected_per_source_top_k,
                    expected_candidate_pool_top_k=args.expected_candidate_pool_top_k,
                    expected_rerank_top_k=args.expected_rerank_top_k,
                    expected_paper_balanced_projection=args.expected_paper_balanced_projection,
                    progress=progress,
                )
                if config.get("oracle_policy") != policy:
                    raise RuntimeError(
                        f"Oracle policy mismatch in {run_dir}: {config.get('oracle_policy')!r} != {policy!r}"
                    )
                keys = sample_keys(rows)
                if reference_keys is None:
                    reference_keys = keys
                elif keys != reference_keys:
                    raise RuntimeError(f"Cross-condition cohort mismatch in {run_dir}")
                conditions.append(
                    {
                        "top_k": top_k,
                        "policy": policy,
                        "display_label": display_label,
                        "run_dir": str(run_dir.resolve()),
                        "metrics": summarize_rows(rows),
                    }
                )

        summary = {
            "summary_version": SUMMARY_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "results_root": str(args.results_root.resolve()),
            "semantic_label_manifest": str(args.semantic_label_manifest.resolve()),
            "paper_balanced_projection": args.expected_paper_balanced_projection,
            "semantic_annotation": {
                key: label_manifest.get(key)
                for key in ("annotation_version", "prompt_version", "model", "reasoning_effort")
            },
            "cohort_questions": expected_questions,
            "conditions": conditions,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table = render(summary)
        progress.set_stage(f"{condition_count + 1}/{condition_count + 1} write summary artifacts", total=3)
        atomic_json(args.output_dir / "summary.json", summary)
        progress.update(1)
        write_csv(args.output_dir / "summary.csv", conditions)
        progress.update(1)
        atomic_text(args.output_dir / "summary_table_pretty.txt", table + "\n")
        progress.update(1)
    finally:
        progress.close()
    print(table)
    print(f"Semantic oracle summary: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

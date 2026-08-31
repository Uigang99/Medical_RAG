#!/usr/bin/env python3
"""Validate and summarize Top-8 semantic behavioral subset Oracle results."""

from __future__ import annotations

import argparse
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
    load_condition,
    sample_keys,
    summarize_rows,
)


SUMMARY_VERSION = "rag2_semantic_behavioral_subset_oracle_summary_v1"
TOP_K = 8
CONDITIONS = (
    ("semantic", "semantic_direct", "Semantic Direct-all"),
    ("semantic", "semantic_direct_supporting", "Semantic Direct+Supporting-all"),
    ("subset", "behavioral_best_direct", "Behavioral Best-Direct subset"),
    (
        "subset",
        "behavioral_best_semantic_candidates",
        "Behavioral Best-(Direct+Supporting) subset",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-results-root", type=Path, required=True)
    parser.add_argument("--subset-results-root", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
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


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    sample = row.get("sample") or {}
    return str(sample.get("dataset") or ""), str(sample.get("id") or "")


def is_correct(row: dict[str, Any]) -> bool:
    return bool((row.get("evaluation") or {}).get("correct"))


def paired_change(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, int]:
    reference_by_key = {row_key(row): is_correct(row) for row in reference}
    candidate_by_key = {row_key(row): is_correct(row) for row in candidate}
    if reference_by_key.keys() != candidate_by_key.keys():
        raise RuntimeError("Paired comparison cohort mismatch")
    wrong_to_correct = sum(
        int(not reference_by_key[key] and candidate_by_key[key]) for key in reference_by_key
    )
    correct_to_wrong = sum(
        int(reference_by_key[key] and not candidate_by_key[key]) for key in reference_by_key
    )
    return {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "net_correct": wrong_to_correct - correct_to_wrong,
    }


def render(summary: dict[str, Any]) -> str:
    headers = [
        "Condition",
        "Avg # docs",
        *[DATASET_LABELS[dataset] for dataset in DATASETS],
        "MMLU pooled",
        "Micro Avg",
        "Macro Avg (8)",
        "Macro Avg (3 groups)",
    ]
    lines = [
        "# Top-8 semantic-candidate behavioral subset Oracle",
        "",
        "All four rows use the same 6,545 questions and exact paper-balanced 4x8 -> MedCPT Top-8 documents. "
        "Subset selection uses gold direct-choice margin; final scoring uses the shared rationale + fixed terminal answer contract.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    by_policy = {}
    for condition in summary["conditions"]:
        metrics = condition["metrics"]
        by_policy[condition["policy"]] = condition
        cells = [
            condition["display_label"],
            f"{metrics['mean_context_documents']:.2f}",
            *[f"{metrics['dataset_accuracy'][dataset] * 100:.2f}" for dataset in DATASETS],
            f"{metrics['mmlu_pooled_accuracy'] * 100:.2f}",
            f"{metrics['micro_accuracy'] * 100:.2f}",
            f"{metrics['macro_8_accuracy'] * 100:.2f}",
            f"{metrics['macro_3_accuracy'] * 100:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    direct = by_policy["behavioral_best_direct"]["metrics"]
    broad = by_policy["behavioral_best_semantic_candidates"]["metrics"]
    change = summary["paired_best_broad_vs_best_direct"]
    selection = summary["selection_summary"]
    lines.extend(
        [
            "",
            "## Core hypothesis",
            "",
            "| Comparison | Micro | Macro (8) | Macro (3 groups) | W->C | C->W | Net correct |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| Best-(Direct+Supporting) minus Best-Direct | "
            f"{(broad['micro_accuracy'] - direct['micro_accuracy']) * 100:+.2f}%p | "
            f"{(broad['macro_8_accuracy'] - direct['macro_8_accuracy']) * 100:+.2f}%p | "
            f"{(broad['macro_3_accuracy'] - direct['macro_3_accuracy']) * 100:+.2f}%p | "
            f"{change['wrong_to_correct']} | {change['correct_to_wrong']} | {change['net_correct']:+d} |",
            "",
            f"- Questions whose best broad margin strictly exceeds best Direct margin: "
            f"{selection['broad_margin_strictly_exceeds_direct_rate'] * 100:.2f}%",
            f"- Questions where the selected broad subset contains Supporting evidence: "
            f"{selection['policies']['behavioral_best_semantic_candidates']['questions_with_supporting_selected_rate'] * 100:.2f}%",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    selection_summary = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    expected_questions = sum(EXPECTED_DATASET_COUNTS.values())
    if selection_summary.get("status") != "complete":
        raise RuntimeError("Subset-selection summary is incomplete")
    if int(selection_summary.get("questions", -1)) != expected_questions:
        raise RuntimeError("Subset-selection cohort is not the full 6,545-question benchmark")
    if int(selection_summary.get("top_k", -1)) != TOP_K:
        raise RuntimeError("Subset-selection summary is not Top-8")
    if selection_summary.get("candidate_semantic_labels") != [
        "direct_support",
        "supporting_evidence",
    ]:
        raise RuntimeError("This summary requires exactly Direct + Supporting semantic candidates")
    progress = PipelineProgress(
        overall_total=len(CONDITIONS) * expected_questions + 2,
        desc="SemanticBehaviorSubsetSummary",
    )
    conditions = []
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    reference_keys: set[tuple[str, str]] | None = None
    try:
        for index, (root_name, policy, display_label) in enumerate(CONDITIONS, start=1):
            progress.set_stage(
                f"{index}/5 validate {display_label}",
                total=expected_questions,
            )
            root = args.semantic_results_root if root_name == "semantic" else args.subset_results_root
            run_root = root / f"oracle_rag_{policy}_top{TOP_K}"
            rows, run_dir, config = load_condition(
                run_root,
                expected_case="oracle_rag",
                expected_top_k=TOP_K,
                expected_prompt_profile=args.expected_prompt_profile,
                expected_answer_decision_mode=args.expected_answer_decision_mode,
                expected_per_source_top_k=args.expected_per_source_top_k,
                expected_candidate_pool_top_k=args.expected_candidate_pool_top_k,
                expected_rerank_top_k=args.expected_rerank_top_k,
                expected_paper_balanced_projection=True,
                progress=progress,
            )
            if config.get("oracle_policy") != policy:
                raise RuntimeError(f"Oracle policy mismatch in {run_dir}")
            keys = sample_keys(rows)
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise RuntimeError(f"Cross-condition cohort mismatch in {run_dir}")
            rows_by_policy[policy] = rows
            conditions.append(
                {
                    "policy": policy,
                    "display_label": display_label,
                    "run_dir": str(run_dir.resolve()),
                    "metrics": summarize_rows(rows),
                }
            )
        summary = {
            "summary_version": SUMMARY_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions": expected_questions,
            "top_k": TOP_K,
            "selection_summary_path": str(args.selection_summary.resolve()),
            "selection_summary": selection_summary,
            "conditions": conditions,
            "paired_best_broad_vs_best_direct": paired_change(
                rows_by_policy["behavioral_best_direct"],
                rows_by_policy["behavioral_best_semantic_candidates"],
            ),
        }
        table = render(summary)
        progress.set_stage("5/5 write validated comparison", total=2)
        atomic_json(args.output_dir / "summary.json", summary)
        progress.update(1)
        atomic_text(args.output_dir / "summary_table_pretty.txt", table + "\n")
        progress.update(1)
    finally:
        progress.close()
    print(table)
    print(f"Subset Oracle summary: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and summarize gold-free semantic-candidate subset policies."""

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


SUMMARY_VERSION = "rag2_gold_free_subset_policy_summary_v1"
TOP_K = 8
POLICIES = (
    "gold_free_max_confidence",
    "gold_free_min_entropy",
    "gold_free_consensus_confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rag-root", type=Path, required=True)
    parser.add_argument("--semantic-results-root", type=Path, required=True)
    parser.add_argument("--gold-oracle-results-root", type=Path, required=True)
    parser.add_argument("--gold-free-results-root", type=Path, required=True)
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
    left = {row_key(row): is_correct(row) for row in reference}
    right = {row_key(row): is_correct(row) for row in candidate}
    if left.keys() != right.keys():
        raise RuntimeError("Paired comparison cohort mismatch")
    wrong_to_correct = sum(not left[key] and right[key] for key in left)
    correct_to_wrong = sum(left[key] and not right[key] for key in left)
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
        "# Top-8 gold-free semantic-candidate subset selection",
        "",
        "All policies use only the four direct-choice logits/probabilities from each subset. "
        "Gold answers are excluded from selection and used only for the reported audit metrics.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    by_policy: dict[str, dict[str, Any]] = {}
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
    semantic_all = by_policy["semantic_direct_supporting"]["metrics"]["micro_accuracy"]
    oracle = by_policy["behavioral_best_semantic_candidates"]["metrics"]["micro_accuracy"]
    denominator = oracle - semantic_all
    lines.extend(
        [
            "",
            "## Gold-free recovery of the final rationale-answer Oracle gap",
            "",
            "| Policy | Micro gain over Direct+Supporting-all | Oracle recovery | W->C | C->W | Net |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        metrics = by_policy[policy]["metrics"]
        change = summary["paired_vs_semantic_all"][policy]
        recovery = (metrics["micro_accuracy"] - semantic_all) / denominator if denominator else None
        lines.append(
            f"| {policy} | {(metrics['micro_accuracy'] - semantic_all) * 100:+.2f}%p | "
            f"{'n/a' if recovery is None else f'{recovery * 100:.1f}%'} | "
            f"{change['wrong_to_correct']} | {change['correct_to_wrong']} | "
            f"{change['net_correct']:+d} |"
        )
    audit = summary["selection_audit"]
    lines.extend(
        [
            "",
            "## Cached direct-choice audit before rationale generation",
            "",
            "| Policy | Direct-choice acc. | Exact gold-Oracle subset | Margin regret |",
            "|---|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        values = audit["policies"][policy]
        lines.append(
            f"| {policy} | {values['direct_choice_accuracy'] * 100:.2f} | "
            f"{values['exact_oracle_subset_rate'] * 100:.2f}% | "
            f"{values['mean_gold_margin_regret']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    expected_questions = sum(EXPECTED_DATASET_COUNTS.values())
    selection_audit = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    if selection_audit.get("status") != "complete":
        raise RuntimeError("Gold-free selection audit is incomplete")
    if bool(selection_audit.get("selection_uses_gold")):
        raise RuntimeError("Selection audit unexpectedly reports gold-dependent selection")
    if int(selection_audit.get("questions", -1)) != expected_questions:
        raise RuntimeError("Gold-free selection cohort mismatch")
    if set(selection_audit.get("policies") or {}) != set(POLICIES):
        raise RuntimeError("Gold-free policy set mismatch")

    specs = [
        (
            "no_rag",
            "No-RAG",
            args.no_rag_root,
            "no_rag",
            None,
            False,
        ),
        (
            "semantic_direct",
            "Semantic Direct-all",
            args.semantic_results_root / "oracle_rag_semantic_direct_top8",
            "oracle_rag",
            TOP_K,
            True,
        ),
        (
            "semantic_direct_supporting",
            "Semantic Direct+Supporting-all",
            args.semantic_results_root / "oracle_rag_semantic_direct_supporting_top8",
            "oracle_rag",
            TOP_K,
            True,
        ),
        (
            "behavioral_best_semantic_candidates",
            "Gold-margin Best-(Direct+Supporting)",
            args.gold_oracle_results_root
            / "oracle_rag_behavioral_best_semantic_candidates_top8",
            "oracle_rag",
            TOP_K,
            True,
        ),
        *[
            (
                policy,
                policy,
                args.gold_free_results_root / f"oracle_rag_{policy}_top8",
                "oracle_rag",
                TOP_K,
                True,
            )
            for policy in POLICIES
        ],
    ]
    progress = PipelineProgress(
        overall_total=len(specs) * expected_questions + 2,
        desc="GoldFreeSubsetSummary",
    )
    conditions: list[dict[str, Any]] = []
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    reference_keys: set[tuple[str, str]] | None = None
    try:
        for index, (policy, label, root, case, top_k, projected) in enumerate(specs, start=1):
            progress.set_stage(f"{index}/{len(specs) + 1} validate {label}", total=expected_questions)
            rows, run_dir, config = load_condition(
                root,
                expected_case=case,
                expected_top_k=top_k,
                expected_prompt_profile=args.expected_prompt_profile,
                expected_answer_decision_mode=args.expected_answer_decision_mode,
                expected_per_source_top_k=args.expected_per_source_top_k,
                expected_candidate_pool_top_k=args.expected_candidate_pool_top_k,
                expected_rerank_top_k=args.expected_rerank_top_k,
                expected_paper_balanced_projection=projected,
                progress=progress,
            )
            if case == "oracle_rag" and config.get("oracle_policy") != policy:
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
                    "display_label": label,
                    "run_dir": str(run_dir.resolve()),
                    "metrics": summarize_rows(rows),
                }
            )
        semantic_rows = rows_by_policy["semantic_direct_supporting"]
        summary = {
            "summary_version": SUMMARY_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions": expected_questions,
            "top_k": TOP_K,
            "selection_summary_path": str(args.selection_summary.resolve()),
            "selection_audit": selection_audit,
            "conditions": conditions,
            "paired_vs_semantic_all": {
                policy: paired_change(semantic_rows, rows_by_policy[policy])
                for policy in POLICIES
            },
        }
        table = render(summary)
        progress.set_stage(f"{len(specs) + 1}/{len(specs) + 1} write comparison", total=2)
        atomic_json(args.output_dir / "summary.json", summary)
        progress.update(1)
        atomic_text(args.output_dir / "summary_table_pretty.txt", table + "\n")
        progress.update(1)
    finally:
        progress.close()
    print(table)
    print(f"Gold-free subset summary: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

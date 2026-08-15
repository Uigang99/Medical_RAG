#!/usr/bin/env python3
from __future__ import annotations

"""Merge rationale and direct-choice oracle sweeps into one comparison table."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationale-base-summary", type=Path, required=True)
    parser.add_argument("--rationale-tau04-summary", type=Path, required=True)
    parser.add_argument("--direct-choice-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path, expected_mode: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    actual_mode = str(value.get("answer_decision_mode") or "paper_exact_terminal")
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"Unexpected answer mode in {path}: expected={expected_mode} actual={actual_mode}"
        )
    return value


def policy(summary: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return summary["conditions"][name]
    except KeyError as error:
        raise RuntimeError(f"Missing policy {name!r} in summary") from error


def validate_cohort(values: list[dict[str, Any]]) -> None:
    question_shapes = []
    for value in values:
        baseline = value["baseline"]
        question_shapes.append(
            {
                key: int(metrics["questions"])
                for key, metrics in baseline.items()
            }
        )
    if any(shape != question_shapes[0] for shape in question_shapes[1:]):
        raise RuntimeError(f"Oracle summaries use different cohorts: {question_shapes}")


def baseline_row(mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    baseline = summary["baseline"]
    return {
        "answer_mode": mode,
        "policy": "no_rag",
        "top_k": None,
        "overall": baseline["overall"],
        "datasets": {
            name: metrics for name, metrics in baseline.items() if name != "overall"
        },
        "mean_context_documents": 0.0,
        "zero_context_questions": int(baseline["overall"]["questions"]),
        "gains_vs_no_rag": 0,
        "losses_vs_no_rag": 0,
        "net_gain_vs_no_rag": 0,
    }


def condition_rows(
    mode: str,
    summary: dict[str, Any],
    names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        values = policy(summary, name)
        for top_k, metrics in sorted(values["top_k"].items(), key=lambda item: int(item[0])):
            overall = metrics["overall"]
            rows.append(
                {
                    "answer_mode": mode,
                    "policy": name,
                    "hidden_threshold": values.get("hidden_threshold"),
                    "top_k": int(top_k),
                    "overall": overall,
                    "datasets": metrics["datasets"],
                    "mean_context_documents": overall["mean_context_documents"],
                    "zero_context_questions": overall["zero_context_questions"],
                    "gains_vs_no_rag": overall["gains_vs_no_rag"],
                    "losses_vs_no_rag": overall["losses_vs_no_rag"],
                    "net_gain_vs_no_rag": overall["net_gain_vs_no_rag"],
                }
            )
    return rows


def accuracy(row: dict[str, Any], dataset: str) -> float:
    metrics = row["overall"] if dataset == "overall" else row["datasets"][dataset]
    return float(metrics["accuracy"]) * 100.0


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "RAG2 vs Hidden-State Oracle: Final-Answer Protocol Ablation",
        "",
        "| Final answer | Policy | Top-k | Avg docs | MedMCQA | MedQA | Overall | Zero docs | Gains | Losses | Net |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        top_k = "-" if row["top_k"] is None else str(row["top_k"])
        lines.append(
            f"| {row['answer_mode']} | {row['policy']} | {top_k} | "
            f"{row['mean_context_documents']:.2f} | {accuracy(row, 'medmcqa'):.2f}% | "
            f"{accuracy(row, 'medqa'):.2f}% | {accuracy(row, 'overall'):.2f}% | "
            f"{row['zero_context_questions']:,} | {row['gains_vs_no_rag']:,} | "
            f"{row['losses_vs_no_rag']:,} | {row['net_gain_vs_no_rag']:+,} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rationale_base = load(args.rationale_base_summary, "paper_exact_terminal")
    rationale_tau04 = load(args.rationale_tau04_summary, "paper_exact_terminal")
    direct = load(args.direct_choice_summary, "constrained_choice")
    validate_cohort([rationale_base, rationale_tau04, direct])

    rows = [baseline_row("rationale", rationale_base)]
    rows.extend(
        condition_rows(
            "rationale",
            rationale_base,
            ["rag2", "hidden_tau_0", "hidden_tau_0p2"],
        )
    )
    rows.extend(condition_rows("rationale", rationale_tau04, ["hidden_tau_0p4"]))
    rows.append(baseline_row("direct_choice", direct))
    rows.extend(
        condition_rows(
            "direct_choice",
            direct,
            ["rag2", "hidden_tau_0", "hidden_tau_0p2", "hidden_tau_0p4"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "rag2_oracle_answer_mode_ablation_summary_v1",
        "rows": rows,
        "sources": {
            "rationale_base": str(args.rationale_base_summary.resolve()),
            "rationale_tau04": str(args.rationale_tau04_summary.resolve()),
            "direct_choice": str(args.direct_choice_summary.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary_table_pretty.txt").write_text(
        render(rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

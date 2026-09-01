#!/usr/bin/env python3
"""Combine continuous-gate preflight diagnostics into one Go/Revise/Stop report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-summary", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis_summary.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    cross = dict(analysis["cross_dataset_checks"])
    gate_pass = bool(gate["gate_contract_pass"])
    core_pass = all(
        bool(cross[name])
        for name in (
            "semantic_probabilities_usable_without_calibration",
            "risk_constrained_policy_has_oracle_headroom",
            "conditional_no_evidence_hypothesis_replicates",
        )
    )
    answer_agreement = bool(cross["direct_and_rationale_utility_targets_agree"])
    if core_pass and gate_pass:
        decision = "GO_BOUNDED_PILOT"
    elif not gate_pass:
        decision = "STOP_CONTINUOUS_ATTENTION_GATE"
    else:
        decision = "REVISE_SEMANTIC_RISK_POLICY"
    summary = {
        "decision": decision,
        "scope": "internal validation preflight; final benchmark test set was not used",
        "core_semantic_and_set_assumptions_pass": core_pass,
        "attention_gate_mechanism_pass": gate_pass,
        "answer_modes_agree": answer_agreement,
        "answer_mode_recommendation": analysis["answer_mode_recommendation"],
        "analysis_summary": str(args.analysis_summary.resolve()),
        "gate_summary": str(args.gate_summary.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "preflight_summary.json", summary)

    lines = [
        "# Continuous document-gate preflight",
        "",
        f"**Decision: {decision}**",
        "",
        "This report uses only question-level internal validation data. Gold answers are used only for diagnostics and Oracle headroom.",
        "",
        "| Dataset | Semantic AUROC | ECE | Hard semantic acc. | Best risk-Oracle acc. | Direct↔rationale median Spearman | Sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in analysis["datasets"].items():
        calibration = values["semantic_calibration"]
        subsets = values["subset_analysis"]
        modes = values["answer_mode_analysis"]
        best = subsets["best_risk_policy"]
        lines.append(
            f"| {dataset} | {number(calibration['auroc'])} | {number(calibration['ece'])} | "
            f"{percent(subsets['policies']['hard_semantic']['accuracy'])} | "
            f"{percent(subsets['policies'][best]['accuracy'])} | "
            f"{number(modes['median_per_question_utility_spearman'])} | "
            f"{percent(modes['utility_sign_agreement'])} |"
        )
    metrics = gate["metrics"]
    lines.extend(
        [
            "",
            "## Attention-gate mechanism",
            "",
            f"- `gate=0` vs physical deletion prediction agreement: {percent(metrics['zero_vs_delete_prediction_agreement'])}",
            f"- Physical deletion vs `gate=0` median per-question Spearman: {number(metrics['median_per_question_physical_vs_zero_gate_spearman'])}",
            f"- Physical value vs amplification median per-question Spearman: {number(metrics['median_per_question_physical_vs_amplification_spearman'])}",
            f"- Monotonic direction agreement: {percent(metrics['monotonic_direction_agreement'])}",
            "",
            "## Answer-format decision",
            "",
            analysis["answer_mode_recommendation"],
            "",
            "- If answer-mode agreement passes, direct-choice may be used for the cheaper teacher diagnostic while both modes remain separate reported evaluations.",
            "- If it fails, select one format and use it consistently for target construction, baselines, and final evaluation. Do not train on direct-choice utility and claim rationale-mode utility.",
        ]
    )
    (args.output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

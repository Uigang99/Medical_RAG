#!/usr/bin/env python3
"""Compare the matched SFT control and semantic-utilization pilot runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--proposed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def format_seconds(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_run(path: Path, expected_objective: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = path / "training_summary.json"
    predictions_path = path / "final_test_predictions.jsonl"
    for required in (summary_path, predictions_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("objective") != expected_objective:
        raise ValueError(f"Expected {expected_objective}, found {summary.get('objective')}: {path}")
    rows = list(iter_jsonl(predictions_path))
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate test predictions: {predictions_path}")
    return summary, indexed


def mean_metric(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows]))


def paired_deltas(
    control: list[dict[str, Any]], proposed: list[dict[str, Any]], indices: np.ndarray
) -> dict[str, float]:
    return {
        name: mean_metric([proposed[i] for i in indices], name)
        - mean_metric([control[i] for i in indices], name)
        for name in ("semantic_preference", "direct_positive", "noise_abs", "no_rag_correct")
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Semantic-utilization pilot comparison",
        "",
        f"- Dataset: {report['dataset']}",
        f"- Paired test N: {report['test_questions']}",
        "- Comparison: semantic-utilization LoRA minus matched SFT-control LoRA",
        "",
        "| Metric | SFT control | Semantic utilization | Change | Paired 95% bootstrap CI | Meaning |",
        "|---|---:|---:|---:|---:|---|",
    ]
    meanings = {
        "semantic_preference": "valid context has higher fixed-response likelihood than invalid context; higher is better",
        "direct_positive": "removing the reference Direct-Support document lowers likelihood; higher is better",
        "noise_abs": "absolute likelihood change after No/Misleading documents are added; lower is better",
        "no_rag_correct": "No-RAG choice accuracy preservation diagnostic; higher is better",
    }
    for name in meanings:
        metric = report["metrics"][name]
        lines.append(
            f"| {name} | {metric['control']:.6f} | {metric['proposed']:.6f} | "
            f"{metric['delta']:+.6f} | [{metric['ci95'][0]:+.6f}, {metric['ci95'][1]:+.6f}] | {meanings[name]} |"
        )
    lines.extend(
        [
            "",
            f"- Pre-registered pilot pass: **{report['success_criterion']['passed']}**",
            "- Pass rule: semantic preference >= +5%p, direct-positive >= +3%p, "
            "noise_abs <= control +0.005, No-RAG accuracy >= control -1%p.",
            "- This pilot does not use behavioral utility or final MCQ accuracy as its optimization target.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    print("[overall 1/2] [load and verify matched runs | 0/2 0.0% | ETA unknown]", flush=True)
    control_summary, control_by_id = load_run(args.control_dir, "sft_control")
    proposed_summary, proposed_by_id = load_run(args.proposed_dir, "semantic_utilization")
    if control_summary["dataset"] != proposed_summary["dataset"]:
        raise RuntimeError("Dataset mismatch between runs")
    if control_summary["data_contract_fingerprint"] != proposed_summary["data_contract_fingerprint"]:
        raise RuntimeError("Prepared-data contract mismatch between runs")
    if set(control_by_id) != set(proposed_by_id):
        raise RuntimeError("Paired test cohort mismatch between runs")
    sample_ids = sorted(control_by_id)
    control = [control_by_id[sample_id] for sample_id in sample_ids]
    proposed = [proposed_by_id[sample_id] for sample_id in sample_ids]
    print(
        f"[overall 1/2] [load and verify matched runs | 2/2 100.0% | "
        f"elapsed {format_seconds(time.time()-started)} | ETA 00:00:00]",
        flush=True,
    )

    replicates = args.bootstrap_replicates
    rng = np.random.default_rng(args.seed)
    bootstrap: dict[str, list[float]] = {
        name: [] for name in ("semantic_preference", "direct_positive", "noise_abs", "no_rag_correct")
    }
    bootstrap_started = time.time()
    for replicate in range(replicates):
        indices = rng.integers(0, len(sample_ids), size=len(sample_ids))
        deltas = paired_deltas(control, proposed, indices)
        for name, value in deltas.items():
            bootstrap[name].append(value)
        if (replicate + 1) % 100 == 0 or replicate + 1 == replicates:
            elapsed = time.time() - bootstrap_started
            rate = (replicate + 1) / max(elapsed, 1e-9)
            eta = (replicates - replicate - 1) / rate
            print(
                f"\r[overall 2/2] [paired bootstrap | {replicate+1}/{replicates} "
                f"{100*(replicate+1)/replicates:5.1f}% | {rate:.1f} replicate/s | "
                f"elapsed {format_seconds(elapsed)} | ETA {format_seconds(eta)}]",
                end="\n" if replicate + 1 == replicates else "",
                flush=True,
            )

    full_indices = np.arange(len(sample_ids))
    observed = paired_deltas(control, proposed, full_indices)
    metrics = {}
    for name, delta in observed.items():
        metrics[name] = {
            "control": mean_metric(control, name),
            "proposed": mean_metric(proposed, name),
            "delta": delta,
            "ci95": [float(value) for value in np.quantile(bootstrap[name], [0.025, 0.975])],
        }
    success = {
        "semantic_preference_delta_at_least_0p05": observed["semantic_preference"] >= 0.05,
        "direct_positive_delta_at_least_0p03": observed["direct_positive"] >= 0.03,
        "noise_not_worse_by_more_than_0p005": observed["noise_abs"] <= 0.005,
        "no_rag_accuracy_drop_at_most_0p01": observed["no_rag_correct"] >= -0.01,
    }
    success["passed"] = all(success.values())
    report = {
        "dataset": control_summary["dataset"],
        "test_questions": len(sample_ids),
        "data_contract_fingerprint": control_summary["data_contract_fingerprint"],
        "control_dir": str(args.control_dir.resolve()),
        "proposed_dir": str(args.proposed_dir.resolve()),
        "bootstrap_replicates": replicates,
        "metrics": metrics,
        "success_criterion": success,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "comparison.json", report)
    write_markdown(report, args.output_dir / "COMPARISON.md")
    print(f"Comparison complete: {args.output_dir} pass={success['passed']}", flush=True)


if __name__ == "__main__":
    main()

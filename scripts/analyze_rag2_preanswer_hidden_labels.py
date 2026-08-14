#!/usr/bin/env python3
"""Compare hidden-direction labels with observed MCQ answer transitions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


TRANSITIONS = ("C->C", "C->W", "W->C", "W->W")
LAYERS = ("layer_16", "layer_24", "layer_28", "final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-layer", choices=LAYERS, default="layer_28")
    parser.add_argument("--confidence-thresholds", nargs="+", type=float, default=[0, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def load_pairs(path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    original: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            flat = {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "transition": row["answer_transition"],
                "gold_answer": row["gold_answer"],
                "no_document_answer": row["no_document_answer"],
                "with_document_answer": row["with_document_answer"],
                "gold_choice_logprob_delta": row["gold_choice_logprob_delta"],
            }
            for layer in LAYERS:
                flat[layer] = row["utility_projection_by_layer"][layer]
            records.append(flat)
            original.append(row)
    return pd.DataFrame(records), original


def metric_row(frame: pd.DataFrame, layer: str) -> dict[str, Any]:
    y = (frame["transition"] == "W->C").astype(int).to_numpy()
    scores = frame[layer].to_numpy()
    predictions = (scores > 0).astype(int)
    precision, recall, binary_f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "n": len(frame),
        "helpful_gold": int(y.sum()),
        "not_helpful_gold": int((1 - y).sum()),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "helpful_precision": precision,
        "helpful_recall": recall,
        "helpful_f1": binary_f1,
        "mcc": matthews_corrcoef(y, predictions),
        "auroc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def clustered_bootstrap(
    frame: pd.DataFrame, layer: str, replicates: int, rng: np.random.Generator
) -> list[float]:
    question_ids = frame["sample_id"].unique()
    groups = {question_id: frame[frame["sample_id"] == question_id] for question_id in question_ids}
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(question_ids, size=len(question_ids), replace=True)
        pieces = [groups[question_id] for question_id in sampled]
        y = np.concatenate([(piece["transition"] == "W->C").astype(int) for piece in pieces])
        scores = np.concatenate([piece[layer].to_numpy() for piece in pieces])
        values.append(float(np.mean((scores > 0) == y)))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def run(args: argparse.Namespace) -> None:
    frame, original = load_pairs(args.input_dir / "pairs.jsonl")
    if len(frame) == 0:
        raise RuntimeError("No pair rows found")
    definitive = frame[frame["transition"].isin(["W->C", "C->W"])].copy()
    rng = np.random.default_rng(args.bootstrap_seed)
    report: dict[str, Any] = {
        "pairs": len(frame),
        "definitive_pairs": len(definitive),
        "definitive_fraction": len(definitive) / len(frame),
        "transition_counts": {},
        "definitive_metrics": {},
        "score_distribution": {},
        "confidence_thresholds": {},
    }
    scopes = [(dataset, group) for dataset, group in frame.groupby("dataset")]
    scopes.append(("overall", frame))
    for scope, group in scopes:
        report["transition_counts"][scope] = {
            transition: int((group["transition"] == transition).sum()) for transition in TRANSITIONS
        }
        report["score_distribution"][scope] = {}
        for layer in LAYERS:
            report["score_distribution"][scope][layer] = {
                transition: {
                    "n": int(len(values)),
                    "mean": float(values[layer].mean()),
                    "median": float(values[layer].median()),
                    "positive_rate": float((values[layer] > 0).mean()),
                }
                for transition, values in group.groupby("transition")
            }
    definitive_scopes = [(dataset, group) for dataset, group in definitive.groupby("dataset")]
    definitive_scopes.append(("overall", definitive))
    for scope, group in definitive_scopes:
        report["definitive_metrics"][scope] = {}
        for layer in LAYERS:
            metrics = metric_row(group, layer)
            metrics["question_cluster_bootstrap_accuracy_95ci"] = clustered_bootstrap(
                group, layer, args.bootstrap_replicates, rng
            )
            report["definitive_metrics"][scope][layer] = metrics
    scores = definitive[args.primary_layer].to_numpy()
    y = (definitive["transition"] == "W->C").astype(int).to_numpy()
    for threshold in args.confidence_thresholds:
        retained = np.abs(scores) >= threshold
        predictions = (scores[retained] > 0).astype(int)
        report["confidence_thresholds"][str(threshold)] = {
            "retained": int(retained.sum()),
            "coverage": float(retained.mean()),
            "accuracy": float(accuracy_score(y[retained], predictions)),
            "errors": int(np.sum(y[retained] != predictions)),
        }
    error_pair_ids = set(
        definitive.loc[
            ((definitive[args.primary_layer] > 0).astype(int) != y), "pair_id"
        ].tolist()
    )
    error_rows = [row for row in original if row["pair_id"] in error_pair_ids]
    for row in error_rows:
        row["hidden_label_primary_layer"] = (
            "helpful" if row["utility_projection_by_layer"][args.primary_layer] > 0 else "not helpful"
        )
        row["answer_transition_label"] = (
            "helpful" if row["answer_transition"] == "W->C" else "not helpful"
        )
    atomic_json(args.output_dir / "comparison_report.json", report)
    atomic_jsonl(args.output_dir / "primary_layer_errors.jsonl", error_rows)

    overall = report["definitive_metrics"]["overall"]
    lines = [
        "# Hidden-direction label vs. observed answer-transition label",
        "",
        f"- all pairs: {len(frame):,}",
        f"- definitive changed-answer pairs: {len(definitive):,} ({len(definitive)/len(frame):.2%})",
        "- gold Helpful: W->C; gold Not Helpful: C->W",
        "- hidden Helpful: (hD-h0) dot c > 0",
        "",
        "| Layer | Accuracy | Balanced acc. | Macro-F1 | Helpful P | Helpful R | AUROC | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in LAYERS:
        metric = overall[layer]
        errors = metric["confusion"]["fp"] + metric["confusion"]["fn"]
        lines.append(
            f"| {layer} | {metric['accuracy']:.4f} | {metric['balanced_accuracy']:.4f} | "
            f"{metric['macro_f1']:.4f} | {metric['helpful_precision']:.4f} | "
            f"{metric['helpful_recall']:.4f} | {metric['auroc']:.4f} | {errors} |"
        )
    lines.extend(
        [
            "",
            "Unchanged C->C and W->W pairs are excluded from definitive accuracy because answer correctness "
            "alone does not reveal whether the document strengthened or weakened the gold answer.",
        ]
    )
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Join the pre-answer hidden pilot with Terra semantic evidence labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


LAYERS = ("layer_16", "layer_24", "layer_28", "final")
TERRA_HELPFUL = {"direct_support", "supporting_evidence"}
TERRA_NOT_HELPFUL = {"no_evidence", "misleading_evidence", "indeterminate_or_mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-pilot-dir", type=Path, required=True)
    parser.add_argument("--terra-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-layer", choices=LAYERS, default="layer_28")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def canonical_key(sample_id: str, rank: Any, stable_id: str) -> tuple[str, int, str]:
    return str(sample_id), int(rank), str(stable_id)


def metrics(reference: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    y = reference.astype(int).to_numpy()
    predicted = prediction.astype(int).to_numpy()
    precision, recall, binary_f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "helpful_precision": float(precision),
        "helpful_recall": float(recall),
        "helpful_f1": float(binary_f1),
        "cohen_kappa": float(cohen_kappa_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run(args: argparse.Namespace) -> None:
    pair_rows: list[dict[str, Any]] = []
    targets: set[tuple[str, int, str]] = set()
    with (args.hidden_pilot_dir / "pairs.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            document = row["document"]
            key = canonical_key(row["sample_id"], document["rerank_rank"], document["stable_id"])
            targets.add(key)
            pair_rows.append({"key": key, "row": row})

    terra: dict[tuple[str, int, str], dict[str, Any]] = {}
    for dataset in ("medmcqa", "medqa"):
        path = args.terra_root / dataset / "codex_semantic_labels.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = canonical_key(row["sample_id"], row["doc_rank"], row["doc_stable_id"])
                if key in targets:
                    terra[key] = row
    missing = targets - terra.keys()
    if missing:
        raise RuntimeError(f"Terra labels missing for {len(missing)} pilot pairs")

    joined_rows: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    for item in pair_rows:
        hidden = item["row"]
        label = terra[item["key"]]
        semantic_label = label["semantic_label"]
        if semantic_label not in TERRA_HELPFUL | TERRA_NOT_HELPFUL:
            raise ValueError(f"Unexpected Terra semantic label: {semantic_label}")
        hidden_score = hidden["utility_projection_by_layer"][args.primary_layer]
        joined = dict(hidden)
        joined["terra_annotation"] = label
        joined["terra_binary_label"] = "helpful" if semantic_label in TERRA_HELPFUL else "not helpful"
        joined["hidden_primary_layer"] = args.primary_layer
        joined["hidden_binary_label"] = "helpful" if hidden_score > 0 else "not helpful"
        joined_rows.append(joined)
        record = {
            "dataset": hidden["dataset"],
            "sample_id": hidden["sample_id"],
            "pair_id": label["pair_id"],
            "transition": hidden["answer_transition"],
            "semantic_label": semantic_label,
            "terra_confidence": float(label["confidence"]),
            "terra_y": int(semantic_label in TERRA_HELPFUL),
            "hidden_y": int(hidden_score > 0),
        }
        for layer in LAYERS:
            record[layer] = float(hidden["utility_projection_by_layer"][layer])
        flat.append(record)
    frame = pd.DataFrame(flat)
    report: dict[str, Any] = {
        "pairs": len(frame),
        "join_missing": 0,
        "terra_mapping": {
            "helpful": sorted(TERRA_HELPFUL),
            "not_helpful": sorted(TERRA_NOT_HELPFUL),
        },
        "semantic_label_counts": {
            label: int(count) for label, count in frame["semantic_label"].value_counts().items()
        },
        "binary_contingency": {
            "terra_helpful_hidden_helpful": int(((frame.terra_y == 1) & (frame.hidden_y == 1)).sum()),
            "terra_helpful_hidden_not_helpful": int(((frame.terra_y == 1) & (frame.hidden_y == 0)).sum()),
            "terra_not_helpful_hidden_helpful": int(((frame.terra_y == 0) & (frame.hidden_y == 1)).sum()),
            "terra_not_helpful_hidden_not_helpful": int(((frame.terra_y == 0) & (frame.hidden_y == 0)).sum()),
        },
        "hidden_vs_terra": {},
        "methods_vs_answer_transition": {},
        "hidden_by_semantic_label": {},
        "agreement_by_transition": {},
        "agreement_by_hidden_confidence": {},
    }
    scopes = [(dataset, group) for dataset, group in frame.groupby("dataset")]
    scopes.append(("overall", frame))
    for scope, group in scopes:
        report["hidden_vs_terra"][scope] = {
            layer: metrics(group.terra_y, group[layer] > 0) for layer in LAYERS
        }
        report["agreement_by_transition"][scope] = {
            transition: {
                "n": len(values),
                "agreement": float((values.terra_y == values.hidden_y).mean()),
                "terra_helpful_rate": float(values.terra_y.mean()),
                "hidden_helpful_rate": float(values.hidden_y.mean()),
            }
            for transition, values in group.groupby("transition")
        }
    changed = frame[frame.transition.isin(["W->C", "C->W"])].copy()
    changed["answer_transition_y"] = (changed.transition == "W->C").astype(int)
    changed_scopes = [(dataset, group) for dataset, group in changed.groupby("dataset")]
    changed_scopes.append(("overall", changed))
    for scope, group in changed_scopes:
        report["methods_vs_answer_transition"][scope] = {
            "hidden": metrics(group.answer_transition_y, group.hidden_y),
            "terra": metrics(group.answer_transition_y, group.terra_y),
        }
    for semantic_label, group in frame.groupby("semantic_label"):
        report["hidden_by_semantic_label"][semantic_label] = {
            "n": len(group),
            "hidden_helpful_rate": float(group.hidden_y.mean()),
            "hidden_score_mean": float(group[args.primary_layer].mean()),
            "hidden_score_median": float(group[args.primary_layer].median()),
            "terra_confidence_mean": float(group.terra_confidence.mean()),
        }
    for threshold in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        retained = frame[frame[args.primary_layer].abs() >= threshold]
        report["agreement_by_hidden_confidence"][str(threshold)] = {
            "n": len(retained),
            "coverage": len(retained) / len(frame),
            "agreement": float((retained.terra_y == retained.hidden_y).mean()),
            "cohen_kappa": float(cohen_kappa_score(retained.terra_y, retained.hidden_y)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison_report.json", report)
    write_jsonl(args.output_dir / "joined_labels.jsonl", joined_rows)
    write_jsonl(
        args.output_dir / "disagreements.jsonl",
        (row for row in joined_rows if row["terra_binary_label"] != row["hidden_binary_label"]),
    )
    overall = report["hidden_vs_terra"]["overall"]
    transition = report["methods_vs_answer_transition"]["overall"]
    contingency = report["binary_contingency"]
    lines = [
        "# Hidden-direction vs GPT-5.6-terra semantic labels",
        "",
        f"Pairs joined: {len(frame):,}/{len(frame):,}",
        "",
        "| Terra label | Hidden Helpful | Hidden Not Helpful |",
        "|---|---:|---:|",
        f"| Helpful | {contingency['terra_helpful_hidden_helpful']:,} | {contingency['terra_helpful_hidden_not_helpful']:,} |",
        f"| Not Helpful | {contingency['terra_not_helpful_hidden_helpful']:,} | {contingency['terra_not_helpful_hidden_not_helpful']:,} |",
        "",
        "| Hidden layer | Agreement | Balanced accuracy | Macro-F1 | Kappa |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer in LAYERS:
        value = overall[layer]
        lines.append(
            f"| {layer} | {value['accuracy']:.4f} | {value['balanced_accuracy']:.4f} | "
            f"{value['macro_f1']:.4f} | {value['cohen_kappa']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Agreement with observed correctness transitions (changed-answer pairs only)",
            "",
            "| Method | Accuracy | Balanced accuracy | Macro-F1 | Helpful precision | Helpful recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("hidden", "terra"):
        value = transition[method]
        lines.append(
            f"| {method} | {value['accuracy']:.4f} | {value['balanced_accuracy']:.4f} | "
            f"{value['macro_f1']:.4f} | {value['helpful_precision']:.4f} | "
            f"{value['helpful_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Terra labels semantic evidence for the gold answer; hidden labels the behavioral direction of the "
            "specific target LLM. Their disagreement is therefore a construct difference, not automatically an "
            "annotation error.",
        ]
    )
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())

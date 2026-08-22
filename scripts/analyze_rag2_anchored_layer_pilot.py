#!/usr/bin/env python3
"""Rank decoder blocks from the three-anchor hidden-state pilot.

The primary criterion is worst-subgroup ROC-AUC for predicting whether a
document increased the exact gold-choice log probability.  Subgroups are the
cross-product of dataset and no-RAG correctness.  This deliberately prevents
the majority no-RAG-correct group from selecting a shortcut layer.  Macro AUC,
Spearman correlation, transition AUC, and zero-gradient rates are reported as
diagnostics rather than folded into an opaque weighted score.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from safetensors.torch import load_file


ANALYSIS_VERSION = "rag2_three_anchor_layer_selection_analysis_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-score", choices=["utility_projection", "delta_c_cosine"], default="utility_projection")
    parser.add_argument("--min-subgroup-examples", type=int, default=100)
    parser.add_argument("--min-nonzero-c-rate", type=float, default=0.95)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from error
    return rows


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(np.int8)
    scores = scores[mask].astype(np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan")
    left = left[mask].astype(np.float64)
    right = right[mask].astype(np.float64)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan")
    return correlation(average_ranks(left[mask]), average_ranks(right[mask]))


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def finite_min(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else float("nan")


def load_arrays(feature_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest_path = feature_dir / "feature_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata: list[dict[str, Any]] = []
    tensors: dict[str, list[np.ndarray]] = {
        "utility_projection": [],
        "delta_c_cosine": [],
        "delta_h_norm": [],
        "c_norm": [],
        "gold_choice_logprob_delta": [],
    }
    shard_roots = sorted((feature_dir / "feature_shards").glob("shard_*"))
    if len(shard_roots) != int(manifest.get("shards", -1)):
        raise RuntimeError("Feature shard count does not match manifest")
    for root in shard_roots:
        rows = read_jsonl(root / "pairs.jsonl")
        values = load_file(str(root / "pair_features.safetensors"), device="cpu")
        if len(rows) != int(values["utility_projection"].shape[0]):
            raise RuntimeError(f"Metadata/tensor row mismatch: {root}")
        metadata.extend(rows)
        for name in tensors:
            tensors[name].append(values[name].float().numpy())
    merged = {name: np.concatenate(parts, axis=0) for name, parts in tensors.items()}
    if len(metadata) != int(manifest.get("pairs", -1)):
        raise RuntimeError(f"Pair count mismatch: loaded={len(metadata)} manifest={manifest.get('pairs')}")
    return manifest, metadata, merged


def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    manifest, metadata, tensors = load_arrays(args.feature_dir)
    layer_names = list(manifest["layer_order"])
    anchor_names = list(manifest["anchor_order"])
    datasets = np.asarray([str(row["dataset"]) for row in metadata], dtype=object)
    no_doc_correct = np.asarray([bool(row["no_document_correct"]) for row in metadata])
    transitions = np.asarray([str(row["answer_transition"]) for row in metadata], dtype=object)
    valid = np.asarray([bool(row.get("valid_for_layer_analysis")) for row in metadata])
    generated_hf_match = np.asarray(
        [bool(row.get("generated_hf_answer_match")) for row in metadata]
    )
    gold_delta = tensors["gold_choice_logprob_delta"].reshape(-1)
    target = gold_delta > 0
    primary = tensors[args.primary_score]
    c_norm = tensors["c_norm"]
    delta_norm = tensors["delta_h_norm"]
    if primary.shape[1:] != (len(layer_names), len(anchor_names)):
        raise RuntimeError(
            f"Unexpected score shape {primary.shape}; layers={len(layer_names)} anchors={len(anchor_names)}"
        )

    subgroup_masks: dict[str, np.ndarray] = {}
    for dataset in sorted(set(datasets.tolist())):
        for correctness in (False, True):
            name = f"{dataset}__no_rag_{'correct' if correctness else 'wrong'}"
            subgroup_masks[name] = valid & (datasets == dataset) & (no_doc_correct == correctness)

    metrics: list[dict[str, Any]] = []
    changed = valid & np.isin(transitions, ["W->C", "C->W"])
    transition_target = transitions == "W->C"
    for layer_index, layer_name in enumerate(layer_names):
        for anchor_index, anchor_name in enumerate(anchor_names):
            scores = primary[:, layer_index, anchor_index]
            subgroup_aucs: dict[str, float] = {}
            subgroup_counts: dict[str, int] = {}
            for subgroup, mask in subgroup_masks.items():
                count = int(mask.sum())
                subgroup_counts[subgroup] = count
                subgroup_aucs[subgroup] = (
                    roc_auc(target[mask], scores[mask])
                    if count >= args.min_subgroup_examples
                    else float("nan")
                )
            correct_auc = finite_mean(
                value for key, value in subgroup_aucs.items() if key.endswith("no_rag_correct")
            )
            wrong_auc = finite_mean(
                value for key, value in subgroup_aucs.items() if key.endswith("no_rag_wrong")
            )
            nonzero_c = c_norm[:, layer_index, anchor_index] > 1e-10
            nonzero_delta = delta_norm[:, layer_index, anchor_index] > 1e-10
            row: dict[str, Any] = {
                "layer_index": layer_index,
                "layer": layer_name,
                "anchor_index": anchor_index,
                "anchor": anchor_name,
                "valid_pairs": int(valid.sum()),
                "positive_gold_delta_rate": float(target[valid].mean()),
                "auc_all": roc_auc(target[valid], scores[valid]),
                "auc_macro_subgroup": finite_mean(subgroup_aucs.values()),
                "auc_worst_subgroup": finite_min(subgroup_aucs.values()),
                "auc_no_rag_correct": correct_auc,
                "auc_no_rag_wrong": wrong_auc,
                "auc_correct_wrong_gap": abs(correct_auc - wrong_auc)
                if math.isfinite(correct_auc) and math.isfinite(wrong_auc)
                else float("nan"),
                "transition_auc_wc_vs_cw": roc_auc(
                    transition_target[changed], scores[changed]
                ),
                "transition_examples": int(changed.sum()),
                "pearson_gold_logprob_delta": correlation(scores[valid], gold_delta[valid]),
                "spearman_gold_logprob_delta": spearman(scores[valid], gold_delta[valid]),
                "sign_accuracy": float(((scores[valid] > 0) == target[valid]).mean()),
                "c_nonzero_rate": float(nonzero_c[valid].mean()),
                "delta_nonzero_rate": float(nonzero_delta[valid].mean()),
                "mean_abs_score": float(np.mean(np.abs(scores[valid]))),
                "generated_hf_answer_match_rate": float(generated_hf_match[valid].mean()),
            }
            for subgroup, value in subgroup_aucs.items():
                row[f"auc__{subgroup}"] = value
                row[f"n__{subgroup}"] = subgroup_counts[subgroup]
            metrics.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics[0])
    csv_path = args.output_dir / "layer_anchor_metrics.csv"
    temporary = csv_path.with_name(csv_path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)

    eligible = [
        row
        for row in metrics
        if row["c_nonzero_rate"] >= args.min_nonzero_c_rate
        and math.isfinite(row["auc_worst_subgroup"])
    ]

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["auc_worst_subgroup"]),
            float(row["auc_macro_subgroup"]),
            float(row["transition_auc_wc_vs_cw"])
            if math.isfinite(row["transition_auc_wc_vs_cw"])
            else -1.0,
            abs(float(row["spearman_gold_logprob_delta"])),
        )

    by_anchor: dict[str, list[dict[str, Any]]] = {}
    for anchor in anchor_names:
        by_anchor[anchor] = sorted(
            [row for row in eligible if row["anchor"] == anchor],
            key=rank_key,
            reverse=True,
        )
    recommendation_by_anchor = {
        anchor: rows[0] if rows else None for anchor, rows in by_anchor.items()
    }

    shared_candidates: list[dict[str, Any]] = []
    for layer_name in layer_names:
        rows = [row for row in eligible if row["layer"] == layer_name]
        if len(rows) != len(anchor_names):
            continue
        shared_candidates.append(
            {
                "layer": layer_name,
                "worst_anchor_subgroup_auc": min(row["auc_worst_subgroup"] for row in rows),
                "macro_anchor_subgroup_auc": float(
                    np.mean([row["auc_macro_subgroup"] for row in rows])
                ),
                "max_correct_wrong_gap": max(row["auc_correct_wrong_gap"] for row in rows),
            }
        )
    shared_candidates.sort(
        key=lambda row: (
            row["worst_anchor_subgroup_auc"],
            row["macro_anchor_subgroup_auc"],
            -row["max_correct_wrong_gap"],
        ),
        reverse=True,
    )
    recommendations = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": utc_now(),
        "selection_rule": (
            "For each anchor, maximize worst dataset x no-RAG-correctness subgroup AUC; "
            "break ties with macro subgroup AUC, W->C-vs-C->W AUC, and absolute Spearman correlation."
        ),
        "primary_score": args.primary_score,
        "recommended_primary_post_rationale": recommendation_by_anchor.get("post_rationale"),
        "recommended_by_anchor": recommendation_by_anchor,
        "recommended_shared_layer": shared_candidates[0] if shared_candidates else None,
        "shared_layer_ranking": shared_candidates[:10],
        "valid_pairs": int(valid.sum()),
        "invalid_pairs": int((~valid).sum()),
        "generated_hf_answer_match_rate": float(generated_hf_match[valid].mean()),
    }
    atomic_write_json(args.output_dir / "layer_recommendations.json", recommendations)

    lines = [
        "# RAG2 Three-Anchor Layer Pilot",
        "",
        f"- Valid pairs: **{int(valid.sum()):,}** / {len(valid):,}",
        f"- Primary score: `{args.primary_score}`",
        f"- vLLM/HF constrained-answer agreement: **{float(generated_hf_match[valid].mean()):.2%}**",
        "- Selection rule: maximize the worst MedMCQA/MedQA × no-RAG correct/wrong subgroup AUC.",
        "",
    ]
    primary_row = recommendation_by_anchor.get("post_rationale")
    if primary_row:
        lines.extend(
            [
                "## Primary recommendation",
                "",
                f"Use **{primary_row['layer']}** at `post_rationale` for the main hidden-utility label.",
                "",
                f"- Worst subgroup AUC: {fmt(primary_row['auc_worst_subgroup'])}",
                f"- Macro subgroup AUC: {fmt(primary_row['auc_macro_subgroup'])}",
                f"- No-RAG correct/wrong AUC gap: {fmt(primary_row['auc_correct_wrong_gap'])}",
                f"- Spearman with exact gold log-probability delta: {fmt(primary_row['spearman_gold_logprob_delta'])}",
                f"- W->C vs C->W AUC: {fmt(primary_row['transition_auc_wc_vs_cw'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top blocks by anchor",
            "",
            "| Anchor | Rank | Block | Worst subgroup AUC | Macro subgroup AUC | Correct/wrong gap | Spearman | Transition AUC | c nonzero |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for anchor in anchor_names:
        for rank, row in enumerate(by_anchor[anchor][:5], 1):
            lines.append(
                f"| {anchor} | {rank} | {row['layer']} | {fmt(row['auc_worst_subgroup'])} | "
                f"{fmt(row['auc_macro_subgroup'])} | {fmt(row['auc_correct_wrong_gap'])} | "
                f"{fmt(row['spearman_gold_logprob_delta'])} | {fmt(row['transition_auc_wc_vs_cw'])} | "
                f"{row['c_nonzero_rate']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This pilot selects representation layers; it does not estimate final filtering accuracy.",
            "- Existing reranked documents are reused only for this pilot. Full data must rerun retrieval with the frozen rationale query.",
            "- Inspect subgroup metrics before accepting the automatic recommendation, especially the no-RAG-wrong strata.",
            "- A block with zero answer-direction gradients at an anchor is excluded even if another metric appears high.",
            "",
        ]
    )
    atomic_write_text(args.output_dir / "layer_selection_summary.md", "\n".join(lines))
    logging.info("Layer analysis complete: %s", args.output_dir)
    if primary_row:
        logging.info(
            "Primary post-rationale recommendation: %s worst_auc=%.4f macro_auc=%.4f",
            primary_row["layer"],
            primary_row["auc_worst_subgroup"],
            primary_row["auc_macro_subgroup"],
        )


if __name__ == "__main__":
    main()

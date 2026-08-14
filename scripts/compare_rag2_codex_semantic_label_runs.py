from __future__ import annotations

"""Compare paired Codex semantic-labelling pilot runs.

Every candidate run must use the same candidate JSONL, Top-k, question order,
and batch size as the reference run.  This tool deliberately treats the
reference labels as a *stability baseline*, not ground truth: it measures how
closely a cheaper model/effort reproduces the selected reference configuration.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


LABELS = ("helpful", "harmful", "neutral", "uncertain")
COMPLETE_RE = re.compile(r"completed\s+(\d+)\s+pair\(s\)\s+in\s+([0-9.]+)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare same-pair Codex semantic label runs against a reference configuration."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-run",
        action="append",
        required=True,
        metavar="NAME=ROOT",
        help="Repeat for each candidate configuration, e.g. terra_medium=/path/to/run.",
    )
    parser.add_argument("--dataset", default="medmcqa")
    parser.add_argument(
        "--expected-batches",
        type=int,
        required=True,
        help="Require batches [0, expected_batches) in every run; prevents partial comparisons.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_labels(root: Path, dataset: str, expected_batches: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for batch_index in range(expected_batches):
        path = root / "batches" / dataset / f"batch_{batch_index:06d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required completed pilot batch: {path}")
        batch = read_json(path)
        labels = batch.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Invalid labels in {path}")
        for row in labels:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid label row in {path}")
            pair_id = str(row.get("pair_id") or "")
            label = str(row.get("semantic_label") or "").lower()
            if not pair_id or label not in LABELS:
                raise ValueError(f"Invalid pair_id/semantic_label in {path}: {row}")
            if pair_id in result:
                raise ValueError(f"Duplicate pair_id in {root}: {pair_id}")
            result[pair_id] = row
    return result


def parse_candidate_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--candidate-run must be NAME=ROOT, got: {value}")
    name, root = value.split("=", 1)
    name = name.strip()
    if not name or not root.strip():
        raise ValueError(f"--candidate-run must be NAME=ROOT, got: {value}")
    return name, Path(root).expanduser().resolve()


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def class_metrics(reference: list[str], candidate: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = sum(ref == label and pred == label for ref, pred in zip(reference, candidate))
        false_positive = sum(ref != label and pred == label for ref, pred in zip(reference, candidate))
        false_negative = sum(ref == label and pred != label for ref, pred in zip(reference, candidate))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        value = f1(precision, recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": value,
            "support_reference": sum(ref == label for ref in reference),
            "count_candidate": sum(pred == label for pred in candidate),
        }
        f1_values.append(value)
    metrics["per_class"] = per_class
    metrics["macro_f1"] = mean(f1_values)
    metrics["balanced_accuracy"] = mean(per_class[label]["recall"] for label in LABELS)
    return metrics


def cohen_kappa(reference: list[str], candidate: list[str]) -> float:
    count = len(reference)
    observed = sum(ref == pred for ref, pred in zip(reference, candidate)) / count
    reference_distribution = Counter(reference)
    candidate_distribution = Counter(candidate)
    expected = sum(
        (reference_distribution[label] / count) * (candidate_distribution[label] / count)
        for label in LABELS
    )
    return 0.0 if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def binary_helpful_metrics(reference: list[str], candidate: list[str]) -> dict[str, float]:
    true_positive = sum(ref == "helpful" and pred == "helpful" for ref, pred in zip(reference, candidate))
    false_positive = sum(ref != "helpful" and pred == "helpful" for ref, pred in zip(reference, candidate))
    false_negative = sum(ref == "helpful" and pred != "helpful" for ref, pred in zip(reference, candidate))
    true_negative = sum(ref != "helpful" and pred != "helpful" for ref, pred in zip(reference, candidate))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def timing_summary(root: Path) -> dict[str, Any]:
    durations: list[float] = []
    pairs = 0
    for path in sorted((root / "logs").glob("worker_*.log")):
        for matched in COMPLETE_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            pairs += int(matched.group(1))
            durations.append(float(matched.group(2)))
    if not durations:
        return {"completed_pairs_from_logs": 0, "active_seconds": 0.0, "pairs_per_active_second": None}
    active_seconds = sum(durations)
    return {
        "completed_pairs_from_logs": pairs,
        "completed_batches_from_logs": len(durations),
        "active_seconds": active_seconds,
        "mean_batch_seconds": mean(durations),
        "median_batch_seconds": median(durations),
        "pairs_per_active_second": pairs / active_seconds if active_seconds else None,
    }


def summarize(reference_rows: dict[str, dict[str, Any]], candidate_rows: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_ids = set(reference_rows)
    candidate_ids = set(candidate_rows)
    if reference_ids != candidate_ids:
        missing = sorted(reference_ids - candidate_ids)[:10]
        extra = sorted(candidate_ids - reference_ids)[:10]
        raise ValueError(f"Pair sets differ. missing_candidate={missing} extra_candidate={extra}")

    pair_ids = sorted(reference_ids)
    reference = [str(reference_rows[pair_id]["semantic_label"]).lower() for pair_id in pair_ids]
    candidate = [str(candidate_rows[pair_id]["semantic_label"]).lower() for pair_id in pair_ids]
    agreement = sum(ref == pred for ref, pred in zip(reference, candidate)) / len(pair_ids)
    disagreements: list[dict[str, Any]] = []
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair_id, ref, pred in zip(pair_ids, reference, candidate):
        row = reference_rows[pair_id]
        by_source[str(row.get("source") or "unknown")].append((ref, pred))
        if ref != pred:
            disagreements.append(
                {
                    "pair_id": pair_id,
                    "dataset": row.get("dataset"),
                    "sample_id": row.get("sample_id"),
                    "source": row.get("source"),
                    "doc_rank": row.get("doc_rank"),
                    "reference_label": ref,
                    "candidate_label": pred,
                    "reference_confidence": row.get("confidence"),
                    "candidate_confidence": candidate_rows[pair_id].get("confidence"),
                    "reference_short_reason": row.get("short_reason"),
                    "candidate_short_reason": candidate_rows[pair_id].get("short_reason"),
                }
            )
    source_metrics = {}
    for source, values in sorted(by_source.items()):
        refs = [item[0] for item in values]
        preds = [item[1] for item in values]
        source_metrics[source] = {
            "pairs": len(values),
            "exact_agreement": sum(ref == pred for ref, pred in values) / len(values),
            "helpful_binary_f1": binary_helpful_metrics(refs, preds)["f1"],
        }
    return (
        {
            "pairs": len(pair_ids),
            "four_class_exact_agreement": agreement,
            "cohen_kappa": cohen_kappa(reference, candidate),
            "four_class": class_metrics(reference, candidate),
            "helpful_vs_rest": binary_helpful_metrics(reference, candidate),
            "reference_label_distribution": dict(Counter(reference)),
            "candidate_label_distribution": dict(Counter(candidate)),
            "by_source": source_metrics,
            "disagreement_pairs": len(disagreements),
        },
        disagreements,
    )


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Codex semantic-label model-selection pilot",
        "",
        f"Reference: `{summary['reference_root']}`",
        f"Dataset: `{summary['dataset']}`; paired batches: {summary['expected_batches']}",
        "",
        "| Configuration | Pairs | 4-class agreement | Cohen's kappa | Macro-F1 vs ref. | Helpful F1 | Helpful recall | Active pair/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["candidates"].items():
        metrics = result["agreement"]
        timing = result["timing"]
        active_rate = timing.get("pairs_per_active_second")
        lines.append(
            "| {name} | {pairs} | {agreement:.2%} | {kappa:.3f} | {macro:.2%} | {helpful_f1:.2%} | {helpful_recall:.2%} | {rate} |".format(
                name=name,
                pairs=metrics["pairs"],
                agreement=metrics["four_class_exact_agreement"],
                kappa=metrics["cohen_kappa"],
                macro=metrics["four_class"]["macro_f1"],
                helpful_f1=metrics["helpful_vs_rest"]["f1"],
                helpful_recall=metrics["helpful_vs_rest"]["recall"],
                rate="n/a" if active_rate is None else f"{active_rate:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: agreement is stability relative to the xhigh reference, not semantic-label accuracy. "
            "Review the generated disagreement JSONL before choosing a configuration.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.expected_batches <= 0:
        raise ValueError("--expected-batches must be positive")
    dataset = args.dataset.strip().lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_root = args.reference_root.expanduser().resolve()
    reference = load_labels(reference_root, dataset, args.expected_batches)
    seen_names: set[str] = set()
    candidates: dict[str, Any] = {}
    for raw in args.candidate_run:
        name, root = parse_candidate_run(raw)
        if name in seen_names:
            raise ValueError(f"Duplicate candidate name: {name}")
        seen_names.add(name)
        candidate = load_labels(root, dataset, args.expected_batches)
        agreement, disagreements = summarize(reference, candidate)
        candidates[name] = {"root": str(root), "agreement": agreement, "timing": timing_summary(root)}
        (args.output_dir / f"{name}_disagreements.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in disagreements), encoding="utf-8"
        )
    output = {
        "reference_root": str(reference_root),
        "dataset": dataset,
        "expected_batches": args.expected_batches,
        "candidates": candidates,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown(output), encoding="utf-8")
    print(markdown(output), end="")


if __name__ == "__main__":
    main()

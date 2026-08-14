#!/usr/bin/env python3
"""Materialize Helpful/Not Helpful(/Neutral) labels from hidden projections."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


LABEL_VERSION = "rag2_preanswer_hidden_projection_label_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--primary-layer", choices=["layer_16", "layer_24", "layer_28", "final"], default="layer_28"
    )
    parser.add_argument("--neutral-threshold", type=float, default=0.0)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def classify(score: float, threshold: float) -> str:
    if score > threshold:
        return "Helpful"
    if score < -threshold:
        return "Not Helpful"
    return "Neutral"


def run(args: argparse.Namespace) -> None:
    if args.neutral_threshold < 0:
        raise ValueError("--neutral-threshold cannot be negative")
    input_path = args.input_dir / "pairs.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "hidden_labels.jsonl"
    temporary = output_path.with_name(output_path.name + ".partial")
    counts: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    total_bytes = input_path.stat().st_size
    with input_path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination, tqdm(
        total=total_bytes,
        desc="materialize-hidden-labels",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for line in source:
            progress.update(len(line.encode("utf-8")))
            row = json.loads(line)
            score = float(row["utility_projection_by_layer"][args.primary_layer])
            label = classify(score, args.neutral_threshold)
            document = row["document"]
            output = {
                "label_version": LABEL_VERSION,
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "pair_id": document["pair_id"],
                "doc_rank": document["rerank_rank"],
                "source": document["source"],
                "doc_stable_id": document["stable_id"],
                "primary_layer": args.primary_layer,
                "projection_score": score,
                "neutral_threshold": args.neutral_threshold,
                "hidden_label": label,
                "use_for_binary_training": label in {"Helpful", "Not Helpful"},
                "gold_answer": row["gold_answer"],
                "no_document_answer": row["no_document_answer"],
                "with_document_answer": row["with_document_answer"],
                "answer_transition": row["answer_transition"],
                "gold_choice_logprob_delta": row["gold_choice_logprob_delta"],
                "utility_projection_by_layer": row["utility_projection_by_layer"],
            }
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            counts[label] += 1
            by_dataset[row["dataset"]][label] += 1
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, output_path)
    summary = {
        "label_version": LABEL_VERSION,
        "input_dir": str(args.input_dir.resolve()),
        "output_path": str(output_path.resolve()),
        "primary_layer": args.primary_layer,
        "neutral_threshold": args.neutral_threshold,
        "label_rule": {
            "Helpful": f"score > {args.neutral_threshold}",
            "Not Helpful": f"score < {-args.neutral_threshold}",
            "Neutral": f"abs(score) <= {args.neutral_threshold}",
        },
        "rows": sum(counts.values()),
        "labels": dict(counts),
        "by_dataset": {dataset: dict(values) for dataset, values in by_dataset.items()},
    }
    atomic_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Build question-level train/val/test splits for hidden-utility filter ablations.

The binary target is materialized elsewhere from the layer-28 projection

    score = (hD - h0) dot c_unit

at a fixed threshold.  This builder deliberately does *not* expose ``c`` or
the scalar projection to the model input.  It writes the released RAG2 text
prompt plus references to the already stored ``h0`` and ``hD`` tensors.  The
trainer reconstructs ``delta_h = hD - h0`` lazily and can therefore compare:

* text_only: Question + document text
* hidden_only: h0 + delta_h
* text_hidden: both modalities

All three modes consume the exact same rows and question-level split.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import build_official_filter_input, format_options


MATERIALIZATION_VERSION = "rag2_hidden_utility_filter_inputs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build common text/hidden/hybrid filter splits from pre-answer hidden states."
    )
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--hidden-feature-dir", type=Path, required=True)
    parser.add_argument("--hidden-label-dir", type=Path, required=True)
    parser.add_argument("--reference-split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-primary-layer", default="layer_28")
    parser.add_argument("--expected-threshold", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_split_map(root: Path, dataset: str) -> tuple[dict[str, str], dict[str, int]]:
    split_map: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = root / dataset / "sample_ids" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Duplicate sample IDs in {path}")
        overlap = set(values).intersection(split_map)
        if overlap:
            raise RuntimeError(f"Question split overlap in {path}: {next(iter(overlap))}")
        split_map.update({sample_id: split for sample_id in values})
        counts[split] = len(values)
    return split_map, counts


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    feature_manifest_path = args.hidden_feature_dir / "run_manifest.json"
    label_summary_path = args.hidden_label_dir / "summary.json"
    if not feature_manifest_path.is_file():
        raise FileNotFoundError(feature_manifest_path)
    if not label_summary_path.is_file():
        raise FileNotFoundError(label_summary_path)
    feature = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    labels = json.loads(label_summary_path.read_text(encoding="utf-8"))
    if feature.get("dataset") != args.dataset:
        raise RuntimeError("Hidden feature dataset mismatch")
    requested_layers = [str(value) for value in (feature.get("layers") or [])]
    normalized_layers = [
        value if value.startswith("layer_") else f"layer_{value}"
        for value in requested_layers
    ]
    if normalized_layers != [args.expected_primary_layer]:
        raise RuntimeError(
            f"Expected only {args.expected_primary_layer}, found {feature.get('layers')}"
        )
    if int(feature.get("docs_per_question") or 0) != 8:
        raise RuntimeError("Expected the completed Top-8 hidden feature contract")
    if labels.get("primary_layer") != args.expected_primary_layer:
        raise RuntimeError("Hidden label primary layer mismatch")
    if float(labels.get("neutral_threshold")) != float(args.expected_threshold):
        raise RuntimeError("Hidden label threshold mismatch")
    if int(labels.get("rows") or 0) != int(feature.get("total_pairs") or 0):
        raise RuntimeError("Hidden label/feature pair totals differ")
    return {"feature_manifest": feature, "label_summary": labels}


def output_row(pair: dict[str, Any], label: dict[str, Any], split: str) -> dict[str, Any]:
    document = pair["document"]
    hidden_label = str(label["hidden_label"])
    if hidden_label not in {"Helpful", "Not Helpful"}:
        raise RuntimeError(
            "This binary ablation requires tau=0 labels without Neutral rows; "
            f"got {hidden_label!r} for {pair['pair_id']}"
        )
    target = "helpful" if hidden_label == "Helpful" else "not helpful"
    return {
        "materialization_version": MATERIALIZATION_VERSION,
        "id": pair["pair_id"],
        "pair_id": pair["pair_id"],
        "dataset": pair["dataset"],
        "sample_id": pair["sample_id"],
        "split": split,
        "source": document["source"],
        "doc_stable_id": document["stable_id"],
        "doc_rank": int(document["rerank_rank"]),
        "input": build_official_filter_input(
            question=pair["question"],
            options=format_options(pair["options"]),
            evidence=document["text"],
        ),
        "target": target,
        "label": hidden_label,
        "label_origin": "layer28_hidden_projection_sign_tau0",
        # Audit-only values. The collator never exposes these fields to the model.
        "hidden_projection_score_audit_only": float(label["projection_score"]),
        "answer_transition_audit_only": str(label["answer_transition"]),
        # Lazy tensor references into the immutable source feature shards.
        "feature_shard_index": int(pair["shard_index"]),
        "feature_pair_row": int(pair["shard_pair_index"]),
        "feature_question_row": int(pair["shard_question_index"]),
    }


def run(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    contract = validate_contract(args)
    split_map, question_counts = read_split_map(args.reference_split_root, args.dataset)
    expected_questions = int(contract["feature_manifest"]["total_questions"])
    if len(split_map) != expected_questions:
        raise RuntimeError(
            f"Question split has {len(split_map)} IDs, hidden run has {expected_questions} questions"
        )

    pair_path = args.hidden_feature_dir / "pairs.jsonl"
    label_path = args.hidden_label_dir / "hidden_labels.jsonl"
    if not pair_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing pair/label JSONL: {pair_path}, {label_path}")

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {split: output_dir / f"{split}.jsonl.partial" for split in ("train", "val", "test")}
    final_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
    handles: dict[str, TextIO] = {
        split: path.open("w", encoding="utf-8", buffering=16 * 1024 * 1024)
        for split, path in temporary_paths.items()
    }
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    seen_questions: set[str] = set()
    rows = 0
    try:
        pair_iter = iter_jsonl(pair_path)
        label_iter = iter_jsonl(label_path)
        with tqdm(
            total=int(contract["feature_manifest"]["total_pairs"]),
            desc=f"BuildHiddenFilter:{args.dataset}",
            unit="pair",
            dynamic_ncols=True,
        ) as progress:
            while True:
                try:
                    pair = next(pair_iter)
                except StopIteration:
                    try:
                        next(label_iter)
                    except StopIteration:
                        break
                    raise RuntimeError("Hidden labels contain more rows than pair metadata")
                try:
                    label = next(label_iter)
                except StopIteration as error:
                    raise RuntimeError("Pair metadata contains more rows than hidden labels") from error
                if pair["pair_id"] != label["pair_id"]:
                    raise RuntimeError(
                        f"Pair/label order mismatch: {pair['pair_id']} != {label['pair_id']}"
                    )
                sample_id = str(pair["sample_id"])
                split = split_map.get(sample_id)
                if split is None:
                    raise RuntimeError(f"Question missing from reference split: {sample_id}")
                row = output_row(pair, label, split)
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                seen_questions.add(sample_id)
                counters[split]["rows"] += 1
                counters[split][f"target:{row['target']}"] += 1
                counters[split][f"source:{row['source']}"] += 1
                counters[split][f"rank:{row['doc_rank']}"] += 1
                rows += 1
                progress.update(1)
        if rows != int(contract["feature_manifest"]["total_pairs"]):
            raise RuntimeError(f"Wrote {rows} pairs, expected {contract['feature_manifest']['total_pairs']}")
        if seen_questions != set(split_map):
            raise RuntimeError(f"Observed {len(seen_questions)}/{len(split_map)} split questions")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for split in ("train", "val", "test"):
            os.replace(temporary_paths[split], final_paths[split])
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        raise

    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        counter = counters[split]
        summary[split] = {
            "rows": counter["rows"],
            "questions": question_counts[split],
            "targets": {
                "helpful": counter["target:helpful"],
                "not helpful": counter["target:not helpful"],
            },
            "sources": {
                key.removeprefix("source:"): value
                for key, value in sorted(counter.items())
                if key.startswith("source:")
            },
            "ranks": {
                key.removeprefix("rank:"): value
                for key, value in sorted(counter.items())
                if key.startswith("rank:")
            },
        }
    manifest = {
        "materialization_version": MATERIALIZATION_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "hidden_feature_dir": str(args.hidden_feature_dir.resolve()),
        "hidden_label_dir": str(args.hidden_label_dir.resolve()),
        "reference_split_root": str(args.reference_split_root.resolve()),
        "primary_layer": args.expected_primary_layer,
        "threshold": args.expected_threshold,
        "label_rule": {
            "helpful": f"projection_score > {args.expected_threshold}",
            "not_helpful": f"projection_score <= {args.expected_threshold}",
        },
        "model_input_contract": {
            "text_only": "official RAG2 Question + Evidence text",
            "hidden_only": "h0 and delta_h=hD-h0 only",
            "text_hidden": "official RAG2 text plus h0 and delta_h",
            "forbidden_as_model_inputs": [
                "gold-derived c",
                "projection score",
                "gold answer",
                "answer transition",
            ],
        },
        "summary": summary,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    logging.info("Hidden utility filter splits complete: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())

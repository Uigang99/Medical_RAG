#!/usr/bin/env python3
"""Build question-disjoint pointer splits for text-to-utility regression.

The output stores only trace pointers and the single continuous supervision
target.  Question/options/document text is read lazily from the existing
anchored trace shards during training, avoiding another multi-gigabyte text
copy.  Gold answers, logits, margins, transitions, and hidden states are never
declared as model inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


MATERIALIZATION_VERSION = "rag2_margin_regression_pointer_splits_v1"
SCORE_VERSION = "rag2_anchored_gold_margin_scores_v1"
SPLITS = ("train", "val", "test")


try:
    import msgspec

    _DECODER = msgspec.json.Decoder()

    def decode_json(line: bytes) -> dict[str, Any]:
        return _DECODER.decode(line)

except ImportError:

    def decode_json(line: bytes) -> dict[str, Any]:
        return json.loads(line)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-root",
        type=Path,
        default=base / "gold_margin_utility_source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=base / "document_traces_source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--reference-split-root",
        type=Path,
        default=base / "filter_training_inputs_rag2_paper_reproduction_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base / "gold_margin_regression_v1/prepared",
    )
    parser.add_argument("--datasets", nargs="+", choices=("medmcqa", "medqa"), default=["medmcqa", "medqa"])
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--target-field",
        choices=("boundary_probability_delta",),
        default="boundary_probability_delta",
    )
    parser.add_argument("--exclude-quality-flags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield decode_json(line)
            except Exception as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_assignments(root: Path, datasets: list[str]) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    for dataset in datasets:
        values: dict[str, str] = {}
        for split in SPLITS:
            path = root / dataset / "sample_ids" / f"{split}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            for sample_id in path.read_text(encoding="utf-8").splitlines():
                if not sample_id:
                    continue
                if sample_id in values:
                    raise RuntimeError(f"Question split leakage: {sample_id}")
                values[sample_id] = split
        assignments[dataset] = values
    return assignments


def score_paths(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    for dataset in args.datasets:
        root = args.score_root / "score_shards" / dataset / args.source_split
        for path in sorted(root.glob("shard_*/rows.jsonl")):
            result.append((dataset, path.parent.name, path))
    if not result:
        raise FileNotFoundError(f"No score rows under {args.score_root}")
    return result


def already_complete(args: argparse.Namespace) -> bool:
    path = args.output_root / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("materialization_version") == MATERIALIZATION_VERSION
        and manifest.get("target_field") == args.target_field
        and manifest.get("datasets") == args.datasets
        and bool(manifest.get("exclude_quality_flags")) == bool(args.exclude_quality_flags)
        and all((args.output_root / dataset / f"{split}.jsonl").is_file() for dataset in args.datasets for split in SPLITS)
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    score_manifest = json.loads((args.score_root / "manifest.json").read_text(encoding="utf-8"))
    if score_manifest.get("run_version") != SCORE_VERSION:
        raise RuntimeError("Gold-margin score contract mismatch")
    if not math.isclose(float(score_manifest.get("temperature", float("nan"))), 1.0, abs_tol=1e-12):
        raise RuntimeError("This baseline requires the materialized T=1 utility target")
    if already_complete(args) and not args.overwrite:
        logging.info("Prepared regression dataset already complete: %s", args.output_root)
        return
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite after auditing it: {args.output_root}")

    paths = score_paths(args)
    total_pairs = sum(
        int(json.loads((path.parent / "COMPLETE.json").read_text(encoding="utf-8"))["pair_count"])
        for _, _, path in paths
    )
    if args.dry_run:
        logging.info("Dry-run preparation plan: shards=%d pairs=%d", len(paths), total_pairs)
        return

    assignments = load_assignments(args.reference_split_root, args.datasets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = PipelineProgress(
        overall_total=total_pairs,
        desc="MarginRegressionPrepare",
        enabled=args.show_progress,
    )
    progress.set_stage("1/1 materialize question-disjoint utility pointers", total=total_pairs)
    counters = {dataset: {split: Counter() for split in SPLITS} for dataset in args.datasets}
    questions = {dataset: {split: set() for split in SPLITS} for dataset in args.datasets}
    temporary_paths: dict[tuple[str, str], Path] = {}
    try:
        with ExitStack() as stack:
            handles: dict[str, dict[str, Any]] = {}
            for dataset in args.datasets:
                dataset_root = args.output_root / dataset
                dataset_root.mkdir(parents=True, exist_ok=True)
                handles[dataset] = {}
                for split in SPLITS:
                    final = dataset_root / f"{split}.jsonl"
                    temporary = final.with_name(final.name + ".partial")
                    temporary_paths[(dataset, split)] = temporary
                    handles[dataset][split] = stack.enter_context(
                        temporary.open("w", encoding="utf-8", buffering=16 * 1024 * 1024)
                    )
            for dataset, shard_name, path in paths:
                trace_path = args.trace_root / "trace_shards" / dataset / args.source_split / shard_name / "pairs.jsonl"
                if not trace_path.is_file():
                    raise FileNotFoundError(trace_path)
                for local, row in enumerate(iter_jsonl(path)):
                    progress.update(1)
                    if row.get("run_version") != SCORE_VERSION or int(row.get("tensor_row", -1)) != local:
                        raise RuntimeError(f"Score-row contract mismatch: {path}:{local}")
                    sample_id = str(row["sample_id"])
                    split = assignments[dataset].get(sample_id)
                    if split is None:
                        raise RuntimeError(f"No reference split for {sample_id}")
                    quality_flags = list(row.get("quality_flags") or [])
                    if args.exclude_quality_flags and quality_flags:
                        counters[dataset][split]["excluded_quality"] += 1
                        continue
                    target = float(row[args.target_field])
                    if not math.isfinite(target) or target < -1.000001 or target > 1.000001:
                        raise ValueError(f"Invalid utility target for {row['pair_id']}: {target}")
                    pointer = {
                        "materialization_version": MATERIALIZATION_VERSION,
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "pair_id": str(row["pair_id"]),
                        "doc_rank": int(row["doc_rank"]),
                        "source": str(row["document_source"]),
                        "trace_shard": shard_name,
                        "trace_pair_row": local,
                        "utility_target": target,
                        "no_rag_correct_audit_only": bool(row["no_document_correct"]),
                        "answer_transition_audit_only": str(row["answer_transition"]),
                    }
                    handles[dataset][split].write(json.dumps(pointer, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counters[dataset][split]["rows"] += 1
                    counters[dataset][split]["positive"] += int(target > 0)
                    counters[dataset][split]["zero"] += int(target == 0)
                    counters[dataset][split]["negative"] += int(target < 0)
                    questions[dataset][split].add(sample_id)
                    progress.set_detail(f"dataset={dataset} shard={shard_name}")
            for dataset in args.datasets:
                for split in SPLITS:
                    handles[dataset][split].flush()
                    os.fsync(handles[dataset][split].fileno())
        for dataset in args.datasets:
            for split in SPLITS:
                os.replace(temporary_paths[(dataset, split)], args.output_root / dataset / f"{split}.jsonl")
    finally:
        progress.close()

    summary = {
        dataset: {
            split: {"questions": len(questions[dataset][split]), **dict(counters[dataset][split])}
            for split in SPLITS
        }
        for dataset in args.datasets
    }
    manifest = {
        "materialization_version": MATERIALIZATION_VERSION,
        "created_at": utc_now(),
        "datasets": args.datasets,
        "source_split": args.source_split,
        "target_field": args.target_field,
        "target_definition": "sigmoid(with-document gold margin, T=1) - sigmoid(no-document gold margin, T=1)",
        "score_root": str(args.score_root.resolve()),
        "trace_root": str(args.trace_root.resolve()),
        "reference_split_root": str(args.reference_split_root.resolve()),
        "exclude_quality_flags": args.exclude_quality_flags,
        "model_input_contract": {
            "included": ["question text", "answer options when present", "one document text"],
            "supervision": ["one continuous utility_target"],
            "forbidden": [
                "gold answer",
                "teacher logits or margins",
                "No-RAG answer/correctness",
                "answer transition",
                "hidden states",
                "RAG2 pseudo-label",
            ],
            "audit_only": ["No-RAG correctness", "answer transition"],
        },
        "splits": summary,
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    logging.info("Margin-regression pointer splits complete: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

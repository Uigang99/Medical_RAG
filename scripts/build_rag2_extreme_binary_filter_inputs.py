#!/usr/bin/env python3
"""Materialize extreme utility pairs for direct RAG2-style binary training.

Only pairs with ``utility_target >= tau`` or ``utility_target <= -tau`` are
written.  The former map to Helpful and the latter to Not Helpful (semantically
Harmful in this experiment).  Neutral pairs never enter any split.

The model input contains only deployment-available information: question,
options, one document, and the target model's cached No-RAG predicted answer.
Gold answers, No-RAG correctness, utility magnitudes, transitions, and teacher
states remain audit/supervision metadata and are not rendered into the input.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from medrag.progress import PipelineProgress  # noqa: E402
from train_rag2_margin_regressor import (  # noqa: E402
    NoRAGAnswerIndex,
    TraceTextStore,
    atomic_json,
    decode_json,
)


MATERIALIZATION_VERSION = "rag2_extreme_utility_binary_answer_aware_v1"
PREPARED_VERSION = "rag2_margin_regression_pointer_splits_v1"
SPLIT_FILES = (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test.jsonl"))


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--prepared-root", type=Path, default=base / "gold_margin_regression_v1/prepared")
    parser.add_argument("--no-rag-generation-root", type=Path, default=base / "train_no_rag_anchored_features_v1/no_rag")
    parser.add_argument("--output-root", type=Path, default=base / "extreme_utility_binary_answer_aware_tau0p2_v1")
    parser.add_argument("--extreme-threshold", type=float, default=0.2)
    parser.add_argument("--trace-shard-cache-size", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument(
        "--max-source-rows-per-split",
        type=int,
        default=None,
        help="Testing only: stop after this many source pointer rows in each split.",
    )
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def count_lines(path: Path) -> int:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        return sum(1 for line in handle if line.strip())


def load_source_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = args.prepared_root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("materialization_version") != PREPARED_VERSION:
        raise RuntimeError(f"Prepared-data contract mismatch: {path}")
    if args.dataset not in value.get("datasets", []):
        raise RuntimeError(f"Dataset absent from prepared manifest: {args.dataset}")
    if value.get("target_field") != "boundary_probability_delta":
        raise RuntimeError("Extreme binary labels require boundary_probability_delta")
    return value


def initial_state() -> dict[str, Any]:
    return {
        "processed": 0,
        "written": 0,
        "source_offset": 0,
        "output_bytes": 0,
        "target_counts": {},
        "balance_group_counts": {},
    }


def load_resume_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return initial_state()
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"processed", "written", "source_offset", "output_bytes"}
    if not required.issubset(value):
        raise RuntimeError(f"Incomplete resume state: {path}")
    return value


def balance_group(no_rag_correct: bool, target: str) -> str:
    state = "no_rag_correct" if no_rag_correct else "no_rag_wrong"
    normalized = "helpful" if target == "helpful" else "not_helpful"
    return f"{state}__{normalized}"


def write_split(
    *,
    args: argparse.Namespace,
    source_path: Path,
    output_path: Path,
    total: int,
    store: TraceTextStore,
    progress: PipelineProgress,
    split_name: str,
) -> dict[str, Any]:
    complete_path = output_path.with_suffix(".complete.json")
    if output_path.is_file() and complete_path.is_file():
        report = json.loads(complete_path.read_text(encoding="utf-8"))
        progress.set_stage(f"reuse {args.dataset}/{split_name}", total=total, initial=total)
        return report

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    state = load_resume_state(state_path)
    processed = int(state["processed"])
    written = int(state["written"])
    source_offset = int(state["source_offset"])
    output_bytes = int(state["output_bytes"])
    target_counts = Counter({str(k): int(v) for k, v in state.get("target_counts", {}).items()})
    group_counts = Counter({str(k): int(v) for k, v in state.get("balance_group_counts", {}).items()})

    if processed and not partial_path.is_file():
        raise RuntimeError(f"Resume state exists but partial output is missing: {partial_path}")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "r+b" if partial_path.exists() else "w+b"
    stage_total = min(total, args.max_source_rows_per_split or total)
    if processed > stage_total:
        raise RuntimeError(f"Resume position exceeds requested source rows: {state_path}")
    progress.set_stage(
        f"materialize {args.dataset}/{split_name}",
        total=stage_total,
        initial=processed,
    )

    with source_path.open("rb", buffering=64 * 1024 * 1024) as source, partial_path.open(output_mode) as output:
        source.seek(source_offset)
        output.truncate(output_bytes)
        output.seek(output_bytes)
        while processed < stage_total:
            line = source.readline()
            if not line:
                break
            source_offset = source.tell()
            if not line.strip():
                continue
            pointer = decode_json(line)
            utility = float(pointer["utility_target"])
            target: str | None
            if utility >= args.extreme_threshold:
                target = "helpful"
            elif utility <= -args.extreme_threshold:
                target = "not helpful"
            else:
                target = None
            if target is not None:
                no_rag_correct = bool(pointer["no_rag_correct_audit_only"])
                group = balance_group(no_rag_correct, target)
                row = {
                    "materialization_version": MATERIALIZATION_VERSION,
                    "dataset": args.dataset,
                    "sample_id": str(pointer["sample_id"]),
                    "pair_id": str(pointer["pair_id"]),
                    "doc_rank": int(pointer["doc_rank"]),
                    "source": str(pointer["source"]),
                    "input": store.official_input(pointer),
                    "target": target,
                    "semantic_target": "helpful" if target == "helpful" else "harmful",
                    "balance_group": group,
                    "no_rag_correct_audit_only": no_rag_correct,
                    "answer_transition_audit_only": str(pointer["answer_transition_audit_only"]),
                    "utility_target_audit_only": utility,
                }
                output.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                )
                written += 1
                target_counts[target] += 1
                group_counts[group] += 1
            processed += 1
            progress.update(1)
            if processed % args.checkpoint_interval == 0:
                output.flush()
                os.fsync(output.fileno())
                atomic_json(
                    state_path,
                    {
                        "processed": processed,
                        "written": written,
                        "source_offset": source_offset,
                        "output_bytes": output.tell(),
                        "target_counts": dict(target_counts),
                        "balance_group_counts": dict(group_counts),
                    },
                )
        output.flush()
        os.fsync(output.fileno())
        output_bytes = output.tell()

    if processed != stage_total:
        raise RuntimeError(f"Source ended early: {source_path} processed={processed} expected={stage_total}")
    os.replace(partial_path, output_path)
    state_path.unlink(missing_ok=True)
    report = {
        "split": split_name,
        "source_rows": processed,
        "written_extreme_rows": written,
        "retained_fraction": written / processed if processed else 0.0,
        "target_counts": dict(target_counts),
        "balance_group_counts": dict(group_counts),
        "threshold": args.extreme_threshold,
        "complete": True,
        "output_bytes": output_bytes,
    }
    atomic_json(complete_path, report)
    return report


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not 0.0 < args.extreme_threshold < 1.0:
        raise ValueError("--extreme-threshold must be in (0,1)")
    if args.checkpoint_interval < 1 or args.trace_shard_cache_size < 1:
        raise ValueError("Checkpoint interval and cache size must be positive")

    contract = load_source_contract(args)
    source_split = str(contract["source_split"])
    trace_root = Path(str(contract["trace_root"]))
    source_paths = {
        split_name: args.prepared_root / args.dataset / filename
        for split_name, filename in SPLIT_FILES
    }
    totals = {name: count_lines(path) for name, path in source_paths.items()}
    requested_totals = {
        name: min(total, args.max_source_rows_per_split or total)
        for name, total in totals.items()
    }
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Indexing cached No-RAG answers for %s", args.dataset)
    answers = NoRAGAnswerIndex(
        args.no_rag_generation_root,
        args.dataset,
        source_split,
        show_progress=args.show_progress,
    )
    store = TraceTextStore(
        trace_root,
        args.dataset,
        source_split,
        args.trace_shard_cache_size,
        answers,
    )
    overall_initial = 0
    for split_name, filename in SPLIT_FILES:
        output_path = output_dir / filename
        complete_path = output_path.with_suffix(".complete.json")
        state_path = output_path.with_suffix(output_path.suffix + ".state.json")
        if output_path.is_file() and complete_path.is_file():
            overall_initial += requested_totals[split_name]
        elif state_path.is_file():
            overall_initial += min(
                int(load_resume_state(state_path)["processed"]),
                requested_totals[split_name],
            )
    progress = PipelineProgress(
        overall_total=sum(requested_totals.values()),
        overall_initial=overall_initial,
        desc=f"ExtremeBinaryPrepare:{args.dataset}",
        enabled=args.show_progress,
    )
    reports: dict[str, Any] = {}
    try:
        for split_name, filename in SPLIT_FILES:
            reports[split_name] = write_split(
                args=args,
                source_path=source_paths[split_name],
                output_path=output_dir / filename,
                total=totals[split_name],
                store=store,
                progress=progress,
                split_name=split_name,
            )
    finally:
        progress.close()

    manifest = {
        "materialization_version": MATERIALIZATION_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "threshold": args.extreme_threshold,
        "training_label_mode": "binary",
        "training_target_labels": ["helpful", "not helpful"],
        "semantic_target_labels": ["helpful", "harmful"],
        "label_protocol": {
            "helpful": f"utility_target >= +{args.extreme_threshold}",
            "not helpful": f"utility_target <= -{args.extreme_threshold} (semantic Harmful)",
            "excluded": f"abs(utility_target) < {args.extreme_threshold} (Neutral)",
        },
        "filter_input": {
            "format": "rag2_answer_aware_evidence_question_v1",
            "included": ["question", "options", "No-RAG predicted answer", "one document"],
            "excluded": [
                "gold answer",
                "No-RAG correctness",
                "No-RAG confidence/logits",
                "No-RAG rationale",
                "utility score",
                "answer transition",
                "hidden states",
            ],
        },
        "source_prepared_manifest": str((args.prepared_root / "manifest.json").resolve()),
        "source_no_rag_manifest": answers.manifest,
        "summary": reports,
        "limited_source_rows_per_split": args.max_source_rows_per_split,
        "resume_contract": "byte-offset source/output checkpoints; completed splits are reused",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    logging.info("Extreme binary inputs complete: %s", output_dir)


if __name__ == "__main__":
    main()

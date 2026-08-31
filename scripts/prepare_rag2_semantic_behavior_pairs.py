#!/usr/bin/env python3
"""Prepare same-question Direct-vs-distractor pairs for Llama LoRA training.

Semantic and frozen-target behavioral measurements are used only to choose one
under-used Direct Support document and one sensitive No-Evidence/Misleading
document per question.  The resulting JSONL files contain ordinary text plus
cached frozen No-RAG targets; no semantic or behavioral feature is intended as
an inference-time model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402
from medrag.training.semantic_behavior_lora import (  # noqa: E402
    choose_semantic_behavior_pair,
    jensen_shannon_divergence,
    stratified_pair_limit,
)


RUN_VERSION = "rag2_semantic_behavior_single_document_pairs_v1"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=base / "filter_training_inputs_semantic_top8_four_class_v1",
    )
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
        "--no-rag-root",
        type=Path,
        default=base / "train_no_rag_anchored_features_v1/no_rag",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base / "semantic_behavior_single_document_pairs_v1",
    )
    parser.add_argument("--max-train-pairs", type=int, default=3000)
    parser.add_argument("--max-eval-pairs", type=int, default=1000)
    parser.add_argument("--hard-fraction", type=float, default=0.7)
    parser.add_argument("--violation-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = args.semantic_root / args.dataset / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dataset") != args.dataset or int(value.get("top_k", -1)) != 8:
        raise ValueError(f"Unexpected semantic manifest: {path}")
    return value


def no_rag_path(args: argparse.Namespace) -> Path:
    return args.no_rag_root / args.dataset / "train" / "no_rag_generations.jsonl"


def score_shards(args: argparse.Namespace) -> list[Path]:
    root = args.score_root / "score_shards" / args.dataset / "train"
    paths = sorted(root.glob("shard_*/rows.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No score shards: {root}")
    return paths


def trace_path(args: argparse.Namespace, score_rows_path: Path) -> Path:
    shard_name = score_rows_path.parent.name
    return args.trace_root / "trace_shards" / args.dataset / "train" / shard_name / "pairs.jsonl"


def load_semantic_index(
    args: argparse.Namespace,
    progress: PipelineProgress,
    counts: dict[str, int],
) -> tuple[dict[str, str], dict[str, str]]:
    pair_to_label: dict[str, str] = {}
    sample_to_split: dict[str, str] = {}
    progress.set_stage("1/4 semantic labels", total=sum(counts.values()))
    for split in SPLITS:
        path = args.semantic_root / args.dataset / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            pair_id = str(row["pair_id"])
            label = str(row["target"])
            if pair_id in pair_to_label:
                raise RuntimeError(f"Duplicate semantic pair: {pair_id}")
            pair_to_label[pair_id] = label
            sample_id = str(row["sample_id"])
            previous = sample_to_split.setdefault(sample_id, split)
            if previous != split:
                raise RuntimeError(f"Question appears in multiple splits: {sample_id}")
            progress.update()
    return pair_to_label, sample_to_split


def load_no_rag(
    args: argparse.Namespace,
    progress: PipelineProgress,
    total: int,
) -> dict[str, dict[str, Any]]:
    path = no_rag_path(args)
    if not path.is_file():
        raise FileNotFoundError(path)
    values: dict[str, dict[str, Any]] = {}
    progress.set_stage("2/4 frozen no-RAG traces", total=total)
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in values:
            raise RuntimeError(f"Duplicate no-RAG question: {sample_id}")
        values[sample_id] = row
        progress.update()
    return values


def score_document(
    metadata: dict[str, Any],
    trace: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    tensor_row: int,
    semantic_label: str,
) -> dict[str, Any]:
    no_prob = tensors["no_document_choice_probabilities"][tensor_row].tolist()
    doc_prob = tensors["with_document_choice_probabilities"][tensor_row].tolist()
    return {
        "pair_id": str(metadata["pair_id"]),
        "doc_rank": int(metadata["doc_rank"]),
        "semantic_label": semantic_label,
        "document_source": str(metadata["document_source"]),
        "document_stable_id": str(metadata["document_stable_id"]),
        "document_text": str(trace["document_text_used"]),
        "rationale": str(trace["rationale"]),
        "base_answer": str(metadata["with_document_prediction"]),
        "base_correct": bool(metadata["with_document_correct"]),
        "with_document_gold_margin": float(tensors["with_document_gold_margin"][tensor_row]),
        "gold_margin_delta": float(tensors["gold_margin_delta"][tensor_row]),
        "answer_js_divergence": jensen_shannon_divergence(no_prob, doc_prob),
        "frozen_choice_probabilities": [float(value) for value in doc_prob],
    }


def materialize_candidates(
    args: argparse.Namespace,
    progress: PipelineProgress,
    pair_to_label: dict[str, str],
    sample_to_split: dict[str, str],
    no_rag: dict[str, dict[str, Any]],
    total_pairs: int,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_sample_shards: dict[str, str] = {}
    progress.set_stage("3/4 same-question candidate pairing", total=total_pairs)
    for rows_path in score_shards(args):
        shard_name = rows_path.parent.name
        traces_path = trace_path(args, rows_path)
        tensors_path = rows_path.with_name("scores.safetensors")
        if not traces_path.is_file() or not tensors_path.is_file():
            raise FileNotFoundError(f"Missing aligned trace/tensor for {rows_path}")
        metadata_rows = list(iter_jsonl(rows_path))
        trace_rows = list(iter_jsonl(traces_path))
        if len(metadata_rows) != len(trace_rows):
            raise RuntimeError(f"Score/trace count mismatch: {rows_path}")
        with safe_open(str(tensors_path), framework="pt", device="cpu") as handle:
            names = (
                "no_document_choice_logits",
                "no_document_choice_probabilities",
                "with_document_choice_probabilities",
                "with_document_gold_margin",
                "gold_margin_delta",
            )
            tensors = {name: handle.get_tensor(name) for name in names}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        first_metadata: dict[str, dict[str, Any]] = {}
        for metadata, trace in zip(metadata_rows, trace_rows):
            pair_id = str(metadata["pair_id"])
            if pair_id != str(trace["pair_id"]):
                raise RuntimeError(f"Score/trace pair mismatch: {pair_id}")
            sample_id = str(metadata["sample_id"])
            first_metadata.setdefault(sample_id, metadata)
            label = pair_to_label.get(pair_id)
            if label is not None:
                grouped[sample_id].append(
                    score_document(
                        metadata,
                        trace,
                        tensors,
                        int(metadata["tensor_row"]),
                        label,
                    )
                )
            progress.update()

        for sample_id, documents in grouped.items():
            previous_shard = seen_sample_shards.setdefault(sample_id, shard_name)
            if previous_shard != shard_name:
                raise RuntimeError(
                    "A question crosses score-shard boundaries, which would make "
                    f"pair selection incomplete: {sample_id} in {previous_shard} and {shard_name}"
                )
            split = sample_to_split.get(sample_id)
            no_row = no_rag.get(sample_id)
            if split is None or no_row is None:
                continue
            pair = choose_semantic_behavior_pair(
                documents, violation_threshold=args.violation_threshold
            )
            if pair is None:
                continue
            metadata = first_metadata[sample_id]
            no_prob = tensors["no_document_choice_probabilities"][int(metadata["tensor_row"])].tolist()
            no_logits = tensors["no_document_choice_logits"][int(metadata["tensor_row"])].tolist()
            selected[split].append(
                {
                    "run_version": RUN_VERSION,
                    "dataset": args.dataset,
                    "split": split,
                    "sample_id": sample_id,
                    "row_idx": int(metadata["row_idx"]),
                    "question": str(no_row["question"]),
                    "options": no_row["options"],
                    "gold_answer": str(metadata["gold_answer"]),
                    "no_rag_rationale": str(no_row["model_raw_rationale"]),
                    "frozen_no_rag_answer": str(metadata["no_document_prediction"]),
                    "frozen_no_rag_correct": bool(metadata["no_document_correct"]),
                    "frozen_no_rag_choice_logits": [float(value) for value in no_logits],
                    "frozen_no_rag_choice_probabilities": [float(value) for value in no_prob],
                    "pair_group": pair["pair_group"],
                    "semantic_preference_violation": float(pair["semantic_preference_violation"]),
                    "negative_semantic_label": str(pair["negative"]["semantic_label"]),
                    "positive": pair["positive"],
                    "negative": pair["negative"],
                }
            )
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "pair_group": dict(Counter(str(row["pair_group"]) for row in rows)),
        "negative_semantic_label": dict(
            Counter(str(row["negative_semantic_label"]) for row in rows)
        ),
        "no_rag_correct": dict(
            Counter(str(bool(row["frozen_no_rag_correct"])).lower() for row in rows)
        ),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not 0.0 <= args.hard_fraction <= 1.0:
        raise ValueError("--hard-fraction must be in [0,1]")
    semantic = semantic_manifest(args)
    semantic_counts = {
        split: int(semantic["materialized"]["splits"][split]["rows"]) for split in SPLITS
    }
    total_semantic = sum(semantic_counts.values())
    total_questions = int(semantic["materialized"]["candidate_questions"])
    total_pairs = int(semantic["materialized"]["candidate_pairs"])
    total_no_rag = count_nonempty_lines(no_rag_path(args))
    output_dir = args.output_root / args.dataset
    output_paths = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "semantic_manifest": file_identity(args.semantic_root / args.dataset / "manifest.json"),
        "score_manifest": file_identity(args.score_root / "manifest.json"),
        "trace_manifest": file_identity(args.trace_root / "generation_manifest.json"),
        "no_rag": file_identity(no_rag_path(args)),
        "max_train_pairs": args.max_train_pairs,
        "max_eval_pairs": args.max_eval_pairs,
        "hard_fraction": args.hard_fraction,
        "violation_threshold": args.violation_threshold,
        "seed": args.seed,
    }
    contract_hash = fingerprint(contract)
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.is_file() and all(path.is_file() for path in output_paths.values()):
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("contract_fingerprint") == contract_hash:
            logging.info("Prepared pair cache is complete and reusable: %s", output_dir)
            return
        raise RuntimeError("Prepared pair cache contract changed; use a new output root")

    logging.info(
        "Pair preparation plan: dataset=%s semantic_rows=%d questions=%d document_pairs=%d",
        args.dataset,
        total_semantic,
        total_questions,
        total_pairs,
    )
    if args.plan_only:
        return
    progress = PipelineProgress(
        overall_total=total_semantic + total_no_rag + total_pairs + total_questions,
        desc=f"SemanticBehaviorPairs:{args.dataset}",
    )
    pair_to_label, sample_to_split = load_semantic_index(
        args, progress, semantic_counts
    )
    no_rag = load_no_rag(args, progress, total_no_rag)
    candidates = materialize_candidates(
        args,
        progress,
        pair_to_label,
        sample_to_split,
        no_rag,
        total_pairs,
    )
    progress.set_stage("4/4 deterministic split selection", total=total_questions)
    selected: dict[str, list[dict[str, Any]]] = {}
    rng_seed = args.seed
    for split in SPLITS:
        limit = args.max_train_pairs if split == "train" else args.max_eval_pairs
        if split != "train" and (limit <= 0 or len(candidates[split]) <= limit):
            selected[split] = sorted(
                candidates[split], key=lambda row: str(row["sample_id"])
            )
        elif split == "train":
            split_hard_fraction = args.hard_fraction
            selected[split] = stratified_pair_limit(
                candidates[split],
                limit=limit,
                hard_fraction=split_hard_fraction,
                seed=rng_seed,
            )
        else:
            hard_count = sum(
                str(row["pair_group"]) == "hard" for row in candidates[split]
            )
            split_hard_fraction = (
                hard_count / len(candidates[split]) if candidates[split] else 0.0
            )
            selected[split] = stratified_pair_limit(
                candidates[split],
                limit=limit,
                hard_fraction=split_hard_fraction,
                seed=rng_seed,
            )
        requested = len(candidates[split]) if limit <= 0 else min(limit, len(candidates[split]))
        if split == "train" and len(selected[split]) < requested:
            logging.warning(
                "Selected %d/%d %s pairs because the requested %.2f hard fraction "
                "is limited by available hard/aligned questions",
                len(selected[split]),
                requested,
                split,
                args.hard_fraction,
            )
        rng_seed += 1
        # Candidate questions that were not eligible still count as inspected.
        progress.update(sum(1 for value in sample_to_split.values() if value == split))
        atomic_jsonl(output_paths[split], selected[split])
    progress.close()
    summary = {split: summarize(rows) for split, rows in selected.items()}
    atomic_json(
        manifest_path,
        {
            **contract,
            "contract_fingerprint": contract_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "eligible": {split: summarize(rows) for split, rows in candidates.items()},
            "selected": summary,
            "files": {split: str(path.resolve()) for split, path in output_paths.items()},
            "file_sha256": {
                split: hashlib.sha256(path.read_bytes()).hexdigest()
                for split, path in output_paths.items()
            },
        },
    )
    logging.info("Semantic-behavior pair preparation complete: %s", json.dumps(summary))


if __name__ == "__main__":
    main()

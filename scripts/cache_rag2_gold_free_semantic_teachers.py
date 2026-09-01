#!/usr/bin/env python3
"""Cache gold-free valid-only responses for the semantic gate pilot.

Selection uses only semantic support labels and prepared-feature availability.
Gold answers are retained only in an evaluation-only field and never affect
selection or generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_layer_pilot import generate_specs, init_llm  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.training.rag2_semantic_attention_data import (  # noqa: E402
    RAG2SemanticAttentionDataset,
)


RUN_VERSION = "rag2_gold_free_semantic_teacher_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-questions", type=int, default=256)
    parser.add_argument("--val-questions", type=int, default=128)
    parser.add_argument("--test-questions", type=int, default=128)
    parser.add_argument("--questions-per-shard", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--llm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=80)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_sample_ids(feature_dir: Path, split: str) -> list[str]:
    selected: list[str] = []
    for path in sorted((feature_dir / "feature_shards" / split).glob("shard_*/features.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        masks = payload["semantic_masks"].bool()
        targets = payload["semantic_targets"].long()
        for index, sample_id in enumerate(payload["sample_ids"]):
            active = masks[index]
            values = targets[index][active]
            # Mixed/indeterminate documents are excluded from this pilot, and
            # both support and non-support must exist in the same Top-8 set.
            if bool(active.all()) and bool((values == 1).any()) and bool((values == 0).any()):
                selected.append(str(sample_id))
    if not selected:
        raise RuntimeError(f"No eligible prepared-feature questions for {split}")
    return selected


def question_row(question: Any) -> dict[str, Any]:
    return {
        "dataset": question.dataset,
        "sample_id": question.sample_id,
        "question": question.question,
        "options": dict(question.options),
        # generate_specs needs this only to report an evaluation flag.  It is
        # never used in the prompt, selection, or training target.
        "answer": question.gold_answers[0],
    }


def shard_paths(output_dir: Path, split: str, index: int) -> tuple[Path, Path]:
    root = output_dir / "teacher_shards" / split / f"shard_{index:05d}"
    return root / "questions.jsonl", root / "COMPLETE.json"


def complete(rows: Path, marker: Path, expected: int, fingerprint: str) -> bool:
    if not rows.is_file() or not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("run_version") == RUN_VERSION
        and value.get("contract_fingerprint") == fingerprint
        and int(value.get("questions", -1)) == expected
        and value.get("sha256") == sha256(rows)
    )


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    for path in (args.feature_dir, args.index_path, args.model_name_or_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_path = args.feature_dir / "preparation_manifest.json"
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("dataset") != args.dataset:
        raise RuntimeError("Feature/dataset mismatch")
    requested = {
        "train": args.train_questions,
        "val": args.val_questions,
        "test": args.test_questions,
    }
    selection: dict[str, list[str]] = {}
    questions_by_split: dict[str, list[Any]] = {}
    for split, limit in requested.items():
        eligible = feature_sample_ids(args.feature_dir, split)
        if limit <= 0 or len(eligible) < limit:
            raise RuntimeError(f"{split}: eligible={len(eligible)} requested={limit}")
        wanted = set(eligible[:limit])
        dataset = RAG2SemanticAttentionDataset(args.index_path, split)
        rows = []
        for index in range(len(dataset)):
            question = dataset[index]
            if question.sample_id in wanted:
                rows.append(question)
        dataset.close()
        by_id = {row.sample_id: row for row in rows}
        missing = wanted - set(by_id)
        if missing:
            raise RuntimeError(f"Index misses {len(missing)} selected {split} questions")
        selection[split] = eligible[:limit]
        questions_by_split[split] = [by_id[sample_id] for sample_id in selection[split]]

    contract = {
        "run_version": RUN_VERSION,
        "hypothesis": (
            "Without gold supervision, a document gate can preserve frozen-Llama valid-only "
            "behavior for full contexts and no-RAG behavior for invalid-only contexts."
        ),
        "dataset": args.dataset,
        "feature_contract": feature_manifest["contract_fingerprint"],
        "feature_manifest_sha256": sha256(manifest_path),
        "index_path": str(args.index_path.resolve()),
        "index_manifest_sha256": sha256(args.index_path.with_suffix(args.index_path.suffix + ".manifest.json")),
        "model": str(args.model_name_or_path.resolve()),
        "split_questions": requested,
        "selection_rule": "all Top-8 labels determinate, at least one support and one non-support",
        "selection_uses_gold": False,
        "seed": args.seed,
        "questions_per_shard": args.questions_per_shard,
        "generation": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
    }
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract["contract_fingerprint"] = fingerprint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "experiment_manifest.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Teacher resume contract mismatch; use a new output directory")
    else:
        atomic_json(contract_path, contract)
    atomic_json(args.output_dir / "selected_sample_ids.json", selection)

    total = sum(requested.values())
    completed_count = 0
    for split, rows in questions_by_split.items():
        for index, block in enumerate(chunks(rows, args.questions_per_shard)):
            rows_path, marker_path = shard_paths(args.output_dir, split, index)
            if args.resume and complete(rows_path, marker_path, len(block), fingerprint):
                completed_count += len(block)
    logging.info(
        "Teacher plan: splits=%s total=%d cached=%d remaining=%d",
        requested, total, completed_count, total - completed_count,
    )
    if args.plan_only:
        return

    generation_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        max_new_tokens=args.max_new_tokens,
        retry_max_new_tokens=args.retry_max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        llm_max_model_len=args.llm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        vllm_performance_mode="throughput",
        max_doc_chars=0,
        generation_batch_size=args.generation_batch_size,
    )
    resources = init_llm(generation_args) if completed_count < total else None
    progress = PipelineProgress(total, completed_count, desc=f"GoldFreeTeachers:{args.dataset}")
    try:
        progress.set_stage("1/1 valid-only frozen-Llama response generation", total=total, initial=completed_count)
        for split, rows in questions_by_split.items():
            blocks = list(chunks(rows, args.questions_per_shard))
            for index, block in enumerate(blocks):
                rows_path, marker_path = shard_paths(args.output_dir, split, index)
                if args.resume and complete(rows_path, marker_path, len(block), fingerprint):
                    continue
                if resources is None:
                    raise RuntimeError("Missing vLLM resources")
                progress.set_detail(f"split={split} shard={index + 1}/{len(blocks)}")
                specs: list[dict[str, Any]] = []
                for question in block:
                    valid_documents = [
                        document for document in question.documents
                        if document.semantic_support_target == 1
                    ]
                    evidence = "\n\n".join(" ".join(document.text.split()) for document in valid_documents)
                    specs.append({
                        "kind": "semantic_valid_only",
                        "dataset": args.dataset,
                        "sample_id": question.sample_id,
                        "pair_id": None,
                        "row": question_row(question),
                        "document": {"text": evidence},
                    })
                tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, choice_ids = resources
                traces = generate_specs(
                    generation_args, tokenizer, llm, rationale_sampling, retry_sampling,
                    choice_sampling, choice_ids, specs,
                )
                compact: list[dict[str, Any]] = []
                for question, trace in zip(block, traces, strict=True):
                    compact.append({
                        "run_version": RUN_VERSION,
                        "dataset": args.dataset,
                        "split": split,
                        "sample_id": question.sample_id,
                        "valid_pair_ids": [d.pair_id for d in question.documents if d.semantic_support_target == 1],
                        "invalid_pair_ids": [d.pair_id for d in question.documents if d.semantic_support_target == 0],
                        "valid_only_response": trace["canonical_response"],
                        "valid_only_answer": trace["answer"],
                        "no_rag_response": question.no_rag.canonical_generation,
                        "no_rag_answer": question.no_rag.predicted_answer,
                        "quality_flags": trace["quality_flags"],
                        # Evaluation-only; the trainer contract forbids reading these fields.
                        "evaluation_only": {
                            "gold_answers": list(question.gold_answers),
                            "valid_only_correct": bool(trace["answer_correct"]),
                            "no_rag_correct": question.no_rag.answer_correct,
                        },
                    })
                atomic_jsonl(rows_path, compact)
                atomic_json(marker_path, {
                    "run_version": RUN_VERSION,
                    "contract_fingerprint": fingerprint,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "questions": len(compact),
                    "sha256": sha256(rows_path),
                })
                progress.update(len(compact))
        atomic_json(args.output_dir / "teacher_manifest.json", {
            **contract,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "teacher_rows": total,
        })
        logging.info("Gold-free semantic teacher cache complete: %s", args.output_dir)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

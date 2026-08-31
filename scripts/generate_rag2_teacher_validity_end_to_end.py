#!/usr/bin/env python3
"""Generate regenerated-rationale Top-8 document-removal teachers.

This bounded validity audit reuses an already completed fixed-rationale LOO
teacher to select a deterministic MedQA cohort.  For every selected question,
the target Llama is run on the full Top-8 context twice and on each of the
eight leave-one-document-out contexts.  Both the rationale and final
constrained choice are regenerated for every intervention.

The output is a construct-validity reference, not a scalable training target.
Question rows are atomic and resumable, and progress reports both the active
stage and total workflow ETA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_layer_pilot import generate_specs, init_llm  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.training.rag2_semantic_attention_data import (  # noqa: E402
    RAG2SemanticAttentionDataset,
    SemanticAttentionQuestion,
)


RUN_VERSION = "rag2_teacher_validity_regenerated_rationale_loo_v1"
VARIANTS_PER_QUESTION = 10  # full, exact repeat, and eight removals
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medqa",), default="medqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--fixed-teacher-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--questions-per-batch", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--llm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=80)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument(
        "--vllm-performance-mode",
        choices=("balanced", "interactivity", "throughput"),
        default="throughput",
    )
    parser.add_argument("--max-doc-chars", type=int, default=0)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def row_path(output_dir: Path, sample_id: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return output_dir / "rows" / f"{digest}.json"


def valid_row(path: Path, sample_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        row.get("run_version") == RUN_VERSION
        and row.get("sample_id") == sample_id
        and row.get("contract_fingerprint") == fingerprint
        and len(row.get("variants") or []) == VARIANTS_PER_QUESTION
    )


def load_fixed_teacher_rows(root: Path, split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "rows" / split).glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = str(payload.get("sample_id") or "")
        if not sample_id or sample_id in rows:
            raise RuntimeError(f"Invalid or duplicate fixed teacher row: {path}")
        rows[sample_id] = payload
    if not rows:
        raise FileNotFoundError(f"No fixed teacher rows for {split}: {root}")
    return rows


def select_ids(values: list[str], maximum: int, seed: int) -> list[str]:
    ordered = sorted(values)
    random.Random(seed).shuffle(ordered)
    if maximum > 0:
        ordered = ordered[:maximum]
    return sorted(ordered)


def question_as_row(question: SemanticAttentionQuestion) -> dict[str, Any]:
    if not question.gold_answers:
        raise ValueError(f"Question has no gold answer: {question.sample_id}")
    return {
        "dataset": question.dataset,
        "split": question.split,
        "sample_id": question.sample_id,
        "row_idx": question.row_idx,
        "question": question.question,
        "options": dict(question.options),
        "answer": question.gold_answers[0],
        "documents": [
            {
                "pair_id": document.pair_id,
                "text": " ".join(document.text.split()),
                "title": document.title,
                "source": document.source,
            }
            for document in question.documents
        ],
    }


def document_context(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(document["text"]).strip() for document in documents)


def build_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    documents = list(row["documents"])
    variants: list[tuple[str, list[dict[str, Any]]]] = [
        ("full", documents),
        ("repeat", documents),
    ]
    variants.extend(
        (f"remove_{index}", [document for slot, document in enumerate(documents) if slot != index])
        for index in range(len(documents))
    )
    return [
        {
            "kind": "teacher_validity_end_to_end",
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "pair_id": variant,
            "row": row,
            "document": {"text": document_context(active_documents)},
        }
        for variant, active_documents in variants
    ]


def compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": str(trace["pair_id"]),
        "rationale": str(trace["rationale"]),
        "answer": str(trace["answer"]),
        "answer_correct": bool(trace["answer_correct"]),
        "choice_logprobs": trace["choice_logprobs"],
        "quality_flags": list(trace["quality_flags"]),
        "rationale_finish_reason": trace["rationale_finish_reason"],
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.max_questions <= 0 or args.questions_per_batch <= 0:
        raise ValueError("Question limits and batch size must be positive")
    if args.max_doc_chars != 0:
        raise ValueError("Validity audit requires complete document contexts (--max-doc-chars 0)")
    for path in (args.fixed_teacher_dir, args.index_path, args.model_name_or_path):
        if not path.exists():
            raise FileNotFoundError(path)
    fixed_manifest_path = args.fixed_teacher_dir / "preparation_manifest.json"
    if not fixed_manifest_path.is_file():
        raise FileNotFoundError(fixed_manifest_path)
    fixed_rows = load_fixed_teacher_rows(args.fixed_teacher_dir, args.split)
    selected_ids = select_ids(list(fixed_rows), args.max_questions, args.sample_seed)
    dataset = RAG2SemanticAttentionDataset(args.index_path, args.split)
    indexed: dict[str, SemanticAttentionQuestion] = {}
    try:
        for index in range(len(dataset)):
            question = dataset[index]
            if question.sample_id in selected_ids:
                indexed[question.sample_id] = question
    finally:
        dataset.close()
    missing = sorted(set(selected_ids) - set(indexed))
    if missing:
        raise RuntimeError(f"Selected fixed-teacher IDs are absent from grouped index: {missing[:5]}")
    rows = {sample_id: question_as_row(indexed[sample_id]) for sample_id in selected_ids}
    for sample_id in selected_ids:
        expected = [str(value) for value in fixed_rows[sample_id]["pair_ids"]]
        actual = [str(document["pair_id"]) for document in rows[sample_id]["documents"]]
        if actual != expected:
            raise RuntimeError(f"Document identity/order mismatch for {sample_id}")

    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "fixed_teacher_dir": str(args.fixed_teacher_dir.resolve()),
        "fixed_teacher_manifest_sha256": sha256_file(fixed_manifest_path),
        "index_path": str(args.index_path.resolve()),
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "selected_sample_ids": selected_ids,
        "sample_seed": args.sample_seed,
        "variants_per_question": VARIANTS_PER_QUESTION,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "retry_max_new_tokens": args.retry_max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_doc_chars": args.max_doc_chars,
        },
    }
    fingerprint = canonical_hash(run_contract)
    run_contract["contract_fingerprint"] = fingerprint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError("End-to-end teacher resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(contract_path, run_contract)

    completed_ids = {
        sample_id
        for sample_id in selected_ids
        if valid_row(row_path(args.output_dir, sample_id), sample_id, fingerprint)
    }
    pending_ids = [sample_id for sample_id in selected_ids if sample_id not in completed_ids]
    total_units = len(selected_ids) * VARIANTS_PER_QUESTION
    cached_units = len(completed_ids) * VARIANTS_PER_QUESTION
    logging.info(
        "End-to-end teacher validity plan: questions=%d cached=%d remaining=%d "
        "rationale_generations=%d",
        len(selected_ids),
        len(completed_ids),
        len(pending_ids),
        len(pending_ids) * VARIANTS_PER_QUESTION,
    )
    if args.plan_only:
        return

    progress = PipelineProgress(
        overall_total=2 * total_units,
        overall_initial=cached_units,
        desc="TeacherValidityEndToEnd:medqa",
    )
    resources = None
    try:
        progress.set_stage(
            "1/2 regenerate rationale + constrained choice for full/repeat/8 removals",
            total=total_units,
            initial=cached_units,
        )
        if pending_ids:
            resources = init_llm(args)
            for start in range(0, len(pending_ids), args.questions_per_batch):
                batch_ids = pending_ids[start : start + args.questions_per_batch]
                specs = [spec for sample_id in batch_ids for spec in build_specs(rows[sample_id])]
                traces = generate_specs(args, *resources, specs)
                grouped: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in batch_ids}
                for trace in traces:
                    grouped[str(trace["sample_id"])].append(compact_trace(trace))
                for sample_id in batch_ids:
                    variants = grouped[sample_id]
                    expected_order = ["full", "repeat"] + [f"remove_{index}" for index in range(8)]
                    observed = [str(value["variant"]) for value in variants]
                    if observed != expected_order:
                        raise RuntimeError(
                            f"Counterfactual variant order mismatch for {sample_id}: {observed}"
                        )
                    atomic_write_json(
                        row_path(args.output_dir, sample_id),
                        {
                            "run_version": RUN_VERSION,
                            "contract_fingerprint": fingerprint,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "sample_id": sample_id,
                            "pair_ids": [document["pair_id"] for document in rows[sample_id]["documents"]],
                            "variants": variants,
                        },
                    )
                    progress.update(VARIANTS_PER_QUESTION)
                    progress.set_detail(f"sample={sample_id}")
        progress.set_stage("2/2 validate atomic question rows", total=total_units)
        for sample_id in selected_ids:
            if not valid_row(row_path(args.output_dir, sample_id), sample_id, fingerprint):
                raise RuntimeError(f"Invalid regenerated teacher row: {sample_id}")
            progress.update(VARIANTS_PER_QUESTION)
        atomic_write_json(
            args.output_dir / "generation_manifest.json",
            {
                **run_contract,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "question_count": len(selected_ids),
                "generation_count": total_units,
            },
        )
        logging.info("Regenerated-rationale LOO teacher complete: %s", args.output_dir)
    finally:
        progress.close()
        del resources


if __name__ == "__main__":
    main()

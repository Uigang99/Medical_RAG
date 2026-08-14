#!/usr/bin/env python3
"""Stream the complete rerank Top-k dataset through pre-answer feature extraction.

Unlike the pilot sampler, this entry point processes every candidate question
sequentially and never holds the complete corpus in memory.  It reuses the
same fixed prompt, constrained A/B/C/D decoding, h0/hD states, and gold-answer
direction contract as ``extract_rag2_preanswer_hidden_pilot.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

from tqdm.auto import tqdm

from extract_rag2_preanswer_hidden_pilot import (
    FORMAT_VERSION,
    PROMPT_VERSION,
    FeatureExtractor,
    atomic_write_json,
    complete_shard_valid,
    consolidate_outputs,
    document_sort_key,
    normalize_document,
    normalize_gold,
    normalize_options,
    process_shard,
    shard_paths,
    utc_now,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
FULL_RUN_VERSION = "rag2_preanswer_hidden_gold_direction_full_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract h0, hD, c and direct-choice answers for every candidate Top-k pair."
    )
    parser.add_argument("--dataset", choices=["medmcqa", "medqa"], required=True)
    parser.add_argument("--candidates-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--layers", nargs="+", default=["16", "24", "28", "final"])
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=64)
    parser.add_argument("--questions-per-shard", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--start-question", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def count_lines(path: Path) -> int:
    count = 0
    final_byte = b""
    with path.open("rb") as handle, tqdm(
        total=path.stat().st_size,
        desc="count-source",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        while block := handle.read(64 * 1024 * 1024):
            count += block.count(b"\n")
            final_byte = block[-1:]
            progress.update(len(block))
    if final_byte and final_byte != b"\n":
        count += 1
    return count


def selected_question_count(total: int, start: int, limit: int | None) -> int:
    available = max(0, total - start)
    return min(available, limit) if limit is not None else available


def normalize_row(
    row: dict[str, Any], dataset: str, source_row_index: int, docs_per_question: int
) -> dict[str, Any]:
    actual_dataset = str(row.get("dataset") or dataset).lower()
    if actual_dataset != dataset:
        raise ValueError(
            f"Dataset mismatch at source row {source_row_index}: {actual_dataset} != {dataset}"
        )
    sample_id = str(row.get("sample_id") or "")
    question = str(row.get("question") or "").strip()
    options = normalize_options(row)
    gold_answer = normalize_gold(row)
    raw_documents = row.get("candidate_documents")
    if not sample_id or not question or options is None or gold_answer is None:
        raise ValueError(f"Invalid MCQ contract at source row {source_row_index}")
    if not isinstance(raw_documents, list) or len(raw_documents) < docs_per_question:
        raise ValueError(
            f"Need {docs_per_question} documents at source row {source_row_index}, "
            f"found {len(raw_documents) if isinstance(raw_documents, list) else 'non-list'}"
        )
    ordered = sorted(
        enumerate(raw_documents, start=1),
        key=lambda item: document_sort_key(item[1], item[0]),
    )
    documents: list[dict[str, Any]] = []
    for fallback_rank, raw_document in ordered:
        document = normalize_document(raw_document, fallback_rank)
        if document is not None:
            document["document_index"] = len(documents)
            document["pair_id"] = f"{sample_id}::{document['rerank_rank']}::{document['stable_id']}"
            documents.append(document)
        if len(documents) == docs_per_question:
            break
    if len(documents) != docs_per_question:
        raise ValueError(
            f"Only {len(documents)} nonempty documents at source row {source_row_index}"
        )
    return {
        "dataset": dataset,
        "source_split": str(row.get("split") or "train"),
        "source_row_index": source_row_index,
        "sample_id": sample_id,
        "question": question,
        "options": options,
        "gold_answer": gold_answer,
        "documents": documents,
    }


def iter_selected_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    selected_index = 0
    with args.candidates_path.open("r", encoding="utf-8") as handle:
        for source_row_index, line in enumerate(handle):
            if source_row_index < args.start_question:
                continue
            if args.limit_questions is not None and selected_index >= args.limit_questions:
                break
            if not line.strip():
                continue
            row = normalize_row(
                json.loads(line), args.dataset, source_row_index, args.docs_per_question
            )
            row["question_index"] = selected_index
            selected_index += 1
            yield row


def stream_chunks(values: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def invocation_contract(args: argparse.Namespace, total_questions: int) -> dict[str, Any]:
    stat = args.candidates_path.stat()
    return {
        "run_version": FULL_RUN_VERSION,
        "feature_format_version": FORMAT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": args.dataset,
        "candidates_path": str(args.candidates_path.resolve()),
        "candidates_size": stat.st_size,
        "candidates_mtime_ns": stat.st_mtime_ns,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "total_questions": total_questions,
        "total_pairs": total_questions * args.docs_per_question,
        "docs_per_question": args.docs_per_question,
        "layers": [str(layer) for layer in args.layers],
        "questions_per_shard": args.questions_per_shard,
        "max_input_tokens": args.max_input_tokens,
        "start_question": args.start_question,
        "limit_questions": args.limit_questions,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "answer_contract": "one constrained argmax token over A/B/C/D after Final answer:",
        "direction_contract": "c = -gradient of gold choice-normalized NLL with respect to h0",
    }


def validate_manifest(args: argparse.Namespace, total_questions: int) -> None:
    path = args.output_dir / "run_manifest.json"
    contract = invocation_contract(args, total_questions)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in contract}
        if comparable != contract:
            raise RuntimeError("Existing run manifest does not match this invocation; use a new output directory")
        if not args.resume:
            raise FileExistsError(f"Output exists with --no-resume: {args.output_dir}")
        return
    manifest = dict(contract)
    manifest["created_at"] = utc_now()
    atomic_write_json(path, manifest)


def run(args: argparse.Namespace) -> None:
    if not args.candidates_path.is_file():
        raise FileNotFoundError(args.candidates_path)
    if args.docs_per_question < 1 or args.questions_per_shard < 1:
        raise ValueError("Document and shard sizes must be positive")
    if args.start_question < 0 or (args.limit_questions is not None and args.limit_questions < 1):
        raise ValueError("Invalid question range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_questions = count_lines(args.candidates_path)
    total_questions = selected_question_count(
        source_questions, args.start_question, args.limit_questions
    )
    if total_questions == 0:
        raise RuntimeError("Selected question range is empty")
    total_pairs = total_questions * args.docs_per_question
    shard_count = math.ceil(total_questions / args.questions_per_shard)
    validate_manifest(args, total_questions)

    completed_pairs = 0
    pending_shards = 0
    for shard_index in range(shard_count):
        questions = min(
            args.questions_per_shard,
            total_questions - shard_index * args.questions_per_shard,
        )
        pairs = questions * args.docs_per_question
        if args.resume and complete_shard_valid(
            shard_paths(args.output_dir, shard_index), questions, pairs
        ):
            completed_pairs += pairs
        else:
            pending_shards += 1
    logging.info(
        "Full extraction contract: dataset=%s questions=%d pairs=%d shards=%d "
        "completed_pairs=%d pending_shards=%d",
        args.dataset,
        total_questions,
        total_pairs,
        shard_count,
        completed_pairs,
        pending_shards,
    )

    extractor: FeatureExtractor | None = None
    progress = tqdm(
        total=total_pairs,
        initial=completed_pairs,
        desc=f"PreAnswerHidden:{args.dataset}",
        unit="pair",
        dynamic_ncols=True,
    )
    observed_questions = 0
    for shard_index, shard_rows in enumerate(
        stream_chunks(iter_selected_rows(args), args.questions_per_shard)
    ):
        observed_questions += len(shard_rows)
        expected_pairs = len(shard_rows) * args.docs_per_question
        paths = shard_paths(args.output_dir, shard_index)
        if args.resume and complete_shard_valid(paths, len(shard_rows), expected_pairs):
            continue
        if extractor is None:
            extractor = FeatureExtractor(args)
        process_shard(args, extractor, shard_index, shard_rows)
        progress.update(expected_pairs)
        progress.set_postfix(shard=shard_index)
    progress.close()
    if observed_questions != total_questions:
        raise RuntimeError(
            f"Observed {observed_questions} selected questions, expected {total_questions}"
        )
    summary = consolidate_outputs(args, shard_count)
    summary.update(
        {
            "run_version": FULL_RUN_VERSION,
            "dataset": args.dataset,
            "expected_questions": total_questions,
            "expected_pairs": total_pairs,
        }
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    logging.info("Full extraction complete: %s", json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()

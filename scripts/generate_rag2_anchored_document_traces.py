#!/usr/bin/env python3
"""Generate independent anchored rationale+answer traces for reranked documents.

Every reranked document is evaluated alone: ``question + options + one
document``.  The model freely generates a rationale and is then constrained to
one A/B/C/D token after the fixed ``Final answer: (`` anchor.  Outputs are
atomic, question-sharded, resumable, and intentionally retain generated-token
log probabilities for later RAG2-style PPL analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_layer_pilot import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    generate_specs,
    init_llm,
)
from medrag.io_utils import iter_jsonl  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    GENERATION_POLICY_VERSION,
    PROMPT_VERSION,
    TRACE_VERSION,
    normalized_mcq_row,
)

RUN_VERSION = "rag2_anchored_independent_document_generation_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "candidates/source_balanced32_rerank8_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["medmcqa", "medqa"], default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-file", default="candidates_top8.jsonl")
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--questions-per-shard", type=int, default=256)
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
    parser.add_argument("--vllm-performance-mode", choices=["balanced", "interactivity", "throughput"], default="throughput")
    parser.add_argument("--max-doc-chars", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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


def stream_chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def candidate_paths(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    root = args.candidate_root / dataset / args.split
    return root / args.candidate_file, root / "candidate_manifest.json"


def document_pair_id(sample_id: str, document: dict[str, Any], rank: int) -> str:
    stable_id = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    if not stable_id:
        stable_id = f"{document.get('source')}:{document.get('local_id')}"
    return f"{sample_id}::{rank}::{stable_id}"


def load_candidate_contract(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    candidate_path, manifest_path = candidate_paths(args, dataset)
    if not candidate_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing candidates for {dataset}: {candidate_path} / {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "type": "rag2_filter_candidate_dataset",
        "dataset": dataset,
        "split": args.split,
        "top_k": args.docs_per_question,
        "candidate_layout": "source_balanced",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate contract mismatch for {dataset}: {mismatches}")
    question_count = int(manifest.get("selected_question_count", -1))
    if question_count <= 0:
        raise ValueError(f"Invalid selected_question_count for {dataset}: {question_count}")
    stat = candidate_path.stat()
    return {
        "dataset": dataset,
        "candidate_path": str(candidate_path.resolve()),
        "candidate_manifest_path": str(manifest_path.resolve()),
        "candidate_size_bytes": stat.st_size,
        "candidate_mtime_ns": stat.st_mtime_ns,
        "question_count": question_count,
        "pair_count": question_count * args.docs_per_question,
        "candidate_pool_top_k": manifest.get("candidate_pool_top_k"),
        "per_source_top_k": manifest.get("per_source_top_k"),
        "rerank_top_k": manifest.get("top_k"),
        "query_prompt_version": manifest.get("query_prompt_version"),
    }


def normalized_candidate_rows(
    path: Path,
    dataset: str,
    split: str,
    docs_per_question: int,
) -> Iterator[dict[str, Any]]:
    for line_index, raw in enumerate(iter_jsonl(path)):
        row = normalized_mcq_row(raw)
        sample_id = str(raw.get("sample_id") or row.get("id") or f"{dataset}:{split}:{line_index:06d}")
        documents = list(raw.get("candidate_documents") or [])
        documents.sort(key=lambda item: int(item.get("rerank_rank") or 10**9))
        if len(documents) != docs_per_question:
            raise ValueError(
                f"Expected {docs_per_question} reranked documents for {sample_id}, found {len(documents)}"
            )
        seen: set[str] = set()
        normalized_documents: list[dict[str, Any]] = []
        for fallback_rank, original in enumerate(documents, 1):
            document = dict(original)
            rank = int(document.get("rerank_rank") or fallback_rank)
            if rank != fallback_rank:
                raise ValueError(f"Non-contiguous rerank rank for {sample_id}: {rank} != {fallback_rank}")
            text = str(document.get("text") or "").strip()
            if not text:
                raise ValueError(f"Empty reranked document for {sample_id} rank={rank}")
            pair_id = document_pair_id(sample_id, document, rank)
            if pair_id in seen:
                raise ValueError(f"Duplicate pair_id for {sample_id}: {pair_id}")
            seen.add(pair_id)
            document["pair_id"] = pair_id
            normalized_documents.append(document)
        yield {
            **row,
            "dataset": dataset,
            "split": split,
            "sample_id": sample_id,
            "row_idx": int(raw.get("row_idx", line_index)),
            "documents": normalized_documents,
        }


def shard_paths(root: Path, dataset: str, split: str, shard_index: int) -> dict[str, Path]:
    base = root / "trace_shards" / dataset / split / f"shard_{shard_index:05d}"
    return {"root": base, "pairs": base / "pairs.jsonl", "complete": base / "COMPLETE.json"}


def valid_complete(paths: dict[str, Path], question_count: int, pair_count: int) -> bool:
    if not paths["pairs"].is_file() or not paths["complete"].is_file():
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and int(marker.get("question_count", -1)) == question_count
        and int(marker.get("pair_count", -1)) == pair_count
        and int(marker.get("pairs_size_bytes", -1)) == paths["pairs"].stat().st_size
    )


def compact_trace(trace: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    document = dict(trace.get("document") or {})
    document.pop("text", None)
    trace = dict(trace)
    trace.pop("pilot_version", None)
    trace.update(
        {
            "run_version": RUN_VERSION,
            "stage": "rag2_anchored_independent_with_document",
            "split": str(source.get("split") or trace.get("split") or "train"),
            "row_idx": int(source["row_idx"]),
            "doc_rank": int(document["rerank_rank"]),
            "pair_id": str(document["pair_id"]),
            "document": document,
            "ppl_scope_version": "generated_rationale_v1",
        }
    )
    return trace


def build_specs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in rows:
        for document in row["documents"]:
            specs.append(
                {
                    "kind": "with_document",
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "pair_id": document["pair_id"],
                    "row": row,
                    "document": document,
                }
            )
    return specs


def immutable_contract(args: argparse.Namespace, contracts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "datasets": list(args.datasets),
        "split": args.split,
        "docs_per_question": args.docs_per_question,
        "questions_per_shard": args.questions_per_shard,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_doc_chars": args.max_doc_chars,
        "candidate_contracts": contracts,
    }


def ensure_run_contract(path: Path, contract: dict[str, Any]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(f"Cannot resume with a different generation contract: {path}")
    else:
        atomic_write_json(path, contract)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.questions_per_shard <= 0 or args.generation_batch_size <= 0:
        raise ValueError("Shard and generation batch sizes must be positive")
    if args.docs_per_question <= 0:
        raise ValueError("--docs-per-question must be positive")
    contracts = {dataset: load_candidate_contract(args, dataset) for dataset in args.datasets}
    question_counts = {dataset: int(contracts[dataset]["question_count"]) for dataset in args.datasets}
    pair_counts = {dataset: int(contracts[dataset]["pair_count"]) for dataset in args.datasets}
    total_pairs = sum(pair_counts.values())
    logging.info(
        "Anchored document generation plan: questions=%s pairs=%s total_pairs=%d",
        question_counts,
        pair_counts,
        total_pairs,
    )
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_run_contract(args.output_root / "generation_contract.json", immutable_contract(args, contracts))

    completed_pairs = 0
    for dataset in args.datasets:
        shard_count = math.ceil(question_counts[dataset] / args.questions_per_shard)
        for shard_index in range(shard_count):
            expected_questions = min(
                args.questions_per_shard,
                question_counts[dataset] - shard_index * args.questions_per_shard,
            )
            expected_pairs = expected_questions * args.docs_per_question
            if args.resume and valid_complete(
                shard_paths(args.output_root, dataset, args.split, shard_index),
                expected_questions,
                expected_pairs,
            ):
                completed_pairs += expected_pairs

    resources = None
    choice_token_ids: dict[str, int] = {}
    if completed_pairs < total_pairs:
        resources = init_llm(args)
        choice_token_ids = resources[-1]
    else:
        manifest_path = args.output_root / "generation_manifest.json"
        if manifest_path.is_file():
            choice_token_ids = json.loads(manifest_path.read_text(encoding="utf-8")).get("choice_token_ids") or {}
        logging.info("All document generation shards are already complete; skipping vLLM load.")

    progress = PipelineProgress(
        overall_total=2 * total_pairs,
        overall_initial=completed_pairs,
        desc="AnchoredDocumentPipeline",
    )
    try:
        progress.set_stage(
            "1/2 independent document rationale+answer generation",
            total=total_pairs,
            initial=completed_pairs,
        )
        for dataset in args.datasets:
            candidate_path = Path(contracts[dataset]["candidate_path"])
            rows = normalized_candidate_rows(candidate_path, dataset, args.split, args.docs_per_question)
            observed_questions = 0
            dataset_shards = math.ceil(question_counts[dataset] / args.questions_per_shard)
            for shard_index, shard_rows in enumerate(stream_chunks(rows, args.questions_per_shard)):
                progress.set_detail(
                    f"dataset={dataset} shard={shard_index + 1}/{dataset_shards}"
                )
                observed_questions += len(shard_rows)
                expected_pairs = len(shard_rows) * args.docs_per_question
                paths = shard_paths(args.output_root, dataset, args.split, shard_index)
                if args.resume and valid_complete(paths, len(shard_rows), expected_pairs):
                    continue
                if resources is None:
                    raise RuntimeError("Missing vLLM resources for an incomplete document shard")
                tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, _ = resources
                specs = build_specs(shard_rows)
                traces = generate_specs(
                    args,
                    tokenizer,
                    llm,
                    rationale_sampling,
                    retry_sampling,
                    choice_sampling,
                    choice_token_ids,
                    specs,
                )
                if len(traces) != expected_pairs:
                    raise RuntimeError(f"Generated pair mismatch: {len(traces)} != {expected_pairs}")
                source_by_sample = {row["sample_id"]: row for row in shard_rows}
                output_rows = [compact_trace(trace, source_by_sample[trace["sample_id"]]) for trace in traces]
                paths["root"].mkdir(parents=True, exist_ok=True)
                atomic_write_jsonl(paths["pairs"], output_rows)
                marker = {
                    "run_version": RUN_VERSION,
                    "completed_at": utc_now(),
                    "dataset": dataset,
                    "split": args.split,
                    "shard_index": shard_index,
                    "question_count": len(shard_rows),
                    "pair_count": len(output_rows),
                    "valid_pair_count": sum(bool(row.get("valid_for_layer_analysis")) for row in output_rows),
                    "quality_flags": dict(Counter(flag for row in output_rows for flag in (row.get("quality_flags") or []))),
                    "pairs_size_bytes": paths["pairs"].stat().st_size,
                }
                atomic_write_json(paths["complete"], marker)
                progress.update(len(output_rows))
            if observed_questions != question_counts[dataset]:
                raise RuntimeError(
                    f"Candidate coverage mismatch for {dataset}: {observed_questions} != {question_counts[dataset]}"
                )
        markers: list[dict[str, Any]] = []
        for dataset in args.datasets:
            for marker_path in sorted((args.output_root / "trace_shards" / dataset / args.split).glob("shard_*/COMPLETE.json")):
                markers.append(json.loads(marker_path.read_text(encoding="utf-8")))
        atomic_write_json(
            args.output_root / "generation_manifest.json",
            {
                "run_version": RUN_VERSION,
                "trace_version": TRACE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "generation_policy_version": GENERATION_POLICY_VERSION,
                "ppl_scope_version": "generated_rationale_v1",
                "created_at": utc_now(),
                "model_name_or_path": str(args.model_name_or_path.resolve()),
                "datasets": question_counts,
                "pairs_by_dataset": pair_counts,
                "total_questions": sum(question_counts.values()),
                "total_pairs": total_pairs,
                "docs_per_question": args.docs_per_question,
                "questions_per_shard": args.questions_per_shard,
                "choice_token_ids": choice_token_ids,
                "valid_pairs": sum(int(marker.get("valid_pair_count", 0)) for marker in markers),
                "quality_flags": dict(Counter({
                    flag: sum(int((marker.get("quality_flags") or {}).get(flag, 0)) for marker in markers)
                    for flag in {name for marker in markers for name in (marker.get("quality_flags") or {})}
                })),
                "candidate_contracts": contracts,
                "next_stage": "extract_rag2_anchored_document_features.py",
            },
        )
        logging.info("Independent anchored document generation complete: %s", args.output_root)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

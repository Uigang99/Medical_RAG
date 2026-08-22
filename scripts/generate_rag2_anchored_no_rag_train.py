#!/usr/bin/env python3
"""Generate complete no-RAG anchored traces for MCQ training splits.

The model first generates a free rationale up to a fixed reasoning boundary.
It then emits exactly one constrained A/B/C/D token.  The exact option text is
appended deterministically.  The complete rationale and terminal answer are
stored as the retrieval query.  Outputs are atomic, sharded, and resumable.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
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

RUN_VERSION = "rag2_anchored_no_rag_train_generation_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "datasets/benchmark/mcq/unified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["medmcqa", "medqa"], default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
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
    parser.add_argument("--max-doc-chars", type=int, default=0, help=argparse.SUPPRESS)
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


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def stream_chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def source_path(args: argparse.Namespace, dataset: str) -> Path:
    return args.benchmark_root / dataset / f"{args.split}.jsonl"


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def normalized_source_rows(path: Path, dataset: str, split: str) -> Iterator[dict[str, Any]]:
    for row_idx, raw in enumerate(iter_jsonl(path)):
        row = normalized_mcq_row(raw)
        yield {
            **row,
            "dataset": str(row.get("dataset") or dataset),
            "split": str(row.get("split") or split),
            "sample_id": str(row.get("id") or row.get("sample_id") or f"{dataset}:{split}:{row_idx:06d}"),
            "row_idx": row_idx,
        }


def shard_paths(root: Path, dataset: str, split: str, shard_index: int) -> dict[str, Path]:
    base = root / "trace_shards" / dataset / split / f"shard_{shard_index:05d}"
    return {"root": base, "rows": base / "questions.jsonl", "complete": base / "COMPLETE.json"}


def valid_complete(paths: dict[str, Path], expected: int) -> bool:
    if not paths["rows"].is_file() or not paths["complete"].is_file():
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("run_version") == RUN_VERSION and int(marker.get("question_count", -1)) == expected


def artifact_row(trace: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    rationale_stats = trace.get("rationale_stats") or {}
    answer = str(trace["answer"])
    correct = bool(trace["answer_correct"])
    full_query = str((trace.get("retrieval_queries") or {}).get("rationale_answer") or "").strip()
    return {
        "schema_version": 3,
        "stage": "rag2_anchored_no_rag",
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "ppl_scope_version": "generated_rationale_v1",
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "sample_id": trace["sample_id"],
        "row_idx": int(source["row_idx"]),
        "dataset": trace["dataset"],
        "split": source["split"],
        "subject": source.get("subject"),
        "question": trace["question"],
        "options": trace["options"],
        "gold_answer": trace["gold_answer"],
        "gold_answers": [trace["gold_answer"]],
        "no_rag_generation": trace["canonical_response"],
        "model_raw_rationale": trace["model_raw_rationale"],
        "canonical_generation": trace["canonical_response"],
        "retrieval_query": full_query,
        "retrieval_queries": trace["retrieval_queries"],
        "parsed": {
            "visible_text": trace["canonical_response"],
            "rationale": trace["rationale"],
            "rationale_only": trace["rationale"],
            "rationale_query": full_query,
            "final_answer": answer,
            "final_answer_correct": correct,
            "parse_errors": [],
        },
        "answer": answer,
        "answer_text": trace["answer_text"],
        "answer_correct": correct,
        "quality_flags": trace.get("quality_flags") or [],
        "valid": bool(trace.get("valid_for_layer_analysis")),
        "finish_reason": trace.get("rationale_finish_reason"),
        "stop_reason": trace.get("rationale_stop_reason"),
        "truncated_by_max_tokens": trace.get("rationale_finish_reason") == "length",
        "rationale_token_ids": trace.get("rationale_token_ids") or [],
        "rationale_stats": rationale_stats,
        "generation_stats": {"rationale": rationale_stats, "rationale_only": rationale_stats},
        "choice_token_id": trace["choice_token_id"],
        "choice_logprobs": trace["choice_logprobs"],
        "user_prompt_sha256": trace["user_prompt_sha256"],
        "rendered_rationale_prompt_sha256": trace["rendered_rationale_prompt_sha256"],
    }


def read_shard_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def materialize_dataset(
    args: argparse.Namespace,
    dataset: str,
    expected: int,
    progress: PipelineProgress,
) -> Path:
    target_dir = args.output_root / "no_rag" / dataset / args.split
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "no_rag_generations.jsonl"
    temporary = target.with_name(target.name + ".partial")
    written = 0
    shard_count = math.ceil(expected / args.questions_per_shard)
    with temporary.open("w", encoding="utf-8") as output:
        for shard_index in range(shard_count):
            expected_shard = min(args.questions_per_shard, expected - shard_index * args.questions_per_shard)
            paths = shard_paths(args.output_root, dataset, args.split, shard_index)
            if not valid_complete(paths, expected_shard):
                raise RuntimeError(f"Incomplete trace shard: {paths['root']}")
            for row in read_shard_rows(paths["rows"]):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                progress.update()
        output.flush()
        os.fsync(output.fileno())
    if written != expected:
        raise RuntimeError(f"Materialized count mismatch for {dataset}: {written} != {expected}")
    os.replace(temporary, target)
    atomic_write_json(
        target_dir / "manifest.json",
        {
            "type": "rag2_no_rag_rationale_artifact",
            "run_version": RUN_VERSION,
            "trace_version": TRACE_VERSION,
            "prompt_profile": "paper_compatible_three_anchor",
            "prompt_version": PROMPT_VERSION,
            "ppl_scope_version": "generated_rationale_v1",
            "generation_policy_version": GENERATION_POLICY_VERSION,
            "dataset": dataset,
            "split": args.split,
            "rows": written,
            "output_path": str(target.resolve()),
            "retrieval_query_policy": "complete_rationale_plus_fixed_terminal_answer_v1",
            "created_at": utc_now(),
        },
    )
    return target


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.questions_per_shard <= 0 or args.generation_batch_size <= 0:
        raise ValueError("Shard and batch sizes must be positive")
    sources = {dataset: source_path(args, dataset) for dataset in args.datasets}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = {dataset: count_jsonl(path) for dataset, path in sources.items()}
    total = sum(counts.values())
    logging.info("No-RAG plan: datasets=%s questions=%d", counts, total)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed_generation = 0
    for dataset in args.datasets:
        shard_count = math.ceil(counts[dataset] / args.questions_per_shard)
        for shard_index in range(shard_count):
            expected = min(
                args.questions_per_shard,
                counts[dataset] - shard_index * args.questions_per_shard,
            )
            if args.resume and valid_complete(
                shard_paths(args.output_root, dataset, args.split, shard_index), expected
            ):
                completed_generation += expected
    resources = None
    choice_token_ids: dict[str, int] = {}
    if completed_generation < total:
        resources = init_llm(args)
        choice_token_ids = resources[-1]
    else:
        previous_manifest = args.output_root / "generation_manifest.json"
        if previous_manifest.is_file():
            choice_token_ids = json.loads(
                previous_manifest.read_text(encoding="utf-8")
            ).get("choice_token_ids") or {}
        logging.info("All generation shards already complete; skipping vLLM load.")
    progress = PipelineProgress(
        overall_total=3 * total,
        overall_initial=completed_generation,
        desc="AnchoredNoRAG",
    )
    try:
        progress.set_stage(
            "1/3 no-RAG rationale+answer generation",
            total=total,
            initial=completed_generation,
        )
        for dataset in args.datasets:
            rows_iter = normalized_source_rows(sources[dataset], dataset, args.split)
            for shard_index, shard_rows in enumerate(stream_chunks(rows_iter, args.questions_per_shard)):
                paths = shard_paths(args.output_root, dataset, args.split, shard_index)
                if args.resume and valid_complete(paths, len(shard_rows)):
                    continue
                if resources is None:
                    raise RuntimeError("Missing vLLM resources for an incomplete generation shard")
                tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, _ = resources
                paths["root"].mkdir(parents=True, exist_ok=True)
                specs = [
                    {"kind": "no_document", "dataset": dataset, "sample_id": row["sample_id"], "row": row, "document": None}
                    for row in shard_rows
                ]
                traces = generate_specs(
                    args, tokenizer, llm, rationale_sampling, retry_sampling,
                    choice_sampling, choice_token_ids, specs,
                )
                output_rows = [artifact_row(trace, source) for trace, source in zip(traces, shard_rows)]
                atomic_write_jsonl(paths["rows"], output_rows)
                atomic_write_json(
                    paths["complete"],
                    {
                        "run_version": RUN_VERSION,
                        "completed_at": utc_now(),
                        "dataset": dataset,
                        "split": args.split,
                        "shard_index": shard_index,
                        "question_count": len(output_rows),
                        "valid_count": sum(bool(row["valid"]) for row in output_rows),
                    },
                )
                progress.update(len(output_rows))
        progress.set_stage("2/3 artifact materialization", total=total)
        artifacts = {
            dataset: str(materialize_dataset(args, dataset, counts[dataset], progress).resolve())
            for dataset in args.datasets
        }
        atomic_write_json(
            args.output_root / "generation_manifest.json",
            {
                "run_version": RUN_VERSION,
                "trace_version": TRACE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "generation_policy_version": GENERATION_POLICY_VERSION,
                "created_at": utc_now(),
                "model_name_or_path": str(args.model_name_or_path.resolve()),
                "datasets": counts,
                "split": args.split,
                "total_questions": total,
                "questions_per_shard": args.questions_per_shard,
                "choice_token_ids": choice_token_ids,
                "artifacts": artifacts,
                "next_stage": "extract_rag2_anchored_no_rag_features.py",
            },
        )
        logging.info("Generation and materialization complete: %s", args.output_root)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

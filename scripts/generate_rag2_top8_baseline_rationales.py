#!/usr/bin/env python3
"""Cache unbiased Top-8 anchored rationales for attention-controller training.

Each question is generated once with all eight reranked documents and no
attention intervention.  The resulting rationale is frozen during controller
training so the learned gate is optimized and evaluated at the same final
choice query.  Outputs are atomic question shards and safely resumable.
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
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_document_traces import normalized_candidate_rows  # noqa: E402
from generate_rag2_anchored_layer_pilot import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    generate_specs,
    init_llm,
)
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    GENERATION_POLICY_VERSION,
    PROMPT_VERSION,
    TRACE_VERSION,
)


RUN_VERSION = "rag2_top8_unbiased_rationale_cache_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "candidates/source_balanced32_rerank8_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-file", default="candidates_top8.jsonl")
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions-per-shard", type=int, default=128)
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
    parser.add_argument("--max-questions", type=int, default=0)
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


def chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalized_document_texts(row: dict[str, Any]) -> list[str]:
    return [" ".join(str(document["text"]).split()) for document in row["documents"]]


def shard_paths(output_dir: Path, index: int) -> tuple[Path, Path]:
    root = output_dir / "rationale_shards" / f"shard_{index:05d}"
    return root / "questions.jsonl", root / "COMPLETE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_identity(root: Path) -> list[dict[str, Any]]:
    names = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    paths = [root / name for name in names if (root / name).is_file()]
    paths.extend(sorted(root.glob("*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No model/tokenizer artifacts under {root}")
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            **({"sha256": sha256_file(path)} if path.stat().st_size < 16 * 1024 * 1024 else {}),
        }
        for path in paths
    ]


def valid_complete(
    rows_path: Path,
    marker_path: Path,
    expected_questions: int,
    fingerprint: str,
) -> bool:
    if not rows_path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and marker.get("contract_fingerprint") == fingerprint
        and int(marker.get("question_count", -1)) == expected_questions
        and int(marker.get("rows_size_bytes", -1)) == rows_path.stat().st_size
        and marker.get("rows_sha256") == sha256_file(rows_path)
    )


def generation_manifest_matches(
    path: Path,
    run_contract: dict[str, Any],
    shard_count: int,
) -> bool:
    """Return whether a completed manifest already represents this cache.

    ``completed_at`` is intentionally ignored.  Rewriting an otherwise
    identical manifest on every resume changes its file hash/mtime and makes
    downstream immutable feature caches appear stale.
    """

    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        all(manifest.get(key) == value for key, value in run_contract.items())
        and int(manifest.get("rationale_shards", -1)) == shard_count
    )


def count_candidates(path: Path, maximum: int) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
            if maximum > 0 and count >= maximum:
                break
    return count


def contract(
    args: argparse.Namespace,
    candidate_path: Path,
    candidate_manifest_path: Path,
    total: int,
) -> dict[str, Any]:
    stat = candidate_path.stat()
    return {
        "run_version": RUN_VERSION,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "candidate_path": str(candidate_path.resolve()),
        "candidate_size_bytes": stat.st_size,
        "candidate_mtime_ns": stat.st_mtime_ns,
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "model_bundle_identity": bundle_identity(args.model_name_or_path),
        "docs_per_question": args.docs_per_question,
        "question_count": total,
        "questions_per_shard": args.questions_per_shard,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_doc_chars": args.max_doc_chars,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.docs_per_question <= 0 or args.questions_per_shard <= 0:
        raise ValueError("Document and shard counts must be positive")
    if args.max_doc_chars != 0:
        raise ValueError("Top-8 controller rationales require --max-doc-chars 0 (no combined-context truncation)")
    candidate_path = args.candidate_root / args.dataset / args.split / args.candidate_file
    candidate_manifest_path = candidate_path.with_name("candidate_manifest.json")
    for required in (candidate_path, candidate_manifest_path, args.model_name_or_path):
        if not required.exists():
            raise FileNotFoundError(required)
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "type": "rag2_filter_candidate_dataset",
        "dataset": args.dataset,
        "split": args.split,
        "candidate_layout": "source_balanced",
        "top_k": args.docs_per_question,
        "query_prompt_version": PROMPT_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": candidate_manifest.get(key)}
        for key, value in expected_manifest.items()
        if candidate_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Top-8 candidate manifest mismatch: {mismatches}")
    total = count_candidates(candidate_path, args.max_questions)
    if total <= 0:
        raise ValueError("Candidate file contains no selected questions")
    manifest_questions = int(candidate_manifest.get("selected_question_count", -1))
    if manifest_questions <= 0 or (args.max_questions <= 0 and total != manifest_questions):
        raise RuntimeError(
            f"Candidate row count/manifest mismatch: rows={total} manifest={manifest_questions}"
        )
    run_contract = contract(args, candidate_path, candidate_manifest_path, total)
    fingerprint = hashlib.sha256(
        json.dumps(run_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_contract = {**run_contract, "contract_fingerprint": fingerprint}
    logging.info(
        "Top-8 rationale plan: dataset=%s questions=%d output=%s",
        args.dataset,
        total,
        args.output_dir,
    )
    if args.dry_run:
        return
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("--no-resume requires an empty or new rationale output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "generation_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError("Rationale cache resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(contract_path, run_contract)

    shard_count = math.ceil(total / args.questions_per_shard)
    expected_shard_roots = {
        args.output_dir / "rationale_shards" / f"shard_{index:05d}"
        for index in range(shard_count)
    }
    actual_shard_roots = set((args.output_dir / "rationale_shards").glob("shard_*"))
    extras = actual_shard_roots - expected_shard_roots
    if extras:
        raise RuntimeError(f"Unexpected stale rationale shards: {sorted(map(str, extras))[:5]}")
    completed = 0
    for shard_index in range(shard_count):
        expected = min(args.questions_per_shard, total - shard_index * args.questions_per_shard)
        rows_path, marker_path = shard_paths(args.output_dir, shard_index)
        if args.resume and valid_complete(rows_path, marker_path, expected, fingerprint):
            completed += expected
    logging.info(
        "Top-8 rationale plan: dataset=%s questions=%d cached=%d remaining=%d",
        args.dataset,
        total,
        completed,
        total - completed,
    )
    resources = init_llm(args) if completed < total else None
    progress = PipelineProgress(
        overall_total=2 * total,
        overall_initial=2 * completed,
        desc=f"Top8Rationale:{args.dataset}",
    )
    try:
        progress.set_stage(
            "1/1 unbiased Top-8 rationale + constrained choice",
            total=2 * total,
            initial=2 * completed,
        )
        row_iterator = normalized_candidate_rows(
            candidate_path,
            args.dataset,
            args.split,
            args.docs_per_question,
        )
        observed = 0
        for shard_index, rows in enumerate(chunks(row_iterator, args.questions_per_shard)):
            if observed >= total:
                break
            if observed + len(rows) > total:
                rows = rows[: total - observed]
            observed += len(rows)
            rows_path, marker_path = shard_paths(args.output_dir, shard_index)
            if args.resume and valid_complete(rows_path, marker_path, len(rows), fingerprint):
                continue
            if resources is None:
                raise RuntimeError("Missing vLLM resources for an incomplete rationale shard")
            progress.set_detail(f"shard={shard_index + 1}/{shard_count}")
            specs: list[dict[str, Any]] = []
            document_ids: dict[str, list[str]] = {}
            for row in rows:
                texts = normalized_document_texts(row)
                ids = [str(document["pair_id"]) for document in row["documents"]]
                document_ids[row["sample_id"]] = ids
                specs.append(
                    {
                        "kind": "top8_unbiased",
                        "dataset": args.dataset,
                        "sample_id": row["sample_id"],
                        "pair_id": None,
                        "row": row,
                        "document": {"text": "\n\n".join(texts)},
                    }
                )
            tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, choice_ids = resources
            traces = generate_specs(
                args,
                tokenizer,
                llm,
                rationale_sampling,
                retry_sampling,
                choice_sampling,
                choice_ids,
                specs,
            )
            compact_rows: list[dict[str, Any]] = []
            for trace in traces:
                compact_rows.append(
                    {
                        "run_version": RUN_VERSION,
                        "dataset": trace["dataset"],
                        "split": args.split,
                        "sample_id": trace["sample_id"],
                        "document_pair_ids": document_ids[trace["sample_id"]],
                        "rationale": trace["rationale"],
                        "answer": trace["answer"],
                        "answer_correct": trace["answer_correct"],
                        "quality_flags": trace["quality_flags"],
                        "canonical_response": trace["canonical_response"],
                        "choice_logprobs": trace["choice_logprobs"],
                    }
                )
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_jsonl(rows_path, compact_rows)
            atomic_write_json(
                marker_path,
                {
                    "run_version": RUN_VERSION,
                    "contract_fingerprint": fingerprint,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "question_count": len(compact_rows),
                    "rows_size_bytes": rows_path.stat().st_size,
                    "rows_sha256": sha256_file(rows_path),
                    "sample_ids_sha256": hashlib.sha256(
                        "\n".join(row["sample_id"] for row in compact_rows).encode()
                    ).hexdigest(),
                },
            )
            progress.update(2 * len(compact_rows))
        if observed != total:
            raise RuntimeError(f"Candidate coverage mismatch: observed={observed} expected={total}")
        manifest_path = args.output_dir / "generation_manifest.json"
        if completed == total and generation_manifest_matches(
            manifest_path,
            run_contract,
            shard_count,
        ):
            logging.info("Complete rationale manifest is unchanged; preserving cache identity")
        else:
            atomic_write_json(
                manifest_path,
                {
                    **run_contract,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "rationale_shards": shard_count,
                },
            )
        logging.info("Top-8 unbiased rationale cache complete: %s", args.output_dir)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

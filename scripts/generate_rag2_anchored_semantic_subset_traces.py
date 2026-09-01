#!/usr/bin/env python3
"""Generate anchored rationale+answer traces for planned document subsets.

This stage consumes a pre-materialized semantic subset plan.  Every planned
subset contains at least two reranked documents.  Their bodies are presented
to Llama in the original rerank order, separated only by blank lines; semantic
labels, ranks, sources, and policy names are never exposed in the prompt.

The model first generates a free rationale and then emits exactly one
constrained A/B/C/D token after the fixed ``Final answer: (`` anchor.  There is
no direct-choice generation mode in this script.  Outputs are atomically
question-sharded, strictly contract-keyed, and safely resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_document_traces import (  # noqa: E402
    normalized_candidate_rows,
)
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
    rationale_generation_prompt,
)

RUN_VERSION = "rag2_anchored_semantic_subset_rationale_generation_v2"
SUBSET_PROMPT_LAYOUT_VERSION = "ordered_document_bodies_blank_line_v1"
PPL_SCOPE_VERSION = "generated_rationale_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)
DEFAULT_PLAN_ROOT = DEFAULT_DATA_ROOT / "semantic_subset_rationale_traces_v1/subset_plan"
DEFAULT_CANDIDATE_ROOT = DEFAULT_DATA_ROOT / "candidates/source_balanced32_rerank8_v1"
SUPPORTED_DATASETS = ("medmcqa", "medqa")
CHOICES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=["medmcqa", "medqa"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-file", default="candidates_top8.jsonl")
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, required=True)
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
        choices=["balanced", "interactivity", "throughput"],
        default="throughput",
    )
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=0,
        help=(
            "Per-document character cap applied before concatenation. Zero preserves each retrieved body "
            "without truncation. Combined-context truncation is never applied."
        ),
    )
    parser.add_argument(
        "--context-safety-tokens",
        type=int,
        default=32,
        help="Reserved tokens beyond the rationale retry budget for the fixed terminal anchor and choice.",
    )
    parser.add_argument(
        "--preflight-tokenizer-batch-size",
        type=int,
        default=128,
        help="Question batch size for the full tokenizer-only Top-8 context audit.",
    )
    parser.add_argument(
        "--estimated-output-bytes-per-subset",
        type=int,
        default=12_288,
        help="Conservative trace-size estimate used by the disk preflight.",
    )
    parser.add_argument(
        "--disk-reserve-gib",
        type=float,
        default=20.0,
        help="Free-space reserve retained after projected output and atomic-shard headroom.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate plan/candidate alignment and print work counts without loading Llama or writing outputs.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Write/validate the immutable contract, run the full cached tokenizer context audit and disk "
            "preflight, then stop before loading Llama."
        ),
    )
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, hash_content: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        result["sha256"] = sha256_file(path)
    return result


def bundle_identity(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = [
        root / name
        for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
        if (root / name).is_file()
    ]
    paths.extend(sorted(root.glob("*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No local model/tokenizer artifacts under {root}")
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            **({"sha256": sha256_file(path)} if path.stat().st_size < 16 * 1024 * 1024 else {}),
        }
        for path in paths
    ]


def stream_chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def plan_paths(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    root = args.plan_root / dataset / args.split
    return root / "subset_plan.jsonl", root / "subset_plan_manifest.json"


def candidate_paths(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    root = args.candidate_root / dataset / args.split
    return root / args.candidate_file, root / "candidate_manifest.json"


def shard_paths(root: Path, dataset: str, split: str, shard_index: int) -> dict[str, Path]:
    base = root / "trace_shards" / dataset / split / f"shard_{shard_index:05d}"
    return {
        "root": base,
        "rows": base / "subsets.jsonl",
        "complete": base / "COMPLETE.json",
    }


def truncate_document_body(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if not value:
        raise ValueError("Encountered an empty candidate document body")
    if max_chars > 0 and len(value) > max_chars:
        return value[: max(0, max_chars - 3)].rstrip() + "..."
    return value


def subset_mask(subset: dict[str, Any], docs_per_question: int) -> tuple[int, list[int]]:
    raw_mask = subset.get("document_mask", subset.get("mask"))
    raw_ranks = subset.get("document_ranks", subset.get("doc_ranks"))
    mask = int(raw_mask) if raw_mask is not None else 0
    ranks = [int(value) for value in (raw_ranks or [])]
    if not ranks and mask:
        ranks = [rank for rank in range(1, docs_per_question + 1) if mask & (1 << (rank - 1))]
    if not mask and ranks:
        mask = sum(1 << (rank - 1) for rank in ranks)
    if ranks != sorted(set(ranks)):
        raise ValueError(f"Subset document ranks must be unique and sorted: {ranks}")
    if any(rank < 1 or rank > docs_per_question for rank in ranks):
        raise ValueError(f"Subset document rank outside 1..{docs_per_question}: {ranks}")
    expected_mask = sum(1 << (rank - 1) for rank in ranks)
    if mask != expected_mask:
        raise ValueError(f"Subset mask/rank mismatch: mask={mask} ranks={ranks}")
    if len(ranks) < 2:
        raise ValueError(f"Generation plan must contain only multi-document subsets: {ranks}")
    return mask, ranks


def normalized_subset_rows(
    path: Path,
    dataset: str,
    split: str,
    docs_per_question: int,
) -> Iterator[dict[str, Any]]:
    seen_samples: set[str] = set()
    for line_index, raw in enumerate(iter_jsonl(path)):
        actual_dataset = str(raw.get("dataset") or dataset)
        actual_split = str(raw.get("source_split") or raw.get("split") or split)
        if actual_dataset != dataset or actual_split != split:
            raise ValueError(
                f"Subset plan scope mismatch at line {line_index + 1}: "
                f"{actual_dataset}/{actual_split} != {dataset}/{split}"
            )
        sample_id = str(raw.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"Missing sample_id in subset plan line {line_index + 1}")
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate subset-plan sample_id: {sample_id}")
        seen_samples.add(sample_id)
        normalized_subsets: list[dict[str, Any]] = []
        seen_masks: set[int] = set()
        seen_subset_ids: set[str] = set()
        for subset_index, original in enumerate(raw.get("generation_subsets") or []):
            subset = dict(original)
            mask, ranks = subset_mask(subset, docs_per_question)
            subset_id = str(subset.get("subset_id") or f"{sample_id}::subset::{mask:02x}")
            if mask in seen_masks or subset_id in seen_subset_ids:
                raise ValueError(f"Duplicate planned subset for {sample_id}: id={subset_id} mask={mask}")
            seen_masks.add(mask)
            seen_subset_ids.add(subset_id)
            subset["subset_id"] = subset_id
            subset["document_mask"] = mask
            subset["document_ranks"] = ranks
            subset["plan_subset_index"] = subset_index
            normalized_subsets.append(subset)
        yield {
            **raw,
            "dataset": dataset,
            "split": split,
            "sample_id": sample_id,
            "row_idx": int(raw.get("row_idx", line_index)),
            "subsets": normalized_subsets,
        }


def validate_candidate_manifest(args: argparse.Namespace, dataset: str, path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "type": "rag2_filter_candidate_dataset",
        "dataset": dataset,
        "split": args.split,
        "candidate_layout": "source_balanced",
        "top_k": args.docs_per_question,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate contract mismatch for {dataset}: {mismatches}")
    return manifest


def validate_plan_manifest(args: argparse.Namespace, dataset: str, path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (("dataset", dataset), ("source_split", args.split)):
        if key in manifest and manifest[key] != expected:
            raise ValueError(
                f"Subset-plan manifest mismatch for {dataset}: {key}={manifest[key]!r} != {expected!r}"
            )
    if manifest.get("run_version") != "rag2_semantic_subset_plan_v1":
        raise ValueError(
            f"Unsupported subset-plan run_version for {dataset}: {manifest.get('run_version')!r}"
        )
    if manifest.get("policy_version") != "rag2_semantic_structured_subset_policies_v1":
        raise ValueError(
            f"Unsupported subset-plan policy_version for {dataset}: {manifest.get('policy_version')!r}"
        )
    top_k = manifest.get("contract", {}).get("top_k", manifest.get("docs_per_question"))
    if top_k is not None and int(top_k) != args.docs_per_question:
        raise ValueError(
            f"Subset-plan top_k mismatch: {top_k} != {args.docs_per_question}"
        )
    return manifest


def validate_plan_file_identity(plan_path: Path, manifest: dict[str, Any]) -> None:
    identity = manifest.get("plan")
    if not isinstance(identity, dict):
        raise ValueError(f"Subset-plan manifest has no immutable plan identity: {plan_path}")
    expected_size = int(identity.get("size_bytes", -1))
    expected_sha256 = str(identity.get("sha256") or "")
    actual_size = plan_path.stat().st_size
    if expected_size != actual_size:
        raise RuntimeError(
            f"Subset-plan size mismatch: path={plan_path} actual={actual_size} expected={expected_size}"
        )
    actual_sha256 = sha256_file(plan_path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Subset-plan SHA-256 mismatch: path={plan_path} actual={actual_sha256} "
            f"expected={expected_sha256 or '<missing>'}"
        )


def plan_manifest_question_count(manifest: dict[str, Any]) -> int | None:
    for key in ("question_count", "selected_question_count", "total_questions"):
        if key in manifest:
            return int(manifest[key])
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        for key in ("questions", "question_count", "total_questions"):
            if key in counts:
                return int(counts[key])
    return None


def plan_manifest_subset_count(manifest: dict[str, Any]) -> int | None:
    for key in (
        "unique_multi_document_subsets",
        "generated_subset_count",
        "multi_document_subset_count",
        "total_subsets",
        "subset_count",
    ):
        if key in manifest:
            return int(manifest[key])
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        for key in ("generated_subsets", "multi_document_subsets", "subsets", "subset_count"):
            if key in counts:
                return int(counts[key])
    return None


def scan_plan(
    args: argparse.Namespace,
    dataset: str,
    path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    question_count = 0
    subset_count = 0
    shard_subset_counts: list[int] = []
    shard_question_counts: list[int] = []
    shard_sample_hashes: list[str] = []
    for chunk in stream_chunks(
        normalized_subset_rows(path, dataset, args.split, args.docs_per_question),
        args.questions_per_shard,
    ):
        question_count += len(chunk)
        current_subsets = sum(len(row["subsets"]) for row in chunk)
        subset_count += current_subsets
        shard_subset_counts.append(current_subsets)
        shard_question_counts.append(len(chunk))
        shard_sample_hashes.append(sha256_bytes("\n".join(row["sample_id"] for row in chunk).encode("utf-8")))
    expected_questions = plan_manifest_question_count(manifest)
    expected_subsets = plan_manifest_subset_count(manifest)
    if expected_questions is not None and question_count != expected_questions:
        raise RuntimeError(
            f"Subset-plan question count mismatch for {dataset}: {question_count} != {expected_questions}"
        )
    if expected_subsets is not None and subset_count != expected_subsets:
        raise RuntimeError(
            f"Subset-plan generation count mismatch for {dataset}: {subset_count} != {expected_subsets}"
        )
    if question_count <= 0 or subset_count <= 0:
        raise ValueError(f"Subset plan for {dataset} contains no generation work")
    return {
        "question_count": question_count,
        "subset_count": subset_count,
        "shard_question_counts": shard_question_counts,
        "shard_subset_counts": shard_subset_counts,
        "shard_sample_hashes": shard_sample_hashes,
    }


def candidate_doc_stable_id(document: dict[str, Any]) -> str:
    return str(
        document.get("stable_id")
        or document.get("corpus_id")
        or document.get("db_id")
        or f"{document.get('source')}:{document.get('local_id')}"
    )


def validate_subset_against_candidate(
    plan_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> None:
    if plan_row["sample_id"] != candidate_row["sample_id"]:
        raise RuntimeError(
            f"Plan/candidate lockstep mismatch: {plan_row['sample_id']} != {candidate_row['sample_id']}"
        )
    documents = candidate_row["documents"]
    for subset in plan_row["subsets"]:
        selected = [documents[rank - 1] for rank in subset["document_ranks"]]
        expected_pair_ids = subset.get("document_pair_ids", subset.get("pair_ids"))
        if expected_pair_ids is not None:
            actual = [str(document["pair_id"]) for document in selected]
            if [str(value) for value in expected_pair_ids] != actual:
                raise RuntimeError(
                    f"Subset pair identity mismatch for {subset['subset_id']}: "
                    f"{expected_pair_ids} != {actual}"
                )
        expected_stable_ids = subset.get("document_stable_ids", subset.get("doc_stable_ids"))
        if expected_stable_ids is not None:
            actual = [candidate_doc_stable_id(document) for document in selected]
            if [str(value) for value in expected_stable_ids] != actual:
                raise RuntimeError(
                    f"Subset document identity mismatch for {subset['subset_id']}: "
                    f"{expected_stable_ids} != {actual}"
                )


def validate_plan_candidate_alignment(
    args: argparse.Namespace,
    dataset: str,
    plan_path: Path,
    candidate_path: Path,
    expected_questions: int,
) -> None:
    plans = normalized_subset_rows(plan_path, dataset, args.split, args.docs_per_question)
    candidates = normalized_candidate_rows(
        candidate_path,
        dataset,
        args.split,
        args.docs_per_question,
    )
    observed = 0
    sentinel = object()
    for plan_row, candidate_row in zip_longest(plans, candidates, fillvalue=sentinel):
        if plan_row is sentinel:
            # Bounded plans may intentionally use a candidate prefix.  Their exact
            # selected count remains part of the immutable plan manifest.
            break
        if candidate_row is sentinel:
            raise RuntimeError(f"Candidate file ended before subset plan for {dataset}")
        validate_subset_against_candidate(plan_row, candidate_row)
        observed += 1
    if observed != expected_questions:
        raise RuntimeError(
            f"Plan/candidate preflight coverage mismatch for {dataset}: {observed} != {expected_questions}"
        )


def immutable_contract(
    args: argparse.Namespace,
    sources: dict[str, dict[str, Any]],
    plan_stats: dict[str, dict[str, Any]],
    model_bundle: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "subset_prompt_layout_version": SUBSET_PROMPT_LAYOUT_VERSION,
        "ppl_scope_version": PPL_SCOPE_VERSION,
        "generation_mode": "rationale_then_fixed_constrained_terminal_choice_only",
        "direct_choice_generation": False,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "model_bundle_identity": model_bundle,
        "datasets": list(args.datasets),
        "split": args.split,
        "docs_per_question": args.docs_per_question,
        "minimum_documents_per_generated_subset": 2,
        "document_order": "ascending_rerank_rank",
        "document_separator": "\\n\\n",
        "document_metadata_exposed_to_prompt": False,
        "questions_per_shard": args.questions_per_shard,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_document_chars": args.max_document_chars,
        "combined_context_truncation": False,
        "llm_max_model_len": args.llm_max_model_len,
        "context_safety_tokens": args.context_safety_tokens,
        "source_contracts": sources,
        "counts": {
            dataset: {
                "questions": plan_stats[dataset]["question_count"],
                "subsets": plan_stats[dataset]["subset_count"],
            }
            for dataset in args.datasets
        },
    }


def valid_complete(
    paths: dict[str, Path],
    *,
    fingerprint: str,
    expected_questions: int,
    expected_subsets: int,
    expected_sample_hash: str,
) -> bool:
    if not paths["rows"].is_file() or not paths["complete"].is_file():
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and marker.get("contract_fingerprint") == fingerprint
        and int(marker.get("question_count", -1)) == expected_questions
        and int(marker.get("subset_count", -1)) == expected_subsets
        and marker.get("sample_ids_sha256") == expected_sample_hash
        and int(marker.get("rows_size_bytes", -1)) == paths["rows"].stat().st_size
        and marker.get("rows_sha256") == sha256_file(paths["rows"])
    )


def render_subset_document_text(
    candidate_row: dict[str, Any],
    subset: dict[str, Any],
    max_document_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    documents = candidate_row["documents"]
    selected = [documents[rank - 1] for rank in subset["document_ranks"]]
    bodies = [truncate_document_body(document["text"], max_document_chars) for document in selected]
    metadata = [
        {
            "rerank_rank": int(document.get("rerank_rank") or rank),
            "pair_id": str(document["pair_id"]),
            "stable_id": candidate_doc_stable_id(document),
            "source": str(document.get("source") or ""),
            "text_sha256": sha256_bytes(body.encode("utf-8")),
            "text_chars": len(body),
        }
        for rank, document, body in zip(subset["document_ranks"], selected, bodies)
    ]
    return "\n\n".join(bodies), metadata


def build_specs(
    args: argparse.Namespace,
    plan_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for plan_row, candidate_row in zip(plan_rows, candidate_rows):
        validate_subset_against_candidate(plan_row, candidate_row)
        for subset in plan_row["subsets"]:
            text, document_metadata = render_subset_document_text(
                candidate_row,
                subset,
                args.max_document_chars,
            )
            subset_id = subset["subset_id"]
            if subset_id in metadata:
                raise RuntimeError(f"Duplicate subset_id within shard: {subset_id}")
            specs.append(
                {
                    "kind": "with_document_subset",
                    "dataset": plan_row["dataset"],
                    "sample_id": plan_row["sample_id"],
                    "pair_id": subset_id,
                    "row": candidate_row,
                    # generate_specs receives only this joined body.  No labels,
                    # ranks, sources, or policy metadata enter the model prompt.
                    "document": {"text": text},
                }
            )
            semantic_labels = list(subset.get("semantic_labels") or [])
            metadata[subset_id] = {
                "dataset": plan_row["dataset"],
                "split": plan_row["split"],
                "sample_id": plan_row["sample_id"],
                "row_idx": int(plan_row["row_idx"]),
                "analysis_split": plan_row.get("analysis_split"),
                "subset_id": subset_id,
                "document_mask": int(subset["document_mask"]),
                "document_ranks": list(subset["document_ranks"]),
                "document_count": len(subset["document_ranks"]),
                "policies": list(subset.get("policies") or subset.get("policy_names") or []),
                "semantic_labels": semantic_labels,
                "semantic_counts": dict(
                    subset.get("semantic_counts") or Counter(semantic_labels)
                ),
                "documents": document_metadata,
                "document_text_sha256": sha256_bytes(text.encode("utf-8")),
            }
    return specs, metadata


def context_audit_path(output_root: Path, dataset: str) -> Path:
    return output_root / "preflight" / "context_audit" / f"{dataset}.json"


def context_audit_contract(
    args: argparse.Namespace,
    dataset: str,
    source: dict[str, Any],
    question_count: int,
    model_bundle: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "audit_version": "rag2_semantic_subset_full_context_audit_v1",
        "dataset": dataset,
        "split": args.split,
        "question_count": question_count,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "subset_prompt_layout_version": SUBSET_PROMPT_LAYOUT_VERSION,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "model_bundle_identity": model_bundle,
        "candidate_contract": source["candidate"],
        "candidate_manifest_contract": source["candidate_manifest"],
        "plan_contract": source["plan"],
        "plan_manifest_contract": source["plan_manifest"],
        "docs_per_question": args.docs_per_question,
        "max_document_chars": args.max_document_chars,
        "combined_context_truncation": False,
        "llm_max_model_len": args.llm_max_model_len,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "context_safety_tokens": args.context_safety_tokens,
    }


def valid_context_audit(
    path: Path,
    contract_sha256: str,
    expected_questions: int,
    prompt_token_limit: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("audit_version") != "rag2_semantic_subset_full_context_audit_v1"
        or value.get("contract_sha256") != contract_sha256
        or int(value.get("question_count", -1)) != expected_questions
        or int(value.get("prompt_token_limit", -1)) != prompt_token_limit
        or not bool(value.get("passed"))
        or int(value.get("overflow_count", -1)) != 0
        or int(value.get("max_prompt_tokens", prompt_token_limit + 1)) > prompt_token_limit
    ):
        return None
    return value


def percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def iter_audit_candidate_rows(
    args: argparse.Namespace,
    dataset: str,
    plan_path: Path,
    candidate_path: Path,
    expected_questions: int,
) -> Iterator[dict[str, Any]]:
    plans = normalized_subset_rows(plan_path, dataset, args.split, args.docs_per_question)
    candidates = normalized_candidate_rows(
        candidate_path,
        dataset,
        args.split,
        args.docs_per_question,
    )
    for index in range(expected_questions):
        try:
            plan_row = next(plans)
            candidate_row = next(candidates)
        except StopIteration as error:  # pragma: no cover - guarded by structural preflight
            raise RuntimeError(
                f"Context-audit source ended early for {dataset}: {index}/{expected_questions}"
            ) from error
        validate_subset_against_candidate(plan_row, candidate_row)
        yield candidate_row


def run_full_context_audit(
    args: argparse.Namespace,
    sources: dict[str, dict[str, Any]],
    plan_stats: dict[str, dict[str, Any]],
    model_bundle: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Audit every selected Top-8 prompt before any Llama weights are loaded."""

    prompt_token_limit = (
        args.llm_max_model_len - args.retry_max_new_tokens - args.context_safety_tokens
    )
    if prompt_token_limit <= 0:
        raise ValueError("Rationale retry budget and safety reserve exhaust --llm-max-model-len")
    contracts = {
        dataset: context_audit_contract(
            args,
            dataset,
            sources[dataset],
            plan_stats[dataset]["question_count"],
            model_bundle,
        )
        for dataset in args.datasets
    }
    fingerprints = {dataset: sha256_json(contracts[dataset]) for dataset in args.datasets}
    audits: dict[str, dict[str, Any]] = {}
    cached_questions = 0
    for dataset in args.datasets:
        cached = valid_context_audit(
            context_audit_path(args.output_root, dataset),
            fingerprints[dataset],
            plan_stats[dataset]["question_count"],
            prompt_token_limit,
        )
        if cached is not None:
            audits[dataset] = cached
            cached_questions += plan_stats[dataset]["question_count"]

    total_questions = sum(plan_stats[dataset]["question_count"] for dataset in args.datasets)
    logging.info(
        "Full context audit: questions=%d cached=%d remaining=%d prompt_token_limit=%d "
        "(model_len=%d retry=%d safety=%d); tokenizer only, Llama weights not loaded",
        total_questions,
        cached_questions,
        total_questions - cached_questions,
        prompt_token_limit,
        args.llm_max_model_len,
        args.retry_max_new_tokens,
        args.context_safety_tokens,
    )
    tokenizer = None
    if cached_questions < total_questions:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, trust_remote_code=True
        )
    progress = PipelineProgress(
        overall_total=total_questions,
        overall_initial=cached_questions,
        desc="SemanticSubsetContextAudit",
    )
    try:
        progress.set_stage(
            "tokenizer-only full Top-8 context audit before Llama load",
            total=total_questions,
            initial=cached_questions,
        )
        for dataset in args.datasets:
            if dataset in audits:
                progress.set_detail(
                    f"dataset={dataset} cached={plan_stats[dataset]['question_count']}"
                )
                continue
            if tokenizer is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Tokenizer was not initialized for an uncached context audit")
            plan_path, _ = plan_paths(args, dataset)
            candidate_path, _ = candidate_paths(args, dataset)
            token_lengths: list[int] = []
            max_sample_id = ""
            max_tokens = -1
            overflow_samples: list[dict[str, Any]] = []
            rows = iter_audit_candidate_rows(
                args,
                dataset,
                plan_path,
                candidate_path,
                plan_stats[dataset]["question_count"],
            )
            processed = 0
            for batch in stream_chunks(rows, args.preflight_tokenizer_batch_size):
                prompts: list[str] = []
                for row in batch:
                    all_documents = {
                        "document_ranks": list(range(1, len(row["documents"]) + 1))
                    }
                    text, _ = render_subset_document_text(
                        row,
                        all_documents,
                        args.max_document_chars,
                    )
                    prompts.append(rationale_generation_prompt(tokenizer, row, text))
                encoded = tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )
                lengths = list(encoded.get("length") or [])
                if len(lengths) != len(batch):
                    lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
                for row, length in zip(batch, lengths):
                    value = int(length)
                    token_lengths.append(value)
                    if value > max_tokens:
                        max_tokens = value
                        max_sample_id = row["sample_id"]
                    if value > prompt_token_limit and len(overflow_samples) < 20:
                        overflow_samples.append(
                            {"sample_id": row["sample_id"], "prompt_tokens": value}
                        )
                processed += len(batch)
                progress.set_detail(
                    f"dataset={dataset} questions={processed}/{plan_stats[dataset]['question_count']} "
                    f"max_prompt_tokens={max_tokens}/{prompt_token_limit}"
                )
                progress.update(len(batch))
            if processed != plan_stats[dataset]["question_count"]:
                raise RuntimeError(
                    f"Context-audit coverage mismatch for {dataset}: {processed} != "
                    f"{plan_stats[dataset]['question_count']}"
                )
            overflow_count = sum(value > prompt_token_limit for value in token_lengths)
            audit = {
                "audit_version": "rag2_semantic_subset_full_context_audit_v1",
                "created_at": utc_now(),
                "contract": contracts[dataset],
                "contract_sha256": fingerprints[dataset],
                "question_count": processed,
                "prompt_token_limit": prompt_token_limit,
                "max_prompt_tokens": max_tokens,
                "max_prompt_sample_id": max_sample_id,
                "mean_prompt_tokens": sum(token_lengths) / len(token_lengths),
                "p50_prompt_tokens": percentile(token_lengths, 0.50),
                "p95_prompt_tokens": percentile(token_lengths, 0.95),
                "p99_prompt_tokens": percentile(token_lengths, 0.99),
                "overflow_count": overflow_count,
                "overflow_examples": overflow_samples,
                "passed": overflow_count == 0,
            }
            if overflow_count:
                raise RuntimeError(
                    f"Full context audit failed for {dataset}: {overflow_count}/{processed} prompts exceed "
                    f"{prompt_token_limit} tokens; max={max_tokens} sample={max_sample_id}. "
                    "No Llama weights were loaded. Use a new output directory and a bounded "
                    "--max-document-chars value."
                )
            atomic_write_json(context_audit_path(args.output_root, dataset), audit)
            audits[dataset] = audit
            logging.info(
                "Context audit complete: dataset=%s questions=%d max=%d p95=%d limit=%d cache=%s",
                dataset,
                processed,
                max_tokens,
                audit["p95_prompt_tokens"],
                prompt_token_limit,
                context_audit_path(args.output_root, dataset),
            )
    finally:
        progress.close()
    return audits


def disk_preflight(
    args: argparse.Namespace,
    *,
    total_subsets: int,
    completed_subsets: int,
    completed_output_bytes: int,
    largest_remaining_shard_subsets: int,
) -> dict[str, Any]:
    remaining_subsets = max(0, total_subsets - completed_subsets)
    observed_bytes_per_subset = (
        completed_output_bytes / completed_subsets if completed_subsets else 0.0
    )
    effective_bytes_per_subset = max(
        args.estimated_output_bytes_per_subset,
        int(math.ceil(observed_bytes_per_subset * 1.25)),
    )
    projected_output = remaining_subsets * effective_bytes_per_subset
    atomic_headroom = largest_remaining_shard_subsets * effective_bytes_per_subset
    reserve = int(args.disk_reserve_gib * 1024**3)
    required_free = projected_output + atomic_headroom + reserve
    disk = shutil.disk_usage(args.output_root)
    result = {
        "remaining_subsets": remaining_subsets,
        "configured_bytes_per_subset": args.estimated_output_bytes_per_subset,
        "observed_bytes_per_subset": observed_bytes_per_subset,
        "effective_bytes_per_subset": effective_bytes_per_subset,
        "projected_remaining_output_bytes": projected_output,
        "atomic_largest_shard_headroom_bytes": atomic_headroom,
        "reserve_bytes": reserve,
        "required_free_bytes": required_free,
        "available_free_bytes": disk.free,
        "passed": disk.free >= required_free,
    }
    logging.info(
        "Disk preflight: remaining_subsets=%d effective=%.1f KiB/subset projected=%.2f GiB "
        "atomic_headroom=%.2f GiB reserve=%.2f GiB required=%.2f GiB free=%.2f GiB passed=%s",
        remaining_subsets,
        effective_bytes_per_subset / 1024,
        projected_output / 1024**3,
        atomic_headroom / 1024**3,
        reserve / 1024**3,
        required_free / 1024**3,
        disk.free / 1024**3,
        result["passed"],
    )
    if not result["passed"]:
        raise RuntimeError(
            f"Insufficient disk before Llama load: required={required_free / 1024**3:.2f} GiB "
            f"free={disk.free / 1024**3:.2f} GiB. This includes projected traces, one atomic "
            "largest-shard copy, and the configured reserve."
        )
    return result


def compact_trace(trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    flags = set(str(value) for value in (trace.get("quality_flags") or []))
    choice_logprobs = dict(trace.get("choice_logprobs") or {})
    bad_choices: list[str] = []
    for choice in CHOICES:
        value = choice_logprobs.get(choice)
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            bad_choices.append(choice)
    if bad_choices:
        flags.add("missing_or_nonfinite_choice_logprobs")
    ordered_flags = sorted(flags)
    return {
        "run_version": RUN_VERSION,
        "trace_version": trace.get("trace_version"),
        "prompt_version": trace.get("prompt_version"),
        "generation_policy_version": trace.get("generation_policy_version"),
        "subset_prompt_layout_version": SUBSET_PROMPT_LAYOUT_VERSION,
        "ppl_scope_version": PPL_SCOPE_VERSION,
        "kind": "with_document_subset",
        **metadata,
        "model_raw_rationale": trace.get("model_raw_rationale"),
        "rationale": trace.get("rationale"),
        "answer": trace.get("answer"),
        "answer_text": trace.get("answer_text"),
        "gold_answer": trace.get("gold_answer"),
        "answer_correct": bool(trace.get("answer_correct")),
        "canonical_response": trace.get("canonical_response"),
        "quality_flags": ordered_flags,
        "valid_for_subset_analysis": not ordered_flags,
        "rationale_finish_reason": trace.get("rationale_finish_reason"),
        "rationale_stop_reason": trace.get("rationale_stop_reason"),
        "rationale_token_ids": list(trace.get("rationale_token_ids") or []),
        "rationale_stats": dict(trace.get("rationale_stats") or {}),
        "choice_token_id": trace.get("choice_token_id"),
        "choice_logprobs": choice_logprobs,
        "choice_logprob_invalid_labels": bad_choices,
        "user_prompt_sha256": trace.get("user_prompt_sha256"),
        "rendered_rationale_prompt_sha256": trace.get("rendered_rationale_prompt_sha256"),
    }


def generation_manifest_matches(
    path: Path,
    contract: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    previous.pop("completed_at", None)
    expected = {**contract, **summary}
    return previous == expected


def tokenizer_choice_ids(model_path: Path) -> dict[str, int]:
    """Recover the fixed A/B/C/D token contract without loading model weights."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    result: dict[str, int] = {}
    for choice in CHOICES:
        token_ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"Choice {choice!r} is not one tokenizer token: {token_ids}")
        result[choice] = int(token_ids[0])
    return result


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.questions_per_shard <= 0 or args.generation_batch_size <= 0:
        raise ValueError("Shard and generation batch sizes must be positive")
    if args.preflight_tokenizer_batch_size <= 0:
        raise ValueError("--preflight-tokenizer-batch-size must be positive")
    if args.docs_per_question < 2:
        raise ValueError("--docs-per-question must be at least two")
    if args.max_document_chars < 0 or args.context_safety_tokens < 0:
        raise ValueError("Document character limit and context safety reserve cannot be negative")
    if args.max_new_tokens <= 0 or args.retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("Retry token limit must be at least the primary token limit")
    if args.estimated_output_bytes_per_subset <= 0 or args.disk_reserve_gib < 0:
        raise ValueError("Disk estimate must be positive and reserve cannot be negative")
    if args.dry_run and args.preflight_only:
        raise ValueError("Choose either --dry-run or --preflight-only, not both")

    # Artifact inspection is part of preflight but does not load Llama weights.
    model_bundle = bundle_identity(args.model_name_or_path)
    sources: dict[str, dict[str, Any]] = {}
    plan_stats: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        plan_path, plan_manifest_path = plan_paths(args, dataset)
        candidate_path, candidate_manifest_path = candidate_paths(args, dataset)
        for required in (plan_path, plan_manifest_path, candidate_path, candidate_manifest_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        plan_manifest = validate_plan_manifest(args, dataset, plan_manifest_path)
        validate_plan_file_identity(plan_path, plan_manifest)
        candidate_manifest = validate_candidate_manifest(args, dataset, candidate_manifest_path)
        plan_stats[dataset] = scan_plan(args, dataset, plan_path, plan_manifest)
        validate_plan_candidate_alignment(
            args,
            dataset,
            plan_path,
            candidate_path,
            plan_stats[dataset]["question_count"],
        )
        sources[dataset] = {
            "plan": file_identity(plan_path),
            "plan_manifest": file_identity(plan_manifest_path, hash_content=True),
            "plan_contract_sha256": plan_manifest.get("contract_sha256"),
            "candidate": file_identity(candidate_path),
            "candidate_manifest": file_identity(candidate_manifest_path, hash_content=True),
            "candidate_selected_question_count": candidate_manifest.get("selected_question_count"),
            "candidate_selected_pair_count": candidate_manifest.get("selected_pair_count"),
        }

    total_questions = sum(stats["question_count"] for stats in plan_stats.values())
    total_subsets = sum(stats["subset_count"] for stats in plan_stats.values())
    logging.info(
        "Semantic-subset rationale generation preflight complete: questions=%s subsets=%s "
        "total_questions=%d total_subsets=%d generation_mode=rationale+fixed_terminal_only",
        {dataset: plan_stats[dataset]["question_count"] for dataset in args.datasets},
        {dataset: plan_stats[dataset]["subset_count"] for dataset in args.datasets},
        total_questions,
        total_subsets,
    )
    if args.dry_run:
        logging.info("Dry-run complete; Llama was not loaded and no output was written.")
        return

    contract = immutable_contract(args, sources, plan_stats, model_bundle)
    fingerprint = sha256_json(contract)
    contract = {**contract, "contract_fingerprint": fingerprint}
    if not args.resume and args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("--no-resume requires an empty or new output directory")
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / "generation_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                f"Semantic-subset rationale resume contract mismatch; use a new output directory: {contract_path}"
            )
    else:
        atomic_write_json(contract_path, contract)

    completed_subsets = 0
    completed_output_bytes = 0
    largest_remaining_shard_subsets = 0
    expected_shard_roots: set[Path] = set()
    for dataset in args.datasets:
        stats = plan_stats[dataset]
        for shard_index, (question_count, subset_count, sample_hash) in enumerate(
            zip(
                stats["shard_question_counts"],
                stats["shard_subset_counts"],
                stats["shard_sample_hashes"],
            )
        ):
            paths = shard_paths(args.output_root, dataset, args.split, shard_index)
            expected_shard_roots.add(paths["root"])
            if args.resume and valid_complete(
                paths,
                fingerprint=fingerprint,
                expected_questions=question_count,
                expected_subsets=subset_count,
                expected_sample_hash=sample_hash,
            ):
                completed_subsets += subset_count
                completed_output_bytes += paths["rows"].stat().st_size
            else:
                largest_remaining_shard_subsets = max(
                    largest_remaining_shard_subsets,
                    subset_count,
                )
    actual_shard_roots = set((args.output_root / "trace_shards").glob("*/*/shard_*"))
    extras = actual_shard_roots - expected_shard_roots
    if extras:
        raise RuntimeError(f"Unexpected stale semantic-subset shards: {sorted(map(str, extras))[:5]}")
    logging.info(
        "Semantic-subset generation cache: total=%d cached=%d remaining=%d output=%s",
        total_subsets,
        completed_subsets,
        total_subsets - completed_subsets,
        args.output_root,
    )

    disk_report = disk_preflight(
        args,
        total_subsets=total_subsets,
        completed_subsets=completed_subsets,
        completed_output_bytes=completed_output_bytes,
        largest_remaining_shard_subsets=largest_remaining_shard_subsets,
    )
    context_audits = run_full_context_audit(args, sources, plan_stats, model_bundle)
    atomic_write_json(
        args.output_root / "preflight" / "preflight_summary.json",
        {
            "run_version": RUN_VERSION,
            "created_at": utc_now(),
            "generation_contract_fingerprint": fingerprint,
            "disk": disk_report,
            "context_audits": {
                dataset: {
                    "contract_sha256": context_audits[dataset]["contract_sha256"],
                    "question_count": context_audits[dataset]["question_count"],
                    "max_prompt_tokens": context_audits[dataset]["max_prompt_tokens"],
                    "prompt_token_limit": context_audits[dataset]["prompt_token_limit"],
                    "passed": context_audits[dataset]["passed"],
                }
                for dataset in args.datasets
            },
            "passed": True,
        },
    )
    if args.preflight_only:
        logging.info(
            "Preflight-only complete; full context audit is cached and Llama weights were not loaded: %s",
            args.output_root / "preflight",
        )
        return

    resources = init_llm(args) if completed_subsets < total_subsets else None
    choice_token_ids: dict[str, int] = resources[-1] if resources is not None else {}
    if resources is None:
        manifest_path = args.output_root / "generation_manifest.json"
        if manifest_path.is_file():
            choice_token_ids = dict(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("choice_token_ids") or {}
            )
        if set(choice_token_ids) != set(CHOICES):
            choice_token_ids = tokenizer_choice_ids(args.model_name_or_path)
        logging.info("All semantic-subset shards are complete; skipping vLLM load.")

    # generate_specs historically applies --max-doc-chars to its single
    # evidence string.  Per-document truncation has already been applied above;
    # disabling that second cap prevents accidental combined-context truncation.
    args.max_doc_chars = 0
    progress = PipelineProgress(
        overall_total=total_subsets,
        overall_initial=completed_subsets,
        desc="SemanticSubsetRationale",
    )
    last_durable_shard = "none"
    try:
        progress.set_stage(
            "1/1 multi-document rationale + fixed constrained terminal answer",
            total=total_subsets,
            initial=completed_subsets,
        )
        for dataset in args.datasets:
            plan_path, _ = plan_paths(args, dataset)
            candidate_path, _ = candidate_paths(args, dataset)
            plan_chunks = stream_chunks(
                normalized_subset_rows(plan_path, dataset, args.split, args.docs_per_question),
                args.questions_per_shard,
            )
            candidate_chunks = stream_chunks(
                normalized_candidate_rows(
                    candidate_path,
                    dataset,
                    args.split,
                    args.docs_per_question,
                ),
                args.questions_per_shard,
            )
            shard_count = len(plan_stats[dataset]["shard_question_counts"])
            observed_questions = 0
            observed_subsets = 0
            for shard_index, (plan_rows, candidate_rows) in enumerate(zip(plan_chunks, candidate_chunks)):
                expected_questions = plan_stats[dataset]["shard_question_counts"][shard_index]
                expected_subsets = plan_stats[dataset]["shard_subset_counts"][shard_index]
                sample_hash = plan_stats[dataset]["shard_sample_hashes"][shard_index]
                if len(plan_rows) != expected_questions or len(candidate_rows) < expected_questions:
                    raise RuntimeError(
                        f"Shard coverage mismatch for {dataset} shard={shard_index}: "
                        f"plan={len(plan_rows)} candidate={len(candidate_rows)} expected={expected_questions}"
                    )
                candidate_rows = candidate_rows[:expected_questions]
                observed_questions += len(plan_rows)
                observed_subsets += expected_subsets
                paths = shard_paths(args.output_root, dataset, args.split, shard_index)
                progress.set_detail(
                    f"dataset={dataset} shard={shard_index + 1}/{shard_count} "
                    f"subsets={expected_subsets}"
                )
                if args.resume and valid_complete(
                    paths,
                    fingerprint=fingerprint,
                    expected_questions=expected_questions,
                    expected_subsets=expected_subsets,
                    expected_sample_hash=sample_hash,
                ):
                    continue
                if resources is None:
                    raise RuntimeError("Missing vLLM resources for an incomplete subset shard")
                tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, _ = resources
                specs, metadata = build_specs(args, plan_rows, candidate_rows)
                if len(specs) != expected_subsets:
                    raise RuntimeError(
                        f"Prepared subset count mismatch: {len(specs)} != {expected_subsets}"
                    )
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
                if len(traces) != expected_subsets:
                    raise RuntimeError(
                        f"Generated subset count mismatch: {len(traces)} != {expected_subsets}"
                    )
                output_rows: list[dict[str, Any]] = []
                seen_subset_ids: set[str] = set()
                for trace in traces:
                    subset_id = str(trace.get("pair_id") or "")
                    if subset_id not in metadata or subset_id in seen_subset_ids:
                        raise RuntimeError(f"Unexpected or duplicate generated subset_id: {subset_id}")
                    seen_subset_ids.add(subset_id)
                    output_rows.append(compact_trace(trace, metadata[subset_id]))
                paths["root"].mkdir(parents=True, exist_ok=True)
                atomic_write_jsonl(paths["rows"], output_rows)
                quality_flags = Counter(
                    flag for row in output_rows for flag in (row.get("quality_flags") or [])
                )
                atomic_write_json(
                    paths["complete"],
                    {
                        "run_version": RUN_VERSION,
                        "contract_fingerprint": fingerprint,
                        "completed_at": utc_now(),
                        "dataset": dataset,
                        "split": args.split,
                        "shard_index": shard_index,
                        "question_count": expected_questions,
                        "subset_count": len(output_rows),
                        "valid_subset_count": sum(
                            bool(row["valid_for_subset_analysis"]) for row in output_rows
                        ),
                        "quality_flags": dict(quality_flags),
                        "sample_ids_sha256": sample_hash,
                        "rows_size_bytes": paths["rows"].stat().st_size,
                        "rows_sha256": sha256_file(paths["rows"]),
                    },
                )
                last_durable_shard = str(paths["complete"])
                progress.update(len(output_rows))
            if observed_questions != plan_stats[dataset]["question_count"]:
                raise RuntimeError(
                    f"Observed question mismatch for {dataset}: {observed_questions} != "
                    f"{plan_stats[dataset]['question_count']}"
                )
            if observed_subsets != plan_stats[dataset]["subset_count"]:
                raise RuntimeError(
                    f"Observed subset mismatch for {dataset}: {observed_subsets} != "
                    f"{plan_stats[dataset]['subset_count']}"
                )

        markers: list[dict[str, Any]] = []
        for dataset in args.datasets:
            root = args.output_root / "trace_shards" / dataset / args.split
            for marker_path in sorted(root.glob("shard_*/COMPLETE.json")):
                markers.append(json.loads(marker_path.read_text(encoding="utf-8")))
        summary = {
            "choice_token_ids": choice_token_ids,
            "total_questions": total_questions,
            "total_subsets": total_subsets,
            "shard_count": len(markers),
            "valid_subsets": sum(int(marker.get("valid_subset_count", 0)) for marker in markers),
            "quality_flags": dict(
                Counter(
                    {
                        flag: sum(
                            int((marker.get("quality_flags") or {}).get(flag, 0))
                            for marker in markers
                        )
                        for flag in {
                            name for marker in markers for name in (marker.get("quality_flags") or {})
                        }
                    }
                )
            ),
            "output_layout": "trace_shards/{dataset}/{split}/shard_N/subsets.jsonl",
        }
        manifest_path = args.output_root / "generation_manifest.json"
        if generation_manifest_matches(manifest_path, contract, summary):
            logging.info("Complete semantic-subset manifest is unchanged; preserving cache identity")
        else:
            atomic_write_json(manifest_path, {**contract, **summary, "completed_at": utc_now()})
        logging.info(
            "Semantic-subset rationale+answer generation complete: subsets=%d output=%s",
            total_subsets,
            args.output_root,
        )
    except BaseException:
        logging.exception(
            "Semantic-subset generation stopped: active_stage=%s detail=%s completed=%d/%d "
            "remaining=%d last_durable_complete=%s. Rerunning the same command with --resume is safe.",
            progress.stage,
            progress.detail,
            progress.stage_done,
            progress.stage_total,
            max(0, progress.stage_total - progress.stage_done),
            last_durable_shard,
        )
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

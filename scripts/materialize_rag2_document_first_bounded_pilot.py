#!/usr/bin/env python3
"""Materialize a bounded document-first direct-choice training cache.

The workflow deliberately reuses the immutable paper-compatible Top-8
retrieval/reranking candidates and semantic labels.  It recomputes only the
frozen Llama outcomes whose meaning changes when evidence is moved before the
question.  Gold answers are never rendered into prompts; they are used after
the forward pass to derive evaluation features.

One durable outcome row contains a No-RAG condition and all eight independent
single-document conditions.  A final question-level dataset joins semantic
labels, behavioral features, and a deterministic source/length-matched
cross-question support donor.  The latter is metadata for a later Dswap loss;
it is not scored in this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import (  # noqa: E402
    ExactDirectChoiceScorer,
    choice_probabilities,
    document_metadata,
    document_text,
    make_sample,
    summary_features,
    transition_label,
)
from evaluate_rag2_document_first_prompt_order import (  # noqa: E402
    HierarchicalProgress,
    sequence_for_order,
)
from generate_rag2_anchored_document_traces import normalized_candidate_rows  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_document_first_bounded_direct_outcomes_v1"
PROMPT_POLICY_VERSION = "rag2_paper_compatible_direct_choice_document_first_v1"
SPLITS = ("train", "val", "test")
NEGATIVE_LABELS = frozenset(("no_evidence", "misleading_evidence"))
POSITIVE_LABELS = frozenset(("direct_support",))
ALL_SEMANTIC_LABELS = frozenset(
    (
        "direct_support",
        "supporting_evidence",
        "no_evidence",
        "misleading_evidence",
        "indeterminate_or_mixed",
    )
)
BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), default="medmcqa")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=BASE / "candidates/source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=BASE / "filter_training_inputs_semantic_top8_four_class_v1",
    )
    parser.add_argument(
        "--raw-semantic-root",
        type=Path,
        default=Path(
            "/home/user/codex_rag2_outputs/"
            "codex_evidence_utility_labels_three_anchor_top8_terra_medium_v1_incremental/terra_medium"
        ),
        help="Complete five-label annotations, including indeterminate_or_mixed rows excluded from four-class training files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE / "document_first_bounded_direct_outcomes_v1",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument("--train-questions", type=int, default=20_000)
    parser.add_argument("--val-questions", type=int, default=4_000)
    parser.add_argument("--test-questions", type=int, default=4_000)
    parser.add_argument("--safety-questions", type=int, default=4_000)
    parser.add_argument("--questions-per-shard", type=int, default=256)
    parser.add_argument("--prompt-batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    parser.add_argument("--expected-prompts-per-second", type=float, default=70.0)
    parser.add_argument("--disk-reserve-gib", type=float, default=20.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        value["sha256"] = sha256_file(path)
    return value


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stable_priority(seed: int, split: str, sample_id: str) -> int:
    payload = f"{seed}\0{split}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class LowestHashes:
    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self.heap: list[tuple[int, str]] = []

    def add(self, priority: int, sample_id: str) -> None:
        if self.limit == 0:
            return
        item = (-int(priority), str(sample_id))
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def values(self) -> list[str]:
        return [item[1] for item in sorted(self.heap, key=lambda item: (-item[0], item[1]))]


def semantic_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {split: args.semantic_root / args.dataset / f"{split}.jsonl" for split in SPLITS}


def raw_semantic_path(args: argparse.Namespace) -> Path:
    return args.raw_semantic_root / args.dataset / "codex_semantic_labels.jsonl"


def load_semantic_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, int]]:
    path = args.semantic_root / args.dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != args.dataset or int(manifest.get("top_k", -1)) != 8:
        raise RuntimeError(f"Semantic manifest contract mismatch: {path}")
    split_manifest = manifest.get("materialized", {}).get("splits", {})
    counts = {split: int(split_manifest[split]["rows"]) for split in SPLITS}
    return path, manifest, counts


def candidate_contract(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    path = args.candidate_root / args.dataset / "train/candidates_top8.jsonl"
    manifest_path = path.with_name("candidate_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": args.dataset,
        "split": "train",
        "candidate_layout": "source_balanced",
        "per_source_top_k": 8,
        "candidate_pool_top_k": 32,
        "top_k": 8,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Candidate contract mismatch: {mismatches}")
    return path, manifest_path, manifest


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.output_root / args.dataset
    return {
        "root": root,
        "contract": root / "run_contract.json",
        "cohort": root / "cohort.jsonl",
        "cohort_manifest": root / "cohort_manifest.json",
        "candidates": root / "selected_candidates.jsonl",
        "candidate_manifest": root / "selected_candidate_manifest.json",
        "prompt_audit": root / "prompt_audit.json",
        "outcome_manifest": root / "outcome_manifest.json",
        "dataset_manifest": root / "training_dataset/manifest.json",
    }


def build_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, int]]:
    candidate_path, candidate_manifest_path, candidate_manifest = candidate_contract(args)
    semantic_manifest_path, _semantic_manifest, semantic_counts = load_semantic_manifest(args)
    counts = {
        "train": int(args.train_questions),
        "val": int(args.val_questions),
        "test": int(args.test_questions),
        "safety": int(args.safety_questions),
    }
    contract = {
        "run_version": RUN_VERSION,
        "hypothesis": (
            "Document-token-restricted adaptation can improve semantic-support use relative to a frozen "
            "document-first baseline without changing No-RAG behavior."
        ),
        "dataset": args.dataset,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "prompt_order": "documents_then_question_options_then_fixed_final_answer_prefix",
        "answer_mode": "exact_four_choice_next_token_logits",
        "candidate_file": file_identity(candidate_path),
        "candidate_manifest": file_identity(candidate_manifest_path, content_hash=True),
        "candidate_retrieval_contract": {
            key: candidate_manifest.get(key)
            for key in ("candidate_layout", "per_source_top_k", "candidate_pool_top_k", "top_k", "sources")
        },
        "semantic_manifest": file_identity(semantic_manifest_path, content_hash=True),
        "semantic_split_files": {
            split: file_identity(path) for split, path in semantic_paths(args).items()
        },
        "complete_five_label_semantics": file_identity(raw_semantic_path(args)),
        "semantic_split_rows": semantic_counts,
        "cohort_counts": counts,
        "eligible_definition": {
            "positive": sorted(POSITIVE_LABELS),
            "negative": sorted(NEGATIVE_LABELS),
            "train_val_test": "same question contains at least one positive and one negative Top-8 document",
            "safety": "random held-out internal-test question not selected for the mechanism test",
        },
        "selection_seed": int(args.seed),
        "model_path": str(args.model_name_or_path.resolve()),
        "model_config": file_identity(args.model_name_or_path / "config.json", content_hash=True),
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "max_input_tokens": int(args.max_input_tokens),
        "questions_per_shard": int(args.questions_per_shard),
        "raw_data_immutable": True,
        "external_final_test_used": False,
    }
    return contract, counts


def ensure_contract(args: argparse.Namespace, paths: dict[str, Path], contract: dict[str, Any]) -> str:
    paths["root"].mkdir(parents=True, exist_ok=True)
    contract_hash = fingerprint(contract)
    if paths["contract"].is_file():
        previous = json.loads(paths["contract"].read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract_hash or previous.get("contract") != contract:
            raise RuntimeError(
                "Document-first bounded cache contract mismatch; use a new --output-root. "
                "Existing durable artifacts were not modified."
            )
    else:
        atomic_json(
            paths["contract"],
            {
                "contract_sha256": contract_hash,
                "created_at": utc_now(),
                "code_commit": git_commit(),
                "code_sha256": sha256_file(Path(__file__)),
                "contract": contract,
            },
        )
    return contract_hash


def grouped_semantic_rows(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if current_id is None:
            current_id = sample_id
        if sample_id != current_id:
            if current_id in seen:
                raise RuntimeError(f"Semantic rows are not question-contiguous: {current_id}")
            seen.add(current_id)
            yield current_id, current
            current_id, current = sample_id, []
        current.append(row)
    if current_id is not None:
        if current_id in seen:
            raise RuntimeError(f"Semantic rows are not question-contiguous: {current_id}")
        yield current_id, current


def select_cohort(
    args: argparse.Namespace,
    counts: dict[str, int],
    paths: dict[str, Path],
    progress: HierarchicalProgress,
) -> list[dict[str, Any]]:
    if args.resume and paths["cohort"].is_file() and paths["cohort_manifest"].is_file():
        manifest = json.loads(paths["cohort_manifest"].read_text(encoding="utf-8"))
        rows = list(iter_jsonl(paths["cohort"]))
        if manifest.get("contract_sha256") == fingerprint(build_contract(args)[0]) and len(rows) == sum(counts.values()):
            progress.set_initial(sum(manifest["semantic_rows_scanned"].values()))
            return rows

    limits = {"train": counts["train"], "val": counts["val"], "test": counts["test"]}
    reservoirs = {split: LowestHashes(limits[split]) for split in SPLITS}
    safety_reservoir = LowestHashes(counts["safety"] + counts["test"])
    semantic_row_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    sample_rows: dict[str, dict[str, Any]] = {}
    scanned = 0
    for split, path in semantic_paths(args).items():
        split_rows = 0
        eligible = 0
        for sample_id, rows in grouped_semantic_rows(path):
            split_rows += len(rows)
            labels = {str(row["target"]) for row in rows}
            if not labels.issubset(ALL_SEMANTIC_LABELS):
                raise RuntimeError(f"Unexpected semantic labels for {sample_id}: {sorted(labels)}")
            first = rows[0]
            sample_rows[sample_id] = {
                "sample_id": sample_id,
                "row_idx": int(first["row_idx"]),
                "semantic_split": split,
            }
            priority = stable_priority(args.seed, split, sample_id)
            if labels.intersection(POSITIVE_LABELS) and labels.intersection(NEGATIVE_LABELS):
                reservoirs[split].add(priority, sample_id)
                eligible += 1
            if split == "test":
                safety_reservoir.add(stable_priority(args.seed + 1000, "safety", sample_id), sample_id)
            scanned += len(rows)
            if scanned % 8192 < len(rows):
                progress.set_absolute(scanned)
        semantic_row_counts[split] = split_rows
        eligible_counts[split] = eligible
    progress.set_absolute(scanned, force=True)

    selected = {split: reservoirs[split].values() for split in SPLITS}
    for split in SPLITS:
        if len(selected[split]) != limits[split]:
            raise RuntimeError(
                f"Not enough eligible {split} questions: requested={limits[split]} found={len(selected[split])}"
            )
    test_set = set(selected["test"])
    safety = [sample_id for sample_id in safety_reservoir.values() if sample_id not in test_set][: counts["safety"]]
    if len(safety) != counts["safety"]:
        raise RuntimeError(f"Not enough disjoint safety questions: requested={counts['safety']} found={len(safety)}")
    cohort = []
    for split in SPLITS:
        for sample_id in selected[split]:
            cohort.append({**sample_rows[sample_id], "cohort_split": split, "eligible_pair": True})
    for sample_id in safety:
        cohort.append({**sample_rows[sample_id], "cohort_split": "safety", "eligible_pair": False})
    cohort.sort(key=lambda row: (tuple((*SPLITS, "safety")).index(row["cohort_split"]), row["row_idx"], row["sample_id"]))
    atomic_jsonl(paths["cohort"], cohort)
    atomic_json(
        paths["cohort_manifest"],
        {
            "run_version": RUN_VERSION,
            "contract_sha256": fingerprint(build_contract(args)[0]),
            "created_at": utc_now(),
            "counts": dict(Counter(row["cohort_split"] for row in cohort)),
            "eligible_available": eligible_counts,
            "semantic_rows_scanned": semantic_row_counts,
            "cohort_sha256": sha256_file(paths["cohort"]),
        },
    )
    return cohort


def load_selected_semantics(
    args: argparse.Namespace,
    cohort: Sequence[dict[str, Any]],
    total_semantic_rows: int,
    progress: HierarchicalProgress,
) -> dict[str, dict[str, dict[str, Any]]]:
    wanted = {str(row["sample_id"]) for row in cohort}
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    scanned = 0
    for row in iter_jsonl(raw_semantic_path(args)):
        sample_id = str(row["sample_id"])
        if sample_id in wanted:
            pair_id = str(row["pair_id"])
            if pair_id in result[sample_id]:
                raise RuntimeError(f"Duplicate semantic pair: {pair_id}")
            label = str(row["semantic_label"])
            if label not in ALL_SEMANTIC_LABELS:
                raise RuntimeError(f"Unexpected semantic label for {pair_id}: {label}")
            result[sample_id][pair_id] = {
                "semantic_label": label,
                "semantic_confidence": float(row.get("confidence") or 0.0),
                "semantic_reason": str(row.get("short_reason") or ""),
            }
        scanned += 1
        if scanned % 8192 == 0:
            progress.set_absolute(scanned)
    progress.set_absolute(scanned, force=True)
    if scanned != total_semantic_rows:
        raise RuntimeError(f"Semantic row count mismatch: expected={total_semantic_rows} actual={scanned}")
    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(f"Selected questions missing semantic rows: count={len(missing)} first={missing[:3]}")
    incomplete = [(sample_id, len(values)) for sample_id, values in result.items() if len(values) != 8]
    if incomplete:
        raise RuntimeError(f"Selected questions do not have exactly eight semantic labels: first={incomplete[:3]}")
    return result


def compact_candidate(row: dict[str, Any], split: str, semantics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    documents = []
    for document in row["documents"]:
        pair_id = str(document["pair_id"])
        if pair_id not in semantics:
            raise RuntimeError(f"Missing selected semantic pair: {pair_id}")
        documents.append(
            {
                key: document.get(key)
                for key in (
                    "pair_id", "source", "local_id", "db_id", "corpus_id", "chunk_id", "doc_id",
                    "title", "text", "retrieval_score", "retrieval_rank", "rerank_score", "rerank_rank",
                    "stable_id", "source_retrieval_rank",
                )
            }
            | semantics[pair_id]
        )
    return {
        "sample_id": str(row["sample_id"]),
        "row_idx": int(row["row_idx"]),
        "dataset": str(row["dataset"]),
        "split": "bounded",
        "cohort_split": split,
        "question": str(row["question"]),
        "options": dict(row["options"]),
        "answers": list(row["answers"]),
        "answer": str(row["answer"]),
        "query_text": str(row.get("query_text") or ""),
        "retrieval_query_mode": str(row.get("retrieval_query_mode") or ""),
        "rerank_query_mode": str(row.get("rerank_query_mode") or ""),
        "candidate_documents": documents,
    }


def materialize_candidates(
    args: argparse.Namespace,
    cohort: Sequence[dict[str, Any]],
    semantics: dict[str, dict[str, dict[str, Any]]],
    source_path: Path,
    expected_source_rows: int,
    paths: dict[str, Path],
    contract_sha256: str,
    progress: HierarchicalProgress,
) -> list[dict[str, Any]]:
    if args.resume and paths["candidates"].is_file() and paths["candidate_manifest"].is_file():
        manifest = json.loads(paths["candidate_manifest"].read_text(encoding="utf-8"))
        rows = list(iter_jsonl(paths["candidates"]))
        if (
            manifest.get("contract_sha256") == contract_sha256
            and manifest.get("candidates_sha256") == sha256_file(paths["candidates"])
            and len(rows) == len(cohort)
        ):
            progress.set_initial(expected_source_rows)
            return rows

    split_by_id = {str(row["sample_id"]): str(row["cohort_split"]) for row in cohort}
    selected: list[dict[str, Any]] = []
    observed = 0
    for row in normalized_candidate_rows(source_path, args.dataset, "train", 8):
        sample_id = str(row["sample_id"])
        if sample_id in split_by_id:
            selected.append(compact_candidate(row, split_by_id[sample_id], semantics[sample_id]))
        observed += 1
        if observed % 4096 == 0:
            progress.set_absolute(observed)
    progress.set_absolute(observed, force=True)
    if observed != expected_source_rows:
        raise RuntimeError(f"Candidate source row mismatch: expected={expected_source_rows} actual={observed}")
    if len(selected) != len(cohort):
        raise RuntimeError(f"Selected candidate count mismatch: expected={len(cohort)} actual={len(selected)}")
    selected.sort(key=lambda row: (tuple((*SPLITS, "safety")).index(row["cohort_split"]), row["row_idx"], row["sample_id"]))
    atomic_jsonl(paths["candidates"], selected)
    atomic_json(
        paths["candidate_manifest"],
        {
            "run_version": RUN_VERSION,
            "contract_sha256": contract_sha256,
            "created_at": utc_now(),
            "selected_question_count": len(selected),
            "selected_pair_count": len(selected) * 8,
            "split_counts": dict(Counter(row["cohort_split"] for row in selected)),
            "candidate_layout": "source_balanced",
            "per_source_top_k": 8,
            "candidate_pool_top_k": 32,
            "top_k": 8,
            "candidates_sha256": sha256_file(paths["candidates"]),
        },
    )
    return selected


def prompt_preflight(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    paths: dict[str, Path],
    contract_sha256: str,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    if args.resume and paths["prompt_audit"].is_file():
        audit = json.loads(paths["prompt_audit"].read_text(encoding="utf-8"))
        if audit.get("contract_sha256") == contract_sha256 and int(audit.get("questions", -1)) == len(rows):
            progress.set_initial(len(rows))
            return audit

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
    )
    max_tokens = 0
    max_case: dict[str, Any] = {}
    token_counts: list[int] = []
    for index, row in enumerate(rows, 1):
        sample = make_sample(row)
        no_ids, _ = sequence_for_order(tokenizer, sample, None, "document_first")
        current = [("no_rag", 0, no_ids)]
        for document in row["candidate_documents"]:
            ids, _ = sequence_for_order(tokenizer, sample, document_text(document), "document_first")
            current.append(("single_document", int(document["rerank_rank"]), ids))
        for condition, rank, ids in current:
            count = len(ids)
            token_counts.append(count)
            if count > args.max_input_tokens:
                raise RuntimeError(
                    f"Document-first prompt exceeds max_input_tokens={args.max_input_tokens}: "
                    f"sample={sample.id} condition={condition} rank={rank} tokens={count}"
                )
            if count > max_tokens:
                max_tokens = count
                max_case = {"sample_id": sample.id, "condition": condition, "rerank_rank": rank, "tokens": count}
        if index % 128 == 0 or index == len(rows):
            progress.set_absolute(index)
    array = np.asarray(token_counts, dtype=np.int64)
    source_candidates = args.candidate_root / args.dataset / "train/candidates_top8.jsonl"
    source_manifest = json.loads(source_candidates.with_name("candidate_manifest.json").read_text(encoding="utf-8"))
    source_questions = int(source_manifest["selected_question_count"])
    # Selected candidates, outcome rows, and final training rows each retain
    # document text.  Four times the proportional source size plus score
    # metadata is a conservative bound for these durable copies.
    proportional_text_bytes = source_candidates.stat().st_size * len(rows) / source_questions
    projected_bytes = int(proportional_text_bytes * 4.0 + len(rows) * 9 * 2600)
    free = shutil.disk_usage(paths["root"]).free
    reserve = int(args.disk_reserve_gib * 1024**3)
    if free < projected_bytes + reserve:
        raise RuntimeError(
            f"Insufficient disk: free={free/1024**3:.2f}GiB projected={projected_bytes/1024**3:.2f}GiB "
            f"reserve={args.disk_reserve_gib:.2f}GiB"
        )
    audit = {
        "run_version": RUN_VERSION,
        "contract_sha256": contract_sha256,
        "created_at": utc_now(),
        "questions": len(rows),
        "prompts": len(token_counts),
        "max_prompt": max_case,
        "token_count": {
            "min": int(array.min()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": int(array.max()),
        },
        "projected_output_gib": projected_bytes / 1024**3,
        "free_gib": free / 1024**3,
        "reserve_gib": args.disk_reserve_gib,
    }
    atomic_json(paths["prompt_audit"], audit)
    return audit


def jsd(probability_a: np.ndarray, probability_b: np.ndarray) -> float:
    midpoint = 0.5 * (probability_a + probability_b)
    a = np.clip(probability_a.astype(np.float64), 1e-12, 1.0)
    b = np.clip(probability_b.astype(np.float64), 1e-12, 1.0)
    m = np.clip(midpoint.astype(np.float64), 1e-12, 1.0)
    return float(0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m)))


def condition_record(logits: np.ndarray, gold_index: int, prompt: str, token_count: int) -> dict[str, Any]:
    probabilities = choice_probabilities(logits[np.newaxis, :])[0]
    return {
        **summary_features(logits, gold_index),
        "choice_logits": [float(value) for value in logits],
        "choice_probabilities": [float(value) for value in probabilities],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count": int(token_count),
    }


def score_rows(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    paths: dict[str, Path],
    contract_sha256: str,
    progress: HierarchicalProgress,
) -> list[Path]:
    shard_root = paths["root"] / "outcome_shards"
    shards = [rows[start : start + args.questions_per_shard] for start in range(0, len(rows), args.questions_per_shard)]
    complete_paths: list[Path] = []
    cached_prompts = 0
    cached_indices: set[int] = set()
    for shard_index, shard in enumerate(shards):
        directory = shard_root / f"shard_{shard_index:05d}"
        data_path = directory / "outcomes.jsonl"
        marker_path = directory / "COMPLETE.json"
        if args.resume and data_path.is_file() and marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                marker = {}
            if (
                marker.get("contract_sha256") == contract_sha256
                and int(marker.get("question_count", -1)) == len(shard)
                and int(marker.get("prompt_count", -1)) == len(shard) * 9
                and int(marker.get("data_size_bytes", -1)) == data_path.stat().st_size
            ):
                cached_indices.add(shard_index)
                cached_prompts += len(shard) * 9
                complete_paths.append(data_path)
    progress.set_initial(cached_prompts)
    progress.log(
        f"[stage {progress.stage_index}/{progress.stage_count} resume] cached={cached_prompts}/{len(rows)*9} "
        f"remaining={len(rows)*9-cached_prompts} durable_cache={shard_root}"
    )
    if cached_prompts == len(rows) * 9:
        return complete_paths

    scorer = ExactDirectChoiceScorer(args)
    try:
        for shard_index, shard in enumerate(shards):
            if shard_index in cached_indices:
                continue
            shard_started = time.monotonic()
            sequences: list[list[int]] = []
            prompts: list[str] = []
            specs: list[tuple[int, int | None]] = []
            for question_index, row in enumerate(shard):
                sample = make_sample(row)
                ids, prompt = sequence_for_order(scorer.tokenizer, sample, None, "document_first")
                sequences.append(ids)
                prompts.append(prompt)
                specs.append((question_index, None))
                for document_index, document in enumerate(row["candidate_documents"]):
                    ids, prompt = sequence_for_order(
                        scorer.tokenizer, sample, document_text(document), "document_first"
                    )
                    sequences.append(ids)
                    prompts.append(prompt)
                    specs.append((question_index, document_index))
            # Score in visible micro-batches so the active-stage rate and ETA
            # keep updating while a durable question shard is in flight.
            logit_chunks = []
            visible_batch = max(1, int(args.prompt_batch_size))
            for batch_start in range(0, len(sequences), visible_batch):
                batch_sequences = sequences[batch_start : batch_start + visible_batch]
                logit_chunks.append(scorer.score(batch_sequences))
                progress.update(len(batch_sequences))
            logits = np.concatenate(logit_chunks, axis=0)
            by_question: list[dict[str, Any]] = [dict() for _ in shard]
            cursor = 0
            for question_index, row in enumerate(shard):
                gold_index = CHOICES.index(str(row["answer"]))
                no_rag = condition_record(logits[cursor], gold_index, prompts[cursor], len(sequences[cursor]))
                cursor += 1
                documents = []
                no_logits = np.asarray(no_rag["choice_logits"], dtype=np.float32)
                no_probabilities = np.asarray(no_rag["choice_probabilities"], dtype=np.float32)
                for document in row["candidate_documents"]:
                    current = condition_record(logits[cursor], gold_index, prompts[cursor], len(sequences[cursor]))
                    current_logits = np.asarray(current["choice_logits"], dtype=np.float32)
                    current_probabilities = np.asarray(current["choice_probabilities"], dtype=np.float32)
                    metadata = document_metadata(document, document_text(document))
                    metadata.update(
                        {
                            "semantic_label": str(document["semantic_label"]),
                            "semantic_confidence": float(document["semantic_confidence"]),
                            "semantic_reason": str(document.get("semantic_reason") or ""),
                            "document_text": document_text(document),
                            "title": str(document.get("title") or ""),
                        }
                    )
                    documents.append(
                        {
                            **current,
                            "delta_gold_probability": float(current["gold_probability"] - no_rag["gold_probability"]),
                            "delta_gold_margin": float(current["gold_margin"] - no_rag["gold_margin"]),
                            "delta_choice_logits": [float(value) for value in current_logits - no_logits],
                            "delta_choice_probabilities": [
                                float(value) for value in current_probabilities - no_probabilities
                            ],
                            "jsd_from_no_rag": jsd(current_probabilities, no_probabilities),
                            "prediction_changed_from_no_rag": current["prediction"] != no_rag["prediction"],
                            "correctness_transition": transition_label(
                                bool(no_rag["answer_correct"]), bool(current["answer_correct"])
                            ),
                            "document": metadata,
                        }
                    )
                    cursor += 1
                by_question[question_index] = {
                    "run_version": RUN_VERSION,
                    "prompt_policy_version": PROMPT_POLICY_VERSION,
                    "dataset": args.dataset,
                    "cohort_split": str(row["cohort_split"]),
                    "sample_id": str(row["sample_id"]),
                    "row_idx": int(row["row_idx"]),
                    "question": str(row["question"]),
                    "options": dict(row["options"]),
                    "gold_answer": str(row["answer"]),
                    "no_rag": no_rag,
                    "documents": documents,
                }
            if cursor != len(sequences):
                raise RuntimeError(f"Scoring cursor mismatch: {cursor} != {len(sequences)}")
            directory = shard_root / f"shard_{shard_index:05d}"
            data_path = directory / "outcomes.jsonl"
            marker_path = directory / "COMPLETE.json"
            atomic_jsonl(data_path, by_question)
            atomic_json(
                marker_path,
                {
                    "run_version": RUN_VERSION,
                    "contract_sha256": contract_sha256,
                    "created_at": utc_now(),
                    "shard_index": shard_index,
                    "question_count": len(shard),
                    "prompt_count": len(shard) * 9,
                    "data_size_bytes": data_path.stat().st_size,
                    "elapsed_seconds": time.monotonic() - shard_started,
                },
            )
            complete_paths.append(data_path)
    finally:
        scorer.close()
    complete_paths = [shard_root / f"shard_{index:05d}/outcomes.jsonl" for index in range(len(shards))]
    if not all(path.is_file() for path in complete_paths):
        raise RuntimeError("Some outcome shards are missing after scoring")
    atomic_json(
        paths["outcome_manifest"],
        {
            "run_version": RUN_VERSION,
            "contract_sha256": contract_sha256,
            "completed_at": utc_now(),
            "questions": len(rows),
            "single_document_pairs": len(rows) * 8,
            "prompts": len(rows) * 9,
            "shards": len(complete_paths),
            "stored_features": [
                "A/B/C/D raw logits and conditional probabilities",
                "prediction, correctness, entropy, top-1 probability/margin",
                "gold probability/margin/rank",
                "No-RAG deltas, JSD, prediction change, correctness transition",
                "semantic label/confidence/reason and retrieval/rerank/source metadata",
                "prompt hash and token count",
            ],
        },
    )
    return complete_paths


def document_word_count(document: dict[str, Any]) -> int:
    return len(str(document["document"]["document_text"]).split())


def semantic_positive(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in documents if row["document"]["semantic_label"] in POSITIVE_LABELS]


def semantic_negative(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in documents if row["document"]["semantic_label"] in NEGATIVE_LABELS]


def donor_map(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    # Donors never cross an internal split; otherwise a train prompt could
    # contain validation/test evidence even though it comes from another question.
    pools: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        positives = sorted(
            semantic_positive(row["documents"]),
            key=lambda value: (-float(value["document"]["semantic_confidence"]), int(value["document"]["rerank_rank"])),
        )
        if not positives:
            continue
        chosen = positives[0]
        source = str(chosen["document"]["source"])
        split = str(row["cohort_split"])
        pools[(split, source)].append((document_word_count(chosen), str(row["sample_id"]), chosen))
    result: dict[str, dict[str, Any]] = {}
    for (split, source), values in pools.items():
        values.sort(key=lambda item: (item[0], stable_priority(seed + 2000, source, item[1])))
        if len(values) < 2:
            continue
        for index, (_length, sample_id, _document) in enumerate(values):
            offsets = (1, -1, 2, -2)
            donor = None
            for offset in offsets:
                candidate = values[(index + offset) % len(values)]
                if candidate[1] != sample_id:
                    donor = candidate
                    break
            if donor is None:
                raise RuntimeError(f"Cannot find cross-question donor in split={split} source={source}")
            result[sample_id] = {
                "donor_sample_id": donor[1],
                "donor_split": split,
                "source": source,
                "recipient_primary_support_word_count": values[index][0],
                "donor_word_count": donor[0],
                "document": donor[2]["document"],
            }
    return result


def build_training_dataset(
    args: argparse.Namespace,
    outcome_paths: Sequence[Path],
    paths: dict[str, Path],
    contract_sha256: str,
    total_questions: int,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    output_dir = paths["root"] / "training_dataset"
    output_files = {split: output_dir / f"{split}.jsonl" for split in (*SPLITS, "safety")}
    if args.resume and paths["dataset_manifest"].is_file() and all(path.is_file() for path in output_files.values()):
        manifest = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
        if manifest.get("contract_sha256") == contract_sha256:
            progress.set_initial(total_questions)
            return manifest

    rows = [row for path in outcome_paths for row in iter_jsonl(path)]
    if len(rows) != total_questions or len({row["sample_id"] for row in rows}) != total_questions:
        raise RuntimeError("Outcome question cardinality or uniqueness mismatch")
    donors = donor_map(rows, args.seed)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summaries: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(rows, 1):
        positives = semantic_positive(row["documents"])
        negatives = semantic_negative(row["documents"])
        primary_positive = min(
            positives,
            key=lambda value: (
                -float(value["document"]["semantic_confidence"]),
                int(value["document"]["rerank_rank"]),
            ),
            default=None,
        )
        primary_negative = min(
            negatives,
            key=lambda value: (
                0 if value["document"]["semantic_label"] == "misleading_evidence" else 1,
                -float(value["document"]["semantic_confidence"]),
                int(value["document"]["rerank_rank"]),
            ),
            default=None,
        )
        hardest_positive = min(positives, key=lambda value: float(value["delta_gold_margin"]), default=None)
        hardest_negative = min(negatives, key=lambda value: float(value["delta_gold_margin"]), default=None)
        split = str(row["cohort_split"])
        if split != "safety" and (not positives or not negatives):
            raise RuntimeError(f"Eligible cohort lost a semantic side: {row['sample_id']}")
        prepared = {
            "run_version": RUN_VERSION,
            "dataset": row["dataset"],
            "split": split,
            "sample_id": row["sample_id"],
            "row_idx": row["row_idx"],
            "question": row["question"],
            "options": row["options"],
            "gold_answer": row["gold_answer"],
            "frozen_no_rag": row["no_rag"],
            "documents": row["documents"],
            "semantic_positive_pair_ids": [value["document"]["pair_id"] for value in positives],
            "semantic_negative_pair_ids": [value["document"]["pair_id"] for value in negatives],
            "semantic_primary_positive_pair_id": (
                primary_positive["document"]["pair_id"] if primary_positive else None
            ),
            "semantic_primary_negative_pair_id": (
                primary_negative["document"]["pair_id"] if primary_negative else None
            ),
            "behaviorally_underused_positive_pair_id": (
                hardest_positive["document"]["pair_id"] if hardest_positive else None
            ),
            "behaviorally_disruptive_negative_pair_id": (
                hardest_negative["document"]["pair_id"] if hardest_negative else None
            ),
            "cross_question_support_donor": donors.get(str(row["sample_id"])),
        }
        by_split[split].append(prepared)
        summaries[split]["questions"] += 1
        summaries[split]["documents"] += len(row["documents"])
        summaries[split]["positive_documents"] += len(positives)
        summaries[split]["negative_documents"] += len(negatives)
        summaries[split]["no_rag_correct"] += int(bool(row["no_rag"]["answer_correct"]))
        for document in row["documents"]:
            summaries[split][f"semantic_{document['document']['semantic_label']}"] += 1
            summaries[split][f"transition_{document['correctness_transition']}"] += 1
        if index % 64 == 0 or index == len(rows):
            progress.set_absolute(index)
    for split, output in output_files.items():
        atomic_jsonl(output, by_split[split])
    manifest = {
        "run_version": RUN_VERSION,
        "contract_sha256": contract_sha256,
        "completed_at": utc_now(),
        "purpose": "Question-level input for document-token-restricted semantic adaptation",
        "label_visibility": {
            "gold_and_semantic_labels_in_model_prompt": False,
            "gold_and_behavioral_features": "training selection, weighting, and evaluation only",
        },
        "split_summary": {split: dict(values) for split, values in summaries.items()},
        "files": {
            split: {**file_identity(path), "sha256": sha256_file(path)}
            for split, path in output_files.items()
        },
        "dswap": (
            "A source-matched, nearest-length Direct-Support document from another question is stored as a "
            "deterministic donor. It has not been scored on the recipient question."
        ),
        "next_stage": "tiny overfit test for document-token K/V adapters, then the preregistered bounded pilot",
    }
    atomic_json(paths["dataset_manifest"], manifest)
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if min(
        args.train_questions,
        args.val_questions,
        args.test_questions,
        args.safety_questions,
        args.questions_per_shard,
        args.prompt_batch_size,
        args.max_input_tokens,
    ) <= 0:
        raise ValueError("Question counts, shard/batch sizes, and max tokens must be positive")
    if args.expected_prompts_per_second <= 0:
        raise ValueError("--expected-prompts-per-second must be positive")

    contract, requested_counts = build_contract(args)
    paths = output_paths(args)
    contract_sha256 = ensure_contract(args, paths, contract)
    source_candidate_path, _candidate_manifest_path, candidate_manifest = candidate_contract(args)
    semantic_manifest_path, _semantic_manifest, semantic_counts = load_semantic_manifest(args)
    del semantic_manifest_path
    split_semantic_rows = sum(semantic_counts.values())
    total_semantic_rows = int(candidate_manifest["selected_question_count"]) * 8
    source_candidate_rows = int(candidate_manifest["selected_question_count"])
    selected_questions = sum(requested_counts.values())
    total_prompts = selected_questions * 9
    score_seconds = total_prompts / args.expected_prompts_per_second
    stage_names = (
        "select fixed semantic cohort",
        "join selected semantic labels",
        "materialize compact Top-8 candidates",
        "audit document-first prompt lengths",
        "score frozen No-RAG and eight single-document conditions",
        "materialize question-level training data",
    )
    stage_estimates = (75.0, 75.0, 150.0, 180.0, score_seconds, 150.0)
    progress = HierarchicalProgress(stage_names, stage_estimates)
    progress.log(
        "[workflow plan] "
        f"dataset={args.dataset} cohort={requested_counts} questions={selected_questions} prompts={total_prompts} "
        f"attention={args.attn_implementation} batch={args.prompt_batch_size} output={paths['root']}"
    )
    try:
        progress.start_stage(1, split_semantic_rows, "semantic-row")
        cohort = select_cohort(args, requested_counts, paths, progress)
        progress.complete_stage(f"selected={len(cohort)} cohort={paths['cohort']}")

        progress.start_stage(2, total_semantic_rows, "semantic-row")
        semantics = load_selected_semantics(args, cohort, total_semantic_rows, progress)
        progress.complete_stage(f"selected_questions={len(semantics)}")

        progress.start_stage(3, source_candidate_rows, "question")
        candidates = materialize_candidates(
            args,
            cohort,
            semantics,
            source_candidate_path,
            source_candidate_rows,
            paths,
            contract_sha256,
            progress,
        )
        progress.complete_stage(f"selected={len(candidates)} candidates={paths['candidates']}")
        del semantics

        progress.start_stage(4, len(candidates), "question")
        audit = prompt_preflight(args, candidates, paths, contract_sha256, progress)
        progress.complete_stage(
            f"prompts={audit['prompts']} max_tokens={audit['token_count']['max']} audit={paths['prompt_audit']}"
        )
        if args.preflight_only:
            progress.finish("preflight-only: no frozen Llama scoring was run")
            return

        progress.start_stage(5, total_prompts, "prompt")
        outcome_paths = score_rows(args, candidates, paths, contract_sha256, progress)
        progress.complete_stage(f"shards={len(outcome_paths)} manifest={paths['outcome_manifest']}")
        del candidates

        progress.start_stage(6, selected_questions, "question")
        manifest = build_training_dataset(
            args,
            outcome_paths,
            paths,
            contract_sha256,
            selected_questions,
            progress,
        )
        progress.complete_stage(f"dataset={paths['dataset_manifest'].parent}")
        progress.finish(
            f"training_data={paths['dataset_manifest'].parent} split_summary={manifest['split_summary']}"
        )
    except Exception:
        progress.log(
            f"[workflow FAILED] active_stage={progress.stage_index}/{progress.stage_count} "
            f"completed={progress.stage_done}/{progress.stage_total}; rerun the identical command to resume "
            f"from durable artifacts under {paths['root']}"
        )
        raise


if __name__ == "__main__":
    main()

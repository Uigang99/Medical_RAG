#!/usr/bin/env python3
"""Prepare and merge semantic labels for the exact dynamic Top-k union.

The original external semantic run annotated the global rerank Top-32 for each
question.  The paper-balanced sweep instead reconstructs a different 4k pool
for each k in {1,2,4,8,16,32}; a small number of documents selected only at a
smaller k therefore lie outside the global Top-32.  This utility reuses labels
by stable question-document identity, materializes only those missing pairs,
then merges and verifies the exact dynamic-k union.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm


DATASETS = (
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)
ANNOTATION_VERSION = "rag2_codex_evidence_utility_label_v2"
PROMPT_VERSION = "rag2_codex_evidence_utility_prompt_v3_compact_item_index"
PREPARATION_VERSION = "rag2_external_dynamic_topk_semantic_delta_prepare_v1"
MERGE_VERSION = "rag2_external_dynamic_topk_semantic_union_merge_v1"
VALID_LABELS = {
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or merge external dynamic-Top-k semantic label deltas.")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidate-union-root", type=Path, required=True)
    prepare.add_argument("--existing-label-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    prepare.add_argument("--max-documents-per-block", type=int, default=8)
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--prepared-root", type=Path, required=True)
    merge.add_argument("--new-label-root", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    merge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_document_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    value = str(value or "").strip()
    if not value:
        raise ValueError("Document has no stable ID")
    return value


def union_rank(document: dict[str, Any]) -> int:
    value = document.get("oracle_union_rank") or document.get("rerank_rank")
    try:
        rank = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid dynamic-union rank: {value!r}") from error
    if rank <= 0:
        raise ValueError(f"Invalid dynamic-union rank: {rank}")
    return rank


def semantic_key(dataset: str, sample_id: str, stable_id: str) -> tuple[str, str, str]:
    return dataset, sample_id, stable_id


def pair_id(sample_id: str, rank: int, stable_id: str) -> str:
    return f"{sample_id}::{rank}::{stable_id}"


def union_candidate_paths(root: Path, datasets: Iterable[str]) -> dict[str, Path]:
    paths = {dataset: root / dataset / "test" / "candidates_topk_union.jsonl" for dataset in datasets}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dynamic-union candidates: " + ", ".join(missing))
    return paths


def existing_label_paths(root: Path, datasets: Iterable[str]) -> dict[str, Path]:
    paths = {dataset: root / dataset / "codex_semantic_labels.jsonl" for dataset in datasets}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing existing semantic labels: " + ", ".join(missing))
    return paths


def validate_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "docs_per_question": 8,
        "allow_fewer_documents": False,
        "questions_per_batch": 10,
        "max_doc_chars": 0,
        "codex_bin": "codex",
        "codex_model_request": "gpt-5.6-terra",
        "codex_reasoning_effort": "medium",
        "web_search_enabled": False,
        "worker_count": 8,
    }
    mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Existing semantic-label contract mismatch: {mismatches}")
    return value


def validate_union_manifest(path: Path, datasets: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("type") != "rag2_paper_balanced_dynamic_oracle_candidate_union":
        raise ValueError(f"Unexpected dynamic-union manifest type: {value.get('type')}")
    if value.get("dynamic_top_k_values") != [1, 2, 4, 8, 16, 32]:
        raise ValueError(f"Unexpected dynamic Top-k contract: {value.get('dynamic_top_k_values')}")
    question_counts = value.get("questions_by_dataset")
    pair_counts = value.get("pairs_by_dataset")
    if not isinstance(question_counts, dict) or not isinstance(pair_counts, dict):
        raise ValueError("Dynamic-union manifest omits per-dataset counts")
    missing = [dataset for dataset in datasets if dataset not in question_counts or dataset not in pair_counts]
    if missing:
        raise ValueError(f"Dynamic-union manifest omits datasets: {missing}")
    return value


def output_files_match(manifest: dict[str, Any], output_root: Path, datasets: Iterable[str]) -> bool:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return False
    for dataset in datasets:
        for name in ("reused_labels", "pending_candidates"):
            path = output_root / name / f"{dataset}.jsonl"
            expected = outputs.get(dataset, {}).get(name)
            if not path.is_file() or not isinstance(expected, dict):
                return False
            actual = path_identity(path)
            if actual["size"] != expected.get("size") or actual["mtime_ns"] != expected.get("mtime_ns"):
                return False
    return True


def remap_label(
    label: dict[str, Any], dataset: str, sample_id: str, document: dict[str, Any], origin: str
) -> dict[str, Any]:
    rank = union_rank(document)
    stable_id = stable_document_id(document)
    semantic_label = str(label.get("semantic_label") or "")
    if semantic_label not in VALID_LABELS:
        raise ValueError(f"Invalid existing semantic label for {sample_id}/{stable_id}: {semantic_label}")
    return {
        **label,
        "id": pair_id(sample_id, rank, stable_id),
        "pair_id": pair_id(sample_id, rank, stable_id),
        "dataset": dataset,
        "sample_id": sample_id,
        "doc_rank": rank,
        "source": str(document.get("source") or label.get("source") or "unknown"),
        "doc_stable_id": stable_id,
        "title": str(document.get("title") or label.get("title") or ""),
        "dynamic_union_origin": origin,
        "master_rerank_rank": document.get("master_rerank_rank"),
        "dynamic_top_k_membership": document.get("metadata", {}).get("oracle_dynamic_top_k_membership"),
        "dynamic_rerank_rank_by_top_k": document.get("metadata", {}).get(
            "oracle_dynamic_rerank_rank_by_top_k"
        ),
    }


def pending_document(document: dict[str, Any]) -> dict[str, Any]:
    rank = union_rank(document)
    stable_id = stable_document_id(document)
    return {
        **document,
        "rerank_rank": rank,
        "oracle_union_rank": rank,
        "stable_id": stable_id,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_documents_per_block <= 0:
        raise ValueError("--max-documents-per-block must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate --datasets entry")
    union_manifest_path = args.candidate_union_root / "manifest.json"
    existing_manifest_path = args.existing_label_root / "manifest.json"
    union_manifest = validate_union_manifest(union_manifest_path, args.datasets)
    existing_manifest = validate_existing_manifest(existing_manifest_path)
    candidate_paths = union_candidate_paths(args.candidate_union_root, args.datasets)
    label_paths = existing_label_paths(args.existing_label_root, args.datasets)
    expected_existing_pairs = sum(int(existing_manifest["datasets"][dataset]["pairs"]) for dataset in args.datasets)
    expected_union_pairs = sum(int(union_manifest["pairs_by_dataset"][dataset]) for dataset in args.datasets)
    expected_union_questions = sum(
        int(union_manifest["questions_by_dataset"][dataset]) for dataset in args.datasets
    )
    contract = {
        "preparation_version": PREPARATION_VERSION,
        "candidate_union_manifest": path_identity(union_manifest_path),
        "existing_label_manifest": path_identity(existing_manifest_path),
        "candidate_files": {dataset: path_identity(path) for dataset, path in candidate_paths.items()},
        "existing_label_files": {dataset: path_identity(path) for dataset, path in label_paths.items()},
        "datasets": list(args.datasets),
        "max_documents_per_block": args.max_documents_per_block,
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
    }
    input_fingerprint = fingerprint(contract)
    manifest_path = args.output_root / "prepare_manifest.json"
    if args.resume and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            old.get("status") == "complete"
            and old.get("input_fingerprint") == input_fingerprint
            and output_files_match(old, args.output_root, args.datasets)
        ):
            logging.info(
                "Dynamic semantic delta already prepared: reused=%d pending=%d root=%s",
                old["totals"]["reused_pairs"],
                old["totals"]["pending_pairs"],
                args.output_root,
            )
            return old

    overall = tqdm(
        total=expected_existing_pairs + expected_union_pairs,
        desc="DynamicSemanticDeltaPrepareOverall",
        unit="pair",
        position=0,
        dynamic_ncols=True,
    )
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    stage = tqdm(
        total=expected_existing_pairs,
        desc="Stage 1/2 index existing Top-32 labels",
        unit="pair",
        position=1,
        dynamic_ncols=True,
    )
    for dataset, path in label_paths.items():
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            stable_id = str(row.get("doc_stable_id") or "")
            key = semantic_key(dataset, sample_id, stable_id)
            if not sample_id or not stable_id or key in existing:
                raise ValueError(f"Invalid or duplicate existing semantic pair: {key}")
            existing[key] = row
            stage.update(1)
            overall.update(1)
    stage.close()
    if len(existing) != expected_existing_pairs:
        raise ValueError(f"Existing semantic count mismatch: {len(existing)} != {expected_existing_pairs}")

    reused_dir = args.output_root / "reused_labels"
    pending_dir = args.output_root / "pending_candidates"
    reused_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)
    reused_temporary = {dataset: reused_dir / f".{dataset}.jsonl.tmp" for dataset in args.datasets}
    pending_temporary = {dataset: pending_dir / f".{dataset}.jsonl.tmp" for dataset in args.datasets}
    reused_handles = {dataset: path.open("w", encoding="utf-8") for dataset, path in reused_temporary.items()}
    pending_handles = {dataset: path.open("w", encoding="utf-8") for dataset, path in pending_temporary.items()}
    reused_keys: set[tuple[str, str, str]] = set()
    union_keys: set[tuple[str, str, str]] = set()
    dataset_stats: dict[str, Counter[str]] = defaultdict(Counter)
    stage = tqdm(
        total=expected_union_pairs,
        desc="Stage 2/2 diff exact dynamic Top-k union",
        unit="pair",
        position=1,
        dynamic_ncols=True,
    )
    try:
        for dataset, path in candidate_paths.items():
            observed_questions = 0
            for row in iter_jsonl(path):
                observed_questions += 1
                sample_id = str(row.get("sample_id") or "")
                if str(row.get("dataset") or "").lower() != dataset or not sample_id:
                    raise ValueError(f"Invalid dynamic-union row in {path}")
                documents = row.get("candidate_documents")
                if not isinstance(documents, list) or not documents:
                    raise ValueError(f"No dynamic-union documents for {sample_id}")
                ranks = [union_rank(document) for document in documents]
                if ranks != list(range(1, len(documents) + 1)):
                    raise ValueError(f"Non-contiguous dynamic-union ranks for {sample_id}: {ranks}")
                missing_documents: list[dict[str, Any]] = []
                for document in documents:
                    stable_id = stable_document_id(document)
                    key = semantic_key(dataset, sample_id, stable_id)
                    if key in union_keys:
                        raise ValueError(f"Duplicate dynamic-union semantic pair: {key}")
                    union_keys.add(key)
                    if key in existing:
                        output = remap_label(existing[key], dataset, sample_id, document, "reused_global_top32")
                        reused_handles[dataset].write(
                            json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                        reused_keys.add(key)
                        dataset_stats[dataset]["reused_pairs"] += 1
                    else:
                        missing_documents.append(pending_document(document))
                        dataset_stats[dataset]["pending_pairs"] += 1
                    dataset_stats[dataset]["union_pairs"] += 1
                    stage.update(1)
                    overall.update(1)
                for block_index, offset in enumerate(
                    range(0, len(missing_documents), args.max_documents_per_block)
                ):
                    block = missing_documents[offset : offset + args.max_documents_per_block]
                    pending_row = {
                        **{key: value for key, value in row.items() if key != "candidate_documents"},
                        "semantic_delta_block_index": block_index,
                        "semantic_delta_block_count": (
                            len(missing_documents) + args.max_documents_per_block - 1
                        )
                        // args.max_documents_per_block,
                        "candidate_documents": block,
                    }
                    pending_handles[dataset].write(
                        json.dumps(pending_row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    dataset_stats[dataset]["pending_transport_rows"] += 1
                dataset_stats[dataset]["questions"] += 1
            wanted_questions = int(union_manifest["questions_by_dataset"][dataset])
            if observed_questions != wanted_questions:
                raise ValueError(
                    f"Dynamic-union question count mismatch for {dataset}: {observed_questions} != {wanted_questions}"
                )
    finally:
        for handle in reused_handles.values():
            handle.close()
        for handle in pending_handles.values():
            handle.close()
        stage.close()
        overall.close()

    if len(union_keys) != expected_union_pairs:
        raise ValueError(f"Dynamic-union pair count mismatch: {len(union_keys)} != {expected_union_pairs}")
    if reused_keys != set(existing):
        extra = sorted(set(existing) - reused_keys)
        raise ValueError(f"Existing Top-32 labels outside dynamic union: {len(extra)}; first={extra[:1]}")
    pending_total = expected_union_pairs - len(reused_keys)
    if pending_total <= 0:
        raise ValueError("No missing dynamic-union semantic pairs were found")

    reused_paths: dict[str, Path] = {}
    pending_paths: dict[str, Path] = {}
    for dataset in args.datasets:
        reused_path = reused_dir / f"{dataset}.jsonl"
        pending_path = pending_dir / f"{dataset}.jsonl"
        os.replace(reused_temporary[dataset], reused_path)
        os.replace(pending_temporary[dataset], pending_path)
        reused_paths[dataset] = reused_path
        pending_paths[dataset] = pending_path
    manifest = {
        "preparation_version": PREPARATION_VERSION,
        "created_at": utc_now(),
        "status": "complete",
        "input_fingerprint": input_fingerprint,
        "input_contract": contract,
        "annotation_contract": {
            "annotation_version": ANNOTATION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "web_search_enabled": False,
            "max_doc_chars": 0,
            "workers": 8,
            "questions_per_batch": 10,
            "max_documents_per_transport_row": args.max_documents_per_block,
        },
        "totals": {
            "questions": expected_union_questions,
            "union_pairs": expected_union_pairs,
            "reused_pairs": len(reused_keys),
            "pending_pairs": pending_total,
            "pending_transport_rows": sum(
                dataset_stats[dataset]["pending_transport_rows"] for dataset in args.datasets
            ),
        },
        "datasets": {dataset: dict(dataset_stats[dataset]) for dataset in args.datasets},
        "outputs": {
            dataset: {
                "reused_labels": path_identity(reused_paths[dataset]),
                "pending_candidates": path_identity(pending_paths[dataset]),
            }
            for dataset in args.datasets
        },
    }
    atomic_json(manifest_path, manifest)
    logging.info(
        "Dynamic Top-k semantic delta ready: union=%d reused=%d pending=%d transport_rows=%d root=%s",
        expected_union_pairs,
        len(reused_keys),
        pending_total,
        manifest["totals"]["pending_transport_rows"],
        args.output_root,
    )
    return manifest


def validate_new_label_manifest(path: Path, prepared: dict[str, Any], datasets: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "docs_per_question": prepared["annotation_contract"]["max_documents_per_transport_row"],
        "allow_fewer_documents": True,
        "questions_per_batch": 10,
        "max_doc_chars": 0,
        "codex_bin": "codex",
        "codex_model_request": "gpt-5.6-terra",
        "codex_reasoning_effort": "medium",
        "web_search_enabled": False,
        "worker_count": 8,
    }
    mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    dataset_values = value.get("datasets") if isinstance(value.get("datasets"), dict) else {}
    for dataset in datasets:
        wanted = prepared["datasets"][dataset]
        actual = dataset_values.get(dataset, {})
        if actual.get("pairs") != wanted.get("pending_pairs", 0):
            mismatches[f"datasets.{dataset}.pairs"] = {
                "expected": wanted.get("pending_pairs", 0),
                "actual": actual.get("pairs"),
            }
        if actual.get("questions") != wanted.get("pending_transport_rows", 0):
            mismatches[f"datasets.{dataset}.questions"] = {
                "expected": wanted.get("pending_transport_rows", 0),
                "actual": actual.get("questions"),
            }
    if mismatches:
        raise ValueError(f"New semantic-label contract mismatch: {mismatches}")
    return value


def load_semantic_rows(
    paths: dict[str, Path], expected_pairs: int, origin: str, overall: tqdm, stage: tqdm
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for dataset, path in paths.items():
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            stable_id = str(row.get("doc_stable_id") or "")
            key = semantic_key(dataset, sample_id, stable_id)
            if not sample_id or not stable_id or key in rows:
                raise ValueError(f"Invalid or duplicate {origin} semantic pair: {key}")
            label = str(row.get("semantic_label") or "")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid {origin} semantic label for {key}: {label}")
            rows[key] = row
            stage.update(1)
            overall.update(1)
    if len(rows) != expected_pairs:
        raise ValueError(f"{origin} semantic count mismatch: {len(rows)} != {expected_pairs}")
    return rows


def merge(args: argparse.Namespace) -> dict[str, Any]:
    prepared_path = args.prepared_root / "prepare_manifest.json"
    if not prepared_path.is_file():
        raise FileNotFoundError(prepared_path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("status") != "complete" or prepared.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError(f"Prepared semantic delta is incomplete or incompatible: {prepared_path}")
    new_manifest = validate_new_label_manifest(args.new_label_root / "manifest.json", prepared, args.datasets)
    expected_union_pairs = int(prepared["totals"]["union_pairs"])
    expected_reused_pairs = int(prepared["totals"]["reused_pairs"])
    expected_pending_pairs = int(prepared["totals"]["pending_pairs"])
    contract = {
        "merge_version": MERGE_VERSION,
        "prepared_manifest": path_identity(prepared_path),
        "new_label_manifest": path_identity(args.new_label_root / "manifest.json"),
        "datasets": list(args.datasets),
    }
    input_fingerprint = fingerprint(contract)
    output_manifest_path = args.output_root / "manifest.json"
    if args.resume and output_manifest_path.is_file():
        old = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        outputs = old.get("outputs") if isinstance(old.get("outputs"), dict) else {}
        valid = old.get("status") == "complete" and old.get("input_fingerprint") == input_fingerprint
        for dataset in args.datasets:
            path = args.output_root / dataset / "codex_semantic_labels.jsonl"
            expected = outputs.get(dataset)
            if not path.is_file() or not isinstance(expected, dict):
                valid = False
                break
            actual = path_identity(path)
            valid = valid and actual["size"] == expected.get("size") and actual["mtime_ns"] == expected.get(
                "mtime_ns"
            )
        if valid:
            logging.info("Dynamic Top-k semantic union already merged and verified: %s", args.output_root)
            return old

    overall = tqdm(
        total=expected_union_pairs * 2,
        desc="DynamicSemanticUnionMergeOverall",
        unit="pair",
        position=0,
        dynamic_ncols=True,
    )
    stage = tqdm(
        total=expected_union_pairs,
        desc="Stage 1/2 index reused and new labels",
        unit="pair",
        position=1,
        dynamic_ncols=True,
    )
    reused_paths = {
        dataset: args.prepared_root / "reused_labels" / f"{dataset}.jsonl" for dataset in args.datasets
    }
    new_paths = {
        dataset: args.new_label_root / dataset / "codex_semantic_labels.jsonl" for dataset in args.datasets
    }
    reused = load_semantic_rows(reused_paths, expected_reused_pairs, "reused", overall, stage)
    new = load_semantic_rows(new_paths, expected_pending_pairs, "new", overall, stage)
    overlap = set(reused) & set(new)
    if overlap:
        raise ValueError(f"Reused/new semantic labels overlap: {len(overlap)}; first={next(iter(overlap))}")
    labels = {**reused, **new}
    stage.close()
    if len(labels) != expected_union_pairs:
        raise ValueError(f"Merged semantic count mismatch: {len(labels)} != {expected_union_pairs}")

    candidate_paths = {
        dataset: Path(prepared["input_contract"]["candidate_files"][dataset]["path"])
        for dataset in args.datasets
    }
    output_paths = {
        dataset: args.output_root / dataset / "codex_semantic_labels.jsonl" for dataset in args.datasets
    }
    temporary_paths = {dataset: path.with_name(path.name + ".tmp") for dataset, path in output_paths.items()}
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {dataset: path.open("w", encoding="utf-8") for dataset, path in temporary_paths.items()}
    seen: set[tuple[str, str, str]] = set()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    origin_counts: dict[str, Counter[str]] = defaultdict(Counter)
    question_counts = Counter()
    pair_counts = Counter()
    stage = tqdm(
        total=expected_union_pairs,
        desc="Stage 2/2 write and verify exact union",
        unit="pair",
        position=1,
        dynamic_ncols=True,
    )
    try:
        for dataset, path in candidate_paths.items():
            for candidate_row in iter_jsonl(path):
                sample_id = str(candidate_row.get("sample_id") or "")
                question_counts[dataset] += 1
                documents = candidate_row.get("candidate_documents")
                if not isinstance(documents, list):
                    raise ValueError(f"Invalid union documents for {sample_id}")
                for document in documents:
                    stable_id = stable_document_id(document)
                    key = semantic_key(dataset, sample_id, stable_id)
                    if key in seen or key not in labels:
                        raise ValueError(f"Duplicate or missing merged semantic label: {key}")
                    origin = "reused_global_top32" if key in reused else "new_dynamic_delta"
                    output = remap_label(labels[key], dataset, sample_id, document, origin)
                    handles[dataset].write(
                        json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    seen.add(key)
                    pair_counts[dataset] += 1
                    label_counts[dataset][str(output["semantic_label"])] += 1
                    origin_counts[dataset][origin] += 1
                    stage.update(1)
                    overall.update(1)
    finally:
        for handle in handles.values():
            handle.close()
        stage.close()
        overall.close()
    if seen != set(labels):
        missing = sorted(set(labels) - seen)
        raise ValueError(f"Union output omitted {len(missing)} semantic labels; first={missing[:1]}")
    for dataset in args.datasets:
        expected_pairs = int(prepared["datasets"][dataset]["union_pairs"])
        expected_questions = int(prepared["datasets"][dataset]["questions"])
        if pair_counts[dataset] != expected_pairs or question_counts[dataset] != expected_questions:
            raise ValueError(
                f"Final dataset count mismatch for {dataset}: questions={question_counts[dataset]}/{expected_questions} "
                f"pairs={pair_counts[dataset]}/{expected_pairs}"
            )
        os.replace(temporary_paths[dataset], output_paths[dataset])

    manifest = {
        "merge_version": MERGE_VERSION,
        "created_at": utc_now(),
        "status": "complete",
        "input_fingerprint": input_fingerprint,
        "input_contract": contract,
        "questions": int(prepared["totals"]["questions"]),
        "pairs": expected_union_pairs,
        "reused_pairs": expected_reused_pairs,
        "new_pairs": expected_pending_pairs,
        "dynamic_top_k_values": [1, 2, 4, 8, 16, 32],
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": new_manifest["codex_model_request"],
        "reasoning_effort": new_manifest["codex_reasoning_effort"],
        "web_search_enabled": new_manifest["web_search_enabled"],
        "datasets": {
            dataset: {
                "questions": question_counts[dataset],
                "pairs": pair_counts[dataset],
                "origin_distribution": dict(sorted(origin_counts[dataset].items())),
                "label_distribution": dict(sorted(label_counts[dataset].items())),
            }
            for dataset in args.datasets
        },
        "outputs": {dataset: path_identity(output_paths[dataset]) for dataset in args.datasets},
    }
    atomic_json(output_manifest_path, manifest)
    logging.info(
        "Exact dynamic Top-k semantic union complete: questions=%d pairs=%d reused=%d new=%d root=%s",
        manifest["questions"],
        manifest["pairs"],
        manifest["reused_pairs"],
        manifest["new_pairs"],
        args.output_root,
    )
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.command == "prepare":
        prepare(args)
    elif args.command == "merge":
        merge(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

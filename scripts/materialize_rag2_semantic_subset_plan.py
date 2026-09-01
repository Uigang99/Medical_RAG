#!/usr/bin/env python3
"""Materialize reproducible semantic Top-8 subset-generation plans.

This script does not run the target LLM.  It joins the reranked Top-8
candidate rows with the raw five-class semantic labels in strict lockstep and
records a bounded collection of semantically meaningful document subsets.
Only subsets containing at least two documents require new generation;
single-document observations point at the existing independent-document
trace cache, and unavailable policies remain explicitly null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_semantic_subset_plan_v1"
POLICY_VERSION = "rag2_semantic_structured_subset_policies_v1"
SUPPORTED_DATASETS = ("medmcqa", "medqa")
SEMANTIC_LABELS = (
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
)
POLICY_NAMES = (
    "all_top8",
    "direct_all",
    "supporting_all",
    "semantic_valid_all",
    "semantic_invalid_all",
    "valid_pair",
    "valid_pair_plus_no",
    "valid_pair_plus_misleading",
    "valid_all_plus_no",
    "valid_all_plus_misleading",
)
INTERNAL_SPLITS = ("train", "val", "test")

DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "candidates/source_balanced32_rerank8_v1"
)
DEFAULT_SEMANTIC_LABEL_ROOT = (
    Path("/home/user/codex_rag2_outputs")
    / "codex_evidence_utility_labels_three_anchor_top8_terra_medium_v1_incremental"
    / "terra_medium"
)
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "filter_training_inputs_rag2_paper_reproduction_three_class_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-file", default="candidates_top8.jsonl")
    parser.add_argument(
        "--semantic-label-root", type=Path, default=DEFAULT_SEMANTIC_LABEL_ROOT
    )
    parser.add_argument("--semantic-label-file", default="codex_semantic_labels.jsonl")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--max-questions-per-dataset",
        type=int,
        default=0,
        help="Deterministic candidate-prefix limit; zero selects the complete dataset.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a completed plan only when its immutable contract and SHA-256 match.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate every selected join and report counts without writing a plan.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def path_identity(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    value: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        value["sha256"] = file_sha256(resolved)
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield row


def require_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def document_stable_id(document: Mapping[str, Any]) -> str:
    value = (
        document.get("stable_id")
        or document.get("corpus_id")
        or document.get("chunk_id")
        or document.get("db_id")
    )
    if not value:
        value = f"{document.get('source')}:{document.get('local_id')}"
    value = str(value or "")
    if not value or value == "None:None":
        raise ValueError(f"Document has no stable identity: {dict(document)}")
    return value


def pair_id(sample_id: str, rank: int, stable_id: str) -> str:
    return f"{sample_id}::{rank}::{stable_id}"


def subset_id(sample_id: str, mask: int, top_k: int = 8) -> str:
    width = max(2, (int(top_k) + 3) // 4)
    return f"{sample_id}::semantic_subset::{mask:0{width}x}"


def load_split_assignments(split_root: Path, dataset: str) -> tuple[dict[str, str], dict[str, Any]]:
    assignments: dict[str, str] = {}
    identities: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for split in INTERNAL_SPLITS:
        path = require_file(
            split_root / dataset / "sample_ids" / f"{split}.txt",
            f"{dataset} internal {split} IDs",
        )
        identities[split] = path_identity(path, include_sha256=True)
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                sample_id_value = line.strip()
                if not sample_id_value:
                    continue
                previous = assignments.setdefault(sample_id_value, split)
                if previous != split:
                    raise ValueError(
                        f"Internal split leakage for {sample_id_value}: {previous} and {split} "
                        f"({path}:{line_number})"
                    )
                count += 1
        counts[split] = count
    if not assignments:
        raise ValueError(f"No internal split IDs found for {dataset}")
    return assignments, {"files": identities, "counts": counts, "unique_ids": len(assignments)}


def candidate_contract(
    args: argparse.Namespace, dataset: str
) -> tuple[Path, Path, dict[str, Any], int]:
    root = args.candidate_root / dataset / args.split
    candidate_path = require_file(root / args.candidate_file, f"{dataset} candidates")
    manifest_path = require_file(root / "candidate_manifest.json", f"{dataset} candidate manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "type": "rag2_filter_candidate_dataset",
        "dataset": dataset,
        "split": args.split,
        "candidate_layout": "source_balanced",
        "top_k": args.top_k,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate manifest mismatch for {dataset}: {mismatches}")
    question_count = int(manifest.get("selected_question_count", -1))
    if question_count <= 0:
        raise ValueError(f"Invalid selected_question_count for {dataset}: {question_count}")
    return candidate_path, manifest_path, manifest, question_count


def indices_mask(indices: Iterable[int]) -> int:
    value = 0
    for index in indices:
        value |= 1 << int(index)
    return value


def first_index(labels: Sequence[str], target: str) -> int | None:
    return next((index for index, value in enumerate(labels) if value == target), None)


def policy_masks(labels: Sequence[str]) -> dict[str, int | None]:
    """Return the ten preregistered policy masks in rerank order.

    ``valid_pair_plus_*`` is a two-document interaction probe: the highest
    reranked semantic-valid document plus the highest reranked document of the
    named invalid class.  ``valid_all_plus_*`` uses all valid documents plus
    that same highest-ranked invalid document.
    """

    if len(labels) != 8:
        raise ValueError(f"Policy version {POLICY_VERSION} requires exactly 8 labels")
    unknown = set(labels) - set(SEMANTIC_LABELS)
    if unknown:
        raise ValueError(f"Unknown semantic labels: {sorted(unknown)}")

    direct = [index for index, label in enumerate(labels) if label == "direct_support"]
    supporting = [index for index, label in enumerate(labels) if label == "supporting_evidence"]
    valid = [
        index
        for index, label in enumerate(labels)
        if label in {"direct_support", "supporting_evidence"}
    ]
    invalid = [
        index
        for index, label in enumerate(labels)
        if label in {"no_evidence", "misleading_evidence"}
    ]
    no_index = first_index(labels, "no_evidence")
    misleading_index = first_index(labels, "misleading_evidence")
    valid_anchor = valid[0] if valid else None

    def nonempty(values: Sequence[int]) -> int | None:
        return indices_mask(values) if values else None

    def interaction(invalid_index: int | None, *, all_valid: bool) -> int | None:
        if valid_anchor is None or invalid_index is None:
            return None
        left = valid if all_valid else [valid_anchor]
        return indices_mask([*left, invalid_index])

    result: dict[str, int | None] = {
        "all_top8": (1 << 8) - 1,
        "direct_all": nonempty(direct),
        "supporting_all": nonempty(supporting),
        "semantic_valid_all": nonempty(valid),
        "semantic_invalid_all": nonempty(invalid),
        "valid_pair": indices_mask(valid[:2]) if len(valid) >= 2 else None,
        "valid_pair_plus_no": interaction(no_index, all_valid=False),
        "valid_pair_plus_misleading": interaction(misleading_index, all_valid=False),
        "valid_all_plus_no": interaction(no_index, all_valid=True),
        "valid_all_plus_misleading": interaction(misleading_index, all_valid=True),
    }
    if tuple(result) != POLICY_NAMES:
        raise AssertionError("Policy order drifted from the immutable contract")
    return result


def make_question_plan(
    candidate: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    source_split: str,
    analysis_split: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id_value = str(candidate.get("sample_id") or "")
    if not sample_id_value:
        raise ValueError(f"Candidate row has no sample_id: {candidate}")
    if str(candidate.get("dataset") or "") != dataset:
        raise ValueError(
            f"Candidate dataset mismatch for {sample_id_value}: {candidate.get('dataset')} != {dataset}"
        )
    if str(candidate.get("split") or "") != source_split:
        raise ValueError(
            f"Candidate split mismatch for {sample_id_value}: {candidate.get('split')} != {source_split}"
        )
    documents = list(candidate.get("candidate_documents") or [])
    if len(documents) != top_k:
        raise ValueError(
            f"Expected exactly Top-{top_k} candidates for {sample_id_value}, found {len(documents)}"
        )
    if len(label_rows) != top_k:
        raise ValueError(f"Expected exactly {top_k} semantic rows for {sample_id_value}")

    document_order: list[dict[str, Any]] = []
    labels: list[str] = []
    seen_pairs: set[str] = set()
    for rank, (document, label_row) in enumerate(zip(documents, label_rows), start=1):
        if not isinstance(document, dict):
            raise ValueError(f"Candidate document is not an object: {sample_id_value} rank={rank}")
        rerank_rank = int(document.get("rerank_rank") or rank)
        if rerank_rank != rank:
            raise ValueError(
                f"Candidate file is not in contiguous rerank order for {sample_id_value}: "
                f"position={rank} rerank_rank={rerank_rank}"
            )
        stable_id = document_stable_id(document)
        expected_pair = pair_id(sample_id_value, rank, stable_id)
        actual_pair = str(label_row.get("pair_id") or label_row.get("id") or "")
        checks = {
            "dataset": (str(label_row.get("dataset") or ""), dataset),
            "sample_id": (str(label_row.get("sample_id") or ""), sample_id_value),
            "doc_rank": (int(label_row.get("doc_rank", -1)), rank),
            "doc_stable_id": (str(label_row.get("doc_stable_id") or ""), stable_id),
            "pair_id": (actual_pair, expected_pair),
        }
        mismatches = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in checks.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"Candidate/semantic lockstep mismatch: {mismatches}")
        if actual_pair in seen_pairs:
            raise ValueError(f"Duplicate pair within question: {actual_pair}")
        seen_pairs.add(actual_pair)
        label = str(label_row.get("semantic_label") or "")
        if label not in SEMANTIC_LABELS:
            raise ValueError(f"Invalid semantic label for {actual_pair}: {label!r}")
        try:
            confidence = float(label_row.get("confidence"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid semantic confidence for {actual_pair}") from error
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Semantic confidence outside [0,1] for {actual_pair}: {confidence}")
        if not str(document.get("text") or "").strip():
            raise ValueError(f"Empty candidate document text for {actual_pair}")
        labels.append(label)
        document_order.append(
            {
                "doc_rank": rank,
                "rerank_rank": rerank_rank,
                "pair_id": actual_pair,
                "doc_stable_id": stable_id,
                "source": str(document.get("source") or ""),
                "semantic_label": label,
                "semantic_confidence": confidence,
            }
        )

    masks = policy_masks(labels)
    grouped_policies: dict[int, list[str]] = defaultdict(list)
    assignments: dict[str, Any] = {}
    for policy_name in POLICY_NAMES:
        mask = masks[policy_name]
        if mask is None:
            assignments[policy_name] = {
                "status": "unavailable",
                "mask": None,
                "subset_id": None,
                "document_count": 0,
                "document_ranks": [],
                "observation_source": None,
            }
            continue
        ranks = [index + 1 for index in range(top_k) if mask & (1 << index)]
        count = len(ranks)
        stable_subset_id = subset_id(sample_id_value, mask, top_k)
        if count == 1:
            source = "existing_independent_document_trace"
            status = "cached_singleton"
        elif count >= 2:
            source = "new_rationale_answer_generation"
            status = "planned_generation"
            grouped_policies[mask].append(policy_name)
        else:  # Defensive: current policies encode empty selections as null.
            raise AssertionError(f"Empty mask should be represented as unavailable: {policy_name}")
        assignments[policy_name] = {
            "status": status,
            "mask": mask,
            "subset_id": stable_subset_id,
            "document_count": count,
            "document_ranks": ranks,
            "observation_source": source,
        }

    generation_subsets = []
    for mask in sorted(grouped_policies):
        ranks = [index + 1 for index in range(top_k) if mask & (1 << index)]
        selected = [document_order[rank - 1] for rank in ranks]
        generation_subsets.append(
            {
                "subset_id": subset_id(sample_id_value, mask, top_k),
                "mask": mask,
                "mask_binary_rerank_order": format(mask, f"0{top_k}b")[::-1],
                "document_count": len(ranks),
                "document_ranks": ranks,
                "pair_ids": [row["pair_id"] for row in selected],
                "doc_stable_ids": [row["doc_stable_id"] for row in selected],
                "semantic_labels": [row["semantic_label"] for row in selected],
                "policies": grouped_policies[mask],
            }
        )

    row = {
        "run_version": RUN_VERSION,
        "policy_version": POLICY_VERSION,
        "dataset": dataset,
        "source_split": source_split,
        "analysis_split": analysis_split,
        "row_idx": int(candidate.get("row_idx", -1)),
        "sample_id": sample_id_value,
        "top_k": top_k,
        "document_order": document_order,
        "policy_assignments": assignments,
        "generation_subsets": generation_subsets,
    }
    stats = {
        "labels": Counter(labels),
        "policy_status": Counter(
            (policy_name, assignment["status"])
            for policy_name, assignment in assignments.items()
        ),
        "generated_subset_sizes": Counter(
            subset["document_count"] for subset in generation_subsets
        ),
        "generated_subsets": len(generation_subsets),
    }
    return row, stats


def build_dataset_contract(args: argparse.Namespace, dataset: str) -> tuple[dict[str, Any], int]:
    candidate_path, candidate_manifest_path, candidate_manifest, available_questions = candidate_contract(
        args, dataset
    )
    labels_path = require_file(
        args.semantic_label_root / dataset / args.semantic_label_file,
        f"{dataset} raw five-class semantic labels",
    )
    assignments, split_contract = load_split_assignments(args.split_root, dataset)
    selected_questions = (
        min(args.max_questions_per_dataset, available_questions)
        if args.max_questions_per_dataset > 0
        else available_questions
    )
    contract = {
        "run_version": RUN_VERSION,
        "policy_version": POLICY_VERSION,
        "dataset": dataset,
        "source_split": args.split,
        "top_k": args.top_k,
        "max_questions_per_dataset": args.max_questions_per_dataset,
        "available_question_count": available_questions,
        "selected_question_count": selected_questions,
        "candidate": path_identity(candidate_path),
        "candidate_manifest": path_identity(candidate_manifest_path, include_sha256=True),
        "candidate_manifest_fields": {
            key: candidate_manifest.get(key)
            for key in (
                "type",
                "dataset",
                "split",
                "candidate_layout",
                "top_k",
                "candidate_pool_top_k",
                "per_source_top_k",
                "query_prompt_version",
            )
        },
        "semantic_labels": path_identity(labels_path),
        "semantic_label_classes": list(SEMANTIC_LABELS),
        "internal_splits": split_contract,
        "policy_names": list(POLICY_NAMES),
        "document_order": "ascending rerank_rank from candidate file; never reordered by semantic label",
        "mixed_label_policy": (
            "indeterminate_or_mixed is included only by all_top8 and excluded from semantic-valid/invalid sets"
        ),
        "generation_rule": "deduplicate identical masks per question and generate only masks with >=2 documents",
    }
    # The assignments are intentionally loaded here to make split-file
    # corruption a preflight failure; only their immutable contract is stored.
    del assignments
    return contract, selected_questions


def completed_dataset_manifest(
    output_root: Path, dataset: str, split: str, contract_hash: str
) -> dict[str, Any] | None:
    root = output_root / dataset / split
    plan_path = root / "subset_plan.jsonl"
    manifest_path = root / "subset_plan_manifest.json"
    if not plan_path.exists() and not manifest_path.exists():
        return None
    # A crash can occur after the atomic plan rename but before its completion
    # manifest is committed.  The uncommitted plan is safe to regenerate under
    # the same immutable contract; the inverse state cannot be trusted.
    if plan_path.is_file() and not manifest_path.exists():
        logging.warning("Regenerating uncommitted subset plan without manifest: %s", plan_path)
        return None
    if not plan_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Incomplete prior plan output: {plan_path} / {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_sha256") != contract_hash:
        raise RuntimeError(
            f"Existing plan contract mismatch for {dataset}; use a new --output-root"
        )
    identity = manifest.get("plan") or {}
    if int(identity.get("size_bytes", -1)) != plan_path.stat().st_size:
        raise RuntimeError(f"Existing plan size mismatch: {plan_path}")
    if identity.get("sha256") != file_sha256(plan_path):
        raise RuntimeError(f"Existing plan SHA-256 mismatch: {plan_path}")
    return manifest


def merge_counter(target: Counter[Any], value: Mapping[Any, int]) -> None:
    for key, count in value.items():
        target[key] += int(count)


def materialize_dataset(
    args: argparse.Namespace,
    dataset: str,
    contract: dict[str, Any],
    contract_hash: str,
    progress: PipelineProgress,
) -> dict[str, Any]:
    candidate_path = Path(contract["candidate"]["path"])
    label_path = Path(contract["semantic_labels"]["path"])
    assignments, split_contract = load_split_assignments(args.split_root, dataset)
    selected_questions = int(contract["selected_question_count"])
    available_questions = int(contract["available_question_count"])
    limited = selected_questions < available_questions

    output_dir = args.output_root / dataset / args.split
    plan_path = output_dir / "subset_plan.jsonl"
    manifest_path = output_dir / "subset_plan_manifest.json"
    temporary = plan_path.with_name(plan_path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    if not args.preflight_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "materialization_contract.json", {**contract, "sha256": contract_hash})

    candidate_rows = iter_jsonl(candidate_path)
    label_rows = iter_jsonl(label_path)
    label_counts: Counter[str] = Counter()
    analysis_split_counts: Counter[str] = Counter()
    policy_status: Counter[tuple[str, str]] = Counter()
    generated_subset_sizes: Counter[int] = Counter()
    total_generated_subsets = 0
    plan_hasher = hashlib.sha256()
    seen_sample_ids: set[str] = set()
    processed = 0
    handle = None
    if not args.preflight_only:
        handle = temporary.open("w", encoding="utf-8", buffering=64 * 1024 * 1024)
    try:
        for _ in range(selected_questions):
            try:
                candidate = next(candidate_rows)
            except StopIteration as error:
                raise RuntimeError(
                    f"Candidate file ended early for {dataset}: {processed}/{selected_questions}"
                ) from error
            sample_id_value = str(candidate.get("sample_id") or "")
            if not sample_id_value:
                raise ValueError(f"Candidate row {processed + 1} has no sample_id")
            if sample_id_value in seen_sample_ids:
                raise ValueError(f"Duplicate candidate sample_id: {sample_id_value}")
            seen_sample_ids.add(sample_id_value)
            analysis_split = assignments.get(sample_id_value)
            if analysis_split is None:
                raise ValueError(
                    f"Candidate question is absent from frozen internal splits: {sample_id_value}"
                )
            semantic_rows = []
            for _ in range(args.top_k):
                try:
                    semantic_rows.append(next(label_rows))
                except StopIteration as error:
                    raise RuntimeError(
                        f"Semantic labels ended before candidate {sample_id_value}"
                    ) from error
            row, row_stats = make_question_plan(
                candidate,
                semantic_rows,
                dataset=dataset,
                source_split=args.split,
                analysis_split=analysis_split,
                top_k=args.top_k,
            )
            merge_counter(label_counts, row_stats["labels"])
            merge_counter(policy_status, row_stats["policy_status"])
            merge_counter(generated_subset_sizes, row_stats["generated_subset_sizes"])
            total_generated_subsets += int(row_stats["generated_subsets"])
            analysis_split_counts[analysis_split] += 1
            if handle is not None:
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                encoded = line.encode("utf-8")
                handle.write(line)
                plan_hasher.update(encoded)
            processed += 1
            if processed % 128 == 0 or processed == selected_questions:
                progress.set_detail(
                    f"dataset={dataset} questions={processed}/{selected_questions} "
                    f"planned_multi_doc={total_generated_subsets}"
                )
            progress.update(1)

        if processed != selected_questions:  # pragma: no cover - defensive invariant
            raise AssertionError(
                f"Processed-count invariant failed for {dataset}: {processed}/{selected_questions}"
            )
        if not limited:
            try:
                extra_candidate = next(candidate_rows)
            except StopIteration:
                extra_candidate = None
            if extra_candidate is not None:
                raise RuntimeError(
                    f"Candidate manifest under-counts rows for {dataset}; first extra="
                    f"{extra_candidate.get('sample_id')}"
                )
            try:
                extra_label = next(label_rows)
            except StopIteration:
                extra_label = None
            if extra_label is not None:
                raise RuntimeError(
                    f"Semantic file has rows not joined to candidate Top-{args.top_k}: "
                    f"first={extra_label.get('pair_id')}"
                )
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            os.replace(temporary, plan_path)
    except BaseException:
        if handle is not None:
            handle.close()
        if temporary.exists():
            temporary.unlink()
        raise

    unused_split_ids = len(assignments) - len(seen_sample_ids)
    summary = {
        "run_version": RUN_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at": utc_now(),
        "dataset": dataset,
        "source_split": args.split,
        "contract": contract,
        "contract_sha256": contract_hash,
        "question_count": processed,
        "pair_count": processed * args.top_k,
        "semantic_label_counts": dict(sorted(label_counts.items())),
        "analysis_split_counts": dict(sorted(analysis_split_counts.items())),
        "frozen_internal_split_counts": split_contract["counts"],
        "unused_frozen_split_ids": unused_split_ids,
        "policy_status_counts": {
            policy: {
                status: policy_status[(policy, status)]
                for status in (
                    "planned_generation",
                    "cached_singleton",
                    "unavailable",
                )
                if policy_status[(policy, status)]
            }
            for policy in POLICY_NAMES
        },
        "unique_multi_document_subsets": total_generated_subsets,
        "mean_multi_document_subsets_per_question": (
            total_generated_subsets / processed if processed else 0.0
        ),
        "generated_subset_size_counts": {
            str(key): value for key, value in sorted(generated_subset_sizes.items())
        },
        "preflight_only": bool(args.preflight_only),
    }
    if not args.preflight_only:
        summary["plan"] = {
            "path": str(plan_path.resolve()),
            "size_bytes": plan_path.stat().st_size,
            "sha256": plan_hasher.hexdigest(),
        }
        atomic_json(manifest_path, summary)
    logging.info(
        "Semantic subset plan %s: dataset=%s questions=%s pairs=%s unique_multi_doc=%s "
        "mean=%.3f output=%s",
        "validated" if args.preflight_only else "complete",
        dataset,
        processed,
        processed * args.top_k,
        total_generated_subsets,
        summary["mean_multi_document_subsets_per_question"],
        "not written (preflight-only)" if args.preflight_only else plan_path,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    if args.top_k != 8:
        raise ValueError(f"{POLICY_VERSION} is defined only for Top-8; got --top-k={args.top_k}")
    if args.max_questions_per_dataset < 0:
        raise ValueError("--max-questions-per-dataset must be non-negative")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError(f"Duplicate datasets requested: {args.datasets}")

    contracts: dict[str, dict[str, Any]] = {}
    contract_hashes: dict[str, str] = {}
    selected_counts: dict[str, int] = {}
    for dataset in args.datasets:
        contract, selected = build_dataset_contract(args, dataset)
        contracts[dataset] = contract
        contract_hashes[dataset] = fingerprint(contract)
        selected_counts[dataset] = selected

    if not args.preflight_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(args.output_root).free
        # The plan contains references rather than document text.  The very
        # conservative 8 KiB/question bound protects atomic temporary output.
        estimated_bytes = sum(selected_counts.values()) * 8192
        if free_bytes < estimated_bytes * 2 + 1_000_000_000:
            raise RuntimeError(
                f"Insufficient disk for atomic plan materialization: free={free_bytes} "
                f"estimated_plan={estimated_bytes}"
            )

    completed: dict[str, dict[str, Any]] = {}
    if not args.preflight_only:
        for dataset in args.datasets:
            manifest = completed_dataset_manifest(
                args.output_root, dataset, args.split, contract_hashes[dataset]
            )
            if manifest is not None:
                if not args.resume:
                    raise FileExistsError(
                        f"Completed output exists for {dataset}; use a new --output-root"
                    )
                completed[dataset] = manifest

    root_contract = {
        "run_version": RUN_VERSION,
        "policy_version": POLICY_VERSION,
        "datasets": list(args.datasets),
        "source_split": args.split,
        "top_k": args.top_k,
        "max_questions_per_dataset": args.max_questions_per_dataset,
        "dataset_contract_sha256": contract_hashes,
    }
    root_contract_hash = fingerprint(root_contract)
    if not args.preflight_only:
        contract_path = args.output_root / "materialization_contract.json"
        if contract_path.is_file():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing.get("sha256") != root_contract_hash:
                raise RuntimeError(
                    f"Output-root contract mismatch; use a new --output-root: {contract_path}"
                )
        else:
            atomic_json(contract_path, {**root_contract, "sha256": root_contract_hash})

    total_questions = sum(selected_counts.values())
    cached_questions = sum(int(value["question_count"]) for value in completed.values())
    logging.info(
        "Semantic subset materialization plan: datasets=%s questions=%s cached=%s remaining=%s "
        "policies=%s preflight_only=%s",
        selected_counts,
        total_questions,
        cached_questions,
        total_questions - cached_questions,
        list(POLICY_NAMES),
        args.preflight_only,
    )
    progress = PipelineProgress(
        overall_total=total_questions,
        overall_initial=cached_questions,
        desc="SemanticSubsetPlan",
    )
    summaries: dict[str, dict[str, Any]] = dict(completed)
    try:
        for dataset in args.datasets:
            if dataset in completed:
                logging.info(
                    "Reusing complete subset plan: dataset=%s questions=%s multi_doc=%s",
                    dataset,
                    completed[dataset]["question_count"],
                    completed[dataset]["unique_multi_document_subsets"],
                )
                continue
            progress.set_stage(
                "join candidates + semantic labels and plan subsets",
                total=selected_counts[dataset],
            )
            progress.set_detail(
                f"dataset={dataset} selected={selected_counts[dataset]} active_stage=preflight+plan"
            )
            summaries[dataset] = materialize_dataset(
                args,
                dataset,
                contracts[dataset],
                contract_hashes[dataset],
                progress,
            )
    finally:
        progress.close()

    aggregate = {
        "run_version": RUN_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at": utc_now(),
        "contract": root_contract,
        "contract_sha256": root_contract_hash,
        "datasets": {
            dataset: {
                "question_count": int(summaries[dataset]["question_count"]),
                "pair_count": int(summaries[dataset]["pair_count"]),
                "unique_multi_document_subsets": int(
                    summaries[dataset]["unique_multi_document_subsets"]
                ),
                "mean_multi_document_subsets_per_question": float(
                    summaries[dataset]["mean_multi_document_subsets_per_question"]
                ),
                "manifest_path": (
                    None
                    if args.preflight_only
                    else str(
                        (
                            args.output_root
                            / dataset
                            / args.split
                            / "subset_plan_manifest.json"
                        ).resolve()
                    )
                ),
            }
            for dataset in args.datasets
        },
        "totals": {
            "questions": sum(int(value["question_count"]) for value in summaries.values()),
            "pairs": sum(int(value["pair_count"]) for value in summaries.values()),
            "unique_multi_document_subsets": sum(
                int(value["unique_multi_document_subsets"]) for value in summaries.values()
            ),
        },
        "preflight_only": bool(args.preflight_only),
    }
    if not args.preflight_only:
        atomic_json(args.output_root / "manifest.json", aggregate)
    logging.info("Aggregate semantic subset plan: %s", canonical_json(aggregate["totals"]))


if __name__ == "__main__":
    main()

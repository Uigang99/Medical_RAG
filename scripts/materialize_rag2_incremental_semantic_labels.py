#!/usr/bin/env python3
"""Prepare and merge pair-level incremental RAG2 semantic annotations.

The semantic judgment is reusable when ``dataset``, ``sample_id``, and the
stable corpus-document ID are unchanged.  Document rank is deliberately not
part of that key because reranking can move an otherwise identical chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm


ANNOTATION_VERSION = "rag2_codex_evidence_utility_label_v2"
PROMPT_VERSION = "rag2_codex_evidence_utility_prompt_v3_compact_item_index"
SEMANTIC_LABELS = {
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse semantic labels for unchanged question-document pairs, "
            "materialize only pending candidates, or merge both sources."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidates-paths", type=Path, nargs="+", required=True)
    prepare.add_argument("--existing-label-paths", type=Path, nargs="+", required=True)
    prepare.add_argument("--existing-manifest-path", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--docs-per-question", type=int, default=8)
    prepare.add_argument("--sqlite-work-dir", type=Path, default=Path("/tmp"))
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--prepared-root", type=Path, required=True)
    merge.add_argument(
        "--new-label-root",
        type=Path,
        required=True,
        help="Terra variant root containing DATASET/codex_semantic_labels.jsonl.",
    )
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--sqlite-work-dir", type=Path, default=Path("/tmp"))
    merge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stable_document_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    value = str(value or "").strip()
    if not value:
        raise ValueError("Candidate document has no stable document ID")
    return value


def pair_id(sample_id: str, rank: int, stable_id: str) -> str:
    return f"{sample_id}::{rank}::{stable_id}"


def json_line_count(path: Path) -> Iterator[tuple[dict[str, Any], int]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw), len(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error


def first_row(path: Path) -> dict[str, Any]:
    for row, _size in json_line_count(path):
        return row
    raise ValueError(f"Empty JSONL file: {path}")


def dataset_paths(paths: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset = str(first_row(path).get("dataset") or "").strip().lower()
        if not dataset:
            raise ValueError(f"Missing dataset in first row: {path}")
        if dataset in result:
            raise ValueError(f"Multiple files supplied for dataset={dataset}")
        result[dataset] = path.resolve()
    return result


def path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_existing_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "codex_model_request": "gpt-5.6-terra",
        "codex_reasoning_effort": "medium",
        "web_search_enabled": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Existing semantic-label manifest is incompatible: {mismatches}")
    if value.get("status") != "complete":
        raise ValueError(f"Existing semantic-label run is not complete: {path}")
    return value


def configure_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute(
        """
        CREATE TABLE labels (
            dataset TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            doc_stable_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            origin TEXT NOT NULL,
            PRIMARY KEY(dataset, sample_id, doc_stable_id)
        ) WITHOUT ROWID
        """
    )
    return connection


def validate_label(row: dict[str, Any], path: Path) -> tuple[str, str, str]:
    dataset = str(row.get("dataset") or "").strip().lower()
    sample_id = str(row.get("sample_id") or "").strip()
    stable_id = str(row.get("doc_stable_id") or "").strip()
    label = str(row.get("semantic_label") or "").strip()
    if not dataset or not sample_id or not stable_id:
        raise ValueError(f"Malformed semantic label in {path}: missing reuse key")
    if label not in SEMANTIC_LABELS:
        raise ValueError(f"Malformed semantic label in {path}: {label!r}")
    return dataset, sample_id, stable_id


def insert_label(
    connection: sqlite3.Connection,
    row: dict[str, Any],
    origin: str,
    source_path: Path,
) -> bool:
    dataset, sample_id, stable_id = validate_label(row, source_path)
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    cursor = connection.execute(
        "INSERT OR IGNORE INTO labels VALUES (?, ?, ?, ?, ?)",
        (dataset, sample_id, stable_id, payload, origin),
    )
    if cursor.rowcount:
        return True
    previous = connection.execute(
        "SELECT payload FROM labels WHERE dataset=? AND sample_id=? AND doc_stable_id=?",
        (dataset, sample_id, stable_id),
    ).fetchone()
    if previous is None:
        raise RuntimeError("SQLite label index lost an existing row")
    old = json.loads(previous[0])
    comparison_fields = (
        "semantic_label",
        "topic_relation",
        "confidence",
        "evidence_sentence_indices",
        "short_reason",
    )
    if any(old.get(field) != row.get(field) for field in comparison_fields):
        raise ValueError(
            f"Conflicting labels for {(dataset, sample_id, stable_id)} in {source_path}"
        )
    return False


def remap_label(
    label: dict[str, Any],
    candidate_row: dict[str, Any],
    document: dict[str, Any],
    origin: str,
) -> dict[str, Any]:
    sample_id = str(candidate_row["sample_id"])
    dataset = str(candidate_row["dataset"]).lower()
    stable_id = stable_document_id(document)
    rank = int(document["rerank_rank"])
    result = dict(label)
    result.update(
        {
            "id": pair_id(sample_id, rank, stable_id),
            "pair_id": pair_id(sample_id, rank, stable_id),
            "dataset": dataset,
            "sample_id": sample_id,
            "doc_rank": rank,
            "source": str(document.get("source") or "unknown"),
            "doc_stable_id": stable_id,
            "title": str(document.get("title") or ""),
            "incremental_origin": origin,
        }
    )
    return result


def ordered_documents(row: dict[str, Any], docs_per_question: int) -> list[dict[str, Any]]:
    raw = row.get("candidate_documents")
    if not isinstance(raw, list):
        raise ValueError(f"Missing candidate_documents for {row.get('sample_id')}")
    documents = sorted(
        (document for document in raw if isinstance(document, dict)),
        key=lambda document: int(document.get("rerank_rank") or sys.maxsize),
    )[:docs_per_question]
    if len(documents) != docs_per_question:
        raise ValueError(
            f"Expected exactly {docs_per_question} candidates for {row.get('sample_id')}, found {len(documents)}"
        )
    ranks = [int(document.get("rerank_rank") or 0) for document in documents]
    if ranks != list(range(1, docs_per_question + 1)):
        raise ValueError(f"Non-contiguous rerank ranks for {row.get('sample_id')}: {ranks}")
    return documents


def progress_pair(total: int, overall_initial: int, overall_total: int, stage_name: str) -> tuple[tqdm, tqdm]:
    overall = tqdm(
        total=overall_total,
        initial=overall_initial,
        desc="IncrementalSemanticOverall",
        unit="B",
        unit_scale=True,
        position=0,
        dynamic_ncols=True,
    )
    stage = tqdm(
        total=total,
        desc=stage_name,
        unit="B",
        unit_scale=True,
        position=1,
        dynamic_ncols=True,
    )
    return overall, stage


def prepare(args: argparse.Namespace) -> None:
    manifest_value = validate_existing_manifest(args.existing_manifest_path)
    candidate_paths = dataset_paths(args.candidates_paths)
    label_paths = dataset_paths(args.existing_label_paths)
    if set(candidate_paths) != set(label_paths):
        raise ValueError(
            f"Dataset mismatch: candidates={sorted(candidate_paths)}, existing_labels={sorted(label_paths)}"
        )

    inputs = {
        "candidates": {dataset: path_identity(path) for dataset, path in candidate_paths.items()},
        "existing_labels": {dataset: path_identity(path) for dataset, path in label_paths.items()},
        "existing_manifest": path_identity(args.existing_manifest_path),
        "docs_per_question": args.docs_per_question,
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
    }
    input_fingerprint = fingerprint(inputs)
    output_manifest_path = args.output_root / "prepare_manifest.json"
    if args.resume and output_manifest_path.is_file():
        old = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        required = [
            args.output_root / "pending_candidates" / f"{dataset}.jsonl"
            for dataset in candidate_paths
        ] + [
            args.output_root / "reused_labels" / f"{dataset}.jsonl"
            for dataset in candidate_paths
        ]
        if (
            old.get("status") == "complete"
            and old.get("input_fingerprint") == input_fingerprint
            and all(path.is_file() for path in required)
        ):
            logging.info("Preparation already complete and input identities match: %s", args.output_root)
            return

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.sqlite_work_dir.mkdir(parents=True, exist_ok=True)
    handle, database_name = tempfile.mkstemp(prefix="rag2_semantic_reuse_", suffix=".sqlite", dir=args.sqlite_work_dir)
    os.close(handle)
    database_path = Path(database_name)
    total_bytes = sum(path.stat().st_size for path in label_paths.values()) + sum(
        path.stat().st_size for path in candidate_paths.values()
    )
    consumed = 0
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        connection = configure_database(database_path)
        label_bytes = sum(path.stat().st_size for path in label_paths.values())
        overall, stage = progress_pair(label_bytes, 0, total_bytes, "stage=1/2 index-existing-labels")
        inserted_since_commit = 0
        for expected_dataset, path in sorted(label_paths.items()):
            for row, raw_size in json_line_count(path):
                dataset, _sample, _stable = validate_label(row, path)
                if dataset != expected_dataset:
                    raise ValueError(f"Mixed dataset in {path}: {dataset} != {expected_dataset}")
                inserted = insert_label(connection, row, "reused_existing", path)
                stats[dataset]["existing_unique"] += int(inserted)
                stats[dataset]["existing_duplicates"] += int(not inserted)
                inserted_since_commit += 1
                if inserted_since_commit >= 10_000:
                    connection.commit()
                    inserted_since_commit = 0
                stage.update(raw_size)
                overall.update(raw_size)
                consumed += raw_size
        connection.commit()
        stage.close()
        overall.close()

        candidate_bytes = sum(path.stat().st_size for path in candidate_paths.values())
        overall, stage = progress_pair(candidate_bytes, consumed, total_bytes, "stage=2/2 split-reused-and-pending")
        temporary_outputs: list[tuple[Path, Path]] = []
        for dataset, path in sorted(candidate_paths.items()):
            reused_path = args.output_root / "reused_labels" / f"{dataset}.jsonl"
            pending_path = args.output_root / "pending_candidates" / f"{dataset}.jsonl"
            reused_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path.parent.mkdir(parents=True, exist_ok=True)
            reused_tmp = reused_path.with_name(reused_path.name + ".tmp")
            pending_tmp = pending_path.with_name(pending_path.name + ".tmp")
            temporary_outputs.extend([(reused_tmp, reused_path), (pending_tmp, pending_path)])
            with reused_tmp.open("w", encoding="utf-8") as reused_handle, pending_tmp.open(
                "w", encoding="utf-8"
            ) as pending_handle:
                for row, raw_size in json_line_count(path):
                    row_dataset = str(row.get("dataset") or "").lower()
                    sample_id = str(row.get("sample_id") or "")
                    if row_dataset != dataset or not sample_id:
                        raise ValueError(f"Invalid candidate identity in {path}: {sample_id!r}")
                    documents = ordered_documents(row, args.docs_per_question)
                    pending_documents: list[dict[str, Any]] = []
                    stats[dataset]["questions"] += 1
                    for document in documents:
                        stable_id = stable_document_id(document)
                        found = connection.execute(
                            "SELECT payload FROM labels WHERE dataset=? AND sample_id=? AND doc_stable_id=?",
                            (dataset, sample_id, stable_id),
                        ).fetchone()
                        stats[dataset]["pairs"] += 1
                        if found is None:
                            pending_documents.append(document)
                            stats[dataset]["pending_pairs"] += 1
                        else:
                            label = remap_label(json.loads(found[0]), row, document, "reused_existing")
                            reused_handle.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")
                            stats[dataset]["reused_pairs"] += 1
                    if pending_documents:
                        pending_row = dict(row)
                        pending_row["candidate_documents"] = pending_documents
                        # The semantic labeler only consumes candidate_documents;
                        # keeping initial_documents would multiply incremental files.
                        pending_row.pop("initial_documents", None)
                        pending_handle.write(
                            json.dumps(pending_row, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                        stats[dataset]["pending_questions"] += 1
                        stats[dataset][f"pending_docs_{len(pending_documents)}"] += 1
                    stage.update(raw_size)
                    overall.update(raw_size)
                    consumed += raw_size
        for temporary, target in temporary_outputs:
            os.replace(temporary, target)
        stage.close()
        overall.close()
        connection.close()

        datasets = {dataset: dict(counter) for dataset, counter in sorted(stats.items())}
        total_pairs = sum(value.get("pairs", 0) for value in datasets.values())
        reused_pairs = sum(value.get("reused_pairs", 0) for value in datasets.values())
        pending_pairs = sum(value.get("pending_pairs", 0) for value in datasets.values())
        if reused_pairs + pending_pairs != total_pairs:
            raise RuntimeError("Preparation accounting invariant failed")
        output_manifest = {
            "status": "complete",
            "created_at": utc_now(),
            "annotation_version": ANNOTATION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": manifest_value.get("codex_model_request"),
            "reasoning_effort": manifest_value.get("codex_reasoning_effort"),
            "web_search_enabled": manifest_value.get("web_search_enabled"),
            "reuse_key": ["dataset", "sample_id", "doc_stable_id"],
            "reuse_contract": (
                "The question identity, gold options/answer, corpus snapshot, and stable chunk ID must be unchanged. "
                "Rerank position may change and is remapped to the new candidate row."
            ),
            "inputs": inputs,
            "input_fingerprint": input_fingerprint,
            "docs_per_question": args.docs_per_question,
            "datasets": datasets,
            "totals": {
                "pairs": total_pairs,
                "reused_pairs": reused_pairs,
                "pending_pairs": pending_pairs,
                "reuse_rate": reused_pairs / total_pairs if total_pairs else 0.0,
            },
        }
        atomic_json(output_manifest_path, output_manifest)
        logging.info(
            "Incremental preparation complete: total=%d reused=%d pending=%d reuse=%.2f%% root=%s",
            total_pairs,
            reused_pairs,
            pending_pairs,
            100.0 * reused_pairs / total_pairs,
            args.output_root,
        )
    finally:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(database_path) + suffix)
            if candidate.exists():
                candidate.unlink()


def merge(args: argparse.Namespace) -> None:
    prepare_manifest_path = args.prepared_root / "prepare_manifest.json"
    if not prepare_manifest_path.is_file():
        raise FileNotFoundError(prepare_manifest_path)
    prepared = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    if prepared.get("status") != "complete":
        raise ValueError("Incremental preparation is not complete")
    datasets = sorted(prepared["datasets"])
    candidate_paths = {
        dataset: Path(prepared["inputs"]["candidates"][dataset]["path"]) for dataset in datasets
    }
    reused_paths = {
        dataset: args.prepared_root / "reused_labels" / f"{dataset}.jsonl" for dataset in datasets
    }
    new_paths = {
        dataset: args.new_label_root / dataset / "codex_semantic_labels.jsonl" for dataset in datasets
    }
    new_manifest_path = args.new_label_root / "manifest.json"
    if not new_manifest_path.is_file():
        raise FileNotFoundError(new_manifest_path)
    new_manifest = validate_existing_manifest(new_manifest_path)
    for path in [*candidate_paths.values(), *reused_paths.values(), *new_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    input_identity = {
        "prepare_manifest": path_identity(prepare_manifest_path),
        "new_manifest": path_identity(new_manifest_path),
        "reused": {dataset: path_identity(path) for dataset, path in reused_paths.items()},
        "new": {dataset: path_identity(path) for dataset, path in new_paths.items()},
        "candidates": {dataset: path_identity(path) for dataset, path in candidate_paths.items()},
    }
    input_fingerprint = fingerprint(input_identity)
    final_manifest_path = args.output_root / "manifest.json"
    if args.resume and final_manifest_path.is_file():
        old = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        expected_outputs = [args.output_root / dataset / "codex_semantic_labels.jsonl" for dataset in datasets]
        if (
            old.get("status") == "complete"
            and old.get("input_fingerprint") == input_fingerprint
            and all(path.is_file() for path in expected_outputs)
        ):
            logging.info("Merged semantic labels already complete: %s", args.output_root)
            return

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.sqlite_work_dir.mkdir(parents=True, exist_ok=True)
    descriptor, database_name = tempfile.mkstemp(
        prefix="rag2_semantic_merge_", suffix=".sqlite", dir=args.sqlite_work_dir
    )
    os.close(descriptor)
    database_path = Path(database_name)
    label_bytes = sum(path.stat().st_size for path in reused_paths.values()) + sum(
        path.stat().st_size for path in new_paths.values()
    )
    candidate_bytes = sum(path.stat().st_size for path in candidate_paths.values())
    total_bytes = label_bytes + candidate_bytes
    consumed = 0
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        connection = configure_database(database_path)
        overall, stage = progress_pair(label_bytes, 0, total_bytes, "stage=1/2 index-reused-and-new-labels")
        pending_commit = 0
        for origin, paths in (("reused_existing", reused_paths), ("new_terra_medium", new_paths)):
            for expected_dataset, path in sorted(paths.items()):
                for row, raw_size in json_line_count(path):
                    dataset, _sample, _stable = validate_label(row, path)
                    if dataset != expected_dataset:
                        raise ValueError(f"Mixed dataset in {path}: {dataset} != {expected_dataset}")
                    inserted = insert_label(connection, row, origin, path)
                    if not inserted:
                        raise ValueError(f"Duplicate reuse key across merge inputs: {dataset} in {path}")
                    counts[dataset][f"indexed_{origin}"] += 1
                    pending_commit += 1
                    if pending_commit >= 10_000:
                        connection.commit()
                        pending_commit = 0
                    stage.update(raw_size)
                    overall.update(raw_size)
                    consumed += raw_size
        connection.commit()
        stage.close()
        overall.close()

        overall, stage = progress_pair(candidate_bytes, consumed, total_bytes, "stage=2/2 merge-and-verify-top8")
        outputs: list[tuple[Path, Path]] = []
        for dataset, path in sorted(candidate_paths.items()):
            output_path = args.output_root / dataset / "codex_semantic_labels.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(output_path.name + ".tmp")
            outputs.append((temporary, output_path))
            with temporary.open("w", encoding="utf-8") as handle:
                for row, raw_size in json_line_count(path):
                    row_dataset = str(row.get("dataset") or "").lower()
                    sample_id = str(row.get("sample_id") or "")
                    if row_dataset != dataset:
                        raise ValueError(f"Mixed candidate dataset in {path}")
                    for document in ordered_documents(row, int(prepared["docs_per_question"])):
                        stable_id = stable_document_id(document)
                        found = connection.execute(
                            "SELECT payload, origin FROM labels WHERE dataset=? AND sample_id=? AND doc_stable_id=?",
                            (dataset, sample_id, stable_id),
                        ).fetchone()
                        if found is None:
                            raise RuntimeError(
                                f"Missing semantic label for {(dataset, sample_id, stable_id)}"
                            )
                        label = remap_label(json.loads(found[0]), row, document, str(found[1]))
                        handle.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[dataset]["final_pairs"] += 1
                        counts[dataset][f"final_{found[1]}"] += 1
                        counts[dataset][f"label_{label['semantic_label']}"] += 1
                    counts[dataset]["final_questions"] += 1
                    stage.update(raw_size)
                    overall.update(raw_size)
                    consumed += raw_size
        for temporary, target in outputs:
            os.replace(temporary, target)
        stage.close()
        overall.close()
        connection.close()

        for dataset in datasets:
            expected = int(prepared["datasets"][dataset]["pairs"])
            actual = counts[dataset]["final_pairs"]
            if actual != expected:
                raise RuntimeError(f"Final pair count mismatch for {dataset}: {actual} != {expected}")
            if counts[dataset]["final_reused_existing"] != int(
                prepared["datasets"][dataset]["reused_pairs"]
            ):
                raise RuntimeError(f"Reused pair count mismatch for {dataset}")
            if counts[dataset]["final_new_terra_medium"] != int(
                prepared["datasets"][dataset]["pending_pairs"]
            ):
                raise RuntimeError(f"New pair count mismatch for {dataset}")

        output_manifest = {
            "status": "complete",
            "created_at": utc_now(),
            "annotation_version": ANNOTATION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "codex_model_request": "gpt-5.6-terra",
            "codex_reasoning_effort": "medium",
            "web_search_enabled": False,
            "docs_per_question": prepared["docs_per_question"],
            "reuse_key": prepared["reuse_key"],
            "input_fingerprint": input_fingerprint,
            "input_identity": input_identity,
            "datasets": {dataset: dict(counts[dataset]) for dataset in datasets},
            "source_prepare_manifest": str(prepare_manifest_path.resolve()),
            "source_new_manifest": str(new_manifest_path.resolve()),
            "label_definitions": new_manifest.get("label_definitions"),
            "topic_relation_definitions": new_manifest.get("topic_relation_definitions"),
        }
        atomic_json(final_manifest_path, output_manifest)
        logging.info(
            "Incremental semantic-label merge complete: pairs=%d reused=%d new=%d output=%s",
            sum(value["final_pairs"] for value in counts.values()),
            sum(value["final_reused_existing"] for value in counts.values()),
            sum(value["final_new_terra_medium"] for value in counts.values()),
            args.output_root,
        )
    finally:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(database_path) + suffix)
            if candidate.exists():
                candidate.unlink()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.command == "prepare":
        prepare(args)
    elif args.command == "merge":
        merge(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

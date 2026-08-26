from __future__ import annotations

"""Reuse semantic labels shared by two retrieval runs and materialize only the delta.

The semantic decision is independent of rerank rank, so an existing label is
reusable when ``(dataset, sample_id, doc_stable_id)`` is unchanged.  Prepare
mode writes one pending candidate file per number of missing documents (1..K),
allowing the unchanged Codex labeler to process compact question-grouped
batches.  Finalize mode combines reused and newly generated labels in the exact
order of the new candidate files and refuses omissions or duplicates.
"""

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm


VALID_LABELS = {
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
}
ANNOTATION_VERSION = "rag2_codex_evidence_utility_label_v2"
PROMPT_VERSION = "rag2_codex_evidence_utility_prompt_v3_compact_item_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or finalize pair-level semantic-label reuse.")
    parser.add_argument("--mode", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--candidates-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--existing-labels-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--delta-root", type=Path, required=True)
    parser.add_argument("--label-runs-root", type=Path, default=None)
    parser.add_argument("--final-output-root", type=Path, default=None)
    parser.add_argument("--docs-per-question", type=int, default=8)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_number}")
            yield value


def line_count(path: Path) -> int:
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""))


def dataset_from_jsonl(path: Path) -> str:
    first = next(iter_jsonl(path), None)
    dataset = str((first or {}).get("dataset") or "").lower()
    if not dataset:
        raise ValueError(f"Cannot determine dataset from {path}")
    return dataset


def map_by_dataset(paths: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset = dataset_from_jsonl(path)
        if dataset in result:
            raise ValueError(f"Duplicate dataset input: {dataset}")
        result[dataset] = path
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    if not value:
        raise ValueError("Candidate document has no stable ID")
    return str(value)


def ranked_documents(row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    documents = row.get("candidate_documents")
    if not isinstance(documents, list) or len(documents) < top_k:
        raise ValueError(f"Expected Top-{top_k} documents for {row.get('sample_id')}")
    ordered = sorted(documents, key=lambda item: int(item.get("rerank_rank") or 10**9))[:top_k]
    ranks = [int(item.get("rerank_rank") or index) for index, item in enumerate(ordered, start=1)]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"Duplicate rerank ranks for {row.get('sample_id')}")
    return ordered


def pair_id(sample_id: str, document: dict[str, Any]) -> str:
    rank = int(document.get("rerank_rank"))
    return f"{sample_id}::{rank}::{stable_id(document)}"


def connect_index(path: Path, rebuild: bool) -> sqlite3.Connection:
    if rebuild:
        path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            dataset TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            doc_stable_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY(dataset, sample_id, doc_stable_id)
        )
        """
    )
    connection.commit()
    return connection


def validate_label(row: dict[str, Any], path: Path) -> None:
    if str(row.get("semantic_label") or "") not in VALID_LABELS:
        raise ValueError(f"Invalid semantic label in {path}: {row.get('semantic_label')!r}")
    if not row.get("sample_id") or not row.get("doc_stable_id"):
        raise ValueError(f"Missing semantic label identity in {path}")


def build_existing_index(paths: dict[str, Path], index_path: Path) -> tuple[sqlite3.Connection, dict[str, int]]:
    connection = connect_index(index_path, rebuild=True)
    counts: dict[str, int] = {}
    for dataset, path in paths.items():
        total = line_count(path)
        inserted = 0
        with connection:
            for row in tqdm(
                iter_jsonl(path),
                total=total,
                desc=f"SemanticDelta stage 1/3 index existing {dataset}",
                unit="pair",
                dynamic_ncols=True,
            ):
                validate_label(row, path)
                if str(row.get("dataset") or "").lower() != dataset:
                    raise ValueError(f"Dataset mismatch in {path}")
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                before = connection.total_changes
                connection.execute(
                    "INSERT OR IGNORE INTO labels(dataset,sample_id,doc_stable_id,payload) VALUES(?,?,?,?)",
                    (dataset, str(row["sample_id"]), str(row["doc_stable_id"]), payload),
                )
                if connection.total_changes == before:
                    previous = connection.execute(
                        "SELECT payload FROM labels WHERE dataset=? AND sample_id=? AND doc_stable_id=?",
                        (dataset, str(row["sample_id"]), str(row["doc_stable_id"])),
                    ).fetchone()
                    if previous is None or json.loads(previous[0]).get("semantic_label") != row.get("semantic_label"):
                        raise ValueError(f"Conflicting duplicate semantic label for {row['sample_id']}::{row['doc_stable_id']}")
                else:
                    inserted += 1
        counts[dataset] = inserted
    return connection, counts


def remap_label(old: dict[str, Any], dataset: str, sample_id: str, document: dict[str, Any]) -> dict[str, Any]:
    rank = int(document.get("rerank_rank"))
    new_pair_id = pair_id(sample_id, document)
    return {
        "id": new_pair_id,
        "pair_id": new_pair_id,
        "dataset": dataset,
        "sample_id": sample_id,
        "doc_rank": rank,
        "source": str(document.get("source") or "unknown"),
        "doc_stable_id": stable_id(document),
        "title": " ".join(str(document.get("title") or "").split()),
        "semantic_label": old["semantic_label"],
        "topic_relation": old.get("topic_relation"),
        "confidence": old.get("confidence"),
        "evidence_sentence_indices": old.get("evidence_sentence_indices") or [],
        "short_reason": old.get("short_reason"),
    }


def minimal_pending_row(row: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "sample_id": row["sample_id"],
        "row_idx": row.get("row_idx"),
        "split": row.get("split"),
        "question": row["question"],
        "options": row["options"],
        "answer": row.get("answer"),
        "answers": row.get("answers"),
        "candidate_documents": documents,
    }


def prepare(args: argparse.Namespace) -> None:
    if not args.existing_labels_paths:
        raise ValueError("prepare mode requires --existing-labels-paths")
    candidates = map_by_dataset(args.candidates_paths)
    labels = map_by_dataset(args.existing_labels_paths)
    if set(candidates) != set(labels):
        raise ValueError(f"Dataset mismatch: candidates={sorted(candidates)} labels={sorted(labels)}")
    args.delta_root.mkdir(parents=True, exist_ok=True)
    connection, indexed = build_existing_index(labels, args.delta_root / "existing_labels.sqlite")
    summary: dict[str, Any] = {
        "mode": "prepare",
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "reuse_identity": ["dataset", "sample_id", "doc_stable_id"],
        "docs_per_question": args.docs_per_question,
        "datasets": {},
    }
    try:
        for dataset, path in candidates.items():
            dataset_root = args.delta_root / dataset
            dataset_root.mkdir(parents=True, exist_ok=True)
            reused_tmp = dataset_root / f".reused_labels.jsonl.{os.getpid()}.tmp"
            bucket_tmps = {
                k: dataset_root / f".pending_k{k}.jsonl.{os.getpid()}.tmp"
                for k in range(1, args.docs_per_question + 1)
            }
            bucket_handles = {k: value.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) for k, value in bucket_tmps.items()}
            counters = Counter()
            total = line_count(path)
            with reused_tmp.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as reused_handle:
                try:
                    for row in tqdm(
                        iter_jsonl(path),
                        total=total,
                        desc=f"SemanticDelta stage 2/3 split new candidates {dataset}",
                        unit="question",
                        dynamic_ncols=True,
                    ):
                        sample_id = str(row.get("sample_id") or "")
                        if str(row.get("dataset") or "").lower() != dataset or not sample_id:
                            raise ValueError(f"Invalid candidate identity in {path}")
                        pending: list[dict[str, Any]] = []
                        for document in ranked_documents(row, args.docs_per_question):
                            key = (dataset, sample_id, stable_id(document))
                            found = connection.execute(
                                "SELECT payload FROM labels WHERE dataset=? AND sample_id=? AND doc_stable_id=?", key
                            ).fetchone()
                            if found is None:
                                pending.append(document)
                                counters["pending_pairs"] += 1
                            else:
                                reused_handle.write(json.dumps(remap_label(json.loads(found[0]), dataset, sample_id, document), ensure_ascii=False) + "\n")
                                counters["reused_pairs"] += 1
                        counters["questions"] += 1
                        counters[f"questions_pending_{len(pending)}"] += 1
                        if pending:
                            bucket_handles[len(pending)].write(json.dumps(minimal_pending_row(row, pending), ensure_ascii=False) + "\n")
                finally:
                    for handle in bucket_handles.values():
                        handle.close()
            os.replace(reused_tmp, dataset_root / "reused_labels.jsonl")
            for k, temporary in bucket_tmps.items():
                destination = dataset_root / f"pending_k{k}.jsonl"
                if temporary.stat().st_size:
                    os.replace(temporary, destination)
                else:
                    temporary.unlink()
                    destination.unlink(missing_ok=True)
            counters["indexed_existing_pairs"] = indexed[dataset]
            summary["datasets"][dataset] = dict(sorted(counters.items()))
    finally:
        connection.close()
    summary["total_reused_pairs"] = sum(value["reused_pairs"] for value in summary["datasets"].values())
    summary["total_pending_pairs"] = sum(value["pending_pairs"] for value in summary["datasets"].values())
    write_json_atomic(args.delta_root / "prepare_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_final_label_index(args: argparse.Namespace) -> sqlite3.Connection:
    index_path = args.delta_root / "finalize_labels.sqlite"
    index_path.unlink(missing_ok=True)
    connection = sqlite3.connect(index_path)
    connection.execute("CREATE TABLE labels(pair_id TEXT PRIMARY KEY, payload TEXT NOT NULL, provenance TEXT NOT NULL)")
    sources: list[tuple[Path, str]] = []
    for candidates_path in args.candidates_paths:
        dataset = dataset_from_jsonl(candidates_path)
        sources.append((args.delta_root / dataset / "reused_labels.jsonl", "reused"))
        for k in range(1, args.docs_per_question + 1):
            pending = args.delta_root / dataset / f"pending_k{k}.jsonl"
            if not pending.is_file():
                continue
            generated = args.label_runs_root / f"k{k}" / dataset / "codex_semantic_labels.jsonl"
            sources.append((generated, f"new_k{k}"))
    for path, provenance in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        total = line_count(path)
        with connection:
            for row in tqdm(
                iter_jsonl(path), total=total, desc=f"SemanticDelta finalize 1/2 index {provenance}", unit="pair", dynamic_ncols=True
            ):
                validate_label(row, path)
                pid = str(row.get("pair_id") or row.get("id") or "")
                try:
                    connection.execute(
                        "INSERT INTO labels(pair_id,payload,provenance) VALUES(?,?,?)",
                        (pid, json.dumps(row, ensure_ascii=False, separators=(",", ":")), provenance),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate final pair label: {pid}") from exc
    return connection


def finalize(args: argparse.Namespace) -> None:
    if args.label_runs_root is None or args.final_output_root is None:
        raise ValueError("finalize mode requires --label-runs-root and --final-output-root")
    connection = load_final_label_index(args)
    summary: dict[str, Any] = {
        "mode": "finalize",
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "datasets": {},
    }
    try:
        for path in args.candidates_paths:
            dataset = dataset_from_jsonl(path)
            output = args.final_output_root / dataset / "codex_semantic_labels.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            counters = Counter()
            total = line_count(path)
            with temporary.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
                for row in tqdm(
                    iter_jsonl(path), total=total, desc=f"SemanticDelta finalize 2/2 assemble {dataset}", unit="question", dynamic_ncols=True
                ):
                    sample_id = str(row["sample_id"])
                    for document in ranked_documents(row, args.docs_per_question):
                        pid = pair_id(sample_id, document)
                        found = connection.execute("SELECT payload,provenance FROM labels WHERE pair_id=?", (pid,)).fetchone()
                        if found is None:
                            raise RuntimeError(f"Missing final semantic label: {pid}")
                        label = json.loads(found[0])
                        # Generated rows already have the target identity; reused rows were remapped in prepare mode.
                        if str(label.get("pair_id") or "") != pid:
                            raise RuntimeError(f"Final semantic identity mismatch: {pid}")
                        handle.write(json.dumps(label, ensure_ascii=False) + "\n")
                        counters["pairs"] += 1
                        counters[f"provenance_{found[1]}"] += 1
                        counters[f"label_{label['semantic_label']}"] += 1
                    counters["questions"] += 1
            os.replace(temporary, output)
            summary["datasets"][dataset] = dict(sorted(counters.items()))
    finally:
        connection.close()
    write_json_atomic(args.final_output_root / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.docs_per_question <= 0:
        raise ValueError("--docs-per-question must be positive")
    if args.mode == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()

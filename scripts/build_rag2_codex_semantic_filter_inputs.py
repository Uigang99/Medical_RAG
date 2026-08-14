#!/usr/bin/env python3
from __future__ import annotations

"""Materialize a controlled document-level RAG² filter dataset from Codex labels.

The original question-level 8:1:1 assignments and the released RAG²
evidence-then-question filter prompt are preserved.  The only supervision
change is replacing PPL-derived labels with the semantic evidence labels:

  Helpful     = direct_support or supporting_evidence
  Not Helpful = no_evidence, misleading_evidence, or indeterminate_or_mixed

The production Codex annotation covers rerank Top-8 candidates.  This builder
therefore intentionally never reads ranks 9 or 10 and never mixes PPL labels
into the resulting dataset.
"""

import argparse
import json
import logging
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import build_official_filter_input, clean_text, format_options


SPLITS = ("train", "val", "test")
SEMANTIC_HELPFUL = {"direct_support", "supporting_evidence"}
SEMANTIC_NOT_HELPFUL = {"no_evidence", "misleading_evidence", "indeterminate_or_mixed"}
VALID_SEMANTIC_LABELS = SEMANTIC_HELPFUL | SEMANTIC_NOT_HELPFUL
MATERIALIZATION_VERSION = "rag2_codex_semantic_top8_binary_filter_input_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build official RAG² document-filter inputs from Codex semantic Top-k labels."
    )
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--candidates-path",
        type=Path,
        required=True,
        help="quality-selected candidates_top32.jsonl used for the Codex Top-8 labelling run.",
    )
    parser.add_argument(
        "--codex-labels-path",
        type=Path,
        required=True,
        help="Final codex_semantic_labels.jsonl for the same dataset.",
    )
    parser.add_argument(
        "--reference-split-root",
        type=Path,
        required=True,
        help=(
            "Existing document-level split root. Only its sample_ids/{train,val,test}.txt "
            "files are used, preserving the original question-level split."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Rerank prefix covered by Codex labels. Production contract is Top-8.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=0,
        help="Optional evidence cap. Zero preserves the exact complete chunk, matching the PPL document baseline.",
    )
    parser.add_argument("--sqlite-work-dir", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every join and print counts without creating JSONL output files.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


def require_file(path: Path, description: str) -> Path:
    value = path.resolve()
    if not value.is_file():
        raise FileNotFoundError(f"Missing {description}: {value}")
    return value


def load_question_split_assignments(reference_root: Path, dataset: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split in SPLITS:
        path = require_file(
            reference_root / dataset / "sample_ids" / f"{split}.txt",
            f"{split} sample-ID assignment file",
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                sample_id = line.strip()
                if not sample_id:
                    continue
                previous = assignments.setdefault(sample_id, split)
                if previous != split:
                    raise ValueError(
                        f"Question split leakage for {sample_id}: {previous} and {split} "
                        f"({path}:{line_number})"
                    )
    if not assignments:
        raise ValueError("No sample IDs were loaded from the reference split root.")
    return assignments


def create_label_index(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE codex_labels (
            pair_id TEXT PRIMARY KEY,
            semantic_label TEXT NOT NULL,
            topic_relation TEXT,
            confidence REAL NOT NULL,
            evidence_sentence_indices TEXT NOT NULL,
            short_reason TEXT NOT NULL,
            source TEXT NOT NULL,
            doc_stable_id TEXT NOT NULL,
            doc_rank INTEGER NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def index_codex_labels(connection: sqlite3.Connection, path: Path, dataset: str) -> dict[str, Any]:
    statement = """
        INSERT INTO codex_labels (
            pair_id,semantic_label,topic_relation,confidence,evidence_sentence_indices,
            short_reason,source,doc_stable_id,doc_rank
        ) VALUES (?,?,?,?,?,?,?,?,?)
    """
    pending: list[tuple[Any, ...]] = []
    counts: Counter[str] = Counter()
    total = 0
    for row in tqdm(iter_jsonl(path), desc=f"index-codex:{dataset}", unit="pair"):
        pair_id = str(row.get("pair_id") or row.get("id") or "")
        semantic_label = str(row.get("semantic_label") or "")
        if not pair_id or semantic_label not in VALID_SEMANTIC_LABELS:
            raise ValueError(f"Invalid Codex semantic row: pair_id={pair_id!r}, label={semantic_label!r}")
        if str(row.get("dataset") or "") != dataset:
            raise ValueError(f"Codex dataset mismatch: {pair_id}")
        try:
            confidence = float(row.get("confidence"))
            doc_rank = int(row.get("doc_rank"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Codex confidence/rank: {pair_id}") from error
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Codex confidence outside [0,1]: {pair_id}")
        source = str(row.get("source") or "")
        stable_id = str(row.get("doc_stable_id") or "")
        if not source or not stable_id:
            raise ValueError(f"Missing Codex document provenance: {pair_id}")
        pending.append(
            (
                pair_id,
                semantic_label,
                row.get("topic_relation"),
                confidence,
                json.dumps(row.get("evidence_sentence_indices") or [], ensure_ascii=False),
                str(row.get("short_reason") or ""),
                source,
                stable_id,
                doc_rank,
            )
        )
        total += 1
        counts[semantic_label] += 1
        if len(pending) >= 10_000:
            try:
                connection.executemany(statement, pending)
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate Codex pair_id detected") from error
            pending.clear()
    if pending:
        try:
            connection.executemany(statement, pending)
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate Codex pair_id detected") from error
    connection.commit()
    return {"rows": total, "semantic_label": dict(sorted(counts.items()))}


def target_from_semantic_label(value: str) -> tuple[str, str]:
    if value in SEMANTIC_HELPFUL:
        return "helpful", "Helpful"
    if value in SEMANTIC_NOT_HELPFUL:
        return "not helpful", "Not Helpful"
    raise ValueError(f"Unexpected semantic label: {value}")


def evidence_text(document: dict[str, Any], max_doc_chars: int) -> str:
    text = clean_text(document.get("text")) or clean_text(document.get("title"))
    if not text:
        raise ValueError(f"Candidate document has no text/title: {document.get('stable_id')}")
    if max_doc_chars > 0 and len(text) > max_doc_chars:
        return text[: max_doc_chars - 3].rstrip() + "..."
    return text


def make_pair_id(sample_id: str, rank: int, stable_id: str) -> str:
    return f"{sample_id}::{rank}::{stable_id}"


def make_output_row(
    candidate_row: dict[str, Any],
    document: dict[str, Any],
    rank: int,
    split: str,
    label_row: sqlite3.Row,
    max_doc_chars: int,
) -> dict[str, Any]:
    sample_id = str(candidate_row["sample_id"])
    stable_id = str(document.get("stable_id") or "")
    source = str(document.get("source") or "")
    if source != label_row["source"] or stable_id != label_row["doc_stable_id"] or rank != int(label_row["doc_rank"]):
        raise ValueError(f"Codex/candidate provenance mismatch: {label_row['pair_id']}")
    target, public_label = target_from_semantic_label(str(label_row["semantic_label"]))
    return {
        "materialization_version": MATERIALIZATION_VERSION,
        "id": str(label_row["pair_id"]),
        "pair_id": str(label_row["pair_id"]),
        "dataset": candidate_row.get("dataset"),
        "sample_id": sample_id,
        "row_idx": candidate_row.get("row_idx"),
        "split": split,
        "source": source,
        "doc_stable_id": stable_id,
        "doc_rank": rank,
        "input": build_official_filter_input(
            question=clean_text(candidate_row.get("question")),
            options=format_options(candidate_row.get("options") if isinstance(candidate_row.get("options"), dict) else {}),
            evidence=evidence_text(document, max_doc_chars),
        ),
        "target": target,
        "label": public_label,
        "pseudo_label": public_label,
        "label_origin": "codex_semantic_evidence_utility_v2",
        "codex_semantic_label": str(label_row["semantic_label"]),
        "codex_topic_relation": label_row["topic_relation"],
        "codex_confidence": float(label_row["confidence"]),
        "codex_evidence_sentence_indices": json.loads(label_row["evidence_sentence_indices"]),
        "codex_short_reason": str(label_row["short_reason"]),
        "answer": candidate_row.get("answer"),
        "answers": candidate_row.get("answers"),
    }


def empty_split_counter() -> dict[str, Any]:
    return {
        "rows": 0,
        "sample_ids": set(),
        "target": Counter(),
        "semantic_label": Counter(),
        "source": Counter(),
        "doc_rank": Counter(),
    }


def serialize_split_counter(counter: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(counter["rows"]),
        "sample_ids": len(counter["sample_ids"]),
        "target": dict(sorted(counter["target"].items())),
        "semantic_label": dict(sorted(counter["semantic_label"].items())),
        "source": dict(sorted(counter["source"].items())),
        "doc_rank": dict(sorted(counter["doc_rank"].items(), key=lambda item: int(item[0]))),
    }


def materialize(args: argparse.Namespace, connection: sqlite3.Connection, assignments: dict[str, str]) -> dict[str, Any]:
    output_dir = args.output_root / args.dataset
    handles: dict[str, Any] = {}
    if not args.dry_run:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output dataset directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        handles = {
            split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", buffering=64 * 1024 * 1024)
            for split in SPLITS
        }

    counters = {split: empty_split_counter() for split in SPLITS}
    candidate_questions = 0
    candidate_pairs = 0
    try:
        for row in tqdm(iter_jsonl(args.candidates_path), desc=f"materialize-codex:{args.dataset}", unit="question"):
            if str(row.get("dataset") or "") != args.dataset:
                raise ValueError(f"Candidate dataset mismatch: {row.get('sample_id')}")
            sample_id = str(row.get("sample_id") or "")
            split = assignments.get(sample_id)
            if split is None:
                raise ValueError(f"Candidate question missing reference split assignment: {sample_id}")
            documents = row.get("candidate_documents")
            if not isinstance(documents, list) or len(documents) < args.top_k:
                raise ValueError(f"Candidate row has fewer than Top-{args.top_k} reranked documents: {sample_id}")
            candidate_questions += 1
            for rank, document in enumerate(documents[: args.top_k], start=1):
                if not isinstance(document, dict):
                    raise ValueError(f"Non-object candidate document: {sample_id} rank={rank}")
                stable_id = str(document.get("stable_id") or "")
                pair_id = make_pair_id(sample_id, rank, stable_id)
                label_row = connection.execute(
                    "SELECT * FROM codex_labels WHERE pair_id = ?", (pair_id,)
                ).fetchone()
                if label_row is None:
                    raise ValueError(f"Missing Codex label for candidate pair: {pair_id}")
                if int(label_row["used"]):
                    raise ValueError(f"Candidate pair occurs more than once: {pair_id}")
                output = make_output_row(row, document, rank, split, label_row, args.max_doc_chars)
                connection.execute("UPDATE codex_labels SET used = 1 WHERE pair_id = ?", (pair_id,))
                candidate_pairs += 1
                counter = counters[split]
                counter["rows"] += 1
                counter["sample_ids"].add(sample_id)
                counter["target"][output["target"]] += 1
                counter["semantic_label"][output["codex_semantic_label"]] += 1
                counter["source"][source := str(output["source"])] += 1
                counter["doc_rank"][str(rank)] += 1
                if not args.dry_run:
                    handles[split].write(json.dumps(output, ensure_ascii=False) + "\n")
        connection.commit()
    finally:
        for handle in handles.values():
            handle.close()

    unused = int(connection.execute("SELECT COUNT(*) FROM codex_labels WHERE used = 0").fetchone()[0])
    if unused:
        first = connection.execute("SELECT pair_id FROM codex_labels WHERE used = 0 ORDER BY pair_id LIMIT 1").fetchone()[0]
        raise RuntimeError(f"{unused} Codex label(s) were not found in candidate Top-{args.top_k}; first={first}")
    expected_pairs = candidate_questions * args.top_k
    if candidate_pairs != expected_pairs:
        raise RuntimeError(f"Candidate count mismatch: pairs={candidate_pairs}, expected={expected_pairs}")
    return {
        "candidate_questions": candidate_questions,
        "candidate_pairs": candidate_pairs,
        "splits": {split: serialize_split_counter(counter) for split, counter in counters.items()},
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.top_k != 8:
        raise ValueError("This production builder is intentionally fixed to --top-k 8, matching Codex coverage.")
    if args.max_doc_chars < 0:
        raise ValueError("--max-doc-chars must be non-negative")
    args.candidates_path = require_file(args.candidates_path, "candidate JSONL")
    args.codex_labels_path = require_file(args.codex_labels_path, "Codex labels JSONL")
    args.reference_split_root = args.reference_split_root.resolve()
    args.sqlite_work_dir = args.sqlite_work_dir.resolve()
    if not args.sqlite_work_dir.is_dir():
        raise FileNotFoundError(f"SQLite work directory does not exist: {args.sqlite_work_dir}")

    assignments = load_question_split_assignments(args.reference_split_root, args.dataset)
    with tempfile.TemporaryDirectory(prefix=f"rag2_codex_{args.dataset}_", dir=args.sqlite_work_dir) as temporary_dir:
        db_path = Path(temporary_dir) / "codex_labels.sqlite"
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        try:
            create_label_index(connection)
            label_summary = index_codex_labels(connection, args.codex_labels_path, args.dataset)
            materialized = materialize(args, connection, assignments)
        finally:
            connection.close()

    manifest = {
        "type": "rag2_codex_semantic_binary_filter_inputs",
        "materialization_version": MATERIALIZATION_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "training_label_mode": "binary",
        "training_target_labels": ["helpful", "not helpful"],
        "label_mapping": {
            "helpful": sorted(SEMANTIC_HELPFUL),
            "not_helpful": sorted(SEMANTIC_NOT_HELPFUL),
        },
        "label_protocol": "Codex semantic evidence-utility annotation; no PPL labels or ranks 9/10 are used.",
        "codex_labels_path": str(args.codex_labels_path),
        "candidates_path": str(args.candidates_path),
        "reference_split_root": str(args.reference_split_root),
        "question_split_source": "reference sample_ids/{train,val,test}.txt copied logically without re-splitting",
        "input_format": "official_rag2_evidence_then_question",
        "filter_input": {
            "format": "official RAG2 evidence-then-question template",
            "document_character_cap": args.max_doc_chars,
            "encoder_overlength_policy": "handled by the trainer: exclude > max_seq_length without truncation/windows",
        },
        "max_doc_chars": args.max_doc_chars,
        "codex_label_summary": label_summary,
        "materialized": materialized,
        "summary": {"splits": materialized["splits"]},
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        output_dir = args.output_root / args.dataset
        manifest["files"] = {split: str((output_dir / f"{split}.jsonl").resolve()) for split in SPLITS}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    logging.info("Codex Top-8 filter-input materialization complete: %s", json.dumps(manifest["materialized"], ensure_ascii=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

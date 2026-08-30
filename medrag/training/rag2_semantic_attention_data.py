from __future__ import annotations

"""Validated, question-grouped data for semantic document-attention training.

The semantic classifier inputs used elsewhere in this repository omit
``indeterminate_or_mixed`` rows.  That is appropriate for a four-way
classifier, but not for attention training: removing the row also removes the
document from the Llama context.  This module instead joins the authoritative
Top-8 candidate rows with the raw five-way annotations and masks only the
semantic loss for mixed documents.

The persistent SQLite index is deliberately question-grouped.  It avoids
loading the multi-gigabyte MedMCQA candidate JSONL into memory and gives a
PyTorch DataLoader deterministic random access at question granularity.
"""

import hashlib
import json
import math
import os
import random
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


SCHEMA_VERSION = "rag2_semantic_attention_grouped_sqlite_v1"
SPLITS = ("train", "val", "test")
SEMANTIC_LABELS = (
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
)
SEMANTIC_CLASS_IDS = {
    "misleading_evidence": 0,
    "no_evidence": 1,
    "supporting_evidence": 2,
    "direct_support": 3,
}
SEMANTIC_SUPPORT_TARGETS = {
    "direct_support": 1,
    "supporting_evidence": 1,
    "no_evidence": 0,
    "misleading_evidence": 0,
}
MIXED_LABEL = "indeterminate_or_mixed"

__all__ = [
    "DeterministicQuestionBatchSampler",
    "MIXED_LABEL",
    "NoRAGTrace",
    "RAG2SemanticAttentionDataset",
    "SEMANTIC_CLASS_IDS",
    "SEMANTIC_LABELS",
    "SEMANTIC_SUPPORT_TARGETS",
    "SemanticAttentionBuildPlan",
    "SemanticAttentionDataSources",
    "SemanticAttentionDocument",
    "SemanticAttentionIndexResult",
    "SemanticAttentionQuestion",
    "build_semantic_attention_index",
    "make_semantic_attention_build_plan",
]


@dataclass(frozen=True)
class SemanticAttentionDataSources:
    dataset: str
    candidates_path: Path
    semantic_labels_path: Path
    split_ids_root: Path
    no_rag_path: Path
    expected_documents: int = 8

    def resolved(self) -> "SemanticAttentionDataSources":
        return SemanticAttentionDataSources(
            dataset=self.dataset,
            candidates_path=self.candidates_path.expanduser().resolve(),
            semantic_labels_path=self.semantic_labels_path.expanduser().resolve(),
            split_ids_root=self.split_ids_root.expanduser().resolve(),
            no_rag_path=self.no_rag_path.expanduser().resolve(),
            expected_documents=self.expected_documents,
        )


@dataclass(frozen=True)
class SemanticAttentionBuildPlan:
    schema_version: str
    dataset: str
    expected_documents: int
    split_id_counts: Mapping[str, int]
    expected_questions: int | None
    expected_semantic_rows: int | None
    expected_no_rag_rows: int | None
    overall_work_units: int | None
    source_fingerprint: str
    source_files: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NoRAGTrace:
    sample_id: str
    valid: bool
    gold_answer: str
    predicted_answer: str | None
    answer_correct: bool | None
    canonical_generation: str
    choice_logprobs: Mapping[str, float | None]


@dataclass(frozen=True)
class SemanticAttentionDocument:
    pair_id: str
    rank: int
    stable_id: str
    source: str
    title: str
    text: str
    semantic_label: str
    semantic_class_id: int | None
    semantic_support_target: int | None
    semantic_loss_mask: bool
    semantic_confidence: float
    topic_relation: str | None
    evidence_sentence_indices: tuple[int, ...]
    short_reason: str


@dataclass(frozen=True)
class SemanticAttentionQuestion:
    dataset: str
    split: str
    sample_id: str
    row_idx: int | None
    question: str
    options: Mapping[str, str]
    gold_answers: tuple[str, ...]
    no_rag: NoRAGTrace
    documents: tuple[SemanticAttentionDocument, ...]

    def __post_init__(self) -> None:
        ranks = tuple(document.rank for document in self.documents)
        if ranks != tuple(range(1, len(self.documents) + 1)):
            raise ValueError(f"Documents are not in contiguous rerank order for {self.sample_id}: {ranks}")


@dataclass(frozen=True)
class SemanticAttentionIndexResult:
    index_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    reused: bool


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, value


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_split_assignments(split_ids_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    assignments: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in SPLITS:
        path = _required_file(split_ids_root / f"{split}.txt", f"{split} sample-ID file")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                sample_id = line.strip()
                if not sample_id:
                    continue
                previous = assignments.get(sample_id)
                if previous is not None:
                    raise ValueError(
                        f"Question split overlap for {sample_id}: {previous} and {split} "
                        f"({path}:{line_number})"
                    )
                assignments[sample_id] = split
                count += 1
        counts[split] = count
    if not assignments:
        raise ValueError(f"No sample IDs found under {split_ids_root}")
    return assignments, counts


def make_semantic_attention_build_plan(
    sources: SemanticAttentionDataSources,
) -> SemanticAttentionBuildPlan:
    sources = sources.resolved()
    if not sources.dataset:
        raise ValueError("dataset must be non-empty")
    if sources.expected_documents <= 0:
        raise ValueError("expected_documents must be positive")
    _required_file(sources.candidates_path, "Top-k candidates JSONL")
    _required_file(sources.semantic_labels_path, "raw semantic labels JSONL")
    _required_file(sources.no_rag_path, "no-RAG trace JSONL")
    _, split_counts = _load_split_assignments(sources.split_ids_root)

    candidate_manifest = _read_json(sources.candidates_path.parent / "candidate_manifest.json") or {}
    annotation_manifest = _read_json(sources.semantic_labels_path.parent.parent / "manifest.json") or {}
    no_rag_manifest = _read_json(sources.no_rag_path.parent / "manifest.json") or {}
    candidate_questions = _optional_int(candidate_manifest.get("selected_question_count"))
    dataset_annotation = annotation_manifest.get("datasets", {}).get(sources.dataset, {})
    semantic_rows = _optional_int(dataset_annotation.get("final_pairs"))
    annotation_questions = _optional_int(dataset_annotation.get("final_questions"))
    no_rag_rows = _optional_int(no_rag_manifest.get("rows"))
    if candidate_questions is not None and annotation_questions is not None:
        if candidate_questions != annotation_questions:
            raise ValueError(
                "Candidate/semantic manifest question mismatch: "
                f"{candidate_questions} != {annotation_questions}"
            )
    if candidate_questions is not None and semantic_rows is not None:
        expected_pairs = candidate_questions * sources.expected_documents
        if semantic_rows != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} semantic rows for {candidate_questions} questions, "
                f"found {semantic_rows} in the annotation manifest"
            )

    files: dict[str, Mapping[str, Any]] = {
        "candidates": _file_identity(sources.candidates_path),
        "semantic_labels": _file_identity(sources.semantic_labels_path),
        "no_rag": _file_identity(sources.no_rag_path),
    }
    for split in SPLITS:
        files[f"split_{split}"] = _file_identity(sources.split_ids_root / f"{split}.txt")
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": sources.dataset,
        "expected_documents": sources.expected_documents,
        "files": files,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    totals = (semantic_rows, no_rag_rows, candidate_questions)
    overall = sum(value for value in totals if value is not None) if all(value is not None for value in totals) else None
    return SemanticAttentionBuildPlan(
        schema_version=SCHEMA_VERSION,
        dataset=sources.dataset,
        expected_documents=sources.expected_documents,
        split_id_counts=split_counts,
        expected_questions=candidate_questions,
        expected_semantic_rows=semantic_rows,
        expected_no_rag_rows=no_rag_rows,
        overall_work_units=overall,
        source_fingerprint=fingerprint,
        source_files=files,
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _manifest_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".manifest.json")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS semantic_labels (
            pair_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            doc_rank INTEGER NOT NULL,
            source TEXT NOT NULL,
            stable_id TEXT NOT NULL,
            semantic_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            topic_relation TEXT,
            evidence_indices_json TEXT NOT NULL,
            short_reason TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS no_rag (
            sample_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            valid INTEGER NOT NULL,
            gold_answer TEXT NOT NULL,
            predicted_answer TEXT,
            answer_correct INTEGER,
            canonical_generation TEXT NOT NULL,
            choice_logprobs_json TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS questions (
            sample_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            split_ordinal INTEGER NOT NULL,
            row_idx INTEGER,
            question TEXT NOT NULL,
            options_json TEXT NOT NULL,
            gold_answers_json TEXT NOT NULL,
            no_rag_valid INTEGER NOT NULL,
            no_rag_gold_answer TEXT NOT NULL,
            no_rag_prediction TEXT,
            no_rag_answer_correct INTEGER,
            no_rag_generation TEXT NOT NULL,
            no_rag_choice_logprobs_json TEXT NOT NULL,
            UNIQUE(split, split_ordinal)
        );
        CREATE TABLE IF NOT EXISTS documents (
            sample_id TEXT NOT NULL,
            doc_rank INTEGER NOT NULL,
            pair_id TEXT NOT NULL UNIQUE,
            stable_id TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            semantic_label TEXT NOT NULL,
            semantic_class_id INTEGER,
            semantic_support_target INTEGER,
            semantic_loss_mask INTEGER NOT NULL,
            semantic_confidence REAL NOT NULL,
            topic_relation TEXT,
            evidence_indices_json TEXT NOT NULL,
            short_reason TEXT NOT NULL,
            PRIMARY KEY(sample_id, doc_rank),
            FOREIGN KEY(sample_id) REFERENCES questions(sample_id)
        );
        CREATE INDEX IF NOT EXISTS questions_split_index ON questions(split, split_ordinal);
        CREATE INDEX IF NOT EXISTS documents_sample_index ON documents(sample_id, doc_rank);
        """
    )
    connection.commit()


def _set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False)),
    )


def _get_metadata(connection: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row is not None else default


def build_semantic_attention_index(
    sources: SemanticAttentionDataSources,
    index_path: Path,
    *,
    resume: bool = True,
    checkpoint_every: int = 1_000,
    show_progress: bool = True,
) -> SemanticAttentionIndexResult:
    """Build or resume a validated grouped SQLite index.

    Progress is checkpointed after ``checkpoint_every`` source rows.  A rerun
    with unchanged inputs skips completed rows and resumes the active stage.
    Completed indexes are reused only when every source path/size/mtime matches
    the stored fingerprint.
    """

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    sources = sources.resolved()
    plan = make_semantic_attention_build_plan(sources)
    index_path = index_path.expanduser().resolve()
    manifest_path = _manifest_path(index_path)
    existing_manifest = _read_json(manifest_path)
    if existing_manifest and existing_manifest.get("status") == "complete":
        if existing_manifest.get("source_fingerprint") != plan.source_fingerprint:
            raise ValueError(
                f"Completed index inputs changed; choose a new index path: {index_path}"
            )
        if not index_path.is_file():
            raise FileNotFoundError(f"Completed manifest exists but SQLite index is missing: {index_path}")
        return SemanticAttentionIndexResult(index_path, manifest_path, existing_manifest, True)
    if (index_path.exists() or manifest_path.exists()) and not resume:
        raise FileExistsError(f"Incomplete semantic-attention index already exists: {index_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    assignments, _ = _load_split_assignments(sources.split_ids_root)
    connection = _connect(index_path)
    _create_schema(connection)
    stored_fingerprint = _get_metadata(connection, "source_fingerprint")
    if stored_fingerprint is not None and stored_fingerprint != plan.source_fingerprint:
        connection.close()
        raise ValueError(f"Incomplete index was built from different inputs: {index_path}")
    _set_metadata(connection, "schema_version", SCHEMA_VERSION)
    _set_metadata(connection, "source_fingerprint", plan.source_fingerprint)
    _set_metadata(connection, "status", "building")
    connection.commit()

    stage_counts = _current_stage_counts(connection)
    overall = tqdm(
        total=plan.overall_work_units,
        initial=sum(stage_counts.values()),
        desc=f"SemanticAttentionData:{sources.dataset}",
        unit="item",
        disable=not show_progress,
    )
    try:
        if not _get_metadata(connection, "labels_complete", False):
            _index_semantic_labels(
                connection,
                sources,
                plan,
                manifest_path,
                overall,
                checkpoint_every,
                show_progress,
            )
        if not _get_metadata(connection, "no_rag_complete", False):
            _index_no_rag(
                connection,
                sources,
                plan,
                manifest_path,
                overall,
                checkpoint_every,
                show_progress,
            )
        if not _get_metadata(connection, "candidates_complete", False):
            _join_candidates(
                connection,
                sources,
                plan,
                assignments,
                manifest_path,
                overall,
                checkpoint_every,
                show_progress,
            )
        summary = _validate_completed_index(connection, sources, plan)
        _set_metadata(connection, "status", "complete")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = _progress_manifest(connection, plan, status="complete", summary=summary)
        _write_json_atomic(manifest_path, manifest)
    except Exception as error:
        connection.rollback()
        _set_metadata(connection, "status", "incomplete")
        _set_metadata(connection, "last_error", f"{type(error).__name__}: {error}")
        connection.commit()
        _write_json_atomic(
            manifest_path,
            _progress_manifest(connection, plan, status="incomplete", last_error=str(error)),
        )
        raise
    finally:
        overall.close()
        connection.close()
    return SemanticAttentionIndexResult(index_path, manifest_path, manifest, False)


def _current_stage_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "semantic_labels": int(connection.execute("SELECT COUNT(*) FROM semantic_labels").fetchone()[0]),
        "no_rag": int(connection.execute("SELECT COUNT(*) FROM no_rag").fetchone()[0]),
        "questions": int(connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0]),
    }


def _progress_manifest(
    connection: sqlite3.Connection,
    plan: SemanticAttentionBuildPlan,
    *,
    status: str,
    summary: Mapping[str, Any] | None = None,
    last_error: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "source_fingerprint": plan.source_fingerprint,
        "plan": plan.to_dict(),
        "completed": _current_stage_counts(connection),
        "stage_complete": {
            "semantic_labels": bool(_get_metadata(connection, "labels_complete", False)),
            "no_rag": bool(_get_metadata(connection, "no_rag_complete", False)),
            "candidates": bool(_get_metadata(connection, "candidates_complete", False)),
        },
    }
    if summary is not None:
        value["summary"] = dict(summary)
    if last_error:
        value["last_error"] = last_error
    return value


def _checkpoint(
    connection: sqlite3.Connection,
    manifest_path: Path,
    plan: SemanticAttentionBuildPlan,
) -> None:
    connection.commit()
    _write_json_atomic(manifest_path, _progress_manifest(connection, plan, status="building"))


def _stage_progress(
    description: str,
    total: int | None,
    initial: int,
    enabled: bool,
) -> tqdm:
    return tqdm(
        total=total,
        initial=initial,
        desc=description,
        unit="item",
        leave=False,
        disable=not enabled,
    )


def _index_semantic_labels(
    connection: sqlite3.Connection,
    sources: SemanticAttentionDataSources,
    plan: SemanticAttentionBuildPlan,
    manifest_path: Path,
    overall: tqdm,
    checkpoint_every: int,
    show_progress: bool,
) -> None:
    completed = int(connection.execute("SELECT COUNT(*) FROM semantic_labels").fetchone()[0])
    stage = _stage_progress(
        "stage=1/3 index semantic labels",
        plan.expected_semantic_rows,
        completed,
        show_progress,
    )
    inserted = completed
    try:
        for logical_index, (line_number, row) in enumerate(_iter_jsonl(sources.semantic_labels_path)):
            if logical_index < completed:
                continue
            dataset = str(row.get("dataset") or "")
            sample_id = str(row.get("sample_id") or "")
            pair_id = str(row.get("pair_id") or row.get("id") or "")
            source = str(row.get("source") or "")
            stable_id = str(row.get("doc_stable_id") or "")
            label = str(row.get("semantic_label") or "")
            try:
                rank = int(row.get("doc_rank"))
                confidence = float(row.get("confidence"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid semantic rank/confidence at line {line_number}") from error
            if dataset != sources.dataset or not sample_id or not pair_id:
                raise ValueError(f"Invalid semantic identity at line {line_number}: {pair_id!r}")
            if label not in SEMANTIC_LABELS:
                raise ValueError(f"Unknown semantic label at line {line_number}: {label!r}")
            if not 1 <= rank <= sources.expected_documents:
                raise ValueError(f"Semantic doc rank outside Top-{sources.expected_documents}: {pair_id}")
            if not source or not stable_id or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Invalid semantic provenance/confidence: {pair_id}")
            connection.execute(
                """
                INSERT INTO semantic_labels(
                    pair_id,dataset,sample_id,doc_rank,source,stable_id,semantic_label,
                    confidence,topic_relation,evidence_indices_json,short_reason
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pair_id,
                    dataset,
                    sample_id,
                    rank,
                    source,
                    stable_id,
                    label,
                    confidence,
                    row.get("topic_relation"),
                    json.dumps(row.get("evidence_sentence_indices") or []),
                    str(row.get("short_reason") or ""),
                ),
            )
            inserted += 1
            stage.update(1)
            overall.update(1)
            if inserted % checkpoint_every == 0:
                _checkpoint(connection, manifest_path, plan)
        if plan.expected_semantic_rows is not None and inserted != plan.expected_semantic_rows:
            raise RuntimeError(
                f"Semantic row count mismatch: expected={plan.expected_semantic_rows}, actual={inserted}"
            )
        _set_metadata(connection, "labels_complete", True)
        _checkpoint(connection, manifest_path, plan)
    finally:
        stage.close()


def _index_no_rag(
    connection: sqlite3.Connection,
    sources: SemanticAttentionDataSources,
    plan: SemanticAttentionBuildPlan,
    manifest_path: Path,
    overall: tqdm,
    checkpoint_every: int,
    show_progress: bool,
) -> None:
    completed = int(connection.execute("SELECT COUNT(*) FROM no_rag").fetchone()[0])
    stage = _stage_progress(
        "stage=2/3 index no-RAG traces",
        plan.expected_no_rag_rows,
        completed,
        show_progress,
    )
    inserted = completed
    try:
        for logical_index, (line_number, row) in enumerate(_iter_jsonl(sources.no_rag_path)):
            if logical_index < completed:
                continue
            dataset = str(row.get("dataset") or "")
            sample_id = str(row.get("sample_id") or "")
            gold = str(row.get("gold_answer") or row.get("answer") or "").strip().upper()
            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            prediction = str(
                row.get("answer") or parsed.get("final_answer") or ""
            ).strip().upper() or None
            correct = row.get("answer_correct")
            if correct is None:
                correct = parsed.get("final_answer_correct")
            if dataset != sources.dataset or not sample_id or not gold:
                raise ValueError(f"Invalid no-RAG identity/gold answer at line {line_number}")
            if correct is not None and not isinstance(correct, bool):
                raise ValueError(f"Non-boolean no-RAG answer_correct for {sample_id}")
            choice_logprobs = row.get("choice_logprobs")
            if not isinstance(choice_logprobs, dict):
                choice_logprobs = {}
            connection.execute(
                """
                INSERT INTO no_rag(
                    sample_id,dataset,valid,gold_answer,predicted_answer,answer_correct,
                    canonical_generation,choice_logprobs_json
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    sample_id,
                    dataset,
                    int(bool(row.get("valid"))),
                    gold,
                    prediction,
                    None if correct is None else int(correct),
                    str(row.get("canonical_generation") or row.get("no_rag_generation") or ""),
                    json.dumps(choice_logprobs),
                ),
            )
            inserted += 1
            stage.update(1)
            overall.update(1)
            if inserted % checkpoint_every == 0:
                _checkpoint(connection, manifest_path, plan)
        if plan.expected_no_rag_rows is not None and inserted != plan.expected_no_rag_rows:
            raise RuntimeError(
                f"No-RAG row count mismatch: expected={plan.expected_no_rag_rows}, actual={inserted}"
            )
        _set_metadata(connection, "no_rag_complete", True)
        _checkpoint(connection, manifest_path, plan)
    finally:
        stage.close()


def _canonical_gold_answers(row: Mapping[str, Any], options: Mapping[str, str]) -> tuple[str, ...]:
    values = row.get("answers")
    if not isinstance(values, list) or not values:
        values = [row.get("answer")]
    answers = tuple(sorted({str(value or "").strip().upper() for value in values if str(value or "").strip()}))
    if not answers or any(answer not in options for answer in answers):
        raise ValueError(f"Invalid gold answers for {row.get('sample_id')}: {answers}")
    return answers


def _join_candidates(
    connection: sqlite3.Connection,
    sources: SemanticAttentionDataSources,
    plan: SemanticAttentionBuildPlan,
    assignments: Mapping[str, str],
    manifest_path: Path,
    overall: tqdm,
    checkpoint_every: int,
    show_progress: bool,
) -> None:
    completed = int(connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0])
    ordinals = {
        split: int(
            connection.execute("SELECT COUNT(*) FROM questions WHERE split=?", (split,)).fetchone()[0]
        )
        for split in SPLITS
    }
    stage = _stage_progress(
        "stage=3/3 join full Top-8 question groups",
        plan.expected_questions,
        completed,
        show_progress,
    )
    inserted = completed
    try:
        for logical_index, (line_number, row) in enumerate(_iter_jsonl(sources.candidates_path)):
            if logical_index < completed:
                continue
            dataset = str(row.get("dataset") or "")
            sample_id = str(row.get("sample_id") or "")
            if dataset != sources.dataset or not sample_id:
                raise ValueError(f"Candidate dataset/sample mismatch at line {line_number}")
            split = assignments.get(sample_id)
            if split is None:
                raise ValueError(f"Candidate question has no split assignment: {sample_id}")
            question = str(row.get("question") or "").strip()
            options_value = row.get("options")
            if not question or not isinstance(options_value, dict) or len(options_value) < 2:
                raise ValueError(f"Invalid candidate question/options: {sample_id}")
            options = {str(key).strip().upper(): str(value) for key, value in options_value.items()}
            gold_answers = _canonical_gold_answers(row, options)
            documents = row.get("candidate_documents")
            if not isinstance(documents, list) or len(documents) != sources.expected_documents:
                raise ValueError(
                    f"Expected exactly {sources.expected_documents} documents for {sample_id}, "
                    f"found {len(documents) if isinstance(documents, list) else 'non-list'}"
                )
            ranks = [document.get("rerank_rank") for document in documents if isinstance(document, dict)]
            if ranks != list(range(1, sources.expected_documents + 1)):
                raise ValueError(f"Non-contiguous rerank ranks for {sample_id}: {ranks}")
            stable_ids = [str(document.get("stable_id") or "") for document in documents]
            if not all(stable_ids) or len(set(stable_ids)) != sources.expected_documents:
                raise ValueError(f"Missing or duplicate document stable IDs for {sample_id}")
            no_rag = connection.execute("SELECT * FROM no_rag WHERE sample_id=?", (sample_id,)).fetchone()
            if no_rag is None:
                raise ValueError(f"Missing no-RAG trace for candidate question: {sample_id}")
            if not bool(no_rag["valid"]):
                raise ValueError(f"Candidate question has an invalid no-RAG trace: {sample_id}")
            if str(no_rag["gold_answer"]) not in gold_answers:
                raise ValueError(
                    f"Candidate/no-RAG gold mismatch for {sample_id}: "
                    f"{gold_answers} != {no_rag['gold_answer']}"
                )
            connection.execute(
                """
                INSERT INTO questions(
                    sample_id,dataset,split,split_ordinal,row_idx,question,options_json,
                    gold_answers_json,no_rag_valid,no_rag_gold_answer,no_rag_prediction,
                    no_rag_answer_correct,no_rag_generation,no_rag_choice_logprobs_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sample_id,
                    dataset,
                    split,
                    ordinals[split],
                    row.get("row_idx"),
                    question,
                    json.dumps(options, ensure_ascii=False),
                    json.dumps(gold_answers),
                    int(no_rag["valid"]),
                    no_rag["gold_answer"],
                    no_rag["predicted_answer"],
                    no_rag["answer_correct"],
                    no_rag["canonical_generation"],
                    no_rag["choice_logprobs_json"],
                ),
            )
            for rank, document in enumerate(documents, start=1):
                stable_id = stable_ids[rank - 1]
                source = str(document.get("source") or "")
                pair_id = f"{sample_id}::{rank}::{stable_id}"
                label = connection.execute(
                    "SELECT * FROM semantic_labels WHERE pair_id=?", (pair_id,)
                ).fetchone()
                if label is None:
                    raise ValueError(f"Missing raw semantic label for candidate pair: {pair_id}")
                if bool(label["used"]):
                    raise ValueError(f"Candidate pair is duplicated: {pair_id}")
                if (
                    label["sample_id"] != sample_id
                    or int(label["doc_rank"]) != rank
                    or label["stable_id"] != stable_id
                    or label["source"] != source
                ):
                    raise ValueError(f"Candidate/semantic provenance mismatch: {pair_id}")
                text = str(document.get("text") or document.get("title") or "").strip()
                if not source or not text:
                    raise ValueError(f"Candidate document lacks source/text: {pair_id}")
                semantic_label = str(label["semantic_label"])
                is_mixed = semantic_label == MIXED_LABEL
                connection.execute(
                    """
                    INSERT INTO documents(
                        sample_id,doc_rank,pair_id,stable_id,source,title,text,semantic_label,
                        semantic_class_id,semantic_support_target,semantic_loss_mask,
                        semantic_confidence,topic_relation,evidence_indices_json,short_reason
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        sample_id,
                        rank,
                        pair_id,
                        stable_id,
                        source,
                        str(document.get("title") or ""),
                        text,
                        semantic_label,
                        SEMANTIC_CLASS_IDS.get(semantic_label),
                        SEMANTIC_SUPPORT_TARGETS.get(semantic_label),
                        int(not is_mixed),
                        float(label["confidence"]),
                        label["topic_relation"],
                        label["evidence_indices_json"],
                        label["short_reason"],
                    ),
                )
                connection.execute("UPDATE semantic_labels SET used=1 WHERE pair_id=?", (pair_id,))
            connection.execute("UPDATE no_rag SET used=1 WHERE sample_id=?", (sample_id,))
            ordinals[split] += 1
            inserted += 1
            stage.update(1)
            overall.update(1)
            if inserted % checkpoint_every == 0:
                _checkpoint(connection, manifest_path, plan)
        if plan.expected_questions is not None and inserted != plan.expected_questions:
            raise RuntimeError(
                f"Candidate question count mismatch: expected={plan.expected_questions}, actual={inserted}"
            )
        _set_metadata(connection, "candidates_complete", True)
        _checkpoint(connection, manifest_path, plan)
    finally:
        stage.close()


def _validate_completed_index(
    connection: sqlite3.Connection,
    sources: SemanticAttentionDataSources,
    plan: SemanticAttentionBuildPlan,
) -> dict[str, Any]:
    question_count = int(connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0])
    document_count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    label_count = int(connection.execute("SELECT COUNT(*) FROM semantic_labels").fetchone()[0])
    unused_labels = int(connection.execute("SELECT COUNT(*) FROM semantic_labels WHERE used=0").fetchone()[0])
    if document_count != question_count * sources.expected_documents:
        raise RuntimeError(
            f"Grouped document count mismatch: {document_count} != "
            f"{question_count}*{sources.expected_documents}"
        )
    if label_count != document_count or unused_labels:
        first = connection.execute(
            "SELECT pair_id FROM semantic_labels WHERE used=0 ORDER BY pair_id LIMIT 1"
        ).fetchone()
        raise RuntimeError(
            f"Semantic coverage mismatch: labels={label_count}, documents={document_count}, "
            f"unused={unused_labels}, first_unused={first[0] if first else None}"
        )
    bad_group = connection.execute(
        """
        SELECT q.sample_id, COUNT(d.doc_rank) AS n
        FROM questions q LEFT JOIN documents d ON d.sample_id=q.sample_id
        GROUP BY q.sample_id HAVING n != ? LIMIT 1
        """,
        (sources.expected_documents,),
    ).fetchone()
    if bad_group is not None:
        raise RuntimeError(f"Question does not retain full Top-k: {bad_group['sample_id']} n={bad_group['n']}")
    split_counts = {
        row["split"]: int(row["n"])
        for row in connection.execute("SELECT split,COUNT(*) AS n FROM questions GROUP BY split")
    }
    semantic_counts = {
        row["semantic_label"]: int(row["n"])
        for row in connection.execute(
            "SELECT semantic_label,COUNT(*) AS n FROM documents GROUP BY semantic_label"
        )
    }
    semantic_masked = int(
        connection.execute("SELECT COUNT(*) FROM documents WHERE semantic_loss_mask=0").fetchone()[0]
    )
    no_rag_count = int(connection.execute("SELECT COUNT(*) FROM no_rag").fetchone()[0])
    unused_no_rag = int(connection.execute("SELECT COUNT(*) FROM no_rag WHERE used=0").fetchone()[0])
    no_rag_counts = {
        f"{row['split']}:{'correct' if row['no_rag_answer_correct'] else 'wrong'}": int(row["n"])
        for row in connection.execute(
            """
            SELECT split,no_rag_answer_correct,COUNT(*) AS n
            FROM questions GROUP BY split,no_rag_answer_correct
            """
        )
    }
    return {
        "dataset": sources.dataset,
        "questions": question_count,
        "documents": document_count,
        "documents_per_question": sources.expected_documents,
        "split_questions": split_counts,
        "semantic_labels": semantic_counts,
        "semantic_loss_masked_documents": semantic_masked,
        "semantic_loss_active_documents": document_count - semantic_masked,
        "no_rag_rows": no_rag_count,
        "unused_no_rag_rows": unused_no_rag,
        "no_rag_question_groups": no_rag_counts,
        "declared_split_ids": dict(plan.split_id_counts),
        "unused_split_ids": {
            split: int(plan.split_id_counts.get(split, 0)) - int(split_counts.get(split, 0))
            for split in SPLITS
        },
    }


class RAG2SemanticAttentionDataset(Sequence[SemanticAttentionQuestion]):
    """Lazy, process-safe random access to one split of a completed index."""

    def __init__(self, index_path: Path, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.index_path = index_path.expanduser().resolve()
        self.manifest_path = _manifest_path(self.index_path)
        manifest = _read_json(self.manifest_path)
        if manifest is None or manifest.get("status") != "complete":
            raise ValueError(f"Semantic-attention index is not complete: {self.index_path}")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported semantic-attention index schema: {manifest.get('schema_version')}")
        self.split = split
        summary = manifest.get("summary", {})
        self._length = int(summary.get("split_questions", {}).get(split, 0))
        self._expected_documents = int(summary.get("documents_per_question", 0))
        if self._expected_documents <= 0:
            raise ValueError(f"Completed index manifest lacks documents_per_question: {self.manifest_path}")
        self._connection: sqlite3.Connection | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return self._length

    def _db(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._connection is None or self._pid != pid:
            if self._connection is not None:
                self._connection.close()
            uri = f"file:{self.index_path}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True, timeout=120.0)
            self._connection.row_factory = sqlite3.Row
            self._pid = pid
        return self._connection

    def __getitem__(self, index: int | slice) -> SemanticAttentionQuestion | list[SemanticAttentionQuestion]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        connection = self._db()
        question = connection.execute(
            "SELECT * FROM questions WHERE split=? AND split_ordinal=?",
            (self.split, index),
        ).fetchone()
        if question is None:
            raise RuntimeError(f"Missing indexed question: split={self.split} ordinal={index}")
        documents = tuple(
            _document_from_row(row)
            for row in connection.execute(
                "SELECT * FROM documents WHERE sample_id=? ORDER BY doc_rank",
                (question["sample_id"],),
            )
        )
        if len(documents) != self._expected_documents:
            raise RuntimeError(
                f"Corrupt document group for {question['sample_id']}: "
                f"{len(documents)} != {self._expected_documents}"
            )
        correct_value = question["no_rag_answer_correct"]
        no_rag = NoRAGTrace(
            sample_id=question["sample_id"],
            valid=bool(question["no_rag_valid"]),
            gold_answer=question["no_rag_gold_answer"],
            predicted_answer=question["no_rag_prediction"],
            answer_correct=None if correct_value is None else bool(correct_value),
            canonical_generation=question["no_rag_generation"],
            choice_logprobs=json.loads(question["no_rag_choice_logprobs_json"]),
        )
        return SemanticAttentionQuestion(
            dataset=question["dataset"],
            split=question["split"],
            sample_id=question["sample_id"],
            row_idx=question["row_idx"],
            question=question["question"],
            options=json.loads(question["options_json"]),
            gold_answers=tuple(json.loads(question["gold_answers_json"])),
            no_rag=no_rag,
            documents=documents,
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _document_from_row(row: sqlite3.Row) -> SemanticAttentionDocument:
    return SemanticAttentionDocument(
        pair_id=row["pair_id"],
        rank=int(row["doc_rank"]),
        stable_id=row["stable_id"],
        source=row["source"],
        title=row["title"],
        text=row["text"],
        semantic_label=row["semantic_label"],
        semantic_class_id=row["semantic_class_id"],
        semantic_support_target=row["semantic_support_target"],
        semantic_loss_mask=bool(row["semantic_loss_mask"]),
        semantic_confidence=float(row["semantic_confidence"]),
        topic_relation=row["topic_relation"],
        evidence_sentence_indices=tuple(json.loads(row["evidence_indices_json"])),
        short_reason=row["short_reason"],
    )


class DeterministicQuestionBatchSampler(Sequence[list[int]]):
    """Epoch-deterministic question batches with an explicit resume cursor."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        epoch: int = 0,
        start_batch: int = 0,
    ) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epoch < 0 or start_batch < 0:
            raise ValueError("epoch and start_batch must be non-negative")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = epoch
        self.start_batch = start_batch
        if start_batch > self.total_batches:
            raise ValueError(f"start_batch {start_batch} exceeds total batches {self.total_batches}")

    @property
    def total_batches(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size) if self.dataset_size else 0

    @property
    def remaining_batches(self) -> int:
        return self.total_batches - self.start_batch

    def _all_batches(self) -> list[list[int]]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches = [indices[offset : offset + self.batch_size] for offset in range(0, len(indices), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._all_batches()[self.start_batch :]

    def __len__(self) -> int:
        return self.remaining_batches

    def __getitem__(self, index: int | slice) -> list[int] | list[list[int]]:
        batches = self._all_batches()[self.start_batch :]
        return batches[index]

    def set_epoch(self, epoch: int, *, start_batch: int = 0) -> None:
        if epoch < 0 or start_batch < 0:
            raise ValueError("epoch and start_batch must be non-negative")
        self.epoch = epoch
        self.start_batch = start_batch
        if start_batch > self.total_batches:
            raise ValueError(f"start_batch {start_batch} exceeds total batches {self.total_batches}")

    def state_dict(self, *, next_batch: int | None = None) -> dict[str, Any]:
        cursor = self.start_batch if next_batch is None else next_batch
        if not 0 <= cursor <= self.total_batches:
            raise ValueError(f"Invalid next_batch cursor: {cursor}")
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "next_batch": cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Sampler resume state is incompatible: expected={expected}, actual={actual}")
        self.set_epoch(int(state["epoch"]), start_batch=int(state["next_batch"]))

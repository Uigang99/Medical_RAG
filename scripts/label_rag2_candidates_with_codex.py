from __future__ import annotations

"""Label RAG² question--document pairs with Codex in resumable batches.

This is intentionally separate from the existing PPL-based RAG² labelling
pipeline.  It never reads a PPL value, a generated rationale, or a prior
pseudo-label.  Codex receives only the MCQ, its options, the gold option, and
one retrieved corpus chunk, then returns a semantic utility judgement.

The job is designed for a long-running 1.87M-pair annotation pass:

* deterministic Top-k document selection from the reranked candidate JSONL;
* question-grouped batches so a question is not repeated ten times in context;
* one independently validated JSON result per batch;
* SQLite progress metadata and atomic batch writes;
* safe --resume behaviour after an interruption; and
* one consolidated JSONL per dataset when every batch has finished.

Codex itself is invoked through ``codex exec``.  No GPU is used by this
script.  The optional --enable-web-search flag is off by default because a
web-grounded decision for millions of pairs is slow and non-reproducible.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm


ANNOTATION_VERSION = "rag2_codex_evidence_utility_label_v2"
PROMPT_VERSION = "rag2_codex_evidence_utility_prompt_v3_compact_item_index"
VALID_UTILITY_LABELS = (
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
)
VALID_TOPIC_RELATIONS = ("related", "unrelated", "unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantically label reranked RAG² question-document pairs with Codex batches."
    )
    parser.add_argument(
        "--candidates-paths",
        nargs="+",
        type=Path,
        required=True,
        help="One candidates_top32.jsonl file per dataset.  Candidate rows must contain rerank_rank and text.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--completed-batches-root",
        type=Path,
        default=None,
        help=(
            "Optional read-only prior-run root containing completed batch JSONs. "
            "Validated artifacts are reused; only newly generated batches are written under --output-root."
        ),
    )
    parser.add_argument(
        "--docs-per-question",
        type=int,
        default=10,
        help="Take this many highest ranked reranked chunks per question (default: 10).",
    )
    parser.add_argument(
        "--questions-per-batch",
        type=int,
        default=10,
        help="Number of questions submitted to each Codex call (default: 10 = 100 pairs at Top-10).",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=0,
        help="Optional character cap for each document.  Zero preserves the complete corpus chunk.",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--model",
        default="",
        help="Optional Codex model name.  Empty uses the account's configured Codex default.",
    )
    parser.add_argument(
        "--model-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="",
        help=(
            "Override Codex reasoning effort for this labelling job.  Empty uses the configured default. "
            "For this short, repeated classification task, medium is the recommended starting point."
        ),
    )
    parser.add_argument(
        "--enable-web-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expose Codex web search. Off by default for reproducible, scalable semantic labelling.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=0,
        help="Per-Codex-call timeout.  Zero means no local timeout.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=30.0)
    parser.add_argument(
        "--retry-jitter-fraction",
        type=float,
        default=0.25,
        help=(
            "Deterministic per-batch jitter added to retry waits so concurrent workers do not "
            "resubmit capacity-limited requests at the same instant (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip fully validated batch result files and retry incomplete/failed batches.",
    )
    parser.add_argument(
        "--reuse-validated-completed-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accept completed batch artifacts whose input hash predates a transport-only prompt/schema upgrade, "
            "after revalidating every stored label against the current candidate pairs."
        ),
    )
    parser.add_argument(
        "--progress-db-path",
        type=Path,
        default=None,
        help=(
            "Optional writable SQLite status database.  Batch JSON artifacts remain in --output-root and "
            "are still the authoritative resumable results, so this permits safe continuation from another "
            "Linux account when the original progress.sqlite is read-only."
        ),
    )
    parser.add_argument(
        "--pending-plan-path",
        type=Path,
        default=None,
        help="Optional frozen pending-batch plan used to rebalance unfinished work across workers.",
    )
    parser.add_argument(
        "--write-pending-plan-only",
        action="store_true",
        help="Validate existing artifacts, write --pending-plan-path, and exit without invoking Codex.",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="For a bounded pilot only.  Zero processes every question in each candidates file.",
    )
    parser.add_argument(
        "--stop-after-batches",
        type=int,
        default=0,
        help="Stop cleanly after this many newly completed batches.  Zero has no limit.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Number of deterministic Codex workers sharing this output root (default: 1).",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Zero-based worker index.  Worker i processes batches where batch_index %% worker_count == i.",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Do not call Codex; verify completed batch files and write final JSONLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input selection and report the exact number of planned batches without invoking Codex.",
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
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object: {path}:{line_number}")
            yield value


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def canonical_options(row: dict[str, Any]) -> dict[str, str]:
    options = row.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(f"Missing options for {row.get('sample_id')}")
    normalized = {str(key).strip().upper(): clean_text(value) for key, value in options.items()}
    if not all(normalized) or not all(normalized.values()):
        raise ValueError(f"Invalid options for {row.get('sample_id')}")
    return dict(sorted(normalized.items()))


def canonical_answers(row: dict[str, Any], options: dict[str, str]) -> list[str]:
    values = row.get("answers")
    if not isinstance(values, list):
        values = [row.get("answer")]
    answers = sorted({str(value or "").strip().upper() for value in values if str(value or "").strip()})
    if not answers or any(answer not in options for answer in answers):
        raise ValueError(f"Invalid gold answer for {row.get('sample_id')}: {answers}")
    return answers


def document_pair_id(sample_id: str, document: dict[str, Any], rank: int) -> str:
    stable_id = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    if not stable_id:
        stable_id = f"{document.get('source')}:{document.get('local_id')}"
    return f"{sample_id}::{rank}::{stable_id}"


def selected_documents(row: dict[str, Any], docs_per_question: int, max_doc_chars: int) -> list[dict[str, Any]]:
    raw_documents = row.get("candidate_documents")
    if not isinstance(raw_documents, list):
        raise ValueError(f"Missing candidate_documents for {row.get('sample_id')}")

    def rank_key(document: dict[str, Any]) -> tuple[int, str]:
        try:
            rank = int(document.get("rerank_rank"))
        except (TypeError, ValueError):
            rank = sys.maxsize
        stable = str(document.get("stable_id") or document.get("corpus_id") or document.get("db_id") or "")
        return rank, stable

    ordered = sorted((item for item in raw_documents if isinstance(item, dict)), key=rank_key)
    if len(ordered) < docs_per_question:
        raise ValueError(
            f"Expected at least {docs_per_question} reranked documents for {row.get('sample_id')}, found {len(ordered)}"
        )
    result: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for fallback_rank, document in enumerate(ordered[:docs_per_question], start=1):
        try:
            rank = int(document.get("rerank_rank") or fallback_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rerank rank for {row.get('sample_id')}") from exc
        if rank in seen_ranks:
            raise ValueError(f"Duplicate rerank rank {rank} for {row.get('sample_id')}")
        seen_ranks.add(rank)
        text = clean_text(document.get("text"))
        if not text:
            raise ValueError(f"Missing document text for {row.get('sample_id')} rerank_rank={rank}")
        if max_doc_chars and len(text) > max_doc_chars:
            text = text[: max_doc_chars - 3].rstrip() + "..."
        stable_id = str(document.get("stable_id") or document.get("corpus_id") or document.get("db_id") or "")
        if not stable_id:
            raise ValueError(f"Missing stable document ID for {row.get('sample_id')} rerank_rank={rank}")
        result.append(
            {
                "pair_id": document_pair_id(str(row.get("sample_id") or ""), document, rank),
                "doc_rank": rank,
                "source": clean_text(document.get("source")) or "unknown",
                "doc_stable_id": stable_id,
                "title": clean_text(document.get("title")),
                "text": text,
            }
        )
    return result


def make_question_item(row: dict[str, Any], docs_per_question: int, max_doc_chars: int) -> dict[str, Any]:
    sample_id = clean_text(row.get("sample_id"))
    dataset = clean_text(row.get("dataset")).lower()
    question = clean_text(row.get("question"))
    if not sample_id or not dataset or not question:
        raise ValueError(f"Candidate row missing sample_id/dataset/question: {row.get('sample_id')}")
    options = canonical_options(row)
    answers = canonical_answers(row, options)
    documents = selected_documents(row, docs_per_question, max_doc_chars)
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "question": question,
        "options": options,
        "gold_answers": answers,
        "documents": documents,
    }


def iter_question_items(path: Path, args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    expected_dataset: str | None = None
    for index, row in enumerate(iter_jsonl(path)):
        if args.limit_questions and index >= args.limit_questions:
            break
        item = make_question_item(row, args.docs_per_question, args.max_doc_chars)
        if expected_dataset is None:
            expected_dataset = item["dataset"]
        elif item["dataset"] != expected_dataset:
            raise ValueError(f"A candidates file must contain one dataset: {path}")
        yield item


def count_question_items(path: Path, args: argparse.Namespace) -> tuple[str, int, int]:
    dataset = ""
    questions = 0
    pairs = 0
    for item in iter_question_items(path, args):
        dataset = item["dataset"]
        questions += 1
        pairs += len(item["documents"])
    if not dataset or not questions:
        raise RuntimeError(f"No selected question rows: {path}")
    return dataset, questions, pairs


def batched(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def sentence_indexed_text(text: str) -> str:
    """Provide stable, human-readable sentence references without changing the evidence text."""
    import re

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    pieces = [piece.strip() for piece in pieces if piece.strip()]
    if not pieces:
        return text
    return "\n".join(f"[S{index}] {piece}" for index, piece in enumerate(pieces, start=1))


def prompt_payload(question_batch: list[dict[str, Any]]) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    item_index = 1
    for question_item in question_batch:
        # Keep the semantic decision generalisable beyond MCQ.  The reference
        # answer is supplied as text, while the non-gold answer choices are
        # intentionally withheld: a chunk is judged by its evidence for the
        # answer, not by how well it separates a fixed set of distractors.
        reference_answers = [question_item["options"][option] for option in question_item["gold_answers"]]
        documents = []
        for document in question_item["documents"]:
            documents.append(
                {
                    # Long corpus IDs are vulnerable to transcription errors in
                    # a large structured response.  The model returns this
                    # compact ordinal; we restore the immutable pair ID locally.
                    "item_index": item_index,
                    "document_sentences": sentence_indexed_text(document["text"]),
                }
            )
            item_index += 1
        questions.append(
            {
                "dataset": question_item["dataset"],
                "sample_id": question_item["sample_id"],
                "question": question_item["question"],
                "reference_answer": reference_answers,
                "candidate_documents": documents,
            }
        )
    return {"annotation_version": ANNOTATION_VERSION, "questions": questions}


def annotation_instruction() -> str:
    return """You are a careful medical evidence adjudicator. Label every candidate question-document pair in the supplied JSON batch independently.

The task is general answer-evidence assessment, not multiple-choice comparison.  Each item gives a question, a REFERENCE ANSWER, and one retrieved document chunk.  Decide only what information is actually present in that chunk and whether it helps a medically competent solver produce or justify the reference answer.  Do not use answer choices not shown, PPL, a prior RAG trace, retrieved ranking score, a model prediction, or medical knowledge to fill in missing document evidence.

Choose exactly one utility label:
- direct_support: the chunk contains a direct, answer-justifying medical fact, criterion, relationship, or recommendation.  Given the question and this chunk, the reference answer is substantially justified.
- supporting_evidence: the chunk contains a medically valid and case-relevant premise that can be used in a correct reasoning chain toward the reference answer, but cannot by itself justify the answer.
- no_evidence: the chunk contains no usable proposition that materially supports the reference answer or pushes toward an incompatible answer.  It may be topically related or entirely unrelated.
- misleading_evidence: an explicit chunk claim, when reasonably applied to this question, contradicts the reference answer or plausibly leads toward an incompatible answer.  Mere omission of useful information, a different disease topic, or generic background is not misleading.
- indeterminate_or_mixed: use only when the displayed text itself is uninterpretable, or contains both answer-supporting and answer-opposing claims of comparable material strength.  A normal 120-word chunk boundary, missing surrounding context, or an incomplete answer chain is NOT indeterminate; judge the evidence actually displayed and otherwise use supporting_evidence or no_evidence.

Set topic_relation only for no_evidence:
- related: the chunk shares a clinically meaningful topic with the question but does not provide answer-useful evidence.
- unrelated: the chunk is not clinically relevant to solving the question.
- unclear: use only when related versus unrelated cannot be established from the displayed text.
For every other utility label, topic_relation must be null.

Important rules:
1. Shared words, diseases, entities, or broad topics alone are no_evidence, not support.
2. Support must come from a concrete proposition in the supplied chunk, not from your own knowledge of the reference answer.
3. Treat each chunk as its own evidence unit.  Do not penalize it merely because the parent document may have had additional context outside this 120-word chunk.
4. The reference answer is provided to evaluate evidence utility.  Do not solve the MCQ or compare alternative answer choices in the output.
5. For direct_support, supporting_evidence, and misleading_evidence, cite the [S#] sentence(s) containing the material claim.  For no_evidence, use an empty list.  For indeterminate_or_mixed, cite only sentences that create the conflict; otherwise use an empty list.
6. Return one record for every input item_index, no duplicates and no omissions.  Copy each small integer item_index exactly; never output a document ID.  Keep short_reason factual and at most 30 words.  Do not add prose outside the schema-conforming JSON response.
"""


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "item_index": {"type": "integer", "minimum": 1},
                        "label": {"type": "string", "enum": list(VALID_UTILITY_LABELS)},
                        "topic_relation": {"type": ["string", "null"], "enum": [*VALID_TOPIC_RELATIONS, None]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "evidence_sentence_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                        "short_reason": {"type": "string"},
                    },
                    "required": [
                        "item_index",
                        "label",
                        "topic_relation",
                        "confidence",
                        "evidence_sentence_indices",
                        "short_reason",
                    ],
                },
            }
        },
        "required": ["labels"],
    }


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_database(path: Path) -> sqlite3.Connection:
    # Multiple Codex workers share one small progress database.  WAL allows
    # concurrent readers, while this explicit timeout prevents brief status
    # updates from failing merely because another worker is committing.
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS batches (
            dataset TEXT NOT NULL,
            batch_index INTEGER NOT NULL,
            input_sha256 TEXT NOT NULL,
            pair_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            last_error TEXT,
            PRIMARY KEY (dataset, batch_index)
        )
        """
    )
    connection.commit()
    return connection


def write_progress_database_pointer(output_root: Path, progress_db_path: Path) -> None:
    """Let the monitor find a per-user status DB without moving batch artifacts."""
    write_json_atomic(
        output_root / "active_progress_database.json",
        {
            "path": str(progress_db_path.resolve()),
            "updated_at": utc_now(),
        },
    )


def batch_path(output_root: Path, dataset: str, batch_index: int) -> Path:
    return output_root / "batches" / dataset / f"batch_{batch_index:06d}.json"


def raw_attempt_path(output_root: Path, dataset: str, batch_index: int, attempt: int) -> Path:
    return output_root / "failed_attempts" / dataset / f"batch_{batch_index:06d}.attempt_{attempt}.txt"


def flatten_batch_metadata(question_batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for question_item in question_batch:
        for document in question_item["documents"]:
            pair_id = document["pair_id"]
            if pair_id in metadata:
                raise ValueError(f"Duplicate candidate pair ID: {pair_id}")
            metadata[pair_id] = {
                "id": pair_id,
                "pair_id": pair_id,
                "dataset": question_item["dataset"],
                "sample_id": question_item["sample_id"],
                "doc_rank": document["doc_rank"],
                "source": document["source"],
                "doc_stable_id": document["doc_stable_id"],
                "title": document["title"],
            }
    return metadata


def validate_response(value: Any, expected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("labels"), list):
        raise ValueError("Codex final response is not a JSON object with a labels list")
    labels = value["labels"]
    expected_pair_ids = list(expected)
    received: dict[str, dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("Codex label list contains a non-object")
        # New compact transport uses item_index.  The pair_id branch keeps old
        # completed artifacts readable during a safe resume after this upgrade.
        if "item_index" in item:
            item_index = item.get("item_index")
            if not isinstance(item_index, int) or isinstance(item_index, bool) or not 1 <= item_index <= len(expected_pair_ids):
                raise ValueError(f"Invalid item_index in Codex response: {item_index!r}")
            pair_id = expected_pair_ids[item_index - 1]
        else:
            pair_id = str(item.get("pair_id") or "")
        if not pair_id or pair_id in received:
            raise ValueError(f"Missing or duplicate pair_id in Codex response: {pair_id!r}")
        label = str(item.get("label") or "").strip().lower()
        if label not in VALID_UTILITY_LABELS:
            raise ValueError(f"Invalid Codex label for {pair_id}: {label!r}")
        topic_relation = item.get("topic_relation")
        if topic_relation is not None:
            topic_relation = str(topic_relation).strip().lower()
        if label == "no_evidence":
            if topic_relation not in VALID_TOPIC_RELATIONS:
                raise ValueError(f"no_evidence requires a valid topic_relation for {pair_id}: {topic_relation!r}")
        elif topic_relation is not None:
            raise ValueError(f"topic_relation must be null for {label} at {pair_id}: {topic_relation!r}")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(float(confidence)):
            raise ValueError(f"Invalid confidence for {pair_id}: {confidence!r}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"Out-of-range confidence for {pair_id}: {confidence!r}")
        sentence_indices = item.get("evidence_sentence_indices")
        if not isinstance(sentence_indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 1 for index in sentence_indices
        ):
            raise ValueError(f"Invalid evidence_sentence_indices for {pair_id}")
        reason = clean_text(item.get("short_reason"))
        if not reason:
            raise ValueError(f"Missing short_reason for {pair_id}")
        if len(reason) > 600:
            raise ValueError(f"short_reason too long for {pair_id}")
        received[pair_id] = {
            "semantic_label": label,
            "topic_relation": topic_relation,
            "confidence": float(confidence),
            "evidence_sentence_indices": sentence_indices,
            "short_reason": reason,
        }
    expected_ids = set(expected)
    received_ids = set(received)
    if received_ids != expected_ids:
        missing = sorted(expected_ids - received_ids)
        extra = sorted(received_ids - expected_ids)
        raise ValueError(
            f"Codex response pair IDs mismatch: missing={len(missing)} {missing[:3]} extra={len(extra)} {extra[:3]}"
        )
    return [{**expected[pair_id], **received[pair_id]} for pair_id in sorted(expected)]


def load_completed_batch(
    path: Path,
    input_hash: str,
    expected: dict[str, dict[str, Any]],
    allow_hash_mismatch: bool = False,
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("input_sha256") != input_hash and not allow_hash_mismatch:
            return None
        stored_labels = value.get("labels")
        if not isinstance(stored_labels, list):
            return None
        # Completed artifacts retain the explicit ``semantic_label`` field for
        # downstream training, whereas Codex's schema names the transient field
        # ``label``.  Convert only for the common validator.
        validator_labels = []
        for label_row in stored_labels:
            if not isinstance(label_row, dict):
                return None
            converted = dict(label_row)
            converted["label"] = converted.get("label", converted.get("semantic_label"))
            validator_labels.append(converted)
        return validate_response({"labels": validator_labels}, expected)
    except PermissionError as exc:
        # A resumable run must never silently regenerate an already-completed
        # batch merely because another Linux account owns its JSON artifact.
        # Propagate a clear failure so the operator can grant read access (or
        # copy the artifacts) before spending any additional Codex usage.
        raise PermissionError(
            f"Existing completed batch is not readable: {path}. "
            "Grant the current Linux account read access to the batch artifact, "
            "then resume with the same output root."
        ) from exc
    except Exception as exc:
        logging.warning("Ignoring invalid completed batch %s: %s", path, exc)
        return None


def load_reusable_batch(
    args: argparse.Namespace,
    output_root: Path,
    dataset: str,
    batch_index: int,
    input_hash: str,
    expected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Path] | None:
    """Resolve a batch from the writable run first, then an immutable prior cache."""
    roots = [output_root]
    if args.completed_batches_root is not None and args.completed_batches_root.resolve() != output_root.resolve():
        roots.append(args.completed_batches_root)
    for root in roots:
        candidate_path = batch_path(root, dataset, batch_index)
        labels = load_completed_batch(
            candidate_path,
            input_hash,
            expected,
            allow_hash_mismatch=args.reuse_validated_completed_batches,
        )
        if labels is not None:
            return labels, candidate_path
    return None


def codex_command(args: argparse.Namespace, schema_path: Path, response_path: Path) -> list[str]:
    command = [args.codex_bin]
    if args.enable_web_search:
        command.append("--search")
    command.extend(["--sandbox", "read-only", "--ask-for-approval", "never"])
    if args.model:
        command.extend(["--model", args.model])
    if args.model_reasoning_effort:
        command.extend(["--config", f"model_reasoning_effort={json.dumps(args.model_reasoning_effort)}"])
    command.extend(
        [
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--json",
            "--output-last-message",
            str(response_path),
            annotation_instruction(),
        ]
    )
    return command


def extract_codex_usage(jsonl_stdout: str) -> dict[str, int] | None:
    """Extract the final token accounting emitted by ``codex exec --json``."""
    usage: dict[str, int] | None = None
    for line in jsonl_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn.completed":
            continue
        value = event.get("usage")
        if not isinstance(value, dict):
            continue
        parsed: dict[str, int] = {}
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
            raw = value.get(key, 0)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                parsed[key] = raw
            elif isinstance(raw, float) and math.isfinite(raw) and raw >= 0 and raw.is_integer():
                parsed[key] = int(raw)
            else:
                parsed[key] = 0
        usage = parsed
    return usage


def final_message_from_codex_events(jsonl_stdout: str) -> str:
    """Fallback for an unexpected missing --output-last-message artifact."""
    message = ""
    for line in jsonl_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            message = text
    return message


def codex_event_errors(jsonl_stdout: str) -> list[str]:
    """Return the useful failure messages hidden in ``codex exec --json`` stdout.

    The CLI emits transport/model failures as JSONL events on stdout while its
    stderr may contain only the harmless "Reading additional input" notice.
    Keeping these messages in the batch status is essential for distinguishing
    account exhaustion from a transient model-capacity event.
    """
    messages: list[str] = []
    for line in jsonl_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        candidates: list[object] = []
        if event.get("type") == "error":
            candidates.append(event.get("message"))
        error = event.get("error")
        if isinstance(error, dict):
            candidates.append(error.get("message"))
        elif isinstance(error, str):
            candidates.append(error)
        for candidate in candidates:
            message = clean_text(candidate)
            if message and message not in messages:
                messages.append(message)
    return messages


def codex_failure_summary(returncode: int, stdout: str, stderr: str) -> str:
    parts = [f"codex_exit={returncode}"]
    event_errors = codex_event_errors(stdout)
    if event_errors:
        parts.append("errors=" + " | ".join(event_errors))
    stderr_lines = [
        line.strip()
        for line in stderr.splitlines()
        if line.strip() and line.strip() != "Reading additional input from stdin..."
    ]
    if stderr_lines:
        parts.append("stderr=" + " | ".join(stderr_lines[-8:]))
    return "; ".join(parts)


def retry_delay_seconds(args: argparse.Namespace, dataset: str, batch_index: int, attempt: int) -> float:
    """Linear backoff plus stable jitter, reproducible across resume runs."""
    base = args.retry_backoff_seconds * attempt
    if base <= 0 or args.retry_jitter_fraction <= 0:
        return base
    seed = hashlib.sha256(f"{dataset}:{batch_index}:{attempt}".encode("utf-8")).digest()
    unit_interval = int.from_bytes(seed[:4], "big") / 0xFFFFFFFF
    return base * (1.0 + args.retry_jitter_fraction * unit_interval)


def invoke_codex(
    args: argparse.Namespace,
    schema_path: Path,
    response_path: Path,
    payload: dict[str, Any],
) -> tuple[int, str, str, dict[str, int] | None]:
    command = codex_command(args, schema_path, response_path)
    timeout = args.request_timeout_seconds if args.request_timeout_seconds > 0 else None
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            124,
            str(exc.stdout or ""),
            f"Codex timeout after {args.request_timeout_seconds} seconds\n{exc.stderr or ''}",
            extract_codex_usage(str(exc.stdout or "")),
        )
    except OSError as exc:
        return 127, "", f"Could not invoke Codex: {exc}", None
    return completed.returncode, completed.stdout, completed.stderr, extract_codex_usage(completed.stdout)


def record_batch_status(
    connection: sqlite3.Connection,
    dataset: str,
    batch_index: int,
    input_hash: str,
    pair_count: int,
    status: str,
    attempts: int,
    error: str | None = None,
    completed: bool = False,
) -> None:
    now = utc_now()
    connection.execute(
        """
        INSERT INTO batches(dataset, batch_index, input_sha256, pair_count, status, attempts, started_at, completed_at, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset, batch_index) DO UPDATE SET
            input_sha256=excluded.input_sha256,
            pair_count=excluded.pair_count,
            status=excluded.status,
            attempts=excluded.attempts,
            started_at=CASE
                WHEN excluded.status = 'running' THEN excluded.started_at
                ELSE batches.started_at
            END,
            completed_at=excluded.completed_at,
            last_error=excluded.last_error
        """,
        (
            dataset,
            batch_index,
            input_hash,
            pair_count,
            status,
            attempts,
            now,
            now if completed else None,
            (error or "")[:4000] or None,
        ),
    )
    connection.commit()


def process_batch(
    args: argparse.Namespace,
    connection: sqlite3.Connection,
    schema_path: Path,
    output_root: Path,
    dataset: str,
    batch_index: int,
    question_batch: list[dict[str, Any]],
) -> tuple[int, bool]:
    expected = flatten_batch_metadata(question_batch)
    payload = prompt_payload(question_batch)
    input_hash = sha256_json(payload)
    result_path = batch_path(output_root, dataset, batch_index)
    existing = load_reusable_batch(args, output_root, dataset, batch_index, input_hash, expected) if args.resume else None
    if existing is not None:
        labels, existing_path = existing
        record_batch_status(
            connection, dataset, batch_index, input_hash, len(labels), "completed", 0, completed=True
        )
        if existing_path != result_path:
            logging.debug("[%s batch %d] reused prior-run artifact: %s", dataset, batch_index, existing_path)
        return len(labels), False
    if args.consolidate_only:
        raise RuntimeError(f"Missing or invalid batch result required for consolidation: {result_path}")
    if args.dry_run:
        return len(expected), False

    last_error = ""
    for attempt in range(1, args.max_attempts + 1):
        raw_response_path = output_root / "work" / dataset / f"batch_{batch_index:06d}.attempt_{attempt}.json"
        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_response_path.exists():
            raw_response_path.unlink()
        record_batch_status(
            connection, dataset, batch_index, input_hash, len(expected), "running", attempt, error=last_error
        )
        logging.info(
            "[%s batch %d] submitting %d pair(s) to Codex (attempt %d/%d, reasoning=%s)",
            dataset,
            batch_index,
            len(expected),
            attempt,
            args.max_attempts,
            args.model_reasoning_effort or "configured_default",
        )
        started = time.monotonic()
        returncode, stdout, stderr, usage = invoke_codex(args, schema_path, raw_response_path, payload)
        try:
            if returncode != 0:
                raise RuntimeError(codex_failure_summary(returncode, stdout, stderr))
            elapsed_seconds = time.monotonic() - started
            response_text = (
                raw_response_path.read_text(encoding="utf-8")
                if raw_response_path.is_file()
                else final_message_from_codex_events(stdout)
            )
            response = json.loads(response_text)
            normalized = validate_response(response, expected)
            batch_result = {
                "annotation_version": ANNOTATION_VERSION,
                "prompt_version": PROMPT_VERSION,
                "dataset": dataset,
                "batch_index": batch_index,
                "input_sha256": input_hash,
                "codex_model_request": args.model or "configured_default",
                "codex_reasoning_effort": args.model_reasoning_effort or "configured_default",
                "web_search_enabled": bool(args.enable_web_search),
                "elapsed_seconds": elapsed_seconds,
                "usage": usage,
                "completed_at": utc_now(),
                "labels": normalized,
            }
            write_json_atomic(result_path, batch_result)
            record_batch_status(
                connection, dataset, batch_index, input_hash, len(normalized), "completed", attempt, completed=True
            )
            raw_response_path.unlink(missing_ok=True)
            logging.info(
                "[%s batch %d] completed %d pair(s) in %.1fs",
                dataset,
                batch_index,
                len(normalized),
                elapsed_seconds,
            )
            return len(normalized), True
        except Exception as exc:
            last_error = str(exc)
            failure_path = raw_attempt_path(output_root, dataset, batch_index, attempt)
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                "COMMAND FAILED OR INVALID RESPONSE\n\n"
                + last_error
                + "\n\n--- STDERR ---\n"
                + stderr[-12000:]
                + "\n\n--- STDOUT ---\n"
                + stdout[-12000:],
                encoding="utf-8",
            )
            raw_response_path.unlink(missing_ok=True)
            record_batch_status(
                connection, dataset, batch_index, input_hash, len(expected), "failed", attempt, error=last_error
            )
            if attempt < args.max_attempts:
                delay = retry_delay_seconds(args, dataset, batch_index, attempt)
                logging.warning(
                    "[%s batch %d] attempt %d/%d failed: %s; retrying in %.1fs",
                    dataset,
                    batch_index,
                    attempt,
                    args.max_attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"[{dataset} batch {batch_index}] failed after {args.max_attempts} attempts: {last_error}")


def expected_batches(path: Path, args: argparse.Namespace) -> Iterator[tuple[str, int, list[dict[str, Any]]]]:
    dataset = ""
    for batch_index, question_batch in enumerate(batched(iter_question_items(path, args), args.questions_per_batch)):
        batch_dataset = question_batch[0]["dataset"]
        if not dataset:
            dataset = batch_dataset
        elif dataset != batch_dataset:
            raise ValueError(f"Dataset changed within candidates file: {path}")
        yield batch_dataset, batch_index, question_batch


def consolidated_rows(path: Path, args: argparse.Namespace, output_root: Path) -> Iterator[dict[str, Any]]:
    for dataset, batch_index, question_batch in expected_batches(path, args):
        expected = flatten_batch_metadata(question_batch)
        payload = prompt_payload(question_batch)
        input_hash = sha256_json(payload)
        loaded = load_reusable_batch(args, output_root, dataset, batch_index, input_hash, expected)
        if loaded is None:
            raise RuntimeError(f"Cannot consolidate unfinished or invalid batch: {batch_path(output_root, dataset, batch_index)}")
        labels, _ = loaded
        yield from labels


def write_pending_plan(
    args: argparse.Namespace,
    path_dataset: dict[Path, str],
    dataset_counts: dict[str, dict[str, int]],
) -> None:
    if args.pending_plan_path is None:
        raise ValueError("--write-pending-plan-only requires --pending-plan-path")
    entries: list[dict[str, Any]] = []
    reusable_pairs = 0
    for path in args.candidates_paths:
        dataset = path_dataset[path]
        for batch_dataset, batch_index, question_batch in expected_batches(path, args):
            if batch_dataset != dataset:
                raise RuntimeError(f"Unexpected batch dataset: {batch_dataset} != {dataset}")
            expected = flatten_batch_metadata(question_batch)
            input_hash = sha256_json(prompt_payload(question_batch))
            loaded = load_reusable_batch(args, args.output_root, dataset, batch_index, input_hash, expected)
            if loaded is not None:
                labels, _ = loaded
                reusable_pairs += len(labels)
                continue
            entries.append(
                {
                    "dataset": dataset,
                    "batch_index": batch_index,
                    "pair_count": len(expected),
                }
            )
    plan = {
        "version": "rag2_codex_pending_batch_plan_v1",
        "created_at": utc_now(),
        "candidates_paths": [str(path.resolve()) for path in args.candidates_paths],
        "docs_per_question": args.docs_per_question,
        "questions_per_batch": args.questions_per_batch,
        "dataset_counts": dataset_counts,
        "reusable_pairs": reusable_pairs,
        "pending_pairs": sum(int(entry["pair_count"]) for entry in entries),
        "pending_batches": entries,
    }
    args.pending_plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.pending_plan_path, plan)
    logging.info(
        "Pending plan written: reusable_pairs=%d pending_pairs=%d pending_batches=%d path=%s",
        reusable_pairs,
        plan["pending_pairs"],
        len(entries),
        args.pending_plan_path,
    )


def load_pending_plan(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if args.pending_plan_path is None:
        return None
    value = json.loads(args.pending_plan_path.read_text(encoding="utf-8"))
    if value.get("version") != "rag2_codex_pending_batch_plan_v1":
        raise ValueError(f"Unsupported pending plan: {args.pending_plan_path}")
    expected_paths = [str(path.resolve()) for path in args.candidates_paths]
    if value.get("candidates_paths") != expected_paths:
        raise ValueError("Pending plan candidates paths do not match this run")
    if value.get("docs_per_question") != args.docs_per_question:
        raise ValueError("Pending plan docs-per-question does not match this run")
    if value.get("questions_per_batch") != args.questions_per_batch:
        raise ValueError("Pending plan questions-per-batch does not match this run")
    entries = value.get("pending_batches")
    if not isinstance(entries, list):
        raise ValueError("Pending plan has no pending_batches list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Pending plan contains a non-object entry")
        dataset = str(entry.get("dataset") or "")
        batch_index = entry.get("batch_index")
        pair_count = entry.get("pair_count")
        if not dataset or not isinstance(batch_index, int) or not isinstance(pair_count, int) or pair_count <= 0:
            raise ValueError(f"Invalid pending plan entry: {entry!r}")
        key = (dataset, batch_index)
        if key in seen:
            raise ValueError(f"Duplicate pending plan entry: {key}")
        seen.add(key)
        normalized.append({"dataset": dataset, "batch_index": batch_index, "pair_count": pair_count})
    return normalized


def write_run_manifest(
    output_root: Path,
    args: argparse.Namespace,
    dataset_counts: dict[str, dict[str, int]],
    status: str,
) -> None:
    manifest = {
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "updated_at": utc_now(),
        "status": status,
        "candidates_paths": [str(path.resolve()) for path in args.candidates_paths],
        "docs_per_question": args.docs_per_question,
        "questions_per_batch": args.questions_per_batch,
        "max_doc_chars": args.max_doc_chars,
        "codex_bin": args.codex_bin,
        "codex_model_request": args.model or "configured_default",
        "codex_reasoning_effort": args.model_reasoning_effort or "configured_default",
        "web_search_enabled": bool(args.enable_web_search),
        "worker_count": args.worker_count,
        "reuse_validated_completed_batches": bool(args.reuse_validated_completed_batches),
        "completed_batches_root": str(args.completed_batches_root.resolve()) if args.completed_batches_root else None,
        "pending_plan_path": str(args.pending_plan_path.resolve()) if args.pending_plan_path else None,
        "label_definitions": {
            "direct_support": "The chunk contains a direct medical fact, criterion, relationship, or recommendation that substantially justifies the reference answer.",
            "supporting_evidence": "The chunk supplies a medically valid, case-relevant premise for a correct reasoning chain, but cannot itself justify the reference answer.",
            "no_evidence": "The chunk contains no usable proposition that materially supports the reference answer or pushes toward an incompatible answer.",
            "misleading_evidence": "An explicit chunk claim, reasonably applied to the question, contradicts the reference answer or plausibly leads toward an incompatible answer.",
            "indeterminate_or_mixed": "Only for uninterpretable text or comparable answer-supporting and answer-opposing claims in the displayed chunk; not for ordinary chunk boundaries or missing context.",
        },
        "topic_relation_definitions": {
            "related": "For no_evidence only: clinically related topic but no answer-useful evidence.",
            "unrelated": "For no_evidence only: not clinically relevant to solving the question.",
            "unclear": "For no_evidence only: related versus unrelated cannot be determined from the displayed chunk.",
        },
        "datasets": dataset_counts,
    }
    write_json_atomic(output_root / "manifest.json", manifest)


def validate_args(args: argparse.Namespace) -> None:
    if args.docs_per_question <= 0:
        raise ValueError("--docs-per-question must be positive")
    if args.questions_per_batch <= 0:
        raise ValueError("--questions-per-batch must be positive")
    if args.max_doc_chars < 0:
        raise ValueError("--max-doc-chars must be non-negative")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.request_timeout_seconds < 0 or args.retry_backoff_seconds < 0:
        raise ValueError("Timeout and retry backoff must be non-negative")
    if not 0.0 <= args.retry_jitter_fraction <= 1.0:
        raise ValueError("--retry-jitter-fraction must be between 0 and 1")
    if args.limit_questions < 0 or args.stop_after_batches < 0:
        raise ValueError("Limits must be non-negative")
    if args.worker_count <= 0:
        raise ValueError("--worker-count must be positive")
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("--worker-index must be in [0, --worker-count)")
    if len(set(args.candidates_paths)) != len(args.candidates_paths):
        raise ValueError("Duplicate --candidates-paths entry")
    for path in args.candidates_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.completed_batches_root is not None and not args.completed_batches_root.is_dir():
        raise FileNotFoundError(args.completed_batches_root)
    if args.consolidate_only and args.dry_run:
        raise ValueError("--consolidate-only and --dry-run cannot be used together")
    if args.write_pending_plan_only and args.pending_plan_path is None:
        raise ValueError("--write-pending-plan-only requires --pending-plan-path")
    if args.write_pending_plan_only and (args.consolidate_only or args.dry_run):
        raise ValueError("--write-pending-plan-only cannot be combined with --consolidate-only or --dry-run")
    if args.pending_plan_path is not None and not args.write_pending_plan_only and not args.pending_plan_path.is_file():
        raise FileNotFoundError(args.pending_plan_path)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    schema_path = args.output_root / "codex_output_schema.json"
    if schema_path.exists():
        old_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if old_schema != schema():
            if not args.reuse_validated_completed_batches:
                raise RuntimeError(f"Existing output root has an incompatible schema: {schema_path}")
            logging.warning(
                "Upgrading response schema in %s; prior completed batches will be revalidated and reused.", schema_path
            )
            write_json_atomic(schema_path, schema())
    else:
        write_json_atomic(schema_path, schema())

    dataset_counts: dict[str, dict[str, int]] = {}
    path_dataset: dict[Path, str] = {}
    for path in args.candidates_paths:
        dataset, questions, pairs = count_question_items(path, args)
        if dataset in dataset_counts:
            raise ValueError(f"Only one candidates file per dataset is supported: {dataset}")
        path_dataset[path] = dataset
        dataset_counts[dataset] = {
            "questions": questions,
            "pairs": pairs,
            "planned_batches": math.ceil(questions / args.questions_per_batch),
        }
    write_run_manifest(args.output_root, args, dataset_counts, "planned")
    total_pairs = sum(value["pairs"] for value in dataset_counts.values())
    total_batches = sum(value["planned_batches"] for value in dataset_counts.values())
    if args.write_pending_plan_only:
        write_pending_plan(args, path_dataset, dataset_counts)
        return

    pending_entries = None if args.consolidate_only else load_pending_plan(args)
    assigned_pending_keys: set[tuple[str, int]] | None = None
    assigned_pairs = total_pairs
    if pending_entries is not None:
        assigned_entries = [
            entry for ordinal, entry in enumerate(pending_entries) if ordinal % args.worker_count == args.worker_index
        ]
        assigned_pending_keys = {
            (str(entry["dataset"]), int(entry["batch_index"])) for entry in assigned_entries
        }
        assigned_pairs = sum(int(entry["pair_count"]) for entry in assigned_entries)
    elif not args.consolidate_only and args.worker_count > 1:
        assigned_pairs = 0
        for value in dataset_counts.values():
            questions = value["questions"]
            for batch_index in range(value["planned_batches"]):
                if batch_index % args.worker_count != args.worker_index:
                    continue
                batch_questions = min(args.questions_per_batch, questions - batch_index * args.questions_per_batch)
                assigned_pairs += batch_questions * args.docs_per_question
    logging.info(
        "Planned Codex semantic labelling: datasets=%s questions=%d pairs=%d batches=%d (%d question(s)/batch, Top-%d)",
        ",".join(sorted(dataset_counts)),
        sum(value["questions"] for value in dataset_counts.values()),
        total_pairs,
        total_batches,
        args.questions_per_batch,
        args.docs_per_question,
    )
    if args.worker_count > 1 and not args.consolidate_only:
        logging.info(
            "Worker %d/%d owns %d pair(s)%s.",
            args.worker_index,
            args.worker_count,
            assigned_pairs,
            " from the frozen pending plan" if pending_entries is not None else f" of {total_pairs} planned",
        )
    if args.dry_run:
        write_run_manifest(args.output_root, args, dataset_counts, "dry_run_complete")
        return

    progress_db_path = args.progress_db_path or (args.output_root / "progress.sqlite")
    progress_db_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress_database_pointer(args.output_root, progress_db_path)
    connection = configure_database(progress_db_path)
    completed_new_batches = 0
    try:
        progress = tqdm(total=assigned_pairs, desc="CodexSemanticLabels", unit="pair")
        for path in args.candidates_paths:
            dataset = path_dataset[path]
            for batch_dataset, batch_index, question_batch in expected_batches(path, args):
                if batch_dataset != dataset:
                    raise RuntimeError(f"Unexpected batch dataset: {batch_dataset} != {dataset}")
                if not args.consolidate_only:
                    if assigned_pending_keys is not None:
                        if (dataset, batch_index) not in assigned_pending_keys:
                            continue
                    elif batch_index % args.worker_count != args.worker_index:
                        continue
                pair_count, newly_completed = process_batch(
                    args,
                    connection,
                    schema_path,
                    args.output_root,
                    dataset,
                    batch_index,
                    question_batch,
                )
                progress.update(pair_count)
                if newly_completed:
                    completed_new_batches += 1
                    if args.stop_after_batches and completed_new_batches >= args.stop_after_batches:
                        progress.close()
                        write_run_manifest(args.output_root, args, dataset_counts, "stopped_after_requested_batches")
                        logging.info("Stopped cleanly after %d newly completed batch(es).", completed_new_batches)
                        return
        progress.close()
    finally:
        connection.close()

    if args.worker_count > 1 and not args.consolidate_only:
        write_run_manifest(args.output_root, args, dataset_counts, "worker_complete_waiting_for_consolidation")
        logging.info(
            "Worker %d completed.  After every worker exits, run one --consolidate-only command.", args.worker_index
        )
        return

    for path in args.candidates_paths:
        dataset = path_dataset[path]
        final_path = args.output_root / dataset / "codex_semantic_labels.jsonl"
        logging.info("Consolidating completed %s batches into %s", dataset, final_path)
        write_jsonl_atomic(final_path, consolidated_rows(path, args, args.output_root))
    write_run_manifest(args.output_root, args, dataset_counts, "complete")
    logging.info("Codex semantic labelling complete: %s", args.output_root)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        run(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted. Completed batch files are preserved; rerun with --resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a compact, stratified audit set from completed Codex RAG² labels.

The output is intended for independent LLM or human review.  It contains only
the question, reference answer, retrieved chunk, and the stored Codex judgement
with its explanation; it deliberately omits the original model prompt and
retrieval rationale to keep the review task focused on evidence utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


LABEL_DEFINITIONS = {
    "direct_support": (
        "The chunk contains a direct medical fact, criterion, relationship, or recommendation "
        "that substantially justifies the reference answer."
    ),
    "supporting_evidence": (
        "The chunk supplies a medically valid, case-relevant premise for a correct reasoning chain, "
        "but cannot itself justify the reference answer."
    ),
    "no_evidence": (
        "The chunk contains no usable proposition that materially supports the reference answer or "
        "pushes toward an incompatible answer. For this label only, topic_relation specifies related, "
        "unrelated, or unclear."
    ),
    "misleading_evidence": (
        "An explicit chunk claim, reasonably applied to the question, contradicts the reference answer "
        "or plausibly leads toward an incompatible answer."
    ),
    "indeterminate_or_mixed": (
        "Use only for uninterpretable text or comparable answer-supporting and answer-opposing claims "
        "in the displayed chunk; ordinary chunk boundaries or missing context are not sufficient."
    ),
}
LABEL_ORDER = tuple(LABEL_DEFINITIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medmcqa-candidates", type=Path, required=True)
    parser.add_argument("--medqa-candidates", type=Path, required=True)
    parser.add_argument(
        "--label-batches-root",
        type=Path,
        required=True,
        help="Run root containing batches/medmcqa and batches/medqa.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score(seed: int, pair_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sentence_indexed_text(text: str) -> str:
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    pieces = [piece.strip() for piece in pieces if piece.strip()]
    return "\n".join(f"[S{index}] {piece}" for index, piece in enumerate(pieces, start=1))


def iter_label_rows(root: Path, dataset: str) -> Iterator[tuple[dict[str, Any], Path]]:
    for path in sorted((root / "batches" / dataset).glob("batch_*.json")):
        value = read_json(path)
        labels = value.get("labels")
        if not isinstance(labels, list):
            continue
        for row in labels:
            if isinstance(row, dict) and row.get("semantic_label") in LABEL_DEFINITIONS:
                yield row, path


def select_labels(root: Path, dataset: str, sample_count: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically sample a balanced label set, then fill rare-label gaps."""
    target = sample_count // len(LABEL_ORDER)
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {label: [] for label in LABEL_ORDER}
    fallback: list[tuple[int, dict[str, Any]]] = []
    observed = Counter()
    for row, path in iter_label_rows(root, dataset):
        label = str(row["semantic_label"])
        observed[label] += 1
        item = {**row, "label_source_batch": str(path)}
        item_score = score(seed, str(row["pair_id"]))
        buckets[label].append((item_score, item))
        fallback.append((item_score, item))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for label in LABEL_ORDER:
        for _, item in sorted(buckets[label], key=lambda value: value[0])[:target]:
            selected.append(item)
            selected_ids.add(str(item["pair_id"]))
    for _, item in sorted(fallback, key=lambda value: value[0]):
        if len(selected) >= sample_count:
            break
        pair_id = str(item["pair_id"])
        if pair_id not in selected_ids:
            selected.append(item)
            selected_ids.add(pair_id)

    if len(selected) < sample_count:
        raise RuntimeError(f"Only {len(selected)} valid {dataset} labels were available; requested {sample_count}.")
    return selected


def candidate_index(path: Path, required_pair_ids: set[str]) -> dict[str, dict[str, Any]]:
    joined: dict[str, dict[str, Any]] = {}
    required_sample_ids = {pair_id.split("::", 1)[0] for pair_id in required_pair_ids}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if sample_id not in required_sample_ids:
                continue
            documents = row.get("candidate_documents")
            if not isinstance(documents, list):
                continue
            options = row.get("options") if isinstance(row.get("options"), dict) else {}
            answer_key = str(row.get("answer") or "")
            answer_text = options.get(answer_key)
            for document in documents[:8]:
                if not isinstance(document, dict):
                    continue
                stable_id = str(document.get("stable_id") or "")
                doc_rank = document.get("rerank_rank")
                if not stable_id or not isinstance(doc_rank, int):
                    continue
                pair_id = f"{sample_id}::{doc_rank}::{stable_id}"
                if pair_id not in required_pair_ids:
                    continue
                joined[pair_id] = {
                    "question": row.get("question"),
                    "options": options,
                    "reference_answer_key": answer_key,
                    "reference_answer_text": answer_text,
                    "document_source": document.get("source"),
                    "document_title": document.get("title"),
                    "document_rerank_rank": doc_rank,
                    "document_rerank_score": document.get("rerank_score"),
                    "document_text": document.get("text"),
                }
    return joined


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_definitions(path: Path) -> None:
    lines = [
        "# Codex Evidence-Utility Label Definitions",
        "",
        "Review each chunk as a standalone evidence unit for the supplied reference answer.",
        "Do not infer support from topic, keywords, or entities alone; identify a concrete proposition in the chunk.",
        "",
    ]
    for label in LABEL_ORDER:
        lines.extend([f"## `{label}`", "", LABEL_DEFINITIONS[label], ""])
    lines.extend(
        [
            "## Review fields",
            "",
            "- `topic_relation` is populated only for `no_evidence`: `related`, `unrelated`, or `unclear`.",
            "- `evidence_sentence_indices` refer to the `[S#]` sentences in `document_sentences`.",
            "- `codex_short_reason` is the original short explanation from the Codex labeler, not an independently verified rationale.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_dataset < len(LABEL_ORDER):
        raise ValueError("--samples-per-dataset must be at least the number of labels")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_by_dataset = {
        "medmcqa": select_labels(args.label_batches_root, "medmcqa", args.samples_per_dataset, args.seed),
        "medqa": select_labels(args.label_batches_root, "medqa", args.samples_per_dataset, args.seed + 1),
    }
    candidate_paths = {"medmcqa": args.medmcqa_candidates, "medqa": args.medqa_candidates}
    exported: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "seed": args.seed,
        "samples_per_dataset_requested": args.samples_per_dataset,
        "label_batches_root": str(args.label_batches_root),
        "label_definitions": LABEL_DEFINITIONS,
        "datasets": {},
    }
    for dataset, selected in selected_by_dataset.items():
        required_pair_ids = {str(row["pair_id"]) for row in selected}
        joined = candidate_index(candidate_paths[dataset], required_pair_ids)
        missing = sorted(required_pair_ids - set(joined))
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} selected labels could not be joined to Top-8 candidates: {missing[:3]}")
        label_counts = Counter()
        source_counts = Counter()
        for label_row in selected:
            pair_id = str(label_row["pair_id"])
            candidate = joined[pair_id]
            text = str(candidate.get("document_text") or "")
            label = str(label_row["semantic_label"])
            label_counts[label] += 1
            source_counts[str(candidate.get("document_source") or "unknown")] += 1
            exported.append(
                {
                    "dataset": dataset,
                    "sample_id": label_row["sample_id"],
                    "pair_id": pair_id,
                    "question": candidate["question"],
                    "options": candidate["options"],
                    "reference_answer_key": candidate["reference_answer_key"],
                    "reference_answer_text": candidate["reference_answer_text"],
                    "document_source": candidate["document_source"],
                    "document_title": candidate["document_title"],
                    "document_rerank_rank": candidate["document_rerank_rank"],
                    "document_rerank_score": candidate["document_rerank_score"],
                    "document_text": text,
                    "document_sentences": sentence_indexed_text(text),
                    "codex_label": label,
                    "codex_topic_relation": label_row.get("topic_relation"),
                    "codex_confidence": label_row.get("confidence"),
                    "codex_evidence_sentence_indices": label_row.get("evidence_sentence_indices"),
                    "codex_short_reason": label_row.get("short_reason"),
                    "label_definition": LABEL_DEFINITIONS[label],
                    "label_source_batch": label_row["label_source_batch"],
                }
            )
        summary["datasets"][dataset] = {
            "sampled_pairs": len(selected),
            "label_counts": dict(sorted(label_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
        }

    random.Random(args.seed).shuffle(exported)
    write_jsonl(args.output_dir / "codex_evidence_utility_audit_sample_400.jsonl", exported)
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_definitions(args.output_dir / "label_definitions.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {len(exported)} review pairs to {args.output_dir}")


if __name__ == "__main__":
    main()

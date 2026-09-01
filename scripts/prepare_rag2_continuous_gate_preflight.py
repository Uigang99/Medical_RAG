#!/usr/bin/env python3
"""Build an internal-validation Top-k cohort for continuous-gate preflight.

The source candidate files contain every MedMCQA/MedQA training question,
whereas semantic filter splits are question-level train/val/test partitions of
that source.  This utility joins the validation partition back to the exact
reranked documents and writes the compact contract expected by the existing
exact-subset scorer.  The final benchmark test set is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_continuous_gate_preflight_cohort_v1"
DEFAULT_BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)
SEMANTIC_LABELS = {
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["medmcqa", "medqa"])
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--semantic-split", choices=("val", "test"), default="val")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_BASE / "candidates/source_balanced32_rerank8_v1")
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_BASE / "filter_training_inputs_semantic_top8_four_class_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-questions-per-dataset", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("chunk_id") or document.get("db_id")
    if not value:
        raise ValueError("Candidate document has no stable identity")
    return str(value)


def candidate_total(path: Path) -> int:
    manifest_path = path.with_name("candidate_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total = int(manifest.get("selected_question_count") or 0)
        if total > 0:
            return total
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def load_semantic_rows(path: Path, top_k: int) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    label_counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        rank = int(row.get("doc_rank") or 0)
        if rank <= 0 or rank > top_k:
            continue
        sample_id = str(row.get("sample_id") or "")
        document_id = str(row.get("doc_stable_id") or "")
        label = str(row.get("label") or row.get("target") or "")
        if not sample_id or not document_id or label not in SEMANTIC_LABELS:
            raise ValueError(f"Invalid semantic row: {row}")
        if document_id in by_sample[sample_id]:
            raise ValueError(f"Duplicate semantic question-document pair: {(sample_id, document_id)}")
        by_sample[sample_id][document_id] = row
        label_counts[label] += 1
    return dict(by_sample), dict(label_counts)


def choose_sample_ids(
    semantic: dict[str, dict[str, dict[str, Any]]],
    top_k: int,
    maximum: int,
    seed: int,
) -> list[str]:
    complete = sorted(sample_id for sample_id, rows in semantic.items() if len(rows) == top_k)
    if maximum > 0 and len(complete) > maximum:
        complete = sorted(random.Random(seed).sample(complete, maximum))
    return complete


def prepare_dataset(args: argparse.Namespace, dataset: str, progress: PipelineProgress) -> dict[str, Any]:
    candidate_path = args.candidate_root / dataset / args.source_split / "candidates_top8.jsonl"
    semantic_path = args.semantic_root / dataset / f"{args.semantic_split}.jsonl"
    for required in (candidate_path, semantic_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    semantic, source_label_counts = load_semantic_rows(semantic_path, args.top_k)
    selected_ids = set(
        choose_sample_ids(
            semantic,
            args.top_k,
            args.max_questions_per_dataset,
            args.sample_seed + sum(map(ord, dataset)),
        )
    )
    if not selected_ids:
        raise RuntimeError(f"No complete Top-{args.top_k} semantic questions for {dataset}")

    total_candidates = candidate_total(candidate_path)
    progress.set_stage(
        f"1/1 join {dataset} internal-{args.semantic_split} Top-{args.top_k}",
        total=total_candidates,
    )
    compact_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    found: set[str] = set()
    label_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    for row in iter_jsonl(candidate_path):
        progress.update(1)
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in selected_ids:
            continue
        if sample_id in found:
            raise ValueError(f"Duplicate candidate question: {sample_id}")
        documents = list(row.get("candidate_documents") or [])[: args.top_k]
        if len(documents) != args.top_k:
            raise ValueError(f"Incomplete candidate Top-{args.top_k}: {sample_id}")
        joined_documents: list[dict[str, Any]] = []
        ids: list[str] = []
        labels: list[str] = []
        for rank, document in enumerate(documents, start=1):
            document_id = stable_id(document)
            decision = semantic[sample_id].get(document_id)
            if decision is None:
                raise KeyError(f"Missing semantic decision: {(sample_id, document_id)}")
            label = str(decision.get("label") or decision.get("target"))
            ids.append(document_id)
            labels.append(label)
            label_counts[label] += 1
            joined_documents.append({**document, "stable_id": document_id})
            semantic_rows.append(
                {
                    "run_version": RUN_VERSION,
                    "sample_key": sample_id,
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "split": args.semantic_split,
                    "row_idx": int(row.get("row_idx", -1)),
                    "doc_rank": rank,
                    "doc_stable_id": document_id,
                    "semantic_label": label,
                    "semantic_confidence": decision.get("codex_confidence"),
                }
            )
        has_support = any(label in {"direct_support", "supporting_evidence"} for label in labels)
        has_nonsupport = any(label in {"no_evidence", "misleading_evidence"} for label in labels)
        composition = (
            "mixed_support_nonsupport"
            if has_support and has_nonsupport
            else "support_only"
            if has_support
            else "nonsupport_only"
        )
        composition_counts[composition] += 1
        options = row.get("options") or {}
        if set(options) != {"A", "B", "C", "D"}:
            raise ValueError(f"Expected four choices for {sample_id}")
        compact_rows.append(
            {
                "run_version": RUN_VERSION,
                "key": sample_id,
                "sample_id": sample_id,
                "dataset": dataset,
                "split": args.semantic_split,
                "source_split": args.source_split,
                "row_idx": int(row.get("row_idx", -1)),
                "question": row["question"],
                "options": options,
                "answer": row.get("answer"),
                "answers": row.get("answers") or [row.get("answer")],
                "selected_document_ids_by_top_k": {str(args.top_k): ids},
                "candidate_documents": joined_documents,
                "semantic_composition": composition,
            }
        )
        found.add(sample_id)

    missing = selected_ids - found
    if missing:
        raise RuntimeError(f"Candidate join missed {len(missing)} selected questions for {dataset}")
    compact_rows.sort(key=lambda row: int(row["row_idx"]))
    semantic_rows.sort(key=lambda row: (int(row["row_idx"]), int(row["doc_rank"])))
    dataset_root = args.output_root / "candidate_union" / dataset / args.semantic_split
    candidate_output = dataset_root / "candidates_topk_union.jsonl"
    semantic_output = args.output_root / "semantic_labels" / f"{dataset}.jsonl"
    atomic_jsonl(candidate_output, compact_rows)
    atomic_jsonl(semantic_output, semantic_rows)
    return {
        "dataset": dataset,
        "questions": len(compact_rows),
        "documents": len(semantic_rows),
        "candidate_source": str(candidate_path.resolve()),
        "semantic_source": str(semantic_path.resolve()),
        "candidate_output": str(candidate_output.resolve()),
        "semantic_output": str(semantic_output.resolve()),
        "candidate_output_sha256": sha256_file(candidate_output),
        "semantic_output_sha256": sha256_file(semantic_output),
        "semantic_source_label_counts_top_k": source_label_counts,
        "selected_label_counts": dict(label_counts),
        "composition_counts": dict(composition_counts),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.top_k <= 0 or args.top_k > 8:
        raise ValueError("--top-k must be in [1, 8]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "cohort_manifest.json"
    if manifest_path.is_file() and args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "run_version": RUN_VERSION,
            "datasets": args.datasets,
            "source_split": args.source_split,
            "semantic_split": args.semantic_split,
            "top_k": args.top_k,
            "max_questions_per_dataset": args.max_questions_per_dataset,
            "sample_seed": args.sample_seed,
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            outputs_exist = all(
                (args.output_root / "candidate_union" / dataset / args.semantic_split / "candidates_topk_union.jsonl").is_file()
                and (args.output_root / "semantic_labels" / f"{dataset}.jsonl").is_file()
                for dataset in args.datasets
            ) and (args.output_root / "semantic_labels" / "all.jsonl").is_file()
            if outputs_exist:
                logging.info("Complete preflight cohort is unchanged: %s", args.output_root)
                return
        raise RuntimeError("Preflight cohort resume contract mismatch; use a new output directory")

    totals = []
    for dataset in args.datasets:
        path = args.candidate_root / dataset / args.source_split / "candidates_top8.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        totals.append(candidate_total(path))
    progress = PipelineProgress(overall_total=sum(totals), desc="PrepareGatePreflight")
    try:
        datasets = [prepare_dataset(args, dataset, progress) for dataset in args.datasets]
    finally:
        progress.close()
    manifest = {
        "run_version": RUN_VERSION,
        "datasets": args.datasets,
        "source_split": args.source_split,
        "semantic_split": args.semantic_split,
        "top_k": args.top_k,
        "max_questions_per_dataset": args.max_questions_per_dataset,
        "sample_seed": args.sample_seed,
        "dataset_outputs": datasets,
    }
    combined_semantic_path = args.output_root / "semantic_labels" / "all.jsonl"
    combined_semantic_rows = (
        row
        for dataset in args.datasets
        for row in iter_jsonl(args.output_root / "semantic_labels" / f"{dataset}.jsonl")
    )
    combined_count = atomic_jsonl(combined_semantic_path, combined_semantic_rows)
    manifest["combined_semantic_output"] = str(combined_semantic_path.resolve())
    manifest["combined_semantic_rows"] = combined_count
    manifest["combined_semantic_sha256"] = sha256_file(combined_semantic_path)
    atomic_json(manifest_path, manifest)
    logging.info("Continuous-gate preflight cohort ready: %s", manifest_path)


if __name__ == "__main__":
    main()

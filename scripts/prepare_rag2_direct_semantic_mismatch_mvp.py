#!/usr/bin/env python3
"""Prepare a bounded direct-choice semantic/behavior mismatch LoRA pilot.

The immutable direct-choice cache supplies frozen Llama-3 four-option
distributions.  Semantic labels and correctness transitions are used only to
choose training cases; neither is written into the model prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import logging
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import (  # noqa: E402
    PROMPT_POLICY_VERSION,
    RUN_VERSION as DIRECT_OUTCOME_VERSION,
)
from generate_rag2_anchored_document_traces import normalized_candidate_rows  # noqa: E402
from medrag.progress import StageProgress  # noqa: E402
from medrag.training.direct_semantic_mismatch import TRAIN_CASES  # noqa: E402


RUN_VERSION = "rag2_direct_semantic_mismatch_mvp_pairs_v1"
SPLITS = ("train", "val", "test")
PRIMARY_FAILURE_CASES = (
    "direct_support_w2w",
    "direct_support_c2w",
    "no_evidence_c2w",
)
BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), default="medmcqa")
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=BASE / "filter_training_inputs_semantic_top8_four_class_v1",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=BASE / "candidates/source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--outcome-root",
        type=Path,
        default=BASE / "anchored_direct_choice_single_document_outcomes_source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE / "direct_semantic_mismatch_mvp_pairs_v1",
    )
    parser.add_argument("--max-train-questions", type=int, default=4000)
    parser.add_argument("--max-eval-questions", type=int, default=4000)
    parser.add_argument("--train-failure-fraction", type=float, default=0.80)
    parser.add_argument("--questions-per-shard", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def semantic_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return fields that determine prepared-data meaning, excluding provenance."""
    return {
        key: value
        for key, value in contract.items()
        if key not in {"code_commit", "code_sha256"}
    }


def prepared_contract_matches(
    stored: dict[str, Any], current: dict[str, Any], current_fingerprint: str
) -> bool:
    if stored.get("contract_fingerprint") == current_fingerprint:
        return True
    # Legacy manifests included the repository commit in their fingerprint.
    # Semantic preparation changes must bump RUN_VERSION; source hash and
    # repository commit are provenance and may change after compatibility fixes.
    expected = semantic_contract(current)
    return all(stored.get(key) == value for key, value in expected.items())


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        value["sha256"] = sha256_file(path)
    return value


def stable_priority(seed: int, sample_id: str) -> int:
    payload = f"{seed}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class LowestPriorityRecords:
    """Bounded deterministic reservoir retaining the lowest hash priorities."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, priority: int, sample_id: str, record: dict[str, Any]) -> None:
        item = (-int(priority), str(sample_id), record)
        if self.limit <= 0:
            heapq.heappush(self.heap, item)
        elif len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def records(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heap, key=lambda item: (-item[0], item[1]))]


def semantic_index(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str], int]:
    manifest_path = args.semantic_root / args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != args.dataset or int(manifest.get("top_k", -1)) != 8:
        raise RuntimeError(f"Unexpected semantic manifest: {manifest_path}")
    counts = manifest["materialized"]["splits"]
    total = sum(int(counts[split]["rows"]) for split in SPLITS)
    labels: dict[str, str] = {}
    sample_splits: dict[str, str] = {}
    progress = StageProgress(total, f"[stage 1/3 semantic index:{args.dataset}]")
    for split in SPLITS:
        for row in iter_jsonl(args.semantic_root / args.dataset / f"{split}.jsonl"):
            pair_id = str(row["pair_id"])
            if pair_id in labels:
                raise RuntimeError(f"Duplicate semantic pair_id: {pair_id}")
            labels[pair_id] = str(row["target"])
            sample_id = str(row["sample_id"])
            previous = sample_splits.setdefault(sample_id, split)
            if previous != split:
                raise RuntimeError(f"Question crosses internal splits: {sample_id}")
            progress.update()
    progress.close()
    if len(labels) != total:
        raise RuntimeError(f"Semantic count mismatch: expected={total} actual={len(labels)}")
    return labels, sample_splits, total


def case_name(label: str, transition: str) -> str | None:
    if label == "direct_support":
        if transition == "W2W":
            return "direct_support_w2w"
        if transition == "C2W":
            return "direct_support_c2w"
        if transition in {"C2C", "W2C"}:
            return "direct_support_preserve"
    if label == "no_evidence":
        if transition == "C2W":
            return "no_evidence_c2w"
        if transition == "C2C":
            return "no_evidence_preserve"
        if transition in {"W2C", "W2W"}:
            return "no_evidence_excluded"
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = {str(row["sample_id"]) for row in rows}
    return {
        "questions": len(samples),
        "examples": len(rows),
        "case": dict(Counter(str(row["case"]) for row in rows)),
        "semantic_label": dict(Counter(str(row["semantic_label"]) for row in rows)),
        "transition": dict(Counter(str(row["frozen_transition"]) for row in rows)),
        "no_rag_correct": dict(Counter(str(bool(row["frozen_no_rag_correct"])).lower() for row in rows)),
    }


def select_train_questions(
    reservoirs: dict[str, LowestPriorityRecords],
    natural: LowestPriorityRecords,
    limit: int,
    failure_fraction: float,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return natural.records()
    selected: dict[str, dict[str, Any]] = {}
    failure_target = int(round(limit * failure_fraction))
    per_case = max(1, failure_target // len(PRIMARY_FAILURE_CASES))
    for case in PRIMARY_FAILURE_CASES:
        taken = 0
        for record in reservoirs[case].records():
            sample_id = str(record["sample_id"])
            if sample_id in selected:
                continue
            selected[sample_id] = record
            taken += 1
            if taken >= per_case or len(selected) >= failure_target:
                break
    for record in natural.records():
        selected.setdefault(str(record["sample_id"]), record)
        if len(selected) >= limit:
            break
    return list(selected.values())[:limit]


def flatten(records: list[dict[str, Any]], *, training: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        documents = list(record["documents"])
        if training:
            chosen: dict[str, dict[str, Any]] = {}
            for document in documents:
                case = str(document["case"])
                if case not in TRAIN_CASES:
                    continue
                previous = chosen.get(case)
                if previous is None or int(document["rerank_rank"]) < int(previous["rerank_rank"]):
                    chosen[case] = document
            documents = list(chosen.values())
        question_repeat_weight = 1.0 / max(1, len(documents))
        for document in documents:
            if training and str(document["case"]) not in TRAIN_CASES:
                continue
            rows.append(
                {
                    "run_version": RUN_VERSION,
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "sample_id": record["sample_id"],
                    "row_idx": record["row_idx"],
                    "question": record["question"],
                    "options": record["options"],
                    "gold_answer": record["gold_answer"],
                    "frozen_no_rag_correct": record["frozen_no_rag_correct"],
                    "frozen_no_rag_prediction": record["frozen_no_rag_prediction"],
                    "frozen_no_rag_prompt_sha256": record["frozen_no_rag_prompt_sha256"],
                    "frozen_no_rag_probabilities": record["frozen_no_rag_probabilities"],
                    "question_repeat_weight": question_repeat_weight,
                    **document,
                }
            )
    rows.sort(key=lambda row: (str(row["sample_id"]), int(row["rerank_rank"])))
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not 0.0 <= args.train_failure_fraction <= 1.0:
        raise ValueError("--train-failure-fraction must be in [0,1]")
    candidate_path = args.candidate_root / args.dataset / "train/candidates_top8.jsonl"
    candidate_manifest_path = candidate_path.with_name("candidate_manifest.json")
    semantic_manifest_path = args.semantic_root / args.dataset / "manifest.json"
    outcome_manifest_path = args.outcome_root / "outcome_manifest.json"
    outcome_contract_path = args.outcome_root / "run_contract.json"
    outcome_manifest = json.loads(outcome_manifest_path.read_text(encoding="utf-8"))
    outcome_contract = json.loads(outcome_contract_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    expected_questions = int(outcome_manifest["datasets"][args.dataset])
    if outcome_manifest.get("run_version") != DIRECT_OUTCOME_VERSION:
        raise RuntimeError("Direct-choice outcome version mismatch")
    if outcome_contract.get("prompt_policy_version") != PROMPT_POLICY_VERSION:
        raise RuntimeError("Direct-choice prompt policy mismatch")
    expected_candidate = {
        "candidate_layout": "source_balanced",
        "per_source_top_k": 8,
        "candidate_pool_top_k": 32,
        "top_k": 8,
    }
    mismatches = {
        key: {"expected": value, "actual": candidate_manifest.get(key)}
        for key, value in expected_candidate.items()
        if candidate_manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Candidate retrieval contract mismatch: {mismatches}")
    output_dir = args.output_root / args.dataset
    outputs = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    contract = {
        "run_version": RUN_VERSION,
        "hypothesis": "Semantic-behavior mismatch correction improves Direct-Support rescue without changing no-RAG behavior or increasing No-Evidence destruction.",
        "dataset": args.dataset,
        "candidate_manifest": file_identity(candidate_manifest_path, content_hash=True),
        "retrieval_reranking": expected_candidate,
        "semantic_manifest": file_identity(semantic_manifest_path, content_hash=True),
        "direct_outcome_manifest": file_identity(outcome_manifest_path, content_hash=True),
        "direct_outcome_contract": file_identity(outcome_contract_path, content_hash=True),
        "max_train_questions": args.max_train_questions,
        "max_eval_questions": args.max_eval_questions,
        "train_failure_fraction": args.train_failure_fraction,
        "seed": args.seed,
        "model_input": "question + four options + zero or one document; direct-choice prompt v1",
        "gold_or_semantic_in_prompt": False,
        "train_cases": list(TRAIN_CASES),
        "excluded_train_cases": ["no_evidence_w2c", "no_evidence_w2w", "supporting_evidence", "misleading_evidence"],
        "primary_metric": "held-out Direct-Support accuracy among frozen no-RAG-wrong pairs",
        "minimum_worthwhile_improvement": "+0.02 absolute primary accuracy with <=0.005 no-RAG accuracy loss and no destruction-rate increase",
        "safety_metrics": ["no-RAG accuracy drop", "Direct-Support C2W rate", "No-Evidence C2W rate"],
        "code_sha256": sha256_file(Path(__file__)),
        "code_commit": git_commit(),
    }
    contract_hash = fingerprint(semantic_contract(contract))
    contract_path = output_dir / "run_contract.json"
    completed_path = output_dir / "manifest.json"
    if args.resume and completed_path.is_file() and all(path.is_file() for path in outputs.values()):
        current = json.loads(completed_path.read_text(encoding="utf-8"))
        if prepared_contract_matches(current, contract, contract_hash):
            logging.info("Prepared MVP data are complete and reusable: %s", output_dir)
            return
        raise RuntimeError("Prepared MVP contract mismatch; use a versioned output root")
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if not prepared_contract_matches(previous, contract, contract_hash):
            raise RuntimeError("Incomplete prepared-data contract mismatch; use a new output root")
    atomic_json(contract_path, {**contract, "contract_fingerprint": contract_hash})
    logging.info(
        "Preparation plan: dataset=%s questions=%d train_limit=%d eval_limit=%d",
        args.dataset,
        expected_questions,
        args.max_train_questions,
        args.max_eval_questions,
    )
    if args.plan_only:
        return

    labels, sample_splits, semantic_rows = semantic_index(args)
    limits = {"train": args.max_train_questions, "val": args.max_eval_questions, "test": args.max_eval_questions}
    natural = {
        split: LowestPriorityRecords(
            limits[split] * 4
            if split == "train" and limits[split] > 0
            else limits[split]
        )
        for split in SPLITS
    }
    failure_reservoirs = {
        case: LowestPriorityRecords(max(1, args.max_train_questions))
        for case in PRIMARY_FAILURE_CASES
    }
    outcome_shards = sorted(
        (args.outcome_root / "outcome_shards" / args.dataset / "train").glob("shard_*")
    )
    if not outcome_shards:
        raise FileNotFoundError("No direct-choice outcome shards")
    candidate_iterator = normalized_candidate_rows(candidate_path, args.dataset, "train", 8)
    progress = StageProgress(expected_questions, f"[stage 2/3 join caches:{args.dataset}]")
    joined_questions = 0
    joined_pairs = 0
    missing_semantic = 0
    eligible_counts: Counter[str] = Counter()
    for shard_dir in outcome_shards:
        question_rows = list(iter_jsonl(shard_dir / "questions.jsonl"))
        pair_rows = list(iter_jsonl(shard_dir / "pairs.jsonl"))
        candidate_rows = list(itertools.islice(candidate_iterator, len(question_rows)))
        if len(candidate_rows) != len(question_rows) or len(pair_rows) != len(question_rows) * 8:
            raise RuntimeError(f"Shard cardinality mismatch: {shard_dir}")
        with safe_open(str(shard_dir / "scores.safetensors"), framework="pt", device="cpu") as handle:
            no_probs = handle.get_tensor("no_rag_choice_probabilities")
            doc_probs = handle.get_tensor("single_document_choice_probabilities")
        for q_index, (candidate, question) in enumerate(zip(candidate_rows, question_rows)):
            sample_id = str(candidate["sample_id"])
            if sample_id != str(question["sample_id"]):
                raise RuntimeError(f"Candidate/outcome question mismatch: {sample_id}")
            split = sample_splits.get(sample_id)
            if split is None:
                progress.update()
                continue
            documents: list[dict[str, Any]] = []
            for doc_offset, (candidate_doc, pair) in enumerate(
                zip(candidate["documents"], pair_rows[q_index * 8 : (q_index + 1) * 8])
            ):
                pair_id = str(candidate_doc["pair_id"])
                if pair_id != str(pair["document"]["pair_id"]):
                    raise RuntimeError(f"Candidate/outcome pair mismatch: {pair_id}")
                text = str(candidate_doc["text"]).strip()
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != pair["document"]["document_text_sha256"]:
                    raise RuntimeError(f"Document hash mismatch: {pair_id}")
                label = labels.get(pair_id)
                if label is None:
                    missing_semantic += 1
                    continue
                transition = str(pair["correctness_transition"])
                case = case_name(label, transition)
                if case is None:
                    continue
                tensor_row = int(pair["tensor_row"])
                documents.append(
                    {
                        "pair_id": pair_id,
                        "case": case,
                        "semantic_label": label,
                        "frozen_transition": transition,
                        "frozen_document_correct": bool(pair["answer_correct"]),
                        "frozen_document_prediction": str(pair["prediction"]),
                        "frozen_document_prompt_sha256": str(pair["prompt_sha256"]),
                        "frozen_document_probabilities": [float(value) for value in doc_probs[tensor_row].tolist()],
                        "document_text": text,
                        "document_source": str(pair["document"]["source"]),
                        "document_stable_id": str(pair["document"]["stable_id"]),
                        "rerank_rank": int(pair["document"]["rerank_rank"]),
                    }
                )
                eligible_counts[case] += 1
                joined_pairs += 1
            if documents:
                record = {
                    "dataset": args.dataset,
                    "split": split,
                    "sample_id": sample_id,
                    "row_idx": int(candidate["row_idx"]),
                    "question": str(candidate["question"]),
                    "options": dict(candidate["options"]),
                    "gold_answer": str(candidate["answer"]),
                    "frozen_no_rag_correct": bool(question["answer_correct"]),
                    "frozen_no_rag_prediction": str(question["prediction"]),
                    "frozen_no_rag_prompt_sha256": str(question["prompt_sha256"]),
                    "frozen_no_rag_probabilities": [float(value) for value in no_probs[int(question["tensor_row"])].tolist()],
                    "documents": documents,
                }
                priority = stable_priority(args.seed + SPLITS.index(split), sample_id)
                if split == "train":
                    present = {str(document["case"]) for document in documents}
                    if present.intersection(TRAIN_CASES):
                        natural[split].add(priority, sample_id, record)
                    for case in PRIMARY_FAILURE_CASES:
                        if case in present:
                            failure_reservoirs[case].add(priority, sample_id, record)
                else:
                    natural[split].add(priority, sample_id, record)
                joined_questions += 1
            progress.update()
    progress.close()
    try:
        next(candidate_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Candidate rows remain after all outcome shards")
    if joined_questions <= 0 or joined_pairs <= 0:
        raise RuntimeError("No semantic/direct-choice examples were joined")

    selected_records = {
        "train": select_train_questions(
            failure_reservoirs,
            natural["train"],
            args.max_train_questions,
            args.train_failure_fraction,
        ),
        "val": natural["val"].records(),
        "test": natural["test"].records(),
    }
    selected_question_total = sum(len(values) for values in selected_records.values())
    progress = StageProgress(
        selected_question_total,
        f"[stage 3/3 select/write:{args.dataset}]",
    )
    selected_rows = {
        split: flatten(selected_records[split], training=(split == "train"))
        for split in SPLITS
    }
    for split in SPLITS:
        if not selected_rows[split]:
            raise RuntimeError(f"No selected {split} examples")
        atomic_jsonl(outputs[split], selected_rows[split])
        progress.update(len(selected_records[split]))
    progress.close()
    summary = {split: summarize(rows) for split, rows in selected_rows.items()}
    atomic_json(
        completed_path,
        {
            **contract,
            "contract_fingerprint": contract_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "semantic_rows": semantic_rows,
            "joined_questions": joined_questions,
            "joined_pairs": joined_pairs,
            "missing_semantic_pairs": missing_semantic,
            "eligible_case_counts": dict(eligible_counts),
            "selected": summary,
            "files": {split: file_identity(path, content_hash=True) for split, path in outputs.items()},
        },
    )
    logging.info("Preparation complete: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

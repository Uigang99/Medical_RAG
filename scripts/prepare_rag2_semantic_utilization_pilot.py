#!/usr/bin/env python3
"""Build question-level semantic-utilization contrast sets for a LoRA pilot.

The target is deliberately semantic-only.  Gold-margin changes and every
other behavioral-utility measurement are excluded.  Each example contains:

* a valid context: Direct Support + Supporting Evidence documents;
* the original labelled context: valid documents plus No/Misleading noise;
* an invalid-only context: No Evidence + Misleading Evidence documents;
* a valid context with one Direct Support document removed; and
* a correct, fixed-format response produced from that Direct Support document.

Semantic labels construct the interventions but are never rendered into the
model prompt.  Outputs are immutable, question-disjoint train/val/test JSONL
files with a versioned manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_semantic_utilization_contrast_sets_v1"
SPLITS = ("train", "val", "test")
VALID_LABELS = {"direct_support", "supporting_evidence"}
INVALID_LABELS = {"no_evidence", "misleading_evidence"}
BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medqa", "medmcqa"), default="medqa")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=BASE / "candidates/source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=BASE / "filter_training_inputs_semantic_top8_four_class_v1",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=BASE / "document_traces_source_balanced32_rerank8_v1/trace_shards",
    )
    parser.add_argument(
        "--no-rag-root",
        type=Path,
        default=BASE / "train_no_rag_anchored_features_v1/no_rag",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=BASE / "filter_training_inputs_rag2_paper_reproduction_three_class_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE / "semantic_utilization_contrast_pilot_v1",
    )
    parser.add_argument("--max-train-questions", type=int, default=512)
    parser.add_argument("--max-eval-questions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StageStatus:
    """A stable stage-local progress display with rolling rate and ETA."""

    def __init__(self, stage: int, stages: int, name: str, total: int) -> None:
        self.stage = stage
        self.stages = stages
        self.name = name
        self.total = max(0, int(total))
        self.done = 0
        self.started = time.time()
        self.last_render = 0.0
        self.render(force=True)

    def update(self, value: int = 1) -> None:
        self.done += int(value)
        self.render(force=self.done >= self.total)

    def render(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render < 1.0:
            return
        self.last_render = now
        elapsed = max(1e-9, now - self.started)
        rate = self.done / elapsed
        remaining = max(0, self.total - self.done)
        eta = remaining / rate if rate > 0 else None
        percent = 100.0 * self.done / self.total if self.total else 100.0
        eta_text = "unknown" if eta is None else format_seconds(eta)
        print(
            f"\r[overall {self.stage}/{self.stages}] [{self.name} | "
            f"{self.done}/{self.total} {percent:5.1f}% | {rate:,.1f}/s | "
            f"elapsed {format_seconds(elapsed)} | ETA {eta_text}]",
            end="\n" if force and self.done >= self.total else "",
            flush=True,
        )


def format_seconds(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_split_ids(root: Path, dataset: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    all_ids: set[str] = set()
    for split in SPLITS:
        path = root / dataset / "sample_ids" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        overlap = all_ids & ids
        if overlap:
            raise RuntimeError(f"Question leakage across splits: {next(iter(overlap))}")
        all_ids.update(ids)
        values[split] = ids
    return values


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("chunk_id")
    if not value:
        raise ValueError("Candidate document has no stable ID")
    return str(value)


def probabilities_from_logprobs(values: dict[str, Any]) -> tuple[list[float], int]:
    finite = [float(values[choice]) for choice in CHOICES if values.get(choice) is not None]
    finite = [value for value in finite if math.isfinite(value)]
    if not finite:
        raise ValueError("No finite No-RAG choice log-probabilities")
    # vLLM only retained a bounded top-logprob set. A missing option is therefore
    # below every retained option; assigning min-5 keeps its mass negligible
    # without excluding highly confident questions and biasing the cohort.
    floor = min(finite) - 5.0
    raw = []
    imputed = 0
    for choice in CHOICES:
        value = values.get(choice)
        if value is None or not math.isfinite(float(value)):
            raw.append(floor)
            imputed += 1
        else:
            raw.append(float(value))
    logits = np.asarray(raw, dtype=np.float64)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32).tolist(), imputed


def deterministic_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row['sample_id']}".encode("utf-8")).hexdigest()

    return sorted(rows, key=key)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.max_train_questions <= 0 or args.max_eval_questions <= 0:
        raise ValueError("Pilot split limits must be positive")

    semantic_dir = args.semantic_root / args.dataset
    candidate_path = args.candidate_root / args.dataset / "train/candidates_top8.jsonl"
    no_rag_path = args.no_rag_root / args.dataset / "train/no_rag_generations.jsonl"
    trace_dir = args.trace_root / args.dataset / "train"
    semantic_paths = {split: semantic_dir / f"{split}.jsonl" for split in SPLITS}
    required = [candidate_path, no_rag_path, *semantic_paths.values(), trace_dir]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_root / args.dataset
    outputs = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "candidate": identity(candidate_path),
        "semantic": {split: identity(path) for split, path in semantic_paths.items()},
        "no_rag": identity(no_rag_path),
        "split_root": str((args.split_root / args.dataset / "sample_ids").resolve()),
        "trace_manifest": identity(args.trace_root.parent / "generation_manifest.json"),
        "max_train_questions": args.max_train_questions,
        "max_eval_questions": args.max_eval_questions,
        "seed": args.seed,
        "semantic_policy": {
            "valid": sorted(VALID_LABELS),
            "invalid": sorted(INVALID_LABELS),
            "reference": "highest-reranked direct_support trace with correct fixed-format answer",
            "behavioral_utility_used": False,
        },
    }
    contract_hash = fingerprint(contract)
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.is_file() and all(path.is_file() for path in outputs.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract_fingerprint") == contract_hash:
            logging.info("Semantic-utilization pilot data are complete and reusable: %s", output_dir)
            return
        raise RuntimeError("Prepared-data contract mismatch; use a new output root")

    split_ids = load_split_ids(args.split_root, args.dataset)
    semantic_total = sum(sum(1 for _ in iter_jsonl(path)) for path in semantic_paths.values())
    candidate_total = sum(1 for _ in iter_jsonl(candidate_path))
    no_rag_total = sum(1 for _ in iter_jsonl(no_rag_path))
    trace_paths = sorted(trace_dir.glob("shard_*/pairs.jsonl"))
    if not trace_paths:
        raise FileNotFoundError(f"No document traces under {trace_dir}")
    trace_total = sum(sum(1 for _ in iter_jsonl(path)) for path in trace_paths)
    logging.info(
        "Preparation plan: dataset=%s stages=5 semantic=%d candidates=%d no_rag=%d traces=%d",
        args.dataset,
        semantic_total,
        candidate_total,
        no_rag_total,
        trace_total,
    )
    if args.plan_only:
        return

    labels: dict[tuple[str, str], str] = {}
    status = StageStatus(1, 5, f"semantic labels {args.dataset}", semantic_total)
    for split, path in semantic_paths.items():
        for row in iter_jsonl(path):
            key = (str(row["sample_id"]), str(row["doc_stable_id"]))
            label = str(row["target"])
            if label not in VALID_LABELS | INVALID_LABELS:
                raise ValueError(f"Unexpected four-class semantic label: {label}")
            if key in labels:
                raise RuntimeError(f"Duplicate semantic document label: {key}")
            labels[key] = label
            status.update()

    questions: dict[str, dict[str, Any]] = {}
    needed_direct: set[tuple[str, str]] = set()
    status = StageStatus(2, 5, f"Top-8 candidate join {args.dataset}", candidate_total)
    for row in iter_jsonl(candidate_path):
        sample_id = str(row["sample_id"])
        split = next((name for name, ids in split_ids.items() if sample_id in ids), None)
        if split is None:
            status.update()
            continue
        documents = []
        for rank, document in enumerate(row.get("candidate_documents") or [], 1):
            doc_id = stable_id(document)
            label = labels.get((sample_id, doc_id))
            if label is None:
                continue
            normalized = {
                "stable_id": doc_id,
                "rank": int(document.get("rerank_rank") or rank),
                "source": str(document.get("source") or ""),
                "text": " ".join(str(document.get("text") or "").split()),
                "semantic_label": label,
            }
            if not normalized["text"]:
                raise ValueError(f"Empty document text: {sample_id}/{doc_id}")
            documents.append(normalized)
            if label == "direct_support":
                needed_direct.add((sample_id, doc_id))
        documents.sort(key=lambda value: value["rank"])
        valid = [doc for doc in documents if doc["semantic_label"] in VALID_LABELS]
        invalid = [doc for doc in documents if doc["semantic_label"] in INVALID_LABELS]
        if any(doc["semantic_label"] == "direct_support" for doc in valid) and invalid:
            questions[sample_id] = {
                "dataset": args.dataset,
                "split": split,
                "sample_id": sample_id,
                "row_idx": int(row["row_idx"]),
                "question": str(row["question"]),
                "options": {str(k): str(v) for k, v in row["options"].items()},
                "gold_answer": str(row["answer"]).upper(),
                "documents": documents,
            }
        status.update()

    no_rag: dict[str, dict[str, Any]] = {}
    no_rag_imputed_rows = 0
    status = StageStatus(3, 5, f"no-RAG preservation targets {args.dataset}", no_rag_total)
    for row in iter_jsonl(no_rag_path):
        sample_id = str(row["sample_id"])
        if sample_id in questions:
            rationale = str(row.get("model_raw_rationale") or "").strip()
            logprobs = row.get("choice_logprobs") or {}
            if rationale and all(choice in logprobs for choice in CHOICES):
                probabilities, imputed = probabilities_from_logprobs(logprobs)
                no_rag_imputed_rows += int(imputed > 0)
                no_rag[sample_id] = {
                    "rationale": rationale,
                    "choice_probabilities": probabilities,
                    "imputed_choice_logprob_count": imputed,
                    "answer": str(row.get("answer") or "").upper(),
                    "answer_correct": bool(row.get("answer_correct")),
                }
        status.update()

    direct_traces: dict[tuple[str, str], dict[str, Any]] = {}
    status = StageStatus(4, 5, f"correct Direct-Support references {args.dataset}", trace_total)
    for path in trace_paths:
        for row in iter_jsonl(path):
            sample_id = str(row["sample_id"])
            document = row.get("document") or {}
            doc_id = str(document.get("stable_id") or document.get("corpus_id") or "")
            key = (sample_id, doc_id)
            if (
                key in needed_direct
                and bool(row.get("answer_correct"))
                and not (row.get("quality_flags") or [])
                and str(row.get("rationale") or "").strip()
                and str(row.get("canonical_response") or "").strip()
            ):
                direct_traces[key] = {
                    "rationale": str(row["rationale"]).strip(),
                    "canonical_response": str(row["canonical_response"]).strip(),
                    "answer": str(row["answer"]).upper(),
                }
            status.update()

    eligible: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for sample_id, question in questions.items():
        if sample_id not in no_rag:
            continue
        direct = [
            doc
            for doc in question["documents"]
            if doc["semantic_label"] == "direct_support"
            and (sample_id, doc["stable_id"]) in direct_traces
        ]
        if not direct:
            continue
        anchor = min(direct, key=lambda doc: (doc["rank"], doc["stable_id"]))
        reference = direct_traces[(sample_id, anchor["stable_id"])]
        if reference["answer"] != question["gold_answer"]:
            raise RuntimeError(f"Correct reference/gold mismatch: {sample_id}")
        valid = [doc for doc in question["documents"] if doc["semantic_label"] in VALID_LABELS]
        invalid = [doc for doc in question["documents"] if doc["semantic_label"] in INVALID_LABELS]
        ablated = [doc for doc in valid if doc["stable_id"] != anchor["stable_id"]]
        eligible[question["split"]].append(
            {
                "run_version": RUN_VERSION,
                **question,
                "reference": {
                    **reference,
                    "source_document_id": anchor["stable_id"],
                    "selection_rule": "highest-reranked correct direct_support trace",
                },
                "valid_documents": valid,
                "full_documents": question["documents"],
                "invalid_documents": invalid,
                "direct_ablated_documents": ablated,
                "no_rag": no_rag[sample_id],
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        limit = args.max_train_questions if split == "train" else args.max_eval_questions
        selected[split] = deterministic_order(eligible[split], args.seed)[:limit]
        if not selected[split]:
            raise RuntimeError(f"No eligible semantic-utilization questions for {split}")
    selected_ids = {split: {row["sample_id"] for row in rows} for split, rows in selected.items()}
    if any(selected_ids[left] & selected_ids[right] for left in SPLITS for right in SPLITS if left < right):
        raise RuntimeError("Prepared question leakage across splits")

    status = StageStatus(5, 5, f"atomic split materialization {args.dataset}", sum(map(len, selected.values())))
    split_summary = {}
    for split in SPLITS:
        rows = selected[split]
        atomic_jsonl(outputs[split], rows)
        for _ in rows:
            status.update()
        split_summary[split] = {
            "questions": len(rows),
            "available_before_limit": len(eligible[split]),
            "mean_valid_documents": float(np.mean([len(row["valid_documents"]) for row in rows])),
            "mean_invalid_documents": float(np.mean([len(row["invalid_documents"]) for row in rows])),
            "semantic_labels": dict(
                Counter(doc["semantic_label"] for row in rows for doc in row["full_documents"])
            ),
            "no_rag_correct": dict(Counter(str(row["no_rag"]["answer_correct"]).lower() for row in rows)),
            "no_rag_rows_with_imputed_low_probability_choices": sum(
                int(row["no_rag"]["imputed_choice_logprob_count"] > 0) for row in rows
            ),
        }

    manifest = {
        "run_version": RUN_VERSION,
        "contract_fingerprint": contract_hash,
        "contract": contract,
        "splits": split_summary,
        "reference_quality_filter": {
            "semantic_label": "direct_support",
            "answer_must_be_correct": True,
            "quality_flags_must_be_empty": True,
            "gold_answer_used_as_model_input": False,
            "behavioral_utility_used": False,
        },
        "no_rag_choice_logprob_policy": {
            "rows_with_imputation_before_split_selection": no_rag_imputed_rows,
            "missing_choice_value": "minimum finite retained log-probability minus 5.0",
            "reason": "vLLM top-logprob truncation; missing choices are below retained choices",
        },
        "outputs": {
            split: {**identity(path), "sha256": sha256_file(path)}
            for split, path in outputs.items()
        },
    }
    atomic_json(manifest_path, manifest)
    logging.info("Semantic-utilization pilot data complete: %s", output_dir)
    logging.info("Split summary: %s", json.dumps(split_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

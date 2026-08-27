#!/usr/bin/env python3
"""Materialize exact choice-margin utility for anchored RAG2 training pairs.

The expensive model work has already been completed by the anchored no-RAG
and one-document replay pipelines.  Both caches retain the exact A/B/C/D
logits at ``pre_choice``.  This script joins those cached values and computes,
for every question-document pair:

* the gold-vs-strongest-wrong logit margin before and after the document;
* the exact margin change ``m_D - m_0``;
* the bounded two-way boundary probability ``sigmoid(m / T)`` and its change;
* the four-choice gold probability and its change; and
* the observed C->C, C->W, W->C, or W->W transition.

No generation, retrieval, reranking, hidden-state extraction, or GPU forward
pass is performed.  Outputs are sharded, atomic, resumable, and accompanied by
an aggregate audit.  Progress includes the active stage and ETA.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_anchored_gold_margin_scores_v1"
NO_RAG_RUN_VERSION = "rag2_anchored_no_rag_selected_layer_features_v1"
DOCUMENT_RUN_VERSION = "rag2_anchored_independent_document_selected_layer_features_v1"
TRACE_VERSION = "rag2_paper_compatible_three_anchor_v1"
PROMPT_VERSION = "rag2_paper_compatible_three_anchor_prompt_v1"
CHOICES = ("A", "B", "C", "D")


try:
    import msgspec

    _DECODER = msgspec.json.Decoder()

    def decode_json(line: bytes) -> dict[str, Any]:
        return _DECODER.decode(line)

except ImportError:

    def decode_json(line: bytes) -> dict[str, Any]:
        return json.loads(line)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-rag-feature-root",
        type=Path,
        default=base / "train_no_rag_anchored_features_v1",
    )
    parser.add_argument(
        "--document-feature-root",
        type=Path,
        default=base / "document_traces_source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=base / "candidates/source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base / "gold_margin_utility_source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(
            "medmcqa",
            "medqa",
            "mmlu_anatomy",
            "mmlu_clinical_knowledge",
            "mmlu_college_biology",
            "mmlu_college_medicine",
            "mmlu_medical_genetics",
            "mmlu_professional_medicine",
        ),
        default=["medmcqa", "medqa"],
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--candidate-contract",
        choices=("source_balanced_top8", "dynamic_topk_union"),
        default="source_balanced_top8",
        help=(
            "Validate either the training 4x8 -> rerank Top-8 candidates or the external-test "
            "union of independently reconstructed 4k -> rerank Top-k conditions."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sigmoid(margin / T); raw margins are never rescaled.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield decode_json(line)
            except Exception as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        str(temporary),
        metadata=metadata,
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class QuestionScore:
    dataset: str
    sample_id: str
    row_idx: int
    gold_answer: str
    gold_index: int
    generated_answer: str
    generated_answer_correct: bool
    logits: torch.Tensor


@dataclass(frozen=True)
class FeatureShard:
    dataset: str
    name: str
    root: Path
    pair_count: int
    tensor_size: int
    tensor_mtime_ns: int


def require_manifest_contract(manifest: dict[str, Any], expected_run_version: str, label: str) -> None:
    if manifest.get("run_version") != expected_run_version:
        raise RuntimeError(
            f"{label} run version mismatch: {manifest.get('run_version')!r} != {expected_run_version!r}"
        )
    if manifest.get("trace_version") != TRACE_VERSION:
        raise RuntimeError(f"{label} trace version mismatch: {manifest.get('trace_version')!r}")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise RuntimeError(f"{label} prompt version mismatch: {manifest.get('prompt_version')!r}")
    if list(manifest.get("choice_token_ids") or {}) != list(CHOICES):
        raise RuntimeError(f"{label} choice ordering is not A/B/C/D")


def validate_candidate_contract(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    manifest_path = args.candidate_root / dataset / args.source_split / "candidate_manifest.json"
    manifest = read_json(manifest_path)
    expected_sources = ["pubmed", "pmc", "cpg", "textbooks"]
    actual_source_top_k = (manifest.get("candidate_pool_policy") or {}).get("source_top_k") or {}
    problems: list[str] = []
    if manifest.get("dataset") != dataset or manifest.get("split") != args.source_split:
        problems.append("dataset/split mismatch")
    if manifest.get("sources") != expected_sources:
        problems.append(f"sources={manifest.get('sources')!r}")
    if manifest.get("candidate_layout") != "source_balanced":
        problems.append(f"candidate_layout={manifest.get('candidate_layout')!r}")
    if args.candidate_contract == "source_balanced_top8":
        if int(manifest.get("per_source_top_k", -1)) != 8:
            problems.append(f"per_source_top_k={manifest.get('per_source_top_k')!r}")
        if int(manifest.get("candidate_pool_top_k", -1)) != 32:
            problems.append(f"candidate_pool_top_k={manifest.get('candidate_pool_top_k')!r}")
        if int(manifest.get("top_k", -1)) != 8:
            problems.append(f"rerank top_k={manifest.get('top_k')!r}")
        if actual_source_top_k != {source: 8 for source in expected_sources}:
            problems.append(f"source_top_k={actual_source_top_k!r}")
    else:
        if manifest.get("candidate_protocol") != "rag2_paper_balanced_dynamic_topk_union_v1":
            problems.append(f"candidate_protocol={manifest.get('candidate_protocol')!r}")
        if int(manifest.get("per_source_top_k", -1)) != 32:
            problems.append(f"per_source_top_k={manifest.get('per_source_top_k')!r}")
        if int(manifest.get("candidate_pool_top_k", -1)) != 128:
            problems.append(f"candidate_pool_top_k={manifest.get('candidate_pool_top_k')!r}")
        if not bool(manifest.get("variable_docs_per_question")):
            problems.append("variable_docs_per_question is false")
        if list(manifest.get("dynamic_top_k_values") or []) != [1, 2, 4, 8, 16, 32]:
            problems.append(f"dynamic_top_k_values={manifest.get('dynamic_top_k_values')!r}")
        if int(manifest.get("selected_pair_count", -1)) <= 0:
            problems.append(f"selected_pair_count={manifest.get('selected_pair_count')!r}")
    if problems:
        raise RuntimeError(
            f"Candidate contract mismatch ({args.candidate_contract}) for {dataset}: "
            + "; ".join(problems)
        )
    return manifest


def output_paths(output_root: Path, dataset: str, source_split: str, shard_name: str) -> dict[str, Path]:
    root = output_root / "score_shards" / dataset / source_split / shard_name
    return {
        "root": root,
        "rows": root / "rows.jsonl",
        "scores": root / "scores.safetensors",
        "complete": root / "COMPLETE.json",
    }


def output_complete(paths: dict[str, Path], source: FeatureShard, temperature: float) -> bool:
    if any(not paths[name].is_file() for name in ("rows", "scores", "complete")):
        return False
    try:
        marker = read_json(paths["complete"])
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and int(marker.get("pair_count", -1)) == source.pair_count
        and int(marker.get("source_tensor_size", -1)) == source.tensor_size
        and int(marker.get("source_tensor_mtime_ns", -1)) == source.tensor_mtime_ns
        and math.isclose(float(marker.get("temperature", float("nan"))), temperature, abs_tol=1e-12)
    )


def discover_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], list[FeatureShard], int, int]:
    no_manifest = read_json(args.no_rag_feature_root / "feature_manifest.json")
    doc_manifest = read_json(args.document_feature_root / "document_feature_manifest.json")
    require_manifest_contract(no_manifest, NO_RAG_RUN_VERSION, "no-RAG feature")
    require_manifest_contract(doc_manifest, DOCUMENT_RUN_VERSION, "document feature")
    if no_manifest.get("model_name_or_path") != doc_manifest.get("model_name_or_path"):
        raise RuntimeError("No-RAG and document logits were extracted with different models")
    if no_manifest.get("choice_token_ids") != doc_manifest.get("choice_token_ids"):
        raise RuntimeError("No-RAG and document choice token IDs differ")

    candidate_manifests: dict[str, dict[str, Any]] = {}
    feature_shards: list[FeatureShard] = []
    total_questions = 0
    total_pairs = 0
    for dataset in args.datasets:
        candidate = validate_candidate_contract(args, dataset)
        candidate_manifests[dataset] = candidate
        total_questions += int((no_manifest.get("datasets") or {}).get(dataset, 0))
        expected_pairs = int((doc_manifest.get("datasets") or {}).get(dataset, -1))
        selected = int(candidate.get("selected_question_count", -1))
        expected_from_candidate = (
            selected * 8
            if args.candidate_contract == "source_balanced_top8"
            else int(candidate.get("selected_pair_count", -1))
        )
        if expected_pairs != expected_from_candidate:
            raise RuntimeError(
                f"Candidate/feature pair count mismatch for {dataset}: "
                f"document_manifest={expected_pairs}, candidate={expected_from_candidate}"
            )
        roots = sorted(
            (
                args.document_feature_root
                / "with_document_features"
                / dataset
                / args.source_split
                / "shards"
            ).glob("shard_*")
        )
        observed = 0
        for root in roots:
            complete = read_json(root / "COMPLETE.json")
            count = int(complete.get("pair_count", -1))
            tensor_path = root / "features.safetensors"
            meta_path = root / "pairs.jsonl"
            if count <= 0 or not tensor_path.is_file() or not meta_path.is_file():
                raise RuntimeError(f"Incomplete document feature shard: {root}")
            stat = tensor_path.stat()
            feature_shards.append(
                FeatureShard(
                    dataset=dataset,
                    name=root.name,
                    root=root,
                    pair_count=count,
                    tensor_size=stat.st_size,
                    tensor_mtime_ns=stat.st_mtime_ns,
                )
            )
            observed += count
        if observed != expected_pairs:
            raise RuntimeError(f"Document shard coverage mismatch for {dataset}: {observed} != {expected_pairs}")
        total_pairs += observed
    return no_manifest, doc_manifest, candidate_manifests, feature_shards, total_questions, total_pairs


def build_question_index(
    args: argparse.Namespace,
    progress: PipelineProgress,
) -> dict[str, QuestionScore]:
    result: dict[str, QuestionScore] = {}
    for dataset in args.datasets:
        roots = sorted(
            (
                args.no_rag_feature_root
                / "no_rag_features"
                / dataset
                / args.source_split
                / "shards"
            ).glob("shard_*")
        )
        for root in roots:
            metadata = list(iter_jsonl(root / "questions.jsonl"))
            with safe_open(root / "features.safetensors", framework="pt", device="cpu") as handle:
                tensor_metadata = handle.metadata() or {}
                if json.loads(tensor_metadata.get("choice_order", "[]")) != list(CHOICES):
                    raise RuntimeError(f"Choice tensor ordering mismatch: {root}")
                logits = handle.get_tensor("choice_logits").float()
            if logits.shape != (len(metadata), 4):
                raise RuntimeError(f"No-RAG logit shape mismatch: {root} has {tuple(logits.shape)}")
            for local, row in enumerate(metadata):
                if int(row.get("tensor_row", -1)) != local:
                    raise RuntimeError(f"No-RAG tensor row mismatch: {root}:{local}")
                sample_id = str(row.get("sample_id") or "")
                gold = str(row.get("gold_answer") or "").upper()
                if not sample_id or gold not in CHOICES:
                    raise RuntimeError(f"Invalid no-RAG metadata: {root}:{local}")
                if sample_id in result:
                    raise RuntimeError(f"Duplicate no-RAG sample_id: {sample_id}")
                result[sample_id] = QuestionScore(
                    dataset=dataset,
                    sample_id=sample_id,
                    row_idx=int(row["row_idx"]),
                    gold_answer=gold,
                    gold_index=CHOICES.index(gold),
                    generated_answer=str(row.get("generated_answer") or ""),
                    generated_answer_correct=bool(row.get("generated_answer_correct")),
                    logits=logits[local].clone(),
                )
            progress.update(len(metadata))
            progress.set_detail(f"dataset={dataset} shard={root.name}")
    return result


def batch_choice_scores(
    no_document_logits: torch.Tensor,
    with_document_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Compute exact four-choice decision quantities for one aligned batch."""
    if no_document_logits.shape != with_document_logits.shape:
        raise ValueError("No-RAG and document logit shapes differ")
    if no_document_logits.ndim != 2 or no_document_logits.shape[1] != 4:
        raise ValueError(f"Expected [N,4] logits, got {tuple(no_document_logits.shape)}")
    if gold_indices.shape != (no_document_logits.shape[0],):
        raise ValueError("Gold-index shape mismatch")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive")

    rows = torch.arange(no_document_logits.shape[0], dtype=torch.long)
    wrong_mask = F.one_hot(gold_indices, num_classes=4).bool()
    no_wrong_logits = no_document_logits.masked_fill(wrong_mask, float("-inf"))
    doc_wrong_logits = with_document_logits.masked_fill(wrong_mask, float("-inf"))
    no_best_wrong_logits, no_best_wrong = torch.max(no_wrong_logits, dim=-1)
    doc_best_wrong_logits, doc_best_wrong = torch.max(doc_wrong_logits, dim=-1)
    no_gold_logits = no_document_logits[rows, gold_indices]
    doc_gold_logits = with_document_logits[rows, gold_indices]
    no_margin = no_gold_logits - no_best_wrong_logits
    doc_margin = doc_gold_logits - doc_best_wrong_logits
    no_probabilities = F.softmax(no_document_logits, dim=-1)
    doc_probabilities = F.softmax(with_document_logits, dim=-1)
    no_gold_probability = no_probabilities[rows, gold_indices]
    doc_gold_probability = doc_probabilities[rows, gold_indices]
    no_boundary_probability = torch.sigmoid(no_margin / temperature)
    doc_boundary_probability = torch.sigmoid(doc_margin / temperature)
    no_prediction = torch.argmax(no_document_logits, dim=-1)
    doc_prediction = torch.argmax(with_document_logits, dim=-1)
    return {
        "no_document_choice_logits": no_document_logits,
        "with_document_choice_logits": with_document_logits,
        "no_document_choice_probabilities": no_probabilities,
        "with_document_choice_probabilities": doc_probabilities,
        "gold_index": gold_indices,
        "no_document_prediction_index": no_prediction,
        "with_document_prediction_index": doc_prediction,
        "no_document_best_wrong_index": no_best_wrong,
        "with_document_best_wrong_index": doc_best_wrong,
        "no_document_gold_margin": no_margin,
        "with_document_gold_margin": doc_margin,
        "gold_margin_delta": doc_margin - no_margin,
        "no_document_boundary_probability": no_boundary_probability,
        "with_document_boundary_probability": doc_boundary_probability,
        "boundary_probability_delta": doc_boundary_probability - no_boundary_probability,
        "no_document_gold_choice_probability": no_gold_probability,
        "with_document_gold_choice_probability": doc_gold_probability,
        "gold_choice_probability_delta": doc_gold_probability - no_gold_probability,
    }


def process_feature_shard(
    args: argparse.Namespace,
    source: FeatureShard,
    question_index: dict[str, QuestionScore],
    paths: dict[str, Path],
    progress: PipelineProgress,
) -> None:
    pair_rows = list(iter_jsonl(source.root / "pairs.jsonl"))
    if len(pair_rows) != source.pair_count:
        raise RuntimeError(
            f"Pair metadata count mismatch in {source.root}: {len(pair_rows)} != {source.pair_count}"
        )
    with safe_open(source.root / "features.safetensors", framework="pt", device="cpu") as handle:
        tensor_metadata = handle.metadata() or {}
        if json.loads(tensor_metadata.get("choice_order", "[]")) != list(CHOICES):
            raise RuntimeError(f"Document choice tensor ordering mismatch: {source.root}")
        document_logits = handle.get_tensor("choice_logits").float()
    if document_logits.shape != (source.pair_count, 4):
        raise RuntimeError(f"Document logit shape mismatch in {source.root}: {tuple(document_logits.shape)}")

    refs: list[QuestionScore] = []
    for local, row in enumerate(pair_rows):
        if int(row.get("tensor_row", -1)) != local:
            raise RuntimeError(f"Document tensor row mismatch: {source.root}:{local}")
        ref = question_index.get(str(row.get("sample_id") or ""))
        if ref is None:
            raise RuntimeError(f"Missing no-RAG logits for {row.get('sample_id')} in {source.root}")
        if ref.dataset != source.dataset or ref.gold_answer != str(row.get("gold_answer") or "").upper():
            raise RuntimeError(f"Question/document metadata mismatch: {source.root}:{local}")
        refs.append(ref)

    no_document_logits = torch.stack([ref.logits for ref in refs]).float()
    gold_indices = torch.tensor([ref.gold_index for ref in refs], dtype=torch.long)
    scores = batch_choice_scores(no_document_logits, document_logits, gold_indices, args.temperature)

    output_rows: list[dict[str, Any]] = []
    for local, (row, ref) in enumerate(zip(pair_rows, refs)):
        no_prediction = int(scores["no_document_prediction_index"][local].item())
        doc_prediction = int(scores["with_document_prediction_index"][local].item())
        no_correct = no_prediction == ref.gold_index
        doc_correct = doc_prediction == ref.gold_index
        transition = ("C" if no_correct else "W") + "->" + ("C" if doc_correct else "W")
        document_generated_answer = str(row.get("generated_answer") or "")
        document_generated_correct = bool(row.get("generated_answer_correct"))
        generated_transition = ("C" if ref.generated_answer_correct else "W") + "->" + (
            "C" if document_generated_correct else "W"
        )
        no_margin = float(scores["no_document_gold_margin"][local].item())
        doc_margin = float(scores["with_document_gold_margin"][local].item())
        output_rows.append(
            {
                "run_version": RUN_VERSION,
                "dataset": source.dataset,
                "source_split": args.source_split,
                "sample_id": ref.sample_id,
                "row_idx": ref.row_idx,
                "pair_id": row["pair_id"],
                "doc_rank": int(row.get("doc_rank") or 0),
                "document_source": str(row.get("document_source") or "unknown"),
                "document_stable_id": str(row.get("document_stable_id") or ""),
                "gold_answer": ref.gold_answer,
                "no_document_generated_answer": ref.generated_answer,
                "with_document_generated_answer": document_generated_answer,
                "no_document_generated_correct": ref.generated_answer_correct,
                "with_document_generated_correct": document_generated_correct,
                "generated_answer_transition": generated_transition,
                "no_document_prediction": CHOICES[no_prediction],
                "with_document_prediction": CHOICES[doc_prediction],
                "no_document_best_wrong": CHOICES[
                    int(scores["no_document_best_wrong_index"][local].item())
                ],
                "with_document_best_wrong": CHOICES[
                    int(scores["with_document_best_wrong_index"][local].item())
                ],
                "no_document_correct": no_correct,
                "with_document_correct": doc_correct,
                "answer_transition": transition,
                "answer_transition_basis": "exact_hf_replay_four_choice_logits",
                "no_document_generated_replay_match": ref.generated_answer == CHOICES[no_prediction],
                "with_document_generated_replay_match": document_generated_answer == CHOICES[doc_prediction],
                "positive_boundary_crossing": transition == "W->C",
                "negative_boundary_crossing": transition == "C->W",
                "no_document_gold_margin": no_margin,
                "with_document_gold_margin": doc_margin,
                "gold_margin_delta": float(scores["gold_margin_delta"][local].item()),
                "distance_to_positive_boundary": max(0.0, -no_margin),
                "with_document_boundary_surplus": doc_margin,
                "no_document_boundary_probability": float(
                    scores["no_document_boundary_probability"][local].item()
                ),
                "with_document_boundary_probability": float(
                    scores["with_document_boundary_probability"][local].item()
                ),
                "boundary_probability_delta": float(scores["boundary_probability_delta"][local].item()),
                "no_document_gold_choice_probability": float(
                    scores["no_document_gold_choice_probability"][local].item()
                ),
                "with_document_gold_choice_probability": float(
                    scores["with_document_gold_choice_probability"][local].item()
                ),
                "gold_choice_probability_delta": float(
                    scores["gold_choice_probability_delta"][local].item()
                ),
                "rationale_ppl": row.get("rationale_ppl"),
                "quality_flags": row.get("quality_flags") or [],
                "tensor_row": local,
            }
        )

    atomic_jsonl(paths["rows"], output_rows)
    atomic_safetensors(
        paths["scores"],
        scores,
        {
            "run_version": RUN_VERSION,
            "dataset": source.dataset,
            "source_split": args.source_split,
            "choice_order": json.dumps(list(CHOICES)),
            "temperature": str(args.temperature),
            "score_definition": "margin=gold_logit-max_wrong_logit; utility=sigmoid(mD/T)-sigmoid(m0/T)",
        },
    )
    atomic_json(
        paths["complete"],
        {
            "run_version": RUN_VERSION,
            "completed_at": utc_now(),
            "dataset": source.dataset,
            "source_split": args.source_split,
            "shard": source.name,
            "pair_count": source.pair_count,
            "temperature": args.temperature,
            "source_tensor_size": source.tensor_size,
            "source_tensor_mtime_ns": source.tensor_mtime_ns,
            "transition_counts": dict(Counter(row["answer_transition"] for row in output_rows)),
        },
    )
    progress.update(source.pair_count)
    progress.set_detail(f"dataset={source.dataset} shard={source.name}")


def update_group_summary(group: dict[str, Any], row: dict[str, Any]) -> None:
    group["pairs"] += 1
    group["margin_delta_sum"] += float(row["gold_margin_delta"])
    group["utility_sum"] += float(row["boundary_probability_delta"])
    group["gold_probability_delta_sum"] += float(row["gold_choice_probability_delta"])
    group["positive_margin_delta"] += int(float(row["gold_margin_delta"]) > 0)
    group["negative_margin_delta"] += int(float(row["gold_margin_delta"]) < 0)
    group["positive_utility"] += int(float(row["boundary_probability_delta"]) > 0)
    group["negative_utility"] += int(float(row["boundary_probability_delta"]) < 0)
    group["transition_counts"][str(row["answer_transition"])] += 1


def finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    count = int(group["pairs"])
    return {
        "pairs": count,
        "mean_gold_margin_delta": group["margin_delta_sum"] / count if count else None,
        "mean_boundary_probability_delta": group["utility_sum"] / count if count else None,
        "mean_gold_choice_probability_delta": group["gold_probability_delta_sum"] / count if count else None,
        "positive_margin_delta_rate": group["positive_margin_delta"] / count if count else None,
        "negative_margin_delta_rate": group["negative_margin_delta"] / count if count else None,
        "positive_utility_rate": group["positive_utility"] / count if count else None,
        "negative_utility_rate": group["negative_utility"] / count if count else None,
        "transition_counts": dict(group["transition_counts"]),
        "transition_rates": {
            key: value / count if count else None
            for key, value in sorted(group["transition_counts"].items())
        },
    }


def empty_group() -> dict[str, Any]:
    return {
        "pairs": 0,
        "margin_delta_sum": 0.0,
        "utility_sum": 0.0,
        "gold_probability_delta_sum": 0.0,
        "positive_margin_delta": 0,
        "negative_margin_delta": 0,
        "positive_utility": 0,
        "negative_utility": 0,
        "transition_counts": Counter(),
    }


def build_summary(
    args: argparse.Namespace,
    feature_shards: Sequence[FeatureShard],
    progress: PipelineProgress,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(empty_group)
    margin_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    utility_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for source in feature_shards:
        paths = output_paths(args.output_root, source.dataset, args.source_split, source.name)
        if not output_complete(paths, source, args.temperature):
            raise RuntimeError(f"Cannot summarize incomplete score shard: {paths['root']}")
        rows = list(iter_jsonl(paths["rows"]))
        if len(rows) != source.pair_count:
            raise RuntimeError(f"Output row count mismatch: {paths['rows']}")
        for row in rows:
            update_group_summary(groups["overall"], row)
            update_group_summary(groups[f"dataset:{source.dataset}"], row)
            state = "correct" if row["no_document_correct"] else "wrong"
            update_group_summary(groups[f"dataset:{source.dataset}:no_rag_{state}"], row)
        with safe_open(paths["scores"], framework="pt", device="cpu") as handle:
            margin = handle.get_tensor("gold_margin_delta").float()
            utility = handle.get_tensor("boundary_probability_delta").float()
        margin_values["overall"].append(margin)
        utility_values["overall"].append(utility)
        margin_values[f"dataset:{source.dataset}"].append(margin)
        utility_values[f"dataset:{source.dataset}"].append(utility)
        progress.update(source.pair_count)
        progress.set_detail(f"dataset={source.dataset} shard={source.name}")

    quantiles = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    finalized = {name: finalize_group(value) for name, value in sorted(groups.items())}
    for name in ("overall", *(f"dataset:{dataset}" for dataset in args.datasets)):
        margin = torch.cat(margin_values[name])
        utility = torch.cat(utility_values[name])
        finalized[name]["gold_margin_delta_quantiles"] = {
            f"q{int(q.item() * 100):02d}": float(value.item())
            for q, value in zip(quantiles, torch.quantile(margin, quantiles))
        }
        finalized[name]["boundary_probability_delta_quantiles"] = {
            f"q{int(q.item() * 100):02d}": float(value.item())
            for q, value in zip(quantiles, torch.quantile(utility, quantiles))
        }
    return {
        "run_version": RUN_VERSION,
        "created_at": utc_now(),
        "temperature": args.temperature,
            "definitions": {
            "margin": "gold choice logit - strongest wrong choice logit",
            "margin_delta": "with-document margin - no-document margin",
            "boundary_probability": "sigmoid(margin / temperature), a gold-vs-strongest-wrong two-way probability",
            "utility": "with-document boundary probability - no-document boundary probability",
            "positive_boundary_crossing": "W->C under exact four-choice argmax",
            "negative_boundary_crossing": "C->W under exact four-choice argmax",
            "answer_transition_basis": "exact HF replay A/B/C/D logits; original generated transitions are retained separately",
        },
        "groups": finalized,
    }


def write_pretty_summary(path: Path, summary: dict[str, Any], datasets: Sequence[str]) -> None:
    lines = [
        "RAG2 anchored gold-margin utility summary",
        f"temperature: {summary['temperature']}",
        "",
        (
            "group".ljust(34)
            + "pairs".rjust(12)
            + " mean_dm".rjust(14)
            + " mean_du".rjust(14)
            + " W->C".rjust(12)
            + " C->W".rjust(12)
        ),
        "-" * 98,
    ]
    names = ["overall", *(f"dataset:{dataset}" for dataset in datasets)]
    for name in names:
        row = summary["groups"][name]
        transitions = row["transition_counts"]
        lines.append(
            name.ljust(34)
            + f"{row['pairs']:12d}"
            + f"{row['mean_gold_margin_delta']:14.6f}"
            + f"{row['mean_boundary_probability_delta']:14.6f}"
            + f"{transitions.get('W->C', 0):12d}"
            + f"{transitions.get('C->W', 0):12d}"
        )
    lines.extend(
        [
            "",
            "dm = (gold logit - strongest wrong logit)_D - the same no-RAG margin",
            "du = sigmoid(m_D / T) - sigmoid(m_0 / T)",
            "This stage does not assign Helpful/Neutral/Harmful thresholds.",
        ]
    )
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        raise ValueError("--temperature must be finite and positive")

    (
        no_manifest,
        doc_manifest,
        candidate_manifests,
        feature_shards,
        total_questions,
        total_pairs,
    ) = discover_inputs(args)
    completed_pairs = sum(
        source.pair_count
        for source in feature_shards
        if args.resume
        and output_complete(
            output_paths(args.output_root, source.dataset, args.source_split, source.name),
            source,
            args.temperature,
        )
    )
    logging.info(
        "Gold-margin plan: questions=%d pairs=%d completed_pairs=%d remaining_pairs=%d "
        "temperature=%.6g GPU_forward=false",
        total_questions,
        total_pairs,
        completed_pairs,
        total_pairs - completed_pairs,
        args.temperature,
    )
    for dataset in args.datasets:
        manifest = candidate_manifests[dataset]
        if args.candidate_contract == "source_balanced_top8":
            contract_text = "4 corpora x 8 dense = 32 -> MedCPT rerank Top-8"
        else:
            contract_text = "dynamic union of 4k -> MedCPT rerank Top-k, k=1/2/4/8/16/32"
        logging.info(
            "[%s] retrieval contract: %s; questions=%d pairs=%d",
            dataset,
            contract_text,
            int(manifest["selected_question_count"]),
            int((doc_manifest["datasets"])[dataset]),
        )
    if args.dry_run:
        logging.info("Dry run complete; no scores were written.")
        return

    progress = PipelineProgress(
        overall_total=total_questions + 2 * total_pairs,
        overall_initial=completed_pairs,
        desc="GoldMarginPipeline",
    )
    try:
        progress.set_stage("1/3 index cached no-RAG choice logits", total=total_questions, initial=0)
        question_index = build_question_index(args, progress)
        # The manifest may describe more datasets than this invocation.  A
        # dataset-scoped smoke run must validate only its requested coverage.
        expected_index = total_questions
        if len(question_index) != expected_index:
            raise RuntimeError(f"No-RAG question index mismatch: {len(question_index)} != {expected_index}")

        progress.set_stage(
            "2/3 compute pair margin and boundary utility",
            total=total_pairs,
            initial=completed_pairs,
        )
        for source in feature_shards:
            paths = output_paths(args.output_root, source.dataset, args.source_split, source.name)
            if args.resume and output_complete(paths, source, args.temperature):
                continue
            process_feature_shard(args, source, question_index, paths, progress)

        progress.set_stage("3/3 aggregate score audit", total=total_pairs, initial=0)
        summary = build_summary(args, feature_shards, progress)
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_root / "summary.json", summary)
        write_pretty_summary(args.output_root / "summary_table_pretty.txt", summary, args.datasets)
        atomic_json(
            args.output_root / "manifest.json",
            {
                "run_version": RUN_VERSION,
                "created_at": utc_now(),
                "source_split": args.source_split,
                "datasets": args.datasets,
                "questions_in_no_rag_cache": total_questions,
                "scored_questions": {
                    dataset: int(candidate_manifests[dataset]["selected_question_count"])
                    for dataset in args.datasets
                },
                "pairs": total_pairs,
                "temperature": args.temperature,
                "no_rag_feature_root": str(args.no_rag_feature_root.resolve()),
                "document_feature_root": str(args.document_feature_root.resolve()),
                "candidate_root": str(args.candidate_root.resolve()),
                "model_name_or_path": no_manifest["model_name_or_path"],
                "trace_version": TRACE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "candidate_contract": args.candidate_contract,
                "retrieval_contract": (
                    "4 corpora x 8 dense candidates = 32; MedCPT rerank Top-8"
                    if args.candidate_contract == "source_balanced_top8"
                    else "dynamic union of independent 4k -> MedCPT rerank Top-k conditions for k=1/2/4/8/16/32"
                ),
                "gpu_forward_performed": False,
                "score_shard_layout": "score_shards/{dataset}/{split}/{shard}/rows.jsonl + scores.safetensors",
                "answer_transition_basis": "exact_hf_replay_four_choice_logits",
            },
        )
        logging.info("Gold-margin utility materialization complete: %s", args.output_root)
        logging.info("Summary: %s", args.output_root / "summary_table_pretty.txt")
    finally:
        progress.close()


if __name__ == "__main__":
    main()

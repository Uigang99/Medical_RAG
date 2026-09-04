#!/usr/bin/env python3
"""Bounded pilot for Semantic document-influence adjustment.

The frozen Llama receives Question + Options + Top-8 documents in the existing
anchored Direct-Choice format.  Only document-token K/V LoRA parameters are
trainable.  For one Support and one non-support document sampled per question,
the loss makes the Support removal effect larger than the non-support removal
effect while suppressing the latter and preserving the frozen full-context
output.  Removal effects are full-vocabulary JSD under the exact all-layer
hard-mask proxy validated against physical token deletion.

Gold answers, answer correctness, and Semantic labels are never placed in the
prompt. Gold answers are retained only because the shared prompt schema
requires the field; no gold likelihood, margin, selection, checkpoint, or
metric reads it. Semantic labels are the supervision and may themselves have
been assigned with access to the reference answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rag2_direct_choice_document_attribution import (  # noqa: E402
    HierarchicalProgress,
    choice_token_ids,
    direct_sequence,
)
from evaluate_rag2_direct_choice_document_mask_validity import (  # noqa: E402
    canonical_hash,
    iter_jsonl,
    jsd_from_logits,
    model_identity,
    sha256_file,
)
from generate_rag2_anchored_document_traces import document_pair_id  # noqa: E402
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from medrag.training.document_path_lora import DocumentPathAdapter  # noqa: E402


RUN_VERSION = "rag2_semantic_influence_bounded_pilot_v1"
DATA_VERSION = "rag2_semantic_influence_mixed_top8_256_64_64_v1"
INFLUENCE_VERSION = "full_vocabulary_jsd_all_layer_compact_hard_mask_v1"
SUPPORT_LABELS = frozenset({"direct_support", "supporting_evidence"})
NON_SUPPORT_LABELS = frozenset({"no_evidence", "misleading_evidence"})
VALID_LABELS = SUPPORT_LABELS | NON_SUPPORT_LABELS
SPLITS = ("train", "val", "test")

DEFAULT_BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
DEFAULT_SEMANTIC_ROOT = DEFAULT_BASE / "filter_training_inputs_semantic_top8_four_class_v1"
DEFAULT_CANDIDATE_ROOT = DEFAULT_BASE / "candidates/source_balanced32_rerank8_v1"
DEFAULT_PREPARED_ROOT = DEFAULT_BASE / "semantic_influence_bounded_pilot_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "models/RAG2-Semantic-Influence-Pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medqa",), default="medqa")
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="medqa_mixed_top8_256_64_64_v1")
    parser.add_argument("--train-questions", type=int, default=256)
    parser.add_argument("--val-questions", type=int, default=64)
    parser.add_argument("--test-questions", type=int, default=64)
    parser.add_argument("--overfit-questions", type=int, default=32)
    parser.add_argument("--overfit-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--non-support-weight", type=float, default=0.5)
    parser.add_argument("--support-floor-weight", type=float, default=1.0)
    parser.add_argument("--full-preservation-weight", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(values)).encode("utf-8")).hexdigest()


def source_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        value["sha256"] = sha256_file(path)
    return value


def load_split_labels(paths: dict[str, Path]) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, int]]:
    sample_split: dict[str, str] = {}
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    counts: Counter[str] = Counter()
    for split, path in paths.items():
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            pair_id = str(row.get("pair_id") or "")
            label = str(row.get("semantic_label") or row.get("target") or "")
            if not sample_id or not pair_id or label not in VALID_LABELS:
                raise RuntimeError(f"Malformed Semantic row: {path} sample={sample_id} pair={pair_id} label={label}")
            previous_split = sample_split.setdefault(sample_id, split)
            if previous_split != split:
                raise RuntimeError(f"Question appears in multiple splits: {sample_id}")
            previous = labels[sample_id].setdefault(pair_id, label)
            if previous != label:
                raise RuntimeError(f"Conflicting Semantic labels: {pair_id}")
            counts[f"{split}:{label}"] += 1
    return sample_split, labels, dict(counts)


def normalize_documents(sample_id: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    documents = sorted(
        list(raw.get("candidate_documents") or []),
        key=lambda item: int(item.get("rerank_rank", 10**9)),
    )
    if len(documents) != 8:
        raise RuntimeError(f"Expected Top-8 documents: {sample_id} got={len(documents)}")
    normalized = []
    for rank, original in enumerate(documents, 1):
        document = dict(original)
        if int(document.get("rerank_rank") or rank) != rank:
            raise RuntimeError(f"Non-contiguous rerank order: {sample_id}")
        document["text"] = str(document.get("text") or "").strip()
        if not document["text"]:
            raise RuntimeError(f"Empty document: {sample_id}:{rank}")
        document["rerank_rank"] = rank
        document["pair_id"] = document_pair_id(sample_id, document, rank)
        normalized.append(document)
    return normalized


def prepare_data(args: argparse.Namespace, progress: HierarchicalProgress) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    prepared_dir = args.prepared_root / args.dataset / DATA_VERSION
    manifest_path = prepared_dir / "manifest.json"
    split_paths = {split: args.semantic_root / args.dataset / f"{split}.jsonl" for split in SPLITS}
    candidate_path = args.candidate_root / args.dataset / "train/candidates_top8.jsonl"
    for path in [candidate_path, *split_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    limits = {"train": args.train_questions, "val": args.val_questions, "test": args.test_questions}
    source_contract = {
        "data_version": DATA_VERSION,
        "dataset": args.dataset,
        "candidate": source_identity(candidate_path),
        "semantic_splits": {key: source_identity(value) for key, value in split_paths.items()},
        "limits": limits,
        "selection": "complete Top-8 mixed Support/non-support questions; seeded sample_id ordering",
        "document_order": "seeded pair_id permutation fixed across baseline/train/evaluation",
        "seed": args.seed,
        "direct_gold_answer_use": "none; answer field retained only for shared prompt-schema validation",
        "semantic_supervision_provenance": "existing labels may have been assigned with the reference answer",
    }
    fingerprint = canonical_hash(source_contract)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract_fingerprint") != fingerprint:
            raise RuntimeError("Prepared-data contract mismatch; use a new --prepared-root or run name")
        data = {split: list(iter_jsonl(prepared_dir / f"{split}.jsonl")) for split in SPLITS}
        if {split: len(rows) for split, rows in data.items()} != limits:
            raise RuntimeError("Prepared-data row counts do not match the manifest contract")
        progress.set(5)
        return data, manifest

    sample_split, semantic, label_counts = load_split_labels(split_paths)
    progress.set(4)
    eligible: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for raw in iter_jsonl(candidate_path):
        sample_id = str(raw.get("sample_id") or "")
        split = sample_split.get(sample_id)
        if split is None:
            continue
        documents = normalize_documents(sample_id, raw)
        pair_ids = [str(document["pair_id"]) for document in documents]
        if any(pair_id not in semantic[sample_id] for pair_id in pair_ids):
            continue
        labels = [semantic[sample_id][pair_id] for pair_id in pair_ids]
        if not (set(labels) & SUPPORT_LABELS and set(labels) & NON_SUPPORT_LABELS):
            continue
        permutation = sorted(
            range(8),
            key=lambda index: stable_key(args.seed, sample_id, pair_ids[index]),
        )
        documents = [documents[index] for index in permutation]
        labels = [labels[index] for index in permutation]
        eligible[split].append({
            "sample_id": sample_id,
            "row_idx": int(raw["row_idx"]),
            "analysis_split": split,
            "question": str(raw["question"]),
            "options": dict(raw["options"]),
            "gold_answer": str(raw.get("answer") or (raw.get("answers") or [""])[0]),
            "documents": documents,
            "semantic_labels": labels,
        })
    progress.set(5)
    selected: dict[str, list[dict[str, Any]]] = {}
    eligible_counts = {split: len(rows) for split, rows in eligible.items()}
    for split, rows in eligible.items():
        rows.sort(key=lambda row: stable_key(args.seed + 1, split, row["sample_id"]))
        if len(rows) < limits[split]:
            raise RuntimeError(f"Insufficient mixed {split} questions: required={limits[split]} available={len(rows)}")
        selected[split] = rows[: limits[split]]
        atomic_jsonl(prepared_dir / f"{split}.jsonl", selected[split])
    selected_ids = {split: [row["sample_id"] for row in rows] for split, rows in selected.items()}
    if len(set().union(*(set(values) for values in selected_ids.values()))) != sum(map(len, selected_ids.values())):
        raise RuntimeError("Prepared splits are not question-disjoint")
    position_counts = {str(index + 1): Counter() for index in range(8)}
    for rows in selected.values():
        for row in rows:
            for index, label in enumerate(row["semantic_labels"]):
                position_counts[str(index + 1)]["support" if label in SUPPORT_LABELS else "non_support"] += 1
    manifest = {
        **source_contract,
        "contract_fingerprint": fingerprint,
        "created_at": utc_now(),
        "eligible_questions": eligible_counts,
        "selected_ids": selected_ids,
        "source_label_counts": label_counts,
        "selected_position_label_counts": {key: dict(value) for key, value in position_counts.items()},
    }
    atomic_json(manifest_path, manifest)
    return selected, manifest


def encode_rows(
    tokenizer: Any,
    rows: dict[str, list[dict[str, Any]]],
    max_tokens: int,
    progress: HierarchicalProgress,
) -> dict[str, list[dict[str, Any]]]:
    encoded: dict[str, list[dict[str, Any]]] = {}
    completed = 0
    for split in SPLITS:
        values = []
        for row in rows[split]:
            sequence = direct_sequence(tokenizer, row, row["documents"])
            if len(sequence["input_ids"]) > max_tokens:
                raise RuntimeError(
                    f"Prompt exceeds max input tokens: {row['sample_id']} "
                    f"tokens={len(sequence['input_ids'])} limit={max_tokens}"
                )
            mapping = torch.full((len(sequence["input_ids"]),), -1, dtype=torch.long)
            for document_index, positions in enumerate(sequence["document_token_indices"]):
                if not positions:
                    raise RuntimeError(f"Document has no token span: {row['sample_id']}:{document_index+1}")
                mapping[torch.tensor(positions, dtype=torch.long)] = document_index
            values.append({
                "row": row,
                "input_ids": torch.tensor(sequence["input_ids"], dtype=torch.long),
                "token_document_ids": mapping,
                "document_mask": mapping.ge(0),
                "support_indices": [
                    index for index, label in enumerate(row["semantic_labels"])
                    if label in SUPPORT_LABELS
                ],
                "non_support_indices": [
                    index for index, label in enumerate(row["semantic_labels"])
                    if label in NON_SUPPORT_LABELS
                ],
                "input_tokens": len(sequence["input_ids"]),
            })
            completed += 1
            progress.set(completed)
        encoded[split] = values
    return encoded


def compact_positions(mapping: torch.Tensor, blocked_document: int) -> torch.Tensor:
    kept = mapping.ne(int(blocked_document))
    positions = kept.long().cumsum(dim=0) - 1
    return positions.clamp_min(0)


def build_eval_batch(value: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Base full, student full, then eight student hard-mask variants."""
    ids = value["input_ids"]
    mapping = value["token_document_ids"]
    rows = 10
    positions = torch.arange(ids.numel(), dtype=torch.long).unsqueeze(0).repeat(rows, 1)
    for document_index in range(8):
        positions[document_index + 2] = compact_positions(mapping, document_index)
    adapter_mask = value["document_mask"].unsqueeze(0).repeat(rows, 1)
    adapter_mask[0].zero_()
    return {
        "input_ids": ids.unsqueeze(0).repeat(rows, 1),
        "attention_mask": torch.ones((rows, ids.numel()), dtype=torch.long),
        "position_ids": positions,
        "token_document_ids": mapping.unsqueeze(0).repeat(rows, 1),
        "blocked_document_ids": torch.tensor([-2, -2, *range(8)], dtype=torch.long),
        "adapter_document_mask": adapter_mask,
    }


def build_train_batch(value: dict[str, Any], support_index: int, non_support_index: int) -> dict[str, torch.Tensor]:
    """Base full, student full, one Support mask, one non-support mask."""
    ids = value["input_ids"]
    mapping = value["token_document_ids"]
    positions = torch.arange(ids.numel(), dtype=torch.long).unsqueeze(0).repeat(4, 1)
    positions[2] = compact_positions(mapping, support_index)
    positions[3] = compact_positions(mapping, non_support_index)
    adapter_mask = value["document_mask"].unsqueeze(0).repeat(4, 1)
    adapter_mask[0].zero_()
    return {
        "input_ids": ids.unsqueeze(0).repeat(4, 1),
        "attention_mask": torch.ones((4, ids.numel()), dtype=torch.long),
        "position_ids": positions,
        "token_document_ids": mapping.unsqueeze(0).repeat(4, 1),
        "blocked_document_ids": torch.tensor([-2, -2, support_index, non_support_index]),
        "adapter_document_mask": adapter_mask,
    }


def forward_logits(
    model: Any,
    adapter: DocumentPathAdapter,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    adapter.set_document_mask(batch["adapter_document_mask"].to(device))
    output = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        semantic_token_document_ids=batch["token_document_ids"].to(device),
        semantic_blocked_document_ids=batch["blocked_document_ids"].to(device),
        semantic_document_block_layer_start=0,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    return output.logits[:, -1].float()


def pair_win(influences: Sequence[float], support: Sequence[int], non_support: Sequence[int]) -> float:
    comparisons = []
    for left in support:
        for right in non_support:
            comparisons.append(
                1.0 if influences[left] > influences[right]
                else 0.5 if influences[left] == influences[right]
                else 0.0
            )
    return float(np.mean(comparisons))


def summarize_eval_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    support = [value for row in rows for value in row["support_influences"]]
    non_support = [value for row in rows for value in row["non_support_influences"]]
    return {
        "questions": len(rows),
        "documents": len(support) + len(non_support),
        "mean_question_support_over_non_pair_accuracy": float(np.mean([row["pair_accuracy"] for row in rows])),
        "mean_support_influence_jsd": float(np.mean(support)),
        "mean_non_support_influence_jsd": float(np.mean(non_support)),
        "median_support_influence_jsd": float(np.median(support)),
        "median_non_support_influence_jsd": float(np.median(non_support)),
        "mean_full_output_drift_jsd": float(np.mean([row["full_output_drift_jsd"] for row in rows])),
        "constrained_choice_preservation": float(np.mean([row["choice_preserved"] for row in rows])),
    }


@torch.inference_mode()
def evaluate(
    model: Any,
    adapter: DocumentPathAdapter,
    values: Sequence[dict[str, Any]],
    choice_ids: torch.Tensor,
    device: torch.device,
    progress: HierarchicalProgress | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    result_rows = []
    for value in values:
        logits = forward_logits(model, adapter, build_eval_batch(value), device)
        base_full, student_full, masks = logits[0], logits[1], logits[2:]
        influence = jsd_from_logits(student_full.unsqueeze(0), masks)
        support = value["support_indices"]
        non_support = value["non_support_indices"]
        base_choice = int(base_full.index_select(0, choice_ids).argmax().item())
        student_choice = int(student_full.index_select(0, choice_ids).argmax().item())
        result_rows.append({
            "sample_id": value["row"]["sample_id"],
            "semantic_labels": value["row"]["semantic_labels"],
            "influences_jsd": [float(item) for item in influence.cpu().tolist()],
            "support_influences": [float(influence[index].item()) for index in support],
            "non_support_influences": [float(influence[index].item()) for index in non_support],
            "pair_accuracy": pair_win(influence.cpu().tolist(), support, non_support),
            "full_output_drift_jsd": float(jsd_from_logits(base_full, student_full).item()),
            "base_choice": CHOICES[base_choice],
            "student_choice": CHOICES[student_choice],
            "choice_preserved": float(base_choice == student_choice),
        })
        if progress is not None:
            progress.update()
    return summarize_eval_rows(result_rows), result_rows


def compare_metrics(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    base_non = max(float(baseline["mean_non_support_influence_jsd"]), 1e-12)
    base_support = max(float(baseline["mean_support_influence_jsd"]), 1e-12)
    return {
        "pair_accuracy_gain": (
            float(current["mean_question_support_over_non_pair_accuracy"])
            - float(baseline["mean_question_support_over_non_pair_accuracy"])
        ),
        "non_support_relative_reduction": 1.0 - float(current["mean_non_support_influence_jsd"]) / base_non,
        "support_influence_retention": float(current["mean_support_influence_jsd"]) / base_support,
        "full_output_drift_jsd": float(current["mean_full_output_drift_jsd"]),
        "choice_preservation": float(current["constrained_choice_preservation"]),
    }


def selection_score(comparison: dict[str, float]) -> float:
    penalty = (
        5.0 * max(0.0, 0.90 - comparison["support_influence_retention"])
        + 5.0 * max(0.0, comparison["full_output_drift_jsd"] - 0.01)
    )
    return (
        comparison["pair_accuracy_gain"]
        + 0.25 * max(-1.0, min(1.0, comparison["non_support_relative_reduction"]))
        - penalty
    )


def influence_losses(
    logits: torch.Tensor,
    baseline_support_jsd: float,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if logits.shape[0] != 4:
        raise ValueError("Training logits must contain base/full/Support-mask/non-support-mask")
    base_full, student_full, support_mask, non_support_mask = logits
    eps = 1e-8
    support_distance = torch.sqrt(jsd_from_logits(student_full, support_mask) + eps)
    non_support_distance = torch.sqrt(jsd_from_logits(student_full, non_support_mask) + eps)
    baseline_support_distance = math.sqrt(max(0.0, float(baseline_support_jsd)) + eps)
    ranking = F.relu(args.ranking_margin - (support_distance - non_support_distance))
    non_support = non_support_distance
    support_floor = F.relu(
        torch.as_tensor(baseline_support_distance, device=logits.device) - support_distance
    )
    preservation = torch.sqrt(jsd_from_logits(base_full.detach(), student_full) + eps)
    total = (
        args.ranking_weight * ranking
        + args.non_support_weight * non_support
        + args.support_floor_weight * support_floor
        + args.full_preservation_weight * preservation
    )
    return {
        "loss": total,
        "ranking": ranking,
        "non_support": non_support,
        "support_floor": support_floor,
        "full_preservation": preservation,
        "support_distance": support_distance,
        "non_support_distance": non_support_distance,
    }


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_phase(
    *,
    phase: str,
    model: Any,
    adapter: DocumentPathAdapter,
    train_values: Sequence[dict[str, Any]],
    validation_values: Sequence[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    validation_baseline: dict[str, Any],
    choice_ids: torch.Tensor,
    args: argparse.Namespace,
    epochs: int,
    output_dir: Path,
    progress: HierarchicalProgress,
    early_stop_overfit: bool,
    contract_fingerprint: str,
) -> dict[str, Any]:
    device = torch.device(args.device)
    parameters = list(adapter.trainable_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best_adapter.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_fingerprint") != contract_fingerprint:
            raise RuntimeError(f"{phase} checkpoint contract mismatch")
        adapter.load_adapter_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
    already_completed = (start_epoch - 1) * (len(train_values) + len(validation_values))
    progress.set(already_completed)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        indices = list(range(len(train_values)))
        random.Random(args.seed + epoch + (0 if phase == "overfit" else 1000)).shuffle(indices)
        sums: Counter[str] = Counter()
        for ordinal in indices:
            value = train_values[ordinal]
            support_values = value["support_indices"]
            non_values = value["non_support_indices"]
            support_index = support_values[(epoch - 1) % len(support_values)]
            non_support_index = non_values[(epoch - 1) % len(non_values)]
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(
                model, adapter,
                build_train_batch(value, support_index, non_support_index),
                device,
            )
            baseline_jsd = baseline_by_id[value["row"]["sample_id"]]["influences_jsd"][support_index]
            losses = influence_losses(logits, baseline_jsd, args)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for key, tensor in losses.items():
                sums[key] += float(tensor.detach().item())
            progress.update()
        validation_metrics, _ = evaluate(
            model, adapter, validation_values, choice_ids, device, progress
        )
        comparison = compare_metrics(validation_metrics, validation_baseline)
        score = selection_score(comparison)
        epoch_row = {
            "epoch": epoch,
            "train_loss": {key: value / len(train_values) for key, value in sums.items()},
            "validation": validation_metrics,
            "comparison_to_frozen": comparison,
            "selection_score": score,
        }
        history.append(epoch_row)
        if score > best_score:
            best_score = score
            atomic_torch(best_path, {
                "contract_fingerprint": contract_fingerprint,
                "epoch": epoch, "score": score, "adapter": adapter.adapter_state_dict(),
            })
        atomic_torch(checkpoint_path, {
            "contract_fingerprint": contract_fingerprint,
            "epoch": epoch,
            "best_score": best_score,
            "adapter": adapter.adapter_state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        })
        print(
            f"[{phase} epoch {epoch}/{epochs}] loss={epoch_row['train_loss']['loss']:.5f} "
            f"pair_gain={comparison['pair_accuracy_gain']:+.4f} "
            f"non_support_reduction={comparison['non_support_relative_reduction']:+.4f} "
            f"support_retention={comparison['support_influence_retention']:.4f} "
            f"full_drift={comparison['full_output_drift_jsd']:.6f}",
            flush=True,
        )
        if early_stop_overfit and (
            validation_metrics["mean_question_support_over_non_pair_accuracy"] >= 0.80
            and comparison["non_support_relative_reduction"] >= 0.30
            and comparison["support_influence_retention"] >= 0.90
            and comparison["full_output_drift_jsd"] <= 0.02
        ):
            break
    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint was produced for {phase}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("contract_fingerprint") != contract_fingerprint:
        raise RuntimeError(f"{phase} best-checkpoint contract mismatch")
    adapter.load_adapter_state_dict(best["adapter"])
    final_metrics, final_rows = evaluate(
        model, adapter, validation_values, choice_ids, device, progress=None
    )
    comparison = compare_metrics(final_metrics, validation_baseline)
    result = {
        "phase": phase,
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "metrics": final_metrics,
        "comparison_to_frozen": comparison,
        "history": history,
        "adapter_audit": adapter.audit(),
    }
    atomic_json(output_dir / "summary.json", result)
    atomic_jsonl(output_dir / "per_question.jsonl", final_rows)
    return result


def bootstrap_test(
    baseline_rows: Sequence[dict[str, Any]],
    final_rows: Sequence[dict[str, Any]],
    replicates: int,
    seed: int,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    if [row["sample_id"] for row in baseline_rows] != [row["sample_id"] for row in final_rows]:
        raise RuntimeError("Baseline/final test question order mismatch")
    rng = np.random.default_rng(seed)
    pair_gains: list[float] = []
    non_reductions: list[float] = []
    support_retentions: list[float] = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(final_rows), size=len(final_rows))
        base = summarize_eval_rows([baseline_rows[int(index)] for index in indices])
        final = summarize_eval_rows([final_rows[int(index)] for index in indices])
        comparison = compare_metrics(final, base)
        pair_gains.append(comparison["pair_accuracy_gain"])
        non_reductions.append(comparison["non_support_relative_reduction"])
        support_retentions.append(comparison["support_influence_retention"])
        if (replicate + 1) % 25 == 0 or replicate + 1 == replicates:
            progress.set(replicate + 1)
    def ci(values: Sequence[float]) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "pair_accuracy_gain": ci(pair_gains),
        "non_support_relative_reduction": ci(non_reductions),
        "support_influence_retention": ci(support_retentions),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    test = summary.get("test", {})
    comparison = test.get("comparison_to_frozen", {})
    decision = summary["decision"]
    def number(key: str, digits: int = 4) -> str:
        value = comparison.get(key)
        return "NA" if value is None else f"{float(value):.{digits}f}"
    lines = [
        "# Semantic document-influence bounded pilot",
        "",
        f"- Tiny overfit passed: {summary['overfit']['passed']}",
        f"- Held-out pilot executed: {bool(test)}",
        "- Direct gold-answer likelihood or margin objective: none",
        "- Supervision: existing Semantic labels, which may have been assigned with the reference answer",
        "- Semantic labels in prompt: no",
        "",
    ]
    if test:
        lines.extend([
            "| Held-out test criterion | Result | Threshold | Passed |",
            "|---|---:|---:|---:|",
            f"| Support>non-support pair accuracy gain | {number('pair_accuracy_gain')} | ≥0.08 | {decision['pair_gain_pass']} |",
            f"| Non-support mean influence reduction | {number('non_support_relative_reduction')} | ≥0.20 | {decision['non_support_pass']} |",
            f"| Support influence retained | {number('support_influence_retention')} | ≥0.90 | {decision['support_retention_pass']} |",
            f"| Full-output mean JSD drift | {number('full_output_drift_jsd', 6)} | ≤0.01 | {decision['full_drift_pass']} |",
            "",
            f"- **Overall passed: {decision['overall_pass']}**",
        ])
    else:
        lines.append("Held-out training was stopped because the tiny-set overfit criterion failed.")
    lines.extend([
        "",
        "This pilot tests only whether document influence can be adjusted on held-out MedQA questions. It does not establish accuracy improvement or cross-dataset generalization.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if min(
        args.train_questions, args.val_questions, args.test_questions,
        args.overfit_questions, args.overfit_epochs, args.epochs, args.lora_rank,
    ) <= 0:
        raise ValueError("Question, epoch, and LoRA sizes must be positive")
    if args.overfit_questions > args.train_questions:
        raise ValueError("--overfit-questions cannot exceed --train-questions")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_root / args.dataset / args.run_name
    complete_path = output_dir / "COMPLETE.json"
    stages = [
        "materialize disjoint mixed Top-8 splits",
        "tokenize Direct-Choice prompts and exact document spans",
        "measure frozen influence baseline",
        "tiny-set influence overfit gate",
        "held-out influence training and validation selection",
        "one-time held-out test evaluation",
        "paired bootstrap and final report",
    ]
    progress = HierarchicalProgress(stages, [30.0, 20.0, 700.0, 1800.0, 4200.0, 180.0, 10.0])
    try:
        progress.start(1, 5, "source")
        rows, data_manifest = prepare_data(args, progress)
        progress.complete(
            f"selected={{train:{len(rows['train'])},val:{len(rows['val'])},test:{len(rows['test'])}}} "
            f"prepared={args.prepared_root/args.dataset/DATA_VERSION}"
        )

        contract = {
            "run_version": RUN_VERSION,
            "dataset": args.dataset,
            "data_contract_fingerprint": data_manifest["contract_fingerprint"],
            "model": model_identity(args.model),
            "prompt": "anchored question-first Direct-Choice without rationale",
            "trainable_path": "document-token K/V LoRA only; all base Llama parameters frozen",
            "semantic_labels_in_prompt": False,
            "direct_gold_answer_use": "none",
            "semantic_supervision_provenance": (
                "existing labels may have been assigned with the reference answer"
            ),
            "influence": INFLUENCE_VERSION,
            "sizes": {
                "train": args.train_questions, "val": args.val_questions, "test": args.test_questions,
                "overfit": args.overfit_questions,
            },
            "epochs": {"overfit": args.overfit_epochs, "pilot": args.epochs},
            "optimizer": {
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "gradient_clip": 1.0,
            },
            "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
            "loss": {
                "ranking_margin": args.ranking_margin,
                "ranking_weight": args.ranking_weight,
                "non_support_weight": args.non_support_weight,
                "support_floor_weight": args.support_floor_weight,
                "full_preservation_weight": args.full_preservation_weight,
            },
            "checkpoint_selection": "validation mechanism score only; test evaluated once",
            "test_pass_thresholds": {
                "pair_accuracy_gain": 0.08,
                "non_support_relative_reduction": 0.20,
                "support_influence_retention": 0.90,
                "full_output_drift_jsd": 0.01,
                "pair_gain_bootstrap_lower_bound": 0.0,
            },
            "dtype": args.dtype,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "code_commit": git_commit(),
        }
        fingerprint = canonical_hash(contract)
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract_fingerprint") != fingerprint:
                raise RuntimeError("Pilot resume contract mismatch; use a new RUN_NAME")
        else:
            atomic_json(manifest_path, {
                **contract, "contract_fingerprint": fingerprint, "created_at": utc_now(),
            })
        if complete_path.is_file():
            completed = json.loads(complete_path.read_text(encoding="utf-8"))
            if completed.get("contract_fingerprint") != fingerprint:
                raise RuntimeError("Completed output has a different contract")
            progress.finish(f"already complete; report={output_dir/'report.md'}")
            return

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("Fast tokenizer required for exact document spans")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        progress.start(2, sum(len(value) for value in rows.values()), "question")
        encoded = encode_rows(tokenizer, rows, args.max_input_tokens, progress)
        maximum_tokens = max(value["input_tokens"] for split in SPLITS for value in encoded[split])
        progress.complete(f"questions={sum(map(len, encoded.values()))} max_tokens={maximum_tokens}")
        if args.preflight_only:
            progress.finish(f"preflight-only passed; manifest={manifest_path}")
            return

        attention_name = register_semantic_attention()
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        device = torch.device(args.device)
        print(
            f"[model load] model={args.model} device={device} dtype={args.dtype} attention={attention_name}",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, low_cpu_mem_usage=True,
            torch_dtype=dtype, attn_implementation=attention_name,
        ).to(device)
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapter = DocumentPathAdapter(
            model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        )
        audit = adapter.audit()
        if audit["unexpected_trainable_parameters"]:
            raise RuntimeError(f"Unexpected trainable parameters: {audit['unexpected_trainable_parameters'][:5]}")
        initial_state = adapter.adapter_state_dict()
        choice_ids = choice_token_ids(tokenizer, device)

        baseline_metrics: dict[str, dict[str, Any]] = {}
        baseline_rows: dict[str, list[dict[str, Any]]] = {}
        baseline_dir = output_dir / "frozen_baseline"
        progress.start(3, sum(len(value) for value in encoded.values()), "question")
        for split in SPLITS:
            path = baseline_dir / f"{split}.jsonl"
            metric_path = baseline_dir / f"{split}_summary.json"
            if args.resume and path.is_file() and metric_path.is_file():
                current_rows = list(iter_jsonl(path))
                current_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                if len(current_rows) != len(encoded[split]):
                    raise RuntimeError(f"Incomplete cached frozen baseline: {split}")
                progress.update(len(current_rows))
            else:
                current_metrics, current_rows = evaluate(
                    model, adapter, encoded[split], choice_ids, device, progress
                )
                atomic_jsonl(path, current_rows)
                atomic_json(metric_path, current_metrics)
            baseline_metrics[split] = current_metrics
            baseline_rows[split] = current_rows
        baseline_by_id = {
            row["sample_id"]: row for split in SPLITS for row in baseline_rows[split]
        }
        progress.complete(f"baseline={baseline_dir}")

        overfit_values = encoded["train"][: args.overfit_questions]
        overfit_baseline_rows = baseline_rows["train"][: args.overfit_questions]
        overfit_baseline = summarize_eval_rows(overfit_baseline_rows)
        progress.start(
            4,
            args.overfit_epochs * (len(overfit_values) * 2),
            "question-pass",
        )
        adapter.load_adapter_state_dict(initial_state)
        overfit = train_phase(
            phase="overfit", model=model, adapter=adapter,
            train_values=overfit_values, validation_values=overfit_values,
            baseline_by_id=baseline_by_id, validation_baseline=overfit_baseline,
            choice_ids=choice_ids, args=args, epochs=args.overfit_epochs,
            output_dir=output_dir / "overfit", progress=progress,
            early_stop_overfit=True,
            contract_fingerprint=fingerprint,
        )
        overfit_comparison = overfit["comparison_to_frozen"]
        overfit_pass = bool(
            overfit["metrics"]["mean_question_support_over_non_pair_accuracy"] >= 0.80
            and overfit_comparison["non_support_relative_reduction"] >= 0.30
            and overfit_comparison["support_influence_retention"] >= 0.90
            and overfit_comparison["full_output_drift_jsd"] <= 0.02
        )
        progress.complete(f"passed={overfit_pass} best_epoch={overfit['best_epoch']}")
        if not overfit_pass:
            summary = {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "overfit": {"passed": False, **overfit},
                "decision": {"overall_pass": False, "reason": "tiny-set overfit failed; held-out training not run"},
            }
            atomic_json(output_dir / "summary.json", summary)
            report = report_markdown(summary)
            atomic_text(output_dir / "report.md", report)
            atomic_json(complete_path, {
                "contract_fingerprint": fingerprint, "completed_at": utc_now(),
                "overall_pass": False, "stopped_after": "overfit",
            })
            progress.finish(f"STOP: tiny overfit failed; report={output_dir/'report.md'}")
            return

        adapter.load_adapter_state_dict(initial_state)
        progress.start(
            5,
            args.epochs * (len(encoded["train"]) + len(encoded["val"])),
            "question-pass",
        )
        pilot = train_phase(
            phase="pilot", model=model, adapter=adapter,
            train_values=encoded["train"], validation_values=encoded["val"],
            baseline_by_id=baseline_by_id, validation_baseline=baseline_metrics["val"],
            choice_ids=choice_ids, args=args, epochs=args.epochs,
            output_dir=output_dir / "pilot", progress=progress,
            early_stop_overfit=False,
            contract_fingerprint=fingerprint,
        )
        progress.complete(f"best_epoch={pilot['best_epoch']} score={pilot['best_score']:.4f}")

        progress.start(6, len(encoded["test"]), "question")
        test_metrics, test_rows = evaluate(
            model, adapter, encoded["test"], choice_ids, device, progress
        )
        atomic_jsonl(output_dir / "test_per_question.jsonl", test_rows)
        comparison = compare_metrics(test_metrics, baseline_metrics["test"])
        progress.complete(
            f"pair_gain={comparison['pair_accuracy_gain']:+.4f} "
            f"non_support_reduction={comparison['non_support_relative_reduction']:+.4f}"
        )

        progress.start(7, max(1, args.bootstrap_replicates), "replicate")
        confidence = bootstrap_test(
            baseline_rows["test"], test_rows, args.bootstrap_replicates, args.seed, progress
        ) if args.bootstrap_replicates else {}
        pair_ci = confidence.get("pair_accuracy_gain")
        decision = {
            "pair_gain_pass": comparison["pair_accuracy_gain"] >= 0.08,
            "pair_gain_uncertainty_pass": bool(pair_ci and pair_ci[0] > 0.0),
            "non_support_pass": comparison["non_support_relative_reduction"] >= 0.20,
            "support_retention_pass": comparison["support_influence_retention"] >= 0.90,
            "full_drift_pass": comparison["full_output_drift_jsd"] <= 0.01,
        }
        decision["overall_pass"] = all(decision.values())
        summary = {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "completed_at": utc_now(),
            "overfit": {"passed": True, **overfit},
            "validation_selected_pilot": pilot,
            "test": {
                "frozen": baseline_metrics["test"],
                "trained": test_metrics,
                "comparison_to_frozen": comparison,
                "paired_bootstrap_95pct_ci": confidence,
            },
            "decision": decision,
            "scope_limit": (
                "Mechanism-only MedQA Direct-Choice pilot. No gold/accuracy objective, no final benchmark, "
                "no rationale generation, and no cross-dataset generalization claim."
            ),
        }
        atomic_json(output_dir / "summary.json", summary)
        report_path = output_dir / "report.md"
        atomic_text(report_path, report_markdown(summary))
        atomic_json(complete_path, {
            "contract_fingerprint": fingerprint, "completed_at": utc_now(),
            "overall_pass": decision["overall_pass"], "report": str(report_path.resolve()),
        })
        progress.complete(f"overall_pass={decision['overall_pass']} report={report_path}")
        progress.finish(f"overall_pass={decision['overall_pass']} report={report_path}")
    except Exception:
        print(
            f"[workflow FAILED] stage={progress.stage_index}/{len(stages)} "
            f"completed={progress.stage_done}/{progress.stage_total} output={output_dir}; "
            "rerun the identical command to resume durable split/baseline/epoch checkpoints",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()

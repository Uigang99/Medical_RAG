#!/usr/bin/env python3
"""Select and overfit 256 document-first semantic contrast questions.

This is a mechanism/identifiability test, not a generalization experiment.
Only low-rank updates on document-token K/V projections are trainable.  The
question/options and No-RAG paths remain exactly frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import make_sample, pad_left  # noqa: E402
from evaluate_rag2_document_first_prompt_order import (  # noqa: E402
    HierarchicalProgress,
    sequence_for_order,
)
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from medrag.training.document_path_lora import DocumentPathAdapter  # noqa: E402
from medrag.training.semantic_behavior_lora import gold_margins  # noqa: E402


RUN_VERSION = "rag2_document_path_lora_overfit256_v1"
DATA_VERSION = "rag2_document_path_overfit256_data_v1"
BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "document_first_bounded_direct_outcomes_v1"
)
STRATA = (
    "no_rag_correct__aligned",
    "no_rag_correct__violation",
    "no_rag_wrong__aligned",
    "no_rag_wrong__violation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), default="medmcqa")
    parser.add_argument("--data-root", type=Path, default=BASE)
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/RAG2-Document-Path-LoRA",
    )
    parser.add_argument("--run-name", default="medmcqa_document_first_overfit256_v1")
    parser.add_argument("--questions", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--preference-margin", type=float, default=0.5)
    parser.add_argument("--positive-baseline-margin", type=float, default=0.25)
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument("--positive-baseline-weight", type=float, default=0.5)
    parser.add_argument("--negative-invariance-weight", type=float, default=1.0)
    parser.add_argument("--swap-invariance-weight", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--early-stop-accuracy", type=float, default=0.95)
    parser.add_argument("--early-stop-epochs", type=int, default=2)
    parser.add_argument("--base-logit-tolerance", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager",), default="eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        value["sha256"] = sha256_file(path)
    return value


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def stable_priority(seed: int, sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()[:8], "big")


class SmallestRecords:
    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, priority: int, sample_id: str, record: dict[str, Any]) -> None:
        item = (-int(priority), str(sample_id), record)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def rows(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heap, key=lambda item: (-item[0], item[1]))]


def finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    return True


def compact_document(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document["document"]
    return {
        "pair_id": str(metadata["pair_id"]),
        "semantic_label": str(metadata["semantic_label"]),
        "semantic_confidence": float(metadata["semantic_confidence"]),
        "source": str(metadata["source"]),
        "stable_id": str(metadata["stable_id"]),
        "rerank_rank": int(metadata["rerank_rank"]),
        "document_text": str(metadata["document_text"]),
        "frozen_choice_logits": list(document["choice_logits"]),
        "frozen_choice_probabilities": list(document["choice_probabilities"]),
        "frozen_gold_margin": float(document["gold_margin"]),
        "delta_gold_margin": float(document["delta_gold_margin"]),
        "jsd_from_no_rag": float(document["jsd_from_no_rag"]),
        "correctness_transition": str(document["correctness_transition"]),
    }


def prepare_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    documents = {str(value["document"]["pair_id"]): value for value in row["documents"]}
    positive = documents[str(row["behaviorally_underused_positive_pair_id"])]
    negative = documents[str(row["behaviorally_disruptive_negative_pair_id"])]
    donor = row.get("cross_question_support_donor")
    if not donor or donor.get("donor_split") != "train" or donor.get("donor_sample_id") == row["sample_id"]:
        raise RuntimeError(f"Invalid cross-question train donor: {row['sample_id']}")
    if positive["document"]["semantic_label"] != "direct_support":
        raise RuntimeError(f"D+ is not Direct Support: {row['sample_id']}")
    if negative["document"]["semantic_label"] not in {"no_evidence", "misleading_evidence"}:
        raise RuntimeError(f"D- is not a semantic negative: {row['sample_id']}")
    no_rag_correct = bool(row["frozen_no_rag"]["answer_correct"])
    violation = float(positive["gold_margin"]) <= float(negative["gold_margin"])
    stratum = f"no_rag_{'correct' if no_rag_correct else 'wrong'}__{'violation' if violation else 'aligned'}"
    prepared = {
        "data_version": DATA_VERSION,
        "dataset": row["dataset"],
        "split": "train",
        "sample_id": row["sample_id"],
        "row_idx": int(row["row_idx"]),
        "question": row["question"],
        "options": row["options"],
        "gold_answer": row["gold_answer"],
        "stratum": stratum,
        "frozen_no_rag": {
            key: row["frozen_no_rag"][key]
            for key in (
                "prediction", "answer_correct", "choice_logits", "choice_probabilities",
                "gold_margin", "prompt_sha256", "prompt_token_count",
            )
        },
        "positive": compact_document(positive),
        "negative": compact_document(negative),
        "swap": {
            "donor_sample_id": str(donor["donor_sample_id"]),
            "source": str(donor["source"]),
            "recipient_primary_support_word_count": int(donor["recipient_primary_support_word_count"]),
            "donor_word_count": int(donor["donor_word_count"]),
            "pair_id": str(donor["document"]["pair_id"]),
            "semantic_label": str(donor["document"]["semantic_label"]),
            "document_text": str(donor["document"]["document_text"]),
        },
    }
    if not finite_numbers(prepared):
        raise RuntimeError(f"Non-finite prepared values: {row['sample_id']}")
    return stratum, prepared


def materialize_overfit_data(
    args: argparse.Namespace,
    source_path: Path,
    source_questions: int,
    output_path: Path,
    manifest_path: Path,
    source_contract_sha256: str,
    progress: HierarchicalProgress,
) -> list[dict[str, Any]]:
    if args.questions % len(STRATA):
        raise ValueError(f"--questions must be divisible by {len(STRATA)}")
    if args.resume and output_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = list(iter_jsonl(output_path))
        if (
            manifest.get("source_contract_sha256") == source_contract_sha256
            and int(manifest.get("questions", -1)) == args.questions
            and manifest.get("data_sha256") == sha256_file(output_path)
        ):
            progress.set_initial(source_questions)
            return rows

    per_stratum = args.questions // len(STRATA)
    reservoirs = {stratum: SmallestRecords(per_stratum) for stratum in STRATA}
    available: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    observed = 0
    unique: set[str] = set()
    for row in iter_jsonl(source_path):
        observed += 1
        sample_id = str(row["sample_id"])
        if sample_id in unique:
            raise RuntimeError(f"Duplicate train question: {sample_id}")
        unique.add(sample_id)
        if len(row.get("documents") or []) != 8:
            raise RuntimeError(f"Expected eight document outcomes: {sample_id}")
        for document in row["documents"]:
            semantic_counts[str(document["document"]["semantic_label"])] += 1
            transitions[str(document["correctness_transition"])] += 1
        stratum, prepared = prepare_record(row)
        available[stratum] += 1
        reservoirs[stratum].add(stable_priority(args.seed, sample_id), sample_id, prepared)
        if observed % 32 == 0 or observed == source_questions:
            progress.set_absolute(observed)
    if observed != source_questions:
        raise RuntimeError(f"Train question count mismatch: expected={source_questions} actual={observed}")
    selected = []
    for stratum in STRATA:
        rows = reservoirs[stratum].rows()
        if len(rows) != per_stratum:
            raise RuntimeError(
                f"Insufficient {stratum}: requested={per_stratum} available={available[stratum]}"
            )
        selected.extend(rows)
    random.Random(args.seed).shuffle(selected)
    atomic_jsonl(output_path, selected)
    manifest = {
        "data_version": DATA_VERSION,
        "source_contract_sha256": source_contract_sha256,
        "created_at": utc_now(),
        "source": file_identity(source_path),
        "questions_audited": observed,
        "questions": len(selected),
        "selection": "64 per No-RAG correctness x frozen D+/D- alignment stratum",
        "available_strata": dict(available),
        "selected_strata": dict(Counter(row["stratum"] for row in selected)),
        "audited_semantic_labels": dict(semantic_counts),
        "audited_transitions": dict(transitions),
        "data_sha256": sha256_file(output_path),
    }
    atomic_json(manifest_path, manifest)
    return selected


def encode_document_prompt(tokenizer: Any, row: dict[str, Any], text: str) -> dict[str, Any]:
    sample = make_sample(
        {
            "row_idx": row["row_idx"], "sample_id": row["sample_id"], "dataset": row["dataset"],
            "split": row["split"], "question": row["question"], "options": row["options"],
            "answer": row["gold_answer"],
        }
    )
    document = str(text).strip()
    token_ids, prompt = sequence_for_order(tokenizer, sample, document, "document_first")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != token_ids:
        raise RuntimeError(f"Offset/token encoding mismatch: {row['sample_id']}")
    marker = "Documents:\n"
    marker_start = prompt.find(marker)
    if marker_start < 0:
        raise RuntimeError(f"Missing Documents marker: {row['sample_id']}")
    start = marker_start + len(marker)
    if prompt[start : start + len(document)] != document:
        raise RuntimeError(f"Document character span mismatch: {row['sample_id']}")
    end = start + len(document)
    mask = [bool(offset_end > start and offset_start < end) for offset_start, offset_end in encoded["offset_mapping"]]
    if not any(mask):
        raise RuntimeError(f"Empty document token mask: {row['sample_id']}")
    return {"input_ids": token_ids, "document_mask": mask, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


class EncodedDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, max_tokens: int, progress: HierarchicalProgress) -> None:
        self.values = []
        self.max_tokens = 0
        for index, row in enumerate(rows, 1):
            conditions = {
                "positive": encode_document_prompt(tokenizer, row, row["positive"]["document_text"]),
                "negative": encode_document_prompt(tokenizer, row, row["negative"]["document_text"]),
                "swap": encode_document_prompt(tokenizer, row, row["swap"]["document_text"]),
            }
            current_max = max(len(value["input_ids"]) for value in conditions.values())
            if current_max > max_tokens:
                raise RuntimeError(f"Overfit prompt exceeds {max_tokens}: {row['sample_id']} tokens={current_max}")
            self.max_tokens = max(self.max_tokens, current_max)
            self.values.append({"row": row, "conditions": conditions})
            progress.set_absolute(index)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.values[index]


def collate(values: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    sequences: list[list[int]] = []
    masks: list[list[bool]] = []
    # Condition-major order is part of the loss contract: [all D+, all D-, all Dswap].
    for condition in ("positive", "negative", "swap"):
        for value in values:
            sequences.append(value["conditions"][condition]["input_ids"])
            masks.append(value["conditions"][condition]["document_mask"])
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), maximum), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), maximum), dtype=torch.long)
    document_mask = torch.zeros((len(sequences), maximum), dtype=torch.bool)
    for index, (sequence, mask) in enumerate(zip(sequences, masks)):
        input_ids[index, -len(sequence) :] = torch.tensor(sequence)
        attention[index, -len(sequence) :] = 1
        document_mask[index, -len(sequence) :] = torch.tensor(mask)
    positions = attention.cumsum(dim=-1) - 1
    positions.masked_fill_(attention == 0, 0)
    return {
        "rows": [value["row"] for value in values],
        "input_ids": input_ids,
        "attention_mask": attention,
        "position_ids": positions,
        "document_mask": document_mask,
    }


def selected_choice_head(model: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model.get_output_embeddings()
    tokenizer = getattr(model, "_document_path_tokenizer")
    ids = []
    for choice in CHOICES:
        values = tokenizer.encode(choice, add_special_tokens=False)
        if len(values) != 1:
            raise RuntimeError(f"Choice is not a single continuation token: {choice} -> {values}")
        ids.append(values[0])
    index = torch.tensor(ids, device=output.weight.device)
    weight = output.weight.index_select(0, index).detach()
    bias = output.bias.index_select(0, index).detach() if getattr(output, "bias", None) is not None else None
    return weight, bias


def forward_choice_logits(
    model: Any,
    adapter: DocumentPathAdapter,
    batch: dict[str, Any],
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    document_mask = batch["document_mask"].to(device, non_blocking=True)
    adapter.set_document_mask(document_mask)
    output = model.model(
        input_ids=batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        position_ids=batch["position_ids"].to(device, non_blocking=True),
        use_cache=False,
        return_dict=True,
    )
    return F.linear(output.last_hidden_state[:, -1], choice_weight, choice_bias).float()


def batch_targets(rows: Sequence[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gold = torch.tensor([CHOICES.index(row["gold_answer"]) for row in rows], device=device)
    no_logits = torch.tensor([row["frozen_no_rag"]["choice_logits"] for row in rows], device=device, dtype=torch.float32)
    no_probs = torch.tensor([row["frozen_no_rag"]["choice_probabilities"] for row in rows], device=device, dtype=torch.float32)
    return gold, no_logits, no_probs


def losses(
    logits: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    positive, negative, swap = logits[:batch_size], logits[batch_size : 2 * batch_size], logits[2 * batch_size :]
    gold, no_logits, no_probs = batch_targets(rows, device)
    positive_margin = gold_margins(positive, gold)
    negative_margin = gold_margins(negative, gold)
    swap_margin = gold_margins(swap, gold)
    no_margin = gold_margins(no_logits, gold)
    preference_negative = F.relu(args.preference_margin - (positive_margin - negative_margin)).mean()
    preference_swap = F.relu(args.preference_margin - (positive_margin - swap_margin)).mean()
    positive_baseline = F.relu(args.positive_baseline_margin - (positive_margin - no_margin)).mean()
    negative_invariance = F.kl_div(F.log_softmax(negative, dim=-1), no_probs, reduction="batchmean")
    swap_invariance = F.kl_div(F.log_softmax(swap, dim=-1), no_probs, reduction="batchmean")
    total = (
        args.preference_weight * 0.5 * (preference_negative + preference_swap)
        + args.positive_baseline_weight * positive_baseline
        + args.negative_invariance_weight * negative_invariance
        + args.swap_invariance_weight * swap_invariance
    )
    return {
        "loss": total,
        "preference_negative": preference_negative,
        "preference_swap": preference_swap,
        "positive_baseline": positive_baseline,
        "negative_invariance": negative_invariance,
        "swap_invariance": swap_invariance,
    }


@torch.inference_mode()
def evaluate(
    model: Any,
    adapter: DocumentPathAdapter,
    loader: DataLoader[Any],
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    args: argparse.Namespace,
    progress: HierarchicalProgress | None = None,
) -> dict[str, Any]:
    model.eval()
    device = torch.device(args.device)
    totals: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    subgroup: dict[str, Counter[str]] = {stratum: Counter() for stratum in STRATA}
    for batch in loader:
        logits = forward_choice_logits(model, adapter, batch, choice_weight, choice_bias, device)
        batch_size = len(batch["rows"])
        positive, negative, swap = logits[:batch_size], logits[batch_size : 2 * batch_size], logits[2 * batch_size :]
        gold, no_logits, no_probs = batch_targets(batch["rows"], device)
        pm = gold_margins(positive, gold)
        nm = gold_margins(negative, gold)
        sm = gold_margins(swap, gold)
        m0 = gold_margins(no_logits, gold)
        negative_kl = F.kl_div(F.log_softmax(negative, dim=-1), no_probs, reduction="none").sum(-1)
        swap_kl = F.kl_div(F.log_softmax(swap, dim=-1), no_probs, reduction="none").sum(-1)
        for index, row in enumerate(batch["rows"]):
            values = {
                "positive_gt_negative": float(pm[index] > nm[index]),
                "positive_gt_swap": float(pm[index] > sm[index]),
                "both_preferences": float(pm[index] > nm[index] and pm[index] > sm[index]),
                "positive_gt_no_rag": float(pm[index] > m0[index]),
                "negative_kl": float(negative_kl[index]),
                "swap_kl": float(swap_kl[index]),
                "positive_margin": float(pm[index]),
                "negative_margin": float(nm[index]),
                "swap_margin": float(sm[index]),
            }
            for key, value in values.items():
                sums[key] += value
            totals["questions"] += 1
            current = subgroup[row["stratum"]]
            current["questions"] += 1
            current["positive_gt_negative"] += values["positive_gt_negative"]
            current["positive_gt_swap"] += values["positive_gt_swap"]
        if progress is not None:
            progress.update(batch_size)
    count = totals["questions"]
    result = {key: float(value) / count for key, value in sums.items()}
    result["questions"] = count
    result["subgroups"] = {
        name: {
            "questions": int(values["questions"]),
            "positive_gt_negative_accuracy": float(values["positive_gt_negative"]) / max(1, values["questions"]),
            "positive_gt_swap_accuracy": float(values["positive_gt_swap"]) / max(1, values["questions"]),
        }
        for name, values in subgroup.items()
    }
    return result


@torch.inference_mode()
def no_rag_logit_error(
    model: Any,
    adapter: DocumentPathAdapter,
    sample_values: Sequence[dict[str, Any]],
    tokenizer: Any,
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    args: argparse.Namespace,
    progress: HierarchicalProgress | None = None,
) -> float:
    device = torch.device(args.device)
    no_sequences = []
    no_cached = []
    for value in sample_values:
        row = value["row"]
        sample = make_sample(
            {
                "row_idx": row["row_idx"], "sample_id": row["sample_id"], "dataset": row["dataset"],
                "split": row["split"], "question": row["question"], "options": row["options"],
                "answer": row["gold_answer"],
            }
        )
        ids, _prompt = sequence_for_order(tokenizer, sample, None, "document_first")
        no_sequences.append(ids)
        no_cached.append(row["frozen_no_rag"]["choice_logits"])
    input_ids, attention, positions = pad_left(no_sequences, int(tokenizer.pad_token_id), device)
    adapter.set_document_mask(torch.zeros_like(attention, dtype=torch.bool))
    output = model.model(
        input_ids=input_ids, attention_mask=attention, position_ids=positions,
        use_cache=False, return_dict=True,
    )
    no_logits = F.linear(output.last_hidden_state[:, -1], choice_weight, choice_bias).float()
    no_cached_tensor = torch.tensor(no_cached, device=device)
    if progress is not None:
        progress.update(len(sample_values))
    return float((no_logits - no_cached_tensor).abs().max())


@torch.inference_mode()
def fidelity_audit(
    model: Any,
    adapter: DocumentPathAdapter,
    dataset: EncodedDataset,
    tokenizer: Any,
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    args: argparse.Namespace,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    device = torch.device(args.device)
    sample_values = dataset.values[: min(32, len(dataset))]
    loader = DataLoader(
        sample_values,
        batch_size=min(args.batch_size, len(sample_values)),
        shuffle=False,
        collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
    )
    max_document_error = 0.0
    for batch in loader:
        logits = forward_choice_logits(model, adapter, batch, choice_weight, choice_bias, device)
        size = len(batch["rows"])
        positive, negative = logits[:size], logits[size : 2 * size]
        cached_positive = torch.tensor(
            [row["positive"]["frozen_choice_logits"] for row in batch["rows"]], device=device
        )
        cached_negative = torch.tensor(
            [row["negative"]["frozen_choice_logits"] for row in batch["rows"]], device=device
        )
        max_document_error = max(
            max_document_error,
            float((positive - cached_positive).abs().max()),
            float((negative - cached_negative).abs().max()),
        )
        progress.update(size)

    no_rag_error = no_rag_logit_error(
        model, adapter, sample_values, tokenizer, choice_weight, choice_bias, args, progress
    )
    audit = adapter.audit()
    if audit["unexpected_trainable_parameters"]:
        raise RuntimeError(f"Unexpected trainable parameters: {audit['unexpected_trainable_parameters'][:5]}")
    if audit["max_non_document_delta"] != 0.0:
        raise RuntimeError(f"Non-document adapter delta is non-zero: {audit['max_non_document_delta']}")
    if max(max_document_error, no_rag_error) > args.base_logit_tolerance:
        raise RuntimeError(
            f"Frozen-base fidelity failed: document={max_document_error:.6f} no_rag={no_rag_error:.6f} "
            f"tolerance={args.base_logit_tolerance}"
        )
    return {**audit, "max_document_logit_error": max_document_error, "max_no_rag_logit_error": no_rag_error}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.questions != 256:
        raise ValueError("This preregistered mechanism diagnostic is fixed to 256 questions")
    if min(args.epochs, args.batch_size, args.lora_rank) <= 0:
        raise ValueError("epochs, batch size, and LoRA rank must be positive")
    if not args.prepare_only and (
        not torch.cuda.is_available() or torch.device(args.device).type != "cuda"
    ):
        raise RuntimeError("The document-path overfit diagnostic requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_root = args.data_root / args.dataset
    source_manifest_path = source_root / "training_dataset/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_contract_sha256 = str(source_manifest["contract_sha256"])
    train_source = source_root / "training_dataset/train.jsonl"
    source_questions = int(source_manifest["split_summary"]["train"]["questions"])
    prepared_root = source_root / "document_path_overfit256_v1"
    prepared_path = prepared_root / "train.jsonl"
    prepared_manifest_path = prepared_root / "manifest.json"
    output_dir = args.output_root / args.dataset / args.run_name
    contract = {
        "run_version": RUN_VERSION,
        "purpose": "Tiny-set identifiability test for document-token K/V-only LoRA",
        "not_a_generalization_result": True,
        "dataset": args.dataset,
        "source_manifest": file_identity(source_manifest_path, content_hash=True),
        "source_contract_sha256": source_contract_sha256,
        "questions": args.questions,
        "strata": list(STRATA),
        "model_config": file_identity(args.model_name_or_path / "config.json", content_hash=True),
        "prompt_order": "documents_then_question_options",
        "trainable_path": "document-token positions of every Llama k_proj and v_proj only",
        "gold_and_semantic_labels_in_prompt": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "loss": {
            "preference_margin": args.preference_margin,
            "positive_baseline_margin": args.positive_baseline_margin,
            "preference_weight": args.preference_weight,
            "positive_baseline_weight": args.positive_baseline_weight,
            "negative_invariance_weight": args.negative_invariance_weight,
            "swap_invariance_weight": args.swap_invariance_weight,
        },
        "pass_threshold": {
            "positive_gt_negative_accuracy": args.early_stop_accuracy,
            "positive_gt_swap_accuracy": args.early_stop_accuracy,
            "consecutive_epochs": args.early_stop_epochs,
            "no_rag_exact_path": True,
            "max_non_document_delta": 0.0,
        },
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "seed": args.seed,
    }
    contract_hash = fingerprint(contract)
    contract_path = output_dir / "run_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract_hash:
            raise RuntimeError("Overfit run contract mismatch; use a new --run-name")
    else:
        atomic_json(
            contract_path,
            {
                "contract_sha256": contract_hash, "created_at": utc_now(),
                "code_commit": git_commit(), "code_sha256": sha256_file(Path(__file__)),
                "contract": contract,
            },
        )

    passes_per_epoch = args.questions * 2
    stage_names = (
        "audit 20k train cache and select 256 questions",
        "tokenize three document-first conditions",
        "verify frozen-base and document-mask fidelity",
        "overfit document-token K/V adapters",
        "final same-set mechanism evaluation",
    )
    stage_estimates = (20.0, 20.0, 45.0, 2400.0, 30.0)
    progress = HierarchicalProgress(stage_names, stage_estimates)
    progress.log(
        f"[workflow plan] diagnostic=overfit-only dataset={args.dataset} questions={args.questions} "
        f"epochs<={args.epochs} batch={args.batch_size} output={output_dir}"
    )
    try:
        progress.start_stage(1, source_questions, "question")
        selected = materialize_overfit_data(
            args, train_source, source_questions, prepared_path, prepared_manifest_path,
            source_contract_sha256, progress,
        )
        progress.complete_stage(f"selected={len(selected)} data={prepared_path}")
        if args.prepare_only:
            progress.finish("prepare-only: no model was loaded")
            return

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        progress.start_stage(2, len(selected), "question")
        dataset = EncodedDataset(selected, tokenizer, args.max_input_tokens, progress)
        progress.complete_stage(f"max_tokens={dataset.max_tokens}")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
        )
        eval_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
        )

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
        logging.info("Loading frozen Llama for document-path LoRA: %s", args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, low_cpu_mem_usage=True,
            dtype=dtype, attn_implementation=args.attn_implementation,
        ).to(args.device)
        model._document_path_tokenizer = tokenizer
        model.config.use_cache = False
        adapter = DocumentPathAdapter(
            model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
        )
        choice_weight, choice_bias = selected_choice_head(model)
        adapter_audit = adapter.audit()
        logging.info("Adapter plan: %s", json.dumps(adapter_audit, ensure_ascii=False))

        progress.start_stage(3, min(32, len(dataset)) * 2, "question-condition-set")
        fidelity = fidelity_audit(
            model, adapter, dataset, tokenizer, choice_weight, choice_bias, args, progress
        )
        progress.complete_stage(f"fidelity={json.dumps(fidelity, ensure_ascii=False)}")

        parameters = list(adapter.trainable_parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        checkpoint_path = output_dir / "checkpoint.pt"
        history: list[dict[str, Any]] = []
        start_epoch = 1
        consecutive_passes = 0
        if args.resume and checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("contract_sha256") != contract_hash:
                raise RuntimeError("Checkpoint contract mismatch")
            adapter.load_adapter_state_dict(checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(args.device)
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["epoch"]) + 1
            consecutive_passes = int(checkpoint.get("consecutive_passes", 0))
            logging.info("Resuming overfit diagnostic at epoch %d", start_epoch)

        progress.start_stage(4, args.epochs * passes_per_epoch, "question-pass")
        progress.set_initial((start_epoch - 1) * passes_per_epoch)
        device = torch.device(args.device)
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_sums: Counter[str] = Counter()
            trained = 0
            progress.log(f"[stage 4/5 | train epoch {epoch}/{args.epochs}] starting")
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = forward_choice_logits(model, adapter, batch, choice_weight, choice_bias, device)
                current = losses(logits, batch["rows"], args, device)
                current["loss"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                count = len(batch["rows"])
                trained += count
                for key, value in current.items():
                    train_sums[key] += float(value.detach()) * count
                progress.update(count)
            metrics = evaluate(
                model, adapter, eval_loader, choice_weight, choice_bias, args, progress
            )
            epoch_row = {
                "epoch": epoch,
                "train": {key: value / trained for key, value in train_sums.items()},
                "same_set": metrics,
            }
            history.append(epoch_row)
            passed = (
                metrics["positive_gt_negative"] >= args.early_stop_accuracy
                and metrics["positive_gt_swap"] >= args.early_stop_accuracy
            )
            consecutive_passes = consecutive_passes + 1 if passed else 0
            progress.log(
                f"[epoch {epoch}/{args.epochs}] loss={epoch_row['train']['loss']:.4f} "
                f"D+>D-={metrics['positive_gt_negative']:.4f} "
                f"D+>Dswap={metrics['positive_gt_swap']:.4f} both={metrics['both_preferences']:.4f} "
                f"negative_KL={metrics['negative_kl']:.5f} swap_KL={metrics['swap_kl']:.5f} "
                f"pass_streak={consecutive_passes}/{args.early_stop_epochs}"
            )
            atomic_torch(
                checkpoint_path,
                {
                    "contract_sha256": contract_hash,
                    "epoch": epoch,
                    "adapter": adapter.adapter_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "consecutive_passes": consecutive_passes,
                },
            )
            if consecutive_passes >= args.early_stop_epochs:
                logging.info("Overfit pass threshold held for %d epochs; stopping", consecutive_passes)
                break
        progress.complete_stage(f"epochs_completed={history[-1]['epoch'] if history else 0}")

        final_no_rag_sample = dataset.values[: min(32, len(dataset))]
        progress.start_stage(5, len(dataset) + len(final_no_rag_sample), "question")
        final_metrics = evaluate(
            model, adapter, eval_loader, choice_weight, choice_bias, args, progress
        )
        final_no_rag_error = no_rag_logit_error(
            model, adapter, final_no_rag_sample, tokenizer, choice_weight, choice_bias, args, progress
        )
        final_fidelity = adapter.audit()
        final_pass = bool(
            final_metrics["positive_gt_negative"] >= args.early_stop_accuracy
            and final_metrics["positive_gt_swap"] >= args.early_stop_accuracy
            and final_fidelity["max_non_document_delta"] == 0.0
            and not final_fidelity["unexpected_trainable_parameters"]
            and final_no_rag_error <= args.base_logit_tolerance
        )
        final = {
            "run_version": RUN_VERSION,
            "contract_sha256": contract_hash,
            "completed_at": utc_now(),
            "purpose": "same-set overfit/identifiability test only",
            "generalization_claim": False,
            "passed": final_pass,
            "pass_threshold": contract["pass_threshold"],
            "frozen_base_fidelity": fidelity,
            "final_adapter_audit": final_fidelity,
            "final_no_rag_max_logit_error": final_no_rag_error,
            "final_same_set_metrics": final_metrics,
            "history": history,
            "adapter_path": str((output_dir / "final_adapter.pt").resolve()),
            "next_action": (
                "run held-out bounded pilot" if final_pass
                else "stop scaling and diagnose document mask/loss/adapter capacity"
            ),
        }
        atomic_torch(
            output_dir / "final_adapter.pt",
            {"contract_sha256": contract_hash, "adapter": adapter.adapter_state_dict()},
        )
        atomic_json(output_dir / "summary.json", final)
        progress.complete_stage(f"passed={final_pass} summary={output_dir/'summary.json'}")
        progress.finish(f"passed={final_pass} next={final['next_action']}")
    except Exception:
        progress.log(
            f"[workflow FAILED] active_stage={progress.stage_index}/{progress.stage_count} "
            f"completed={progress.stage_done}/{progress.stage_total}; durable data={prepared_root} "
            f"checkpoint={output_dir/'checkpoint.pt'}; rerun the identical command to resume"
        )
        raise


if __name__ == "__main__":
    main()

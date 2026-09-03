#!/usr/bin/env python3
"""Train and compare evidence-counterfactual and ordinary-SFT document-path LoRAs.

The unit of supervision is a same-question triplet:

* full: the complete Direct Support document;
* evidence_removed: only the annotated evidence sentence(s) are removed;
* control_removed: a word-count-matched non-evidence sentence subset is removed.

Only K/V LoRA deltas at document-token positions are trainable.  This makes the
No-RAG path exactly identical to the frozen Llama and prevents question-only
domain fine-tuning from satisfying the objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
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
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import make_sample  # noqa: E402
from evaluate_rag2_document_first_prompt_order import (  # noqa: E402
    HierarchicalProgress,
    sequence_for_order,
)
from evaluate_rag2_evidence_sentence_causal_effect import variants  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from medrag.training.document_path_lora import DocumentPathAdapter  # noqa: E402
from medrag.training.semantic_behavior_lora import gold_margins  # noqa: E402
from train_rag2_document_path_overfit import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    atomic_torch,
    file_identity,
    fingerprint,
    selected_choice_head,
    sha256_file,
)


RUN_VERSION = "rag2_evidence_counterfactual_lora_v5"
DATA_VERSION = "rag2_evidence_counterfactual_direct_support_v3"
CONDITIONS = ("full", "evidence_removed", "control_removed")
GROUPS = ("dependence_demo", "rescue")
SPLITS = ("train", "val", "test")
BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa",), default="medmcqa")
    parser.add_argument("--candidate-root", type=Path, default=BASE / "evidence_ablation_candidates_strict_v1")
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=BASE / "evidence_sentence_causal_audit_direct_choice_document_first_v2",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=BASE / "evidence_counterfactual_direct_support_v3",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "models/RAG2-Evidence-Counterfactual-LoRA")
    parser.add_argument("--run-name", default="medmcqa_direct_support_stable_all_v5")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--pair-improvement", type=float, default=0.5)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--removed-anchor-weight", type=float, default=0.5)
    parser.add_argument("--control-consistency-weight", type=float, default=0.1)
    parser.add_argument("--minimum-differential-margin", type=float, default=1.0)
    parser.add_argument("--maximum-control-word-difference", type=float, default=0.25)
    parser.add_argument(
        "--minimum-top1-gap",
        type=float,
        default=0.125,
        help="Exclude conditions whose cached top-1 versus top-2 logit gap is at or below this value.",
    )
    parser.add_argument("--demo-per-batch", type=int, default=6)
    parser.add_argument("--rescue-per-batch", type=int, default=2)
    parser.add_argument("--tiny-demo", type=int, default=96)
    parser.add_argument("--tiny-rescue", type=int, default=32)
    parser.add_argument("--tiny-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--expected-train-pairs", type=int, default=12949)
    parser.add_argument("--base-logit-tolerance", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager",), default="eager")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Materialize and tokenize every split, then exit before loading the GPU model.",
    )
    parser.add_argument(
        "--gradient-smoke-only",
        action="store_true",
        help="Run preflight, frozen-cache fidelity, and one counterfactual optimizer step, then exit.",
    )
    parser.add_argument(
        "--tiny-only",
        action="store_true",
        help="Run through the 128-pair overfit gate, then exit before held-out/full training.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def parse_question_with_options(text: str) -> tuple[str, dict[str, str]]:
    lines = str(text).splitlines()
    option_positions = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) >= 3 and stripped[0] in CHOICES and stripped[1] == ")":
            option_positions.append((index, stripped[0], stripped[2:].strip()))
    if [item[1] for item in option_positions] != list(CHOICES):
        raise ValueError(f"Could not parse four canonical options: {text[:160]!r}")
    first = option_positions[0][0]
    question = "\n".join(lines[:first]).strip()
    options = {choice: value for _, choice, value in option_positions}
    if not question or any(not value for value in options.values()):
        raise ValueError("Empty question or option")
    return question, options


def group_for_audit(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    effects = row["effects"]
    if float(row["control_relative_word_difference"]) > args.maximum_control_word_difference:
        return None
    if float(effects["differential_gold_margin"]) < args.minimum_differential_margin:
        return None
    if any(
        float(row["conditions"][condition]["top1_margin"]) <= args.minimum_top1_gap
        for condition in ("full", "evidence_removed", "matched_control_removed")
    ):
        return None
    evidence_transition = str(effects["full_to_evidence_removed_transition"])
    control_transition = str(effects["full_to_control_removed_transition"])
    if evidence_transition == "C2W" and control_transition == "C2C":
        return "dependence_demo"
    if evidence_transition == "W2W" and control_transition == "W2C":
        return "rescue"
    return None


def outcome_shards(args: argparse.Namespace, split: str) -> list[Path]:
    root = args.audit_root / "outcome_shards/direct_support" / args.dataset / split
    paths = sorted(root.glob("shard_*/rows.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No outcome shards: {root}")
    return paths


def condition_arrays(shard: Path) -> tuple[torch.Tensor, torch.Tensor]:
    scores = load_safetensors(str(shard.with_name("scores.safetensors")), device="cpu")
    return scores["choice_logits"].float(), scores["choice_probabilities"].float()


def prepare_splits(
    args: argparse.Namespace,
    progress: HierarchicalProgress,
    source_counts: dict[str, int],
    preparation_contract: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    root = args.prepared_root / args.dataset
    manifest_path = root / "manifest.json"
    output_paths = {split: root / f"{split}.jsonl" for split in SPLITS}
    if args.resume and manifest_path.is_file() and all(path.is_file() for path in output_paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("preparation_contract_sha256") == preparation_contract
            and all(manifest["splits"][split]["sha256"] == sha256_file(output_paths[split]) for split in SPLITS)
        ):
            progress.set_initial(sum(source_counts.values()) * 2)
            return {split: list(iter_jsonl(output_paths[split])) for split in SPLITS}, manifest
        raise RuntimeError(
            "Prepared-data contract mismatch; use a new --prepared-root instead of overwriting existing data"
        )

    prepared: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    progress_offset = 0
    for split in SPLITS:
        selected: dict[str, dict[str, Any]] = {}
        observed = 0
        for shard in outcome_shards(args, split):
            logits, probabilities = condition_arrays(shard)
            rows = list(iter_jsonl(shard))
            if len(rows) != logits.shape[0] or logits.shape != probabilities.shape:
                raise RuntimeError(f"Outcome tensor/row mismatch: {shard}")
            for row in rows:
                observed += 1
                group = group_for_audit(row, args)
                if group is not None:
                    tensor_row = int(row["tensor_row"])
                    order = list(row["condition_order"])
                    selected[str(row["pair_id"])] = {
                        "group": group,
                        "audit": row,
                        "base_logits": {
                            name: logits[tensor_row, order.index(name)].tolist() for name in order
                        },
                        "base_probabilities": {
                            name: probabilities[tensor_row, order.index(name)].tolist() for name in order
                        },
                    }
                if observed % 2048 == 0:
                    progress.set_absolute(progress_offset + observed)
        if observed != source_counts[split]:
            raise RuntimeError(f"Audit count mismatch {split}: {observed} != {source_counts[split]}")
        progress_offset += source_counts[split]

        candidate_path = args.candidate_root / args.dataset / f"{split}.jsonl"
        values: list[dict[str, Any]] = []
        candidate_observed = 0
        for candidate in iter_jsonl(candidate_path):
            candidate_observed += 1
            pair_id = str(candidate["pair_id"])
            chosen = selected.get(pair_id)
            if chosen is not None:
                documents, details = variants(candidate)
                audit = chosen["audit"]
                for key in ("canonical_document_sha256", "evidence_removed_sha256", "control_removed_sha256"):
                    if details[key] != audit[key]:
                        raise RuntimeError(f"Variant replay mismatch {pair_id}: {key}")
                question, options = parse_question_with_options(candidate["question_with_options"])
                values.append(
                    {
                        "data_version": DATA_VERSION,
                        "pair_id": pair_id,
                        "sample_id": str(candidate["sample_id"]),
                        "row_idx": int(candidate["row_idx"]),
                        "dataset": str(candidate["dataset"]),
                        "split": split,
                        "source": str(candidate["source"]),
                        "doc_rank": int(candidate["doc_rank"]),
                        "group": chosen["group"],
                        "question": question,
                        "options": options,
                        "gold_answer": str(candidate["answer"]),
                        "documents": {
                            "full": documents["full"],
                            "evidence_removed": documents["evidence_removed"],
                            "control_removed": documents["matched_control_removed"],
                        },
                        "base_logits": {
                            "full": chosen["base_logits"]["full"],
                            "evidence_removed": chosen["base_logits"]["evidence_removed"],
                            "control_removed": chosen["base_logits"]["matched_control_removed"],
                        },
                        "base_probabilities": {
                            "full": chosen["base_probabilities"]["full"],
                            "evidence_removed": chosen["base_probabilities"]["evidence_removed"],
                            "control_removed": chosen["base_probabilities"]["matched_control_removed"],
                        },
                        "selection": {
                            "differential_gold_margin": float(audit["effects"]["differential_gold_margin"]),
                            "control_relative_word_difference": float(audit["control_relative_word_difference"]),
                            "evidence_transition": str(audit["effects"]["full_to_evidence_removed_transition"]),
                            "control_transition": str(audit["effects"]["full_to_control_removed_transition"]),
                        },
                    }
                )
            if candidate_observed % 2048 == 0:
                progress.set_absolute(progress_offset + candidate_observed)
        if candidate_observed != source_counts[split]:
            raise RuntimeError(f"Candidate count mismatch {split}: {candidate_observed} != {source_counts[split]}")
        if len(values) != len(selected):
            missing = sorted(set(selected) - {row["pair_id"] for row in values})[:3]
            raise RuntimeError(f"Selected join mismatch {split}: selected={len(selected)} joined={len(values)} missing={missing}")
        values.sort(key=lambda row: (row["sample_id"], row["pair_id"]))
        atomic_jsonl(output_paths[split], values)
        counts = Counter(row["group"] for row in values)
        summaries[split] = {
            "source_pairs": source_counts[split],
            "selected_pairs": len(values),
            "selected_questions": len({row["sample_id"] for row in values}),
            "groups": dict(sorted(counts.items())),
            "group_questions": {
                group: len({row["sample_id"] for row in values if row["group"] == group})
                for group in GROUPS
            },
            "sha256": sha256_file(output_paths[split]),
        }
        prepared[split] = values
        progress_offset += source_counts[split]
    ids = {split: {row["sample_id"] for row in values} for split, values in prepared.items()}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = ids[left] & ids[right]
            if overlap:
                raise RuntimeError(f"Question leakage {left}/{right}: {sorted(overlap)[:3]}")
    manifest = {
        "data_version": DATA_VERSION,
        "created_at": utc_now(),
        "preparation_contract_sha256": preparation_contract,
        "selection": {
            "semantic_label": "direct_support",
            "minimum_differential_gold_margin": args.minimum_differential_margin,
            "maximum_control_relative_word_difference": args.maximum_control_word_difference,
            "minimum_top1_gap": args.minimum_top1_gap,
            "dependence_demo": "full C, evidence-removed W, control-removed C",
            "rescue": "full W, evidence-removed W, control-removed C",
        },
        "question_overlap_across_splits": 0,
        "splits": summaries,
    }
    atomic_json(manifest_path, manifest)
    return prepared, manifest


def encode_document_prompt(tokenizer: Any, row: dict[str, Any], document: str) -> dict[str, Any]:
    sample = make_sample(
        {
            "row_idx": row["row_idx"],
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["gold_answer"],
        }
    )
    token_ids, prompt = sequence_for_order(tokenizer, sample, document, "document_first")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != token_ids:
        raise RuntimeError(f"Prompt token replay mismatch: {row['pair_id']}")
    marker = "Documents:\n"
    start = prompt.find(marker)
    if start < 0:
        raise RuntimeError(f"Missing document marker: {row['pair_id']}")
    start += len(marker)
    end = start + len(document)
    if prompt[start:end] != document:
        raise RuntimeError(f"Document span mismatch: {row['pair_id']}")
    mask = [bool(right > start and left < end) for left, right in encoded["offset_mapping"]]
    if not any(mask):
        raise RuntimeError(f"Empty document token mask: {row['pair_id']}")
    return {"input_ids": token_ids, "document_mask": mask}


class CounterfactualDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_tokens: int,
        progress: HierarchicalProgress,
        progress_offset: int,
    ) -> None:
        self.values: list[dict[str, Any]] = []
        self.max_tokens = 0
        for index, row in enumerate(rows, 1):
            conditions = {
                condition: encode_document_prompt(tokenizer, row, row["documents"][condition])
                for condition in CONDITIONS
            }
            current = max(len(value["input_ids"]) for value in conditions.values())
            if current > max_tokens:
                raise RuntimeError(f"Prompt exceeds {max_tokens} tokens: {row['pair_id']} tokens={current}")
            self.max_tokens = max(self.max_tokens, current)
            self.values.append({"row": row, "conditions": conditions})
            if index % 32 == 0 or index == len(rows):
                progress.set_absolute(progress_offset + index)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.values[index]


def collate(values: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    sequences: list[list[int]] = []
    masks: list[list[bool]] = []
    for condition in CONDITIONS:
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


class GroupBalancedBatchSampler:
    """Use every demonstration once and cycle rescue examples to a 3:1 ratio."""

    def __init__(self, dataset: CounterfactualDataset, demo: int, rescue: int, seed: int, epoch: int) -> None:
        if demo <= 0 or rescue <= 0:
            raise ValueError("Both per-batch group counts must be positive")
        self.demo = [i for i, value in enumerate(dataset.values) if value["row"]["group"] == "dependence_demo"]
        self.rescue = [i for i, value in enumerate(dataset.values) if value["row"]["group"] == "rescue"]
        if not self.demo or not self.rescue:
            raise RuntimeError("Both training groups are required")
        self.demo_per_batch = demo
        self.rescue_per_batch = rescue
        self.seed = seed + epoch * 1009
        self.batch_count = math.ceil(len(self.demo) / self.demo_per_batch)

    def __len__(self) -> int:
        return self.batch_count

    @property
    def example_passes(self) -> int:
        return len(self.demo) + self.batch_count * self.rescue_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        demos = list(self.demo)
        rescues = list(self.rescue)
        rng.shuffle(demos)
        rng.shuffle(rescues)
        rescue_cursor = 0
        for start in range(0, len(demos), self.demo_per_batch):
            batch = demos[start : start + self.demo_per_batch]
            for _ in range(self.rescue_per_batch):
                if rescue_cursor >= len(rescues):
                    rng.shuffle(rescues)
                    rescue_cursor = 0
                batch.append(rescues[rescue_cursor])
                rescue_cursor += 1
            rng.shuffle(batch)
            yield batch


def forward_choice_logits(
    model: Any,
    adapter: DocumentPathAdapter,
    batch: dict[str, Any],
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    mask = batch["document_mask"].to(device, non_blocking=True)
    adapter.set_document_mask(mask)
    output = model.model(
        input_ids=batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        position_ids=batch["position_ids"].to(device, non_blocking=True),
        use_cache=False,
        return_dict=True,
    )
    return F.linear(output.last_hidden_state[:, -1], choice_weight, choice_bias).float()


def jsd_rows(a_logits: torch.Tensor, b_logits: torch.Tensor) -> torch.Tensor:
    a_log = F.log_softmax(a_logits, dim=-1)
    b_log = F.log_softmax(b_logits, dim=-1)
    a = a_log.exp()
    b = b_log.exp()
    midpoint = 0.5 * (a + b)
    log_midpoint = midpoint.clamp_min(1e-12).log()
    return 0.5 * ((a * (a_log - log_midpoint)).sum(-1) + (b * (b_log - log_midpoint)).sum(-1))


def training_losses(
    logits: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    objective: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    full = logits[:batch_size]
    removed = logits[batch_size : 2 * batch_size]
    control = logits[2 * batch_size :]
    gold = torch.tensor([CHOICES.index(row["gold_answer"]) for row in rows], device=device)
    full_ce = F.cross_entropy(full, gold, reduction="none")
    control_ce = F.cross_entropy(control, gold, reduction="none")
    answer = 0.5 * (full_ce + control_ce)
    zero = answer.sum() * 0.0
    if objective == "sft_control":
        return {
            "loss": answer.mean(),
            "answer": answer.mean(),
            "pair": zero,
            "removed_anchor": zero,
            "control_consistency": zero,
        }
    if objective != "counterfactual":
        raise ValueError(objective)
    full_margin = gold_margins(full, gold)
    removed_margin = gold_margins(removed, gold)
    control_margin = gold_margins(control, gold)
    current_gap = 0.5 * (full_margin + control_margin) - removed_margin
    base_logits = torch.tensor(
        [[row["base_logits"][condition] for row in rows] for condition in CONDITIONS],
        device=device,
        dtype=torch.float32,
    )
    base_full, base_removed, base_control = base_logits
    base_gap = 0.5 * (
        gold_margins(base_full, gold) + gold_margins(base_control, gold)
    ) - gold_margins(base_removed, gold)
    pair = F.relu(args.pair_improvement - (current_gap - base_gap))
    removed_target = torch.tensor(
        [row["base_probabilities"]["evidence_removed"] for row in rows],
        device=device,
        dtype=torch.float32,
    )
    removed_anchor = F.kl_div(
        F.log_softmax(removed, dim=-1), removed_target, reduction="none"
    ).sum(-1)
    consistency = jsd_rows(full, control)
    per_example = (
        args.answer_weight * answer
        + args.pair_weight * pair
        + args.removed_anchor_weight * removed_anchor
        + args.control_consistency_weight * consistency
    )
    return {
        "loss": per_example.mean(),
        "answer": answer.mean(),
        "pair": pair.mean(),
        "removed_anchor": removed_anchor.mean(),
        "control_consistency": consistency.mean(),
    }


def clustered_bootstrap(values: dict[str, list[float]], replicates: int, seed: int) -> dict[str, float]:
    question_values = np.asarray([float(np.mean(item)) for item in values.values()], dtype=np.float64)
    if len(question_values) == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "questions": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        positions = rng.integers(0, len(question_values), len(question_values))
        means[index] = question_values[positions].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(question_values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "questions": int(len(question_values)),
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
    output_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model.eval()
    device = torch.device(args.device)
    records: list[dict[str, Any]] = []
    for batch in loader:
        logits = forward_choice_logits(model, adapter, batch, choice_weight, choice_bias, device)
        size = len(batch["rows"])
        full, removed, control = logits[:size], logits[size : 2 * size], logits[2 * size :]
        gold = torch.tensor([CHOICES.index(row["gold_answer"]) for row in batch["rows"]], device=device)
        margins = {
            "full": gold_margins(full, gold),
            "evidence_removed": gold_margins(removed, gold),
            "control_removed": gold_margins(control, gold),
        }
        predictions = {"full": full.argmax(-1), "evidence_removed": removed.argmax(-1), "control_removed": control.argmax(-1)}
        base_logits = torch.tensor(
            [[row["base_logits"][condition] for row in batch["rows"]] for condition in CONDITIONS],
            device=device,
            dtype=torch.float32,
        )
        base_margins = {
            condition: gold_margins(base_logits[index], gold)
            for index, condition in enumerate(CONDITIONS)
        }
        current_gap = 0.5 * (margins["full"] + margins["control_removed"]) - margins["evidence_removed"]
        base_gap = 0.5 * (base_margins["full"] + base_margins["control_removed"]) - base_margins["evidence_removed"]
        removed_target = torch.tensor(
            [row["base_probabilities"]["evidence_removed"] for row in batch["rows"]],
            device=device,
            dtype=torch.float32,
        )
        removed_kl = F.kl_div(F.log_softmax(removed, -1), removed_target, reduction="none").sum(-1)
        consistency = jsd_rows(full, control)
        for index, row in enumerate(batch["rows"]):
            current = {
                "pair_id": row["pair_id"],
                "sample_id": row["sample_id"],
                "group": row["group"],
                "gap": float(current_gap[index]),
                "base_gap": float(base_gap[index]),
                "gap_gain": float(current_gap[index] - base_gap[index]),
                "removed_kl_from_base": float(removed_kl[index]),
                "full_control_jsd": float(consistency[index]),
                "conditions": {},
            }
            for condition in CONDITIONS:
                current["conditions"][condition] = {
                    "prediction": CHOICES[int(predictions[condition][index])],
                    "correct": bool(predictions[condition][index] == gold[index]),
                    "gold_margin": float(margins[condition][index]),
                }
            records.append(current)
            if output_rows is not None:
                output_rows.append(current)
        if progress is not None:
            progress.update(size)
    result: dict[str, Any] = {
        "pairs": len(records),
        "questions": len({row["sample_id"] for row in records}),
        "groups": {},
    }
    for group in ("all", *GROUPS):
        chosen = records if group == "all" else [row for row in records if row["group"] == group]
        if not chosen:
            continue
        gap_by_question: dict[str, list[float]] = defaultdict(list)
        for row in chosen:
            gap_by_question[row["sample_id"]].append(row["gap_gain"])
        result["groups"][group] = {
            "pairs": len(chosen),
            "questions": len(gap_by_question),
            "accuracy": {
                condition: float(np.mean([row["conditions"][condition]["correct"] for row in chosen]))
                for condition in CONDITIONS
            },
            "mean_gold_margin": {
                condition: float(np.mean([row["conditions"][condition]["gold_margin"] for row in chosen]))
                for condition in CONDITIONS
            },
            "mean_gap": float(np.mean([row["gap"] for row in chosen])),
            "mean_base_gap": float(np.mean([row["base_gap"] for row in chosen])),
            "gap_gain_cluster_bootstrap": clustered_bootstrap(
                gap_by_question, args.bootstrap_replicates, args.seed + len(result["groups"])
            ),
            "mean_removed_kl_from_base": float(np.mean([row["removed_kl_from_base"] for row in chosen])),
            "mean_full_control_jsd": float(np.mean([row["full_control_jsd"] for row in chosen])),
            "full_correct_removed_wrong": float(
                np.mean([
                    row["conditions"]["full"]["correct"] and not row["conditions"]["evidence_removed"]["correct"]
                    for row in chosen
                ])
            ),
        }
    return result


def train_epoch(
    model: Any,
    adapter: DocumentPathAdapter,
    dataset: CounterfactualDataset,
    optimizer: torch.optim.Optimizer,
    objective: str,
    epoch: int,
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    tokenizer: Any,
    args: argparse.Namespace,
    progress: HierarchicalProgress,
) -> dict[str, float]:
    sampler = GroupBalancedBatchSampler(
        dataset, args.demo_per_batch, args.rescue_per_batch, args.seed, epoch
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
    )
    model.train()
    parameters = list(adapter.trainable_parameters())
    sums: Counter[str] = Counter()
    observed = 0
    device = torch.device(args.device)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = forward_choice_logits(model, adapter, batch, choice_weight, choice_bias, device)
        losses = training_losses(logits, batch["rows"], objective, args, device)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"Non-finite {objective} loss at epoch {epoch}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        count = len(batch["rows"])
        observed += count
        for key, value in losses.items():
            sums[key] += float(value.detach()) * count
        progress.update(count)
    return {key: value / observed for key, value in sums.items()}


def selected_score(metrics: dict[str, Any]) -> float:
    rescue = metrics["groups"]["rescue"]
    return float(rescue["full_correct_removed_wrong"] + 0.001 * rescue["gap_gain_cluster_bootstrap"]["mean"])


def train_arm(
    objective: str,
    model: Any,
    adapter: DocumentPathAdapter,
    zero_state: dict[str, torch.Tensor],
    datasets: dict[str, CounterfactualDataset],
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    tokenizer: Any,
    args: argparse.Namespace,
    contract_sha256: str,
    output_dir: Path,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    arm_dir = output_dir / objective
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_dir / "checkpoint.pt"
    best_path = arm_dir / "best_adapter.pt"
    adapter.load_adapter_state_dict(zero_state)
    optimizer = torch.optim.AdamW(
        list(adapter.trainable_parameters()), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_score = -float("inf")
    stale = 0
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha256 or checkpoint.get("objective") != objective:
            raise RuntimeError(f"Checkpoint contract mismatch: {checkpoint_path}")
        adapter.load_adapter_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(args.device)
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        stale = int(checkpoint.get("stale", 0))
        progress.log(f"[{objective}] resume epoch={start_epoch} best_epoch={best_epoch}")
    validation_loader = DataLoader(
        datasets["val"], batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
    )
    sampler = GroupBalancedBatchSampler(
        datasets["train"], args.demo_per_batch, args.rescue_per_batch, args.seed, 1
    )
    passes_per_epoch = sampler.example_passes + len(datasets["val"])
    progress.set_initial((start_epoch - 1) * passes_per_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        progress.log(f"[active training | objective={objective} | epoch {epoch}/{args.epochs}]")
        train_metrics = train_epoch(
            model, adapter, datasets["train"], optimizer, objective, epoch,
            choice_weight, choice_bias, tokenizer, args, progress,
        )
        validation = evaluate(
            model, adapter, validation_loader, choice_weight, choice_bias, args, progress
        )
        score = selected_score(validation)
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            atomic_torch(
                best_path,
                {"contract_sha256": contract_sha256, "objective": objective, "epoch": epoch, "adapter": adapter.adapter_state_dict()},
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "selection_score": score,
            "best_epoch": best_epoch,
        }
        history.append(row)
        progress.log(
            f"[{objective} epoch {epoch}] loss={train_metrics['loss']:.4f} "
            f"val_rescue_specific={validation['groups']['rescue']['full_correct_removed_wrong']:.4f} "
            f"val_gap_gain={validation['groups']['all']['gap_gain_cluster_bootstrap']['mean']:+.4f} "
            f"val_removed_acc={validation['groups']['all']['accuracy']['evidence_removed']:.4f} best={best_epoch}"
        )
        atomic_torch(
            checkpoint_path,
            {
                "contract_sha256": contract_sha256,
                "objective": objective,
                "epoch": epoch,
                "adapter": adapter.adapter_state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "stale": stale,
            },
        )
        if stale >= args.early_stopping_patience:
            skipped = (args.epochs - epoch) * passes_per_epoch
            progress.update(skipped)
            progress.log(f"[{objective}] early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break
    if not best_path.is_file():
        raise RuntimeError(f"No best adapter: {objective}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    adapter.load_adapter_state_dict(best["adapter"])
    return {
        "objective": objective,
        "best_epoch": best_epoch,
        "history": history,
        "adapter_audit": adapter.audit(),
    }


def evaluate_arm_test(
    objective: str,
    summary: dict[str, Any],
    model: Any,
    adapter: DocumentPathAdapter,
    dataset: CounterfactualDataset,
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    tokenizer: Any,
    args: argparse.Namespace,
    output_dir: Path,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    test_rows: list[dict[str, Any]] = []
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
    )
    test = evaluate(
        model, adapter, loader, choice_weight, choice_bias, args, progress, test_rows
    )
    arm_dir = output_dir / objective
    atomic_jsonl(arm_dir / "test_predictions.jsonl", test_rows)
    summary = {**summary, "test": test}
    atomic_json(arm_dir / "completed.json", summary)
    return summary


def tiny_overfit_gate(
    model: Any,
    adapter: DocumentPathAdapter,
    zero_state: dict[str, torch.Tensor],
    dataset: CounterfactualDataset,
    choice_weight: torch.Tensor,
    choice_bias: torch.Tensor | None,
    tokenizer: Any,
    args: argparse.Namespace,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    demos = [value for value in dataset.values if value["row"]["group"] == "dependence_demo"][: args.tiny_demo]
    rescues = [value for value in dataset.values if value["row"]["group"] == "rescue"][: args.tiny_rescue]
    tiny = CounterfactualDataset.__new__(CounterfactualDataset)
    tiny.values = demos + rescues
    tiny.max_tokens = max(
        len(condition["input_ids"])
        for value in tiny.values for condition in value["conditions"].values()
    )
    adapter.load_adapter_state_dict(zero_state)
    optimizer = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=2e-4)
    latest: dict[str, Any] = {}
    for epoch in range(1, args.tiny_epochs + 1):
        train_epoch(
            model, adapter, tiny, optimizer, "counterfactual", epoch,
            choice_weight, choice_bias, tokenizer, args, progress,
        )
        loader = DataLoader(
            tiny, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
        )
        latest = evaluate(model, adapter, loader, choice_weight, choice_bias, args)
        rescue = latest["groups"]["rescue"]
        demo = latest["groups"]["dependence_demo"]
        if (
            rescue["full_correct_removed_wrong"] >= 0.80
            and demo["accuracy"]["full"] >= 0.95
            and latest["groups"]["all"]["gap_gain_cluster_bootstrap"]["mean"] >= 0.25
        ):
            remaining_sampler = GroupBalancedBatchSampler(
                tiny, args.demo_per_batch, args.rescue_per_batch, args.seed, epoch
            )
            progress.update((args.tiny_epochs - epoch) * remaining_sampler.example_passes)
            adapter.load_adapter_state_dict(zero_state)
            return {"passed": True, "epoch": epoch, "metrics": latest}
    adapter.load_adapter_state_dict(zero_state)
    return {"passed": False, "epoch": args.tiny_epochs, "metrics": latest}


def compare_test(
    base: dict[str, Any],
    counterfactual: dict[str, Any],
    sft: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_all = base["groups"]["all"]
    cf_all = counterfactual["groups"]["all"]
    sft_all = sft["groups"]["all"]
    base_rescue = base["groups"]["rescue"]
    cf_rescue = counterfactual["groups"]["rescue"]
    sft_rescue = sft["groups"]["rescue"]
    values = {
        "counterfactual_gap_gain": cf_all["gap_gain_cluster_bootstrap"],
        "counterfactual_rescue_specific": cf_rescue["full_correct_removed_wrong"],
        "sft_rescue_specific": sft_rescue["full_correct_removed_wrong"],
        "counterfactual_minus_sft_rescue_specific": (
            cf_rescue["full_correct_removed_wrong"] - sft_rescue["full_correct_removed_wrong"]
        ),
        "counterfactual_removed_accuracy_change": (
            cf_all["accuracy"]["evidence_removed"] - base_all["accuracy"]["evidence_removed"]
        ),
        "counterfactual_demo_full_accuracy_change": (
            counterfactual["groups"]["dependence_demo"]["accuracy"]["full"]
            - base["groups"]["dependence_demo"]["accuracy"]["full"]
        ),
        "sft_gap_gain": sft_all["gap_gain_cluster_bootstrap"],
        "base_rescue_specific": base_rescue["full_correct_removed_wrong"],
    }
    values["passed"] = bool(
        values["counterfactual_gap_gain"]["mean"] >= 0.5
        and values["counterfactual_gap_gain"]["ci95_low"] > 0.0
        and values["counterfactual_rescue_specific"] >= 0.05
        and values["counterfactual_minus_sft_rescue_specific"] >= 0.03
        and abs(values["counterfactual_removed_accuracy_change"]) <= 0.01
        and values["counterfactual_demo_full_accuracy_change"] >= -0.01
    )
    return values


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    base = summary["frozen_test"]["groups"]
    counterfactual = summary["counterfactual"]["test"]["groups"]
    sft = summary["sft_control"]["test"]["groups"]
    rows = []
    for label, group, metric, condition in (
        ("All: full-document accuracy", "all", "accuracy", "full"),
        ("All: evidence-removed accuracy", "all", "accuracy", "evidence_removed"),
        ("All: matched-control accuracy", "all", "accuracy", "control_removed"),
        ("Dependence demo: full-document accuracy", "dependence_demo", "accuracy", "full"),
    ):
        base_value = float(base[group][metric][condition])
        cf_value = float(counterfactual[group][metric][condition])
        sft_value = float(sft[group][metric][condition])
        rows.append((label, base_value, cf_value, sft_value, cf_value - base_value, cf_value - sft_value))
    base_rescue = float(base["rescue"]["full_correct_removed_wrong"])
    cf_rescue = float(counterfactual["rescue"]["full_correct_removed_wrong"])
    sft_rescue = float(sft["rescue"]["full_correct_removed_wrong"])
    rows.append(
        (
            "Rescue: full correct while evidence-removed wrong",
            base_rescue,
            cf_rescue,
            sft_rescue,
            cf_rescue - base_rescue,
            cf_rescue - sft_rescue,
        )
    )
    lines = [
        "# Evidence-counterfactual document-path LoRA",
        "",
        f"- Held-out test: {summary['frozen_test']['pairs']:,} document pairs / "
        f"{summary['frozen_test']['questions']:,} questions",
        f"- Mechanism criterion passed: **{summary['passed']}**",
        f"- Scope: {summary['scope']}",
        "",
        "| Metric | Frozen Llama | Counterfactual LoRA | Matched SFT | CF - Frozen | CF - SFT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {label} | {base_value:.4f} | {cf_value:.4f} | {sft_value:.4f} | "
        f"{cf_delta:+.4f} | {sft_delta:+.4f} |"
        for label, base_value, cf_value, sft_value, cf_delta, sft_delta in rows
    )
    gap = summary["comparison"]["counterfactual_gap_gain"]
    lines.extend(
        [
            "",
            f"Counterfactual mean evidence-gap gain: {gap['mean']:+.4f} "
            f"(question-cluster bootstrap 95% CI {gap['ci95_low']:+.4f} to {gap['ci95_high']:+.4f}).",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not (args.prepare_only or args.preflight_only) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size != args.demo_per_batch + args.rescue_per_batch:
        raise ValueError("batch-size must equal demo-per-batch + rescue-per-batch")
    if min(args.epochs, args.batch_size, args.lora_rank, args.bootstrap_replicates) <= 0:
        raise ValueError("Invalid non-positive training setting")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    candidate_manifest = args.candidate_root / "manifest.json"
    audit_summary = args.audit_root / "summary.json"
    if not candidate_manifest.is_file() or not audit_summary.is_file():
        raise FileNotFoundError("Candidate manifest or completed audit summary is missing")
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    source_counts = {
        split: int(audit["groups"][f"direct_support:{args.dataset}:{split}"]["pairs"])
        for split in SPLITS
    }
    preparation = {
        "data_version": DATA_VERSION,
        "candidate_manifest": file_identity(candidate_manifest, content_hash=True),
        "audit_summary": file_identity(audit_summary, content_hash=True),
        "dataset": args.dataset,
        "selection": {
            "minimum_differential_margin": args.minimum_differential_margin,
            "maximum_control_word_difference": args.maximum_control_word_difference,
            "minimum_top1_gap": args.minimum_top1_gap,
            "transitions": {
                "dependence_demo": ["C2W", "C2C"],
                "rescue": ["W2W", "W2C"],
            },
        },
    }
    preparation_hash = fingerprint(preparation)
    output_dir = args.output_root / args.dataset / args.run_name
    contract = {
        "run_version": RUN_VERSION,
        "purpose": "same-document evidence-sentence counterfactual training versus matched SFT",
        "preparation": preparation,
        "model": file_identity(args.model_name_or_path / "config.json", content_hash=True),
        "prompt": "paper-compatible direct choice, documents then question/options",
        "trainable_path": "all-layer K/V low-rank deltas emitted at document-token positions only",
        "groups_per_batch": {"dependence_demo": args.demo_per_batch, "rescue": args.rescue_per_batch},
        "objectives": ["counterfactual", "sft_control"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "counterfactual_loss": {
            "pair_improvement": args.pair_improvement,
            "answer_weight": args.answer_weight,
            "pair_weight": args.pair_weight,
            "removed_anchor_weight": args.removed_anchor_weight,
            "control_consistency_weight": args.control_consistency_weight,
        },
        "checkpoint_selection": "validation rescue full-correct AND evidence-removed-wrong rate; gap gain breaks ties",
        "test_used_for_selection": False,
        "preflight": {
            "expected_train_pairs": args.expected_train_pairs,
            "max_input_tokens": args.max_input_tokens,
            "base_logit_tolerance": args.base_logit_tolerance,
        },
        "learnability_gate": {
            "demo_pairs": args.tiny_demo,
            "rescue_pairs": args.tiny_rescue,
            "maximum_epochs": args.tiny_epochs,
            "pass": "rescue-specific>=0.80 AND demo-full-accuracy>=0.95 AND mean-gap-gain>=0.25",
        },
        "evaluation": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "early_stopping_patience": args.early_stopping_patience,
            "primary_pass": "gap-gain>=0.5 with CI_low>0; rescue-specific>=5%; versus-SFT>=3pp; removed drift<=1pp; demo drop<=1pp",
        },
        "runtime": {
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "seed": args.seed,
    }
    contract_hash = fingerprint(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "run_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract_hash:
            raise RuntimeError("Training contract mismatch; use a new --run-name")
    else:
        atomic_json(
            contract_path,
            {
                "contract_sha256": contract_hash,
                "created_at": utc_now(),
                "code_commit": git_commit(),
                "code_sha256": sha256_file(Path(__file__)),
                "contract": contract,
            },
        )

    stage_names = (
        "join causal audit with immutable source documents and materialize splits",
        "tokenize all three document-first counterfactual conditions",
        "load frozen Llama and verify cached-logit/document-mask fidelity",
        "128-pair counterfactual overfit gate",
        "measure frozen validation and held-out test baselines",
        "train counterfactual document-path LoRA and select on validation",
        "evaluate selected counterfactual model on held-out test",
        "train matched ordinary-SFT document-path LoRA and select on validation",
        "evaluate selected SFT control on held-out test",
        "compare mechanisms, write report, and freeze completion manifest",
    )
    stage_estimates = (75, 90, 25, 1200, 360, 9000, 300, 9000, 300, 15)
    progress = HierarchicalProgress(stage_names, stage_estimates)
    progress.log(
        f"[workflow plan] dataset={args.dataset} all qualifying pairs; tiny overfit -> frozen baseline -> "
        f"counterfactual {args.epochs} epochs -> matched SFT {args.epochs} epochs -> held-out comparison"
    )
    try:
        progress.start_stage(1, sum(source_counts.values()) * 2, "source-pair")
        rows, prepared_manifest = prepare_splits(
            args, progress, source_counts, preparation_hash
        )
        train_pairs = int(prepared_manifest["splits"]["train"]["selected_pairs"])
        if args.expected_train_pairs and train_pairs != args.expected_train_pairs:
            raise RuntimeError(
                f"Unexpected train pair count: {train_pairs} != {args.expected_train_pairs}"
            )
        progress.complete_stage(
            f"selected={json.dumps({s: prepared_manifest['splits'][s] for s in SPLITS}, ensure_ascii=False)} "
            f"output={args.prepared_root/args.dataset}"
        )
        if args.prepare_only:
            progress.finish("prepare-only; no GPU model loaded")
            return

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        progress.start_stage(2, sum(len(value) for value in rows.values()), "pair")
        datasets: dict[str, CounterfactualDataset] = {}
        offset = 0
        for split in SPLITS:
            datasets[split] = CounterfactualDataset(
                rows[split], tokenizer, args.max_input_tokens, progress, offset
            )
            offset += len(rows[split])
        progress.complete_stage(
            "max_tokens=" + json.dumps({split: data.max_tokens for split, data in datasets.items()})
        )
        if args.preflight_only:
            progress.finish("preflight-only; all split prompts fit and no GPU model was loaded")
            return

        fidelity_values = (
            [value for value in datasets["train"].values if value["row"]["group"] == "dependence_demo"][:16]
            + [value for value in datasets["train"].values if value["row"]["group"] == "rescue"][:16]
        )
        progress.start_stage(3, len(fidelity_values), "pair")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, low_cpu_mem_usage=True,
            dtype=dtype, attn_implementation=args.attn_implementation,
        ).to(args.device)
        model._document_path_tokenizer = tokenizer
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapter = DocumentPathAdapter(
            model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
        )
        zero_state = adapter.adapter_state_dict()
        choice_weight, choice_bias = selected_choice_head(model)
        fidelity_loader = DataLoader(
            fidelity_values, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
        )
        max_error = 0.0
        max_margin_error = 0.0
        prediction_matches = 0
        prediction_total = 0
        prediction_mismatches: list[dict[str, Any]] = []
        model.eval()
        with torch.inference_mode():
            for batch in fidelity_loader:
                logits = forward_choice_logits(
                    model, adapter, batch, choice_weight, choice_bias, torch.device(args.device)
                ).cpu()
                size = len(batch["rows"])
                cached = torch.tensor(
                    [[row["base_logits"][condition] for row in batch["rows"]] for condition in CONDITIONS],
                    dtype=torch.float32,
                ).reshape(3 * size, 4)
                max_error = max(max_error, float((logits - cached).abs().max()))
                gold = torch.tensor(
                    [CHOICES.index(row["gold_answer"]) for row in batch["rows"]] * 3,
                    dtype=torch.long,
                )
                max_margin_error = max(
                    max_margin_error,
                    float((gold_margins(logits, gold) - gold_margins(cached, gold)).abs().max()),
                )
                actual_predictions = logits.argmax(-1)
                cached_predictions = cached.argmax(-1)
                matches = actual_predictions == cached_predictions
                prediction_matches += int(matches.sum())
                prediction_total += int(logits.shape[0])
                for flat_index in torch.where(~matches)[0].tolist():
                    condition_index, row_index = divmod(flat_index, size)
                    prediction_mismatches.append(
                        {
                            "pair_id": batch["rows"][row_index]["pair_id"],
                            "group": batch["rows"][row_index]["group"],
                            "condition": CONDITIONS[condition_index],
                            "cached_logits": cached[flat_index].tolist(),
                            "actual_logits": logits[flat_index].tolist(),
                            "cached_prediction": CHOICES[int(cached_predictions[flat_index])],
                            "actual_prediction": CHOICES[int(actual_predictions[flat_index])],
                        }
                    )
                progress.update(size)
        if max_error > args.base_logit_tolerance:
            raise RuntimeError(f"Frozen cache fidelity failed: max_abs={max_error} tolerance={args.base_logit_tolerance}")
        if prediction_matches != prediction_total:
            raise RuntimeError(
                "Frozen cache prediction mismatch: "
                f"matches={prediction_matches}/{prediction_total} max_abs={max_error} "
                f"details={json.dumps(prediction_mismatches, ensure_ascii=False)}"
            )
        progress.complete_stage(
            f"cached_prediction_agreement={prediction_matches}/{prediction_total} "
            f"max_cached_logit_error={max_error:.6f} max_gold_margin_error={max_margin_error:.6f} "
            f"adapter={adapter.audit()}"
        )

        if args.gradient_smoke_only:
            smoke = CounterfactualDataset.__new__(CounterfactualDataset)
            smoke.values = (
                [value for value in datasets["train"].values if value["row"]["group"] == "dependence_demo"][: args.demo_per_batch]
                + [value for value in datasets["train"].values if value["row"]["group"] == "rescue"][: args.rescue_per_batch]
            )
            smoke.max_tokens = max(
                len(condition["input_ids"])
                for value in smoke.values for condition in value["conditions"].values()
            )
            before = {key: value.clone() for key, value in adapter.adapter_state_dict().items()}
            optimizer = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=args.learning_rate)
            progress.start_stage(4, len(smoke), "pair-pass")
            losses = train_epoch(
                model, adapter, smoke, optimizer, "counterfactual", 1,
                choice_weight, choice_bias, tokenizer, args, progress,
            )
            after = adapter.adapter_state_dict()
            maximum_update = max(float((after[key] - before[key]).abs().max()) for key in before)
            if maximum_update <= 0.0 or not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError(
                    f"Gradient smoke test failed: maximum_update={maximum_update} losses={losses}"
                )
            result = {
                "status": "passed",
                "pairs": len(smoke),
                "losses": losses,
                "maximum_parameter_update": maximum_update,
                "cached_logit_max_abs_error": max_error,
                "cached_gold_margin_max_abs_error": max_margin_error,
                "cached_prediction_agreement": f"{prediction_matches}/{prediction_total}",
            }
            atomic_json(output_dir / "gradient_smoke.json", result)
            progress.complete_stage(
                f"finite loss and nonzero update verified; max_update={maximum_update:.8f}"
            )
            progress.finish(f"gradient-smoke-only passed; report={output_dir/'gradient_smoke.json'}")
            return

        tiny_sampler = GroupBalancedBatchSampler(
            datasets["train"], args.demo_per_batch, args.rescue_per_batch, args.seed, 1
        )
        tiny_passes_per_epoch = args.tiny_demo + math.ceil(args.tiny_demo / args.demo_per_batch) * args.rescue_per_batch
        progress.start_stage(4, args.tiny_epochs * tiny_passes_per_epoch, "pair-pass")
        tiny = tiny_overfit_gate(
            model, adapter, zero_state, datasets["train"], choice_weight, choice_bias,
            tokenizer, args, progress,
        )
        if not tiny["passed"]:
            atomic_json(output_dir / "tiny_overfit_failure.json", tiny)
            raise RuntimeError("Tiny counterfactual overfit gate failed; full training was not started")
        progress.complete_stage(
            f"passed epoch={tiny['epoch']} rescue_specific={tiny['metrics']['groups']['rescue']['full_correct_removed_wrong']:.4f}"
        )
        if args.tiny_only:
            atomic_json(output_dir / "tiny_overfit_pass.json", tiny)
            progress.finish(
                f"tiny-only passed at epoch={tiny['epoch']}; report={output_dir/'tiny_overfit_pass.json'}"
            )
            return

        progress.start_stage(5, len(datasets["val"]) + len(datasets["test"]), "pair")
        adapter.load_adapter_state_dict(zero_state)
        validation_loader = DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
        )
        test_loader = DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda values: collate(values, int(tokenizer.pad_token_id)),
        )
        frozen_validation_rows: list[dict[str, Any]] = []
        frozen_test_rows: list[dict[str, Any]] = []
        frozen_validation = evaluate(
            model, adapter, validation_loader, choice_weight, choice_bias, args, progress,
            frozen_validation_rows,
        )
        frozen_test = evaluate(
            model, adapter, test_loader, choice_weight, choice_bias, args, progress,
            frozen_test_rows,
        )
        atomic_jsonl(output_dir / "frozen_validation_predictions.jsonl", frozen_validation_rows)
        atomic_jsonl(output_dir / "frozen_test_predictions.jsonl", frozen_test_rows)
        progress.complete_stage(
            f"val_pairs={frozen_validation['pairs']} test_pairs={frozen_test['pairs']} "
            f"test_rescue_specific={frozen_test['groups']['rescue']['full_correct_removed_wrong']:.4f}"
        )

        train_sampler = GroupBalancedBatchSampler(
            datasets["train"], args.demo_per_batch, args.rescue_per_batch, args.seed, 1
        )
        passes_per_arm = args.epochs * (train_sampler.example_passes + len(datasets["val"]))
        progress.start_stage(6, passes_per_arm, "pair-pass")
        cf_summary = train_arm(
            "counterfactual", model, adapter, zero_state, datasets, choice_weight, choice_bias,
            tokenizer, args, contract_hash, output_dir, progress,
        )
        progress.complete_stage(
            f"best_epoch={cf_summary['best_epoch']} val-selected checkpoint={output_dir/'counterfactual/best_adapter.pt'}"
        )

        progress.start_stage(7, len(datasets["test"]), "pair")
        cf_summary = evaluate_arm_test(
            "counterfactual", cf_summary, model, adapter, datasets["test"],
            choice_weight, choice_bias, tokenizer, args, output_dir, progress,
        )
        progress.complete_stage(
            f"test_rescue_specific={cf_summary['test']['groups']['rescue']['full_correct_removed_wrong']:.4f} "
            f"test_gap_gain={cf_summary['test']['groups']['all']['gap_gain_cluster_bootstrap']['mean']:+.4f}"
        )

        progress.start_stage(8, passes_per_arm, "pair-pass")
        sft_summary = train_arm(
            "sft_control", model, adapter, zero_state, datasets, choice_weight, choice_bias,
            tokenizer, args, contract_hash, output_dir, progress,
        )
        progress.complete_stage(
            f"best_epoch={sft_summary['best_epoch']} val-selected checkpoint={output_dir/'sft_control/best_adapter.pt'}"
        )

        progress.start_stage(9, len(datasets["test"]), "pair")
        sft_summary = evaluate_arm_test(
            "sft_control", sft_summary, model, adapter, datasets["test"],
            choice_weight, choice_bias, tokenizer, args, output_dir, progress,
        )
        progress.complete_stage(
            f"test_rescue_specific={sft_summary['test']['groups']['rescue']['full_correct_removed_wrong']:.4f} "
            f"test_gap_gain={sft_summary['test']['groups']['all']['gap_gain_cluster_bootstrap']['mean']:+.4f}"
        )

        progress.start_stage(10, 1, "report")
        comparison = compare_test(frozen_test, cf_summary["test"], sft_summary["test"], args)
        summary = {
            "run_version": RUN_VERSION,
            "completed_at": utc_now(),
            "contract_sha256": contract_hash,
            "prepared_manifest": prepared_manifest,
            "tiny_overfit_gate": tiny,
            "frozen_validation": frozen_validation,
            "frozen_test": frozen_test,
            "counterfactual": cf_summary,
            "sft_control": sft_summary,
            "comparison": comparison,
            "passed": comparison["passed"],
            "scope": "single-document Direct Support mechanism/generalization test; not final Top-k RAG accuracy",
            "next_action": (
                "evaluate the selected adapter on broad Direct Support, No Evidence, and Top-k RAG cohorts"
                if comparison["passed"]
                else "stop scaling and inspect which preregistered mechanism criterion failed"
            ),
        }
        atomic_json(output_dir / "summary.json", summary)
        write_summary_markdown(output_dir / "summary.md", summary)
        atomic_json(
            output_dir / "completion.json",
            {"status": "complete", "completed_at": summary["completed_at"], "passed": summary["passed"]},
        )
        progress.update(1)
        progress.complete_stage(
            f"passed={summary['passed']} summary={output_dir/'summary.json'}"
        )
        progress.finish(f"passed={summary['passed']} next={summary['next_action']}")
    except Exception as error:
        progress.log(
            f"[workflow FAILED] stage={progress.stage_index}/{progress.stage_count} "
            f"completed={progress.stage_done}/{progress.stage_total} error={type(error).__name__}: {error}; "
            f"durable_data={args.prepared_root/args.dataset} durable_checkpoints={output_dir}; "
            "rerun the identical command to resume"
        )
        raise


if __name__ == "__main__":
    main()

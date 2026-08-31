#!/usr/bin/env python3
"""LoRA-train Llama on same-question semantic-behavior document pairs.

Every model forward receives at most one document.  Semantic and frozen-target
behavioral measurements are never model inputs: they only determine which
Direct Support (D+) and No-Evidence/Misleading (D-) documents were paired and
provide a frozen No-RAG categorical teacher.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES, encode_to_pre_choice  # noqa: E402
from medrag.training.semantic_behavior_lora import (  # noqa: E402
    gold_margins,
    semantic_behavior_losses,
    transition_name,
)


RUN_VERSION = "rag2_semantic_behavior_single_document_lora_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--pair-root",
        type=Path,
        default=base / "semantic_behavior_single_document_pairs_v1",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/RAG2-SemanticBehavior-LoRA",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--objective",
        choices=("proposed", "question_only", "rag_ce"),
        default="proposed",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--train-pairs-per-batch", type=int, default=2)
    parser.add_argument("--eval-pairs-per-batch", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--preference-margin", type=float, default=0.5)
    parser.add_argument("--positive-loss-weight", type=float, default=1.0)
    parser.add_argument("--preference-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-invariance-weight", type=float, default=0.1)
    parser.add_argument("--no-rag-preservation-weight", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer moments loaded from a CPU checkpoint to the active device."""

    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(pair_dir: Path, split: str) -> list[dict[str, Any]]:
    path = pair_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"Empty prepared split: {path}")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"Prepared split has duplicate questions: {path}")
    return rows


class EncodedPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_tokens: int,
        *,
        progress: PipelineProgress | None = None,
    ) -> None:
        self.values: list[dict[str, Any]] = []
        self.excluded_overlength = 0
        for row in rows:
            normalized = {
                "question": row["question"],
                "options": row["options"],
                "answer": row["gold_answer"],
            }
            positive = encode_to_pre_choice(
                tokenizer,
                normalized,
                str(row["positive"]["document_text"]),
                str(row["positive"]["rationale"]),
            )
            negative = encode_to_pre_choice(
                tokenizer,
                normalized,
                str(row["negative"]["document_text"]),
                str(row["negative"]["rationale"]),
            )
            no_rag = encode_to_pre_choice(
                tokenizer,
                normalized,
                None,
                str(row["no_rag_rationale"]),
            )
            if max(len(positive.input_ids), len(negative.input_ids), len(no_rag.input_ids)) > max_tokens:
                self.excluded_overlength += 1
                if progress is not None:
                    progress.update()
                continue
            self.values.append(
                {
                    "row": row,
                    "positive_ids": positive.input_ids,
                    "negative_ids": negative.input_ids,
                    "no_rag_ids": no_rag.input_ids,
                }
            )
            if progress is not None:
                progress.update()
        if not self.values:
            raise RuntimeError("Every prepared pair exceeded --max-input-tokens")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.values[index]


class PairCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences = (
            [value["positive_ids"] for value in values]
            + [value["negative_ids"] for value in values]
            + [value["no_rag_ids"] for value in values]
        )
        maximum = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), maximum), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[index, maximum - length :] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, maximum - length :] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        gold = torch.tensor(
            [CHOICES.index(str(value["row"]["gold_answer"])) for value in values],
            dtype=torch.long,
        )
        teacher = torch.tensor(
            [value["row"]["frozen_no_rag_choice_probabilities"] for value in values],
            dtype=torch.float32,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "gold_indices": gold,
            "teacher_probabilities": teacher,
            "rows": [value["row"] for value in values],
        }


def choice_token_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    values = []
    for choice in CHOICES:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Choice {choice} is not one token: {encoded}")
        values.append(int(encoded[0]))
    return torch.tensor(values, dtype=torch.long, device=device)


def forward_choice_logits(
    model: Any,
    batch: dict[str, Any],
    choice_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    position_ids = batch["position_ids"].to(device, non_blocking=True)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=1,
    )
    logits = outputs.logits[:, -1].index_select(-1, choice_ids).float()
    count = len(batch["rows"])
    return logits[:count], logits[count : 2 * count], logits[2 * count :]


def objective_losses(
    args: argparse.Namespace,
    positive: torch.Tensor,
    negative: torch.Tensor,
    no_rag: torch.Tensor,
    gold: torch.Tensor,
    teacher: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if args.objective == "proposed":
        return semantic_behavior_losses(
            positive_logits=positive,
            negative_logits=negative,
            no_rag_logits=no_rag,
            frozen_no_rag_probabilities=teacher,
            gold_indices=gold,
            preference_margin=args.preference_margin,
            positive_weight=args.positive_loss_weight,
            preference_weight=args.preference_loss_weight,
            negative_invariance_weight=args.negative_invariance_weight,
            no_rag_preservation_weight=args.no_rag_preservation_weight,
        )
    zero = positive.sum() * 0.0
    if args.objective == "question_only":
        loss = F.cross_entropy(no_rag.float(), gold)
        return {"loss": loss, "question_only_ce": loss, "zero": zero}
    positive_ce = F.cross_entropy(positive.float(), gold)
    negative_ce = F.cross_entropy(negative.float(), gold)
    loss = 0.5 * (positive_ce + negative_ce)
    return {"loss": loss, "positive_ce": positive_ce, "negative_ce": negative_ce, "zero": zero}


def new_metric_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "loss_sum": 0.0,
        "positive_correct": 0,
        "negative_correct": 0,
        "frozen_positive_correct": 0,
        "frozen_negative_correct": 0,
        "no_rag_correct": 0,
        "frozen_no_rag_correct": 0,
        "preference_correct": 0,
        "positive_changed": 0,
        "negative_changed": 0,
        "no_rag_changed": 0,
        "positive_margin_sum": 0.0,
        "negative_margin_sum": 0.0,
        "positive_js_sum": 0.0,
        "negative_js_sum": 0.0,
        "positive_transitions": Counter(),
        "negative_transitions": Counter(),
        "positive_transitions_from_frozen_document": Counter(),
        "negative_transitions_from_frozen_document": Counter(),
        "groups": defaultdict(lambda: {"count": 0, "preference_correct": 0, "positive_correct": 0, "negative_correct": 0}),
        "predictions": [],
    }


def categorical_js(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.clamp_min(1e-8)
    right = right.clamp_min(1e-8)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def update_metrics(
    accumulator: dict[str, Any],
    *,
    rows: Sequence[dict[str, Any]],
    positive: torch.Tensor,
    negative: torch.Tensor,
    no_rag: torch.Tensor,
    gold: torch.Tensor,
    teacher: torch.Tensor,
    loss: float,
    collect_predictions: bool,
) -> None:
    positive_prediction = positive.argmax(dim=-1).cpu()
    negative_prediction = negative.argmax(dim=-1).cpu()
    no_rag_prediction = no_rag.argmax(dim=-1).cpu()
    teacher_prediction = teacher.argmax(dim=-1).cpu()
    gold_cpu = gold.cpu()
    positive_margin = gold_margins(positive, gold).detach().cpu()
    negative_margin = gold_margins(negative, gold).detach().cpu()
    preference = positive_margin > negative_margin
    positive_js = categorical_js(teacher, F.softmax(positive, dim=-1)).detach().cpu()
    negative_js = categorical_js(teacher, F.softmax(negative, dim=-1)).detach().cpu()
    frozen_positive_correct = torch.tensor(
        [bool(row["positive"]["base_correct"]) for row in rows], dtype=torch.bool
    )
    frozen_negative_correct = torch.tensor(
        [bool(row["negative"]["base_correct"]) for row in rows], dtype=torch.bool
    )
    frozen_positive_prediction = torch.tensor(
        [CHOICES.index(str(row["positive"]["base_answer"])) for row in rows],
        dtype=torch.long,
    )
    frozen_negative_prediction = torch.tensor(
        [CHOICES.index(str(row["negative"]["base_answer"])) for row in rows],
        dtype=torch.long,
    )
    count = len(rows)
    accumulator["count"] += count
    accumulator["loss_sum"] += float(loss) * count
    accumulator["positive_correct"] += int((positive_prediction == gold_cpu).sum())
    accumulator["negative_correct"] += int((negative_prediction == gold_cpu).sum())
    accumulator["frozen_positive_correct"] += int(frozen_positive_correct.sum())
    accumulator["frozen_negative_correct"] += int(frozen_negative_correct.sum())
    accumulator["no_rag_correct"] += int((no_rag_prediction == gold_cpu).sum())
    accumulator["frozen_no_rag_correct"] += int((teacher_prediction == gold_cpu).sum())
    accumulator["preference_correct"] += int(preference.sum())
    accumulator["positive_changed"] += int(
        (positive_prediction != frozen_positive_prediction).sum()
    )
    accumulator["negative_changed"] += int((negative_prediction != teacher_prediction).sum())
    accumulator["no_rag_changed"] += int((no_rag_prediction != teacher_prediction).sum())
    accumulator["positive_margin_sum"] += float(positive_margin.sum())
    accumulator["negative_margin_sum"] += float(negative_margin.sum())
    accumulator["positive_js_sum"] += float(positive_js.sum())
    accumulator["negative_js_sum"] += float(negative_js.sum())
    for index, row in enumerate(rows):
        frozen_correct = bool(teacher_prediction[index] == gold_cpu[index])
        positive_correct = bool(positive_prediction[index] == gold_cpu[index])
        negative_correct = bool(negative_prediction[index] == gold_cpu[index])
        accumulator["positive_transitions"][transition_name(frozen_correct, positive_correct)] += 1
        accumulator["negative_transitions"][transition_name(frozen_correct, negative_correct)] += 1
        accumulator["positive_transitions_from_frozen_document"][
            transition_name(bool(frozen_positive_correct[index]), positive_correct)
        ] += 1
        accumulator["negative_transitions_from_frozen_document"][
            transition_name(bool(frozen_negative_correct[index]), negative_correct)
        ] += 1
        group_names = (
            f"pair_group:{row['pair_group']}",
            f"negative_label:{row['negative_semantic_label']}",
            f"no_rag_correct:{str(frozen_correct).lower()}",
        )
        for name in group_names:
            group = accumulator["groups"][name]
            group["count"] += 1
            group["preference_correct"] += int(preference[index])
            group["positive_correct"] += int(positive_correct)
            group["negative_correct"] += int(negative_correct)
        if collect_predictions:
            accumulator["predictions"].append(
                {
                    "sample_id": row["sample_id"],
                    "gold_answer": row["gold_answer"],
                    "pair_group": row["pair_group"],
                    "negative_semantic_label": row["negative_semantic_label"],
                    "positive_prediction": CHOICES[int(positive_prediction[index])],
                    "negative_prediction": CHOICES[int(negative_prediction[index])],
                    "no_rag_prediction": CHOICES[int(no_rag_prediction[index])],
                    "frozen_no_rag_prediction": CHOICES[int(teacher_prediction[index])],
                    "frozen_positive_prediction": CHOICES[int(frozen_positive_prediction[index])],
                    "frozen_negative_prediction": CHOICES[int(frozen_negative_prediction[index])],
                    "positive_margin": float(positive_margin[index]),
                    "negative_margin": float(negative_margin[index]),
                    "negative_js_divergence": float(negative_js[index]),
                }
            )


def finalize_metrics(accumulator: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = int(accumulator["count"])
    if count <= 0:
        raise RuntimeError("Cannot finalize empty metrics")
    groups = {}
    for name, value in accumulator["groups"].items():
        group_count = int(value["count"])
        groups[name] = {
            "count": group_count,
            "preference_accuracy": value["preference_correct"] / group_count,
            "positive_accuracy": value["positive_correct"] / group_count,
            "negative_accuracy": value["negative_correct"] / group_count,
        }
    metrics = {
        "pairs": count,
        "loss": accumulator["loss_sum"] / count,
        "positive_accuracy": accumulator["positive_correct"] / count,
        "negative_accuracy": accumulator["negative_correct"] / count,
        "frozen_positive_accuracy": accumulator["frozen_positive_correct"] / count,
        "frozen_negative_accuracy": accumulator["frozen_negative_correct"] / count,
        "no_rag_accuracy": accumulator["no_rag_correct"] / count,
        "frozen_no_rag_accuracy": accumulator["frozen_no_rag_correct"] / count,
        "preference_accuracy": accumulator["preference_correct"] / count,
        "positive_answer_change_rate_from_frozen_document": accumulator["positive_changed"] / count,
        "negative_answer_change_rate": accumulator["negative_changed"] / count,
        "no_rag_answer_change_rate": accumulator["no_rag_changed"] / count,
        "mean_positive_margin": accumulator["positive_margin_sum"] / count,
        "mean_negative_margin": accumulator["negative_margin_sum"] / count,
        "mean_positive_js_from_frozen_no_rag": accumulator["positive_js_sum"] / count,
        "mean_negative_js_divergence": accumulator["negative_js_sum"] / count,
        "positive_transitions_from_frozen_no_rag": dict(accumulator["positive_transitions"]),
        "negative_transitions_from_frozen_no_rag": dict(accumulator["negative_transitions"]),
        "positive_transitions_from_frozen_document": dict(
            accumulator["positive_transitions_from_frozen_document"]
        ),
        "negative_transitions_from_frozen_document": dict(
            accumulator["negative_transitions_from_frozen_document"]
        ),
        "groups": groups,
    }
    return metrics, accumulator["predictions"]


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    choice_ids: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage: str,
    *,
    collect_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = new_metric_accumulator()
    progress.set_stage(stage, total=len(loader.dataset))
    for batch in loader:
        positive, negative, no_rag = forward_choice_logits(model, batch, choice_ids, device)
        gold = batch["gold_indices"].to(device)
        teacher = batch["teacher_probabilities"].to(device)
        losses = objective_losses(args, positive, negative, no_rag, gold, teacher)
        update_metrics(
            accumulator,
            rows=batch["rows"],
            positive=positive,
            negative=negative,
            no_rag=no_rag,
            gold=gold,
            teacher=teacher,
            loss=float(losses["loss"].detach()),
            collect_predictions=collect_predictions,
        )
        progress.update(len(batch["rows"]))
    return finalize_metrics(accumulator)


def selection_score(metrics: dict[str, Any]) -> float:
    return (
        0.40 * float(metrics["preference_accuracy"])
        + 0.30 * float(metrics["positive_accuracy"])
        + 0.15 * (1.0 - float(metrics["negative_answer_change_rate"]))
        + 0.15 * (1.0 - float(metrics["no_rag_answer_change_rate"]))
    )


def save_predictions(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)
    if args.epochs <= 0 or args.train_pairs_per_batch <= 0 or args.eval_pairs_per_batch <= 0:
        raise ValueError("Epoch and batch sizes must be positive")
    pair_dir = args.pair_root / args.dataset
    pair_manifest_path = pair_dir / "manifest.json"
    if not pair_manifest_path.is_file():
        raise FileNotFoundError(pair_manifest_path)
    raw = {split: load_rows(pair_dir, split) for split in ("train", "val", "test")}
    output_dir = args.output_root / args.dataset / args.run_name
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "objective": args.objective,
        "pair_manifest_sha256": file_sha256(pair_manifest_path),
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "train_pairs_per_batch": args.train_pairs_per_batch,
        "eval_pairs_per_batch": args.eval_pairs_per_batch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "weights": {
            "positive": args.positive_loss_weight,
            "preference": args.preference_loss_weight,
            "negative_invariance": args.negative_invariance_weight,
            "no_rag_preservation": args.no_rag_preservation_weight,
        },
        "preference_margin": args.preference_margin,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(args.lora_target_modules),
        },
        "max_input_tokens": args.max_input_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    logging.info(
        "Semantic-behavior LoRA plan: dataset=%s objective=%s raw=%s output=%s",
        args.dataset,
        args.objective,
        {split: len(rows) for split, rows in raw.items()},
        output_dir,
    )
    if args.plan_only:
        return
    contract_path = output_dir / "contract.json"
    if contract_path.is_file():
        current = json.loads(contract_path.read_text(encoding="utf-8"))
        if current.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Training output contract changed; use a new run name")
    else:
        atomic_json(contract_path, {**contract, "contract_fingerprint": contract_hash})
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    raw_total = sum(len(rows) for rows in raw.values())
    planned_total = (
        raw_total
        + args.epochs * (len(raw["train"]) + len(raw["val"]))
        + len(raw["test"])
    )
    progress = PipelineProgress(
        overall_total=planned_total,
        desc=f"SemanticBehaviorLoRA:{args.dataset}:{args.objective}",
    )
    encoded = {}
    for split, rows in raw.items():
        progress.set_stage(f"1/4 tokenize anchored {split} pairs", total=len(rows))
        encoded[split] = EncodedPairDataset(
            rows,
            tokenizer,
            args.max_input_tokens,
            progress=progress,
        )
    logging.info(
        "Encoded pairs: %s excluded_overlength=%s",
        {split: len(data) for split, data in encoded.items()},
        {split: data.excluded_overlength for split, data in encoded.items()},
    )
    collator = PairCollator(tokenizer.pad_token_id)
    train_generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            encoded["train"],
            batch_size=args.train_pairs_per_batch,
            shuffle=True,
            generator=train_generator,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        ),
        "val": DataLoader(encoded["val"], batch_size=args.eval_pairs_per_batch, shuffle=False, collate_fn=collator),
        "test": DataLoader(encoded["test"], batch_size=args.eval_pairs_per_batch, shuffle=False, collate_fn=collator),
    }
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    logging.info("Loading target Llama with LoRA: %s", args.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_name_or_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    base_model.config.use_cache = False
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=list(args.lora_target_modules),
    )
    model = get_peft_model(base_model, lora)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = torch.device(args.device)
    model.to(device)
    model.print_trainable_parameters()
    choice_ids = choice_token_ids(tokenizer, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(
        len(loaders["train"]) / args.gradient_accumulation_steps
    )
    total_updates = max(1, args.epochs * updates_per_epoch)
    warmup = int(round(total_updates * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_updates)

    checkpoint_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "best_adapter_state.pt"
    start_epoch = 1
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Checkpoint contract mismatch")
        set_peft_model_state_dict(model, checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint["history"])
        logging.info("Resuming after epoch %d", start_epoch - 1)

    epoch_units = len(encoded["train"]) + len(encoded["val"])
    excluded_planned_units = (
        args.epochs
        * (encoded["train"].excluded_overlength + encoded["val"].excluded_overlength)
        + encoded["test"].excluded_overlength
    )
    if excluded_planned_units:
        progress.set_stage(
            "2/4 account for overlength pairs excluded from planned work",
            total=excluded_planned_units,
        )
        progress.update(excluded_planned_units)
    completed_resume_units = (start_epoch - 1) * epoch_units
    if completed_resume_units:
        progress.set_stage(
            "2/4 restore completed epochs from resume checkpoint",
            total=completed_resume_units,
        )
        progress.update(completed_resume_units)
    stopped_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_count = 0
        accumulation_count = 0
        progress.set_stage(f"2/4 train epoch {epoch}/{args.epochs}", total=len(encoded["train"]))
        for batch_index, batch in enumerate(loaders["train"], 1):
            positive, negative, no_rag = forward_choice_logits(model, batch, choice_ids, device)
            gold = batch["gold_indices"].to(device)
            teacher = batch["teacher_probabilities"].to(device)
            losses = objective_losses(args, positive, negative, no_rag, gold, teacher)
            (losses["loss"] / args.gradient_accumulation_steps).backward()
            accumulation_count += 1
            count = len(batch["rows"])
            train_count += count
            train_loss_sum += float(losses["loss"].detach()) * count
            should_step = (
                accumulation_count == args.gradient_accumulation_steps
                or batch_index == len(loaders["train"])
            )
            if should_step:
                if accumulation_count != args.gradient_accumulation_steps:
                    correction = args.gradient_accumulation_steps / accumulation_count
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
            progress.set_detail(
                f"batch={batch_index}/{len(loaders['train'])} "
                f"loss={float(losses['loss'].detach()):.4f}"
            )
            progress.update(count)

        val_metrics, _ = evaluate(
            model,
            loaders["val"],
            choice_ids,
            device,
            args,
            progress,
            f"3/4 validation epoch {epoch}/{args.epochs}",
            collect_predictions=False,
        )
        score = selection_score(val_metrics)
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                best_path,
                {"adapter_state": get_peft_model_state_dict(model), "epoch": epoch, "score": score},
            )
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_count),
            "validation": val_metrics,
            "selection_score": score,
            "best": improved,
        }
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        atomic_torch_save(
            checkpoint_path,
            {
                "contract_fingerprint": contract_hash,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "adapter_state": get_peft_model_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )
        logging.info(
            "Epoch %d: train_loss=%.4f val_preference=%.4f val_positive=%.4f val_no_rag=%.4f score=%.4f best=%d",
            epoch,
            record["train_loss"],
            val_metrics["preference_accuracy"],
            val_metrics["positive_accuracy"],
            val_metrics["no_rag_accuracy"],
            score,
            best_epoch,
        )
        stopped_epoch = epoch
        if bad_epochs >= args.patience:
            skipped = (args.epochs - epoch) * epoch_units
            if skipped:
                progress.set_stage("early stopping: skipped planned epochs", total=skipped)
                progress.update(skipped)
            logging.info("Early stopping after epoch %d", epoch)
            break

    if not best_path.is_file():
        raise RuntimeError("No best adapter checkpoint was produced")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, best["adapter_state"])
    test_metrics, predictions = evaluate(
        model,
        loaders["test"],
        choice_ids,
        device,
        args,
        progress,
        "4/4 held-out single-document test",
        collect_predictions=True,
    )
    progress.close()
    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    save_predictions(output_dir / "test_predictions.jsonl", predictions)
    summary = {
        **contract,
        "contract_fingerprint": contract_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoded_pairs": {split: len(data) for split, data in encoded.items()},
        "excluded_overlength": {split: data.excluded_overlength for split, data in encoded.items()},
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_validation_score": best_score,
        "best_validation": history[best_epoch - 1]["validation"],
        "test": test_metrics,
        "final_model": str(final_dir.resolve()),
    }
    atomic_json(output_dir / "summary.json", summary)
    logging.info("Training complete: %s", json.dumps(summary["test"], ensure_ascii=False))


if __name__ == "__main__":
    main()

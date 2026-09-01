#!/usr/bin/env python3
"""LoRA-train Llama to utilize semantically valid document sets.

This is a bounded feasibility pilot, not an answer-accuracy optimizer.  The
semantic objective uses the likelihood of one fixed, correct, Direct-Support
response under four same-question contexts:

  valid       = Direct Support + Supporting Evidence
  full        = valid + No Evidence/Misleading Evidence
  invalid     = No Evidence/Misleading Evidence only
  direct_drop = valid with the reference Direct Support document removed

The proposed objective (1) raises fixed-response likelihood in valid context,
(2) makes full-context likelihood invariant to adding semantic noise, and (3)
requires valid context to explain the response better than invalid context.
The invalid score is stop-gradient, so the model is never rewarded for
degrading its answer under invalid context.  Frozen no-RAG choice probabilities
are preserved as an anti-forgetting constraint.
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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    build_anchored_user_prompt,
    encode_to_pre_choice,
    render_chat_prompt,
)


RUN_VERSION = "rag2_semantic_utilization_lora_pilot_v1"
DATA_VERSION = "rag2_semantic_utilization_contrast_sets_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medqa", "medmcqa"), default="medqa")
    parser.add_argument("--data-root", type=Path, default=base / "semantic_utilization_contrast_pilot_v1")
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "models/RAG2-Semantic-Utilization-LoRA")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--objective", choices=("sft_control", "semantic_utilization"), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--train-questions-per-batch", type=int, default=1)
    parser.add_argument("--eval-questions-per-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--valid-sft-weight", type=float, default=1.0)
    parser.add_argument("--semantic-margin-weight", type=float, default=0.5)
    parser.add_argument("--semantic-margin", type=float, default=0.05)
    parser.add_argument("--noise-invariance-weight", type=float, default=1.0)
    parser.add_argument("--no-rag-preservation-weight", type=float, default=0.1)
    parser.add_argument("--noise-tolerance", type=float, default=0.05)
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


def configure_attention_backend(name: str) -> None:
    if name == "sdpa" and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


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


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def format_seconds(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class StageStatus:
    def __init__(self, stage: int, stages: int, name: str, total: int, *, unit: str = "question") -> None:
        self.stage = stage
        self.stages = stages
        self.name = name
        self.total = max(0, int(total))
        self.unit = unit
        self.done = 0
        self.started = time.time()
        self.last_render = 0.0
        self.render(force=True)

    def update(self, value: int = 1, detail: str = "") -> None:
        self.done += int(value)
        self.render(force=self.done >= self.total, detail=detail)

    def render(self, *, force: bool = False, detail: str = "") -> None:
        now = time.time()
        if not force and now - self.last_render < 1.0:
            return
        self.last_render = now
        elapsed = max(1e-9, now - self.started)
        rate = self.done / elapsed
        eta = (self.total - self.done) / rate if rate > 0 else None
        percent = 100 * self.done / self.total if self.total else 100.0
        suffix = f" | {detail}" if detail else ""
        print(
            f"\r[overall {self.stage}/{self.stages}] [{self.name} | {self.done}/{self.total} "
            f"{percent:5.1f}% | {rate:.2f} {self.unit}/s | elapsed {format_seconds(elapsed)} | "
            f"ETA {'unknown' if eta is None else format_seconds(eta)}{suffix}]",
            end="\n" if force and self.done >= self.total else "",
            flush=True,
        )


def render_documents(documents: Sequence[dict[str, Any]]) -> str | None:
    texts = [" ".join(str(document["text"]).split()) for document in documents]
    return "\n\n".join(texts) if texts else None


def encode_response(
    tokenizer: Any,
    row: dict[str, Any],
    documents: Sequence[dict[str, Any]],
    response: str,
) -> dict[str, Any]:
    normalized = {"question": row["question"], "options": row["options"], "answer": row["gold_answer"]}
    user_prompt = build_anchored_user_prompt(normalized, render_documents(documents))
    prompt_text = render_chat_prompt(tokenizer, user_prompt)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = [*response_ids, int(tokenizer.eos_token_id)]
    input_ids = [*map(int, prompt_ids), *map(int, response_ids)]
    response_mask = [False] * len(prompt_ids) + [True] * len(response_ids)
    return {"input_ids": input_ids, "response_mask": response_mask, "response_tokens": len(response_ids)}


class EncodedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_tokens: int,
        stage: StageStatus,
    ) -> None:
        self.values: list[dict[str, Any]] = []
        self.excluded_overlength = 0
        self.max_observed_tokens = 0
        for row in rows:
            response = str(row["reference"]["canonical_response"])
            contexts = {
                "valid": row["valid_documents"],
                "full": row["full_documents"],
                "invalid": row["invalid_documents"],
                "ablated": row["direct_ablated_documents"],
            }
            encoded = {
                name: encode_response(tokenizer, row, documents, response)
                for name, documents in contexts.items()
            }
            normalized = {"question": row["question"], "options": row["options"], "answer": row["gold_answer"]}
            no_rag = encode_to_pre_choice(tokenizer, normalized, None, str(row["no_rag"]["rationale"]))
            maximum = max([len(value["input_ids"]) for value in encoded.values()] + [len(no_rag.input_ids)])
            self.max_observed_tokens = max(self.max_observed_tokens, maximum)
            if maximum > max_tokens:
                self.excluded_overlength += 1
                stage.update(detail=f"excluded_overlength={self.excluded_overlength}")
                continue
            self.values.append(
                {
                    "row": row,
                    **encoded,
                    "no_rag_ids": [int(value) for value in no_rag.input_ids],
                }
            )
            stage.update(detail=f"kept={len(self.values)} excluded={self.excluded_overlength}")
        if not self.values:
            raise RuntimeError("Every prepared question exceeded --max-input-tokens")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.values[index]


def collate(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": [value["row"] for value in values],
        **{name: [value[name] for value in values] for name in ("valid", "full", "invalid", "ablated")},
        "no_rag": [value["no_rag_ids"] for value in values],
    }


def pad_response_sequences(
    values: Sequence[dict[str, Any]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(value["input_ids"]) for value in values)
    input_ids = torch.full((len(values), maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(values), maximum), dtype=torch.long)
    response_mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, value in enumerate(values):
        length = len(value["input_ids"])
        input_ids[index, :length] = torch.tensor(value["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = 1
        response_mask[index, :length] = torch.tensor(value["response_mask"], dtype=torch.bool)
    return input_ids.to(device), attention_mask.to(device), response_mask.to(device)


def sequence_log_likelihood(
    model: Any,
    values: Sequence[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention_mask, response_mask = pad_response_sequences(values, pad_token_id, device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
    token_logprob = logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logprob * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def pad_prefixes(
    values: Sequence[Sequence[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(map(len, values))
    ids = torch.full((len(values), maximum), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(values), maximum), dtype=torch.long)
    for index, sequence in enumerate(values):
        length = len(sequence)
        ids[index, maximum - length :] = torch.tensor(sequence, dtype=torch.long)
        mask[index, maximum - length :] = 1
    positions = mask.cumsum(dim=-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids.to(device), mask.to(device), positions.to(device)


def no_rag_choice_logits(
    model: Any,
    values: Sequence[Sequence[int]],
    pad_token_id: int,
    choice_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    ids, mask, positions = pad_prefixes(values, pad_token_id, device)
    outputs = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        logits_to_keep=1,
    )
    return outputs.logits[:, -1].float().index_select(-1, choice_ids)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.clamp_min(1e-8)
    right = right.clamp_min(1e-8)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def objective_loss(
    args: argparse.Namespace,
    valid: torch.Tensor,
    full: torch.Tensor,
    invalid: torch.Tensor,
    student_no_rag: torch.Tensor,
    teacher_no_rag: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid_sft = -valid.mean()
    no_rag_kl = F.kl_div(
        F.log_softmax(student_no_rag, dim=-1), teacher_no_rag, reduction="batchmean"
    )
    zero = valid.sum() * 0.0
    semantic_margin = F.relu(args.semantic_margin - (valid - invalid.detach())).mean()
    noise_invariance = F.mse_loss(full, valid.detach())
    loss = args.valid_sft_weight * valid_sft + args.no_rag_preservation_weight * no_rag_kl
    if args.objective == "semantic_utilization":
        loss = (
            loss
            + args.semantic_margin_weight * semantic_margin
            + args.noise_invariance_weight * noise_invariance
        )
    else:
        semantic_margin = zero
        noise_invariance = zero
    return {
        "loss": loss,
        "valid_sft": valid_sft,
        "semantic_margin": semantic_margin,
        "noise_invariance": noise_invariance,
        "no_rag_kl": no_rag_kl,
    }


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    pad_token_id: int,
    choice_ids: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    status: StageStatus,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals = {
        "count": 0,
        "f_valid": 0.0,
        "f_full": 0.0,
        "f_invalid": 0.0,
        "f_ablated": 0.0,
        "semantic_preference": 0,
        "direct_positive": 0,
        "noise_abs": 0.0,
        "noise_within_tolerance": 0,
        "no_rag_js": 0.0,
        "no_rag_correct": 0,
        "frozen_no_rag_correct": 0,
        "no_rag_changed": 0,
    }
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        count = len(batch["rows"])
        all_values = batch["valid"] + batch["full"] + batch["invalid"] + batch["ablated"]
        scores = sequence_log_likelihood(model, all_values, pad_token_id, device)
        valid, full, invalid, ablated = scores.split(count)
        no_logits = no_rag_choice_logits(model, batch["no_rag"], pad_token_id, choice_ids, device)
        no_prob = F.softmax(no_logits, dim=-1)
        teacher = torch.tensor(
            [row["no_rag"]["choice_probabilities"] for row in batch["rows"]],
            dtype=torch.float32,
            device=device,
        )
        gold = torch.tensor(
            [CHOICES.index(str(row["gold_answer"])) for row in batch["rows"]],
            dtype=torch.long,
            device=device,
        )
        student_prediction = no_prob.argmax(dim=-1)
        teacher_prediction = teacher.argmax(dim=-1)
        noise = (full - valid).abs()
        totals["count"] += count
        totals["f_valid"] += float(valid.sum())
        totals["f_full"] += float(full.sum())
        totals["f_invalid"] += float(invalid.sum())
        totals["f_ablated"] += float(ablated.sum())
        totals["semantic_preference"] += int((valid > invalid).sum())
        totals["direct_positive"] += int((valid > ablated).sum())
        totals["noise_abs"] += float(noise.sum())
        totals["noise_within_tolerance"] += int((noise <= args.noise_tolerance).sum())
        totals["no_rag_js"] += float(js_divergence(teacher, no_prob).sum())
        totals["no_rag_correct"] += int((student_prediction == gold).sum())
        totals["frozen_no_rag_correct"] += int((teacher_prediction == gold).sum())
        totals["no_rag_changed"] += int((student_prediction != teacher_prediction).sum())
        for index, row in enumerate(batch["rows"]):
            predictions.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "f_valid": float(valid[index]),
                    "f_full": float(full[index]),
                    "f_invalid": float(invalid[index]),
                    "f_ablated": float(ablated[index]),
                    "semantic_preference": bool(valid[index] > invalid[index]),
                    "direct_positive": bool(valid[index] > ablated[index]),
                    "noise_abs": float(noise[index]),
                    "no_rag_js": float(js_divergence(teacher[index : index + 1], no_prob[index : index + 1])[0]),
                    "gold_answer": str(row["gold_answer"]),
                    "no_rag_prediction": CHOICES[int(student_prediction[index])],
                    "frozen_no_rag_prediction": CHOICES[int(teacher_prediction[index])],
                    "no_rag_correct": bool(student_prediction[index] == gold[index]),
                }
            )
        status.update(count)
    count = totals.pop("count")
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update(
        {
            "questions": count,
            "semantic_gap": metrics["f_valid"] - metrics["f_invalid"],
            "direct_contribution": metrics["f_valid"] - metrics["f_ablated"],
            "noise_delta": metrics["f_full"] - metrics["f_valid"],
        }
    )
    metrics["selection_score"] = (
        0.50 * metrics["semantic_preference"]
        + 0.30 * metrics["direct_positive"]
        + 0.20 * metrics["noise_within_tolerance"]
        - 0.10 * min(1.0, metrics["no_rag_js"] / 0.05)
    )
    return metrics, predictions


def choice_token_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    ids = []
    for choice in CHOICES:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Choice {choice} is not one token: {encoded}")
        ids.append(int(encoded[0]))
    return torch.tensor(ids, dtype=torch.long, device=device)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"Empty prepared split: {path}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate prepared question: {path}")
    return rows


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def summary_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_test"]
    final = summary["final_test"]
    rows = [
        "# Semantic-utilization LoRA pilot",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Objective: {summary['objective']}",
        f"- Best epoch: {summary['best_epoch']}",
        f"- Test N: {final['questions']}",
        "",
        "| Metric | Frozen baseline | Final | Absolute change | Meaning |",
        "|---|---:|---:|---:|---|",
    ]
    definitions = {
        "semantic_preference": "Fraction where valid documents explain the fixed response better than invalid documents",
        "direct_positive": "Fraction where removing the reference Direct Support document lowers response likelihood",
        "noise_abs": "Mean absolute likelihood change after adding No/Misleading documents; lower is better",
        "no_rag_js": "No-RAG choice-distribution drift from frozen Llama; lower is better",
        "no_rag_correct": "No-RAG MCQ accuracy; preservation diagnostic, not the training objective",
    }
    for name, meaning in definitions.items():
        before = float(baseline[name])
        after = float(final[name])
        rows.append(f"| {name} | {before:.6f} | {after:.6f} | {after-before:+.6f} | {meaning} |")
    passed = summary["success_criterion"]
    rows.extend(
        [
            "",
            f"- Pre-registered pilot pass: **{passed['passed']}**",
            f"- Semantic-preference improvement: {passed['semantic_preference_delta']:+.4f} (required >= +0.10)",
            f"- Direct-contribution improvement: {passed['direct_positive_delta']:+.4f} (required >= +0.05)",
            f"- Noise penalty not worse: {passed['noise_not_worse']}",
            f"- No-RAG accuracy drop: {passed['no_rag_accuracy_delta']:+.4f} (allowed >= -0.01)",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)
    configure_attention_backend(args.attn_implementation)
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Epochs and gradient accumulation must be positive")

    data_dir = args.data_root / args.dataset
    data_manifest_path = data_dir / "manifest.json"
    data_paths = {split: data_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
    for path in [data_manifest_path, args.model_name_or_path, *data_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("run_version") != DATA_VERSION:
        raise ValueError(f"Unexpected prepared-data version: {data_manifest.get('run_version')}")

    output_dir = args.output_root / args.dataset / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < 5 * 1024**3:
        raise RuntimeError(f"Insufficient free disk for checkpoints: {free_bytes / 1024**3:.2f} GiB")
    contract = {
        "run_version": RUN_VERSION,
        "hypothesis": "Semantic-valid evidence should have greater causal contribution while No/Misleading additions should not change the fixed output.",
        "dataset": args.dataset,
        "objective": args.objective,
        "data_manifest": file_identity(data_manifest_path),
        "data_contract_fingerprint": data_manifest["contract_fingerprint"],
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "train_questions_per_batch": args.train_questions_per_batch,
        "eval_questions_per_batch": args.eval_questions_per_batch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_input_tokens": args.max_input_tokens,
        "loss_weights": {
            "valid_sft": args.valid_sft_weight,
            "semantic_margin": args.semantic_margin_weight,
            "noise_invariance": args.noise_invariance_weight,
            "no_rag_preservation": args.no_rag_preservation_weight,
        },
        "semantic_margin": args.semantic_margin,
        "noise_tolerance": args.noise_tolerance,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": list(args.lora_target_modules),
        },
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "git_commit_before_task_changes": git_commit(),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        "selection_metric": "0.50 semantic_preference + 0.30 direct_positive + 0.20 noise_tolerance - no_rag_js penalty",
        "test_not_used_for_selection": True,
    }
    contract_hash = fingerprint(contract)
    experiment_manifest = output_dir / "experiment_manifest.json"
    summary_path = output_dir / "training_summary.json"
    if args.resume and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("contract_fingerprint") == contract_hash:
            logging.info("Completed semantic-utilization run is reusable: %s", output_dir)
            return
        raise RuntimeError("Completed-run contract mismatch; use a new run name")
    if experiment_manifest.is_file():
        existing = json.loads(experiment_manifest.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Experiment contract mismatch; use a new run name")
    else:
        atomic_json(experiment_manifest, {"contract_fingerprint": contract_hash, "contract": contract})

    raw = {split: load_rows(path) for split, path in data_paths.items()}
    total_stages = 3 + 2 + 2 * args.epochs + 2
    logging.info(
        "Training plan: dataset=%s objective=%s stages=%d raw=%s output=%s",
        args.dataset,
        args.objective,
        total_stages,
        {split: len(rows) for split, rows in raw.items()},
        output_dir,
    )
    if args.plan_only:
        return

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("This LoRA pilot requires an available CUDA device")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    datasets = {}
    stage_index = 1
    for split in ("train", "val", "test"):
        status = StageStatus(stage_index, total_stages, f"tokenize {split}", len(raw[split]))
        datasets[split] = EncodedDataset(raw[split], tokenizer, args.max_input_tokens, status)
        stage_index += 1
    split_ids = {split: {value["row"]["sample_id"] for value in dataset.values} for split, dataset in datasets.items()}
    if any(split_ids[left] & split_ids[right] for left in split_ids for right in split_ids if left < right):
        raise RuntimeError("Tokenized question leakage across splits")
    logging.info(
        "Token preflight: %s",
        {
            split: {
                "kept": len(dataset),
                "excluded_overlength": dataset.excluded_overlength,
                "max_observed_tokens": dataset.max_observed_tokens,
            }
            for split, dataset in datasets.items()
        },
    )

    train_loader = DataLoader(
        datasets["train"], batch_size=args.train_questions_per_batch, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    eval_loaders = {
        split: DataLoader(
            datasets[split], batch_size=args.eval_questions_per_batch, shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        for split in ("val", "test")
    }
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)
    logging.info("Loading frozen target architecture with trainable LoRA: %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    model.print_trainable_parameters()
    choice_ids = choice_token_ids(tokenizer, device)
    pad_token_id = int(tokenizer.pad_token_id)

    status = StageStatus(stage_index, total_stages, "frozen baseline validation", len(datasets["val"]))
    baseline_val, _ = evaluate(model, eval_loaders["val"], pad_token_id, choice_ids, device, args, status)
    stage_index += 1
    status = StageStatus(stage_index, total_stages, "frozen baseline internal test", len(datasets["test"]))
    baseline_test, baseline_test_predictions = evaluate(
        model, eval_loaders["test"], pad_token_id, choice_ids, device, args, status
    )
    atomic_jsonl(output_dir / "baseline_test_predictions.jsonl", baseline_test_predictions)
    stage_index += 1

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    checkpoint_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_adapter.pt"
    start_epoch = 1
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Training checkpoint contract mismatch")
        set_peft_model_state_dict(model, checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint["history"])
        global_step = int(checkpoint["global_step"])
        logging.info("Resuming after epoch %d; next epoch=%d", start_epoch - 1, start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        status = StageStatus(stage_index, total_stages, f"train epoch {epoch}/{args.epochs}", len(datasets["train"]))
        stage_index += 1
        loss_sums = {name: 0.0 for name in ("loss", "valid_sft", "semantic_margin", "noise_invariance", "no_rag_kl")}
        train_count = 0
        for batch_index, batch in enumerate(train_loader, 1):
            count = len(batch["rows"])
            if args.objective == "semantic_utilization":
                combined = batch["valid"] + batch["full"]
                scores = sequence_log_likelihood(model, combined, pad_token_id, device)
                valid, full = scores.split(count)
                with torch.no_grad():
                    invalid = sequence_log_likelihood(model, batch["invalid"], pad_token_id, device)
            else:
                valid = sequence_log_likelihood(model, batch["valid"], pad_token_id, device)
                full = valid.detach()
                invalid = valid.detach()
            student_no_rag = no_rag_choice_logits(model, batch["no_rag"], pad_token_id, choice_ids, device)
            teacher_no_rag = torch.tensor(
                [row["no_rag"]["choice_probabilities"] for row in batch["rows"]],
                dtype=torch.float32,
                device=device,
            )
            losses = objective_loss(args, valid, full, invalid, student_no_rag, teacher_no_rag)
            (losses["loss"] / args.gradient_accumulation_steps).backward()
            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            for name in loss_sums:
                loss_sums[name] += float(losses[name].detach()) * count
            train_count += count
            status.update(count, detail=f"batch={batch_index}/{len(train_loader)} loss={float(losses['loss']):.4f}")

        status = StageStatus(stage_index, total_stages, f"validation epoch {epoch}/{args.epochs}", len(datasets["val"]))
        stage_index += 1
        validation, _ = evaluate(model, eval_loaders["val"], pad_token_id, choice_ids, device, args, status)
        record = {
            "epoch": epoch,
            "train": {name: value / max(1, train_count) for name, value in loss_sums.items()},
            "validation": validation,
        }
        history.append(record)
        score = float(validation["selection_score"])
        improved = score > best_score + 1e-8
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch(
                best_path,
                {"contract_fingerprint": contract_hash, "epoch": epoch, "adapter": get_peft_model_state_dict(model)},
            )
        else:
            bad_epochs += 1
        atomic_torch(
            checkpoint_path,
            {
                "contract_fingerprint": contract_hash,
                "epoch": epoch,
                "adapter": get_peft_model_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "global_step": global_step,
            },
        )
        logging.info(
            "Epoch %d: selection=%.4f semantic_pref=%.4f direct_positive=%.4f noise_abs=%.4f no_rag_js=%.6f best=%d",
            epoch,
            score,
            validation["semantic_preference"],
            validation["direct_positive"],
            validation["noise_abs"],
            validation["no_rag_js"],
            best_epoch,
        )
        if bad_epochs >= args.patience:
            logging.info("Early stopping after epoch %d", epoch)
            break

    if not best_path.is_file():
        raise RuntimeError("No best adapter checkpoint was written")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, best["adapter"])
    status = StageStatus(stage_index, total_stages, "final internal test with best validation adapter", len(datasets["test"]))
    final_test, final_test_predictions = evaluate(
        model, eval_loaders["test"], pad_token_id, choice_ids, device, args, status
    )
    atomic_jsonl(output_dir / "final_test_predictions.jsonl", final_test_predictions)
    stage_index += 1

    status = StageStatus(stage_index, total_stages, "save adapter and summary", 1, unit="artifact")
    final_adapter = output_dir / "final_adapter"
    temporary_adapter = output_dir / f"final_adapter.tmp.{os.getpid()}"
    if temporary_adapter.exists():
        shutil.rmtree(temporary_adapter)
    model.save_pretrained(temporary_adapter, safe_serialization=True)
    tokenizer.save_pretrained(temporary_adapter)
    if final_adapter.exists():
        if not args.resume:
            raise RuntimeError(f"Final adapter already exists: {final_adapter}")
        shutil.rmtree(final_adapter)
    os.replace(temporary_adapter, final_adapter)

    semantic_delta = final_test["semantic_preference"] - baseline_test["semantic_preference"]
    direct_delta = final_test["direct_positive"] - baseline_test["direct_positive"]
    no_rag_accuracy_delta = final_test["no_rag_correct"] - baseline_test["no_rag_correct"]
    noise_not_worse = final_test["noise_abs"] <= baseline_test["noise_abs"] + 0.005
    criterion = {
        "semantic_preference_delta": semantic_delta,
        "direct_positive_delta": direct_delta,
        "noise_not_worse": noise_not_worse,
        "no_rag_accuracy_delta": no_rag_accuracy_delta,
        "passed": bool(
            semantic_delta >= 0.10
            and direct_delta >= 0.05
            and noise_not_worse
            and no_rag_accuracy_delta >= -0.01
        ),
    }
    summary = {
        "run_version": RUN_VERSION,
        "contract_fingerprint": contract_hash,
        "data_contract_fingerprint": data_manifest["contract_fingerprint"],
        "dataset": args.dataset,
        "objective": args.objective,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "tokenized": {
            split: {
                "questions": len(dataset),
                "excluded_overlength": dataset.excluded_overlength,
                "max_observed_tokens": dataset.max_observed_tokens,
            }
            for split, dataset in datasets.items()
        },
        "baseline_validation": baseline_val,
        "baseline_test": baseline_test,
        "history": history,
        "final_test": final_test,
        "success_criterion": criterion,
        "final_adapter": str(final_adapter.resolve()),
    }
    atomic_json(summary_path, summary)
    (output_dir / "SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
    status.update(1)
    logging.info("Training complete: %s", output_dir)
    logging.info("Pilot pass=%s test=%s", criterion["passed"], json.dumps(final_test))


if __name__ == "__main__":
    main()

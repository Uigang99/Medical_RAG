#!/usr/bin/env python3
"""LoRA-train Llama-3 on direct-choice semantic/behavior mismatch cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import (  # noqa: E402
    PROMPT_POLICY_VERSION,
    sequence_for_prompt,
)
from medrag.core import BenchmarkSample  # noqa: E402
from medrag.progress import StageProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from medrag.training.direct_semantic_mismatch import (  # noqa: E402
    TRAIN_CASES,
    semantic_mismatch_losses,
)


RUN_VERSION = "rag2_direct_semantic_mismatch_lora_v1"
PAIR_VERSION = "rag2_direct_semantic_mismatch_mvp_pairs_v1"
BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), default="medmcqa")
    parser.add_argument("--pair-root", type=Path, default=BASE / "direct_semantic_mismatch_mvp_pairs_v1")
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "models/RAG2-Direct-Semantic-Mismatch-LoRA")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--objective", choices=("mismatch", "question_only", "rag_ce"), default="mismatch")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--train-examples-per-batch", type=int, default=8)
    parser.add_argument("--eval-examples-per-batch", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--boundary-margin", type=float, default=0.0)
    parser.add_argument("--gain-margin", type=float, default=0.5)
    parser.add_argument("--no-rag-preservation-weight", type=float, default=2.0)
    parser.add_argument("--failure-case-weight", type=float, default=1.0)
    parser.add_argument("--normal-case-weight", type=float, default=0.1)
    parser.add_argument("--max-no-rag-accuracy-drop", type=float, default=0.005)
    parser.add_argument("--max-destruction-rate-increase", type=float, default=0.0)
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
    parser.add_argument("--preflight-only", action="store_true")
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


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        logging.info("SDPA policy: cuDNN disabled; Flash/Efficient/Math enabled")


def make_sample(row: dict[str, Any]) -> BenchmarkSample:
    return BenchmarkSample(
        id=str(row["sample_id"]),
        row_idx=int(row["row_idx"]),
        task=str(row["dataset"]),
        collection="unified",
        dataset=str(row["dataset"]),
        split=str(row["split"]),
        question=str(row["question"]),
        options=dict(row["options"]),
        answer=str(row["gold_answer"]),
        answers=[str(row["gold_answer"])],
        raw=dict(row),
    )


class EncodedDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, max_tokens: int, stage: str) -> None:
        self.values: list[dict[str, Any]] = []
        self.excluded_overlength = 0
        self.max_observed_tokens = 0
        progress = StageProgress(len(rows), stage)
        for row in rows:
            sample = make_sample(row)
            q_ids, q_prompt = sequence_for_prompt(tokenizer, sample, None)
            d_ids, d_prompt = sequence_for_prompt(tokenizer, sample, str(row["document_text"]))
            if hashlib.sha256(q_prompt.encode()).hexdigest() != str(row["frozen_no_rag_prompt_sha256"]):
                raise RuntimeError(f"No-RAG prompt contract mismatch: {sample.id}")
            if hashlib.sha256(d_prompt.encode()).hexdigest() != str(row["frozen_document_prompt_sha256"]):
                raise RuntimeError(f"Document prompt contract mismatch: {row['pair_id']}")
            current_max = max(len(q_ids), len(d_ids))
            self.max_observed_tokens = max(self.max_observed_tokens, current_max)
            if current_max > max_tokens:
                self.excluded_overlength += 1
                progress.update()
                continue
            self.values.append({"row": row, "question_ids": q_ids, "document_ids": d_ids})
            progress.update()
        progress.close()
        if not self.values:
            raise RuntimeError(f"Every row exceeded max tokens in {stage}")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.values[index]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences = [value["document_ids"] for value in values] + [value["question_ids"] for value in values]
        maximum = max(map(len, sequences))
        ids = torch.full((len(sequences), maximum), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            ids[index, maximum - length :] = torch.tensor(sequence)
            mask[index, maximum - length :] = 1
        position = mask.cumsum(dim=-1) - 1
        position.masked_fill_(mask == 0, 0)
        rows = [value["row"] for value in values]
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": position,
            "gold_indices": torch.tensor([CHOICES.index(str(row["gold_answer"])) for row in rows]),
            "case_indices": torch.tensor([TRAIN_CASES.index(str(row["case"])) if row["case"] in TRAIN_CASES else -1 for row in rows]),
            "frozen_document_probabilities": torch.tensor([row["frozen_document_probabilities"] for row in rows], dtype=torch.float32),
            "frozen_no_rag_probabilities": torch.tensor([row["frozen_no_rag_probabilities"] for row in rows], dtype=torch.float32),
            "question_repeat_weights": torch.tensor([float(row["question_repeat_weight"]) for row in rows], dtype=torch.float32),
            "rows": rows,
        }


def choice_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    ids = []
    for choice in CHOICES:
        value = tokenizer.encode(choice, add_special_tokens=False)
        if len(value) != 1:
            raise RuntimeError(f"Choice label is not one token: {choice} -> {value}")
        ids.append(value[0])
    return torch.tensor(ids, device=device)


def forward_logits(model: Any, batch: dict[str, Any], selected_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        position_ids=batch["position_ids"].to(device, non_blocking=True),
        use_cache=False,
        logits_to_keep=1,
    )
    logits = outputs.logits[:, -1].index_select(-1, selected_ids).float()
    count = len(batch["rows"])
    return logits[:count], logits[count:]


def objective_loss(args: argparse.Namespace, document: torch.Tensor, question: torch.Tensor, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    gold = batch["gold_indices"].to(device)
    if args.objective == "question_only":
        value = F.cross_entropy(question, gold)
        return {"loss": value, "question_ce": value}
    if args.objective == "rag_ce":
        value = F.cross_entropy(document, gold)
        return {"loss": value, "document_ce": value}
    weights = {
        "direct_support_w2w": args.failure_case_weight,
        "direct_support_c2w": args.failure_case_weight,
        "no_evidence_c2w": args.failure_case_weight,
        "direct_support_preserve": args.normal_case_weight,
        "no_evidence_preserve": args.normal_case_weight,
    }
    return semantic_mismatch_losses(
        document_logits=document,
        no_rag_logits=question,
        frozen_document_probabilities=batch["frozen_document_probabilities"].to(device),
        frozen_no_rag_probabilities=batch["frozen_no_rag_probabilities"].to(device),
        gold_indices=gold,
        case_indices=batch["case_indices"].to(device),
        boundary_margin=args.boundary_margin,
        gain_margin=args.gain_margin,
        case_weights=weights,
        no_rag_preservation_weight=args.no_rag_preservation_weight,
        no_rag_row_weights=batch["question_repeat_weights"].to(device),
    )


def proportion(correct: int, count: int) -> float | None:
    return correct / count if count else None


def format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def wilson_interval(correct: int, count: int, z: float = 1.96) -> list[float] | None:
    if count <= 0:
        return None
    p = correct / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def finalize_eval(rows: list[dict[str, Any]], document_logits: list[torch.Tensor], question_logits: list[torch.Tensor]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = torch.cat(document_logits).float()
    question = torch.cat(question_logits).float()
    doc_pred = document.argmax(-1).tolist()
    q_pred = question.argmax(-1).tolist()
    gold = [CHOICES.index(str(row["gold_answer"])) for row in rows]
    groups: dict[str, dict[str, int]] = defaultdict(lambda: Counter(count=0, student_correct=0, frozen_correct=0, improved=0, regressed=0))
    q_by_sample: dict[str, tuple[int, int, int]] = {}
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        student_correct = doc_pred[index] == gold[index]
        frozen_correct = bool(row["frozen_document_correct"])
        no_rag_correct = bool(row["frozen_no_rag_correct"])
        semantic = str(row["semantic_label"])
        keys = ["all", f"semantic:{semantic}"]
        if semantic == "direct_support":
            keys.append("direct_support:no_rag_correct" if no_rag_correct else "direct_support:no_rag_wrong")
        elif semantic == "no_evidence":
            keys.append("no_evidence:no_rag_correct" if no_rag_correct else "no_evidence:no_rag_wrong")
        for key in keys:
            value = groups[key]
            value["count"] += 1
            value["student_correct"] += int(student_correct)
            value["frozen_correct"] += int(frozen_correct)
            value["improved"] += int(student_correct and not frozen_correct)
            value["regressed"] += int(frozen_correct and not student_correct)
        sample_id = str(row["sample_id"])
        current_q = (q_pred[index], gold[index], CHOICES.index(str(row["frozen_no_rag_prediction"])))
        previous = q_by_sample.setdefault(sample_id, current_q)
        if previous != current_q:
            raise RuntimeError(f"Inconsistent no-RAG predictions across repeated question: {sample_id}")
        predictions.append(
            {
                "sample_id": sample_id,
                "pair_id": row["pair_id"],
                "semantic_label": semantic,
                "frozen_transition": row["frozen_transition"],
                "gold_answer": row["gold_answer"],
                "student_document_prediction": CHOICES[doc_pred[index]],
                "student_no_rag_prediction": CHOICES[q_pred[index]],
                "frozen_document_prediction": row["frozen_document_prediction"],
                "frozen_no_rag_prediction": row["frozen_no_rag_prediction"],
            }
        )
    group_metrics = {}
    for name, value in groups.items():
        count = int(value["count"])
        student = int(value["student_correct"])
        frozen = int(value["frozen_correct"])
        group_metrics[name] = {
            "n": count,
            "student_accuracy": proportion(student, count),
            "student_accuracy_95ci": wilson_interval(student, count),
            "frozen_accuracy": proportion(frozen, count),
            "absolute_accuracy_delta": (student - frozen) / count if count else None,
            "improved_pairs": int(value["improved"]),
            "regressed_pairs": int(value["regressed"]),
        }
    q_count = len(q_by_sample)
    student_q_correct = sum(value[0] == value[1] for value in q_by_sample.values())
    frozen_q_correct = sum(value[2] == value[1] for value in q_by_sample.values())
    q_changed = sum(value[0] != value[2] for value in q_by_sample.values())
    return {
        "question_level_no_rag": {
            "n": q_count,
            "student_accuracy": proportion(student_q_correct, q_count),
            "frozen_accuracy": proportion(frozen_q_correct, q_count),
            "absolute_accuracy_delta": (student_q_correct - frozen_q_correct) / q_count,
            "answer_change_rate": q_changed / q_count,
        },
        "pair_groups": group_metrics,
    }, predictions


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, selected_ids: torch.Tensor, device: torch.device, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    documents: list[torch.Tensor] = []
    questions: list[torch.Tensor] = []
    progress = StageProgress(len(loader.dataset), stage)
    for batch in loader:
        document, question = forward_logits(model, batch, selected_ids, device)
        rows.extend(batch["rows"])
        documents.append(document.cpu())
        questions.append(question.cpu())
        progress.update(len(batch["rows"]))
    progress.close()
    return finalize_eval(rows, documents, questions)


def selection(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    groups = metrics["pair_groups"]
    primary = groups.get("direct_support:no_rag_wrong", {})
    direct_safe = groups.get("direct_support:no_rag_correct", {})
    no_evidence_safe = groups.get("no_evidence:no_rag_correct", {})
    q = metrics["question_level_no_rag"]
    primary_acc = float(primary.get("student_accuracy") or 0.0)
    q_drop = max(0.0, -float(q["absolute_accuracy_delta"]))
    direct_increase = max(0.0, -float(direct_safe.get("absolute_accuracy_delta") or 0.0))
    no_evidence_increase = max(0.0, -float(no_evidence_safe.get("absolute_accuracy_delta") or 0.0))
    limits = {
        "no_rag_accuracy_drop": args.max_no_rag_accuracy_drop,
        "direct_support_destruction_increase": args.max_destruction_rate_increase,
        "no_evidence_destruction_increase": args.max_destruction_rate_increase,
    }
    measured = {
        "no_rag_accuracy_drop": q_drop,
        "direct_support_destruction_increase": direct_increase,
        "no_evidence_destruction_increase": no_evidence_increase,
    }
    violations = {key: max(0.0, measured[key] - limits[key]) for key in limits}
    normalized = sum(violations[key] / max(limits[key], 0.005) for key in limits)
    feasible = all(value <= 1e-12 for value in violations.values())
    rank = [1.0, primary_acc] if feasible else [0.0, -normalized, primary_acc]
    return {"feasible": feasible, "primary_accuracy": primary_acc, "measured": measured, "limits": limits, "violations": violations, "rank": rank}


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    test = summary["test"]
    groups = test["pair_groups"]
    q = test["question_level_no_rag"]
    rows = [
        (
            "No-RAG question accuracy",
            q["n"],
            q["frozen_accuracy"],
            q["student_accuracy"],
            q["absolute_accuracy_delta"],
        )
    ]
    for key, label in (
        ("direct_support:no_rag_wrong", "Direct Support accuracy when frozen No-RAG was wrong (primary)"),
        ("direct_support:no_rag_correct", "Direct Support accuracy when frozen No-RAG was correct"),
        ("no_evidence:no_rag_correct", "No Evidence accuracy when frozen No-RAG was correct"),
        ("no_evidence:no_rag_wrong", "No Evidence accuracy when frozen No-RAG was wrong (audit only)"),
    ):
        value = groups.get(key)
        if value is not None:
            rows.append(
                (
                    label,
                    value["n"],
                    value["frozen_accuracy"],
                    value["student_accuracy"],
                    value["absolute_accuracy_delta"],
                )
            )
    lines = [
        f"# Direct semantic-mismatch MVP — {summary['dataset']}",
        "",
        f"- Best epoch: {summary['best_epoch']}",
        f"- Objective: `{summary['objective']}`",
        f"- Preregistered pilot criterion: **{'PASS' if summary['pilot_success']['passed'] else 'FAIL'}**",
        "- Frozen is the unchanged Llama-3-8B under the identical direct-choice prompt.",
        "",
        "| Metric | N | Frozen | Student | Absolute change |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, count, frozen, student, delta in rows:
        lines.append(
            f"| {label} | {count:,} | {100*frozen:.2f}% | {100*student:.2f}% | {100*delta:+.2f}%p |"
        )
    lines.extend(
        [
            "",
            "Checkpoint selection uses validation only. Test metrics are reported once and are not used for tuning.",
        ]
    )
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pilot_success(metrics: dict[str, Any]) -> dict[str, Any]:
    groups = metrics["pair_groups"]
    primary_delta = float(
        groups.get("direct_support:no_rag_wrong", {}).get("absolute_accuracy_delta")
        or 0.0
    )
    no_rag_delta = float(metrics["question_level_no_rag"]["absolute_accuracy_delta"])
    direct_safe_delta = float(
        groups.get("direct_support:no_rag_correct", {}).get("absolute_accuracy_delta")
        or 0.0
    )
    no_evidence_safe_delta = float(
        groups.get("no_evidence:no_rag_correct", {}).get("absolute_accuracy_delta")
        or 0.0
    )
    criteria = {
        "direct_support_no_rag_wrong_accuracy_gain_at_least_0p02": primary_delta >= 0.02,
        "no_rag_accuracy_drop_at_most_0p005": no_rag_delta >= -0.005,
        "direct_support_destruction_does_not_increase": direct_safe_delta >= 0.0,
        "no_evidence_destruction_does_not_increase": no_evidence_safe_delta >= 0.0,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "measured": {
            "primary_accuracy_delta": primary_delta,
            "no_rag_accuracy_delta": no_rag_delta,
            "direct_support_no_rag_correct_accuracy_delta": direct_safe_delta,
            "no_evidence_no_rag_correct_accuracy_delta": no_evidence_safe_delta,
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    set_seed(args.seed)
    configure_attention_backend(args.attn_implementation)
    pair_dir = args.pair_root / args.dataset
    pair_manifest_path = pair_dir / "manifest.json"
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if pair_manifest.get("run_version") != PAIR_VERSION:
        raise RuntimeError("Prepared-pair version mismatch")
    raw = {split: list(iter_jsonl(pair_dir / f"{split}.jsonl")) for split in ("train", "val", "test")}
    output_dir = args.output_root / args.dataset / args.run_name
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "objective": args.objective,
        "hypothesis": pair_manifest["hypothesis"],
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "model": str(args.model_name_or_path.resolve()),
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": {"train": args.train_examples_per_batch, "eval": args.eval_examples_per_batch, "accumulation": args.gradient_accumulation_steps},
        "optimizer": {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio, "max_grad_norm": args.max_grad_norm},
        "loss": {"boundary_margin": args.boundary_margin, "gain_margin": args.gain_margin, "no_rag_preservation_weight": args.no_rag_preservation_weight, "failure_case_weight": args.failure_case_weight, "normal_case_weight": args.normal_case_weight},
        "checkpoint_constraints": {"max_no_rag_accuracy_drop": args.max_no_rag_accuracy_drop, "max_destruction_rate_increase": args.max_destruction_rate_increase},
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": list(args.lora_target_modules)},
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "code_sha256": sha256_file(Path(__file__)),
        "code_commit": git_commit(),
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
    }
    contract_hash = fingerprint(contract)
    contract_path = output_dir / "run_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Training contract mismatch; use a versioned run name")
    atomic_json(contract_path, {**contract, "contract_fingerprint": contract_hash, "created_at": datetime.now(timezone.utc).isoformat()})
    free_gib = shutil.disk_usage(output_dir.parent).free / 1024**3
    if free_gib < 10.0:
        raise RuntimeError(f"Insufficient free disk for checkpoints: {free_gib:.2f} GiB < 10 GiB")
    logging.info("Training plan: dataset=%s objective=%s rows=%s epochs=%d", args.dataset, args.objective, {key: len(value) for key, value in raw.items()}, args.epochs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {
        split: EncodedDataset(rows, tokenizer, args.max_input_tokens, f"[preflight {index}/3 tokenize:{args.dataset}:{split}]")
        for index, (split, rows) in enumerate(raw.items(), 1)
    }
    logging.info("Preflight complete: encoded=%s overlength=%s max_tokens=%s", {key: len(value) for key, value in encoded.items()}, {key: value.excluded_overlength for key, value in encoded.items()}, {key: value.max_observed_tokens for key, value in encoded.items()})
    if args.preflight_only:
        return

    collator = Collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(encoded["train"], batch_size=args.train_examples_per_batch, shuffle=True, generator=generator, collate_fn=collator, num_workers=0, pin_memory=True),
        "val": DataLoader(encoded["val"], batch_size=args.eval_examples_per_batch, shuffle=False, collate_fn=collator, num_workers=0, pin_memory=True),
        "test": DataLoader(encoded["test"], batch_size=args.eval_examples_per_batch, shuffle=False, collate_fn=collator, num_workers=0, pin_memory=True),
    }
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    logging.info("Loading target Llama for LoRA: %s", args.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=dtype, attn_implementation=args.attn_implementation, low_cpu_mem_usage=True)
    model = get_peft_model(
        base_model,
        LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=list(args.lora_target_modules), bias="none", task_type="CAUSAL_LM"),
    )
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = torch.device(args.device)
    model.to(device)
    model.print_trainable_parameters()
    selected_ids = choice_ids(tokenizer, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_updates = max(1, args.epochs * updates_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(round(total_updates * args.warmup_ratio)), total_updates)

    checkpoint_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "best_adapter_state.pt"
    start_epoch = 1
    best_rank: tuple[float, ...] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Checkpoint contract mismatch")
        set_peft_model_state_dict(model, checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rank = tuple(checkpoint["best_rank"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint["history"])
        logging.info("Resuming after durable epoch %d", start_epoch - 1)

    measured_epoch_seconds: list[float] = []
    last_validation_seconds = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = StageProgress(len(encoded["train"]), f"[train epoch {epoch}/{args.epochs}:{args.dataset}]")
        loss_sum = 0.0
        count_sum = 0
        accumulation = 0
        for batch_index, batch in enumerate(loaders["train"], 1):
            document, question = forward_logits(model, batch, selected_ids, device)
            losses = objective_loss(args, document, question, batch, device)
            (losses["loss"] / args.gradient_accumulation_steps).backward()
            accumulation += 1
            count = len(batch["rows"])
            count_sum += count
            loss_sum += float(losses["loss"].detach()) * count
            if accumulation == args.gradient_accumulation_steps or batch_index == len(loaders["train"]):
                if accumulation != args.gradient_accumulation_steps:
                    factor = args.gradient_accumulation_steps / accumulation
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.mul_(factor)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
            progress.set_detail(f"batch={batch_index}/{len(loaders['train'])} loss={float(losses['loss'].detach()):.4f}")
            progress.update(count)
        progress.close()
        validation_started = time.time()
        val_metrics, _ = evaluate(model, loaders["val"], selected_ids, device, f"[validation epoch {epoch}/{args.epochs}:{args.dataset}]")
        last_validation_seconds = time.time() - validation_started
        current_selection = selection(val_metrics, args)
        current_rank = tuple(float(value) for value in current_selection["rank"])
        improved = best_rank is None or current_rank > best_rank
        if improved:
            best_rank = current_rank
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch(best_path, {"adapter": get_peft_model_state_dict(model), "epoch": epoch, "selection": current_selection})
        else:
            bad_epochs += 1
        measured_epoch_seconds.append(time.time() - epoch_started)
        will_stop = bad_epochs >= args.patience
        remaining_epochs = 0 if will_stop else args.epochs - epoch
        average_epoch_seconds = sum(measured_epoch_seconds) / len(measured_epoch_seconds)
        estimated_test_seconds = (
            last_validation_seconds * len(encoded["test"]) / max(1, len(encoded["val"]))
        )
        estimated_remaining = remaining_epochs * average_epoch_seconds + estimated_test_seconds
        record = {
            "epoch": epoch,
            "duration_seconds": measured_epoch_seconds[-1],
            "estimated_remaining_seconds": estimated_remaining,
            "train_loss": loss_sum / max(1, count_sum),
            "validation": val_metrics,
            "selection": current_selection,
            "best": improved,
        }
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        atomic_torch(checkpoint_path, {"contract_fingerprint": contract_hash, "epoch": epoch, "best_rank": list(best_rank), "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history, "adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
        logging.info(
            "Epoch %d complete: duration=%s loss=%.4f primary_DS_given_noRAG_wrong=%.4f "
            "noRAG_delta=%+.4f feasible=%s best=%d estimated_remaining=%s",
            epoch,
            format_duration(measured_epoch_seconds[-1]),
            record["train_loss"],
            current_selection["primary_accuracy"],
            val_metrics["question_level_no_rag"]["absolute_accuracy_delta"],
            current_selection["feasible"],
            best_epoch,
            format_duration(estimated_remaining),
        )
        if will_stop:
            logging.info("Early stopping after epoch %d; durable checkpoint=%s", epoch, checkpoint_path)
            break

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was written")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, best["adapter"])
    test_metrics, predictions = evaluate(model, loaders["test"], selected_ids, device, f"[held-out test:{args.dataset}]")
    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    atomic_jsonl(output_dir / "test_predictions.jsonl", predictions)
    summary = {
        **contract,
        "contract_fingerprint": contract_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoded": {split: len(value) for split, value in encoded.items()},
        "excluded_overlength": {split: value.excluded_overlength for split, value in encoded.items()},
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "best_validation_selection": history[best_epoch - 1]["selection"],
        "test": test_metrics,
        "pilot_success": pilot_success(test_metrics),
        "test_selection_audit": selection(test_metrics, args),
        "final_model": str(final_dir.resolve()),
    }
    atomic_json(output_dir / "summary.json", summary)
    write_summary_markdown(output_dir / "summary_table.md", summary)
    logging.info("Training complete: best_epoch=%d output=%s", best_epoch, output_dir)
    logging.info("Held-out test: %s", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

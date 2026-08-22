#!/usr/bin/env python3
"""Train a scalar RAG utility scorer from text, h0, and hD-h0.

Gold-derived answer direction ``c`` is used upstream only to construct the
continuous teacher score stored in the split.  It is never opened or supplied
to this model.  Questions are batched with all of their retrieved documents so
question-centered interventions and within-question ranking are well defined.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import socket
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from safetensors import safe_open
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_latent_utility import (
    MODEL_VERSION,
    LatentUtilityConfig,
    LatentUtilityScorer,
    split_official_filter_input,
)


TRAINER_VERSION = "rag2_latent_utility_trainer_v1"
NO_RAG_STATES = ("no_rag_correct", "no_rag_wrong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--hidden-feature-root", type=Path, default=None)
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models" / "Flan-T5-large")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-label-threshold", type=float, default=0.4)
    parser.add_argument("--expected-label-mode", default="positive_vs_rest")
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--documents-per-train-batch", type=int, default=32)
    parser.add_argument("--documents-per-eval-batch", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--text-encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trainable-text-encoder-layers", type=int, default=4)
    parser.add_argument("--max-question-tokens", type=int, default=512)
    parser.add_argument("--max-document-tokens", type=int, default=384)
    parser.add_argument("--score-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--soft-target-temperature", type=float, default=0.15)
    parser.add_argument("--decision-temperature", type=float, default=0.15)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
    parser.add_argument("--ranking-min-score-gap", type=float, default=0.05)
    parser.add_argument("--group-dro-eta", type=float, default=0.05)
    parser.add_argument(
        "--group-dro-min-weight",
        type=float,
        default=0.1,
        help="Minimum loss mass retained for each no-RAG state to prevent group collapse.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--hidden-shard-cache-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-eval-questions", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def no_rag_state(transition: Any) -> int:
    value = str(transition).strip().upper()
    if value.startswith("C->"):
        return 0
    if value.startswith("W->"):
        return 1
    raise ValueError(f"Unsupported answer transition: {transition!r}")


def load_contract(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    manifest_path = args.split_root / args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("materialization_version") != "rag2_hidden_utility_filter_inputs_v1":
        raise RuntimeError(f"Unsupported split contract: {manifest_path}")
    if manifest.get("dataset") != args.dataset:
        raise RuntimeError("Split dataset mismatch")
    if not math.isclose(float(manifest["threshold"]), args.expected_label_threshold, abs_tol=1e-12):
        raise RuntimeError(f"Threshold mismatch: split={manifest['threshold']} requested={args.expected_label_threshold}")
    if str(manifest.get("label_mode")) != args.expected_label_mode:
        raise RuntimeError("Label mode mismatch")
    forbidden = set((manifest.get("model_input_contract") or {}).get("forbidden_as_model_inputs") or [])
    required = {"gold-derived c", "projection score", "gold answer", "answer transition"}
    if not required.issubset(forbidden):
        raise RuntimeError("Split does not declare all leakage exclusions")
    hidden_root = (args.hidden_feature_root or Path(str(manifest["hidden_feature_dir"]))).resolve()
    hidden_manifest = json.loads((hidden_root / "run_manifest.json").read_text(encoding="utf-8"))
    layers = [str(value).removeprefix("layer_") for value in hidden_manifest.get("layers", [])]
    if hidden_manifest.get("dataset") != args.dataset or layers != ["28"]:
        raise RuntimeError("Hidden feature root is not the expected layer-28 dataset")
    return manifest, hidden_root


def load_splits(args: argparse.Namespace) -> dict[str, Dataset]:
    root = args.split_root / args.dataset
    paths = {"train": root / "train.jsonl", "validation": root / "val.jsonl", "test": root / "test.jsonl"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = PROJECT_ROOT / "cache" / "hf_latent_utility"
    cache.mkdir(parents=True, exist_ok=True)
    loaded = load_dataset("json", data_files={key: str(path) for key, path in paths.items()}, cache_dir=str(cache))
    return {key: loaded[key] for key in paths}


def select_question_limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    for index, (shard, question_row) in enumerate(zip(dataset["feature_shard_index"], dataset["feature_question_row"])):
        key = (int(shard), int(question_row))
        if key not in seen and len(seen) >= int(limit):
            break
        seen.add(key)
        selected.append(index)
    return dataset.select(selected)


class SafeTensorFeatureStore:
    """Memory-mapped h0/hD store that cannot expose c or cached projections."""

    def __init__(self, root: Path, cache_size: int) -> None:
        self.root = root
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
        first = root / "shards" / "shard_00000"
        with safe_open(str(first / "question_features.safetensors"), framework="pt", device="cpu") as handle:
            self.hidden_size = int(handle.get_slice("h0").get_shape()[-1])

    def _handles(self, shard: int) -> tuple[Any, Any]:
        cached = self.cache.pop(shard, None)
        if cached is not None:
            self.cache[shard] = cached
            return cached
        root = self.root / "shards" / f"shard_{shard:05d}"
        handles = (
            safe_open(str(root / "question_features.safetensors"), framework="pt", device="cpu"),
            safe_open(str(root / "pair_features.safetensors"), framework="pt", device="cpu"),
        )
        self.cache[shard] = handles
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return handles

    def get(self, shard: int, question_row: int, pair_row: int) -> tuple[torch.Tensor, torch.Tensor]:
        question, pair = self._handles(int(shard))
        h0 = question.get_slice("h0")[int(question_row), 0, :]
        hD = pair.get_slice("hD")[int(pair_row), 0, :]
        return h0, hD - h0


class QuestionBatchSampler(Sampler[list[int]]):
    """Keep complete question-document groups together and preserve shard locality."""

    def __init__(self, dataset: Dataset, batch_size: int, seed: int, shuffle: bool) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (shard, question) in enumerate(zip(dataset["feature_shard_index"], dataset["feature_question_row"])):
            grouped[int(shard)][int(question)].append(index)
        self.batches: dict[int, list[list[int]]] = {}
        for shard, questions in grouped.items():
            packed: list[list[int]] = []
            current: list[int] = []
            for _, indices in sorted(questions.items()):
                if len(indices) > self.batch_size:
                    raise ValueError(f"Question has {len(indices)} documents but batch capacity is {self.batch_size}")
                if current and len(current) + len(indices) > self.batch_size:
                    packed.append(current)
                    current = []
                current.extend(indices)
            if current:
                packed.append(current)
            self.batches[shard] = packed
        self.length = sum(len(value) for value in self.batches.values())

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.batches)
        if self.shuffle:
            generator.shuffle(shards)
        for shard in shards:
            batches = [list(value) for value in self.batches[shard]]
            if self.shuffle:
                generator.shuffle(batches)
            for batch in batches:
                if self.shuffle:
                    generator.shuffle(batch)
                yield batch


class LatentUtilityCollator:
    def __init__(self, tokenizer: Any, store: SafeTensorFeatureStore, args: argparse.Namespace) -> None:
        self.tokenizer = tokenizer
        self.store = store
        self.args = args

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        question_ids: dict[tuple[int, int], int] = {}
        question_texts: list[str] = []
        h0_values: list[torch.Tensor] = []
        documents: list[str] = []
        deltas: list[torch.Tensor] = []
        doc_to_question: list[int] = []
        scores: list[float] = []
        labels: list[int] = []
        states: list[int] = []
        sample_ids: list[str] = []
        pair_ids: list[str] = []
        for row in rows:
            question, evidence = split_official_filter_input(str(row["input"]))
            key = (int(row["feature_shard_index"]), int(row["feature_question_row"]))
            h0, delta = self.store.get(key[0], key[1], int(row["feature_pair_row"]))
            if key not in question_ids:
                question_ids[key] = len(question_ids)
                question_texts.append(question)
                h0_values.append(h0)
            elif question_texts[question_ids[key]] != question:
                raise RuntimeError(f"Question text mismatch inside group {key}")
            doc_to_question.append(question_ids[key])
            documents.append(evidence)
            deltas.append(delta)
            score = float(row["hidden_projection_score_audit_only"])
            label = int(score > self.args.expected_label_threshold)
            expected = 1 if str(row["target"]) == "helpful" else 0
            if label != expected:
                raise RuntimeError(f"Continuous/binary target mismatch for {row['pair_id']}")
            scores.append(score)
            labels.append(label)
            states.append(no_rag_state(row["answer_transition_audit_only"]))
            sample_ids.append(str(row["sample_id"]))
            pair_ids.append(str(row["pair_id"]))
        q = self.tokenizer(
            question_texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_question_tokens,
            return_tensors="pt",
            pad_to_multiple_of=8 if self.args.bf16 else None,
        )
        d = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=self.args.max_document_tokens,
            return_tensors="pt",
            pad_to_multiple_of=8 if self.args.bf16 else None,
        )
        return {
            "question_input_ids": q["input_ids"],
            "question_attention_mask": q["attention_mask"],
            "document_input_ids": d["input_ids"],
            "document_attention_mask": d["attention_mask"],
            "h0": torch.stack(h0_values),
            "delta_h": torch.stack(deltas),
            "document_to_question": torch.tensor(doc_to_question, dtype=torch.long),
            "teacher_score": torch.tensor(scores, dtype=torch.float32),
            "binary_target": torch.tensor(labels, dtype=torch.float32),
            "no_rag_state": torch.tensor(states, dtype=torch.long),
            "sample_ids": sample_ids,
            "pair_ids": pair_ids,
        }


def move_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "question_input_ids",
        "question_attention_mask",
        "document_input_ids",
        "document_attention_mask",
        "h0",
        "delta_h",
        "document_to_question",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


class GroupDROObjective:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.weights = torch.full((2,), 0.5, dtype=torch.float32, device=device)

    def state_dict(self) -> dict[str, Any]:
        return {"weights": self.weights.detach().cpu()}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        self.weights.copy_(value["weights"].to(self.weights.device))

    def __call__(
        self,
        predicted: torch.Tensor,
        teacher: torch.Tensor,
        states: torch.Tensor,
        document_to_question: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        threshold = float(self.args.expected_label_threshold)
        score_loss = F.smooth_l1_loss(predicted, teacher, reduction="none")
        soft_target = torch.sigmoid((teacher - threshold) / float(self.args.soft_target_temperature))
        binary_logits = (predicted - threshold) / float(self.args.decision_temperature)
        binary_loss = F.binary_cross_entropy_with_logits(binary_logits, soft_target, reduction="none")
        absolute = self.args.score_loss_weight * score_loss + self.args.binary_loss_weight * binary_loss

        ranking_by_state: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        ranking_pairs = 0
        for question_index in torch.unique(document_to_question):
            mask = document_to_question.eq(question_index)
            q_pred = predicted[mask]
            q_teacher = teacher[mask]
            if q_pred.numel() < 2:
                continue
            upper = torch.triu_indices(q_pred.numel(), q_pred.numel(), offset=1, device=q_pred.device)
            target_difference = q_teacher[upper[0]] - q_teacher[upper[1]]
            usable = target_difference.abs().ge(float(self.args.ranking_min_score_gap))
            if not bool(usable.any()):
                continue
            prediction_difference = q_pred[upper[0]] - q_pred[upper[1]]
            sign = target_difference[usable].sign()
            losses = F.softplus(
                -sign * prediction_difference[usable] / float(self.args.ranking_temperature)
            )
            state_values = torch.unique(states[mask])
            if state_values.numel() != 1:
                raise RuntimeError("No-RAG state is inconsistent within a question")
            ranking_by_state[int(state_values.item())].append(losses.mean())
            ranking_pairs += int(usable.sum().item())

        group_losses: list[torch.Tensor] = []
        present: list[bool] = []
        for state in range(2):
            mask = states.eq(state)
            if bool(mask.any()):
                value = absolute[mask].mean()
                if ranking_by_state[state]:
                    value = value + self.args.ranking_loss_weight * torch.stack(ranking_by_state[state]).mean()
                group_losses.append(value)
                present.append(True)
            else:
                group_losses.append(predicted.sum() * 0.0)
                present.append(False)
        stacked = torch.stack(group_losses)
        if training and all(present):
            with torch.no_grad():
                self.weights.mul_(torch.exp(float(self.args.group_dro_eta) * stacked.detach()))
                self.weights.div_(self.weights.sum().clamp_min(1e-12))
                floor = float(self.args.group_dro_min_weight)
                self.weights.clamp_(min=floor, max=1.0 - floor)
                self.weights.div_(self.weights.sum().clamp_min(1e-12))
        active_weights = self.weights * torch.tensor(present, dtype=self.weights.dtype, device=self.weights.device)
        active_weights = active_weights / active_weights.sum().clamp_min(1e-12)
        total = (active_weights * stacked).sum()
        details = {
            "loss": float(total.detach().cpu()),
            "score_loss": float(score_loss.mean().detach().cpu()),
            "binary_loss": float(binary_loss.mean().detach().cpu()),
            "loss_no_rag_correct": float(stacked[0].detach().cpu()),
            "loss_no_rag_wrong": float(stacked[1].detach().cpu()),
            "dro_weight_correct": float(self.weights[0].detach().cpu()),
            "dro_weight_wrong": float(self.weights[1].detach().cpu()),
            "ranking_pairs": float(ranking_pairs),
        }
        return total, details


def safe_binary_metrics(target: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {
        "n": float(target.size),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "predicted_helpful_fraction": float(predicted.mean()),
        "actual_helpful_fraction": float(target.mean()),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=[1, 0], zero_division=0
    )
    result.update(
        {
            "helpful_precision": float(precision[0]),
            "helpful_recall": float(recall[0]),
            "helpful_f1": float(f1[0]),
            "helpful_support": float(support[0]),
            "not_helpful_precision": float(precision[1]),
            "not_helpful_recall": float(recall[1]),
            "not_helpful_f1": float(f1[1]),
            "not_helpful_support": float(support[1]),
            "macro_f1": float(np.mean(f1)),
        }
    )
    if np.unique(target).size == 2:
        result["auroc"] = float(roc_auc_score(target, probability))
        result["auprc"] = float(average_precision_score(target, probability))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


@torch.no_grad()
def evaluate(
    model: LatentUtilityScorer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    description: str,
    overall_progress: tqdm | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    states: list[np.ndarray] = []
    sample_ids: list[str] = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16)
    with autocast:
        iterator = loader if overall_progress is not None else tqdm(loader, desc=description, unit="batch")
        if overall_progress is not None:
            overall_progress.set_postfix(stage=description, refresh=True)
        for batch in iterator:
            output = model(**move_model_inputs(batch, device))
            predictions.append(output["utility_score"].float().cpu().numpy())
            teachers.append(batch["teacher_score"].numpy())
            states.append(batch["no_rag_state"].numpy())
            sample_ids.extend(batch["sample_ids"])
            if overall_progress is not None:
                overall_progress.update(len(batch["pair_ids"]))
    predicted_score = np.concatenate(predictions)
    teacher_score = np.concatenate(teachers)
    state = np.concatenate(states)
    threshold = float(args.expected_label_threshold)
    binary_target = (teacher_score > threshold).astype(np.int64)
    binary_prediction = (predicted_score > threshold).astype(np.int64)
    probability = 1.0 / (1.0 + np.exp(-(predicted_score - threshold) / args.decision_temperature))
    result: dict[str, Any] = {
        "overall": safe_binary_metrics(binary_target, binary_prediction, probability),
        "no_rag_correct": safe_binary_metrics(
            binary_target[state == 0], binary_prediction[state == 0], probability[state == 0]
        ),
        "no_rag_wrong": safe_binary_metrics(
            binary_target[state == 1], binary_prediction[state == 1], probability[state == 1]
        ),
        "continuous": {
            "mae": float(np.mean(np.abs(predicted_score - teacher_score))),
            "rmse": float(np.sqrt(np.mean(np.square(predicted_score - teacher_score)))),
            "pearson": float(np.corrcoef(predicted_score, teacher_score)[0, 1]),
            "spearman": float(spearmanr(predicted_score, teacher_score).statistic),
        },
    }
    comparisons = correct = 0
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped[sample_id].append(index)
    for indices in grouped.values():
        values = np.asarray(indices, dtype=np.int64)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                target_diff = teacher_score[values[left]] - teacher_score[values[right]]
                if abs(target_diff) < args.ranking_min_score_gap:
                    continue
                pred_diff = predicted_score[values[left]] - predicted_score[values[right]]
                comparisons += 1
                correct += int(np.sign(target_diff) == np.sign(pred_diff))
    result["within_question_ranking"] = {
        "comparisons": comparisons,
        "accuracy": correct / comparisons if comparisons else float("nan"),
    }
    c_auc = result["no_rag_correct"]["auroc"]
    w_auc = result["no_rag_wrong"]["auroc"]
    result["selection_metric_worst_group_auroc"] = float(min(c_auc, w_auc))
    return result


def make_loader(
    dataset: Dataset,
    collator: LatentUtilityCollator,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=QuestionBatchSampler(dataset, batch_size, seed, shuffle),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def optimizer_for(model: LatentUtilityScorer, args: argparse.Namespace) -> torch.optim.Optimizer:
    text_parameters: list[torch.nn.Parameter] = []
    head_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (text_parameters if name.startswith("text_encoder.") else head_parameters).append(parameter)
    groups = [
        {"params": head_parameters, "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
    ]
    if text_parameters:
        groups.append(
            {"params": text_parameters, "lr": args.text_encoder_learning_rate, "weight_decay": args.weight_decay}
        )
    return torch.optim.AdamW(groups)


def save_checkpoint(
    path: Path,
    model: LatentUtilityScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    objective: GroupDROObjective,
    state: dict[str, Any],
) -> None:
    model.save_trainable(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "group_dro": objective.state_dict(),
            "trainer_state": state,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "training_state.pt",
    )
    atomic_json(path / "trainer_state.json", state)


def load_checkpoint(
    path: Path,
    model: LatentUtilityScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    objective: GroupDROObjective,
) -> dict[str, Any]:
    weights = torch.load(path / "trainable_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable.intersection(missing) or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={sorted(trainable.intersection(missing))} unexpected={unexpected}")
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    objective.load_state_dict(payload["group_dro"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and payload["cuda_random_state"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    return dict(payload["trainer_state"])


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("This trainer requires a CUDA GPU")
    if args.documents_per_train_batch < 8 or args.documents_per_eval_batch < 8:
        raise ValueError("Batch capacity must fit a complete top-8 question group")
    if args.gradient_accumulation_steps < 1 or args.early_stopping_patience < 0:
        raise ValueError("Invalid accumulation or early-stopping setting")
    if not 0.0 <= args.group_dro_min_weight < 0.5:
        raise ValueError("group-dro-min-weight must be in [0, 0.5)")
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    manifest, hidden_root = load_contract(args)
    datasets = load_splits(args)
    datasets["train"] = select_question_limit(datasets["train"], args.max_train_questions)
    datasets["validation"] = select_question_limit(datasets["validation"], args.max_eval_questions)
    datasets["test"] = select_question_limit(datasets["test"], args.max_eval_questions)

    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        output_dir = checkpoint.parent if checkpoint.name == "last_checkpoint" else checkpoint
        if not (checkpoint / "training_state.pt").is_file():
            raise FileNotFoundError(checkpoint / "training_state.pt")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_root / args.dataset / args.run_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), use_fast=True)
    store = SafeTensorFeatureStore(hidden_root, args.hidden_shard_cache_size)
    collator = LatentUtilityCollator(tokenizer, store, args)
    loaders = {
        "train": make_loader(datasets["train"], collator, args.documents_per_train_batch, args.seed, True),
        "validation": make_loader(datasets["validation"], collator, args.documents_per_eval_batch, args.seed, False),
        "test": make_loader(datasets["test"], collator, args.documents_per_eval_batch, args.seed, False),
    }
    data_summary = {
        name: {
            "pairs": len(dataset),
            "questions": len(set(zip(dataset["feature_shard_index"], dataset["feature_question_row"]))),
            "labels": dict(Counter(str(value) for value in dataset["target"])),
            "no_rag_states": dict(
                Counter(NO_RAG_STATES[no_rag_state(value)] for value in dataset["answer_transition_audit_only"])
            ),
            "batches": len(loaders[name]),
        }
        for name, dataset in datasets.items()
    }
    config = LatentUtilityConfig(
        base_model_name_or_path=str(args.model_name_or_path.resolve()),
        hidden_size=store.hidden_size,
        latent_size=args.latent_size,
        dropout=args.dropout,
        trainable_text_encoder_layers=args.trainable_text_encoder_layers,
        decision_threshold=args.expected_label_threshold,
        decision_temperature=args.decision_temperature,
    )
    reproduction = {
        "trainer_version": TRAINER_VERSION,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": args.dataset,
        "split_manifest": manifest,
        "hidden_feature_root": str(hidden_root),
        "model_input": [
            "question and options text",
            "document text",
            "h0",
            "delta_h=hD-h0",
            "delta_h minus within-question mean(delta_h)",
            "log(1+norm(delta_h))",
        ],
        "forbidden_model_input": [
            "gold answer",
            "gold-derived c",
            "teacher utility score",
            "binary label",
            "answer transition/no-RAG correctness",
            "dataset ID",
        ],
        "supervision": {
            "continuous": "Huber(predicted_score, delta_h dot unit_c)",
            "soft_binary": f"BCE around fixed tau={args.expected_label_threshold}",
            "within_question_ranking": "teacher-score ordering",
            "robustness": "GroupDRO over no-RAG correct/wrong; group is training-only metadata",
        },
        "checkpoint_selection": "minimum validation AUROC over no-RAG correct and wrong groups",
        "config": vars(args) | {"split_root": str(args.split_root), "hidden_feature_root": str(hidden_root), "model_name_or_path": str(args.model_name_or_path), "output_root": str(args.output_root), "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None},
        "data_summary": data_summary,
    }
    atomic_json(output_dir / "reproduction_manifest.json", reproduction)
    logging.info("Output directory: %s", output_dir)
    logging.info("Data summary: %s", json.dumps(data_summary, ensure_ascii=False))
    if args.dry_run:
        batch = next(iter(loaders["train"]))
        atomic_json(
            output_dir / "dry_run_report.json",
            {
                "question_shape": list(batch["question_input_ids"].shape),
                "document_shape": list(batch["document_input_ids"].shape),
                "h0_shape": list(batch["h0"].shape),
                "delta_h_shape": list(batch["delta_h"].shape),
                "teacher_score_range": [float(batch["teacher_score"].min()), float(batch["teacher_score"].max())],
                "leakage_check": "feature store opened only h0 and hD",
            },
        )
        logging.info("Dry run complete")
        return

    device = torch.device("cuda:0")
    model = LatentUtilityScorer(config)
    if args.gradient_checkpointing and args.trainable_text_encoder_layers:
        model.text_encoder.gradient_checkpointing_enable()
        # The embedding and lower encoder blocks are intentionally frozen.
        # Checkpointed trainable upper blocks still need a grad-bearing input.
        model.text_encoder.enable_input_require_grads()
    model.to(device)
    optimizer = optimizer_for(model, args)
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.num_train_epochs
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    objective = GroupDROObjective(args, device)
    trainer_state: dict[str, Any] = {
        "completed_epoch": 0,
        "global_step": 0,
        "best_metric": -float("inf"),
        "best_epoch": None,
        "epochs_without_improvement": 0,
        "history": [],
    }
    if args.resume_from_checkpoint:
        trainer_state = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, objective)
        logging.info("Resumed from %s at epoch %s", args.resume_from_checkpoint, trainer_state["completed_epoch"])

    work_per_epoch = len(datasets["train"]) + len(datasets["validation"])
    final_evaluation_work = len(datasets["validation"]) + len(datasets["test"])
    total_work = args.num_train_epochs * work_per_epoch + final_evaluation_work
    completed_work = int(trainer_state["completed_epoch"]) * work_per_epoch
    overall_progress = tqdm(
        total=total_work,
        initial=completed_work,
        desc=f"LatentUtilityOverall:{args.dataset}",
        unit="pair",
        dynamic_ncols=True,
    )
    try:
        for epoch in range(int(trainer_state["completed_epoch"]) + 1, args.num_train_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rolling: Counter[str] = Counter()
            overall_progress.set_postfix(
                stage=f"train epoch {epoch}/{args.num_train_epochs}",
                epoch_batch=f"0/{len(loaders['train'])}",
                refresh=True,
            )
            for batch_index, batch in enumerate(loaders["train"], start=1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    output = model(**move_model_inputs(batch, device))
                    loss, details = objective(
                        output["utility_score"],
                        batch["teacher_score"].to(device),
                        batch["no_rag_state"].to(device),
                        batch["document_to_question"].to(device),
                        training=True,
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                update = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(loaders["train"])
                if update:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (parameter for parameter in model.parameters() if parameter.requires_grad),
                            args.max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    trainer_state["global_step"] += 1
                for key, value in details.items():
                    rolling[key] += value
                overall_progress.update(len(batch["pair_ids"]))
                if batch_index % args.logging_steps == 0:
                    overall_progress.set_postfix(
                        stage=f"train epoch {epoch}/{args.num_train_epochs}",
                        epoch_batch=f"{batch_index}/{len(loaders['train'])}",
                        loss=f"{rolling['loss']/args.logging_steps:.3f}",
                        dro=f"{objective.weights[0].item():.2f}/{objective.weights[1].item():.2f}",
                        refresh=True,
                    )
                    rolling.clear()
            validation = evaluate(
                model,
                loaders["validation"],
                device,
                args,
                f"validation epoch {epoch}/{args.num_train_epochs}",
                overall_progress=overall_progress,
            )
            metric = float(validation["selection_metric_worst_group_auroc"])
            improved = metric > float(trainer_state["best_metric"]) + args.minimum_improvement
            trainer_state["completed_epoch"] = epoch
            trainer_state["history"].append({"epoch": epoch, "validation": validation, "group_dro_weights": objective.weights.detach().cpu().tolist()})
            if improved:
                trainer_state["best_metric"] = metric
                trainer_state["best_epoch"] = epoch
                trainer_state["epochs_without_improvement"] = 0
                model.save_trainable(output_dir / "best_model")
                atomic_json(output_dir / "best_model" / "validation_metrics.json", validation)
            else:
                trainer_state["epochs_without_improvement"] += 1
            save_checkpoint(output_dir / "last_checkpoint", model, optimizer, scheduler, objective, trainer_state)
            atomic_json(output_dir / "training_history.json", trainer_state)
            logging.info(
                "Epoch %d: robust_auc=%.4f C=%.4f W=%.4f macro_f1=%.4f best_epoch=%s",
                epoch,
                metric,
                validation["no_rag_correct"]["auroc"],
                validation["no_rag_wrong"]["auroc"],
                validation["overall"]["macro_f1"],
                trainer_state["best_epoch"],
            )
            if trainer_state["epochs_without_improvement"] >= args.early_stopping_patience > 0:
                logging.info("Early stopping after %d epochs without robust-AUROC improvement", trainer_state["epochs_without_improvement"])
                # Remove work belonging to epochs that early stopping skipped;
                # the final best-model validation/test still remains.
                overall_progress.total = overall_progress.n + final_evaluation_work
                overall_progress.refresh()
                break

        best_weights = torch.load(output_dir / "best_model" / "trainable_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(best_weights, strict=False)
        validation_best = evaluate(
            model,
            loaders["validation"],
            device,
            args,
            "final best-model validation",
            overall_progress=overall_progress,
        )
        test_best = evaluate(
            model,
            loaders["test"],
            device,
            args,
            "final best-model test",
            overall_progress=overall_progress,
        )
    finally:
        overall_progress.close()
    final_dir = output_dir / "final_model"
    model.save_trainable(final_dir)
    tokenizer.save_pretrained(final_dir)
    atomic_json(final_dir / "validation_metrics.json", validation_best)
    atomic_json(final_dir / "test_metrics.json", test_best)
    atomic_json(
        final_dir / "training_contract.json",
        {
            "trainer_version": TRAINER_VERSION,
            "best_epoch": trainer_state["best_epoch"],
            "best_metric": trainer_state["best_metric"],
            "decision_threshold": args.expected_label_threshold,
            "inference_output": "one scalar utility score; helpful iff score > fixed threshold",
            "gold_or_transition_required_at_inference": False,
        },
    )
    logging.info(
        "Complete: best_epoch=%s test_macro_f1=%.4f test_C_auc=%.4f test_W_auc=%.4f output=%s",
        trainer_state["best_epoch"],
        test_best["overall"]["macro_f1"],
        test_best["no_rag_correct"]["auroc"],
        test_best["no_rag_wrong"]["auroc"],
        final_dir,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

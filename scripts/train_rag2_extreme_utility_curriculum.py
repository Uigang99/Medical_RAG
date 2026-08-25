#!/usr/bin/env python3
"""Train the two-stage anchored hidden-utility document filter.

Stage 1 learns only high-purity Helpful/Harmful extremes with natural
pointwise supervision plus no-RAG-group-balanced within-question ranking.
Stage 2 replays those extremes and softly calibrates the previously excluded
neutral region.  Gold answers, gold-derived directions, answer transitions,
and teacher utility scores are never model inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import socket
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from datasets import Dataset, load_dataset
from safetensors import safe_open
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_extreme_utility import (  # noqa: E402
    MODEL_VERSION,
    ExtremeUtilityConfig,
    ExtremeUtilityScorer,
    extreme_curriculum_loss,
)
from medrag.filtering.rag2_official import build_official_filter_input, format_options  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402


TRAINER_VERSION = "rag2_extreme_utility_curriculum_trainer_v1"
BAND_TO_INT = {
    "extreme_harmful": -2,
    "neutral_negative": -1,
    "neutral_zero": 0,
    "neutral_positive": 1,
    "extreme_helpful": 2,
}


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=base / "hidden_utility_extreme_curriculum_v1/prepared"
    )
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Flan-T5-large")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("extreme", "neutral"), required=True)
    parser.add_argument("--stage1-model", type=Path, default=None)
    parser.add_argument("--input-mode", choices=("text_only", "text_delta", "text_h0_delta"), default="text_delta")
    parser.add_argument("--num-train-epochs", type=int, required=True)
    parser.add_argument("--documents-per-train-batch", type=int, default=32)
    parser.add_argument("--documents-per-eval-batch", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--text-encoder-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trainable-text-encoder-layers", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--neutral-loss-weight", type=float, default=0.1)
    parser.add_argument("--stage2-max-extreme-auroc-drop", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--hidden-shard-cache-size", type=int, default=8)
    parser.add_argument("--trace-shard-cache-size", type=int, default=8)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-eval-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error
    return result


def load_contract(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.prepared_root / "filter_inputs" / args.dataset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("materialization_version") != "rag2_anchored_extreme_utility_dataset_v1":
        raise RuntimeError(f"Unsupported prepared-data contract: {manifest_path}")
    if manifest.get("dataset") != args.dataset:
        raise RuntimeError("Prepared dataset mismatch")
    included = set((manifest.get("model_input_contract") or {}).get("included") or [])
    forbidden = set((manifest.get("model_input_contract") or {}).get("forbidden") or [])
    if "normalized delta_h=hD-h0" not in included:
        raise RuntimeError("Prepared contract does not declare delta_h")
    required_forbidden = {"gold answer", "gold-derived c", "projection score", "answer transition"}
    if not required_forbidden.issubset(forbidden):
        raise RuntimeError("Prepared contract does not declare every leakage exclusion")
    if args.stage == "neutral" and args.stage1_model is None:
        raise ValueError("--stage1-model is required for neutral calibration")
    return manifest


def load_splits(args: argparse.Namespace) -> dict[str, Dataset]:
    root = args.prepared_root / "filter_inputs" / args.dataset
    paths = {"train": root / "train.jsonl", "validation": root / "val.jsonl", "test": root / "test.jsonl"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = PROJECT_ROOT / "cache/hf_extreme_utility"
    cache.mkdir(parents=True, exist_ok=True)
    logging.info("Pipeline stage data/4: loading prepared pointer splits (HF progress shows this stage ETA)")
    loaded = load_dataset("json", data_files={key: str(path) for key, path in paths.items()}, cache_dir=str(cache))
    result = {key: loaded[key] for key in paths}
    if args.stage == "extreme":
        logging.info("Pipeline stage data/4: selecting high-purity extremes")
        result = {
            key: value.filter(
                lambda band: str(band).startswith("extreme_"),
                input_columns=["curriculum_band"],
                desc=f"select-extremes:{args.dataset}:{key}",
            )
            for key, value in result.items()
        }
    return result


def limit_questions(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    selected: list[int] = []
    seen: set[str] = set()
    for index, sample_id in enumerate(dataset["sample_id"]):
        value = str(sample_id)
        if value not in seen and len(seen) >= int(limit):
            break
        seen.add(value)
        selected.append(index)
    return dataset.select(selected)


class TraceFeatureStore:
    """Read only text, h0, and hD; gold directions/scores are inaccessible."""

    def __init__(self, manifest: dict[str, Any], args: argparse.Namespace) -> None:
        self.no_rag_root = Path(str(manifest["no_rag_root"]))
        self.document_root = Path(str(manifest["document_root"]))
        self.dataset = args.dataset
        self.split = str(manifest["source_split"])
        no_manifest = json.loads((self.no_rag_root / "feature_manifest.json").read_text(encoding="utf-8"))
        layers = [int(value) for value in no_manifest["layers"]]
        anchors = [str(value) for value in no_manifest["anchor_order"]]
        self.layer_index = layers.index(int(manifest["layer"]))
        self.anchor_index = anchors.index(str(manifest["anchor"]))
        self.hidden_size = int(no_manifest["hidden_size"])
        self.hidden_capacity = max(1, int(args.hidden_shard_cache_size))
        self.trace_capacity = max(1, int(args.trace_shard_cache_size))
        self.question_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.document_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.trace_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    @staticmethod
    def _insert(cache: OrderedDict[str, Any], key: str, value: Any, capacity: int) -> Any:
        cache[key] = value
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def _h0(self, shard: str) -> torch.Tensor:
        value = self.question_cache.pop(shard, None)
        if value is not None:
            self.question_cache[shard] = value
            return value
        path = self.no_rag_root / "no_rag_features" / self.dataset / self.split / "shards" / shard / "features.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.get_slice("anchor_hidden")[:, self.layer_index, self.anchor_index, :].clone()
        return self._insert(self.question_cache, shard, value, self.hidden_capacity)

    def _hD(self, shard: str) -> torch.Tensor:
        value = self.document_cache.pop(shard, None)
        if value is not None:
            self.document_cache[shard] = value
            return value
        path = self.document_root / "with_document_features" / self.dataset / self.split / "shards" / shard / "features.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.get_slice("anchor_hidden")[:, self.layer_index, self.anchor_index, :].clone()
        return self._insert(self.document_cache, shard, value, self.hidden_capacity)

    def _traces(self, shard: str) -> list[dict[str, Any]]:
        value = self.trace_cache.pop(shard, None)
        if value is not None:
            self.trace_cache[shard] = value
            return value
        path = self.document_root / "trace_shards" / self.dataset / self.split / shard / "pairs.jsonl"
        value = read_jsonl(path)
        return self._insert(self.trace_cache, shard, value, self.trace_capacity)

    def get(self, row: dict[str, Any]) -> tuple[str, torch.Tensor, torch.Tensor]:
        q_shard = str(row["question_feature_shard"])
        d_shard = str(row["document_feature_shard"])
        q_row = int(row["question_tensor_row"])
        d_row = int(row["document_tensor_row"])
        trace_row = int(row["trace_pair_row"])
        trace = self._traces(str(row["trace_shard"]))[trace_row]
        if trace["pair_id"] != row["pair_id"]:
            raise RuntimeError(f"Trace pointer mismatch: {row['pair_id']}")
        text = build_official_filter_input(
            question=str(trace["question"]),
            options=format_options(trace.get("options") or {}),
            evidence=str(trace.get("document_text_used") or ""),
        )
        h0 = self._h0(q_shard)[q_row]
        delta = self._hD(d_shard)[d_row] - h0
        return text, h0, delta


class QuestionBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: Dataset, capacity: int, seed: int, shuffle: bool) -> None:
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        grouped: dict[str, OrderedDict[str, list[int]]] = defaultdict(OrderedDict)
        sample_shards: dict[str, str] = {}
        for index, (shard, sample_id) in enumerate(zip(dataset["document_feature_shard"], dataset["sample_id"])):
            shard_value = str(shard)
            sample_value = str(sample_id)
            previous = sample_shards.setdefault(sample_value, shard_value)
            if previous != shard_value:
                raise RuntimeError(
                    f"Question {sample_value} spans feature shards {previous}/{shard_value}; "
                    "the complete group cannot be split across batches"
                )
            grouped[shard_value].setdefault(sample_value, []).append(index)
        self.by_shard: dict[str, list[list[int]]] = {}
        for shard, questions in grouped.items():
            batches: list[list[int]] = []
            current: list[int] = []
            for indices in questions.values():
                if len(indices) > self.capacity:
                    raise ValueError(f"Question has {len(indices)} rows but batch capacity={self.capacity}")
                if current and len(current) + len(indices) > self.capacity:
                    batches.append(current)
                    current = []
                current.extend(indices)
            if current:
                batches.append(current)
            self.by_shard[shard] = batches
        self.length = sum(len(value) for value in self.by_shard.values())

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.by_shard)
        if self.shuffle:
            rng.shuffle(shards)
        for shard in shards:
            values = [list(batch) for batch in self.by_shard[shard]]
            if self.shuffle:
                rng.shuffle(values)
            for batch in values:
                if self.shuffle:
                    rng.shuffle(batch)
                yield batch


class Collator:
    def __init__(self, tokenizer: Any, store: TraceFeatureStore, args: argparse.Namespace) -> None:
        self.tokenizer = tokenizer
        self.store = store
        self.args = args

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        h0_values: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        teacher: list[float] = []
        bands: list[int] = []
        states: list[int] = []
        sample_ids: list[str] = []
        pair_ids: list[str] = []
        q_indices: dict[str, int] = {}
        doc_to_question: list[int] = []
        for row in rows:
            text, h0, delta = self.store.get(row)
            texts.append(text)
            h0_values.append(h0)
            deltas.append(delta)
            teacher.append(float(row["utility_projection"]))
            bands.append(BAND_TO_INT[str(row["curriculum_band"])])
            states.append(0 if bool(row["no_rag_correct"]) else 1)
            sample_id = str(row["sample_id"])
            if sample_id not in q_indices:
                q_indices[sample_id] = len(q_indices)
            doc_to_question.append(q_indices[sample_id])
            sample_ids.append(sample_id)
            pair_ids.append(str(row["pair_id"]))
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_input_tokens,
            return_tensors="pt",
            pad_to_multiple_of=8 if self.args.bf16 else None,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "h0": torch.stack(h0_values),
            "delta_h": torch.stack(deltas),
            "teacher_score": torch.tensor(teacher, dtype=torch.float32),
            "band": torch.tensor(bands, dtype=torch.long),
            "no_rag_state": torch.tensor(states, dtype=torch.long),
            "document_to_question": torch.tensor(doc_to_question, dtype=torch.long),
            "sample_ids": sample_ids,
            "pair_ids": pair_ids,
        }


def move_inputs(batch: dict[str, Any], device: torch.device, input_mode: str) -> dict[str, torch.Tensor | None]:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "delta_h": batch["delta_h"].to(device, non_blocking=True) if input_mode != "text_only" else None,
        "h0": batch["h0"].to(device, non_blocking=True) if input_mode == "text_h0_delta" else None,
    }


def make_loader(dataset: Dataset, collator: Collator, capacity: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=QuestionBatchSampler(dataset, capacity, seed, shuffle),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def binary_metrics(target: np.ndarray, predicted: np.ndarray, score: np.ndarray) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=[1, 0], zero_division=0
    )
    result = {
        "n": int(target.size),
        "accuracy": float(np.mean(target == predicted)) if target.size else float("nan"),
        "helpful_precision": float(precision[0]),
        "helpful_recall": float(recall[0]),
        "helpful_f1": float(f1[0]),
        "helpful_support": int(support[0]),
        "harmful_precision": float(precision[1]),
        "harmful_recall": float(recall[1]),
        "harmful_f1": float(f1[1]),
        "harmful_support": int(support[1]),
        "macro_f1": float(np.mean(f1)),
    }
    if target.size and np.unique(target).size == 2:
        result["auroc"] = float(roc_auc_score(target, score))
        result["auprc"] = float(average_precision_score(target, score))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


@torch.no_grad()
def evaluate(
    model: ExtremeUtilityScorer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage_name: str,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    states: list[np.ndarray] = []
    sample_ids: list[str] = []
    progress.set_stage(stage_name, total=len(loader.dataset))
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            score = model(**move_inputs(batch, device, args.input_mode))["utility_score"]
        predictions.append(score.float().cpu().numpy())
        teachers.append(batch["teacher_score"].numpy())
        bands.append(batch["band"].numpy())
        states.append(batch["no_rag_state"].numpy())
        sample_ids.extend(batch["sample_ids"])
        progress.update(len(batch["pair_ids"]))
    predicted = np.concatenate(predictions)
    teacher = np.concatenate(teachers)
    band = np.concatenate(bands)
    state = np.concatenate(states)
    extreme = np.abs(band) == 2
    target = (band[extreme] > 0).astype(np.int64)
    decision = (predicted[extreme] > 0).astype(np.int64)
    result: dict[str, Any] = {
        "extreme_overall": binary_metrics(target, decision, predicted[extreme]),
        "continuous": {
            "n": int(predicted.size),
            "mae_tanh_normalized": float(
                np.mean(np.abs(np.tanh(predicted) - np.clip(teacher / float(args.expected_threshold), -1, 1)))
            ),
            "spearman": float(spearmanr(predicted, teacher).statistic) if predicted.size > 1 else float("nan"),
        },
    }
    for state_value, name in ((0, "no_rag_correct"), (1, "no_rag_wrong")):
        selector = extreme & (state == state_value)
        result[name] = binary_metrics(
            (band[selector] > 0).astype(np.int64),
            (predicted[selector] > 0).astype(np.int64),
            predicted[selector],
        )
    comparisons = correct = 0
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped[sample_id].append(index)
    for indices in grouped.values():
        helpful = [index for index in indices if band[index] > 0]
        harmful = [index for index in indices if band[index] < 0]
        for left in helpful:
            for right in harmful:
                comparisons += 1
                correct += int(predicted[left] > predicted[right])
    result["within_question_extreme_ranking"] = {
        "comparisons": comparisons,
        "accuracy": correct / comparisons if comparisons else float("nan"),
    }
    group_auc = [result[name]["auroc"] for name in ("no_rag_correct", "no_rag_wrong")]
    result["worst_group_auroc"] = float(min(value for value in group_auc if not math.isnan(value)))
    return result


def optimizer_for(model: ExtremeUtilityScorer, args: argparse.Namespace) -> torch.optim.Optimizer:
    stage_defaults = {
        "extreme": (2e-4, 1e-5),
        "neutral": (5e-5, 3e-6),
    }
    default_head, default_text = stage_defaults[args.stage]
    head_lr = args.head_learning_rate if args.head_learning_rate is not None else default_head
    text_lr = args.text_encoder_learning_rate if args.text_encoder_learning_rate is not None else default_text
    text_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (text_parameters if name.startswith("text_encoder.") else head_parameters).append(parameter)
    groups = [{"params": head_parameters, "lr": head_lr, "weight_decay": args.weight_decay}]
    if text_parameters:
        groups.append({"params": text_parameters, "lr": text_lr, "weight_decay": args.weight_decay})
    return torch.optim.AdamW(groups)


def save_checkpoint(
    path: Path,
    model: ExtremeUtilityScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_trainable(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer_state": state,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        path / "training_state.pt",
    )
    atomic_json(path / "trainer_state.json", state)


def load_training_state(
    path: Path,
    model: ExtremeUtilityScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    weights = torch.load(path / "trainable_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable.intersection(missing) or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={sorted(trainable.intersection(missing))} unexpected={unexpected}")
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    return dict(payload["trainer_state"])


def copy_best_model(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.num_train_epochs < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Epoch and accumulation settings must be positive")
    if args.documents_per_train_batch < 8 or args.documents_per_eval_batch < 8:
        raise ValueError("Batch capacity must fit one complete top-8 question group")
    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("Training requires CUDA; launch with CUDA_VISIBLE_DEVICES=1")
    set_seed(args.seed)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    manifest = load_contract(args)
    args.expected_threshold = float(manifest["threshold"])
    datasets = load_splits(args)
    datasets["train"] = limit_questions(datasets["train"], args.max_train_questions)
    datasets["validation"] = limit_questions(datasets["validation"], args.max_eval_questions)
    datasets["test"] = limit_questions(datasets["test"], args.max_eval_questions)
    args.output_dir = args.output_dir.resolve()
    if args.resume_from_checkpoint is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), use_fast=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    store = TraceFeatureStore(manifest, args)
    collator = Collator(tokenizer, store, args)
    loaders = {
        "train": make_loader(datasets["train"], collator, args.documents_per_train_batch, args.seed, True),
        "validation": make_loader(
            datasets["validation"], collator, args.documents_per_eval_batch, args.seed, False
        ),
        "test": make_loader(datasets["test"], collator, args.documents_per_eval_batch, args.seed, False),
    }
    data_summary = {
        name: {
            "rows": len(dataset),
            "questions": len(set(str(value) for value in dataset["sample_id"])),
            "bands": dict(Counter(str(value) for value in dataset["curriculum_band"])),
            "no_rag_correct": int(sum(bool(value) for value in dataset["no_rag_correct"])),
            "batches": len(loaders[name]),
        }
        for name, dataset in datasets.items()
    }
    config = ExtremeUtilityConfig(
        base_model_name_or_path=str(args.model_name_or_path.resolve()),
        hidden_size=store.hidden_size,
        latent_size=args.latent_size,
        dropout=args.dropout,
        trainable_text_encoder_layers=args.trainable_text_encoder_layers,
        input_mode=args.input_mode,
        source_layer=int(manifest["layer"]),
        source_anchor=str(manifest["anchor"]),
        label_threshold=float(manifest["threshold"]),
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
        "stage": args.stage,
        "prepared_manifest": manifest,
        "model_input": ["official Question+Options+Evidence text", "delta_h=hD-h0", "log1p(norm(delta_h))"],
        "forbidden_model_input": ["gold answer", "gold-derived c", "teacher score", "answer transition", "no-RAG correctness"],
        "supervision": {
            "stage1": "natural pointwise extreme BCE + C/W-balanced same-question Helpful>Harmful ranking",
            "stage2": "stage1 replay + low-weight continuous neutral Huber calibration",
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data_summary": data_summary,
    }
    atomic_json(args.output_dir / "reproduction_manifest.json", reproduction)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Data summary: %s", json.dumps(data_summary, ensure_ascii=False))
    if args.dry_run:
        batch = next(iter(loaders["train"]))
        atomic_json(
            args.output_dir / "dry_run_report.json",
            {
                "input_ids": list(batch["input_ids"].shape),
                "h0": list(batch["h0"].shape),
                "delta_h": list(batch["delta_h"].shape),
                "teacher_range": [float(batch["teacher_score"].min()), float(batch["teacher_score"].max())],
                "band_counts": dict(Counter(batch["band"].tolist())),
                "leakage_check": "collator opened only document trace text and cached h0/hD",
            },
        )
        logging.info("Dry run complete")
        return

    device = torch.device("cuda:0")
    if args.stage == "neutral":
        source_config = json.loads((args.stage1_model / "extreme_utility_config.json").read_text(encoding="utf-8"))
        if source_config["input_mode"] != args.input_mode:
            raise RuntimeError("Stage-1 and stage-2 input modes differ")
        model = ExtremeUtilityScorer.from_trainable(args.stage1_model)
    else:
        model = ExtremeUtilityScorer(config)
    if args.gradient_checkpointing and args.trainable_text_encoder_layers:
        model.text_encoder.gradient_checkpointing_enable()
        model.text_encoder.enable_input_require_grads()
    model.to(device)
    optimizer = optimizer_for(model, args)
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(round(total_updates * args.warmup_ratio)), total_updates
    )
    state: dict[str, Any] = {
        "completed_epoch": 0,
        "global_step": 0,
        "best_metric": -float("inf"),
        "best_epoch": None,
        "epochs_without_improvement": 0,
        "history": [],
    }
    if args.resume_from_checkpoint is not None:
        state = load_training_state(args.resume_from_checkpoint, model, optimizer, scheduler)
    baseline_rows = len(datasets["validation"]) if args.stage == "neutral" else 0
    total_progress = (
        baseline_rows
        + (len(datasets["train"]) + len(datasets["validation"])) * args.num_train_epochs
    )
    progress = PipelineProgress(
        overall_total=total_progress,
        overall_initial=(len(datasets["train"]) + len(datasets["validation"])) * int(state["completed_epoch"]),
        desc=f"ExtremeUtility:{args.dataset}:{args.stage}",
        enabled=args.show_progress,
    )
    baseline_worst = None
    if args.stage == "neutral":
        baseline = evaluate(model, loaders["validation"], device, args, progress, "stage2 baseline validation")
        baseline_worst = float(baseline["worst_group_auroc"])
        atomic_json(args.output_dir / "stage1_baseline_validation.json", baseline)
    try:
        for epoch in range(int(state["completed_epoch"]) + 1, args.num_train_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress.set_stage(f"epoch {epoch}/{args.num_train_epochs} train", total=len(datasets["train"]))
            sums: Counter[str] = Counter()
            seen = 0
            for batch_index, batch in enumerate(loaders["train"], 1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    score = model(**move_inputs(batch, device, args.input_mode))["utility_score"]
                    loss, details = extreme_curriculum_loss(
                        score,
                        batch["teacher_score"].to(device),
                        batch["band"].to(device),
                        batch["no_rag_state"].to(device),
                        batch["document_to_question"].to(device),
                        stage=args.stage,
                        threshold=args.expected_threshold,
                        neutral_loss_weight=args.neutral_loss_weight,
                        pairwise_loss_weight=args.pairwise_loss_weight,
                        pairwise_temperature=args.pairwise_temperature,
                    )
                    scaled = loss / args.gradient_accumulation_steps
                scaled.backward()
                if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(loaders["train"]):
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["global_step"] += 1
                batch_rows = len(batch["pair_ids"])
                seen += batch_rows
                for key, value in details.items():
                    sums[key] += float(value) * batch_rows
                progress.update(batch_rows)
                progress.set_detail(f"loss={details['loss']:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
            train_metrics = {key: value / max(1, seen) for key, value in sums.items()}
            validation = evaluate(
                model, loaders["validation"], device, args, progress, f"epoch {epoch}/{args.num_train_epochs} validation"
            )
            allowed = True
            if args.stage == "neutral" and baseline_worst is not None:
                allowed = validation["worst_group_auroc"] >= baseline_worst - args.stage2_max_extreme_auroc_drop
                metric = validation["continuous"]["spearman"] if allowed else -float("inf")
            else:
                metric = validation["worst_group_auroc"]
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
                "selection_metric": metric,
                "extreme_guard_passed": allowed,
            }
            state["history"].append(epoch_record)
            state["completed_epoch"] = epoch
            improved = metric > float(state["best_metric"])
            if improved:
                state["best_metric"] = metric
                state["best_epoch"] = epoch
                state["epochs_without_improvement"] = 0
                save_checkpoint(args.output_dir / "best_model", model, optimizer, scheduler, state)
                atomic_json(args.output_dir / "best_model/validation_metrics.json", validation)
            else:
                state["epochs_without_improvement"] += 1
            save_checkpoint(args.output_dir / "last_checkpoint", model, optimizer, scheduler, state)
            atomic_json(args.output_dir / "training_history.json", state)
            logging.info(
                "Epoch %d complete: metric=%.6f best=%.6f guard=%s patience=%d/%d",
                epoch,
                metric,
                state["best_metric"],
                allowed,
                state["epochs_without_improvement"],
                args.early_stopping_patience,
            )
            if state["epochs_without_improvement"] >= args.early_stopping_patience:
                logging.info("Early stopping at epoch %d", epoch)
                break
    finally:
        progress.close()
    if state["best_epoch"] is None:
        raise RuntimeError("No checkpoint satisfied the selection rule")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    best_model = ExtremeUtilityScorer.from_trainable(args.output_dir / "best_model").to(device)
    final_progress = PipelineProgress(
        overall_total=len(datasets["validation"]) + len(datasets["test"]),
        desc=f"ExtremeUtilityFinal:{args.dataset}:{args.stage}",
        enabled=args.show_progress,
    )
    best_validation = evaluate(
        best_model, loaders["validation"], device, args, final_progress, "best checkpoint validation"
    )
    test = evaluate(best_model, loaders["test"], device, args, final_progress, "best checkpoint test")
    final_progress.close()
    copy_best_model(args.output_dir / "best_model", args.output_dir / "final_model")
    atomic_json(args.output_dir / "final_model/validation_metrics.json", best_validation)
    atomic_json(args.output_dir / "final_model/test_metrics.json", test)
    atomic_json(
        args.output_dir / "final_report.json",
        {
            "best_epoch": state["best_epoch"],
            "best_selection_metric": state["best_metric"],
            "validation": best_validation,
            "test": test,
        },
    )
    logging.info("Training complete: best_epoch=%s final_model=%s", state["best_epoch"], args.output_dir / "final_model")


if __name__ == "__main__":
    main()

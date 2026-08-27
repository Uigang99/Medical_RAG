#!/usr/bin/env python3
"""Train Question+Options+Document -> one continuous utility score.

The only supervision consumed by the loss is ``utility_target``.  Gold
answers, teacher margins/logits, transitions, No-RAG correctness, and hidden
states are forbidden model inputs.  Audit metadata is used only after forward
passes to report whether performance differs across No-RAG-correct/wrong
questions.
"""

from __future__ import annotations

import argparse
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
from safetensors.torch import load_file
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_margin_regressor import (  # noqa: E402
    MODEL_VERSION,
    MarginRegressorConfig,
    TextMarginRegressor,
)
from medrag.filtering.rag2_official import build_official_filter_input, format_options  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402


TRAINER_VERSION = "rag2_text_margin_regression_trainer_v4_extreme_neutral_curriculum"
PREPARED_VERSION = "rag2_margin_regression_pointer_splits_v1"
ACTION_NAMES = ("helpful", "neutral", "harmful")


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
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--prepared-root", type=Path, default=base / "gold_margin_regression_v1/prepared")
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Flan-T5-large")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "models/RAG2-MarginRegressor-FlanT5-large")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Use this exact run directory instead of output-root/dataset/run-name/timestamp.",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=("all", "extreme", "calibration"),
        default="all",
        help=(
            "all keeps the original full-distribution training; extreme trains only |u| above "
            "extreme-threshold; calibration initializes from stage1-model and trains a 50/50 "
            "extreme-replay/stratified-neutral set."
        ),
    )
    parser.add_argument("--stage1-model", type=Path, default=None)
    parser.add_argument("--extreme-threshold", type=float, default=0.2)
    parser.add_argument("--calibration-extreme-fraction", type=float, default=0.5)
    parser.add_argument("--neutral-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--neutral-tolerance",
        type=float,
        default=0.03,
        help="Neutral calibration incurs no pointwise penalty inside |prediction-target|<=epsilon.",
    )
    parser.add_argument(
        "--stage2-max-extreme-auroc-drop",
        type=float,
        default=0.02,
        help="Reject a Stage-2 checkpoint if extreme AUROC falls farther than this from Stage 1.",
    )
    parser.add_argument(
        "--stage2-max-extreme-sign-accuracy-drop",
        type=float,
        default=0.02,
        help="Also reject Stage-2 checkpoints that forget the extreme Helpful/Harmful sign.",
    )
    parser.add_argument("--num-train-epochs", type=int, required=True)
    parser.add_argument("--train-documents-per-batch", type=int, default=64)
    parser.add_argument("--eval-documents-per-batch", type=int, default=128)
    parser.add_argument(
        "--train-questions-per-batch",
        type=int,
        default=16,
        help="Used when pairwise ranking is enabled so all documents for a question share a batch.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument(
        "--utility-threshold",
        type=float,
        default=0.1,
        help="Deploy-time threshold: u>=tau Helpful, u<=-tau Harmful, otherwise Neutral.",
    )
    parser.add_argument(
        "--extreme-sample-weight",
        type=float,
        default=2.0,
        help="Huber-loss weight for |u|>=utility-threshold; Neutral remains weight 1.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("action_macro_f1", "spearman", "within_question_spearman", "extreme_auroc"),
        default="action_macro_f1",
    )
    parser.add_argument("--pairwise-ranking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.05)
    parser.add_argument("--pairwise-min-target-gap", type=float, default=0.1)
    parser.add_argument("--pairwise-temperature", type=float, default=0.1)
    parser.add_argument("--head-hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trainable-encoder-layers", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--trace-shard-cache-size", type=int, default=8)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-eval-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(decode_json(line))
            except Exception as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error
    return rows


def load_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = args.prepared_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("materialization_version") != PREPARED_VERSION:
        raise RuntimeError(f"Prepared-data contract mismatch: {path}")
    if args.dataset not in manifest.get("datasets", []):
        raise RuntimeError(f"Dataset {args.dataset} is absent from the prepared manifest")
    if manifest.get("target_field") != "boundary_probability_delta":
        raise RuntimeError("Expected the single T=1 boundary-probability-delta target")
    included = set((manifest.get("model_input_contract") or {}).get("included") or [])
    supervision = set((manifest.get("model_input_contract") or {}).get("supervision") or [])
    if included != {"question text", "answer options when present", "one document text"}:
        raise RuntimeError("Prepared input contract is not text-only")
    if supervision != {"one continuous utility_target"}:
        raise RuntimeError("Prepared supervision contract is not single-target regression")
    return manifest


def load_splits(args: argparse.Namespace) -> dict[str, Dataset]:
    paths = {
        "train": args.prepared_root / args.dataset / "train.jsonl",
        "validation": args.prepared_root / args.dataset / "val.jsonl",
        "test": args.prepared_root / args.dataset / "test.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = PROJECT_ROOT / "cache/hf_margin_regression"
    cache.mkdir(parents=True, exist_ok=True)
    logging.info("Pipeline stage data/3: loading pointer splits (the HF bar reports this stage progress/ETA)")
    loaded = load_dataset("json", data_files={key: str(path) for key, path in paths.items()}, cache_dir=str(cache))
    return {key: loaded[key] for key in paths}


def limit_questions(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None:
        return dataset
    question_ids = list(dict.fromkeys(str(value) for value in dataset["sample_id"]))
    if int(limit) >= len(question_ids):
        return dataset
    selected_questions = set(random.Random(int(seed)).sample(question_ids, int(limit)))
    return dataset.select(
        [index for index, sample_id in enumerate(dataset["sample_id"]) if str(sample_id) in selected_questions]
    )


def _stratified_neutral_indices(
    values: np.ndarray,
    *,
    threshold: float,
    requested: int,
    seed: int,
) -> list[int]:
    """Sample the four signed neutral sub-bands as evenly as availability permits."""

    half = float(threshold) / 2.0
    bands = [
        np.flatnonzero((values > -threshold) & (values < -half)).tolist(),
        np.flatnonzero((values >= -half) & (values < 0.0)).tolist(),
        np.flatnonzero((values >= 0.0) & (values < half)).tolist(),
        np.flatnonzero((values >= half) & (values < threshold)).tolist(),
    ]
    generator = random.Random(int(seed))
    for band in bands:
        generator.shuffle(band)
    target = min(int(requested), sum(len(band) for band in bands))
    quota, remainder = divmod(target, len(bands))
    selected: list[int] = []
    leftovers: list[int] = []
    for band_index, band in enumerate(bands):
        take = min(len(band), quota + (1 if band_index < remainder else 0))
        selected.extend(band[:take])
        leftovers.extend(band[take:])
    if len(selected) < target:
        generator.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    return selected


def select_curriculum_training_data(
    dataset: Dataset,
    args: argparse.Namespace,
) -> tuple[Dataset, dict[str, Any]]:
    """Materialize only the training rows required by the requested curriculum stage."""

    values = np.asarray(dataset["utility_target"], dtype=np.float32)
    threshold = float(args.extreme_threshold)
    helpful = np.flatnonzero(values >= threshold).tolist()
    harmful = np.flatnonzero(values <= -threshold).tolist()
    neutral = np.flatnonzero(np.abs(values) < threshold).tolist()
    before = len(dataset)
    if args.curriculum_stage == "all":
        selected = list(range(before))
    elif args.curriculum_stage == "extreme":
        selected = helpful + harmful
    else:
        extreme = helpful + harmful
        if not extreme:
            raise RuntimeError("Calibration requires at least one extreme replay row")
        requested_neutral = int(
            round(len(extreme) * (1.0 - args.calibration_extreme_fraction) / args.calibration_extreme_fraction)
        )
        neutral_selected = _stratified_neutral_indices(
            values,
            threshold=threshold,
            requested=requested_neutral,
            seed=args.seed + 7001,
        )
        selected = extreme + neutral_selected
    if not selected:
        raise RuntimeError(f"No training rows remain for curriculum stage {args.curriculum_stage}")
    selected.sort()
    result = dataset.select(selected)
    chosen_values = np.asarray(result["utility_target"], dtype=np.float32)
    summary = {
        "stage": args.curriculum_stage,
        "extreme_threshold": threshold,
        "before_rows": before,
        "selected_rows": len(result),
        "helpful_extreme": int(np.sum(chosen_values >= threshold)),
        "harmful_extreme": int(np.sum(chosen_values <= -threshold)),
        "neutral": int(np.sum(np.abs(chosen_values) < threshold)),
        "selected_extreme_fraction": float(np.mean(np.abs(chosen_values) >= threshold)),
    }
    return result, summary


def curriculum_direction_weights(dataset: Dataset, args: argparse.Namespace) -> tuple[float, float]:
    """Return Helpful/Harmful weights whose mean over extreme rows is one."""

    values = np.asarray(dataset["utility_target"], dtype=np.float32)
    positive = int(np.sum(values >= args.extreme_threshold))
    negative = int(np.sum(values <= -args.extreme_threshold))
    if args.curriculum_stage == "all" or positive == 0 or negative == 0:
        return 1.0, 1.0
    total = positive + negative
    return total / (2.0 * positive), total / (2.0 * negative)


class TraceTextStore:
    """Expose only question/options/document text from anchored trace shards."""

    def __init__(self, trace_root: Path, dataset: str, source_split: str, capacity: int) -> None:
        self.trace_root = trace_root
        self.dataset = dataset
        self.source_split = source_split
        self.capacity = max(1, int(capacity))
        self.cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def _rows(self, shard: str) -> list[dict[str, Any]]:
        cached = self.cache.pop(shard, None)
        if cached is not None:
            self.cache[shard] = cached
            return cached
        path = self.trace_root / "trace_shards" / self.dataset / self.source_split / shard / "pairs.jsonl"
        rows = read_jsonl(path)
        self.cache[shard] = rows
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return rows

    def official_input(self, pointer: dict[str, Any]) -> str:
        trace = self._rows(str(pointer["trace_shard"]))[int(pointer["trace_pair_row"])]
        if str(trace.get("pair_id")) != str(pointer["pair_id"]):
            raise RuntimeError(f"Trace pointer mismatch: {pointer['pair_id']}")
        evidence = str(trace.get("document_text_used") or "").strip()
        if not evidence:
            raise ValueError(f"Empty document text: {pointer['pair_id']}")
        return build_official_filter_input(
            question=str(trace["question"]),
            options=format_options(trace.get("options") or {}),
            evidence=evidence,
        )


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle within shard-local batches to avoid random multi-GB trace I/O."""

    def __init__(self, dataset: Dataset, batch_size: int, seed: int, shuffle: bool) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, shard in enumerate(dataset["trace_shard"]):
            grouped[str(shard)].append(index)
        self.grouped = dict(grouped)
        self.length = math.ceil(len(dataset) / self.batch_size)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.grouped)
        if self.shuffle:
            generator.shuffle(shards)
        remainders: list[int] = []
        for shard in shards:
            indices = list(self.grouped[shard])
            if self.shuffle:
                generator.shuffle(indices)
            full_end = (len(indices) // self.batch_size) * self.batch_size
            for start in range(0, full_end, self.batch_size):
                yield indices[start : start + self.batch_size]
            remainders.extend(indices[full_end:])
            while len(remainders) >= self.batch_size:
                yield remainders[: self.batch_size]
                del remainders[: self.batch_size]
        if remainders:
            yield remainders


class ShardQuestionBatchSampler(Sampler[list[int]]):
    """Keep every document for a question together while retaining shard locality."""

    def __init__(self, dataset: Dataset, questions_per_batch: int, seed: int, shuffle: bool) -> None:
        self.questions_per_batch = int(questions_per_batch)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        question_rows: dict[str, list[int]] = defaultdict(list)
        question_shards: dict[str, str] = {}
        for index, (sample_id, shard) in enumerate(zip(dataset["sample_id"], dataset["trace_shard"])):
            sample = str(sample_id)
            shard_name = str(shard)
            previous = question_shards.setdefault(sample, shard_name)
            if previous != shard_name:
                raise RuntimeError(f"Question spans multiple trace shards: {sample}")
            question_rows[sample].append(index)
        grouped: dict[str, list[str]] = defaultdict(list)
        for sample, shard in question_shards.items():
            grouped[shard].append(sample)
        self.question_rows = question_rows
        self.grouped = dict(grouped)
        self.question_count = len(question_rows)
        self.length = math.ceil(self.question_count / self.questions_per_batch)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.grouped)
        if self.shuffle:
            generator.shuffle(shards)
        remainders: list[str] = []
        for shard in shards:
            questions = list(self.grouped[shard])
            if self.shuffle:
                generator.shuffle(questions)
            full_end = (len(questions) // self.questions_per_batch) * self.questions_per_batch
            for start in range(0, full_end, self.questions_per_batch):
                yield [
                    index
                    for sample in questions[start : start + self.questions_per_batch]
                    for index in self.question_rows[sample]
                ]
            remainders.extend(questions[full_end:])
            while len(remainders) >= self.questions_per_batch:
                selected = remainders[: self.questions_per_batch]
                del remainders[: self.questions_per_batch]
                yield [index for sample in selected for index in self.question_rows[sample]]
        if remainders:
            yield [index for sample in remainders for index in self.question_rows[sample]]


class MarginCollator:
    def __init__(self, tokenizer: Any, store: TraceTextStore, args: argparse.Namespace) -> None:
        self.tokenizer = tokenizer
        self.store = store
        self.args = args

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        inputs = [self.store.official_input(row) for row in rows]
        sample_ids = [str(row["sample_id"]) for row in rows]
        sample_to_index = {
            sample: index for index, sample in enumerate(dict.fromkeys(sample_ids))
        }
        tokens = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.args.max_input_tokens,
            pad_to_multiple_of=8 if self.args.bf16 else None,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "utility_target": torch.tensor([float(row["utility_target"]) for row in rows], dtype=torch.float32),
            "no_rag_correct": torch.tensor(
                [bool(row["no_rag_correct_audit_only"]) for row in rows], dtype=torch.bool
            ),
            "sample_ids": sample_ids,
            "pair_ids": [str(row["pair_id"]) for row in rows],
            "question_index": torch.tensor(
                [sample_to_index[sample] for sample in sample_ids],
                dtype=torch.long,
            ),
        }


def make_loader(
    dataset: Dataset,
    collator: MarginCollator,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=ShardBatchSampler(dataset, batch_size, seed, shuffle),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def make_question_loader(
    dataset: Dataset,
    collator: MarginCollator,
    questions_per_batch: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=ShardQuestionBatchSampler(dataset, questions_per_batch, seed, shuffle),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def within_question_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    question_index: torch.Tensor,
    *,
    min_target_gap: float,
    temperature: float,
) -> tuple[torch.Tensor, int, int]:
    """Logistic ranking loss over confidently ordered documents of each question."""

    question_losses: list[torch.Tensor] = []
    comparisons = 0
    for group in torch.unique(question_index):
        positions = question_index == group
        group_prediction = prediction[positions]
        group_target = target[positions]
        target_difference = group_target[:, None] - group_target[None, :]
        preferred = target_difference >= float(min_target_gap)
        count = int(preferred.sum().item())
        if count == 0:
            continue
        prediction_difference = group_prediction[:, None] - group_prediction[None, :]
        question_losses.append(
            F.softplus(-prediction_difference[preferred] / float(temperature)).mean()
        )
        comparisons += count
    if not question_losses:
        return prediction.sum() * 0.0, 0, 0
    return torch.stack(question_losses).mean(), len(question_losses), comparisons


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {"n": 0.0, "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(target, prediction)[0, 1]) if np.std(target) and np.std(prediction) else float("nan")
    spearman = float(spearmanr(target, prediction).statistic) if target.size > 1 else float("nan")
    result = {
        "n": float(target.size),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
        "pearson": pearson,
        "spearman": spearman,
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "target_positive_rate": float((target > 0).mean()),
        "prediction_positive_rate": float((prediction > 0).mean()),
        "sign_accuracy": float(np.mean(np.signbit(target) == np.signbit(prediction))),
    }
    binary = (target > 0).astype(np.int64)
    if np.unique(binary).size == 2:
        result["positive_utility_auroc"] = float(roc_auc_score(binary, prediction))
    return result


def utility_action_labels(values: np.ndarray, threshold: float) -> np.ndarray:
    """Map a continuous utility score to the deployed three-way action."""

    labels = np.full(values.shape, 1, dtype=np.int64)  # Neutral
    labels[values >= float(threshold)] = 0  # Helpful/pass
    labels[values <= -float(threshold)] = 2  # Harmful/block
    return labels


def action_metrics(target: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, Any]:
    true_labels = utility_action_labels(target, threshold)
    predicted_labels = utility_action_labels(prediction, threshold)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(np.mean(true_labels == predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2]).tolist(),
        "true_rates": {
            name: float(np.mean(true_labels == index)) for index, name in enumerate(ACTION_NAMES)
        },
        "predicted_rates": {
            name: float(np.mean(predicted_labels == index)) for index, name in enumerate(ACTION_NAMES)
        },
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(ACTION_NAMES)
        },
    }
    for name, binary_target, score in (
        ("helpful", true_labels == 0, prediction),
        ("harmful", true_labels == 2, -prediction),
    ):
        if np.unique(binary_target).size == 2:
            result[f"{name}_auroc"] = float(roc_auc_score(binary_target.astype(np.int64), score))
            result[f"{name}_average_precision"] = float(
                average_precision_score(binary_target.astype(np.int64), score)
            )
    return result


def extreme_metrics(target: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, Any]:
    selected = np.abs(target) >= float(threshold)
    extreme_target = target[selected]
    extreme_prediction = prediction[selected]
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(selected.sum()),
        "coverage": float(selected.mean()) if selected.size else 0.0,
    }
    if extreme_target.size == 0:
        result.update({"auroc": float("nan"), "sign_accuracy": float("nan"), "mae": float("nan")})
        return result
    binary = (extreme_target > 0).astype(np.int64)
    result["sign_accuracy"] = float(np.mean((extreme_prediction > 0) == binary))
    result["mae"] = float(np.mean(np.abs(extreme_target - extreme_prediction)))
    result["prediction_mean"] = float(extreme_prediction.mean())
    result["predicted_helpful_rate"] = float(np.mean(extreme_prediction > 0))
    result["auroc"] = (
        float(roc_auc_score(binary, extreme_prediction))
        if np.unique(binary).size == 2
        else float("nan")
    )
    return result


@torch.no_grad()
def evaluate(
    model: TextMarginRegressor,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage: str,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    sample_ids: list[str] = []
    progress.set_stage(stage, total=len(loader.dataset))
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            output = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            )
        predictions.append(output["utility_score"].float().cpu().numpy())
        targets.append(batch["utility_target"].numpy())
        groups.append(batch["no_rag_correct"].numpy())
        sample_ids.extend(batch["sample_ids"])
        progress.update(len(batch["pair_ids"]))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    no_rag_correct = np.concatenate(groups).astype(bool)
    result: dict[str, Any] = {
        "overall": regression_metrics(target, prediction),
        "no_rag_correct": regression_metrics(target[no_rag_correct], prediction[no_rag_correct]),
        "no_rag_wrong": regression_metrics(target[~no_rag_correct], prediction[~no_rag_correct]),
        "action": action_metrics(target, prediction, args.utility_threshold),
        "action_no_rag_correct": action_metrics(
            target[no_rag_correct], prediction[no_rag_correct], args.utility_threshold
        ),
        "action_no_rag_wrong": action_metrics(
            target[~no_rag_correct], prediction[~no_rag_correct], args.utility_threshold
        ),
        "extreme": extreme_metrics(target, prediction, args.extreme_threshold),
        "extreme_no_rag_correct": extreme_metrics(
            target[no_rag_correct], prediction[no_rag_correct], args.extreme_threshold
        ),
        "extreme_no_rag_wrong": extreme_metrics(
            target[~no_rag_correct], prediction[~no_rag_correct], args.extreme_threshold
        ),
    }
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped_indices[sample_id].append(index)
    correlations: list[float] = []
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        values = np.asarray(indices, dtype=np.int64)
        if np.std(target[values]) == 0 or np.std(prediction[values]) == 0:
            continue
        correlation = float(spearmanr(target[values], prediction[values]).statistic)
        if math.isfinite(correlation):
            correlations.append(correlation)
    result["within_question"] = {
        "questions": len(grouped_indices),
        "questions_with_defined_spearman": len(correlations),
        "mean_spearman": float(np.mean(correlations)) if correlations else float("nan"),
    }
    return result


def save_checkpoint(
    path: Path,
    model: TextMarginRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    trainer_state: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_trainable(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer_state": trainer_state,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        path / "training_state.pt",
    )
    atomic_json(path / "trainer_state.json", trainer_state)


def load_checkpoint(
    path: Path,
    model: TextMarginRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    state = load_file(str(path / "trainable_model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
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


def load_best(path: Path, model: TextMarginRegressor) -> None:
    state = load_file(str(path / "trainable_model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable.intersection(missing) or unexpected:
        raise RuntimeError("Best checkpoint is incompatible")


def shorten_progress_for_early_stop(progress: PipelineProgress, final_work: int) -> None:
    progress.overall_total = progress.overall_done + int(final_work)
    if progress._pbar is not None:  # tqdm's public total is mutable.
        progress._pbar.total = progress.overall_total
        progress._pbar.refresh()


def checkpoint_metric_value(metrics: dict[str, Any], name: str) -> float:
    if name == "action_macro_f1":
        return float(metrics["action"]["macro_f1"])
    if name == "within_question_spearman":
        return float(metrics["within_question"]["mean_spearman"])
    if name == "extreme_auroc":
        return float(metrics["extreme"]["auroc"])
    return float(metrics["overall"]["spearman"])


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA is required")
    if args.num_train_epochs < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Epochs and accumulation must be positive")
    if args.huber_delta <= 0 or args.max_input_tokens < 32:
        raise ValueError("Invalid Huber delta or token budget")
    if not 0 < args.utility_threshold < 1:
        raise ValueError("utility-threshold must be in (0,1)")
    if args.extreme_sample_weight < 1:
        raise ValueError("extreme-sample-weight must be at least 1")
    if not 0 < args.extreme_threshold < 1:
        raise ValueError("extreme-threshold must be in (0,1)")
    if not 0 < args.calibration_extreme_fraction < 1:
        raise ValueError("calibration-extreme-fraction must be in (0,1)")
    if not 0 < args.neutral_loss_weight <= 1:
        raise ValueError("neutral-loss-weight must be in (0,1]")
    if (
        args.neutral_tolerance < 0
        or args.stage2_max_extreme_auroc_drop < 0
        or args.stage2_max_extreme_sign_accuracy_drop < 0
    ):
        raise ValueError("Neutral tolerance and Stage-2 retention tolerances must be non-negative")
    if args.curriculum_stage == "calibration" and args.stage1_model is None:
        raise ValueError("--stage1-model is required for calibration")
    if args.train_questions_per_batch < 1:
        raise ValueError("train-questions-per-batch must be positive")
    if args.pairwise_ranking and (
        args.pairwise_loss_weight <= 0
        or args.pairwise_min_target_gap <= 0
        or args.pairwise_temperature <= 0
    ):
        raise ValueError("Pairwise loss weight, target gap, and temperature must be positive")
    if args.early_stopping_patience < 1:
        raise ValueError("early-stopping-patience must be positive")
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    manifest = load_contract(args)
    datasets = load_splits(args)
    datasets["train"] = limit_questions(datasets["train"], args.max_train_questions, args.seed)
    datasets["validation"] = limit_questions(
        datasets["validation"], args.max_eval_questions, args.seed + 1
    )
    datasets["test"] = limit_questions(datasets["test"], args.max_eval_questions, args.seed + 2)
    datasets["train"], curriculum_summary = select_curriculum_training_data(datasets["train"], args)
    helpful_direction_weight, harmful_direction_weight = curriculum_direction_weights(
        datasets["train"], args
    )
    trace_root = Path(str(manifest["trace_root"]))
    source_split = str(manifest["source_split"])
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), use_fast=True)
    store = TraceTextStore(trace_root, args.dataset, source_split, args.trace_shard_cache_size)
    collator = MarginCollator(tokenizer, store, args)
    train_loader = (
        make_question_loader(
            datasets["train"], collator, args.train_questions_per_batch, args.seed, True
        )
        if args.pairwise_ranking
        else make_loader(
            datasets["train"], collator, args.train_documents_per_batch, args.seed, True
        )
    )
    loaders = {
        "train": train_loader,
        "validation": make_loader(datasets["validation"], collator, args.eval_documents_per_batch, args.seed, False),
        "test": make_loader(datasets["test"], collator, args.eval_documents_per_batch, args.seed, False),
    }
    data_summary = {
        name: {
            "pairs": len(dataset),
            "questions": len(set(str(value) for value in dataset["sample_id"])),
            "target_actions_at_threshold": {
                "helpful": int(
                    sum(float(value) >= args.utility_threshold for value in dataset["utility_target"])
                ),
                "neutral": int(
                    sum(abs(float(value)) < args.utility_threshold for value in dataset["utility_target"])
                ),
                "harmful": int(
                    sum(float(value) <= -args.utility_threshold for value in dataset["utility_target"])
                ),
            },
            "batches": len(loaders[name]),
        }
        for name, dataset in datasets.items()
    }
    data_summary["curriculum"] = curriculum_summary
    data_summary["curriculum"]["helpful_direction_weight"] = helpful_direction_weight
    data_summary["curriculum"]["harmful_direction_weight"] = harmful_direction_weight
    logging.info("Pipeline stage data/3 complete: %s", json.dumps(data_summary, ensure_ascii=False))
    if args.dry_run:
        batch = next(iter(loaders["train"]))
        logging.info(
            "Dry run complete: input=%s target=[%.6f,%.6f]",
            tuple(batch["input_ids"].shape),
            float(batch["utility_target"].min()),
            float(batch["utility_target"].max()),
        )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        output_dir = checkpoint.parent if checkpoint.name == "last_checkpoint" else checkpoint
    elif args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = None
    else:
        output_dir = args.output_root / args.dataset / args.run_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = None

    if args.curriculum_stage == "calibration":
        model = TextMarginRegressor.from_trainable(args.stage1_model.resolve())
        config = model.margin_config
        if int(config.trainable_encoder_layers) != int(args.trainable_encoder_layers):
            raise RuntimeError(
                "Stage-2 trainable encoder layers must match Stage 1: "
                f"{args.trainable_encoder_layers} != {config.trainable_encoder_layers}"
            )
    else:
        config = MarginRegressorConfig(
            base_model_name_or_path=str(args.model_name_or_path.resolve()),
            hidden_size=args.head_hidden_size,
            dropout=args.dropout,
            trainable_encoder_layers=args.trainable_encoder_layers,
        )
        model = TextMarginRegressor(config)
    if args.gradient_checkpointing and args.trainable_encoder_layers:
        model.encoder.gradient_checkpointing_enable()
        model.encoder.enable_input_require_grads()
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
        )
    )
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.num_train_epochs
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    trainer_state: dict[str, Any] = {
        "completed_epoch": 0,
        "global_step": 0,
        "best_epoch": None,
        "best_metric": -float("inf"),
        "epochs_without_improvement": 0,
        "history": [],
    }
    if checkpoint:
        trainer_state = load_checkpoint(checkpoint, model, optimizer, scheduler)
        logging.info("Resumed exactly from %s after epoch %d", checkpoint, trainer_state["completed_epoch"])

    reproduction = {
        "trainer_version": TRAINER_VERSION,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "dataset": args.dataset,
        "prepared_manifest": manifest,
        "data_summary": data_summary,
        "model_input": ["question text", "answer options when present", "one document text"],
        "sole_supervision": "utility_target = sigmoid(m_D,T=1)-sigmoid(m_0,T=1)",
        "loss": (
            f"curriculum_stage={args.curriculum_stage}; Huber regression(delta={args.huber_delta}); "
            f"extreme_threshold={args.extreme_threshold}; neutral_weight={args.neutral_loss_weight}; "
            f"neutral_tolerance={args.neutral_tolerance}; Helpful/Harmful direction weights="
            f"{helpful_direction_weight:.6f}/{harmful_direction_weight:.6f}; "
            f"pairwise={args.pairwise_ranking} weight={args.pairwise_loss_weight} "
            f"minimum_gap={args.pairwise_min_target_gap} temperature={args.pairwise_temperature}"
        ),
        "forbidden_model_inputs": [
            "gold answer", "teacher logits/margins", "No-RAG correctness", "answer transition", "hidden states", "RAG2 label"
        ],
        "deployment_rule": (
            f"Helpful if predicted u>={args.utility_threshold}; Harmful if predicted "
            f"u<=-{args.utility_threshold}; otherwise Neutral"
        ),
        "checkpoint_selection": f"maximum validation {args.checkpoint_metric}",
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }
    atomic_json(output_dir / "reproduction_manifest.json", reproduction)
    final_work = len(datasets["validation"]) + len(datasets["test"])
    per_epoch_work = len(datasets["train"]) + len(datasets["validation"])
    calibration_baseline_work = (
        len(datasets["validation"])
        if args.curriculum_stage == "calibration"
        else 0
    )
    baseline_already_recorded = "stage1_baseline" in trainer_state
    progress = PipelineProgress(
        overall_total=(
            calibration_baseline_work + args.num_train_epochs * per_epoch_work + final_work
        ),
        overall_initial=(
            int(trainer_state["completed_epoch"]) * per_epoch_work
            + (calibration_baseline_work if baseline_already_recorded else 0)
        ),
        desc=f"MarginRegression:{args.dataset}",
        enabled=args.show_progress,
    )
    try:
        if args.curriculum_stage == "calibration" and not baseline_already_recorded:
            baseline = evaluate(
                model,
                loaders["validation"],
                device,
                args,
                progress,
                "2/3 Stage-1 baseline before neutral calibration",
            )
            baseline_metric = checkpoint_metric_value(baseline, args.checkpoint_metric)
            baseline_extreme_auroc = float(baseline["extreme"]["auroc"])
            baseline_extreme_sign_accuracy = float(baseline["extreme"]["sign_accuracy"])
            if not all(
                math.isfinite(value)
                for value in (baseline_metric, baseline_extreme_auroc, baseline_extreme_sign_accuracy)
            ):
                raise RuntimeError("Stage-1 model produced no finite validation baseline")
            trainer_state["stage1_baseline"] = baseline
            trainer_state["best_epoch"] = 0
            trainer_state["best_metric"] = baseline_metric
            trainer_state["epochs_without_improvement"] = 0
            model.save_trainable(output_dir / "best_model")
            atomic_json(output_dir / "best_model/validation_metrics.json", baseline)
            atomic_json(output_dir / "training_history.json", trainer_state)
            save_checkpoint(output_dir / "last_checkpoint", model, optimizer, scheduler, trainer_state)
            logging.info(
                "Stage-1 replay baseline: metric=%.6f extreme_AUROC/sign=%.6f/%.6f; "
                "Stage-2 checkpoints may drop at most %.4f/%.4f",
                baseline_metric,
                baseline_extreme_auroc,
                baseline_extreme_sign_accuracy,
                args.stage2_max_extreme_auroc_drop,
                args.stage2_max_extreme_sign_accuracy_drop,
            )
        stage1_baseline = trainer_state.get("stage1_baseline")
        baseline_extreme_auroc = (
            float(stage1_baseline["extreme"]["auroc"])
            if stage1_baseline is not None
            else None
        )
        baseline_extreme_sign_accuracy = (
            float(stage1_baseline["extreme"]["sign_accuracy"])
            if stage1_baseline is not None
            else None
        )
        for epoch in range(int(trainer_state["completed_epoch"]) + 1, args.num_train_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress.set_stage(
                f"2/3 train epoch {epoch}/{args.num_train_epochs}", total=len(datasets["train"])
            )
            rolling_loss = 0.0
            rolling_pointwise = 0.0
            rolling_pairwise = 0.0
            rolling_pair_questions = 0
            rolling_comparisons = 0
            rolling_batches = 0
            for batch_index, batch in enumerate(loaders["train"], 1):
                target = batch["utility_target"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    output = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    )
                    exact_per_sample_loss = F.huber_loss(
                        output["utility_score"].float(),
                        target.float(),
                        delta=args.huber_delta,
                        reduction="none",
                    )
                    is_extreme = target.abs() >= args.extreme_threshold
                    if args.curriculum_stage == "calibration":
                        residual_outside_tolerance = (
                            (output["utility_score"].float() - target.float()).abs()
                            - args.neutral_tolerance
                        ).clamp_min(0.0)
                        neutral_per_sample_loss = F.huber_loss(
                            residual_outside_tolerance,
                            torch.zeros_like(residual_outside_tolerance),
                            delta=args.huber_delta,
                            reduction="none",
                        )
                        per_sample_loss = torch.where(
                            is_extreme,
                            exact_per_sample_loss,
                            neutral_per_sample_loss,
                        )
                    else:
                        per_sample_loss = exact_per_sample_loss
                    sample_weight = torch.ones_like(target)
                    if args.curriculum_stage == "all":
                        sample_weight = torch.where(
                            target.abs() >= args.utility_threshold,
                            torch.full_like(target, args.extreme_sample_weight),
                            sample_weight,
                        )
                    else:
                        sample_weight = torch.where(
                            target >= args.extreme_threshold,
                            torch.full_like(target, helpful_direction_weight),
                            sample_weight,
                        )
                        sample_weight = torch.where(
                            target <= -args.extreme_threshold,
                            torch.full_like(target, harmful_direction_weight),
                            sample_weight,
                        )
                        if args.curriculum_stage == "calibration":
                            sample_weight = torch.where(
                                is_extreme,
                                sample_weight,
                                torch.full_like(target, args.neutral_loss_weight),
                            )
                    pointwise_loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1)
                    if args.pairwise_ranking:
                        pairwise_loss, pair_questions, comparisons = within_question_pairwise_loss(
                            output["utility_score"].float(),
                            target.float(),
                            batch["question_index"].to(device, non_blocking=True),
                            min_target_gap=args.pairwise_min_target_gap,
                            temperature=args.pairwise_temperature,
                        )
                    else:
                        pairwise_loss = pointwise_loss * 0.0
                        pair_questions = 0
                        comparisons = 0
                    loss = pointwise_loss + args.pairwise_loss_weight * pairwise_loss
                    scaled = loss / args.gradient_accumulation_steps
                scaled.backward()
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
                rolling_loss += float(loss.detach().cpu())
                rolling_pointwise += float(pointwise_loss.detach().cpu())
                rolling_pairwise += float(pairwise_loss.detach().cpu())
                rolling_pair_questions += int(pair_questions)
                rolling_comparisons += int(comparisons)
                rolling_batches += 1
                progress.update(len(batch["pair_ids"]))
                if batch_index % args.logging_steps == 0:
                    progress.set_detail(
                        f"batch={batch_index}/{len(loaders['train'])} loss={rolling_loss/rolling_batches:.5f} "
                        f"point/pair={rolling_pointwise/rolling_batches:.4f}/"
                        f"{rolling_pairwise/rolling_batches:.4f} "
                        f"pair_q/cmp={rolling_pair_questions}/{rolling_comparisons}"
                    )
                    rolling_loss = 0.0
                    rolling_pointwise = 0.0
                    rolling_pairwise = 0.0
                    rolling_pair_questions = 0
                    rolling_comparisons = 0
                    rolling_batches = 0

            validation = evaluate(
                model,
                loaders["validation"],
                device,
                args,
                progress,
                f"2/3 validation epoch {epoch}/{args.num_train_epochs}",
            )
            metric = checkpoint_metric_value(validation, args.checkpoint_metric)
            if not math.isfinite(metric):
                metric = -float("inf")
            extreme_guard_passed = True
            if baseline_extreme_auroc is not None:
                extreme_guard_passed = (
                    float(validation["extreme"]["auroc"])
                    >= baseline_extreme_auroc - args.stage2_max_extreme_auroc_drop
                    and float(validation["extreme"]["sign_accuracy"])
                    >= baseline_extreme_sign_accuracy - args.stage2_max_extreme_sign_accuracy_drop
                )
            improved = (
                extreme_guard_passed
                and metric > float(trainer_state["best_metric"]) + args.minimum_improvement
            )
            trainer_state["completed_epoch"] = epoch
            trainer_state["history"].append(
                {
                    "epoch": epoch,
                    "validation": validation,
                    "extreme_guard_passed": extreme_guard_passed,
                }
            )
            if improved:
                trainer_state["best_metric"] = metric
                trainer_state["best_epoch"] = epoch
                trainer_state["epochs_without_improvement"] = 0
                model.save_trainable(output_dir / "best_model")
                atomic_json(output_dir / "best_model/validation_metrics.json", validation)
            else:
                trainer_state["epochs_without_improvement"] += 1
            save_checkpoint(output_dir / "last_checkpoint", model, optimizer, scheduler, trainer_state)
            atomic_json(output_dir / "training_history.json", trainer_state)
            logging.info(
                "Epoch %d: val macro-F1=%.5f Spearman=%.5f within-Q=%.5f MAE=%.5f "
                "extreme_AUROC=%.5f guard=%s correct/wrong macro-F1=%.5f/%.5f best=%s",
                epoch,
                validation["action"]["macro_f1"],
                validation["overall"]["spearman"],
                validation["within_question"]["mean_spearman"],
                validation["overall"]["mae"],
                validation["extreme"]["auroc"],
                extreme_guard_passed,
                validation["action_no_rag_correct"]["macro_f1"],
                validation["action_no_rag_wrong"]["macro_f1"],
                trainer_state["best_epoch"],
            )
            if trainer_state["epochs_without_improvement"] >= args.early_stopping_patience:
                logging.info("Early stopping after %d unimproved epochs", args.early_stopping_patience)
                shorten_progress_for_early_stop(progress, final_work)
                break

        if trainer_state["best_epoch"] is None:
            raise RuntimeError("No finite validation Spearman checkpoint was produced")
        load_best(output_dir / "best_model", model)
        best_validation = evaluate(
            model, loaders["validation"], device, args, progress, "3/3 final best-model validation"
        )
        test = evaluate(model, loaders["test"], device, args, progress, "3/3 final held-out test")
        final_metrics = {
            "best_epoch": trainer_state["best_epoch"],
            "validation": best_validation,
            "test": test,
        }
        atomic_json(output_dir / "final_metrics.json", final_metrics)
        model.save_trainable(output_dir / "final_model")
        atomic_json(output_dir / "final_model/metrics.json", final_metrics)
        logging.info("Training complete: %s", output_dir)
        logging.info("Final held-out test: %s", json.dumps(test, ensure_ascii=False))
    finally:
        progress.close()


if __name__ == "__main__":
    main()

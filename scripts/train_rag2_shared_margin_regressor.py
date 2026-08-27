#!/usr/bin/env python3
"""Train one shared text encoder to predict no-evidence and evidence margins.

The model sees only question/options text plus either ``[NO EVIDENCE]`` or one
retrieved document.  Gold answers and teacher margins are supervision only.
Questions are the batching unit, so m0 contributes once while each question's
document and delta losses are averaged over its available documents.
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
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from safetensors.torch import load_file
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_margin_regressor import (  # noqa: E402
    SHARED_MARGIN_MODEL_VERSION,
    SharedMarginRegressorConfig,
    SharedTextMarginRegressor,
)
from medrag.filtering.rag2_official import clean_text, format_options  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402


TRAINER_VERSION = "rag2_shared_margin_regression_trainer_v1"
PREPARED_VERSION = "rag2_shared_margin_question_splits_v1"
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
    parser.add_argument("--prepared-root", type=Path, default=base / "shared_gold_margin_regression_v1/prepared")
    parser.add_argument("--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Flan-T5-large")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "models/RAG2-SharedMarginRegressor-FlanT5-large")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--train-questions-per-batch", type=int, default=4)
    parser.add_argument("--eval-questions-per-batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=0.5, help="Huber delta in scaled-margin units")
    parser.add_argument("--delta-loss-weight", type=float, default=0.5)
    parser.add_argument("--margin-scale", type=float, default=10.0)
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
    parser.add_argument("--logging-steps", type=int, default=50)
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
    expected = {"no_document_gold_margin", "with_document_gold_margin", "gold_margin_delta"}
    if set(manifest.get("supervision") or []) != expected:
        raise RuntimeError("Prepared supervision is not the shared m0/mD/delta contract")
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
    cache = PROJECT_ROOT / "cache/hf_shared_margin_regression"
    cache.mkdir(parents=True, exist_ok=True)
    logging.info("Pipeline stage data/3: loading grouped pointer splits (HF reports stage progress/ETA)")
    loaded = load_dataset("json", data_files={key: str(path) for key, path in paths.items()}, cache_dir=str(cache))
    return {key: loaded[key] for key in paths}


def limit_questions(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or int(limit) >= len(dataset):
        return dataset
    generator = random.Random(int(seed))
    indices = sorted(generator.sample(range(len(dataset)), int(limit)))
    return dataset.select(indices)


def build_margin_input(question: str, options: dict[str, Any], evidence: str) -> str:
    option_text = format_options(options)
    question_block = clean_text(question)
    if option_text:
        question_block = f"{question_block}\n{option_text}"
    return f"Evidence: {clean_text(evidence)}\n\nQuestion: {question_block}"


class TraceTextStore:
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

    def texts(self, group: dict[str, Any]) -> tuple[str, list[str]]:
        rows = self._rows(str(group["trace_shard"]))
        documents = list(group["documents"])
        if not documents:
            raise ValueError(f"Question has no documents: {group['sample_id']}")
        traces: list[dict[str, Any]] = []
        for pointer in documents:
            trace = rows[int(pointer["trace_pair_row"])]
            if str(trace.get("pair_id")) != str(pointer["pair_id"]):
                raise RuntimeError(f"Trace pointer mismatch: {pointer['pair_id']}")
            if str(trace.get("sample_id")) != str(group["sample_id"]):
                raise RuntimeError(f"Question pointer mismatch: {pointer['pair_id']}")
            traces.append(trace)
        first = traces[0]
        no_evidence = build_margin_input(
            str(first["question"]), dict(first.get("options") or {}), "[NO EVIDENCE]"
        )
        evidence_inputs = [
            build_margin_input(
                str(trace["question"]),
                dict(trace.get("options") or {}),
                str(trace.get("document_text_used") or ""),
            )
            for trace in traces
        ]
        if any(not str(trace.get("document_text_used") or "").strip() for trace in traces):
            raise ValueError(f"Empty evidence in {group['sample_id']}")
        return no_evidence, evidence_inputs


class ShardQuestionBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: Dataset, batch_size: int, seed: int, shuffle: bool) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, shard in enumerate(dataset["trace_shard"]):
            grouped[str(shard)].append(index)
        self.batches = {
            shard: [indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size)]
            for shard, indices in grouped.items()
        }
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
            batches = [list(batch) for batch in self.batches[shard]]
            if self.shuffle:
                generator.shuffle(batches)
            for batch in batches:
                if self.shuffle:
                    generator.shuffle(batch)
                yield batch


class SharedMarginCollator:
    def __init__(self, tokenizer: Any, store: TraceTextStore, args: argparse.Namespace) -> None:
        self.tokenizer = tokenizer
        self.store = store
        self.args = args

    def _tokenize(self, values: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            values,
            padding=True,
            truncation=True,
            max_length=self.args.max_input_tokens,
            pad_to_multiple_of=8 if self.args.bf16 else None,
            return_tensors="pt",
        )

    def __call__(self, groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
        no_evidence_inputs: list[str] = []
        evidence_inputs: list[str] = []
        m0_targets: list[float] = []
        md_targets: list[float] = []
        document_question_indices: list[int] = []
        pair_ids: list[str] = []
        sample_ids: list[str] = []
        for question_index, group in enumerate(groups):
            no_evidence, evidence = self.store.texts(group)
            no_evidence_inputs.append(no_evidence)
            evidence_inputs.extend(evidence)
            m0_targets.append(float(group["no_document_margin"]))
            sample_ids.append(str(group["sample_id"]))
            for document in group["documents"]:
                md_targets.append(float(document["document_margin"]))
                document_question_indices.append(question_index)
                pair_ids.append(str(document["pair_id"]))
        return {
            "no_evidence_tokens": self._tokenize(no_evidence_inputs),
            "evidence_tokens": self._tokenize(evidence_inputs),
            "m0_target": torch.tensor(m0_targets, dtype=torch.float32),
            "md_target": torch.tensor(md_targets, dtype=torch.float32),
            "document_question_index": torch.tensor(document_question_indices, dtype=torch.long),
            "sample_ids": sample_ids,
            "pair_ids": pair_ids,
            "work_units": len(no_evidence_inputs) + len(evidence_inputs),
        }


def make_loader(dataset: Dataset, collator: SharedMarginCollator, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=ShardQuestionBatchSampler(dataset, batch_size, seed, shuffle),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def work_units(dataset: Dataset) -> int:
    return len(dataset) + sum(len(documents) for documents in dataset["documents"])


def group_mean(values: torch.Tensor, group_index: torch.Tensor, groups: int) -> torch.Tensor:
    sums = torch.zeros(groups, dtype=values.dtype, device=values.device)
    counts = torch.zeros(groups, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, group_index, values)
    counts.scatter_add_(0, group_index, torch.ones_like(values))
    return (sums / counts.clamp_min(1)).mean()


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {"n": 0.0, "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(target, prediction)[0, 1]) if np.std(target) and np.std(prediction) else float("nan")
    spearman = float(spearmanr(target, prediction).statistic) if target.size > 1 and np.std(target) and np.std(prediction) else float("nan")
    return {
        "n": float(target.size),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
        "pearson": pearson,
        "spearman": spearman,
        "sign_accuracy": float(np.mean((target > 0) == (prediction > 0))),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
    }


def binary_event_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        target.astype(np.int64), prediction.astype(np.int64), labels=[1], average=None, zero_division=0
    )
    return {
        "support": float(target.sum()),
        "predicted": float(prediction.sum()),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


def action_labels(md: np.ndarray, delta: np.ndarray) -> np.ndarray:
    labels = np.full(md.shape, 1, dtype=np.int64)  # neutral
    labels[delta < 0] = 2  # harmful
    labels[(md > 0) & (delta > 0)] = 0  # helpful/pass
    return labels


def calculate_metrics(
    m0_target: np.ndarray,
    m0_prediction: np.ndarray,
    md_target: np.ndarray,
    md_prediction: np.ndarray,
    document_question_index: np.ndarray,
    sample_ids: list[str],
) -> dict[str, Any]:
    expanded_m0_target = m0_target[document_question_index]
    expanded_m0_prediction = m0_prediction[document_question_index]
    delta_target = md_target - expanded_m0_target
    delta_prediction = md_prediction - expanded_m0_prediction
    true_action = action_labels(md_target, delta_target)
    predicted_action = action_labels(md_prediction, delta_prediction)
    action_f1 = f1_score(true_action, predicted_action, labels=[0, 1, 2], average=None, zero_division=0)
    action_macro = float(f1_score(true_action, predicted_action, labels=[0, 1, 2], average="macro", zero_division=0))
    true_wc = (expanded_m0_target <= 0) & (md_target > 0)
    predicted_wc = (expanded_m0_prediction <= 0) & (md_prediction > 0)
    true_cw = (expanded_m0_target > 0) & (md_target <= 0)
    predicted_cw = (expanded_m0_prediction > 0) & (md_prediction <= 0)
    helpful_target = true_action == 0
    result: dict[str, Any] = {
        "m0": regression_metrics(m0_target, m0_prediction),
        "md": regression_metrics(md_target, md_prediction),
        "delta": regression_metrics(delta_target, delta_prediction),
        "action": {
            "macro_f1": action_macro,
            **{f"{name}_f1": float(value) for name, value in zip(ACTION_NAMES, action_f1)},
            "true_rates": {
                name: float(np.mean(true_action == index)) for index, name in enumerate(ACTION_NAMES)
            },
            "predicted_rates": {
                name: float(np.mean(predicted_action == index)) for index, name in enumerate(ACTION_NAMES)
            },
        },
        "wrong_to_correct": binary_event_metrics(true_wc, predicted_wc),
        "correct_to_wrong": binary_event_metrics(true_cw, predicted_cw),
        "no_rag_correct": {
            "documents": int((expanded_m0_target > 0).sum()),
            "md_sign_accuracy": float(np.mean((md_target[expanded_m0_target > 0] > 0) == (md_prediction[expanded_m0_target > 0] > 0))),
            "delta_sign_accuracy": float(np.mean((delta_target[expanded_m0_target > 0] > 0) == (delta_prediction[expanded_m0_target > 0] > 0))),
        },
        "no_rag_wrong": {
            "documents": int((expanded_m0_target <= 0).sum()),
            "md_sign_accuracy": float(np.mean((md_target[expanded_m0_target <= 0] > 0) == (md_prediction[expanded_m0_target <= 0] > 0))),
            "delta_sign_accuracy": float(np.mean((delta_target[expanded_m0_target <= 0] > 0) == (delta_prediction[expanded_m0_target <= 0] > 0))),
        },
    }
    if np.unique(helpful_target).size == 2:
        # A smooth pass score used only for audit; the deployed rule remains md>0 and delta>delta_threshold.
        result["action"]["helpful_auroc"] = float(roc_auc_score(helpful_target.astype(np.int64), np.minimum(md_prediction, delta_prediction)))
    correlations: list[float] = []
    start = 0
    for question_index, _sample_id in enumerate(sample_ids):
        positions = np.flatnonzero(document_question_index == question_index)
        if positions.size >= 2 and np.std(delta_target[positions]) and np.std(delta_prediction[positions]):
            value = float(spearmanr(delta_target[positions], delta_prediction[positions]).statistic)
            if math.isfinite(value):
                correlations.append(value)
        start += positions.size
    result["within_question"] = {
        "questions": len(sample_ids),
        "questions_with_defined_spearman": len(correlations),
        "mean_delta_spearman": float(np.mean(correlations)) if correlations else float("nan"),
    }
    return result


@torch.no_grad()
def evaluate(
    model: SharedTextMarginRegressor,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage: str,
) -> dict[str, Any]:
    model.eval()
    m0_predictions: list[np.ndarray] = []
    m0_targets: list[np.ndarray] = []
    md_predictions: list[np.ndarray] = []
    md_targets: list[np.ndarray] = []
    mappings: list[np.ndarray] = []
    sample_ids: list[str] = []
    question_offset = 0
    progress.set_stage(stage, total=work_units(loader.dataset))
    for batch in loader:
        no_tokens = batch["no_evidence_tokens"]
        doc_tokens = batch["evidence_tokens"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            m0 = model(
                no_tokens["input_ids"].to(device, non_blocking=True),
                no_tokens["attention_mask"].to(device, non_blocking=True),
            )["margin"]
            md = model(
                doc_tokens["input_ids"].to(device, non_blocking=True),
                doc_tokens["attention_mask"].to(device, non_blocking=True),
            )["margin"]
        m0_predictions.append(m0.float().cpu().numpy())
        m0_targets.append(batch["m0_target"].numpy())
        md_predictions.append(md.float().cpu().numpy())
        md_targets.append(batch["md_target"].numpy())
        mappings.append(batch["document_question_index"].numpy() + question_offset)
        question_offset += len(batch["sample_ids"])
        sample_ids.extend(batch["sample_ids"])
        progress.update(int(batch["work_units"]))
    return calculate_metrics(
        np.concatenate(m0_targets),
        np.concatenate(m0_predictions),
        np.concatenate(md_targets),
        np.concatenate(md_predictions),
        np.concatenate(mappings),
        sample_ids,
    )


def save_checkpoint(path: Path, model: SharedTextMarginRegressor, optimizer: Any, scheduler: Any, state: dict[str, Any]) -> None:
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


def load_checkpoint(path: Path, model: SharedTextMarginRegressor, optimizer: Any, scheduler: Any) -> dict[str, Any]:
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


def load_best(path: Path, model: SharedTextMarginRegressor) -> None:
    state = load_file(str(path / "trainable_model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable.intersection(missing) or unexpected:
        raise RuntimeError("Best checkpoint is incompatible")


def shorten_progress_for_early_stop(progress: PipelineProgress, final_work: int) -> None:
    progress.overall_total = progress.overall_done + int(final_work)
    if progress._pbar is not None:
        progress._pbar.total = progress.overall_total
        progress._pbar.refresh()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA is required")
    if min(args.num_train_epochs, args.gradient_accumulation_steps, args.train_questions_per_batch) < 1:
        raise ValueError("Epochs, accumulation, and batch sizes must be positive")
    if args.margin_scale <= 0 or args.huber_delta <= 0 or args.delta_loss_weight < 0:
        raise ValueError("Invalid margin/loss configuration")
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    manifest = load_contract(args)
    datasets = load_splits(args)
    datasets["train"] = limit_questions(datasets["train"], args.max_train_questions, args.seed)
    datasets["validation"] = limit_questions(datasets["validation"], args.max_eval_questions, args.seed + 1)
    datasets["test"] = limit_questions(datasets["test"], args.max_eval_questions, args.seed + 2)
    trace_root = Path(str(manifest["trace_root"]))
    source_split = str(manifest["source_split"])
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), use_fast=True)
    store = TraceTextStore(trace_root, args.dataset, source_split, args.trace_shard_cache_size)
    collator = SharedMarginCollator(tokenizer, store, args)
    loaders = {
        "train": make_loader(datasets["train"], collator, args.train_questions_per_batch, args.seed, True),
        "validation": make_loader(datasets["validation"], collator, args.eval_questions_per_batch, args.seed, False),
        "test": make_loader(datasets["test"], collator, args.eval_questions_per_batch, args.seed, False),
    }
    data_summary = {
        name: {
            "questions": len(dataset),
            "documents": work_units(dataset) - len(dataset),
            "encoder_inputs": work_units(dataset),
            "batches": len(loaders[name]),
        }
        for name, dataset in datasets.items()
    }
    logging.info("Pipeline stage data/3 complete: %s", json.dumps(data_summary, ensure_ascii=False))
    if args.dry_run:
        batch = next(iter(loaders["train"]))
        logging.info(
            "Dry run complete: questions=%d documents=%d no_tokens=%s doc_tokens=%s m0=[%.3f,%.3f] md=[%.3f,%.3f]",
            len(batch["sample_ids"]),
            len(batch["pair_ids"]),
            tuple(batch["no_evidence_tokens"]["input_ids"].shape),
            tuple(batch["evidence_tokens"]["input_ids"].shape),
            float(batch["m0_target"].min()),
            float(batch["m0_target"].max()),
            float(batch["md_target"].min()),
            float(batch["md_target"].max()),
        )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        output_dir = checkpoint.parent if checkpoint.name == "last_checkpoint" else checkpoint
    else:
        output_dir = args.output_root / args.dataset / args.run_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = None

    config = SharedMarginRegressorConfig(
        base_model_name_or_path=str(args.model_name_or_path.resolve()),
        hidden_size=args.head_hidden_size,
        dropout=args.dropout,
        trainable_encoder_layers=args.trainable_encoder_layers,
        margin_scale=args.margin_scale,
    )
    model = SharedTextMarginRegressor(config)
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(round(total_updates * args.warmup_ratio)),
        total_updates,
    )
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
        "model_version": SHARED_MARGIN_MODEL_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "dataset": args.dataset,
        "prepared_manifest": manifest,
        "data_summary": data_summary,
        "model_input": {
            "m0": "Question + Options + [NO EVIDENCE]",
            "mD": "Question + Options + one Document",
        },
        "model_output": "one shared scalar margin head; invoked once for m0 and once per document for mD",
        "supervision_only": ["teacher m0", "teacher mD", "teacher delta=mD-m0"],
        "forbidden_model_inputs": ["gold answer", "teacher margins", "No-RAG correctness", "answer transition", "hidden states"],
        "loss": "question-normalized Huber(m0)+Huber(mD)+delta_weight*Huber(mD-m0,delta)",
        "checkpoint_selection": "maximum validation rule-derived Helpful/Neutral/Harmful macro-F1 at gamma=delta=0",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(output_dir / "reproduction_manifest.json", reproduction)

    train_work = work_units(datasets["train"])
    validation_work = work_units(datasets["validation"])
    test_work = work_units(datasets["test"])
    per_epoch_work = train_work + validation_work
    final_work = validation_work + test_work
    progress = PipelineProgress(
        overall_total=args.num_train_epochs * per_epoch_work + final_work,
        overall_initial=int(trainer_state["completed_epoch"]) * per_epoch_work,
        desc=f"SharedMargin:{args.dataset}",
        enabled=args.show_progress,
    )
    try:
        for epoch in range(int(trainer_state["completed_epoch"]) + 1, args.num_train_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress.set_stage(f"2/3 train epoch {epoch}/{args.num_train_epochs}", total=train_work)
            rolling = {"loss": 0.0, "m0": 0.0, "md": 0.0, "delta": 0.0, "batches": 0}
            for batch_index, batch in enumerate(loaders["train"], 1):
                no_tokens = batch["no_evidence_tokens"]
                doc_tokens = batch["evidence_tokens"]
                group_index = batch["document_question_index"].to(device, non_blocking=True)
                m0_target = batch["m0_target"].to(device, non_blocking=True) / args.margin_scale
                md_target = batch["md_target"].to(device, non_blocking=True) / args.margin_scale
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    m0_prediction = model(
                        no_tokens["input_ids"].to(device, non_blocking=True),
                        no_tokens["attention_mask"].to(device, non_blocking=True),
                    )["scaled_margin"].float()
                    md_prediction = model(
                        doc_tokens["input_ids"].to(device, non_blocking=True),
                        doc_tokens["attention_mask"].to(device, non_blocking=True),
                    )["scaled_margin"].float()
                    m0_per_question = F.huber_loss(
                        m0_prediction, m0_target.float(), delta=args.huber_delta, reduction="none"
                    )
                    md_per_document = F.huber_loss(
                        md_prediction, md_target.float(), delta=args.huber_delta, reduction="none"
                    )
                    delta_prediction = md_prediction - m0_prediction[group_index]
                    delta_target = md_target - m0_target[group_index]
                    delta_per_document = F.huber_loss(
                        delta_prediction, delta_target.float(), delta=args.huber_delta, reduction="none"
                    )
                    m0_loss = m0_per_question.mean()
                    md_loss = group_mean(md_per_document, group_index, len(batch["sample_ids"]))
                    delta_loss = group_mean(delta_per_document, group_index, len(batch["sample_ids"]))
                    loss = m0_loss + md_loss + args.delta_loss_weight * delta_loss
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
                rolling["loss"] += float(loss.detach().cpu())
                rolling["m0"] += float(m0_loss.detach().cpu())
                rolling["md"] += float(md_loss.detach().cpu())
                rolling["delta"] += float(delta_loss.detach().cpu())
                rolling["batches"] += 1
                progress.update(int(batch["work_units"]))
                if batch_index % args.logging_steps == 0:
                    count = max(1, int(rolling["batches"]))
                    progress.set_detail(
                        f"batch={batch_index}/{len(loaders['train'])} loss={rolling['loss']/count:.4f} "
                        f"m0/md/dm={rolling['m0']/count:.3f}/{rolling['md']/count:.3f}/{rolling['delta']/count:.3f}"
                    )
                    rolling = {"loss": 0.0, "m0": 0.0, "md": 0.0, "delta": 0.0, "batches": 0}

            validation = evaluate(
                model,
                loaders["validation"],
                device,
                args,
                progress,
                f"2/3 validation epoch {epoch}/{args.num_train_epochs}",
            )
            metric = float(validation["action"]["macro_f1"])
            improved = math.isfinite(metric) and metric > float(trainer_state["best_metric"]) + args.minimum_improvement
            trainer_state["completed_epoch"] = epoch
            trainer_state["history"].append({"epoch": epoch, "validation": validation})
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
                "Epoch %d: action_macro_f1=%.4f m0/md_sign=%.4f/%.4f delta_spearman=%.4f W->C/C->W recall=%.4f/%.4f best=%s",
                epoch,
                validation["action"]["macro_f1"],
                validation["m0"]["sign_accuracy"],
                validation["md"]["sign_accuracy"],
                validation["delta"]["spearman"],
                validation["wrong_to_correct"]["recall"],
                validation["correct_to_wrong"]["recall"],
                trainer_state["best_epoch"],
            )
            if trainer_state["epochs_without_improvement"] >= args.early_stopping_patience:
                logging.info("Early stopping after %d unimproved epochs", args.early_stopping_patience)
                shorten_progress_for_early_stop(progress, final_work)
                break

        if trainer_state["best_epoch"] is None:
            raise RuntimeError("No finite validation action macro-F1 checkpoint was produced")
        load_best(output_dir / "best_model", model)
        best_validation = evaluate(model, loaders["validation"], device, args, progress, "3/3 final best validation")
        test = evaluate(model, loaders["test"], device, args, progress, "3/3 final held-out test")
        final_metrics = {"best_epoch": trainer_state["best_epoch"], "validation": best_validation, "test": test}
        atomic_json(output_dir / "final_metrics.json", final_metrics)
        model.save_trainable(output_dir / "final_model")
        atomic_json(output_dir / "final_model/metrics.json", final_metrics)
        logging.info("Shared-margin training complete: %s", output_dir)
        logging.info("Final held-out test: %s", json.dumps(test, ensure_ascii=False))
    finally:
        progress.close()


if __name__ == "__main__":
    main()

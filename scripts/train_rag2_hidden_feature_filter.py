#!/usr/bin/env python3
"""Train controlled RAG2 filter ablations over text and pre-answer states.

The three modes share the same Flan-T5 checkpoint, binary special-token
objective, question-level split, optimizer, sampler, and evaluation code:

``text_only``
    Released RAG2 evidence-question text only.
``hidden_only``
    Two continuous encoder prefix tokens derived from h0 and hD-h0.
``text_hidden``
    The same two prefix tokens prepended to the released RAG2 text.

The gold-derived answer direction ``c`` and its scalar projection are never
loaded by this trainer.  Supplying either would leak the formula used to make
the target and would not be available for an unseen question without gold.
"""

from __future__ import annotations

import argparse
import gc
import inspect
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

import datasets as datasets_package
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as transformers_package
from datasets import Dataset, load_dataset
from safetensors import safe_open
from torch.utils.data import Sampler
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import WEIGHTS_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import LABEL_NAMES, LABEL_TOKENS, add_label_tokens


TRAINER_VERSION = "rag2_hidden_feature_filter_ablation_v2"
INPUT_MODES = ("text_only", "hidden_only", "text_hidden")
LABEL_MODES = ("symmetric_neutral", "positive_vs_rest")
TRAIN_BALANCE_MODES = ("natural", "four_group_loss")
BALANCE_GROUPS = (
    "no_rag_correct__helpful",
    "no_rag_correct__not_helpful",
    "no_rag_wrong__helpful",
    "no_rag_wrong__not_helpful",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--hidden-feature-root", type=Path, default=None)
    parser.add_argument("--expected-label-threshold", type=float, default=0.0)
    parser.add_argument(
        "--expected-label-mode",
        choices=LABEL_MODES,
        default="symmetric_neutral",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=WORKSPACE_ROOT / "models" / "Flan-T5-large",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing-num-workers", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-accumulation-steps", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument(
        "--metric-for-best-model",
        choices=("accuracy", "balanced_accuracy", "macro_f1"),
        default="macro_f1",
    )
    parser.add_argument("--hidden-shard-cache-size", type=int, default=32)
    parser.add_argument("--prefix-dropout", type=float, default=0.0)
    parser.add_argument(
        "--train-balance-mode",
        choices=TRAIN_BALANCE_MODES,
        default="natural",
        help=(
            "four_group_loss gives equal total training-loss mass to no-RAG "
            "correct/wrong x Helpful/Not Helpful without duplicating rows."
        ),
    )
    parser.add_argument(
        "--balanced-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an equal-size four-group validation subset for epoch selection. "
            "Natural validation and test metrics are still saved after training."
        ),
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
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


def load_split_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = args.split_root / args.dataset / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("materialization_version") != "rag2_hidden_utility_filter_inputs_v1":
        raise RuntimeError(f"Unsupported hidden split manifest: {path}")
    if value.get("dataset") != args.dataset:
        raise RuntimeError("Split manifest dataset mismatch")
    actual_threshold = float(value.get("threshold"))
    if not math.isclose(
        actual_threshold,
        float(args.expected_label_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Split label threshold mismatch: expected {args.expected_label_threshold}, "
            f"found {actual_threshold}"
        )
    actual_label_mode = str(value.get("label_mode") or "symmetric_neutral")
    if actual_label_mode != args.expected_label_mode:
        raise RuntimeError(
            f"Split label mode mismatch: expected {args.expected_label_mode}, "
            f"found {actual_label_mode}"
        )
    forbidden = set((value.get("model_input_contract") or {}).get("forbidden_as_model_inputs") or [])
    if not {"gold-derived c", "projection score"}.issubset(forbidden):
        raise RuntimeError("Split manifest does not declare the gold leakage exclusions")
    return value


def resolve_hidden_root(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    value = args.hidden_feature_root or Path(str(manifest["hidden_feature_dir"]))
    value = value.resolve()
    if not (value / "run_manifest.json").is_file():
        raise FileNotFoundError(value / "run_manifest.json")
    contract = json.loads((value / "run_manifest.json").read_text(encoding="utf-8"))
    requested_layers = [str(value) for value in (contract.get("layers") or [])]
    normalized_layers = [
        value if value.startswith("layer_") else f"layer_{value}"
        for value in requested_layers
    ]
    if contract.get("dataset") != args.dataset or normalized_layers != ["layer_28"]:
        raise RuntimeError("Hidden feature root does not match the layer-28 dataset contract")
    return value


def load_raw_splits(args: argparse.Namespace) -> dict[str, Dataset]:
    root = args.split_root / args.dataset
    files = {name: root / filename for name, filename in (("train", "train.jsonl"), ("validation", "val.jsonl"), ("test", "test.jsonl"))}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_dir = PROJECT_ROOT / "cache" / "hf_hidden_filter_ablation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_dataset("json", data_files={key: str(value) for key, value in files.items()}, cache_dir=str(cache_dir))
    return {key: loaded[key] for key in files}


def get_balance_group(transition: Any, target: Any) -> str:
    transition_value = str(transition).strip().upper()
    if transition_value.startswith("C->"):
        state = "no_rag_correct"
    elif transition_value.startswith("W->"):
        state = "no_rag_wrong"
    else:
        raise ValueError(f"Unsupported answer transition for balancing: {transition!r}")
    target_value = str(target).strip().lower().replace(" ", "_")
    if target_value not in {"helpful", "not_helpful"}:
        raise ValueError(f"Unsupported binary target for balancing: {target!r}")
    return f"{state}__{target_value}"


def add_four_group_training_weights(train: Dataset) -> tuple[Dataset, dict[str, Any]]:
    counts = Counter(str(value) for value in train["balance_group"])
    missing = [name for name in BALANCE_GROUPS if counts[name] == 0]
    if missing:
        raise RuntimeError(f"Cannot balance training data; empty groups: {missing}")
    total = len(train)
    weights = {name: total / (len(BALANCE_GROUPS) * counts[name]) for name in BALANCE_GROUPS}
    weighted = train.add_column(
        "_sample_weight",
        [float(weights[str(group)]) for group in train["balance_group"]],
    )
    effective_mass = {name: float(counts[name] * weights[name]) for name in BALANCE_GROUPS}
    total_mass = sum(effective_mass.values())
    return weighted, {
        "mode": "four_group_loss",
        "group_counts": dict(counts),
        "group_weights": weights,
        "effective_group_loss_mass": effective_mass,
        "effective_group_fraction": {
            name: effective_mass[name] / total_mass for name in BALANCE_GROUPS
        },
        "row_count_unchanged": len(weighted) == total,
    }


def make_four_group_balanced_validation(validation: Dataset, seed: int) -> tuple[Dataset, dict[str, Any]]:
    indices: dict[str, list[int]] = {name: [] for name in BALANCE_GROUPS}
    for index, group in enumerate(validation["balance_group"]):
        value = str(group)
        if value not in indices:
            raise RuntimeError(f"Unexpected validation balance group: {value}")
        indices[value].append(index)
    minimum = min(len(values) for values in indices.values())
    if minimum == 0:
        raise RuntimeError("Cannot construct balanced validation with an empty group")
    selected: list[int] = []
    for offset, name in enumerate(BALANCE_GROUPS):
        generator = random.Random(int(seed) + offset * 1009)
        values = list(indices[name])
        generator.shuffle(values)
        selected.extend(values[:minimum])
    # Sorting restores feature-shard locality and keeps evaluation deterministic.
    selected.sort()
    balanced = validation.select(selected)
    return balanced, {
        "source_group_counts": {name: len(indices[name]) for name in BALANCE_GROUPS},
        "selected_per_group": minimum,
        "selected_rows": len(balanced),
        "selection": "deterministic equal-size undersampling without replacement",
        "seed": int(seed),
    }


def prepare_splits(
    raw: dict[str, Dataset],
    tokenizer: Any,
    label_tokens: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    prepared: dict[str, Dataset] = {}
    reports: dict[str, Any] = {}

    def preprocess(examples: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(examples["input"], truncation=False, padding=False)
        targets = [label_tokens[str(value)] for value in examples["target"]]
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "_input_length": [len(value) for value in encoded["input_ids"]],
            "feature_shard_index": [int(value) for value in examples["feature_shard_index"]],
            "feature_pair_row": [int(value) for value in examples["feature_pair_row"]],
            "feature_question_row": [int(value) for value in examples["feature_question_row"]],
            "target_name": [str(value) for value in examples["target"]],
            "pair_id": [str(value) for value in examples["pair_id"]],
            "balance_group": [
                get_balance_group(transition, target)
                for transition, target in zip(
                    examples["answer_transition_audit_only"], examples["target"]
                )
            ],
        }

    # Length is measured without truncation and filtered explicitly below.  A
    # temporary large tokenizer limit prevents misleading "will result in
    # indexing errors" warnings for valid 513--768-token examples.  The
    # original checkpoint limit is restored before the tokenizer is saved.
    original_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = max(1_000_000_000, int(args.max_seq_length))
    try:
        for split_name, dataset in raw.items():
            map_processes = args.preprocessing_num_workers if args.preprocessing_num_workers > 1 else None
            tokenized = dataset.map(
                preprocess,
                batched=True,
                batch_size=1024,
                num_proc=map_processes,
                remove_columns=dataset.column_names,
                desc=f"Tokenize:{args.dataset}:{split_name}",
            )
            lengths = np.asarray(tokenized["_input_length"], dtype=np.int32)
            before = len(tokenized)
            tokenized = tokenized.filter(
                lambda length: int(length) <= args.max_seq_length,
                input_columns=["_input_length"],
                num_proc=map_processes,
                desc=f"DropOverlength:{args.dataset}:{split_name}",
            )
            limit = args.max_train_samples if split_name == "train" else args.max_eval_samples
            if limit is not None:
                tokenized = tokenized.select(range(min(int(limit), len(tokenized))))
            reports[split_name] = {
                "rows_before_length_filter": before,
                "rows_after_length_filter": len(tokenized),
                "rows_dropped": before - int(np.sum(lengths <= args.max_seq_length)),
                "max_seq_length": args.max_seq_length,
                "length_percentiles": {
                    "p50": float(np.percentile(lengths, 50)),
                    "p90": float(np.percentile(lengths, 90)),
                    "p95": float(np.percentile(lengths, 95)),
                    "p99": float(np.percentile(lengths, 99)),
                    "max": int(lengths.max()),
                },
                "targets": dict(Counter(str(value) for value in tokenized["target_name"])),
                "balance_groups": dict(Counter(str(value) for value in tokenized["balance_group"])),
            }
            prepared[split_name] = tokenized
    finally:
        tokenizer.model_max_length = original_model_max_length
    if args.train_balance_mode == "four_group_loss":
        prepared["train"], reports["train"]["group_balancing"] = add_four_group_training_weights(
            prepared["train"]
        )
    if args.balanced_validation:
        prepared["validation_group_balanced"], reports["validation_group_balanced"] = (
            make_four_group_balanced_validation(prepared["validation"], args.seed)
        )
    return prepared, reports


class SafeTensorFeatureStore:
    """Small LRU of memory-mapped feature shards; no gold direction is opened."""

    def __init__(self, root: Path, cache_size: int) -> None:
        self.root = root
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
        first = root / "shards" / "shard_00000"
        with safe_open(str(first / "question_features.safetensors"), framework="pt", device="cpu") as handle:
            shape = handle.get_slice("h0").get_shape()
        self.hidden_size = int(shape[-1])

    def _handles(self, shard_index: int) -> tuple[Any, Any]:
        existing = self.cache.pop(shard_index, None)
        if existing is not None:
            self.cache[shard_index] = existing
            return existing
        root = self.root / "shards" / f"shard_{shard_index:05d}"
        question = safe_open(str(root / "question_features.safetensors"), framework="pt", device="cpu")
        pair = safe_open(str(root / "pair_features.safetensors"), framework="pt", device="cpu")
        value = (question, pair)
        self.cache[shard_index] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def get(self, shard_index: int, pair_row: int, question_row: int) -> tuple[torch.Tensor, torch.Tensor]:
        question, pair = self._handles(int(shard_index))
        # Only h0 and hD are opened. c_unit, c_norm, and utility_projection are forbidden.
        h0 = question.get_slice("h0")[int(question_row), 0, :]
        hD = pair.get_slice("hD")[int(pair_row), 0, :]
        return h0, hD - h0


class RAG2AblationCollator:
    def __init__(
        self,
        tokenizer: Any,
        input_mode: str,
        feature_store: SafeTensorFeatureStore | None,
        pad_to_multiple_of: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.store = feature_store
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        text_rows = [
            {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
            for row in rows
        ]
        batch = self.tokenizer.pad(
            text_rows,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_target = max(len(row["labels"]) for row in rows)
        labels = torch.full((len(rows), max_target), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            values = torch.tensor(row["labels"], dtype=torch.long)
            labels[index, : values.numel()] = values
        batch["labels"] = labels
        if "_sample_weight" in rows[0]:
            batch["sample_weight"] = torch.tensor(
                [float(row["_sample_weight"]) for row in rows], dtype=torch.float32
            )
        if self.input_mode != "text_only":
            if self.store is None:
                raise RuntimeError("Hidden feature mode requires a feature store")
            h0_values: list[torch.Tensor] = []
            delta_values: list[torch.Tensor] = []
            for row in rows:
                h0, delta = self.store.get(
                    int(row["feature_shard_index"]),
                    int(row["feature_pair_row"]),
                    int(row["feature_question_row"]),
                )
                h0_values.append(h0)
                delta_values.append(delta)
            batch["h0"] = torch.stack(h0_values)
            batch["delta_h"] = torch.stack(delta_values)
        return batch


class ShardGroupedSampler(Sampler[int]):
    """Shuffle shards and rows within shards while retaining mmap locality."""

    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, list[int]] = defaultdict(list)
        for index, shard in enumerate(dataset["feature_shard_index"]):
            groups[int(shard)].append(index)
        self.groups = groups
        self.length = len(dataset)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.groups)
        generator.shuffle(shards)
        for shard in shards:
            indices = list(self.groups[shard])
            generator.shuffle(indices)
            yield from indices


class LocalitySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args: Any, group_balanced_loss: bool = False, **kwargs: Any) -> None:
        self.group_balanced_loss = bool(group_balanced_loss)
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset: Any | None = None) -> Sampler[int] | None:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return ShardGroupedSampler(dataset, int(self.args.data_seed or self.args.seed))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        sample_weight = inputs.pop("sample_weight", None)
        if not self.group_balanced_loss:
            parent_parameters = inspect.signature(super().compute_loss).parameters
            if "num_items_in_batch" in parent_parameters:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        if sample_weight is None:
            sample_weight = torch.ones(inputs["labels"].shape[0], device=inputs["labels"].device)
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits.float()
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(labels.shape)
        valid = labels.ne(-100)
        per_example = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        # Dataset-level weights have mean one and give every no-RAG-state x
        # utility-label group exactly 25% expected training-loss mass.
        loss = (per_example * sample_weight.to(per_example.device, dtype=per_example.dtype)).mean()
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        """Save custom hidden-prefix models without safetensors alias errors.

        Transformers 5 removed ``TrainingArguments.save_safetensors`` and
        unconditionally uses safetensors for a plain ``nn.Module``.  The T5
        backbone has tied tensors, so hidden-prefix checkpoints are stored as
        the standard PyTorch state dict instead.  Trainer can load this file
        for best-checkpoint restoration and exact resume.
        """
        unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(unwrapped, HiddenPrefixSeq2Seq):
            super()._save(output_dir=output_dir, state_dict=state_dict)
            return
        destination = output_dir or self.args.output_dir
        os.makedirs(destination, exist_ok=True)
        weights = state_dict if state_dict is not None else unwrapped.state_dict()
        torch.save(weights, os.path.join(destination, WEIGHTS_NAME))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(destination)
        torch.save(self.args, os.path.join(destination, TRAINING_ARGS_NAME))


class HiddenPrefixSeq2Seq(nn.Module):
    """Flan-T5 with two continuous prefix tokens; c is intentionally absent."""

    main_input_name = "input_ids"

    def __init__(
        self,
        backbone: Any,
        input_mode: str,
        hidden_size: int,
        prefix_dropout: float,
    ) -> None:
        super().__init__()
        if input_mode not in {"hidden_only", "text_hidden"}:
            raise ValueError(input_mode)
        self.backbone = backbone
        self.config = backbone.config
        self.generation_config = getattr(backbone, "generation_config", None)
        self.input_mode = input_mode
        d_model = int(backbone.config.d_model)
        self.h0_norm = nn.LayerNorm(hidden_size)
        self.h0_projection = nn.Linear(hidden_size, d_model)
        self.delta_projection = nn.Linear(hidden_size, d_model, bias=False)
        self.delta_magnitude = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        self.prefix_type_embedding = nn.Parameter(torch.empty(2, d_model))
        nn.init.normal_(self.prefix_type_embedding, mean=0.0, std=0.02)
        self.prefix_dropout = nn.Dropout(float(prefix_dropout))

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        delta_h: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if h0 is None or delta_h is None:
            raise RuntimeError("Hidden prefix modes require h0 and delta_h")
        h0_float = h0.float()
        delta_float = delta_h.float()
        magnitude = torch.linalg.vector_norm(delta_float, dim=-1, keepdim=True).clamp_min(1e-6)
        delta_unit = delta_float / magnitude
        h0_token = self.h0_projection(self.h0_norm(h0_float))
        delta_token = self.delta_projection(delta_unit) + self.delta_magnitude(torch.log1p(magnitude))
        prefix = torch.stack((h0_token, delta_token), dim=1)
        prefix = self.prefix_dropout(prefix + self.prefix_type_embedding.unsqueeze(0))
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)

        if self.input_mode == "text_hidden":
            if input_ids is None or attention_mask is None:
                raise RuntimeError("Hybrid mode requires text input IDs")
            text_embeddings = self.backbone.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat((prefix.to(text_embeddings.dtype), text_embeddings), dim=1)
            complete_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        else:
            inputs_embeds = prefix
            complete_mask = prefix_mask
        kwargs.pop("num_items_in_batch", None)
        return self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=complete_mask,
            labels=labels,
            **kwargs,
        )


def build_metrics(label_ids: dict[str, int]):
    ordered_ids = [label_ids[name] for name in LABEL_NAMES]

    def preprocess_logits(logits: Any, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, 0, ordered_ids].float()

    def compute_metrics(prediction: Any) -> dict[str, float]:
        scores = np.asarray(prediction.predictions)
        labels = np.asarray(prediction.label_ids)
        predicted = scores.argmax(axis=-1)
        first = labels[:, 0]
        targets = np.full(first.shape, -1, dtype=np.int64)
        for index, token_id in enumerate(ordered_ids):
            targets[first == token_id] = index
        if np.any(targets < 0):
            raise RuntimeError(f"Unexpected target token IDs: {np.unique(first[targets < 0]).tolist()}")
        result: dict[str, float] = {"accuracy": float(np.mean(predicted == targets))}
        recalls: list[float] = []
        f1s: list[float] = []
        for index, name in enumerate(LABEL_NAMES):
            tp = int(np.sum((predicted == index) & (targets == index)))
            fp = int(np.sum((predicted == index) & (targets != index)))
            fn = int(np.sum((predicted != index) & (targets == index)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            key = name.replace(" ", "_")
            result[f"precision_{key}"] = precision
            result[f"recall_{key}"] = recall
            result[f"f1_{key}"] = f1
            result[f"support_{key}"] = float(np.sum(targets == index))
            result[f"predicted_{key}"] = float(np.sum(predicted == index))
            recalls.append(recall)
            f1s.append(f1)
            for prediction_index, prediction_name in enumerate(LABEL_NAMES):
                result[f"confusion_target_{key}_pred_{prediction_name.replace(' ', '_')}"] = float(
                    np.sum((targets == index) & (predicted == prediction_index))
                )
        result["balanced_accuracy"] = float(np.mean(recalls))
        result["macro_f1"] = float(np.mean(f1s))
        return result

    return preprocess_logits, compute_metrics


class EpochCallback(TrainerCallback):
    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.is_local_process_zero:
            logging.info("Epoch %.3f complete at global_step=%s", state.epoch or -1, state.global_step)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.gradient_accumulation_steps < 1 or args.hidden_shard_cache_size < 1:
        raise ValueError("Accumulation and shard cache sizes must be positive")
    if args.max_grad_norm < 0 or args.early_stopping_patience < 0:
        raise ValueError("Gradient norm/patience cannot be negative")
    set_seed(args.seed)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    split_manifest = load_split_manifest(args)
    hidden_root = resolve_hidden_root(args, split_manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / args.dataset / args.run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Output directory: %s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), use_fast=True)
    tokenizer.add_tokens(list(LABEL_TOKENS))
    for token in LABEL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"RAG2 label token must be atomic after registration: {token} -> {ids}")
    raw = load_raw_splits(args)
    label_token_by_name = dict(zip(LABEL_NAMES, LABEL_TOKENS))
    prepared, data_report = prepare_splits(raw, tokenizer, label_token_by_name, args)
    del raw
    gc.collect()

    if args.dry_run:
        sample = prepared["train"][0]
        store = SafeTensorFeatureStore(hidden_root, min(args.hidden_shard_cache_size, 2))
        h0, delta = store.get(
            sample["feature_shard_index"], sample["feature_pair_row"], sample["feature_question_row"]
        )
        report = {
            "trainer_version": TRAINER_VERSION,
            "dry_run": True,
            "dataset": args.dataset,
            "input_mode": args.input_mode,
            "train_balance_mode": args.train_balance_mode,
            "balanced_validation": args.balanced_validation,
            "data_report": data_report,
            "sample_hidden_shapes": {"h0": list(h0.shape), "delta_h": list(delta.shape)},
            "gold_leakage_check": "c and projection score are not loaded",
        }
        atomic_json(output_dir / "dry_run_report.json", report)
        logging.info("Dry run complete: %s", json.dumps(report, ensure_ascii=False))
        return

    backbone = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_name_or_path))
    label_ids = add_label_tokens(tokenizer, backbone)
    if args.gradient_checkpointing:
        backbone.config.use_cache = False
    feature_store: SafeTensorFeatureStore | None = None
    if args.input_mode == "text_only":
        model: nn.Module = backbone
    else:
        feature_store = SafeTensorFeatureStore(hidden_root, args.hidden_shard_cache_size)
        model = HiddenPrefixSeq2Seq(
            backbone=backbone,
            input_mode=args.input_mode,
            hidden_size=feature_store.hidden_size,
            prefix_dropout=args.prefix_dropout,
        )

    collator = RAG2AblationCollator(
        tokenizer=tokenizer,
        input_mode=args.input_mode,
        feature_store=feature_store,
        pad_to_multiple_of=8 if args.bf16 or args.tf32 else None,
    )
    preprocess_logits, compute_metrics = build_metrics(label_ids)
    epoch_validation = (
        prepared["validation_group_balanced"]
        if args.balanced_validation
        else prepared["validation"]
    )
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "do_train": True,
        "do_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "linear",
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "max_grad_norm": args.max_grad_norm,
        "optim": "adamw_torch",
        "bf16": args.bf16 and torch.cuda.is_available(),
        "bf16_full_eval": args.bf16 and torch.cuda.is_available(),
        "tf32": args.tf32 if torch.cuda.is_available() else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "predict_with_generate": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
    }
    parameters = inspect.signature(Seq2SeqTrainingArguments).parameters
    if "eval_strategy" not in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    # Transformers 5 removed save_safetensors from TrainingArguments.  Keep
    # construction version-safe instead of assuming one local API revision.
    unsupported = sorted(set(kwargs).difference(parameters))
    if unsupported:
        logging.warning("Ignoring unsupported TrainingArguments for transformers %s: %s", transformers_package.__version__, unsupported)
        for name in unsupported:
            kwargs.pop(name)
    training_args = Seq2SeqTrainingArguments(**kwargs)
    callbacks: list[TrainerCallback] = [EpochCallback()]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    trainer = LocalitySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared["train"],
        eval_dataset=epoch_validation,
        data_collator=collator,
        processing_class=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        group_balanced_loss=args.train_balance_mode == "four_group_loss",
    )

    reproduction = {
        "trainer_version": TRAINER_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": args.dataset,
        "input_mode": args.input_mode,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "split_root": str(args.split_root.resolve()),
        "hidden_feature_root": str(hidden_root),
        "label_contract": {
            "threshold": args.expected_label_threshold,
            "mode": args.expected_label_mode,
        },
        "training_balance": {
            "mode": args.train_balance_mode,
            "balanced_validation_for_checkpoint_selection": args.balanced_validation,
            "balance_group_is_model_input": False,
            "answer_transition_is_model_input": False,
        },
        "hidden_input": {
            "included": ["h0", "delta_h=hD-h0"] if args.input_mode != "text_only" else [],
            "excluded": ["c", "projection_score", "gold_answer", "answer_transition"],
            "continuous_prefix_tokens": 2 if args.input_mode != "text_only" else 0,
        },
        "data_report": data_report,
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers_package.__version__,
            "datasets": datasets_package.__version__,
        },
    }
    atomic_json(output_dir / "reproduction_manifest.json", reproduction)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    validation_metrics = trainer.evaluate(prepared["validation"], metric_key_prefix="validation_best")
    validation_balanced_metrics = None
    if args.balanced_validation:
        validation_balanced_metrics = trainer.evaluate(
            prepared["validation_group_balanced"], metric_key_prefix="validation_balanced_best"
        )
    test_metrics = trainer.evaluate(prepared["test"], metric_key_prefix="test_best")
    trainer.log_metrics("validation_best", validation_metrics)
    trainer.log_metrics("test_best", test_metrics)
    trainer.save_metrics("validation_best", validation_metrics)
    if validation_balanced_metrics is not None:
        trainer.log_metrics("validation_balanced_best", validation_balanced_metrics)
        trainer.save_metrics("validation_balanced_best", validation_balanced_metrics)
    trainer.save_metrics("test_best", test_metrics)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    atomic_json(
        final_dir / "rag2_hidden_filter_architecture.json",
        {
            "trainer_version": TRAINER_VERSION,
            "input_mode": args.input_mode,
            "label_threshold": args.expected_label_threshold,
            "label_mode": args.expected_label_mode,
            "hidden_size": feature_store.hidden_size if feature_store is not None else None,
            "hidden_inputs": ["h0", "delta_h"] if feature_store is not None else [],
            "forbidden_inputs": ["c", "projection_score", "gold_answer", "answer_transition"],
            "train_balance_mode": args.train_balance_mode,
            "balanced_validation_for_checkpoint_selection": args.balanced_validation,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        },
    )
    logging.info(
        "Training complete: mode=%s best=%s metric=%s test_macro_f1=%s",
        args.input_mode,
        trainer.state.best_model_checkpoint,
        trainer.state.best_metric,
        test_metrics.get("test_best_macro_f1"),
    )


if __name__ == "__main__":
    main()

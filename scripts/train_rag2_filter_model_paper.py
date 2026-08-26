from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import platform
import socket
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import datasets as datasets_package
import transformers as transformers_package
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

from medrag.filtering.rag2_official import (
    build_rationale_aware_filter_input,
    clean_text,
    convert_legacy_filter_input,
)


BINARY_LABEL_NAMES = ("helpful", "not helpful")
BINARY_LABEL_TOKENS = ("[HELPFUL]", "[NOT_HELPFUL]")
THREE_CLASS_LABEL_NAMES = ("helpful", "not helpful", "discard")
THREE_CLASS_LABEL_TOKENS = ("[HELPFUL]", "[NOT_HELPFUL]", "[DISCARD]")


DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "filtering"
    / "rag2"
    / "llama3_8b_paper_exact_free_response_v2"
    / "filter_training_inputs_top10_independent_ppl_v1"
)
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "Flan-T5-large"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "models"
    / "RAG2-Filter-FlanT5-large-PaperExactFreeResponse"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "hf_datasets_rag2_paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RAG2 filter using the public repository's special-token protocol."
    )
    parser.add_argument("--dataset", choices=["medmcqa", "medqa"], required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="top10_epoch5_official_protocol")
    parser.add_argument(
        "--label-mode",
        choices=("binary", "three_class"),
        default="binary",
        help=(
            "Classifier label space. The default reproduces the historical Helpful/Not Helpful "
            "filter. 'three_class' additionally trains Discard as an abstention/no-decision target."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-doc-rank", type=int, default=10)
    parser.add_argument(
        "--include-no-rag-rationale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation: append the no-RAG rationale (without its final answer sentence) to the filter input. "
            "This intentionally differs from the released RAG2 classifier input."
        ),
    )
    parser.add_argument(
        "--no-rag-root",
        type=Path,
        default=None,
        help=(
            "Root containing no_rag/<dataset>/train/no_rag_generations.jsonl. "
            "Defaults to <split-root parent>/no_rag when --include-no-rag-rationale is enabled."
        ),
    )
    parser.add_argument(
        "--no-rag-rationale-field",
        choices=["rationale_only", "rationale"],
        default="rationale_only",
        help=(
            "Nested parsed field read from no-RAG generations. rationale_only excludes the fixed final-answer "
            "sentence and is the default for this ablation."
        ),
    )

    # Public RAG2 settings.
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate micro-batches before one optimizer update.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=768,
        help=(
            "Maximum encoder tokens for one question-document pair. Inputs longer than this "
            "are excluded once, rather than converted into overlapping overflow windows."
        ),
    )
    parser.add_argument(
        "--overlength-policy",
        choices=("drop", "overflow"),
        default="drop",
        help=(
            "'drop' preserves the local one-pair/one-feature protocol. 'overflow' reproduces "
            "the released RAG2 classifier preprocessing by creating overlapping features."
        ),
    )
    parser.add_argument(
        "--doc-stride",
        type=int,
        default=128,
        help="Overlap between encoder features when --overlength-policy=overflow.",
    )
    parser.add_argument("--max-target-length", type=int, default=30)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help=(
            "Maximum global gradient norm. Set to 0 to disable clipping, matching the released RAG2 "
            "training loop (default: 0)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--preprocessing-num-workers", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=16)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--eval-accumulation-steps", type=int, default=32)
    parser.add_argument(
        "--eval-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run validation during training. By default it runs after every epoch; "
            "--validation-interval-steps changes this to a fixed optimizer-step interval. "
            "Disable to match the public RAG2 training loop."
        ),
    )
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        default=None,
        help=(
            "Run validation and save a checkpoint every N optimizer steps instead of only at epoch end. "
            "Requires --eval-each-epoch and uses the same interval for evaluation and saving so the "
            "validation-selected best checkpoint can be restored safely."
        ),
    )
    parser.add_argument(
        "--evaluate-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the final model on validation/test after training completes.",
    )
    parser.add_argument(
        "--metric-for-best-model",
        choices=["accuracy", "balanced_accuracy", "macro_f1"],
        default="macro_f1",
        help=(
            "Validation metric used to select the checkpoint restored at the end of training. "
            "Macro-F1 is the default so a majority-class checkpoint is not selected by raw accuracy."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Stop after this many consecutive validations without an improvement in "
            "--metric-for-best-model. Omit to train for all requested epochs."
        ),
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum absolute improvement in --metric-for-best-model required to reset early-stopping "
            "patience. Used only with --early-stopping-patience."
        ),
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--load-model-in-bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load model parameters in BF16 to reduce full-finetuning memory use.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute activations during backward pass to reduce VRAM use.",
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def training_label_spec(label_mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if label_mode == "binary":
        return BINARY_LABEL_NAMES, BINARY_LABEL_TOKENS
    if label_mode == "three_class":
        return THREE_CLASS_LABEL_NAMES, THREE_CLASS_LABEL_TOKENS
    raise ValueError(f"Unsupported label mode: {label_mode}")


def normalize_training_label(value: Any) -> str:
    normalized = clean_text(value).lower().replace("_", " ").strip("[]")
    if normalized == "helpful":
        return "helpful"
    if normalized in {"not helpful", "nothelpful", "unhelpful"}:
        return "not helpful"
    if normalized in {"discard", "neutral", "no clear local utility"}:
        return "discard"
    raise ValueError(f"Unsupported filter training label: {value!r}")


def add_training_label_tokens(
    tokenizer: Any,
    model: Any,
    label_names: tuple[str, ...],
    label_tokens: tuple[str, ...],
) -> dict[str, int]:
    """Add one atomic decoder token per class without changing binary defaults."""
    model.config.tie_word_embeddings = False
    tokenizer.add_tokens(list(label_tokens))
    model.resize_token_embeddings(len(tokenizer))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    token_ids: dict[str, int] = {}
    for name, token in zip(label_names, label_tokens):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or (unk_id is not None and ids[0] == unk_id):
            raise ValueError(f"Label {token!r} must resolve to one non-UNK token; got {ids}")
        token_ids[name] = int(ids[0])
    model.config.rag2_filter_label_mode = "three_class" if len(label_names) == 3 else "binary"
    model.config.rag2_filter_label_names = list(label_names)
    model.config.rag2_filter_label_tokens = list(label_tokens)
    model.config.rag2_filter_input_format = "rag2_official_evidence_question_v1"
    model.config.rag2_filter_decision_rule = "first_decoder_step_label_token_softmax"
    return token_ids


def read_filter_label_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Capture the pseudo-label contract used by the supplied split files.

    Training does not infer labels again: it must consume the train-only tau
    and question-level split produced by ``build_rag2_filter_training_splits``.
    Keeping that manifest beside the checkpoint makes the filtering result
    auditable when comparing free-response and older PPL protocols.
    """
    path = args.split_root / args.dataset / "manifest.json"
    if not path.exists():
        return {"path": str(path), "available": False}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid filter-label manifest: {path}") from error
    return {
        "path": str(path),
        "available": True,
        "training_label_mode": value.get("training_label_mode"),
        "training_target_labels": value.get("training_target_labels"),
        "label_protocol": value.get("label_protocol"),
        "filter_input": value.get("filter_input"),
        "threshold_summary": value.get("threshold_summary"),
        "summary": value.get("summary"),
    }


def _rationale_file(args: argparse.Namespace) -> Path:
    no_rag_root = args.no_rag_root or (args.split_root.parent / "no_rag")
    path = no_rag_root / args.dataset / "train" / "no_rag_generations.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Missing no-RAG rationale artifact for --include-no-rag-rationale: "
            f"{path}. Set --no-rag-root explicitly if this split comes from another artifact root."
        )
    return path


def _read_no_rag_rationales(
    path: Path,
    requested_ids: set[str],
    field: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    rationales: dict[str, str] = {}
    duplicate_ids: set[str] = set()
    invalid_rows = 0
    rows_scanned = 0
    with path.open("r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows_scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in no-RAG rationale artifact at {path}:{line_number}") from error
            sample_id = str(row.get("sample_id") or "")
            if sample_id not in requested_ids:
                continue
            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            rationale = " ".join(str(parsed.get(field) or "").split())
            if not rationale:
                invalid_rows += 1
                continue
            if sample_id in rationales:
                duplicate_ids.add(sample_id)
                continue
            rationales[sample_id] = rationale

    missing_ids = requested_ids.difference(rationales)
    if duplicate_ids:
        logging.warning(
            "No-RAG rationale artifact has %s duplicate requested sample_id(s); keeping the first occurrence.",
            len(duplicate_ids),
        )
    stats = {
        "path": str(path),
        "field": f"parsed.{field}",
        "rows_scanned": rows_scanned,
        "requested_sample_ids": len(requested_ids),
        "resolved_sample_ids": len(rationales),
        "missing_sample_ids": len(missing_ids),
        "invalid_requested_rows": invalid_rows,
        "duplicate_requested_sample_ids": len(duplicate_ids),
    }
    return rationales, stats


def _attach_no_rag_rationales(
    dataset: Dataset,
    rationales: dict[str, str],
    split_name: str,
) -> tuple[Dataset, int]:
    available_ids = set(rationales)
    missing_rows = sum(1 for sample_id in dataset["sample_id"] if str(sample_id) not in available_ids)
    if missing_rows:
        dataset = dataset.filter(
            lambda sample_id: str(sample_id) in available_ids,
            input_columns=["sample_id"],
            desc=f"Dropping {split_name} rows without no-RAG rationales",
        )

    def add_rationale(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {"_no_rag_rationale": [rationales[str(sample_id)] for sample_id in examples["sample_id"]]}

    dataset = dataset.map(
        add_rationale,
        batched=True,
        desc=f"Joining no-RAG rationales into {split_name}",
    )
    return dataset, missing_rows


def load_splits(
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, Dataset, dict[str, str], dict[str, Any]]:
    split_dir = args.split_root / args.dataset
    paths = {
        "train": split_dir / "train.jsonl",
        "validation": split_dir / "val.jsonl",
        "test": split_dir / "test.jsonl",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} split: {path}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(
        "json",
        data_files={name: str(path) for name, path in paths.items()},
        cache_dir=str(args.cache_dir),
    )

    def select_rank(dataset: Dataset) -> Dataset:
        selected = dataset
        # Document-level pseudo-label rows store the reranker position in
        # ``doc_rank``.  Direct sentence/window rows preserve the same value
        # as ``parent_document_rank`` because their own rank denotes the
        # selected evidence unit inside that document.  Apply the Top-k
        # contract to the parent document in either representation.
        rank_column = (
            "doc_rank"
            if "doc_rank" in selected.column_names
            else "parent_document_rank"
            if "parent_document_rank" in selected.column_names
            else None
        )
        if args.max_doc_rank > 0 and rank_column is not None:
            filter_kwargs: dict[str, Any] = {
                "input_columns": [rank_column],
                "desc": f"Keeping rerank top-{args.max_doc_rank}",
            }
            if args.preprocessing_num_workers > 1:
                filter_kwargs["num_proc"] = args.preprocessing_num_workers
            selected = selected.filter(
                lambda rank: int(rank) <= args.max_doc_rank,
                **filter_kwargs,
            )
        return selected

    train = select_rank(raw["train"])
    validation = select_rank(raw["validation"])
    test = select_rank(raw["test"])
    if args.max_train_samples is not None:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples is not None:
        validation = validation.select(range(min(args.max_eval_samples, len(validation))))
        test = test.select(range(min(args.max_eval_samples, len(test))))

    rationale_join: dict[str, Any] = {"enabled": bool(args.include_no_rag_rationale)}
    if args.include_no_rag_rationale:
        requested_ids = {
            str(sample_id)
            for dataset in (train, validation, test)
            for sample_id in dataset["sample_id"]
        }
        rationale_path = _rationale_file(args)
        rationales, rationale_stats = _read_no_rag_rationales(
            rationale_path,
            requested_ids,
            args.no_rag_rationale_field,
        )
        train, train_missing = _attach_no_rag_rationales(train, rationales, "train")
        validation, validation_missing = _attach_no_rag_rationales(validation, rationales, "validation")
        test, test_missing = _attach_no_rag_rationales(test, rationales, "test")
        rationale_join = {
            "enabled": True,
            **rationale_stats,
            "missing_rows_by_split": {
                "train": train_missing,
                "validation": validation_missing,
                "test": test_missing,
            },
        }
        logging.info("No-RAG rationale join: %s", rationale_join)

    return train, validation, test, {name: str(path) for name, path in paths.items()}, rationale_join


def summarize(dataset: Dataset, name: str) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "rows": len(dataset)}
    for column in (
        "dataset",
        "source",
        "parent_document_source",
        "target",
        "doc_rank",
        "parent_document_rank",
        "window_selection_role",
    ):
        if column in dataset.column_names:
            value[column] = dict(Counter(str(item) for item in dataset[column]))
    if "sample_id" in dataset.column_names:
        value["sample_ids"] = len(set(str(item) for item in dataset["sample_id"]))
    return value


def filter_overlength_inputs(
    dataset: Dataset,
    tokenizer: Any,
    args: argparse.Namespace,
    split_name: str,
) -> tuple[Dataset, dict[str, Any]]:
    """Keep exactly one untruncated feature for each pair that fits the encoder budget."""

    def add_input_lengths(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if args.include_no_rag_rationale:
            inputs = [
                build_rationale_aware_filter_input(value, rationale)
                for value, rationale in zip(examples["input"], examples["_no_rag_rationale"])
            ]
        else:
            inputs = [convert_legacy_filter_input(value) for value in examples["input"]]
        input_ids = tokenizer(inputs, add_special_tokens=True, truncation=False)["input_ids"]
        return {
            "_filter_input": inputs,
            "_filter_input_length": [len(token_ids) for token_ids in input_ids],
        }

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "batch_size": 1024,
        "desc": f"Measuring {split_name} encoder lengths",
    }
    if args.preprocessing_num_workers > 1:
        map_kwargs["num_proc"] = args.preprocessing_num_workers
    measured = dataset.map(add_input_lengths, **map_kwargs)
    lengths = measured["_filter_input_length"]
    keep_mask = [
        True if args.overlength_policy == "overflow" else length <= args.max_seq_length
        for length in lengths
    ]

    def count_by(columns: tuple[str, ...], *, kept: bool) -> dict[str, int]:
        # Document-level rows use ``source`` while the independently labeled
        # window rows retain the same provenance as
        # ``parent_document_source``.  These counters are audit metadata only;
        # an optional provenance column must never abort otherwise valid
        # training data.
        if any(column not in measured.column_names for column in columns):
            return {}
        counter: Counter[str] = Counter()
        column_values = [measured[column] for column in columns]
        for index, is_kept in enumerate(keep_mask):
            if is_kept == kept:
                key = "|".join(str(values[index]) for values in column_values)
                counter[key] += 1
        return dict(counter)

    filter_kwargs: dict[str, Any] = {
        "input_columns": ["_filter_input_length"],
        "desc": f"Dropping {split_name} inputs over {args.max_seq_length} tokens",
    }
    if args.preprocessing_num_workers > 1:
        filter_kwargs["num_proc"] = args.preprocessing_num_workers
    if args.overlength_policy == "overflow":
        filtered = measured
    else:
        filtered = measured.filter(lambda length: length <= args.max_seq_length, **filter_kwargs)
    source_column = (
        "source"
        if "source" in measured.column_names
        else "parent_document_source"
        if "parent_document_source" in measured.column_names
        else None
    )
    stats = {
        "split": split_name,
        "policy": (
            "released_rag2_overflow_windows"
            if args.overlength_policy == "overflow"
            else "drop_overlength_once_no_overflow_windows"
        ),
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride if args.overlength_policy == "overflow" else None,
        "rows_before": len(measured),
        "rows_kept": len(filtered),
        "rows_dropped": len(measured) - len(filtered),
        "dropped_percent": (len(measured) - len(filtered)) / len(measured) * 100 if measured else 0.0,
        "lengths": {
            "min": int(min(lengths)) if lengths else 0,
            "p50": float(np.percentile(lengths, 50)) if lengths else 0.0,
            "p90": float(np.percentile(lengths, 90)) if lengths else 0.0,
            "p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
            "p99": float(np.percentile(lengths, 99)) if lengths else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        },
        "dropped_by_dataset": count_by(("dataset",), kept=False),
        "source_column": source_column,
        "dropped_by_source": count_by((source_column,), kept=False) if source_column else {},
        "dropped_by_label": count_by(("target",), kept=False),
    }
    logging.info(
        "%s length policy: kept=%s/%s, dropped=%s (%.3f%%), max=%s",
        split_name,
        stats["rows_kept"],
        stats["rows_before"],
        stats["rows_dropped"],
        stats["dropped_percent"],
        stats["lengths"]["max"],
    )
    return filtered, stats


def tokenize_split(
    dataset: Dataset,
    tokenizer: Any,
    args: argparse.Namespace,
    split_name: str,
    label_tokens_by_name: dict[str, str],
) -> Dataset:
    def preprocess(examples: dict[str, list[Any]]) -> dict[str, Any]:
        targets = [label_tokens_by_name[normalize_training_label(value)] for value in examples["target"]]
        if args.overlength_policy == "overflow":
            model_inputs = tokenizer(
                examples["_filter_input"],
                max_length=args.max_seq_length,
                truncation=True,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                padding=False,
            )
            sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
            targets = [targets[int(sample_index)] for sample_index in sample_mapping]
        else:
            # Length filtering has already guaranteed one complete pair per feature.
            model_inputs = tokenizer(examples["_filter_input"], truncation=False, padding=False)
        target_tokens = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        model_inputs["labels"] = target_tokens
        return model_inputs

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": dataset.column_names,
        "desc": f"Tokenizing {split_name} with official RAG2 format",
    }
    if args.preprocessing_num_workers > 1:
        map_kwargs["num_proc"] = args.preprocessing_num_workers
    return dataset.map(preprocess, **map_kwargs)


def build_metric_functions(label_ids: dict[str, int], label_names: tuple[str, ...]):
    ordered_ids = [label_ids[name] for name in label_names]

    def preprocess_logits_for_metrics(logits: Any, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, 0, ordered_ids].float()

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        scores, labels = eval_prediction
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        predictions = scores.argmax(axis=-1)
        first_labels = labels[:, 0]
        targets = np.full(first_labels.shape, -1, dtype=np.int64)
        for index, token_id in enumerate(ordered_ids):
            targets[first_labels == token_id] = index
        if np.any(targets < 0):
            unknown = np.unique(first_labels[targets < 0]).tolist()
            raise ValueError(f"Unexpected validation label token ids: {unknown}")

        total = int(targets.shape[0])
        target_counts = [int(np.sum(targets == index)) for index in range(len(label_names))]
        prediction_counts = [int(np.sum(predictions == index)) for index in range(len(label_names))]
        metrics: dict[str, float] = {
            "accuracy": float(np.mean(predictions == targets)),
            "majority_baseline_accuracy": max(target_counts) / total if total else 0.0,
        }
        recalls: list[float] = []
        f1s: list[float] = []
        for index, name in enumerate(label_names):
            tp = int(np.sum((predictions == index) & (targets == index)))
            fp = int(np.sum((predictions == index) & (targets != index)))
            fn = int(np.sum((predictions != index) & (targets == index)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            key = name.replace(" ", "_")
            metrics[f"precision_{key}"] = precision
            metrics[f"recall_{key}"] = recall
            metrics[f"f1_{key}"] = f1
            metrics[f"support_{key}"] = float(target_counts[index])
            metrics[f"predicted_{key}"] = float(prediction_counts[index])
            metrics[f"prediction_rate_{key}"] = prediction_counts[index] / total if total else 0.0
            recalls.append(recall)
            f1s.append(f1)
            for prediction_index, prediction_name in enumerate(label_names):
                prediction_key = prediction_name.replace(" ", "_")
                metrics[f"confusion_target_{key}_pred_{prediction_key}"] = float(
                    np.sum((targets == index) & (predictions == prediction_index))
                )
        metrics["balanced_accuracy"] = float(np.mean(recalls))
        metrics["macro_f1"] = float(np.mean(f1s))
        return metrics

    return preprocess_logits_for_metrics, compute_metrics


class EpochLogCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            logging.info("Epoch %.3f complete (global_step=%s)", state.epoch or -1, state.global_step)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be non-negative")
    if args.doc_stride < 0 or args.doc_stride >= args.max_seq_length:
        raise ValueError("--doc-stride must be non-negative and smaller than --max-seq-length")
    if args.validation_interval_steps is not None:
        if not args.eval_each_epoch:
            raise ValueError("--validation-interval-steps requires --eval-each-epoch")
        if args.validation_interval_steps < 1:
            raise ValueError("--validation-interval-steps must be at least 1")
    if args.early_stopping_patience is not None:
        if not args.eval_each_epoch:
            raise ValueError("--early-stopping-patience requires --eval-each-epoch")
        if args.early_stopping_patience < 1:
            raise ValueError("--early-stopping-patience must be at least 1")
        if args.early_stopping_threshold < 0:
            raise ValueError("--early-stopping-threshold must be non-negative")
    set_seed(args.seed)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / args.dataset / args.run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Output directory: %s", output_dir)

    logging.info("Workflow stage 1/5: loading paper-style train/validation/test splits")
    train_raw, validation_raw, test_raw, split_paths, rationale_join = load_splits(args)
    label_names, label_tokens = training_label_spec(args.label_mode)
    allowed_labels = set(label_names)
    for split_name, split_dataset in (
        ("train", train_raw),
        ("validation", validation_raw),
        ("test", test_raw),
    ):
        observed = {normalize_training_label(value) for value in split_dataset["target"]}
        unexpected = observed - allowed_labels
        if unexpected:
            raise ValueError(
                f"{split_name} contains labels outside --label-mode {args.label_mode}: {sorted(unexpected)}"
            )
        if args.label_mode == "three_class" and "discard" not in observed:
            raise ValueError(
                f"{split_name} has no Discard targets; materialize the split with --training-label-mode three_class."
            )
    label_dataset_manifest = read_filter_label_manifest(args)
    summaries = [
        summarize(train_raw, "train"),
        summarize(validation_raw, "validation"),
        summarize(test_raw, "test"),
    ]
    for summary in summaries:
        logging.info("%s summary: %s", summary["name"], summary)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.load_model_in_bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-model-in-bf16 requires CUDA")
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    label_ids = add_training_label_tokens(tokenizer, model, label_names, label_tokens)
    label_tokens_by_name = dict(zip(label_names, label_tokens))
    if args.gradient_checkpointing:
        model.config.use_cache = False
    # T5 uses relative-position attention; this raises the tokenizer guard for the
    # explicitly documented local 768-token protocol without truncating evidence.
    tokenizer.model_max_length = args.max_seq_length
    logging.info("Added %s filter label tokens: %s", args.label_mode, label_ids)
    logging.info(
        "Encoder-length policy: mode=%s max=%s stride=%s",
        args.overlength_policy,
        args.max_seq_length,
        args.doc_stride if args.overlength_policy == "overflow" else "disabled",
    )

    logging.info("Workflow stage 2/5: measuring encoder lengths and applying %s policy", args.overlength_policy)
    train_filtered, train_length_stats = filter_overlength_inputs(
        train_raw,
        tokenizer,
        args,
        "train",
    )
    validation_filtered, validation_length_stats = filter_overlength_inputs(
        validation_raw,
        tokenizer,
        args,
        "validation",
    )
    test_filtered, test_length_stats = filter_overlength_inputs(
        test_raw,
        tokenizer,
        args,
        "test",
    )

    logging.info("Workflow stage 3/5: tokenizing classifier features (progress/ETA shown per split)")
    train = tokenize_split(
        train_filtered,
        tokenizer,
        args,
        "train",
        label_tokens_by_name,
    )
    need_evaluation_features = args.eval_each_epoch or args.evaluate_final_model
    validation = None
    test = None
    if need_evaluation_features:
        validation = tokenize_split(
            validation_filtered,
            tokenizer,
            args,
            "validation",
            label_tokens_by_name,
        )
        test = tokenize_split(
            test_filtered,
            tokenizer,
            args,
            "test",
            label_tokens_by_name,
        )
    logging.info(
        "Tokenized features: train=%s validation=%s test=%s",
        len(train),
        len(validation) if validation is not None else "skipped",
        len(test) if test is not None else "skipped",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.bf16 or args.tf32 else None,
    )
    preprocess_logits, compute_metrics = build_metric_functions(label_ids, label_names)
    validation_strategy = (
        "steps"
        if args.validation_interval_steps is not None
        else "epoch"
        if args.eval_each_epoch
        else "no"
    )
    save_strategy = "steps" if args.validation_interval_steps is not None else "epoch"
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "do_train": True,
        "do_eval": args.eval_each_epoch,
        "save_strategy": save_strategy,
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
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.eval_each_epoch,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": True,
    }
    training_parameters = inspect.signature(Seq2SeqTrainingArguments).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in training_parameters else "evaluation_strategy"
    training_kwargs[evaluation_key] = validation_strategy
    if args.validation_interval_steps is not None:
        training_kwargs["eval_steps"] = args.validation_interval_steps
        training_kwargs["save_steps"] = args.validation_interval_steps
    if args.eval_each_epoch:
        training_kwargs["metric_for_best_model"] = args.metric_for_best_model
        training_kwargs["greater_is_better"] = True
    if "dataloader_persistent_workers" in training_parameters:
        training_kwargs["dataloader_persistent_workers"] = False
    if "dataloader_prefetch_factor" in training_parameters:
        training_kwargs["dataloader_prefetch_factor"] = (
            args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None
        )
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    reproduction_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": " ".join(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers_package.__version__,
            "datasets": datasets_package.__version__,
        },
        "dataset": args.dataset,
        "split_paths": split_paths,
        "filter_label_dataset_manifest": label_dataset_manifest,
        "dataset_summaries": summaries,
        "no_rag_rationale_ablation": rationale_join,
        "label_mode": args.label_mode,
        "label_tokens": dict(zip(label_names, label_tokens)),
        "label_token_ids": label_ids,
        "method_status": (
            "RAG2 three-class selective extension; Discard is an abstention/no-decision target and only Helpful is eligible for downstream inclusion"
            if args.label_mode == "three_class"
            else "released RAG2 binary filter label space"
        ),
        "public_protocol": {
            "model": "Flan-T5-large",
            "input_format": "official evidence-then-question template",
            "decision": f"first decoder step, softmax over {len(label_names)} label-token logits",
            "optimizer": "AdamW",
            "learning_rate": 3e-5,
            "batch_size": 16,
            "max_seq_length": 512,
            "doc_stride": 128,
            "reported_epochs": 40,
            "public_environment": {
                "python": "3.10.13",
                "torch": "2.1.0",
                "transformers": "4.36.2",
                "datasets": "2.15.0",
                "tokenizers": "0.15.1",
            },
            "public_training_evaluation": "The released training command does not run validation.",
        },
        "local_scale_adjustment": {
            "max_doc_rank": args.max_doc_rank,
            "epochs": args.num_train_epochs,
            "reason": "Local corpus/retrieval setting; see command arguments for the selected rerank rank and epoch count.",
        },
        "local_evaluation_protocol": {
            "encoder_input": (
                "official evidence-then-question template plus no-RAG rationale"
                if args.include_no_rag_rationale
                else "official evidence-then-question template"
            ),
            "max_seq_length": args.max_seq_length,
            "overlength_policy": (
                f"released RAG2 overflow features at max_seq_length with stride={args.doc_stride}"
                if args.overlength_policy == "overflow"
                else "exclude each >max_seq_length question-document pair; do not truncate or create overflow windows"
            ),
            "validation": (
                "one complete feature per retained question-document pair every "
                f"{args.validation_interval_steps} optimizer steps"
                if args.validation_interval_steps is not None
                else "one complete feature per retained question-document pair after every epoch"
            ),
            "checkpoint_selection": (
                (
                    f"highest validation {args.metric_for_best_model}; early stopping patience "
                    f"{args.early_stopping_patience}, threshold {args.early_stopping_threshold}"
                    if args.early_stopping_patience is not None
                    else f"highest validation {args.metric_for_best_model}; no early stopping"
                )
                if args.eval_each_epoch
                else "disabled because epoch validation is disabled"
            ),
            "test": "one complete feature per retained question-document pair, evaluated once using the best validation checkpoint",
            "training_windows": (
                f"enabled with released overflow preprocessing and stride={args.doc_stride}"
                if args.overlength_policy == "overflow"
                else "disabled"
            ),
        },
        "input_length_filtering": {
            "train": train_length_stats,
            "validation": validation_length_stats,
            "test": test_length_stats,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(output_dir / "reproduction_manifest.json", reproduction_manifest)

    callbacks: list[TrainerCallback] = [EpochLogCallback()]
    if args.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
        logging.info(
            "Early stopping enabled: metric=%s patience=%s threshold=%s",
            args.metric_for_best_model,
            args.early_stopping_patience,
            args.early_stopping_threshold,
        )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train,
        "data_collator": collator,
        "callbacks": callbacks,
    }
    if args.eval_each_epoch:
        assert validation is not None
        trainer_kwargs["eval_dataset"] = validation
        trainer_kwargs["compute_metrics"] = compute_metrics
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits
    if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(
        **trainer_kwargs,
    )
    logging.info(
        "Workflow stage 4/5: training for up to %.3f epochs; Trainer progress shows total steps and ETA",
        args.num_train_epochs,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_model_dir = output_dir / "final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    metrics = dict(train_result.metrics)
    validation_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    best_checkpoint = trainer.state.best_model_checkpoint
    if args.eval_each_epoch and not best_checkpoint:
        raise RuntimeError("No validation checkpoint was selected; cannot perform best-checkpoint test evaluation.")
    if args.evaluate_final_model:
        logging.info("Workflow stage 5/5: evaluating validation and test with the validation-selected checkpoint")
        assert validation is not None and test is not None
        # ``load_best_model_at_end`` has already restored this checkpoint into
        # ``trainer.model``.  Both final evaluations therefore use the exact
        # validation-selected model, not the final optimizer step.
        validation_metrics = trainer.evaluate(eval_dataset=validation, metric_key_prefix="validation")
        test_metrics = trainer.evaluate(eval_dataset=test, metric_key_prefix="test")
        metrics.update(validation_metrics)
        metrics.update(test_metrics)
    metrics.update(
        {
            "best_model_checkpoint": best_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "train_rows": len(train_raw),
            "validation_rows": len(validation_raw),
            "test_rows": len(test_raw),
            "train_rows_retained_after_length_filter": len(train_filtered),
            "validation_rows_retained_after_length_filter": len(validation_filtered),
            "test_rows_retained_after_length_filter": len(test_filtered),
            "train_features": len(train),
            "validation_features": len(validation) if validation is not None else None,
            "test_features": len(test) if test is not None else None,
        }
    )
    write_json(output_dir / "final_metrics.json", metrics)
    write_json(
        output_dir / "best_checkpoint_test_metrics.json",
        {
            "best_model_checkpoint": best_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "final_model_dir": str(final_model_dir),
            "final_model_source": "validation-selected best checkpoint",
        },
    )
    validation_history = [
        entry
        for entry in trainer.state.log_history
        if any(key.startswith("eval_") for key in entry)
    ]
    write_json(output_dir / "validation_history.json", validation_history)
    write_json(
        output_dir / "evaluation_summary.json",
        {
            "dataset": args.dataset,
            "best_model_checkpoint": best_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "final_model_dir": str(final_model_dir),
            "validation": validation_metrics,
            "test": test_metrics,
        },
    )
    trainer.save_state()
    logging.info("Training complete. Final model: %s", final_model_dir)


if __name__ == "__main__":
    main()

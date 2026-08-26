from __future__ import annotations

import argparse
import json
import logging
import math
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
from datasets import Dataset
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from medrag.filtering.rag2_two_head import Rag2TwoHeadFilterModel
from scripts.train_rag2_filter_model_paper import (
    configure_logging,
    filter_overlength_inputs,
    load_splits,
    normalize_training_label,
    read_filter_label_manifest,
    summarize,
)


DEFAULT_BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)
DEFAULT_SPLIT_ROOT = DEFAULT_BASE / "filter_training_inputs_rag2_paper_reproduction_three_class_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Flan-T5-large"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "models/RAG2-Filter-FlanT5-large-PaperReproduction-Anchored-TwoHead"
CLASS_INDEX = {"helpful": 0, "not helpful": 1, "discard": 2}
DESIRED_HIERARCHICAL_RATIOS = {"helpful": 0.25, "not helpful": 0.25, "discard": 0.50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a hierarchical RAG2 filter with a Decisive/Discard head and a masked "
            "Helpful/Not-Helpful head."
        )
    )
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="rag2_two_head")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache/hf_datasets_rag2_paper")
    parser.add_argument("--max-doc-rank", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--overlength-policy", choices=("drop", "overflow"), default="drop")
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument(
        "--sampling-mode",
        choices=("hierarchical_balanced", "natural"),
        default="hierarchical_balanced",
        help=(
            "hierarchical_balanced samples H/NH/D at 25/25/50, yielding a 50/50 "
            "Decisive/Discard head and a 50/50 H/NH direction head."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decisive-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument(
        "--metric-for-best-model",
        choices=("macro_f1", "worst_group_macro_f1", "helpful_f1"),
        default="worst_group_macro_f1",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--discard-contamination-limit", type=float, default=0.10)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--preprocessing-num-workers", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=16)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-accumulation-steps", type=int, default=32)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.doc_stride < 0 or args.doc_stride >= args.max_seq_length:
        raise ValueError("--doc-stride must be non-negative and smaller than --max-seq-length")
    if args.threshold_step <= 0 or not 0 <= args.threshold_min <= args.threshold_max <= 1:
        raise ValueError("Invalid threshold grid")
    if not 0 <= args.discard_contamination_limit <= 1:
        raise ValueError("--discard-contamination-limit must be in [0, 1]")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")


def label_targets(label: str, no_doc_correct: Any) -> list[int]:
    normalized = normalize_training_label(label)
    no_doc_group = int(bool(no_doc_correct))
    if normalized == "helpful":
        return [1, 1, no_doc_group]
    if normalized == "not helpful":
        return [1, 0, no_doc_group]
    return [0, -100, no_doc_group]


def tokenize_split(
    dataset: Dataset,
    tokenizer: Any,
    args: argparse.Namespace,
    split_name: str,
) -> Dataset:
    def preprocess(examples: dict[str, list[Any]]) -> dict[str, Any]:
        targets = [
            label_targets(label, correct)
            for label, correct in zip(examples["target"], examples["no_doc_correct"])
        ]
        class_ids = [CLASS_INDEX[normalize_training_label(label)] for label in examples["target"]]
        if args.overlength_policy == "overflow":
            encoded = tokenizer(
                examples["_filter_input"],
                max_length=args.max_seq_length,
                truncation=True,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                padding=False,
            )
            mapping = encoded.pop("overflow_to_sample_mapping")
            encoded["labels"] = [targets[int(index)] for index in mapping]
            encoded["class_id"] = [class_ids[int(index)] for index in mapping]
            return encoded
        encoded = tokenizer(examples["_filter_input"], truncation=False, padding=False)
        encoded["labels"] = targets
        encoded["class_id"] = class_ids
        return encoded

    kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": dataset.column_names,
        "desc": f"Two-head tokenization: {split_name}",
    }
    if args.preprocessing_num_workers > 1:
        kwargs["num_proc"] = args.preprocessing_num_workers
    return dataset.map(preprocess, **kwargs)


def attach_sampling_weights(dataset: Dataset, mode: str) -> tuple[Dataset, dict[str, Any]]:
    counts = Counter(int(value) for value in dataset["class_id"])
    names_by_index = {value: key for key, value in CLASS_INDEX.items()}
    if set(counts) != set(names_by_index):
        raise ValueError(f"Training features must contain all H/NH/D classes, got {counts}")
    if mode == "natural":
        weights_by_class = {index: 1.0 for index in counts}
        expected = {
            names_by_index[index]: count / len(dataset)
            for index, count in sorted(counts.items())
        }
    else:
        weights_by_class = {
            index: DESIRED_HIERARCHICAL_RATIOS[names_by_index[index]] / count
            for index, count in counts.items()
        }
        expected = dict(DESIRED_HIERARCHICAL_RATIOS)
    weights = [float(weights_by_class[int(value)]) for value in dataset["class_id"]]
    dataset = dataset.add_column("sample_weight", weights)
    return dataset, {
        "mode": mode,
        "feature_counts": {names_by_index[index]: count for index, count in sorted(counts.items())},
        "expected_sampled_ratios": expected,
        "samples_per_epoch": len(dataset),
        "replacement": mode == "hierarchical_balanced",
    }


class TwoHeadCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature.pop("labels") for feature in features], dtype=torch.long)
        sample_weights = torch.tensor(
            [float(feature.pop("sample_weight", 1.0)) for feature in features],
            dtype=torch.float32,
        )
        for feature in features:
            feature.pop("class_id", None)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels
        batch["sample_weight"] = sample_weights
        return batch


class HierarchicalBalancedTrainer(Trainer):
    def _get_train_sampler(self, train_dataset: Dataset | None = None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or "sample_weight" not in dataset.column_names:
            return super()._get_train_sampler(train_dataset)
        weights = torch.as_tensor(dataset["sample_weight"], dtype=torch.double)
        if torch.allclose(weights, weights[0]):
            return super()._get_train_sampler(train_dataset)
        generator = torch.Generator()
        generator.manual_seed(int(self.args.data_seed if self.args.data_seed is not None else self.args.seed))
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


def actual_classes(labels: np.ndarray) -> np.ndarray:
    decisive = labels[:, 0].astype(np.int64)
    utility = labels[:, 1].astype(np.int64)
    return np.where(decisive == 0, 2, np.where(utility == 1, 0, 1)).astype(np.int64)


def predicted_classes(logits: np.ndarray, decisive_threshold: float, helpful_threshold: float) -> np.ndarray:
    p_decisive = softmax_np(logits[:, :2])[:, 1]
    p_helpful = softmax_np(logits[:, 2:])[:, 1]
    return np.where(
        p_decisive < decisive_threshold,
        2,
        np.where(p_helpful >= helpful_threshold, 0, 1),
    ).astype(np.int64)


def three_class_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    decisive_threshold: float,
    helpful_threshold: float,
) -> dict[str, float]:
    targets = actual_classes(labels)
    predictions = predicted_classes(logits, decisive_threshold, helpful_threshold)
    names = ("helpful", "not_helpful", "discard")
    metrics: dict[str, float] = {
        "accuracy": float(np.mean(targets == predictions)) if len(targets) else 0.0,
        "predicted_helpful_rate": float(np.mean(predictions == 0)) if len(targets) else 0.0,
        "decisive_threshold": float(decisive_threshold),
        "helpful_threshold": float(helpful_threshold),
    }
    recalls: list[float] = []
    f1s: list[float] = []
    for index, name in enumerate(names):
        tp = int(np.sum((targets == index) & (predictions == index)))
        fp = int(np.sum((targets != index) & (predictions == index)))
        fn = int(np.sum((targets == index) & (predictions != index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall
        metrics[f"f1_{name}"] = f1
        metrics[f"support_{name}"] = float(np.sum(targets == index))
        metrics[f"predicted_{name}"] = float(np.sum(predictions == index))
        recalls.append(recall)
        f1s.append(f1)
    metrics["balanced_accuracy"] = float(np.mean(recalls))
    metrics["macro_f1"] = float(np.mean(f1s))

    predicted_helpful = predictions == 0
    actual_discard = targets == 2
    metrics["discard_contamination_in_predicted_helpful"] = (
        float(np.sum(predicted_helpful & actual_discard) / np.sum(predicted_helpful))
        if np.any(predicted_helpful)
        else 0.0
    )
    metrics["discard_pass_rate"] = (
        float(np.sum(predicted_helpful & actual_discard) / np.sum(actual_discard))
        if np.any(actual_discard)
        else 0.0
    )
    actual_not_helpful = targets == 1
    metrics["not_helpful_pass_rate"] = (
        float(np.sum(predicted_helpful & actual_not_helpful) / np.sum(actual_not_helpful))
        if np.any(actual_not_helpful)
        else 0.0
    )

    group_scores: list[float] = []
    no_doc_correct = labels[:, 2].astype(bool)
    for group_value, group_name in ((True, "no_rag_correct"), (False, "no_rag_wrong")):
        mask = no_doc_correct == group_value
        if not np.any(mask):
            continue
        group_f1s: list[float] = []
        for index in range(3):
            tp = int(np.sum(mask & (targets == index) & (predictions == index)))
            fp = int(np.sum(mask & (targets != index) & (predictions == index)))
            fn = int(np.sum(mask & (targets == index) & (predictions != index)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            group_f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        group_macro_f1 = float(np.mean(group_f1s))
        metrics[f"macro_f1_{group_name}"] = group_macro_f1
        group_scores.append(group_macro_f1)
    metrics["worst_group_macro_f1"] = min(group_scores) if group_scores else 0.0

    p_decisive = softmax_np(logits[:, :2])[:, 1]
    p_helpful = softmax_np(logits[:, 2:])[:, 1]
    decisive_targets = labels[:, 0].astype(np.int64)
    utility_targets = labels[:, 1].astype(np.int64)
    metrics["head1_decisive_accuracy"] = float(np.mean((p_decisive >= 0.5) == decisive_targets))
    utility_mask = utility_targets >= 0
    metrics["head2_direction_accuracy_decisive_only"] = (
        float(np.mean((p_helpful[utility_mask] >= 0.5) == utility_targets[utility_mask]))
        if np.any(utility_mask)
        else 0.0
    )
    metrics["helpful_f1"] = metrics["f1_helpful"]
    return metrics


def build_metric_functions():
    def preprocess_logits_for_metrics(logits: Any, labels: torch.Tensor) -> torch.Tensor:
        del labels
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.float()

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        return three_class_metrics(
            np.asarray(eval_prediction.predictions),
            np.asarray(eval_prediction.label_ids),
            decisive_threshold=0.5,
            helpful_threshold=0.5,
        )

    return preprocess_logits_for_metrics, compute_metrics


def select_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    values = np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step)
    candidates: list[dict[str, Any]] = []
    for decisive_threshold in values:
        for helpful_threshold in values:
            metrics = three_class_metrics(
                logits,
                labels,
                decisive_threshold=float(decisive_threshold),
                helpful_threshold=float(helpful_threshold),
            )
            candidates.append(
                {
                    "eligible": (
                        metrics["predicted_helpful"] > 0
                        and metrics["discard_contamination_in_predicted_helpful"]
                        <= args.discard_contamination_limit
                    ),
                    **metrics,
                }
            )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda value: (
                value["recall_helpful"],
                value["precision_helpful"],
                value["worst_group_macro_f1"],
                -value["predicted_helpful_rate"],
            ),
        )
        fallback = False
    else:
        nonempty = [candidate for candidate in candidates if candidate["predicted_helpful"] > 0]
        selected = min(
            nonempty,
            key=lambda value: (
                value["discard_contamination_in_predicted_helpful"],
                -value["recall_helpful"],
                -value["precision_helpful"],
            ),
        )
        fallback = True
    return {
        "selection_rule": (
            "maximize validation Helpful recall subject to predicted-Helpful Discard contamination limit; "
            "tie-break by Helpful precision and no-RAG worst-group macro-F1"
        ),
        "discard_contamination_limit": args.discard_contamination_limit,
        "fallback_to_minimum_contamination": fallback,
        "selected": selected,
        "grid": candidates,
    }


class EpochStageCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        logging.info(
            "Workflow stage 4/6: training epoch %.0f/%s; batch progress and ETA follow in the live Trainer bar",
            math.floor(state.epoch or 0) + 1,
            math.ceil(state.num_train_epochs),
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, control, kwargs
        logging.info("Validation completed at step=%s: %s", state.global_step, metrics)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    set_seed(args.seed)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / args.dataset / args.run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Output directory: %s", output_dir)
    logging.info("Overall workflow progress=0/6; overall ETA=unknown until tokenized feature counts are known")

    # Imported split utilities expect these deliberately disabled ablation fields.
    args.include_no_rag_rationale = False
    args.no_rag_root = None
    args.no_rag_rationale_field = "rationale_only"

    logging.info("Workflow stage 1/6: loading question-disjoint train/validation/test splits")
    train_raw, validation_raw, test_raw, split_paths, _ = load_splits(args)
    for split_name, dataset in (("train", train_raw), ("validation", validation_raw), ("test", test_raw)):
        observed = {normalize_training_label(value) for value in dataset["target"]}
        if observed != set(CLASS_INDEX):
            raise ValueError(f"{split_name} must contain Helpful/Not Helpful/Discard, got {sorted(observed)}")
        if "no_doc_correct" not in dataset.column_names:
            raise ValueError(f"{split_name} is missing no_doc_correct required for group-robust validation")
        logging.info("%s summary: %s", split_name, summarize(dataset, split_name))
    label_manifest = read_filter_label_manifest(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.model_max_length = args.max_seq_length
    logging.info("Workflow stage 2/6: measuring encoder lengths and applying %s policy", args.overlength_policy)
    train_filtered, train_length_stats = filter_overlength_inputs(train_raw, tokenizer, args, "train")
    validation_filtered, validation_length_stats = filter_overlength_inputs(
        validation_raw, tokenizer, args, "validation"
    )
    test_filtered, test_length_stats = filter_overlength_inputs(test_raw, tokenizer, args, "test")

    logging.info("Workflow stage 3/6: tokenizing all classifier features; per-split progress/ETA follows")
    train = tokenize_split(train_filtered, tokenizer, args, "train")
    validation = tokenize_split(validation_filtered, tokenizer, args, "validation")
    test = tokenize_split(test_filtered, tokenizer, args, "test")
    train, sampling_summary = attach_sampling_weights(train, args.sampling_mode)
    validation = validation.add_column("sample_weight", [1.0] * len(validation))
    test = test.add_column("sample_weight", [1.0] * len(test))
    logging.info(
        "Tokenized features: train=%s validation=%s test=%s sampling=%s",
        len(train), len(validation), len(test), sampling_summary,
    )

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None
    model = Rag2TwoHeadFilterModel.from_base_model(
        args.model_name_or_path,
        dropout=args.dropout,
        decisive_loss_weight=args.decisive_loss_weight,
        utility_loss_weight=args.utility_loss_weight,
        dtype=dtype,
    )
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        model.config.use_cache = False

    steps_per_epoch = math.ceil(
        len(train) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    logging.info(
        "Overall measurable training work: %s optimizer steps (%s/epoch x %s epochs); live ETA begins with Trainer",
        math.ceil(steps_per_epoch * args.num_train_epochs),
        steps_per_epoch,
        args.num_train_epochs,
    )
    preprocess_logits, compute_metrics = build_metric_functions()
    collator = TwoHeadCollator(
        tokenizer,
        pad_to_multiple_of=8 if args.bf16 or args.tf32 else None,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        bf16=args.bf16 and torch.cuda.is_available(),
        bf16_full_eval=args.bf16 and torch.cuda.is_available(),
        tf32=args.tf32 if torch.cuda.is_available() else None,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        eval_accumulation_steps=args.eval_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        label_names=["labels"],
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
    callbacks: list[TrainerCallback] = [
        EpochStageCallback(),
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ),
    ]
    trainer = HierarchicalBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        preprocess_logits_for_metrics=preprocess_logits,
    )
    logging.info("Workflow stage 4/6: two-head optimization; Trainer bar shows exact step progress and ETA")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()

    logging.info("Workflow stage 5/6: validation threshold grid search and held-out test evaluation")
    validation_prediction = trainer.predict(validation, metric_key_prefix="validation_fixed_0p5")
    threshold_selection = select_thresholds(
        np.asarray(validation_prediction.predictions),
        np.asarray(validation_prediction.label_ids),
        args,
    )
    selected = threshold_selection["selected"]
    test_prediction = trainer.predict(test, metric_key_prefix="test_fixed_0p5")
    test_selected_metrics = three_class_metrics(
        np.asarray(test_prediction.predictions),
        np.asarray(test_prediction.label_ids),
        decisive_threshold=float(selected["decisive_threshold"]),
        helpful_threshold=float(selected["helpful_threshold"]),
    )
    write_json(output_dir / "threshold_selection.json", threshold_selection)
    write_json(output_dir / "test_selected_threshold_metrics.json", test_selected_metrics)

    logging.info("Workflow stage 6/6: saving restored best checkpoint and audit report")
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    final_model_dir = output_dir / "final_model"
    unwrapped.save_two_head_pretrained(
        final_model_dir,
        tokenizer=tokenizer,
        metadata={
            "decision_rule": (
                "Discard if p_decisive < theta_decisive; otherwise Helpful if "
                "p_helpful >= theta_helpful, else Not Helpful"
            ),
            "theta_decisive": float(selected["decisive_threshold"]),
            "theta_helpful": float(selected["helpful_threshold"]),
            "only_helpful_passes_downstream": True,
            "no_helpful_fallback": "no_rag",
        },
    )
    report = {
        "type": "rag2_two_head_filter_training",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "final_model": str(final_model_dir),
        "split_paths": split_paths,
        "label_manifest": label_manifest,
        "architecture": {
            "encoder": str(args.model_name_or_path),
            "pooling": "attention-mask mean pooling",
            "head1": "Discard(0) vs Decisive(1), all rows",
            "head2": "Not Helpful(0) vs Helpful(1), H/NH rows only",
            "loss": (
                f"{args.decisive_loss_weight}*CE_decisive + "
                f"{args.utility_loss_weight}*CE_direction(masked on Discard)"
            ),
        },
        "sampling": sampling_summary,
        "length_stats": {
            "train": train_length_stats,
            "validation": validation_length_stats,
            "test": test_length_stats,
        },
        "features": {"train": len(train), "validation": len(validation), "test": len(test)},
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": args.metric_for_best_model,
        "train_metrics": train_result.metrics,
        "validation_fixed_0p5_metrics": validation_prediction.metrics,
        "threshold_selection": threshold_selection,
        "test_fixed_0p5_metrics": test_prediction.metrics,
        "test_selected_threshold_metrics": test_selected_metrics,
        "arguments": vars(args),
    }
    # Convert Paths before JSON serialization.
    report["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["arguments"].items()
    }
    write_json(output_dir / "run_report.json", report)
    logging.info(
        "Two-head workflow complete: best=%s thresholds=(%.2f, %.2f) test_macro_f1=%.4f final=%s",
        trainer.state.best_model_checkpoint,
        selected["decisive_threshold"],
        selected["helpful_threshold"],
        test_selected_metrics["macro_f1"],
        final_model_dir,
    )


if __name__ == "__main__":
    main()

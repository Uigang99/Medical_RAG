#!/usr/bin/env python3
"""Train a document preference ranker from behavioral utility comparisons.

This trainer deliberately does not regress the teacher utility value.  It only
uses two ordinal facts that are available from the cached utility targets:

* document A is preferable to document B when their utility gap is decisive;
* a document is preferable to the NULL (No-RAG) action when its utility is
  decisively positive, and NULL is preferable when it is decisively negative.

The NULL score is fixed to zero.  This removes the otherwise unidentified
question-specific score offset and gives the learned scalar a deployment-time
meaning: positive scores prefer using the document; negative scores prefer
falling back to No-RAG.  Near-ties are not used as preference supervision.
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_margin_regressor import (  # noqa: E402
    MODEL_VERSION,
    MarginRegressorConfig,
    TextMarginRegressor,
)
from medrag.progress import PipelineProgress  # noqa: E402
from scripts.train_rag2_margin_regressor import (  # noqa: E402
    MarginCollator,
    NoRAGAnswerIndex,
    TraceTextStore,
    atomic_json,
    load_best,
    load_checkpoint,
    load_contract,
    load_splits,
    limit_questions,
    make_question_loader,
    save_checkpoint,
    shorten_progress_for_early_stop,
)


TRAINER_VERSION = "rag2_pairwise_utility_null_ranker_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=base / "gold_margin_regression_v1/prepared",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/Flan-T5-large",
    )
    parser.add_argument(
        "--input-mode",
        choices=("text_only", "text_no_rag_answer"),
        default="text_no_rag_answer",
    )
    parser.add_argument(
        "--no-rag-generation-root",
        type=Path,
        default=base / "train_no_rag_anchored_features_v1/no_rag",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/RAG2-PairwiseUtilityRanker-FlanT5-large",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-train-epochs", type=int, required=True)
    parser.add_argument("--train-questions-per-batch", type=int, default=16)
    parser.add_argument("--eval-questions-per-batch", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--document-pair-min-utility-gap",
        type=float,
        default=0.1,
        help="Ignore document-document comparisons whose teacher utility gap is smaller.",
    )
    parser.add_argument(
        "--null-min-absolute-utility",
        type=float,
        default=0.1,
        help="Compare a document with fixed NULL=0 only when |teacher utility| reaches this value.",
    )
    parser.add_argument("--pairwise-temperature", type=float, default=0.1)
    parser.add_argument("--document-pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--null-pair-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--balance-no-rag-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Give No-RAG-correct and No-RAG-wrong questions equal loss mass without exposing that flag to the model.",
    )
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


def _weighted_question_mean(losses: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    stacked = torch.stack(losses)
    weight = stacked.new_tensor(weights)
    return (stacked * weight).sum() / weight.sum().clamp_min(1e-12)


def pairwise_null_preference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    question_index: torch.Tensor,
    no_rag_correct: torch.Tensor,
    *,
    document_min_gap: float,
    null_min_gap: float,
    temperature: float,
    state_weights: Sequence[float] = (1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Question-macro logistic losses for document pairs and NULL pairs.

    ``state_weights`` are ordered as No-RAG-wrong, No-RAG-correct.  They affect
    loss aggregation only; the correctness flag is never part of model input.
    """

    document_losses: list[torch.Tensor] = []
    document_weights: list[float] = []
    null_losses: list[torch.Tensor] = []
    null_weights: list[float] = []
    counters = {
        "document_questions": 0,
        "document_comparisons": 0,
        "null_questions": 0,
        "null_comparisons": 0,
    }
    for group in torch.unique(question_index):
        positions = question_index == group
        group_prediction = prediction[positions]
        group_target = target[positions]
        group_correct = bool(no_rag_correct[positions][0].item())
        state_weight = float(state_weights[int(group_correct)])

        target_difference = group_target[:, None] - group_target[None, :]
        preferred = target_difference >= float(document_min_gap)
        document_count = int(preferred.sum().item())
        if document_count:
            prediction_difference = group_prediction[:, None] - group_prediction[None, :]
            document_losses.append(
                F.softplus(-prediction_difference[preferred] / float(temperature)).mean()
            )
            document_weights.append(state_weight)
            counters["document_questions"] += 1
            counters["document_comparisons"] += document_count

        decisive_null = group_target.abs() >= float(null_min_gap)
        null_count = int(decisive_null.sum().item())
        if null_count:
            direction = torch.sign(group_target[decisive_null])
            # NULL has fixed score zero, so the preferred score difference is
            # +s(D) for helpful documents and -s(D) for harmful documents.
            null_losses.append(
                F.softplus(
                    -direction * group_prediction[decisive_null] / float(temperature)
                ).mean()
            )
            null_weights.append(state_weight)
            counters["null_questions"] += 1
            counters["null_comparisons"] += null_count

    zero = prediction.sum() * 0.0
    document_loss = (
        _weighted_question_mean(document_losses, document_weights)
        if document_losses
        else zero
    )
    null_loss = _weighted_question_mean(null_losses, null_weights) if null_losses else zero
    return document_loss, null_loss, counters


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def preference_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    sample_ids: Sequence[str],
    no_rag_correct: np.ndarray,
    document_ranks: np.ndarray,
    *,
    document_min_gap: float,
    null_min_gap: float,
) -> dict[str, Any]:
    """Evaluate ordinal preference recovery and NULL-anchored selection."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped[str(sample_id)].append(index)

    document_correct = 0
    document_total = 0
    null_correct = 0
    null_total = 0
    null_positive_correct = 0
    null_positive_total = 0
    null_negative_correct = 0
    null_negative_total = 0
    document_question_accuracy: list[float] = []
    null_question_accuracy: list[float] = []
    combined_question_accuracy: list[float] = []
    correlations: list[float] = []
    top1_teacher_matches = 0
    top1_selected_document = 0
    top1_true_utility: list[float] = []
    top1_regret: list[float] = []
    per_state: dict[str, dict[str, list[float]]] = {
        "no_rag_correct": defaultdict(list),
        "no_rag_wrong": defaultdict(list),
    }

    for indices in grouped.values():
        positions = np.asarray(indices, dtype=np.int64)
        truth = target[positions]
        score = prediction[positions]
        state = "no_rag_correct" if bool(no_rag_correct[positions[0]]) else "no_rag_wrong"

        question_document_outcomes: list[float] = []
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                true_difference = float(truth[left] - truth[right])
                if abs(true_difference) < float(document_min_gap):
                    continue
                predicted_difference = float(score[left] - score[right])
                correct = float(true_difference * predicted_difference > 0.0)
                question_document_outcomes.append(correct)
        if question_document_outcomes:
            value = float(np.mean(question_document_outcomes))
            document_question_accuracy.append(value)
            per_state[state]["document_question_accuracy"].append(value)
            document_correct += int(sum(question_document_outcomes))
            document_total += len(question_document_outcomes)

        decisive = np.abs(truth) >= float(null_min_gap)
        question_null_outcomes: list[float] = []
        if np.any(decisive):
            products = truth[decisive] * score[decisive]
            question_null_outcomes = (products > 0.0).astype(np.float32).tolist()
            value = float(np.mean(question_null_outcomes))
            null_question_accuracy.append(value)
            per_state[state]["null_question_accuracy"].append(value)
            null_correct += int(sum(question_null_outcomes))
            null_total += len(question_null_outcomes)
            positive = truth[decisive] > 0
            negative = truth[decisive] < 0
            null_positive_correct += int(np.sum(products[positive] > 0.0))
            null_positive_total += int(np.sum(positive))
            null_negative_correct += int(np.sum(products[negative] > 0.0))
            null_negative_total += int(np.sum(negative))

        combined = question_document_outcomes + question_null_outcomes
        if combined:
            value = float(np.mean(combined))
            combined_question_accuracy.append(value)
            per_state[state]["combined_question_accuracy"].append(value)

        if len(truth) >= 2 and np.std(truth) > 0 and np.std(score) > 0:
            correlation = float(spearmanr(truth, score).statistic)
            if math.isfinite(correlation):
                correlations.append(correlation)
                per_state[state]["spearman"].append(correlation)

        # Add a fixed NULL candidate at index zero to both teacher and model.
        teacher_values = np.concatenate((np.asarray([0.0]), truth))
        model_values = np.concatenate((np.asarray([0.0]), score))
        teacher_choice = int(np.argmax(teacher_values))
        model_choice = int(np.argmax(model_values))
        top1_teacher_matches += int(teacher_choice == model_choice)
        top1_selected_document += int(model_choice != 0)
        selected_utility = float(teacher_values[model_choice])
        regret = float(np.max(teacher_values) - selected_utility)
        top1_true_utility.append(selected_utility)
        top1_regret.append(regret)
        per_state[state]["top1_true_utility"].append(selected_utility)
        per_state[state]["top1_regret"].append(regret)

    positive_recall = (
        null_positive_correct / null_positive_total if null_positive_total else float("nan")
    )
    negative_recall = (
        null_negative_correct / null_negative_total if null_negative_total else float("nan")
    )
    balanced_null = (
        float(np.mean([positive_recall, negative_recall]))
        if math.isfinite(positive_recall) and math.isfinite(negative_recall)
        else float("nan")
    )
    return {
        "questions": len(grouped),
        "document_pair": {
            "comparisons": document_total,
            "micro_accuracy": document_correct / document_total if document_total else float("nan"),
            "question_macro_accuracy": _safe_mean(document_question_accuracy),
        },
        "null_pair": {
            "comparisons": null_total,
            "micro_accuracy": null_correct / null_total if null_total else float("nan"),
            "question_macro_accuracy": _safe_mean(null_question_accuracy),
            "helpful_recall": positive_recall,
            "harmful_recall": negative_recall,
            "balanced_accuracy": balanced_null,
            "helpful_support": null_positive_total,
            "harmful_support": null_negative_total,
        },
        "combined_question_macro_accuracy": _safe_mean(combined_question_accuracy),
        "within_question_mean_spearman": _safe_mean(correlations),
        "top1_with_null": {
            "teacher_choice_match_rate": top1_teacher_matches / len(grouped) if grouped else float("nan"),
            "selected_document_rate": top1_selected_document / len(grouped) if grouped else float("nan"),
            "mean_true_utility": _safe_mean(top1_true_utility),
            "mean_regret": _safe_mean(top1_regret),
        },
        "by_no_rag_state": {
            state: {
                "questions": len(values.get("combined_question_accuracy", [])),
                "document_question_macro_accuracy": _safe_mean(
                    values.get("document_question_accuracy", [])
                ),
                "null_question_macro_accuracy": _safe_mean(
                    values.get("null_question_accuracy", [])
                ),
                "combined_question_macro_accuracy": _safe_mean(
                    values.get("combined_question_accuracy", [])
                ),
                "within_question_mean_spearman": _safe_mean(values.get("spearman", [])),
                "top1_mean_true_utility": _safe_mean(values.get("top1_true_utility", [])),
                "top1_mean_regret": _safe_mean(values.get("top1_regret", [])),
            }
            for state, values in per_state.items()
        },
        "thresholds": {
            "document_min_gap": float(document_min_gap),
            "null_min_absolute_utility": float(null_min_gap),
        },
        "audit": {
            "mean_document_rank": float(np.mean(document_ranks)),
        },
    }


def summarize_supervision(dataset: Any, args: argparse.Namespace, split: str) -> dict[str, Any]:
    sample_ids = [str(value) for value in dataset["sample_id"]]
    targets = np.asarray(dataset["utility_target"], dtype=np.float32)
    states = np.asarray(dataset["no_rag_correct_audit_only"], dtype=bool)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped[sample_id].append(index)
    counters: dict[str, Any] = {
        "questions": len(grouped),
        "documents": len(dataset),
        "document_comparisons": 0,
        "null_comparisons": 0,
        "eligible_questions": {"no_rag_wrong": 0, "no_rag_correct": 0},
    }
    progress = tqdm(
        grouped.values(),
        total=len(grouped),
        desc=f"PreferencePlan:{args.dataset}:{split}",
        unit="question",
        dynamic_ncols=True,
        disable=not args.show_progress,
    )
    for indices in progress:
        values = targets[np.asarray(indices, dtype=np.int64)]
        differences = np.abs(values[:, None] - values[None, :])
        document_comparisons = int(np.sum(np.triu(differences >= args.document_pair_min_utility_gap, 1)))
        null_comparisons = int(np.sum(np.abs(values) >= args.null_min_absolute_utility))
        counters["document_comparisons"] += document_comparisons
        counters["null_comparisons"] += null_comparisons
        if document_comparisons or null_comparisons:
            state = "no_rag_correct" if bool(states[indices[0]]) else "no_rag_wrong"
            counters["eligible_questions"][state] += 1
    counters["total_comparisons"] = (
        counters["document_comparisons"] + counters["null_comparisons"]
    )
    return counters


def no_rag_state_weights(summary: dict[str, Any], enabled: bool) -> tuple[float, float]:
    if not enabled:
        return 1.0, 1.0
    counts = summary["eligible_questions"]
    wrong = int(counts["no_rag_wrong"])
    correct = int(counts["no_rag_correct"])
    if not wrong or not correct:
        return 1.0, 1.0
    total = wrong + correct
    return total / (2.0 * wrong), total / (2.0 * correct)


@torch.no_grad()
def evaluate(
    model: TextMarginRegressor,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage: str,
    *,
    keep_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    sample_ids: list[str] = []
    pair_ids: list[str] = []
    progress.set_stage(stage, total=len(loader.dataset))
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            output = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            )
        predictions.append(output["utility_score"].float().cpu().numpy())
        targets.append(batch["utility_target"].numpy())
        states.append(batch["no_rag_correct"].numpy())
        ranks.append(batch["document_ranks"].numpy())
        sample_ids.extend(batch["sample_ids"])
        pair_ids.extend(batch["pair_ids"])
        progress.update(len(batch["pair_ids"]))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    no_rag_correct = np.concatenate(states).astype(bool)
    document_ranks = np.concatenate(ranks)
    metrics = preference_metrics(
        target,
        prediction,
        sample_ids,
        no_rag_correct,
        document_ranks,
        document_min_gap=args.document_pair_min_utility_gap,
        null_min_gap=args.null_min_absolute_utility,
    )
    rows: list[dict[str, Any]] = []
    if keep_rows:
        rows = [
            {
                "sample_id": sample_ids[index],
                "pair_id": pair_ids[index],
                "document_rank": int(document_ranks[index]),
                "teacher_utility_audit_only": float(target[index]),
                "predicted_preference_score": float(prediction[index]),
                "teacher_prefers_document_to_null": bool(
                    target[index] >= args.null_min_absolute_utility
                ),
                "teacher_prefers_null_to_document": bool(
                    target[index] <= -args.null_min_absolute_utility
                ),
                "model_prefers_document_to_null": bool(prediction[index] > 0.0),
                "no_rag_correct_audit_only": bool(no_rag_correct[index]),
            }
            for index in range(len(target))
        ]
    return metrics, rows


def write_jsonl_with_progress(
    path: Path,
    rows: Sequence[dict[str, Any]],
    progress: PipelineProgress,
    stage: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    progress.set_stage(stage, total=len(rows))
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            progress.update(1)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA is required")
    positive_values = (
        args.num_train_epochs,
        args.train_questions_per_batch,
        args.eval_questions_per_batch,
        args.gradient_accumulation_steps,
        args.document_pair_min_utility_gap,
        args.null_min_absolute_utility,
        args.pairwise_temperature,
        args.document_pair_loss_weight,
        args.null_pair_loss_weight,
    )
    if any(float(value) <= 0 for value in positive_values):
        raise ValueError("Epochs, batch sizes, gaps, temperatures, and loss weights must be positive")
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
    datasets["test"] = limit_questions(
        datasets["test"], args.max_eval_questions, args.seed + 2
    )
    supervision = {
        split: summarize_supervision(dataset, args, split)
        for split, dataset in datasets.items()
    }
    state_weights = no_rag_state_weights(supervision["train"], args.balance_no_rag_states)
    logging.info(
        "Pairwise supervision ready: %s state_weights(wrong/correct)=%s",
        json.dumps(supervision, ensure_ascii=False),
        state_weights,
    )

    source_split = str(manifest["source_split"])
    no_rag_answers = (
        NoRAGAnswerIndex(
            args.no_rag_generation_root,
            args.dataset,
            source_split,
            args.show_progress,
        )
        if args.input_mode == "text_no_rag_answer"
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), use_fast=True)
    store = TraceTextStore(
        Path(str(manifest["trace_root"])),
        args.dataset,
        source_split,
        args.trace_shard_cache_size,
        no_rag_answers,
    )
    collator = MarginCollator(tokenizer, store, args)
    # Keep every document of a question in one batch for exact pair formation.
    loaders = {
        "train": make_question_loader(
            datasets["train"], collator, args.train_questions_per_batch, args.seed, True
        ),
        "validation": make_question_loader(
            datasets["validation"], collator, args.eval_questions_per_batch, args.seed + 1, False
        ),
        "test": make_question_loader(
            datasets["test"], collator, args.eval_questions_per_batch, args.seed + 2, False
        ),
    }
    if args.dry_run:
        batch = next(iter(loaders["train"]))
        fake_prediction = torch.linspace(-0.5, 0.5, len(batch["pair_ids"]))
        document_loss, null_loss, counters = pairwise_null_preference_loss(
            fake_prediction,
            batch["utility_target"],
            batch["question_index"],
            batch["no_rag_correct"],
            document_min_gap=args.document_pair_min_utility_gap,
            null_min_gap=args.null_min_absolute_utility,
            temperature=args.pairwise_temperature,
            state_weights=state_weights,
        )
        logging.info(
            "Dry run complete: input=%s document_loss=%.6f null_loss=%.6f counters=%s",
            tuple(batch["input_ids"].shape),
            float(document_loss),
            float(null_loss),
            counters,
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
        logging.info(
            "Resumed exactly from %s after epoch %d",
            checkpoint,
            trainer_state["completed_epoch"],
        )

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
        "supervision_summary": supervision,
        "model_input": (
            ["question", "options", "No-RAG predicted answer", "one document"]
            if args.input_mode == "text_no_rag_answer"
            else ["question", "options", "one document"]
        ),
        "supervision": {
            "document_document": (
                "prefer D_i over D_j iff teacher utility(D_i)-utility(D_j) "
                f">={args.document_pair_min_utility_gap}"
            ),
            "document_null": (
                f"prefer D over NULL iff utility(D)>={args.null_min_absolute_utility}; "
                f"prefer NULL over D iff utility(D)<=-{args.null_min_absolute_utility}"
            ),
            "null_score": 0.0,
            "near_ties": "excluded from the loss",
            "absolute_utility_regression": False,
        },
        "no_rag_state_balancing": {
            "enabled": args.balance_no_rag_states,
            "wrong_correct_question_weights": list(state_weights),
            "audit_flag_is_not_model_input": True,
        },
        "forbidden_model_inputs": [
            "gold answer",
            "teacher utility value",
            "teacher logits/margins",
            "No-RAG correctness",
            "answer transition",
            "semantic labels",
            "hidden states",
        ],
        "deployment": (
            "score each retrieved document independently; rank by score; use only score>0 documents; "
            "if every score<=0 choose NULL/No-RAG"
        ),
        "checkpoint_selection": "maximum validation combined question-macro preference accuracy",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    atomic_json(output_dir / "reproduction_manifest.json", reproduction)

    initial_validation_work = len(datasets["validation"])
    initial_validation_recorded = "initial_validation" in trainer_state
    per_epoch_work = len(datasets["train"]) + len(datasets["validation"])
    final_work = len(datasets["validation"]) + 2 * len(datasets["test"])
    progress = PipelineProgress(
        overall_total=(
            initial_validation_work + args.num_train_epochs * per_epoch_work + final_work
        ),
        overall_initial=(
            int(trainer_state["completed_epoch"]) * per_epoch_work
            + (initial_validation_work if initial_validation_recorded else 0)
        ),
        desc=f"PairwiseUtility:{args.dataset}",
        enabled=args.show_progress,
    )
    try:
        if not initial_validation_recorded:
            initial_validation, _ = evaluate(
                model,
                loaders["validation"],
                device,
                args,
                progress,
                "1/3 untrained validation baseline",
            )
            trainer_state["initial_validation"] = initial_validation
            save_checkpoint(
                output_dir / "last_checkpoint", model, optimizer, scheduler, trainer_state
            )
            atomic_json(output_dir / "training_history.json", trainer_state)
            logging.info(
                "Untrained baseline: combined/doc/null macro=%.4f/%.4f/%.4f",
                initial_validation["combined_question_macro_accuracy"],
                initial_validation["document_pair"]["question_macro_accuracy"],
                initial_validation["null_pair"]["question_macro_accuracy"],
            )
        for epoch in range(int(trainer_state["completed_epoch"]) + 1, args.num_train_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress.set_stage(
                f"1/3 train pairwise+NULL epoch {epoch}/{args.num_train_epochs}",
                total=len(datasets["train"]),
            )
            rolling = defaultdict(float)
            rolling_batches = 0
            for batch_index, batch in enumerate(loaders["train"], 1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    output = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    )
                    document_loss, null_loss, counters = pairwise_null_preference_loss(
                        output["utility_score"].float(),
                        batch["utility_target"].to(device, non_blocking=True),
                        batch["question_index"].to(device, non_blocking=True),
                        batch["no_rag_correct"].to(device, non_blocking=True),
                        document_min_gap=args.document_pair_min_utility_gap,
                        null_min_gap=args.null_min_absolute_utility,
                        temperature=args.pairwise_temperature,
                        state_weights=state_weights,
                    )
                    loss = (
                        args.document_pair_loss_weight * document_loss
                        + args.null_pair_loss_weight * null_loss
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                update = (
                    batch_index % args.gradient_accumulation_steps == 0
                    or batch_index == len(loaders["train"])
                )
                if update:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (
                                parameter
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ),
                            args.max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    trainer_state["global_step"] += 1
                rolling["loss"] += float(loss.detach().cpu())
                rolling["document_loss"] += float(document_loss.detach().cpu())
                rolling["null_loss"] += float(null_loss.detach().cpu())
                rolling["comparisons"] += (
                    counters["document_comparisons"] + counters["null_comparisons"]
                )
                rolling_batches += 1
                progress.update(len(batch["pair_ids"]))
                if batch_index % args.logging_steps == 0:
                    progress.set_detail(
                        f"batch={batch_index}/{len(loaders['train'])} "
                        f"loss={rolling['loss']/rolling_batches:.4f} "
                        f"doc/null={rolling['document_loss']/rolling_batches:.4f}/"
                        f"{rolling['null_loss']/rolling_batches:.4f} "
                        f"comparisons={int(rolling['comparisons'])}"
                    )
                    rolling = defaultdict(float)
                    rolling_batches = 0

            validation, _ = evaluate(
                model,
                loaders["validation"],
                device,
                args,
                progress,
                f"2/3 validation preferences epoch {epoch}/{args.num_train_epochs}",
            )
            metric = float(validation["combined_question_macro_accuracy"])
            if not math.isfinite(metric):
                metric = -float("inf")
            improved = metric > float(trainer_state["best_metric"]) + args.minimum_improvement
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
            correct = validation["by_no_rag_state"]["no_rag_correct"]
            wrong = validation["by_no_rag_state"]["no_rag_wrong"]
            logging.info(
                "Epoch %d: combined/doc/null macro=%.4f/%.4f/%.4f null balanced=%.4f "
                "within-Q Spearman=%.4f correct/wrong combined=%.4f/%.4f best=%s",
                epoch,
                validation["combined_question_macro_accuracy"],
                validation["document_pair"]["question_macro_accuracy"],
                validation["null_pair"]["question_macro_accuracy"],
                validation["null_pair"]["balanced_accuracy"],
                validation["within_question_mean_spearman"],
                correct["combined_question_macro_accuracy"],
                wrong["combined_question_macro_accuracy"],
                trainer_state["best_epoch"],
            )
            if trainer_state["epochs_without_improvement"] >= args.early_stopping_patience:
                logging.info("Early stopping after %d unimproved epochs", args.early_stopping_patience)
                shorten_progress_for_early_stop(progress, final_work)
                break

        if trainer_state["best_epoch"] is None:
            raise RuntimeError("No finite validation preference checkpoint was produced")
        load_best(output_dir / "best_model", model)
        best_validation, _ = evaluate(
            model,
            loaders["validation"],
            device,
            args,
            progress,
            "3/3 final best-model validation",
        )
        test, test_rows = evaluate(
            model,
            loaders["test"],
            device,
            args,
            progress,
            "3/3 final held-out preference test",
            keep_rows=True,
        )
        write_jsonl_with_progress(
            output_dir / "test_predictions.jsonl",
            test_rows,
            progress,
            "3/3 save held-out predictions",
        )
        final_metrics = {
            "best_epoch": trainer_state["best_epoch"],
            "validation": best_validation,
            "test": test,
        }
        atomic_json(output_dir / "final_metrics.json", final_metrics)
        model.save_trainable(output_dir / "final_model")
        atomic_json(output_dir / "final_model/metrics.json", final_metrics)
        logging.info("Pairwise utility training complete: %s", output_dir)
        logging.info("Final held-out preference test: %s", json.dumps(test, ensure_ascii=False))
    finally:
        progress.close()


if __name__ == "__main__":
    main()

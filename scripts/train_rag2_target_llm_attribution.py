#!/usr/bin/env python3
"""Train a variable-size Target-LLM conditional-removal attribution predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.attribution.target_llm_predictor import (  # noqa: E402
    TargetLLMAttributionPredictor,
    attribution_loss,
    masked_document_distribution,
)
from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_target_llm_conditional_removal_predictor_training_v1"
PREPARED_RUN_VERSION = "rag2_target_llm_conditional_removal_teacher_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--total-loss-weight", type=float, default=1.0)
    parser.add_argument("--share-loss-weight", type=float, default=0.5)
    parser.add_argument("--set-shift-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument("--minimum-total-for-share", type=float, default=1e-6)
    parser.add_argument("--minimum-rank-log-ratio", type=float, default=0.25)
    parser.add_argument(
        "--use-rank-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-length-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shuffle-documents-during-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memorization-check", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
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


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


class AttributionRows(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return torch.load(self.paths[index], map_location="cpu", weights_only=False)


def selected_paths(
    feature_dir: Path,
    split: str,
    *,
    sample_ids: list[str],
    maximum: int,
    seed: int,
) -> list[Path]:
    paths = [
        feature_dir
        / "rows"
        / split
        / f"{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:24]}.pt"
        for sample_id in sample_ids
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared attribution rows are missing for {split}: {missing[:5]}")
    if maximum > 0 and len(paths) > maximum:
        paths = random.Random(seed).sample(paths, maximum)
    return sorted(paths)


def collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    layers = int(rows[0]["document_features"].shape[1])
    hidden = int(rows[0]["document_features"].shape[2])
    maximum_documents = max(int(row["document_count"]) for row in rows)
    batch = len(rows)
    document_features = torch.zeros(
        (batch, maximum_documents, layers, hidden), dtype=torch.bfloat16
    )
    global_features = torch.zeros((batch, layers, hidden), dtype=torch.bfloat16)
    document_mask = torch.zeros((batch, maximum_documents), dtype=torch.bool)
    relative_rank = torch.zeros((batch, maximum_documents), dtype=torch.float32)
    normalized_length = torch.zeros((batch, maximum_documents), dtype=torch.float32)
    teacher_influence = torch.zeros((batch, maximum_documents), dtype=torch.float32)
    total_loo = torch.zeros(batch, dtype=torch.float32)
    set_shift = torch.zeros(batch, dtype=torch.float32)
    sample_ids: list[str] = []
    for index, row in enumerate(rows):
        count = int(row["document_count"])
        features = row["document_features"]
        if tuple(features.shape) != (count, layers, hidden):
            raise ValueError(f"Prepared document feature shape mismatch for {row['sample_id']}")
        document_features[index, :count] = features.to(torch.bfloat16)
        global_features[index] = row["global_features"].to(torch.bfloat16)
        document_mask[index, :count] = True
        ranks = row["rerank_positions"].float()
        relative_rank[index, :count] = ranks / max(1.0, float(count - 1))
        lengths = row["document_lengths"].float()
        normalized_length[index, :count] = torch.log1p(lengths) / math.log(2049.0)
        teacher_influence[index, :count] = row["loo_jsd"].float()
        total_loo[index] = float(row["total_loo_jsd"])
        set_shift[index] = float(row["set_shift_jsd"])
        sample_ids.append(str(row["sample_id"]))
    return {
        "sample_ids": sample_ids,
        "document_features": document_features,
        "global_features": global_features,
        "document_mask": document_mask,
        "relative_rank": relative_rank,
        "normalized_length": normalized_length,
        "teacher_influence": teacher_influence,
        "teacher_total_loo": total_loo,
        "teacher_set_shift": set_shift,
    }


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-15:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rankdata(left), rankdata(right))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, dtype=torch.float32 if value.dtype == torch.bfloat16 else None)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


DOCUMENT_ALIGNED_KEYS = (
    "document_features",
    "document_mask",
    "relative_rank",
    "normalized_length",
    "teacher_influence",
)


def permute_document_aligned_batch(
    batch: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Apply a deterministic per-question permutation to all document axes."""

    result = dict(batch)
    for key in DOCUMENT_ALIGNED_KEYS:
        result[key] = batch[key].clone()
    generator = torch.Generator().manual_seed(int(seed))
    for row in range(len(batch["sample_ids"])):
        count = int(batch["document_mask"][row].sum().item())
        permutation = torch.randperm(count, generator=generator)
        for key in DOCUMENT_ALIGNED_KEYS:
            result[key][row, :count] = batch[key][row, permutation]
    return result


def forward_loss(
    model: TargetLLMAttributionPredictor,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, torch.Tensor]]:
    relative_rank = batch["relative_rank"]
    if not args.use_rank_feature:
        relative_rank = torch.zeros_like(relative_rank)
    normalized_length = batch["normalized_length"]
    if not args.use_length_feature:
        normalized_length = torch.zeros_like(normalized_length)
    prediction = model(
        batch["document_features"],
        batch["global_features"],
        batch["document_mask"],
        relative_rank,
        normalized_length,
    )
    losses = attribution_loss(
        prediction,
        teacher_influence=batch["teacher_influence"],
        teacher_total_loo=batch["teacher_total_loo"],
        teacher_set_shift=batch["teacher_set_shift"],
        document_mask=batch["document_mask"],
        minimum_total_for_share=args.minimum_total_for_share,
        epsilon=args.epsilon,
        total_weight=args.total_loss_weight,
        share_weight=args.share_loss_weight,
        set_shift_weight=args.set_shift_loss_weight,
        rank_weight=args.rank_loss_weight,
        minimum_rank_log_ratio=args.minimum_rank_log_ratio,
    )
    return prediction, losses


def evaluate(
    model: TargetLLMAttributionPredictor,
    loader: DataLoader,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage: str,
) -> dict[str, float | int | None]:
    model.eval()
    progress.set_stage(stage, total=len(loader.dataset))
    loss_sums = {name: 0.0 for name in ("loss", "total", "share", "set_shift", "rank")}
    questions = 0
    measurable = 0
    correlations: list[float] = []
    top1 = 0
    share_absolute = 0.0
    uniform_absolute = 0.0
    share_documents = 0
    total_log_errors: list[float] = []
    set_log_errors: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, torch.device(args.device))
            prediction, losses = forward_loss(model, batch, args)
            batch_size = len(batch["sample_ids"])
            questions += batch_size
            for name in loss_sums:
                loss_sums[name] += float(losses[name].item()) * batch_size
            predicted_share = masked_document_distribution(
                prediction.document_logits,
                batch["document_mask"],
            )
            for row in range(batch_size):
                count = int(batch["document_mask"][row].sum().item())
                target_total = float(batch["teacher_total_loo"][row].item())
                total_log_errors.append(
                    abs(float(prediction.log_total_loo[row].item()) - math.log(max(args.epsilon, target_total)))
                )
                set_target = float(batch["teacher_set_shift"][row].item())
                set_log_errors.append(
                    abs(float(prediction.log_set_shift[row].item()) - math.log(max(args.epsilon, set_target)))
                )
                if target_total < args.minimum_total_for_share:
                    continue
                measurable += 1
                target = (
                    batch["teacher_influence"][row, :count] / max(args.epsilon, target_total)
                ).tolist()
                predicted = predicted_share[row, :count].tolist()
                correlation = spearman(predicted, target)
                if correlation is not None:
                    correlations.append(correlation)
                top1 += int(max(range(count), key=predicted.__getitem__) == max(range(count), key=target.__getitem__))
                share_absolute += sum(abs(a - b) for a, b in zip(predicted, target, strict=True))
                uniform_absolute += sum(abs((1.0 / count) - b) for b in target)
                share_documents += count
            progress.update(batch_size)
    return {
        "questions": questions,
        "measurable_questions": measurable,
        **{f"mean_{name}": loss_sums[name] / max(1, questions) for name in loss_sums},
        "mean_per_question_spearman": statistics.fmean(correlations) if correlations else None,
        "median_per_question_spearman": statistics.median(correlations) if correlations else None,
        "top1_accuracy": top1 / max(1, measurable),
        "share_mae": share_absolute / max(1, share_documents),
        "uniform_share_mae": uniform_absolute / max(1, share_documents),
        "mean_abs_log_total_error": statistics.fmean(total_log_errors) if total_log_errors else None,
        "mean_abs_log_set_shift_error": statistics.fmean(set_log_errors) if set_log_errors else None,
    }


def train_epoch(
    model: TargetLLMAttributionPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    progress: PipelineProgress,
    epoch: int,
) -> dict[str, float]:
    model.train()
    progress.set_stage(f"1/3 train epoch {epoch}/{args.epochs}", total=len(loader.dataset))
    sums = {name: 0.0 for name in ("loss", "total", "share", "set_shift", "rank")}
    samples = 0
    for batch_index, batch in enumerate(loader, start=1):
        if args.shuffle_documents_during_training:
            batch = permute_document_aligned_batch(
                batch,
                seed=args.seed + epoch * 1_000_003 + batch_index,
            )
        batch = move_batch(batch, torch.device(args.device))
        optimizer.zero_grad(set_to_none=True)
        _prediction, losses = forward_loss(model, batch, args)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        batch_size = len(batch["sample_ids"])
        samples += batch_size
        for name in sums:
            sums[name] += float(losses[name].item()) * batch_size
        progress.update(batch_size)
        progress.set_detail(
            f"batch={batch_index}/{len(loader)} loss={float(losses['loss'].item()):.4f}"
        )
    return {f"mean_{name}": sums[name] / max(1, samples) for name in sums}


def make_loader(
    paths: list[Path],
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        AttributionRows(paths),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_rows,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.feature_dir.is_dir():
        raise FileNotFoundError(args.feature_dir)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.minimum_total_for_share < 0 or args.epsilon <= 0:
        raise ValueError("signal threshold must be non-negative and epsilon positive")
    loss_weights = (
        args.total_loss_weight,
        args.share_loss_weight,
        args.set_shift_loss_weight,
        args.rank_loss_weight,
    )
    if not all(math.isfinite(weight) for weight in loss_weights):
        raise ValueError("loss weights must be finite")
    if any(weight < 0 for weight in loss_weights) or not any(
        weight > 0 for weight in loss_weights
    ):
        raise ValueError("loss weights must be non-negative and at least one must be positive")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    manifest_path = args.feature_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("run_version") != PREPARED_RUN_VERSION:
        raise ValueError("Unsupported attribution feature manifest")
    if feature_manifest.get("dataset") != args.dataset:
        raise ValueError("Prepared dataset differs from --dataset")

    train_paths = selected_paths(
        args.feature_dir,
        "train",
        sample_ids=list(feature_manifest["selected_sample_ids"]["train"]),
        maximum=args.max_train_samples,
        seed=args.seed,
    )
    if args.memorization_check:
        validation_paths = train_paths
        test_paths = train_paths
    else:
        validation_paths = selected_paths(
            args.feature_dir,
            "val",
            sample_ids=list(feature_manifest["selected_sample_ids"]["val"]),
            maximum=args.max_eval_samples,
            seed=args.seed + 1,
        )
        test_paths = selected_paths(
            args.feature_dir,
            "test",
            sample_ids=list(feature_manifest["selected_sample_ids"]["test"]),
            maximum=args.max_eval_samples,
            seed=args.seed + 2,
        )
    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "feature_dir": str(args.feature_dir.resolve()),
        "feature_contract_fingerprint": feature_manifest["contract_fingerprint"],
        "train_samples": len(train_paths),
        "validation_samples": len(validation_paths),
        "test_samples": len(test_paths),
        "memorization_check": args.memorization_check,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "model_dim": args.model_dim,
        "transformer_layers": args.transformer_layers,
        "attention_heads": args.attention_heads,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "total_loss_weight": args.total_loss_weight,
        "share_loss_weight": args.share_loss_weight,
        "set_shift_loss_weight": args.set_shift_loss_weight,
        "rank_loss_weight": args.rank_loss_weight,
        "minimum_total_for_share": args.minimum_total_for_share,
        "minimum_rank_log_ratio": args.minimum_rank_log_ratio,
        "use_rank_feature": args.use_rank_feature,
        "use_length_feature": args.use_length_feature,
        "shuffle_documents_during_training": args.shuffle_documents_during_training,
        "epsilon": args.epsilon,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        # Runs created before the relative-only objective was added implicitly
        # used a unit weight for the total-LOO loss.
        previous.setdefault("total_loss_weight", 1.0)
        previous.setdefault("use_rank_feature", True)
        previous.setdefault("use_length_feature", True)
        previous.setdefault("shuffle_documents_during_training", False)
        if previous != run_contract:
            raise RuntimeError("Attribution-predictor resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(contract_path, run_contract)
    logging.info(
        "Attribution training plan: train=%d val=%d test=%d epochs=%d batch=%d "
        "memorization=%s rank_feature=%s length_feature=%s train_permutation=%s",
        len(train_paths),
        len(validation_paths),
        len(test_paths),
        args.epochs,
        args.batch_size,
        args.memorization_check,
        args.use_rank_feature,
        args.use_length_feature,
        args.shuffle_documents_during_training,
    )
    if args.plan_only:
        return

    train_loader = make_loader(
        train_paths,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_paths,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed + 1,
    )
    test_loader = make_loader(
        test_paths,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed + 2,
    )
    model = TargetLLMAttributionPredictor(
        target_hidden_size=int(feature_manifest["target_hidden_size"]),
        selected_layer_count=int(feature_manifest["selected_layer_count"]),
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(torch.device(args.device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    latest_path = args.output_dir / "latest_checkpoint.pt"
    best_path = args.output_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_metric = -math.inf
    bad_epochs = 0
    if latest_path.is_file() and args.resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint.get("history") or [])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -math.inf))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        logging.info("Resuming attribution predictor at epoch %d", start_epoch)

    per_epoch = len(train_paths) + len(validation_paths)
    completed_epochs = max(0, start_epoch - 1)
    progress = PipelineProgress(
        overall_total=args.epochs * per_epoch + len(test_paths),
        overall_initial=completed_epochs * per_epoch,
        desc=f"TargetLLMAttributionTrain:{args.dataset}",
    )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, args, progress, epoch)
            validation_metrics = evaluate(
                model,
                validation_loader,
                args,
                progress,
                f"2/3 validation epoch {epoch}/{args.epochs}",
            )
            correlation = validation_metrics["mean_per_question_spearman"]
            metric = float(correlation) if correlation is not None else -float(validation_metrics["mean_loss"])
            improved = metric > best_metric + 1e-8
            if improved:
                best_metric = metric
                bad_epochs = 0
            else:
                bad_epochs += 1
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_metric": metric,
                "best": improved,
            }
            history.append(row)
            checkpoint = {
                "run_version": RUN_VERSION,
                "run_contract": run_contract,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "bad_epochs": bad_epochs,
            }
            atomic_torch_save(latest_path, checkpoint)
            if improved:
                atomic_torch_save(best_path, checkpoint)
            atomic_write_json(args.output_dir / "history.json", history)
            logging.info(
                "Epoch %d: train_loss=%.4f val_spearman=%s val_top1=%.4f "
                "share_mae=%.4f uniform=%.4f best=%s",
                epoch,
                train_metrics["mean_loss"],
                validation_metrics["mean_per_question_spearman"],
                validation_metrics["top1_accuracy"],
                validation_metrics["share_mae"],
                validation_metrics["uniform_share_mae"],
                improved,
            )
            if not args.memorization_check and args.patience > 0 and bad_epochs >= args.patience:
                logging.info("Early stopping after epoch %d", epoch)
                break
        if not best_path.is_file():
            raise RuntimeError("Training produced no best checkpoint")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        test_metrics = evaluate(
            model,
            test_loader,
            args,
            progress,
            "3/3 final held-out attribution test",
        )
        summary = {
            "run_version": RUN_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_contract": run_contract,
            "best_epoch": int(best["epoch"]),
            "best_validation_metric": float(best["best_metric"]),
            "test": test_metrics,
            "active_objectives": {
                "total_loo": args.total_loss_weight > 0,
                "relative_share": args.share_loss_weight > 0,
                "set_shift": args.set_shift_loss_weight > 0,
                "within_question_rank": args.rank_loss_weight > 0,
            },
            "input_features": {
                "rank": args.use_rank_feature,
                "document_length": args.use_length_feature,
                "training_document_permutation": args.shuffle_documents_during_training,
            },
            "interpretation": {
                "output": "predicted conditional-removal sensitivity, not literal attention usage",
                "teacher": feature_manifest["teacher_mode"],
                "variable_k_architecture": True,
                "observed_document_count": feature_manifest["document_count"],
            },
        }
        atomic_write_json(args.output_dir / "summary.json", summary)
        final_bundle = {
            "run_version": RUN_VERSION,
            "run_contract": run_contract,
            "feature_manifest": feature_manifest,
            "model": model.state_dict(),
            "best_epoch": int(best["epoch"]),
            "test": test_metrics,
        }
        atomic_torch_save(args.output_dir / "final_model.pt", final_bundle)
        logging.info(
            "Attribution predictor complete: best_epoch=%d test_spearman=%s top1=%.4f output=%s",
            int(best["epoch"]),
            test_metrics["mean_per_question_spearman"],
            test_metrics["top1_accuracy"],
            args.output_dir,
        )
    finally:
        progress.close()


if __name__ == "__main__":
    main()

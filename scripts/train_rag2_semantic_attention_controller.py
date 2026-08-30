#!/usr/bin/env python3
"""Train a semantic-prior residual controller at the final MCQ choice query.

The target Llama and semantic Flan-T5 encoder are frozen.  The only trainable
parameters are a small residual MLP that converts independent semantic
question-document features into per-document attention-logit biases.  Prefix
KV states are computed without autograd using SDPA; the final ``("` query is
replayed with differentiable document-key biases and gold-option CE.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.generation.learned_semantic_attention import (  # noqa: E402
    SemanticResidualAttentionController,
    document_bias_to_token_bias,
    freeze_module_for_controller_training,
    group_robust_answer_loss,
    residual_anchor_loss,
    semantic_ordering_hinge_loss,
)
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_semantic_final_choice_attention_controller_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
SEMANTIC_CLASS_NAMES = {
    -1: "indeterminate_or_mixed",
    0: "misleading_evidence",
    1: "no_evidence",
    2: "supporting_evidence",
    3: "direct_support",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--controller-hidden-size", type=int, default=256)
    parser.add_argument("--controller-dropout", type=float, default=0.1)
    parser.add_argument("--semantic-temperature", type=float, default=1.0)
    parser.add_argument("--prior-strength", type=float, default=0.25)
    parser.add_argument("--max-suppression-factor", type=float, default=4.0)
    parser.add_argument("--boundary-epsilon", type=float, default=0.05)
    parser.add_argument("--semantic-layer-start", type=int, default=16)
    parser.add_argument("--ordering-margin", type=float, default=0.1)
    parser.add_argument("--ordering-loss-weight", type=float, default=0.1)
    parser.add_argument("--anchor-loss-weight", type=float, default=1e-3)
    parser.add_argument(
        "--no-rag-group-balance",
        type=float,
        default=1.0,
        help="0=natural CE, 1=equal mean CE for No-RAG wrong/correct groups",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
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


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_bundle_identity(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / name
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        if (root / name).is_file()
    ]
    paths.extend(sorted(root.glob("*.safetensors")))
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            **({"sha256": sha256_file(path)} if path.stat().st_size < 16 * 1024 * 1024 else {}),
        }
        for path in paths
    ]


def list_feature_shards(
    feature_dir: Path,
    split: str,
    feature_manifest: dict[str, Any],
) -> list[Path]:
    total = int(feature_manifest["split_questions"][split])
    questions_per_shard = int(feature_manifest["questions_per_shard"])
    expected_count = math.ceil(total / questions_per_shard)
    expected = [
        feature_dir / "feature_shards" / split / f"shard_{index:05d}" / "features.pt"
        for index in range(expected_count)
    ]
    actual = sorted((feature_dir / "feature_shards" / split).glob("shard_*/features.pt"))
    if actual != expected:
        raise RuntimeError(
            f"Prepared {split} shard set differs from its manifest: "
            f"expected={len(expected)} actual={len(actual)}"
        )
    fingerprint = str(feature_manifest["contract_fingerprint"])
    observed = 0
    for index, path in enumerate(expected):
        marker_path = path.with_name("COMPLETE.json")
        if not path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"Prepared feature shard is incomplete: {path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_questions = min(questions_per_shard, total - index * questions_per_shard)
        if (
            marker.get("run_version") != "rag2_semantic_attention_training_features_v1"
            or marker.get("contract_fingerprint") != fingerprint
            or int(marker.get("question_count", -1)) != expected_questions
            or int(marker.get("data_size_bytes", -1)) != path.stat().st_size
            or marker.get("data_sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"Prepared feature marker contract mismatch: {marker_path}")
        observed += expected_questions
    if observed != total:
        raise RuntimeError(f"Prepared {split} question total mismatch: {observed} != {total}")
    return expected


def load_feature_shard(
    path: Path,
    *,
    dataset: str,
    split: str,
    fingerprint: str,
    hidden_size: int,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, dict)
        or value.get("run_version") != "rag2_semantic_attention_training_features_v1"
        or value.get("dataset") != dataset
        or value.get("split") != split
        or value.get("contract_fingerprint") != fingerprint
    ):
        raise ValueError(f"Unsupported prepared feature shard: {path}")
    count = len(value.get("sample_ids") or [])
    required = (
        "semantic_features",
        "semantic_margins",
        "semantic_logits",
        "semantic_targets",
        "semantic_masks",
        "semantic_class_ids",
        "gold_options",
        "baseline_options",
        "no_rag_correct",
        "input_ids",
        "token_document_ids",
        "pair_ids",
    )
    for name in required:
        if name not in value:
            raise ValueError(f"Prepared shard misses {name}: {path}")
    if any(len(value[name]) != count for name in required):
        raise ValueError(f"Prepared shard question count mismatch: {path}")
    exact_shapes = {
        "semantic_features": (count, 8, hidden_size),
        "semantic_margins": (count, 8),
        "semantic_logits": (count, 8, 2),
        "semantic_targets": (count, 8),
        "semantic_masks": (count, 8),
        "semantic_class_ids": (count, 8),
        "gold_options": (count,),
        "baseline_options": (count,),
        "no_rag_correct": (count,),
    }
    for name, shape in exact_shapes.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"Prepared {name} shape mismatch in {path}: expected={shape}")
    masks = value["semantic_masks"].bool()
    targets = value["semantic_targets"].long()
    class_ids = value["semantic_class_ids"].long()
    if not bool(((targets[masks] == 0) | (targets[masks] == 1)).all()):
        raise ValueError(f"Invalid binary semantic target in {path}")
    if not bool(((class_ids[masks] >= 0) & (class_ids[masks] <= 3)).all()):
        raise ValueError(f"Invalid semantic class ID in {path}")
    if bool((targets[~masks] != -1).any()) or bool((class_ids[~masks] != -1).any()):
        raise ValueError(f"Mixed semantic rows are not consistently masked in {path}")
    for row_index, (ids, mapping, pair_ids) in enumerate(
        zip(value["input_ids"], value["token_document_ids"], value["pair_ids"], strict=True)
    ):
        if not isinstance(ids, torch.Tensor) or not isinstance(mapping, torch.Tensor):
            raise TypeError(f"Non-tensor prompt mapping in {path}:{row_index}")
        if ids.ndim != 1 or mapping.ndim != 1 or ids.numel() != mapping.numel():
            raise ValueError(f"Prompt/mapping length mismatch in {path}:{row_index}")
        if len(pair_ids) != 8 or int(mapping[-1]) != -1:
            raise ValueError(f"Top-8/final-anchor mapping mismatch in {path}:{row_index}")
        if bool(((mapping < -1) | (mapping > 7)).any()):
            raise ValueError(f"Out-of-range document token mapping in {path}:{row_index}")
    return value


def count_questions(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        marker = json.loads(path.with_name("COMPLETE.json").read_text(encoding="utf-8"))
        total += int(marker["question_count"])
    return total


def shuffled_batches(size: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def collate_prefix_batch(
    payload: dict[str, Any],
    indices: list[int],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    full_ids = [payload["input_ids"][index].long() for index in indices]
    mappings = [payload["token_document_ids"][index].long() for index in indices]
    if any(ids.numel() < 2 for ids in full_ids):
        raise RuntimeError("Final-choice prompt is too short to split the anchor query")
    prefix_ids = [ids[:-1] for ids in full_ids]
    query_ids = torch.stack([ids[-1] for ids in full_ids]).reshape(-1, 1)
    prefix_maps = [mapping[:-1] for mapping in mappings]
    max_length = max(int(ids.numel()) for ids in prefix_ids)
    batch = len(indices)
    padded_ids = torch.full((batch, max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long)
    padded_map = torch.full((batch, max_length), -1, dtype=torch.long)
    for row, (ids, mapping) in enumerate(zip(prefix_ids, prefix_maps, strict=True)):
        length = int(ids.numel())
        padded_ids[row, max_length - length :] = ids
        attention_mask[row, max_length - length :] = 1
        padded_map[row, max_length - length :] = mapping
    # Explicit positions make a question invariant to the other sequence
    # lengths in its batch.  Llama otherwise counts left-padding slots when a
    # DynamicCache is used, changing logits as the batch composition changes.
    prefix_position_ids = attention_mask.cumsum(dim=1) - 1
    prefix_position_ids.masked_fill_(attention_mask == 0, 0)
    query_position_ids = attention_mask.sum(dim=1, keepdim=True)
    return {
        "prefix_ids": padded_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_document_ids": padded_map.to(device),
        "query_ids": query_ids.to(device),
        "prefix_position_ids": prefix_position_ids.to(device),
        "query_position_ids": query_position_ids.to(device),
    }


def final_choice_logits(
    model: Any,
    attention_name: str,
    batch: dict[str, torch.Tensor],
    document_bias: torch.Tensor,
    choice_token_ids: torch.Tensor,
    layer_start: int,
) -> torch.Tensor:
    """Frozen SDPA prefix plus differentiable q_len=1 document attention."""

    previous_attention = model.config._attn_implementation
    try:
        model.config._attn_implementation = "sdpa"
        # Only hidden states and KV cache are needed for the fixed prefix.
        # Calling the CausalLM wrapper here would also materialize an unused
        # [batch, sequence, vocabulary] logits tensor.
        base_model = getattr(model, "model", None)
        if base_model is None:
            raise TypeError("Expected a causal-LM wrapper with a .model decoder")
        with torch.no_grad():
            prefix = base_model(
                input_ids=batch["prefix_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["prefix_position_ids"],
                use_cache=True,
                return_dict=True,
            )
        model.config._attn_implementation = attention_name
        batch_size = int(batch["prefix_ids"].shape[0])
        full_mask = torch.cat(
            [
                batch["attention_mask"],
                torch.ones(
                    (batch_size, 1),
                    dtype=batch["attention_mask"].dtype,
                    device=batch["attention_mask"].device,
                ),
            ],
            dim=1,
        )
        full_mapping = torch.cat(
            [
                batch["token_document_ids"],
                torch.full(
                    (batch_size, 1),
                    -1,
                    dtype=torch.long,
                    device=batch["token_document_ids"].device,
                ),
            ],
            dim=1,
        )
        token_bias = document_bias_to_token_bias(document_bias, full_mapping)
        query_mask = torch.zeros_like(full_mask, dtype=torch.float32)
        query_mask[:, -1] = 1.0
        outputs = model(
            input_ids=batch["query_ids"],
            attention_mask=full_mask,
            position_ids=batch["query_position_ids"],
            past_key_values=prefix.past_key_values,
            use_cache=False,
            return_dict=True,
            semantic_token_bias=token_bias,
            semantic_query_mask=query_mask,
            semantic_layer_start=layer_start,
        )
        logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids)
        del outputs, prefix
        return logits
    finally:
        model.config._attn_implementation = previous_attention


def batch_controller_inputs(
    payload: dict[str, Any], indices: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    selected = torch.tensor(indices, dtype=torch.long)
    return {
        "features": payload["semantic_features"].index_select(0, selected).to(device=device, dtype=torch.float32),
        "margins": payload["semantic_margins"].index_select(0, selected).to(device=device, dtype=torch.float32),
        "targets": payload["semantic_targets"].index_select(0, selected).to(device=device, dtype=torch.long),
        "masks": payload["semantic_masks"].index_select(0, selected).to(device=device, dtype=torch.bool),
        "class_ids": payload["semantic_class_ids"].index_select(0, selected).to(device=device, dtype=torch.long),
        "gold": payload["gold_options"].index_select(0, selected).to(device=device, dtype=torch.long),
        "baseline": payload["baseline_options"].index_select(0, selected).to(device=device, dtype=torch.long),
        "no_rag_correct": payload["no_rag_correct"].index_select(0, selected).to(device=device, dtype=torch.bool),
    }


def new_metric_state() -> dict[str, Any]:
    return {
        "questions": 0,
        "correct": 0,
        "baseline_correct": 0,
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "natural_ce_sum": 0.0,
        "loss_sum": 0.0,
        "group_total": defaultdict(int),
        "group_correct": defaultdict(int),
        "class_bias_sum": defaultdict(float),
        "class_gate_sum": defaultdict(float),
        "class_count": defaultdict(int),
    }


def update_metrics(
    state: dict[str, Any],
    logits: torch.Tensor,
    values: dict[str, torch.Tensor],
    document_bias: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(values["gold"])
    baseline_correct = values["baseline"].eq(values["gold"])
    count = int(predictions.numel())
    state["questions"] += count
    state["correct"] += int(correct.sum().item())
    state["baseline_correct"] += int(baseline_correct.sum().item())
    state["wrong_to_correct"] += int((~baseline_correct & correct).sum().item())
    state["correct_to_wrong"] += int((baseline_correct & ~correct).sum().item())
    state["natural_ce_sum"] += float(F.cross_entropy(logits, values["gold"], reduction="sum").item())
    state["loss_sum"] += float(loss.detach().item()) * count
    for flag in (False, True):
        members = values["no_rag_correct"].eq(flag)
        name = "no_rag_correct" if flag else "no_rag_wrong"
        state["group_total"][name] += int(members.sum().item())
        state["group_correct"][name] += int((correct & members).sum().item())
    for class_id, class_name in SEMANTIC_CLASS_NAMES.items():
        members = values["class_ids"].eq(class_id)
        if bool(members.any()):
            state["class_bias_sum"][class_name] += float(document_bias[members].detach().sum().item())
            state["class_gate_sum"][class_name] += float(document_bias[members].detach().exp().sum().item())
            state["class_count"][class_name] += int(members.sum().item())


def finalize_metrics(state: dict[str, Any]) -> dict[str, Any]:
    total = max(1, int(state["questions"]))
    group_accuracy = {
        name: state["group_correct"][name] / max(1, state["group_total"][name])
        for name in state["group_total"]
    }
    class_attention = {
        name: {
            "documents": count,
            "mean_bias": state["class_bias_sum"][name] / count,
            "mean_keep_gate": state["class_gate_sum"][name] / count,
        }
        for name, count in state["class_count"].items()
        if count
    }
    return {
        "questions": state["questions"],
        "correct": state["correct"],
        "accuracy": state["correct"] / total,
        "cached_unfiltered_reference_correct": state["baseline_correct"],
        "cached_unfiltered_reference_accuracy": state["baseline_correct"] / total,
        "delta_vs_cached_unfiltered_reference": (
            state["correct"] - state["baseline_correct"]
        )
        / total,
        "cached_reference_wrong_to_correct": state["wrong_to_correct"],
        "cached_reference_correct_to_wrong": state["correct_to_wrong"],
        "cached_reference_net_answer_gain": (
            state["wrong_to_correct"] - state["correct_to_wrong"]
        ),
        "natural_answer_ce": state["natural_ce_sum"] / total,
        "mean_total_loss": state["loss_sum"] / total,
        "no_rag_group_accuracy": group_accuracy,
        "semantic_class_attention": class_attention,
    }


def controller_loss(
    logits: torch.Tensor,
    values: dict[str, torch.Tensor],
    controller_output: Any,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    answer = group_robust_answer_loss(
        logits,
        values["gold"],
        values["no_rag_correct"],
        balance_strength=args.no_rag_group_balance,
    )
    order = semantic_ordering_hinge_loss(
        controller_output.document_bias,
        values["targets"],
        document_mask=values["masks"],
        margin=args.ordering_margin,
    )
    # Mixed semantic annotations are excluded only from the weak ordering
    # prior.  Every real Top-8 document remains under the residual anchor.
    anchor = residual_anchor_loss(controller_output.residual)
    total = answer + args.ordering_loss_weight * order + args.anchor_loss_weight * anchor
    return total, {
        "answer": float(answer.detach().item()),
        "ordering": float(order.detach().item()),
        "anchor": float(anchor.detach().item()),
    }


def evaluate_split(
    paths: list[Path],
    split: str,
    feature_fingerprint: str,
    feature_hidden_size: int,
    controller: SemanticResidualAttentionController,
    model: Any,
    tokenizer: Any,
    attention_name: str,
    choice_token_ids: torch.Tensor,
    args: argparse.Namespace,
    progress: PipelineProgress,
    stage_name: str,
) -> dict[str, Any]:
    total = count_questions(paths)
    progress.set_stage(stage_name, total=total)
    state = new_metric_state()
    controller.eval()
    with torch.no_grad():
        for shard_number, path in enumerate(paths, start=1):
            payload = load_feature_shard(
                path,
                dataset=args.dataset,
                split=split,
                fingerprint=feature_fingerprint,
                hidden_size=feature_hidden_size,
            )
            for indices in shuffled_batches(len(payload["sample_ids"]), args.question_batch_size, args.seed):
                values = batch_controller_inputs(payload, indices, torch.device(args.device))
                output = controller(values["features"], values["margins"])
                prefix = collate_prefix_batch(
                    payload,
                    indices,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=torch.device(args.device),
                )
                logits = final_choice_logits(
                    model,
                    attention_name,
                    prefix,
                    output.document_bias,
                    choice_token_ids,
                    args.semantic_layer_start,
                )
                loss, _ = controller_loss(logits, values, output, args)
                update_metrics(state, logits, values, output.document_bias, loss)
                progress.update(len(indices))
                progress.set_detail(f"shard={shard_number}/{len(paths)}")
    return finalize_metrics(state)


def save_checkpoint(
    path: Path,
    controller: SemanticResidualAttentionController,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    next_shard_position: int,
    history: list[dict[str, Any]],
    best_metric: tuple[float, float],
    bad_epochs: int,
    global_step: int,
    run_contract: dict[str, Any],
    train_state: dict[str, Any] | None,
) -> None:
    atomic_torch_save(
        path,
        {
            "run_version": RUN_VERSION,
            "run_contract": run_contract,
            "controller": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "next_shard_position": next_shard_position,
            "history": history,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_state": train_state,
        },
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move a CPU-loaded AdamW state to the controller device."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def normalize_accumulated_gradients(
    module: torch.nn.Module,
    microbatches: int,
) -> None:
    """Average gradients over the actual (possibly partial) accumulation."""

    if microbatches <= 0:
        raise ValueError("microbatches must be positive")
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(microbatches))


def shorten_progress(progress: PipelineProgress, new_total: int) -> None:
    progress.overall_total = int(new_total)
    if progress._pbar is not None:  # The helper intentionally owns the one live line.
        progress._pbar.total = int(new_total)
        progress._pbar.refresh()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.epochs <= 0 or args.question_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Epoch and batch parameters must be positive")
    if args.patience < 0 or args.max_grad_norm <= 0:
        raise ValueError("patience must be non-negative and max-grad-norm must be positive")
    if not 0.0 <= args.no_rag_group_balance <= 1.0:
        raise ValueError("no-rag-group-balance must be in [0, 1]")
    if not 0.0 < args.boundary_epsilon < 0.5:
        raise ValueError("boundary-epsilon must be in (0, 0.5)")
    if args.max_suppression_factor <= 1.0:
        raise ValueError("max-suppression-factor must be greater than 1")
    if args.dtype != "bfloat16":
        raise ValueError("Controller training currently requires bfloat16 to avoid FP16 gradient underflow")
    if not 0 <= args.semantic_layer_start < 32:
        raise ValueError("semantic_layer_start must be in [0, 31]")
    if not args.feature_dir.exists() or not args.llm_model.exists():
        raise FileNotFoundError(args.feature_dir if not args.feature_dir.exists() else args.llm_model)
    manifest_path = args.feature_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("dataset") != args.dataset:
        raise ValueError("Prepared feature dataset does not match --dataset")
    hidden_size = int(feature_manifest["feature_hidden_size"])
    current_llm_identity = model_bundle_identity(args.llm_model)
    if feature_manifest.get("llm_model_bundle") != current_llm_identity:
        raise RuntimeError(
            "Prepared prompts were fingerprinted for a different Llama checkpoint/tokenizer"
        )
    split_paths = {
        split: list_feature_shards(args.feature_dir, split, feature_manifest)
        for split in ("train", "val", "test")
    }
    split_counts = {split: count_questions(paths) for split, paths in split_paths.items()}
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("--no-resume requires an empty or new training output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "feature_contract_fingerprint": feature_manifest["contract_fingerprint"],
        "feature_hidden_size": hidden_size,
        "split_questions": split_counts,
        "llm_model": str(args.llm_model.resolve()),
        "llm_model_bundle": current_llm_identity,
        "epochs": args.epochs,
        "patience": args.patience,
        "question_batch_size": args.question_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "controller_hidden_size": args.controller_hidden_size,
        "controller_dropout": args.controller_dropout,
        "semantic_temperature": args.semantic_temperature,
        "prior_strength": args.prior_strength,
        "max_suppression_factor": args.max_suppression_factor,
        "boundary_epsilon": args.boundary_epsilon,
        "semantic_layer_start": args.semantic_layer_start,
        "ordering_margin": args.ordering_margin,
        "ordering_loss_weight": args.ordering_loss_weight,
        "anchor_loss_weight": args.anchor_loss_weight,
        "no_rag_group_balance": args.no_rag_group_balance,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    run_contract_path = args.output_dir / "training_contract.json"
    if run_contract_path.is_file() and args.resume:
        if json.loads(run_contract_path.read_text(encoding="utf-8")) != run_contract:
            raise RuntimeError("Training resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(run_contract_path, run_contract)
    logging.info("Semantic attention training plan: %s", json.dumps(split_counts))
    summary_path = args.output_dir / "summary.json"
    if args.resume and summary_path.is_file() and (args.output_dir / "final_controller.pt").is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        logging.info(
            "Training already complete: best_epoch=%s test_accuracy=%s output=%s",
            completed.get("best_epoch"),
            completed.get("test", {}).get("accuracy"),
            args.output_dir,
        )
        return
    if args.plan_only:
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    controller = SemanticResidualAttentionController(
        input_dim=hidden_size,
        hidden_dim=args.controller_hidden_size,
        dropout=args.controller_dropout,
        temperature=args.semantic_temperature,
        max_suppression_bias=math.log(args.max_suppression_factor),
        prior_strength=args.prior_strength,
        boundary_epsilon=args.boundary_epsilon,
    ).to(device)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_path = args.output_dir / "latest_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    start_shard_position = 0
    best_metric = (-1.0, float("-inf"))
    bad_epochs = 0
    global_step = 0
    resumed_train_state: dict[str, Any] | None = None
    resumed_cpu_rng_state: torch.Tensor | None = None
    resumed_cuda_rng_state: list[torch.Tensor] | None = None
    if checkpoint_path.is_file() and args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_contract") != run_contract:
            raise RuntimeError("Checkpoint contract mismatch")
        controller.load_state_dict(checkpoint["controller"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch"])
        start_shard_position = int(checkpoint["next_shard_position"])
        history = list(checkpoint.get("history") or [])
        best_metric = tuple(checkpoint.get("best_metric") or best_metric)  # type: ignore[assignment]
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        global_step = int(checkpoint.get("global_step", 0))
        resumed_cpu_rng_state = checkpoint["torch_rng_state"]
        resumed_cuda_rng_state = checkpoint.get("cuda_rng_state_all")
        resumed_train_state = checkpoint.get("train_state")
        logging.info(
            "Resuming controller: epoch=%d next_shard=%d global_step=%d",
            start_epoch + 1,
            start_shard_position,
            global_step,
        )

    attention_name = register_semantic_attention()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    choice_ids: list[int] = []
    for label in CHOICES:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice {label} is not one token after the fixed '(' anchor: {ids}")
        choice_ids.append(int(ids[0]))
    choice_token_ids = torch.tensor(choice_ids, dtype=torch.long, device=device)
    logging.info("Loading frozen target Llama for final-choice attention training: %s", args.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module_for_controller_training(model)
    # Model construction can consume RNG state.  Restore checkpoint RNG only
    # after every module has been materialized so resumed dropout is exact.
    if resumed_cpu_rng_state is not None:
        torch.set_rng_state(resumed_cpu_rng_state)
    if resumed_cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(resumed_cuda_rng_state)

    train_per_epoch = split_counts["train"]
    val_per_epoch = split_counts["val"]
    overall_total = args.epochs * (train_per_epoch + val_per_epoch) + split_counts["test"]
    train_shards = split_paths["train"]
    completed_in_epoch = 0
    if start_shard_position:
        order = list(range(len(train_shards)))
        random.Random(args.seed + start_epoch).shuffle(order)
        completed_in_epoch = sum(
            int(json.loads(train_shards[index].with_name("COMPLETE.json").read_text())["question_count"])
            for index in order[:start_shard_position]
        )
    overall_initial = start_epoch * (train_per_epoch + val_per_epoch) + completed_in_epoch
    progress = PipelineProgress(
        overall_total=overall_total,
        overall_initial=overall_initial,
        desc=f"TrainSemanticGate:{args.dataset}",
    )
    stopped_epoch = args.epochs
    try:
        for epoch in range(start_epoch, args.epochs):
            controller.train()
            shard_order = list(range(len(train_shards)))
            random.Random(args.seed + epoch).shuffle(shard_order)
            shard_cursor = start_shard_position if epoch == start_epoch else 0
            initial_questions = sum(
                int(json.loads(train_shards[index].with_name("COMPLETE.json").read_text())["question_count"])
                for index in shard_order[:shard_cursor]
            )
            progress.set_stage(
                f"1/3 train epoch {epoch + 1}/{args.epochs}",
                total=train_per_epoch,
                initial=initial_questions,
            )
            if epoch == start_epoch and shard_cursor > 0:
                if resumed_train_state is None:
                    raise RuntimeError(
                        "Checkpoint resumes mid-epoch but lacks accumulated train metrics"
                    )
                train_state = resumed_train_state
            else:
                train_state = new_metric_state()
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
            for order_position in range(shard_cursor, len(shard_order)):
                shard_index = shard_order[order_position]
                payload = load_feature_shard(
                    train_shards[shard_index],
                    dataset=args.dataset,
                    split="train",
                    fingerprint=feature_manifest["contract_fingerprint"],
                    hidden_size=hidden_size,
                )
                batches = list(
                    shuffled_batches(
                        len(payload["sample_ids"]),
                        args.question_batch_size,
                        args.seed + epoch * 1_000_003 + shard_index,
                    )
                )
                for batch_number, indices in enumerate(batches, start=1):
                    values = batch_controller_inputs(payload, indices, device)
                    output = controller(values["features"], values["margins"])
                    prefix = collate_prefix_batch(
                        payload,
                        indices,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=device,
                    )
                    logits = final_choice_logits(
                        model,
                        attention_name,
                        prefix,
                        output.document_bias,
                        choice_token_ids,
                        args.semantic_layer_start,
                    )
                    loss, loss_parts = controller_loss(logits, values, output, args)
                    loss.backward()
                    accumulated_microbatches += 1
                    should_step = (
                        batch_number % args.gradient_accumulation_steps == 0
                        or batch_number == len(batches)
                    )
                    if should_step:
                        normalize_accumulated_gradients(
                            controller,
                            accumulated_microbatches,
                        )
                        torch.nn.utils.clip_grad_norm_(controller.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_microbatches = 0
                        global_step += 1
                    update_metrics(train_state, logits, values, output.document_bias, loss)
                    progress.update(len(indices))
                    progress.set_detail(
                        f"shard={order_position + 1}/{len(shard_order)} "
                        f"answer={loss_parts['answer']:.3f} order={loss_parts['ordering']:.3f}"
                    )
                save_checkpoint(
                    checkpoint_path,
                    controller,
                    optimizer,
                    epoch=epoch,
                    next_shard_position=order_position + 1,
                    history=history,
                    best_metric=best_metric,
                    bad_epochs=bad_epochs,
                    global_step=global_step,
                    run_contract=run_contract,
                    train_state=train_state,
                )
            train_metrics = finalize_metrics(train_state)
            val_metrics = evaluate_split(
                split_paths["val"],
                "val",
                feature_manifest["contract_fingerprint"],
                hidden_size,
                controller,
                model,
                tokenizer,
                attention_name,
                choice_token_ids,
                args,
                progress,
                f"2/3 validation epoch {epoch + 1}/{args.epochs}",
            )
            epoch_row = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": val_metrics,
            }
            history.append(epoch_row)
            atomic_write_json(args.output_dir / "history.json", history)
            metric = (float(val_metrics["accuracy"]), -float(val_metrics["natural_answer_ce"]))
            if metric > best_metric:
                best_metric = metric
                bad_epochs = 0
                atomic_torch_save(
                    args.output_dir / "best_controller.pt",
                    {
                        "run_version": RUN_VERSION,
                        "run_contract": run_contract,
                        "epoch": epoch + 1,
                        "controller": controller.state_dict(),
                        "validation": val_metrics,
                    },
                )
            else:
                bad_epochs += 1
            logging.info(
                "Epoch %d: train_acc=%.4f val_acc=%.4f cached_reference=%.4f delta=%+.4f "
                "W->C=%d C->W=%d bad_epochs=%d",
                epoch + 1,
                train_metrics["accuracy"],
                val_metrics["accuracy"],
                val_metrics["cached_unfiltered_reference_accuracy"],
                val_metrics["delta_vs_cached_unfiltered_reference"],
                val_metrics["cached_reference_wrong_to_correct"],
                val_metrics["cached_reference_correct_to_wrong"],
                bad_epochs,
            )
            save_checkpoint(
                checkpoint_path,
                controller,
                optimizer,
                epoch=epoch + 1,
                next_shard_position=0,
                history=history,
                best_metric=best_metric,
                bad_epochs=bad_epochs,
                global_step=global_step,
                run_contract=run_contract,
                train_state=None,
            )
            start_shard_position = 0
            resumed_train_state = None
            if args.patience > 0 and bad_epochs >= args.patience:
                stopped_epoch = epoch + 1
                shorten_progress(
                    progress,
                    stopped_epoch * (train_per_epoch + val_per_epoch) + split_counts["test"],
                )
                logging.info("Early stopping after epoch %d", stopped_epoch)
                break

        best = torch.load(args.output_dir / "best_controller.pt", map_location="cpu", weights_only=False)
        controller.load_state_dict(best["controller"])
        test_metrics = evaluate_split(
            split_paths["test"],
            "test",
            feature_manifest["contract_fingerprint"],
            hidden_size,
            controller,
            model,
            tokenizer,
            attention_name,
            choice_token_ids,
            args,
            progress,
            "3/3 final internal test with best validation checkpoint",
        )
        summary = {
            "run_version": RUN_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "cached_reference_definition": (
                "unbiased Top-8 rationale and constrained choice generated by vLLM; "
                "reported deltas are diagnostic because controller logits use HF replay"
            ),
            "best_epoch": int(best["epoch"]),
            "stopped_epoch": stopped_epoch,
            "split_questions": split_counts,
            "best_validation": best["validation"],
            "test": test_metrics,
            "history": history,
        }
        atomic_write_json(summary_path, summary)
        atomic_torch_save(
            args.output_dir / "final_controller.pt",
            {
                "run_version": RUN_VERSION,
                "run_contract": run_contract,
                "best_epoch": int(best["epoch"]),
                "controller": controller.state_dict(),
                "test": test_metrics,
            },
        )
        logging.info(
            "Training complete: best_epoch=%d test_acc=%.4f cached_reference=%.4f "
            "delta=%+.4f output=%s",
            best["epoch"],
            test_metrics["accuracy"],
            test_metrics["cached_unfiltered_reference_accuracy"],
            test_metrics["delta_vs_cached_unfiltered_reference"],
            args.output_dir,
        )
    finally:
        progress.close()


if __name__ == "__main__":
    main()

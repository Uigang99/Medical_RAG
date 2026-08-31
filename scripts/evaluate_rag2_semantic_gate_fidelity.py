#!/usr/bin/env python3
"""Audit whether learned document keep-gates track causal document influence.

For each prepared Top-8 question, the learned semantic-attention controller
produces one keep gate per document.  The gates are normalized across the
eight documents to obtain a descriptive 100% allocation.  The audit then
physically removes each document's mapped prompt tokens, replays the same
cached rationale path, and measures the Jensen-Shannon divergence of the final
four-choice distribution.  Normalized LOO divergences form the causal
reference allocation.

This is deliberately a fixed-rationale diagnostic, not a claim about fully
regenerated counterfactual reasoning.  It tests whether the inexpensive gate
allocation is faithful enough to use as an analysis metric.  Every long stage
is resumable and reports overall/stage progress and ETA.
"""

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.generation.learned_semantic_attention import (  # noqa: E402
    SemanticResidualAttentionController,
    document_bias_to_token_bias,
)
from medrag.generation.semantic_attention import (  # noqa: E402
    DocumentAttentionCollector,
    register_semantic_attention,
)
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from scripts.train_rag2_semantic_attention_controller import (  # noqa: E402
    SEMANTIC_CLASS_NAMES,
    list_feature_shards,
    load_feature_shard,
)


RUN_VERSION = "rag2_semantic_gate_attention_fidelity_fixed_rationale_loo_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=256, help="0 uses the complete split")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--minimum-total-jsd",
        type=float,
        default=1e-6,
        help="Questions below this summed LOO signal are excluded from normalized-share fidelity",
    )
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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}:{line_number}")
            rows[sample_id] = row
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rankdata(values: list[float]) -> list[float]:
    """Return average ranks for ties, using zero-based ascending ranks."""

    order = sorted(range(len(values)), key=lambda index: values[index])
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


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
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


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    return pearson_correlation(rankdata(left), rankdata(right))


def normalize_positive(values: list[float], minimum_total: float = 0.0) -> list[float] | None:
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("normalize_positive requires finite non-negative values")
    total = sum(values)
    if total <= minimum_total:
        return None
    return [value / total for value in values]


def jensen_shannon_divergence(reference: torch.Tensor, alternative: torch.Tensor) -> float:
    """Return base-2 JSD in [0, 1] for two one-dimensional distributions."""

    reference = reference.double().clamp_min(1e-30)
    alternative = alternative.double().clamp_min(1e-30)
    midpoint = 0.5 * (reference + alternative)
    divergence = 0.5 * (
        torch.sum(reference * torch.log2(reference / midpoint))
        + torch.sum(alternative * torch.log2(alternative / midpoint))
    )
    return float(divergence.clamp(min=0.0, max=1.0).item())


def gold_margin(logits: torch.Tensor, gold_index: int) -> float:
    if logits.ndim != 1 or not 0 <= gold_index < int(logits.numel()):
        raise ValueError("Invalid logits or gold index")
    wrong = torch.cat([logits[:gold_index], logits[gold_index + 1 :]])
    return float((logits[gold_index] - wrong.max()).item())


def build_physical_loo_batch(
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    assistant_query_start: int,
    *,
    pad_token_id: int,
    attention_scope: str,
    document_count: int = 8,
) -> dict[str, torch.Tensor]:
    """Create full plus eight physical-token-removal prompt variants."""

    ids = input_ids.long().cpu()
    mapping = token_document_ids.long().cpu()
    if ids.ndim != 1 or mapping.shape != ids.shape:
        raise ValueError("input_ids and token_document_ids must be aligned vectors")
    if attention_scope not in {"final_choice", "rationale_wide"}:
        raise ValueError(f"Unsupported attention scope: {attention_scope}")
    if not 0 <= assistant_query_start < int(ids.numel()):
        raise ValueError("assistant_query_start is out of range")

    original_query_mask = torch.zeros(ids.numel(), dtype=torch.float32)
    if attention_scope == "rationale_wide":
        original_query_mask[assistant_query_start:] = 1.0
    else:
        original_query_mask[-1] = 1.0

    variants: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    variants.append((ids, mapping, original_query_mask))
    for document_index in range(document_count):
        keep = mapping.ne(document_index)
        removed_tokens = int((~keep).sum().item())
        if removed_tokens <= 0:
            raise RuntimeError(f"Document slot {document_index} has no mapped prompt tokens")
        variants.append((ids[keep], mapping[keep], original_query_mask[keep]))

    maximum = max(int(variant_ids.numel()) for variant_ids, _, _ in variants)
    batch_size = len(variants)
    padded_ids = torch.full((batch_size, maximum), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, maximum), dtype=torch.long)
    padded_mapping = torch.full((batch_size, maximum), -1, dtype=torch.long)
    query_mask = torch.zeros((batch_size, maximum), dtype=torch.float32)
    lengths: list[int] = []
    for row, (variant_ids, variant_mapping, variant_query) in enumerate(variants):
        length = int(variant_ids.numel())
        left = maximum - length
        padded_ids[row, left:] = variant_ids
        attention_mask[row, left:] = 1
        padded_mapping[row, left:] = variant_mapping
        query_mask[row, left:] = variant_query
        lengths.append(length)
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_document_ids": padded_mapping,
        "semantic_query_mask": query_mask,
        "variant_lengths": torch.tensor(lengths, dtype=torch.long),
    }


def choice_logits_for_loo_batch(
    model: Any,
    batch: dict[str, torch.Tensor],
    document_bias: torch.Tensor,
    choice_token_ids: torch.Tensor,
    semantic_layer_start: int,
    device: torch.device,
) -> tuple[torch.Tensor, DocumentAttentionCollector]:
    variants = int(batch["input_ids"].shape[0])
    repeated_bias = document_bias.to(device=device, dtype=torch.float32).expand(variants, -1)
    mapping = batch["token_document_ids"].to(device)
    token_bias = document_bias_to_token_bias(repeated_bias, mapping)
    collector = DocumentAttentionCollector(document_count=int(document_bias.shape[1]))
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            semantic_token_bias=token_bias,
            semantic_query_mask=batch["semantic_query_mask"].to(device),
            semantic_layer_start=semantic_layer_start,
            semantic_token_document_ids=mapping,
            semantic_attention_collector=collector,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids)
    del outputs
    return logits, collector


def controller_from_checkpoint(
    checkpoint: dict[str, Any],
    feature_hidden_size: int,
    device: torch.device,
) -> tuple[SemanticResidualAttentionController, dict[str, Any]]:
    contract = dict(checkpoint.get("run_contract") or {})
    required = (
        "controller_hidden_size",
        "controller_dropout",
        "semantic_temperature",
        "max_suppression_factor",
        "prior_strength",
        "boundary_epsilon",
        "semantic_layer_start",
    )
    missing = [name for name in required if name not in contract]
    if missing:
        raise ValueError(f"Controller checkpoint misses contract fields: {missing}")
    controller = SemanticResidualAttentionController(
        input_dim=feature_hidden_size,
        hidden_dim=int(contract["controller_hidden_size"]),
        dropout=float(contract["controller_dropout"]),
        temperature=float(contract["semantic_temperature"]),
        max_suppression_bias=math.log(float(contract["max_suppression_factor"])),
        prior_strength=float(contract["prior_strength"]),
        boundary_epsilon=float(contract["boundary_epsilon"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    return controller, contract


def select_sample_ids(
    shard_paths: list[Path],
    *,
    dataset: str,
    split: str,
    fingerprint: str,
    hidden_size: int,
    max_samples: int,
    seed: int,
) -> list[str]:
    sample_ids: list[str] = []
    for path in shard_paths:
        payload = load_feature_shard(
            path,
            dataset=dataset,
            split=split,
            fingerprint=fingerprint,
            hidden_size=hidden_size,
        )
        sample_ids.extend(str(value) for value in payload["sample_ids"])
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Prepared feature split contains duplicate sample IDs")
    if max_samples > 0 and max_samples < len(sample_ids):
        sample_ids = random.Random(seed).sample(sample_ids, max_samples)
    return sorted(sample_ids)


def evaluate_one(
    payload: dict[str, Any],
    row_index: int,
    controller: SemanticResidualAttentionController,
    model: Any,
    tokenizer: Any,
    choice_token_ids: torch.Tensor,
    contract: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    features = payload["semantic_features"][row_index : row_index + 1].to(
        device=device, dtype=torch.float32
    )
    margins = payload["semantic_margins"][row_index : row_index + 1].to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        output = controller(features, margins)
    keep_gates = output.document_bias[0].exp().float().cpu()
    predicted_share = keep_gates / keep_gates.sum().clamp_min(1e-30)

    attention_scope = str(contract.get("attention_scope") or "final_choice")
    batch = build_physical_loo_batch(
        payload["input_ids"][row_index],
        payload["token_document_ids"][row_index],
        int(payload["assistant_query_starts"][row_index]),
        pad_token_id=int(tokenizer.pad_token_id),
        attention_scope=attention_scope,
    )
    logits, attention_collector = choice_logits_for_loo_batch(
        model,
        batch,
        output.document_bias,
        choice_token_ids,
        int(contract["semantic_layer_start"]),
        device,
    ).cpu()
    attention_all = attention_collector.summarize()
    attention_controlled = attention_collector.summarize(
        layer_start=int(contract["semantic_layer_start"])
    )
    attention_share_all = [
        float(value) for value in attention_all["document_share"][0].tolist()
    ]
    attention_share_controlled = [
        float(value) for value in attention_controlled["document_share"][0].tolist()
    ]
    probabilities = torch.softmax(logits, dim=-1)
    full_probability = probabilities[0]
    loo_probabilities = probabilities[1:]
    loo_jsd = [
        jensen_shannon_divergence(full_probability, loo_probabilities[index])
        for index in range(8)
    ]
    causal_share = normalize_positive(loo_jsd, args.minimum_total_jsd)
    gate_share = [float(value) for value in predicted_share.tolist()]
    gold_index = int(payload["gold_options"][row_index])
    full_margin = gold_margin(logits[0], gold_index)
    margin_contribution = [
        full_margin - gold_margin(logits[index + 1], gold_index)
        for index in range(8)
    ]
    per_question_pearson = (
        pearson_correlation(gate_share, causal_share) if causal_share is not None else None
    )
    per_question_spearman = (
        spearman_correlation(gate_share, causal_share) if causal_share is not None else None
    )
    attention_all_pearson = (
        pearson_correlation(attention_share_all, causal_share)
        if causal_share is not None
        else None
    )
    attention_all_spearman = (
        spearman_correlation(attention_share_all, causal_share)
        if causal_share is not None
        else None
    )
    attention_controlled_pearson = (
        pearson_correlation(attention_share_controlled, causal_share)
        if causal_share is not None
        else None
    )
    attention_controlled_spearman = (
        spearman_correlation(attention_share_controlled, causal_share)
        if causal_share is not None
        else None
    )
    top_gate = max(range(8), key=gate_share.__getitem__)
    top_attention_all = max(range(8), key=attention_share_all.__getitem__)
    top_attention_controlled = max(range(8), key=attention_share_controlled.__getitem__)
    top_causal = max(range(8), key=loo_jsd.__getitem__)
    class_ids = [int(value) for value in payload["semantic_class_ids"][row_index].tolist()]
    class_names = [SEMANTIC_CLASS_NAMES.get(value, "indeterminate_or_mixed") for value in class_ids]
    return {
        "run_version": RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": str(payload["sample_ids"][row_index]),
        "dataset": args.dataset,
        "split": args.split,
        "pair_ids": [str(value) for value in payload["pair_ids"][row_index]],
        "semantic_class_ids": class_ids,
        "semantic_classes": class_names,
        "keep_gates": [float(value) for value in keep_gates.tolist()],
        "predicted_document_share": gate_share,
        "attention_document_share_all_layers": attention_share_all,
        "attention_document_share_controlled_layers": attention_share_controlled,
        "absolute_document_attention_fraction_all_layers": float(
            attention_all["document_attention_fraction"][0].item()
        ),
        "absolute_document_attention_fraction_controlled_layers": float(
            attention_controlled["document_attention_fraction"][0].item()
        ),
        "attention_layers_all": [int(value) for value in attention_all["layers"].tolist()],
        "attention_layers_controlled": [
            int(value) for value in attention_controlled["layers"].tolist()
        ],
        "loo_jsd": loo_jsd,
        "total_loo_jsd": sum(loo_jsd),
        "causal_document_share": causal_share,
        "gold_margin_contribution": margin_contribution,
        "full_choice_probabilities": [float(value) for value in full_probability.tolist()],
        "loo_choice_probabilities": [
            [float(value) for value in row.tolist()] for row in loo_probabilities
        ],
        "gold_option_index": gold_index,
        "full_prediction_index": int(logits[0].argmax().item()),
        "per_question_pearson": per_question_pearson,
        "per_question_spearman": per_question_spearman,
        "attention_all_per_question_pearson": attention_all_pearson,
        "attention_all_per_question_spearman": attention_all_spearman,
        "attention_controlled_per_question_pearson": attention_controlled_pearson,
        "attention_controlled_per_question_spearman": attention_controlled_spearman,
        "top_gate_document": top_gate,
        "top_attention_all_document": top_attention_all,
        "top_attention_controlled_document": top_attention_controlled,
        "top_causal_document": top_causal,
        "top1_agreement": top_gate == top_causal,
        "attention_all_top1_agreement": top_attention_all == top_causal,
        "attention_controlled_top1_agreement": top_attention_controlled == top_causal,
        "attention_scope": attention_scope,
        "fixed_rationale_diagnostic": True,
    }


def finite_values(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    minimum_total_jsd: float,
    progress: PipelineProgress,
) -> dict[str, Any]:
    progress.set_stage("2/2 aggregate fidelity and semantic leakage", total=len(rows))
    valid_rows: list[dict[str, Any]] = []
    flat_gate: list[float] = []
    flat_attention_all: list[float] = []
    flat_attention_controlled: list[float] = []
    flat_causal: list[float] = []
    class_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "documents": 0,
            "keep_gate": [],
            "predicted_share": [],
            "attention_share_all": [],
            "attention_share_controlled": [],
            "loo_jsd": [],
            "causal_share": [],
            "gold_margin_contribution": [],
        }
    )
    support_gate_mass: list[float] = []
    support_causal_mass: list[float] = []
    nonsupport_gate_mass: list[float] = []
    nonsupport_causal_mass: list[float] = []
    support_attention_all_mass: list[float] = []
    nonsupport_attention_all_mass: list[float] = []
    support_attention_controlled_mass: list[float] = []
    nonsupport_attention_controlled_mass: list[float] = []
    for row in rows:
        causal = row.get("causal_document_share")
        if causal is not None:
            valid_rows.append(row)
            flat_gate.extend(float(value) for value in row["predicted_document_share"])
            flat_attention_all.extend(
                float(value) for value in row["attention_document_share_all_layers"]
            )
            flat_attention_controlled.extend(
                float(value)
                for value in row["attention_document_share_controlled_layers"]
            )
            flat_causal.extend(float(value) for value in causal)
        support_gate = 0.0
        support_causal = 0.0
        nonsupport_gate = 0.0
        nonsupport_causal = 0.0
        support_attention_all = 0.0
        nonsupport_attention_all = 0.0
        support_attention_controlled = 0.0
        nonsupport_attention_controlled = 0.0
        for index, class_name in enumerate(row["semantic_classes"]):
            state = class_state[class_name]
            state["documents"] += 1
            state["keep_gate"].append(float(row["keep_gates"][index]))
            state["predicted_share"].append(float(row["predicted_document_share"][index]))
            state["attention_share_all"].append(
                float(row["attention_document_share_all_layers"][index])
            )
            state["attention_share_controlled"].append(
                float(row["attention_document_share_controlled_layers"][index])
            )
            state["loo_jsd"].append(float(row["loo_jsd"][index]))
            state["gold_margin_contribution"].append(
                float(row["gold_margin_contribution"][index])
            )
            if causal is not None:
                state["causal_share"].append(float(causal[index]))
            if class_name in {"direct_support", "supporting_evidence"}:
                support_gate += float(row["predicted_document_share"][index])
                support_attention_all += float(
                    row["attention_document_share_all_layers"][index]
                )
                support_attention_controlled += float(
                    row["attention_document_share_controlled_layers"][index]
                )
                if causal is not None:
                    support_causal += float(causal[index])
            elif class_name in {"no_evidence", "misleading_evidence"}:
                nonsupport_gate += float(row["predicted_document_share"][index])
                nonsupport_attention_all += float(
                    row["attention_document_share_all_layers"][index]
                )
                nonsupport_attention_controlled += float(
                    row["attention_document_share_controlled_layers"][index]
                )
                if causal is not None:
                    nonsupport_causal += float(causal[index])
        support_gate_mass.append(support_gate)
        nonsupport_gate_mass.append(nonsupport_gate)
        support_attention_all_mass.append(support_attention_all)
        nonsupport_attention_all_mass.append(nonsupport_attention_all)
        support_attention_controlled_mass.append(support_attention_controlled)
        nonsupport_attention_controlled_mass.append(nonsupport_attention_controlled)
        if causal is not None:
            support_causal_mass.append(support_causal)
            nonsupport_causal_mass.append(nonsupport_causal)
        progress.update(1)

    correlations = finite_values(row.get("per_question_spearman") for row in valid_rows)
    pearsons = finite_values(row.get("per_question_pearson") for row in valid_rows)
    attention_all_correlations = finite_values(
        row.get("attention_all_per_question_spearman") for row in valid_rows
    )
    attention_all_pearsons = finite_values(
        row.get("attention_all_per_question_pearson") for row in valid_rows
    )
    attention_controlled_correlations = finite_values(
        row.get("attention_controlled_per_question_spearman") for row in valid_rows
    )
    attention_controlled_pearsons = finite_values(
        row.get("attention_controlled_per_question_pearson") for row in valid_rows
    )
    top1 = sum(bool(row["top1_agreement"]) for row in valid_rows) / max(1, len(valid_rows))
    attention_all_top1 = sum(
        bool(row["attention_all_top1_agreement"]) for row in valid_rows
    ) / max(1, len(valid_rows))
    attention_controlled_top1 = sum(
        bool(row["attention_controlled_top1_agreement"]) for row in valid_rows
    ) / max(1, len(valid_rows))
    def fidelity_block(
        predicted: list[float],
        correlations_for_method: list[float],
        pearsons_for_method: list[float],
        method_top1: float,
    ) -> dict[str, Any]:
        median = (
            statistics.median(correlations_for_method)
            if correlations_for_method
            else None
        )
        if median is not None and median >= 0.5 and method_top1 >= 0.5:
            verdict = "supported_for_descriptive_analysis"
        elif median is not None and median >= 0.3 and method_top1 >= 0.35:
            verdict = "exploratory_only"
        else:
            verdict = "insufficient_fidelity"
        mae = (
            statistics.fmean(
                abs(float(value) - float(causal))
                for value, causal in zip(predicted, flat_causal, strict=True)
            )
            if predicted
            else None
        )
        return {
            "mean_per_question_spearman": (
                statistics.fmean(correlations_for_method)
                if correlations_for_method
                else None
            ),
            "median_per_question_spearman": median,
            "mean_per_question_pearson": (
                statistics.fmean(pearsons_for_method) if pearsons_for_method else None
            ),
            "global_document_spearman": spearman_correlation(predicted, flat_causal),
            "global_document_pearson": pearson_correlation(predicted, flat_causal),
            "top1_document_agreement": method_top1,
            "document_share_mae": mae,
            "verdict": verdict,
        }

    semantic_classes: dict[str, Any] = {}
    for class_name, state in sorted(class_state.items()):
        semantic_classes[class_name] = {
            "documents": state["documents"],
            "mean_keep_gate": statistics.fmean(state["keep_gate"]),
            "mean_predicted_document_share": statistics.fmean(state["predicted_share"]),
            "mean_attention_document_share_all_layers": statistics.fmean(
                state["attention_share_all"]
            ),
            "mean_attention_document_share_controlled_layers": statistics.fmean(
                state["attention_share_controlled"]
            ),
            "mean_loo_jsd": statistics.fmean(state["loo_jsd"]),
            "mean_causal_document_share": (
                statistics.fmean(state["causal_share"]) if state["causal_share"] else None
            ),
            "mean_gold_margin_contribution": statistics.fmean(
                state["gold_margin_contribution"]
            ),
            "positive_gold_contribution_rate": sum(
                value > 0 for value in state["gold_margin_contribution"]
            )
            / max(1, len(state["gold_margin_contribution"])),
        }

    return {
        "run_version": RUN_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(rows),
        "questions_with_measurable_loo_signal": len(valid_rows),
        "measurable_loo_coverage": len(valid_rows) / max(1, len(rows)),
        "minimum_total_jsd": minimum_total_jsd,
        "fidelity": {
            "gate": fidelity_block(flat_gate, correlations, pearsons, top1),
            "attention_all_layers": fidelity_block(
                flat_attention_all,
                attention_all_correlations,
                attention_all_pearsons,
                attention_all_top1,
            ),
            "attention_controlled_layers": fidelity_block(
                flat_attention_controlled,
                attention_controlled_correlations,
                attention_controlled_pearsons,
                attention_controlled_top1,
            ),
        },
        "semantic_selectivity": {
            "mean_support_predicted_gate_mass": statistics.fmean(support_gate_mass),
            "mean_nonsupport_predicted_gate_mass": statistics.fmean(nonsupport_gate_mass),
            "mean_support_causal_mass": (
                statistics.fmean(support_causal_mass) if support_causal_mass else None
            ),
            "mean_nonsupport_causal_mass": (
                statistics.fmean(nonsupport_causal_mass) if nonsupport_causal_mass else None
            ),
            "mean_support_attention_mass_all_layers": statistics.fmean(
                support_attention_all_mass
            ),
            "mean_nonsupport_attention_mass_all_layers": statistics.fmean(
                nonsupport_attention_all_mass
            ),
            "mean_support_attention_mass_controlled_layers": statistics.fmean(
                support_attention_controlled_mass
            ),
            "mean_nonsupport_attention_mass_controlled_layers": statistics.fmean(
                nonsupport_attention_controlled_mass
            ),
        },
        "absolute_attention_allocation": {
            "mean_fraction_to_any_document_all_layers": statistics.fmean(
                float(row["absolute_document_attention_fraction_all_layers"])
                for row in rows
            ),
            "mean_fraction_to_any_document_controlled_layers": statistics.fmean(
                float(row["absolute_document_attention_fraction_controlled_layers"])
                for row in rows
            ),
        },
        "semantic_classes": semantic_classes,
        "interpretation": {
            "predicted_document_share": "doc-only normalization of exp(document attention bias); sums to 1",
            "causal_document_share": "normalized physical-token-removal JSD on fixed-rationale final-choice distribution",
            "attention_document_share": "assistant-query attention mass normalized across the eight mapped document spans",
            "limitation": "fixed cached rationale; this does not measure counterfactual rationale regeneration",
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    fidelity = summary["fidelity"]
    semantic = summary["semantic_selectivity"]
    lines = [
        "# Semantic gate fidelity audit",
        "",
        f"- Questions: {summary['questions']}",
        f"- Measurable LOO coverage: {100 * summary['measurable_loo_coverage']:.2f}%",
        "",
        "| Proxy | Median Spearman vs LOO | Top-1 agreement | Share MAE | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in fidelity.items():
        lines.append(
            f"| {name} | {row['median_per_question_spearman']} | "
            f"{100 * row['top1_document_agreement']:.2f}% | "
            f"{row['document_share_mae']} | {row['verdict']} |"
        )
    lines.extend(
        [
        "",
        "## Semantic allocation",
        "",
        f"- Predicted support gate mass: {100 * semantic['mean_support_predicted_gate_mass']:.2f}%",
        f"- Predicted no/misleading gate mass: {100 * semantic['mean_nonsupport_predicted_gate_mass']:.2f}%",
        f"- Causal support mass: {100 * (semantic['mean_support_causal_mass'] or 0.0):.2f}%",
        f"- Causal no/misleading mass: {100 * (semantic['mean_nonsupport_causal_mass'] or 0.0):.2f}%",
        "",
        "## Absolute attention allocation",
        "",
        f"- Any-document attention, all layers: {100 * summary['absolute_attention_allocation']['mean_fraction_to_any_document_all_layers']:.2f}%",
        f"- Any-document attention, controlled layers: {100 * summary['absolute_attention_allocation']['mean_fraction_to_any_document_controlled_layers']:.2f}%",
        "",
        "| Semantic class | N | Keep gate | Gate share | Attention share (all/controlled) | LOO JSD | Causal share | Gold-margin contribution |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for class_name, row in summary["semantic_classes"].items():
        causal = row["mean_causal_document_share"]
        lines.append(
            f"| {class_name} | {row['documents']} | {row['mean_keep_gate']:.4f} | "
            f"{100 * row['mean_predicted_document_share']:.2f}% | "
            f"{100 * row['mean_attention_document_share_all_layers']:.2f}% / "
            f"{100 * row['mean_attention_document_share_controlled_layers']:.2f}% | "
            f"{row['mean_loo_jsd']:.6f} | "
            f"{(f'{100 * causal:.2f}%' if causal is not None else 'N/A')} | "
            f"{row['mean_gold_margin_contribution']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    if args.minimum_total_jsd < 0:
        raise ValueError("minimum-total-jsd must be non-negative")
    for path in (args.feature_dir, args.controller_checkpoint, args.llm_model):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    manifest_path = args.feature_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("dataset") != args.dataset:
        raise ValueError("Prepared feature dataset does not match --dataset")
    hidden_size = int(feature_manifest["feature_hidden_size"])
    fingerprint = str(feature_manifest["contract_fingerprint"])
    shard_paths = list_feature_shards(args.feature_dir, args.split, feature_manifest)
    selected_ids = select_sample_ids(
        shard_paths,
        dataset=args.dataset,
        split=args.split,
        fingerprint=fingerprint,
        hidden_size=hidden_size,
        max_samples=args.max_samples,
        seed=args.sample_seed,
    )
    selected_set = set(selected_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "gate_loo_details.jsonl"
    summary_path = args.output_dir / "summary.json"
    checkpoint_hash = sha256_file(args.controller_checkpoint)
    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "feature_dir": str(args.feature_dir.resolve()),
        "feature_contract_fingerprint": fingerprint,
        "controller_checkpoint": str(args.controller_checkpoint.resolve()),
        "controller_checkpoint_sha256": checkpoint_hash,
        "llm_model": str(args.llm_model.resolve()),
        "sample_ids": selected_ids,
        "sample_seed": args.sample_seed,
        "minimum_total_jsd": args.minimum_total_jsd,
        "dtype": args.dtype,
    }
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError("Gate-fidelity resume contract mismatch; use a new output directory")
    else:
        if not args.resume and detail_path.exists():
            detail_path.unlink()
        atomic_write_json(contract_path, run_contract)

    existing = load_jsonl_by_id(detail_path) if args.resume else {}
    valid_existing = {sample_id: row for sample_id, row in existing.items() if sample_id in selected_set}
    remaining = selected_set - set(valid_existing)
    logging.info(
        "Gate-fidelity plan: dataset=%s split=%s selected=%d cached=%d remaining=%d interventions=%d",
        args.dataset,
        args.split,
        len(selected_ids),
        len(valid_existing),
        len(remaining),
        len(remaining) * 9,
    )
    if args.plan_only:
        return

    progress = PipelineProgress(
        overall_total=2 * len(selected_ids),
        overall_initial=len(valid_existing),
        desc=f"GateFidelity:{args.dataset}",
    )
    try:
        if remaining:
            checkpoint = torch.load(args.controller_checkpoint, map_location="cpu", weights_only=False)
            controller, controller_contract = controller_from_checkpoint(
                checkpoint, hidden_size, torch.device(args.device)
            )
            if controller_contract.get("feature_contract_fingerprint") != fingerprint:
                raise RuntimeError("Controller and prepared feature fingerprints differ")
            attention_scope = str(controller_contract.get("attention_scope") or "final_choice")
            if attention_scope == "rationale_wide" and not all(
                "assistant_query_starts"
                in load_feature_shard(
                    path,
                    dataset=args.dataset,
                    split=args.split,
                    fingerprint=fingerprint,
                    hidden_size=hidden_size,
                )
                for path in shard_paths
            ):
                raise RuntimeError("Rationale-wide audit requires assistant query starts")

            attention_name = register_semantic_attention()
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model, local_files_only=True, use_fast=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            choice_ids: list[int] = []
            for label in CHOICES:
                token_ids = tokenizer.encode(label, add_special_tokens=False)
                if len(token_ids) != 1:
                    raise RuntimeError(f"Choice {label} is not one token: {token_ids}")
                choice_ids.append(int(token_ids[0]))
            choice_token_ids = torch.tensor(choice_ids, dtype=torch.long, device=args.device)
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            logging.info(
                "Loading frozen target Llama for physical-token LOO: model=%s scope=%s",
                args.llm_model,
                attention_scope,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                local_files_only=True,
                dtype=dtype,
                attn_implementation=attention_name,
            ).to(torch.device(args.device))
            model.eval()
            progress.set_stage(
                "1/2 physical Top-8 leave-one-document-out replay",
                total=len(selected_ids),
                initial=len(valid_existing),
            )
            pending_rows: list[dict[str, Any]] = []
            for shard_number, path in enumerate(shard_paths, start=1):
                payload = load_feature_shard(
                    path,
                    dataset=args.dataset,
                    split=args.split,
                    fingerprint=fingerprint,
                    hidden_size=hidden_size,
                )
                for row_index, sample_id_value in enumerate(payload["sample_ids"]):
                    sample_id = str(sample_id_value)
                    if sample_id not in remaining:
                        continue
                    row = evaluate_one(
                        payload,
                        row_index,
                        controller,
                        model,
                        tokenizer,
                        choice_token_ids,
                        controller_contract,
                        args,
                    )
                    pending_rows.append(row)
                    valid_existing[sample_id] = row
                    progress.update(1)
                    progress.set_detail(
                        f"shard={shard_number}/{len(shard_paths)} sample={sample_id}"
                    )
                    if len(pending_rows) >= 8:
                        append_jsonl(detail_path, pending_rows)
                        pending_rows.clear()
                if pending_rows:
                    append_jsonl(detail_path, pending_rows)
                    pending_rows.clear()
            if remaining - set(valid_existing):
                missing = sorted(remaining - set(valid_existing))
                raise RuntimeError(f"Selected feature rows were not evaluated: {missing[:10]}")
            del model, controller
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            progress.set_stage(
                "1/2 physical Top-8 leave-one-document-out replay",
                total=len(selected_ids),
                initial=len(selected_ids),
            )

        ordered_rows = [valid_existing[sample_id] for sample_id in selected_ids]
        summary = summarize_rows(
            ordered_rows,
            minimum_total_jsd=args.minimum_total_jsd,
            progress=progress,
        )
        summary.update(
            {
                "dataset": args.dataset,
                "split": args.split,
                "feature_dir": str(args.feature_dir.resolve()),
                "controller_checkpoint": str(args.controller_checkpoint.resolve()),
                "controller_checkpoint_sha256": checkpoint_hash,
                "llm_model": str(args.llm_model.resolve()),
            }
        )
        atomic_write_json(summary_path, summary)
        write_markdown(summary, args.output_dir / "summary.md")
        logging.info(
            "Gate/attention fidelity audit complete: gate_spearman=%s attention_spearman=%s "
            "coverage=%.4f output=%s",
            summary["fidelity"]["gate"]["median_per_question_spearman"],
            summary["fidelity"]["attention_controlled_layers"]["median_per_question_spearman"],
            summary["measurable_loo_coverage"],
            args.output_dir,
        )
    finally:
        progress.close()


if __name__ == "__main__":
    main()

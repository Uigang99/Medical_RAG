#!/usr/bin/env python3
"""Audit whether an all-layer document gate follows physical deletion.

This is a bounded mechanism test, not controller training.  For every cached
Top-8 question/document pair it evaluates gate factors 1, .75, .5, .25, and 0.
Positive factors multiply that document's attention odds at every Llama layer
for every query.  Gate zero uses an exact ``-inf`` key mask.  Position IDs are
kept fixed across the path so the experiment changes only document information
flow; a separately cached physical-token deletion is the external reference.

The primary question is whether lowering the gate moves the four-choice output
distribution monotonically from the full-document result toward the physical
deletion result.  All probabilities reported here are normalized over the
four MCQ option tokens, rather than over the complete vocabulary.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.generation.learned_semantic_attention import (  # noqa: E402
    document_bias_to_token_bias,
)
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from scripts.evaluate_rag2_all_layer_document_mask_contract import (  # noqa: E402
    append_jsonl,
    atomic_json,
    atomic_text,
    finite_mean,
    finite_median,
    load_details,
)
from scripts.evaluate_rag2_semantic_gate_fidelity import (  # noqa: E402
    gold_margin,
    jensen_shannon_divergence,
    spearman_correlation,
)
from scripts.train_rag2_semantic_attention_controller import (  # noqa: E402
    list_feature_shards,
    load_feature_shard,
    model_bundle_identity,
)


RUN_VERSION = "rag2_all_layer_continuous_gate_contract_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--deletion-reference-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("medqa", "medmcqa"), default="medqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use the first N questions from the deletion-reference cohort; 0 uses all",
    )
    parser.add_argument(
        "--gate-factors",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument("--gate-batch-size", type=int, default=8)
    parser.add_argument("--document-count", type=int, default=8)
    parser.add_argument(
        "--meaningful-probability-delta",
        type=float,
        default=0.01,
        help="Ignore option changes smaller than this in direction metrics",
    )
    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=0.001,
        help="Allowed backward probability movement per adjacent gate step",
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


def canonical_factors(values: Iterable[float]) -> list[float]:
    factors = sorted({float(value) for value in values}, reverse=True)
    if not factors or factors[0] != 1.0 or factors[-1] != 0.0:
        raise ValueError("Gate factors must include both 1 and 0")
    if any(not 0.0 <= factor <= 1.0 for factor in factors):
        raise ValueError("Gate factors must lie in [0, 1]")
    return factors


def trajectory_metrics(
    values: list[float],
    target_value: float,
    gate_factors: list[float],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Measure movement from the gate-1 value toward ``target_value``.

    ``progress`` is signed so that positive values move toward the target.
    Normalized progress 0 is the full-document endpoint and 1 is the target.
    """

    if len(values) != len(gate_factors) or not values:
        raise ValueError("Trajectory values and gate factors must align")
    source = float(values[0])
    delta = float(target_value) - source
    direction = 1.0 if delta >= 0.0 else -1.0
    progress = [(float(value) - source) * direction for value in values]
    magnitude = abs(delta)
    normalized = [value / magnitude for value in progress] if magnitude else [0.0] * len(values)
    adjacent = [
        progress[index + 1] >= progress[index] - tolerance
        for index in range(len(progress) - 1)
    ]
    suppression = [1.0 - factor for factor in gate_factors]
    return {
        "source": source,
        "target": float(target_value),
        "target_delta": delta,
        "progress": progress,
        "normalized_progress": normalized,
        "adjacent_monotonic_rate": statistics.fmean(adjacent) if adjacent else 1.0,
        "strictly_monotonic_with_tolerance": all(adjacent),
        "suppression_progress_spearman": spearman_correlation(suppression, progress),
        "endpoint_moves_toward_target": progress[-1] > tolerance,
    }


def decreasing_distance_metrics(
    distributions: list[torch.Tensor],
    target: torch.Tensor,
    gate_factors: list[float],
    *,
    tolerance: float,
) -> dict[str, Any]:
    distances = [float(torch.abs(value - target).sum().item()) for value in distributions]
    adjacent = [
        distances[index + 1] <= distances[index] + tolerance
        for index in range(len(distances) - 1)
    ]
    return {
        "l1_distances": distances,
        "adjacent_monotonic_rate": statistics.fmean(adjacent) if adjacent else 1.0,
        "strictly_monotonic_with_tolerance": all(adjacent),
        # As gate increases from 0 to 1, distance from deletion should increase.
        "gate_distance_spearman": spearman_correlation(gate_factors, distances),
    }


def choice_logits_for_all_layer_gates(
    model: Any,
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    variants: list[tuple[int, float]],
    choice_token_ids: torch.Tensor,
    *,
    document_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Replay fixed-position traces with an all-query, all-layer document gate."""

    ids = input_ids.long().to(device).unsqueeze(0).expand(len(variants), -1)
    mapping = token_document_ids.long().to(device).unsqueeze(0).expand(len(variants), -1)
    document_bias = torch.zeros((len(variants), document_count), dtype=torch.float32)
    blocked = torch.full((len(variants),), -1, dtype=torch.long)
    for row, (document_index, factor) in enumerate(variants):
        if factor == 0.0:
            blocked[row] = int(document_index)
        elif factor != 1.0:
            document_bias[row, document_index] = math.log(float(factor))
    token_bias = document_bias_to_token_bias(document_bias.to(device), mapping)
    attention_mask = torch.ones_like(ids, dtype=torch.long)
    position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0).expand_as(ids)
    # Every query is active: document information flow is scaled from layer 0,
    # rather than only when the assistant answer attends to the prompt.
    query_mask = torch.ones_like(ids, dtype=torch.float32)
    with torch.inference_mode():
        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            semantic_token_bias=token_bias,
            semantic_query_mask=query_mask,
            semantic_layer_start=0,
            semantic_token_document_ids=mapping,
            semantic_blocked_document_ids=blocked.to(device),
            semantic_document_block_layer_start=0,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids).cpu()
    del outputs
    return logits


def evaluate_sample(
    args: argparse.Namespace,
    payload: dict[str, Any],
    row_index: int,
    reference: dict[str, Any],
    model: Any,
    choice_token_ids: torch.Tensor,
    factors: list[float],
) -> dict[str, Any]:
    variants = [
        (document_index, factor)
        for document_index in range(args.document_count)
        for factor in factors
    ]
    logits_chunks: list[torch.Tensor] = []
    for start in range(0, len(variants), args.gate_batch_size):
        chunk = variants[start : start + args.gate_batch_size]
        logits_chunks.append(
            choice_logits_for_all_layer_gates(
                model,
                payload["input_ids"][row_index],
                payload["token_document_ids"][row_index],
                chunk,
                choice_token_ids,
                document_count=args.document_count,
                device=torch.device(args.device),
            )
        )
    logits = torch.cat(logits_chunks, dim=0)
    by_variant = {variant: logits[index] for index, variant in enumerate(variants)}
    gold_index = int(payload["gold_options"][row_index])
    documents: list[dict[str, Any]] = []
    for document_index in range(args.document_count):
        factor_logits = [by_variant[(document_index, factor)] for factor in factors]
        factor_probabilities = [torch.softmax(value, dim=-1) for value in factor_logits]
        deletion_record = reference["documents"][document_index]["physical_delete"]
        deletion_logits = torch.tensor(deletion_record["choice_logits"], dtype=torch.float32)
        deletion_probability = torch.softmax(deletion_logits, dim=-1)
        full_probability = factor_probabilities[0]
        physical_delta = deletion_probability - full_probability
        dominant_choice = int(torch.abs(physical_delta).argmax().item())
        dominant_values = [float(value[dominant_choice].item()) for value in factor_probabilities]
        dominant = trajectory_metrics(
            dominant_values,
            float(deletion_probability[dominant_choice].item()),
            factors,
            tolerance=args.probability_tolerance,
        )
        option_trajectories: dict[str, dict[str, Any]] = {}
        for option_index, option in enumerate(CHOICES):
            values = [float(value[option_index].item()) for value in factor_probabilities]
            option_trajectories[option] = trajectory_metrics(
                values,
                float(deletion_probability[option_index].item()),
                factors,
                tolerance=args.probability_tolerance,
            )
        factor_margins = [gold_margin(value, gold_index) for value in factor_logits]
        margin_trajectory = trajectory_metrics(
            factor_margins,
            float(deletion_record["gold_margin"]),
            factors,
            tolerance=max(0.01, 4.0 * args.probability_tolerance),
        )
        distribution_trajectory = decreasing_distance_metrics(
            factor_probabilities,
            deletion_probability,
            factors,
            tolerance=4.0 * args.probability_tolerance,
        )
        gate_zero_probability = factor_probabilities[-1]
        zero_delta = gate_zero_probability - full_probability
        zero_dominant_choice = int(torch.abs(zero_delta).argmax().item())
        zero_dominant_trajectory = trajectory_metrics(
            [float(value[zero_dominant_choice].item()) for value in factor_probabilities],
            float(gate_zero_probability[zero_dominant_choice].item()),
            factors,
            tolerance=args.probability_tolerance,
        )
        zero_distribution_trajectory = decreasing_distance_metrics(
            factor_probabilities,
            gate_zero_probability,
            factors,
            tolerance=4.0 * args.probability_tolerance,
        )
        documents.append(
            {
                "document_index": document_index,
                "pair_id": str(payload["pair_ids"][row_index][document_index]),
                "semantic_class_id": int(
                    payload["semantic_class_ids"][row_index, document_index]
                ),
                "gate_factors": factors,
                "choice_logits": {
                    str(factor): [float(value) for value in factor_logits[index].tolist()]
                    for index, factor in enumerate(factors)
                },
                "choice_probabilities": {
                    str(factor): [float(value) for value in factor_probabilities[index].tolist()]
                    for index, factor in enumerate(factors)
                },
                "physical_delete_choice_logits": [float(value) for value in deletion_logits.tolist()],
                "physical_delete_choice_probabilities": [
                    float(value) for value in deletion_probability.tolist()
                ],
                "dominant_changed_choice": CHOICES[dominant_choice],
                "dominant_physical_probability_delta": float(physical_delta[dominant_choice]),
                "dominant_trajectory": dominant,
                "gate_zero_dominant_changed_choice": CHOICES[zero_dominant_choice],
                "gate_zero_dominant_probability_delta": float(zero_delta[zero_dominant_choice]),
                "gate_zero_dominant_trajectory": zero_dominant_trajectory,
                "option_trajectories": option_trajectories,
                "gold_margin_trajectory": margin_trajectory,
                "distribution_trajectory": distribution_trajectory,
                "gate_zero_distribution_trajectory": zero_distribution_trajectory,
                "gate_zero_vs_physical_delete": {
                    "prediction_agreement": int(gate_zero_probability.argmax())
                    == int(deletion_probability.argmax()),
                    "choice_probability_l1": float(
                        torch.abs(gate_zero_probability - deletion_probability).sum().item()
                    ),
                    "choice_distribution_jsd": jensen_shannon_divergence(
                        deletion_probability,
                        gate_zero_probability,
                    ),
                },
            }
        )
    return {
        "run_version": RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "sample_id": str(payload["sample_ids"][row_index]),
        "documents": documents,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, factors: list[float]) -> dict[str, Any]:
    documents = [document for row in rows for document in row["documents"]]
    meaningful = [
        document
        for document in documents
        if abs(float(document["dominant_physical_probability_delta"]))
        >= args.meaningful_probability_delta
    ]
    meaningful_zero = [
        document
        for document in documents
        if abs(float(document["gate_zero_dominant_probability_delta"]))
        >= args.meaningful_probability_delta
    ]
    meaningful_options = [
        trajectory
        for document in documents
        for trajectory in document["option_trajectories"].values()
        if abs(float(trajectory["target_delta"])) >= args.meaningful_probability_delta
    ]

    def by_gate(field: str) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for index, factor in enumerate(factors):
            result[str(factor)] = finite_median(
                document["dominant_trajectory"][field][index] for document in meaningful
            )
        return result

    direction_by_gate: dict[str, float | None] = {}
    for index, factor in enumerate(factors):
        direction_by_gate[str(factor)] = finite_mean(
            float(document["dominant_trajectory"]["progress"][index] >= -args.probability_tolerance)
            for document in meaningful
        )

    metrics = {
        "questions": len(rows),
        "documents": len(documents),
        "meaningful_dominant_documents": len(meaningful),
        "meaningful_gate_zero_dominant_documents": len(meaningful_zero),
        "meaningful_option_trajectories": len(meaningful_options),
        "dominant_choice_direction_agreement_by_gate": direction_by_gate,
        "dominant_choice_median_normalized_progress_by_gate": by_gate("normalized_progress"),
        "dominant_choice_adjacent_monotonic_step_rate": finite_mean(
            document["dominant_trajectory"]["adjacent_monotonic_rate"]
            for document in meaningful
        ),
        "dominant_choice_strict_monotonic_document_rate": finite_mean(
            float(document["dominant_trajectory"]["strictly_monotonic_with_tolerance"])
            for document in meaningful
        ),
        "median_dominant_choice_suppression_spearman": finite_median(
            document["dominant_trajectory"]["suppression_progress_spearman"]
            for document in meaningful
        ),
        "gate_zero_dominant_choice_adjacent_monotonic_step_rate": finite_mean(
            document["gate_zero_dominant_trajectory"]["adjacent_monotonic_rate"]
            for document in meaningful_zero
        ),
        "gate_zero_dominant_choice_strict_monotonic_document_rate": finite_mean(
            float(
                document["gate_zero_dominant_trajectory"][
                    "strictly_monotonic_with_tolerance"
                ]
            )
            for document in meaningful_zero
        ),
        "median_gate_zero_dominant_choice_suppression_spearman": finite_median(
            document["gate_zero_dominant_trajectory"]["suppression_progress_spearman"]
            for document in meaningful_zero
        ),
        "gate_zero_distribution_adjacent_monotonic_step_rate": finite_mean(
            document["gate_zero_distribution_trajectory"]["adjacent_monotonic_rate"]
            for document in documents
        ),
        "gate_zero_distribution_strict_monotonic_document_rate": finite_mean(
            float(
                document["gate_zero_distribution_trajectory"][
                    "strictly_monotonic_with_tolerance"
                ]
            )
            for document in documents
        ),
        "all_meaningful_options_adjacent_monotonic_step_rate": finite_mean(
            trajectory["adjacent_monotonic_rate"] for trajectory in meaningful_options
        ),
        "physical_distribution_adjacent_monotonic_step_rate": finite_mean(
            document["distribution_trajectory"]["adjacent_monotonic_rate"]
            for document in documents
        ),
        "physical_distribution_strict_monotonic_document_rate": finite_mean(
            float(document["distribution_trajectory"]["strictly_monotonic_with_tolerance"])
            for document in documents
        ),
        "median_gate_vs_physical_distribution_distance_spearman": finite_median(
            document["distribution_trajectory"]["gate_distance_spearman"]
            for document in documents
        ),
        "gold_margin_adjacent_monotonic_step_rate": finite_mean(
            document["gold_margin_trajectory"]["adjacent_monotonic_rate"]
            for document in documents
            if abs(float(document["gold_margin_trajectory"]["target_delta"])) >= 0.25
        ),
        "gate_zero_vs_physical_delete_prediction_agreement": finite_mean(
            float(document["gate_zero_vs_physical_delete"]["prediction_agreement"])
            for document in documents
        ),
        "gate_zero_vs_physical_delete_mean_probability_l1": finite_mean(
            document["gate_zero_vs_physical_delete"]["choice_probability_l1"]
            for document in documents
        ),
        "gate_zero_vs_physical_delete_mean_jsd": finite_mean(
            document["gate_zero_vs_physical_delete"]["choice_distribution_jsd"]
            for document in documents
        ),
    }
    criteria = {
        "continuous_path_is_monotonic": bool(
            (metrics["gate_zero_dominant_choice_adjacent_monotonic_step_rate"] or 0.0)
            >= 0.85
            and (metrics["gate_zero_dominant_choice_strict_monotonic_document_rate"] or 0.0)
            >= 0.60
            and (metrics["median_gate_zero_dominant_choice_suppression_spearman"] or -1.0)
            >= 0.80
        ),
        "continuous_path_tracks_physical_deletion": bool(
            (direction_by_gate.get("0.0") or 0.0) >= 0.90
            and (metrics["physical_distribution_adjacent_monotonic_step_rate"] or 0.0) >= 0.80
            and (metrics["median_gate_vs_physical_distribution_distance_spearman"] or -1.0)
            >= 0.70
        ),
    }
    passed = all(criteria.values())
    return {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "gate_factors": factors,
        "probability_definition": "softmax normalized over the four MCQ option tokens",
        "metrics": metrics,
        "go_criteria": criteria,
        "gate_contract_pass": passed,
        "interpretation": (
            "Intermediate all-layer gates provide a monotonic document-control path that "
            "tracks physical deletion; a bounded learned-gate pilot is justified."
            if passed
            else "Intermediate all-layer gates do not reliably trace physical deletion; "
            "do not train a continuous controller from this operator yet."
        ),
        "scope_limit": (
            "This is an MCQ mechanism audit on fixed cached traces. It does not show that a "
            "controller can predict useful gate values or that the relation remains monotonic "
            "during free rationale generation."
        ),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]

    def percent(value: Any) -> str:
        return "NA" if value is None else f"{100.0 * float(value):.2f}%"

    def number(value: Any, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    progress = metrics["dominant_choice_median_normalized_progress_by_gate"]
    direction = metrics["dominant_choice_direction_agreement_by_gate"]
    lines = [
        "# All-layer continuous document-gate contract",
        "",
        f"- Questions: {metrics['questions']}",
        f"- Documents: {metrics['documents']}",
        f"- Documents with >=1% dominant option change under physical deletion: "
        f"{metrics['meaningful_dominant_documents']}",
        f"- Documents with >=1% dominant option change under exact gate zero: "
        f"{metrics['meaningful_gate_zero_dominant_documents']}",
        "- Probability: softmax normalized over A/B/C/D option-token logits",
        "- Gate 1: full document; gate 0: exact all-layer key/value block; positions fixed",
        "",
        "| Gate | Median progress toward physical deletion | Direction agreement |",
        "|---:|---:|---:|",
    ]
    for factor in summary["gate_factors"]:
        key = str(factor)
        lines.append(f"| {factor:g} | {number(progress[key])} | {percent(direction[key])} |")
    lines.extend(
        [
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Dominant-option adjacent monotonic steps | "
            f"{percent(metrics['dominant_choice_adjacent_monotonic_step_rate'])} |",
            f"| Fully monotonic document trajectories | "
            f"{percent(metrics['dominant_choice_strict_monotonic_document_rate'])} |",
            f"| Median suppression/progress Spearman | "
            f"{number(metrics['median_dominant_choice_suppression_spearman'])} |",
            f"| Gate-internal dominant-option adjacent monotonic steps | "
            f"{percent(metrics['gate_zero_dominant_choice_adjacent_monotonic_step_rate'])} |",
            f"| Gate-internal fully monotonic document trajectories | "
            f"{percent(metrics['gate_zero_dominant_choice_strict_monotonic_document_rate'])} |",
            f"| Gate-internal median suppression/progress Spearman | "
            f"{number(metrics['median_gate_zero_dominant_choice_suppression_spearman'])} |",
            f"| Whole-distribution adjacent monotonic steps toward deletion | "
            f"{percent(metrics['physical_distribution_adjacent_monotonic_step_rate'])} |",
            f"| Fully monotonic whole-distribution trajectories | "
            f"{percent(metrics['physical_distribution_strict_monotonic_document_rate'])} |",
            f"| Gate-0 answer agreement with physical deletion | "
            f"{percent(metrics['gate_zero_vs_physical_delete_prediction_agreement'])} |",
            "",
            "## Decision",
            "",
            f"- Continuous path monotonic: {summary['go_criteria']['continuous_path_is_monotonic']}",
            f"- Tracks physical deletion: "
            f"{summary['go_criteria']['continuous_path_tracks_physical_deletion']}",
            f"- Overall pass: {summary['gate_contract_pass']}",
            f"- Interpretation: {summary['interpretation']}",
            f"- Scope: {summary['scope_limit']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    factors = canonical_factors(args.gate_factors)
    if args.gate_batch_size <= 0 or args.document_count <= 0:
        raise ValueError("Batch size and document count must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.meaningful_probability_delta <= 0 or args.probability_tolerance < 0:
        raise ValueError("Invalid probability thresholds")
    feature_manifest_path = args.feature_dir / "preparation_manifest.json"
    reference_contract_path = args.deletion_reference_dir / "run_contract.json"
    reference_detail_path = args.deletion_reference_dir / "mask_contract_details.jsonl"
    for path in (feature_manifest_path, reference_contract_path, reference_detail_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    reference_contract = json.loads(reference_contract_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != args.dataset or reference_contract.get("dataset") != args.dataset:
        raise ValueError("Feature/reference dataset does not match --dataset")
    if reference_contract.get("split") != args.split:
        raise ValueError("Deletion-reference split does not match --split")
    model_identity = model_bundle_identity(args.llm_model)
    if manifest.get("llm_model_bundle") != model_identity:
        raise RuntimeError("Prepared features and requested Llama model differ")
    if reference_contract.get("llm_model_bundle") != model_identity:
        raise RuntimeError("Deletion reference and requested Llama model differ")
    selected_ids = [str(value) for value in reference_contract["selected_ids"]]
    if args.max_samples:
        selected_ids = selected_ids[: args.max_samples]
    reference_rows = load_details(reference_detail_path)
    missing_reference = set(selected_ids) - set(reference_rows)
    if missing_reference:
        raise RuntimeError(f"Deletion reference misses {len(missing_reference)} selected questions")
    hidden_size = int(manifest["feature_hidden_size"])
    fingerprint = str(manifest["contract_fingerprint"])
    shard_paths = list_feature_shards(args.feature_dir, args.split, manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "continuous_gate_details.jsonl"
    summary_path = args.output_dir / "continuous_gate_summary.json"
    report_path = args.output_dir / "continuous_gate_report.md"
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "feature_contract_fingerprint": fingerprint,
        "llm_model_bundle": model_identity,
        "deletion_reference_contract": reference_contract,
        "selected_ids": selected_ids,
        "gate_factors": factors,
        "gate_batch_size": args.gate_batch_size,
        "document_count": args.document_count,
        "meaningful_probability_delta": args.meaningful_probability_delta,
        "probability_tolerance": args.probability_tolerance,
        "dtype": args.dtype,
    }
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Continuous-gate resume mismatch; use a new output directory")
    else:
        if detail_path.exists() and not args.resume:
            detail_path.unlink()
        atomic_json(contract_path, contract)
    existing = load_details(detail_path) if args.resume else {}
    rows = {sample_id: row for sample_id, row in existing.items() if sample_id in set(selected_ids)}
    remaining = set(selected_ids) - set(rows)
    variants_per_question = args.document_count * len(factors)
    logging.info(
        "Continuous all-layer gate plan: dataset=%s split=%s questions=%d cached=%d "
        "remaining=%d documents/question=%d factors=%s variants/question=%d",
        args.dataset,
        args.split,
        len(selected_ids),
        len(rows),
        len(remaining),
        args.document_count,
        factors,
        variants_per_question,
    )
    logging.info(
        "Reference is cached physical deletion; durable output every 4 questions; "
        "rerunning the same command resumes safely"
    )
    if args.plan_only:
        return

    progress = PipelineProgress(
        overall_total=2 * len(selected_ids),
        overall_initial=len(rows),
        desc=f"AllLayerContinuousGate:{args.dataset}",
    )
    try:
        if remaining:
            attention_name = register_semantic_attention()
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model, local_files_only=True, use_fast=True
            )
            choice_ids: list[int] = []
            for choice in CHOICES:
                encoded = tokenizer.encode(choice, add_special_tokens=False)
                if len(encoded) != 1:
                    raise RuntimeError(f"Choice {choice} is not one token: {encoded}")
                choice_ids.append(encoded[0])
            choice_token_ids = torch.tensor(choice_ids, dtype=torch.long, device=args.device)
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            logging.info("Loading frozen target Llama: %s", args.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                local_files_only=True,
                dtype=dtype,
                attn_implementation=attention_name,
            ).to(torch.device(args.device))
            model.eval()
            progress.set_stage(
                "1/2 all-layer continuous gate replay",
                total=len(selected_ids),
                initial=len(rows),
            )
            pending: list[dict[str, Any]] = []
            for shard_index, path in enumerate(shard_paths, start=1):
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
                    row = evaluate_sample(
                        args,
                        payload,
                        row_index,
                        reference_rows[sample_id],
                        model,
                        choice_token_ids,
                        factors,
                    )
                    rows[sample_id] = row
                    pending.append(row)
                    progress.update(1)
                    progress.set_detail(
                        f"shard={shard_index}/{len(shard_paths)} sample={sample_id} "
                        f"variants={variants_per_question}"
                    )
                    if len(pending) >= 4:
                        append_jsonl(detail_path, pending)
                        pending.clear()
                if pending:
                    append_jsonl(detail_path, pending)
                    pending.clear()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        progress.set_stage("2/2 aggregate monotonicity criteria", total=len(selected_ids))
        ordered_rows = [rows[sample_id] for sample_id in selected_ids]
        for _ in ordered_rows:
            progress.update(1)
        summary = summarize(ordered_rows, args, factors)
        atomic_json(summary_path, summary)
        atomic_text(report_path, report_markdown(summary))
    except BaseException:
        logging.exception(
            "Continuous-gate audit failed: completed=%d remaining=%d durable_cache=%s "
            "rerun_same_command_resumes=%s",
            len(rows),
            len(selected_ids) - len(rows),
            detail_path,
            args.resume,
        )
        raise
    finally:
        progress.close()
    logging.info(
        "Continuous-gate audit complete: questions=%d summary=%s report=%s",
        len(selected_ids),
        summary_path,
        report_path,
    )


if __name__ == "__main__":
    main()

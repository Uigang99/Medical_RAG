#!/usr/bin/env python3
"""Test whether a continuous attention gate behaves like document control.

This is a mechanism preflight, not controller training.  For each cached
Top-8 MedQA rationale trace it compares physical document-token removal with
an attention-logit gate approaching zero, then tests whether amplification
and suppression responses track each document's physical leave-one-out effect.
The target Llama is frozen and no gold information is used to choose gates.
"""

from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.generation.learned_semantic_attention import document_bias_to_token_bias  # noqa: E402
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from scripts.evaluate_rag2_semantic_gate_fidelity import (  # noqa: E402
    build_physical_loo_batch,
    choice_logits_for_loo_batch,
    gold_margin,
    jensen_shannon_divergence,
    select_sample_ids,
    spearman_correlation,
)
from scripts.train_rag2_semantic_attention_controller import (  # noqa: E402
    list_feature_shards,
    load_feature_shard,
    model_bundle_identity,
)


RUN_VERSION = "rag2_continuous_attention_gate_contract_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("medqa", "medmcqa"), default="medqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128, help="0 uses the complete split")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--gate-factors", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.5, 2.0])
    parser.add_argument("--gate-batch-size", type=int, default=8)
    parser.add_argument("--zero-log-bias", type=float, default=-20.0)
    parser.add_argument("--attention-scope", choices=("rationale_wide", "final_choice"), default="rationale_wide")
    parser.add_argument("--semantic-layer-start", type=int, default=16)
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_details(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}:{line_number}")
            result[sample_id] = row
    return result


def mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(kept) if kept else None


def median(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(kept) if kept else None


def choice_logits_for_gate_variants(
    model: Any,
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    assistant_query_start: int,
    biases: torch.Tensor,
    choice_token_ids: torch.Tensor,
    *,
    attention_scope: str,
    semantic_layer_start: int,
    device: torch.device,
) -> torch.Tensor:
    """Replay one fixed trace under a batch of different document gates."""

    if biases.ndim != 2 or int(biases.shape[1]) != 8:
        raise ValueError("Gate biases must have shape [variants, 8]")
    variant_count = int(biases.shape[0])
    ids = input_ids.long().to(device).unsqueeze(0).expand(variant_count, -1)
    mapping = token_document_ids.long().to(device).unsqueeze(0).expand(variant_count, -1)
    attention_mask = torch.ones_like(ids, dtype=torch.long)
    position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0).expand_as(ids)
    query_mask = torch.zeros(ids.shape, dtype=torch.float32, device=device)
    if attention_scope == "rationale_wide":
        query_mask[:, assistant_query_start:] = 1.0
    else:
        query_mask[:, -1] = 1.0
    token_bias = document_bias_to_token_bias(biases.to(device=device, dtype=torch.float32), mapping)
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
            semantic_layer_start=semantic_layer_start,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids).cpu()
    del outputs
    return logits


def evaluate_sample(
    args: argparse.Namespace,
    payload: dict[str, Any],
    row_index: int,
    model: Any,
    tokenizer: Any,
    choice_token_ids: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device(args.device)
    input_ids = payload["input_ids"][row_index]
    mapping = payload["token_document_ids"][row_index]
    assistant_start = int(payload["assistant_query_starts"][row_index])
    physical_batch = build_physical_loo_batch(
        input_ids,
        mapping,
        assistant_start,
        pad_token_id=int(tokenizer.pad_token_id),
        attention_scope=args.attention_scope,
    )
    physical_logits, _ = choice_logits_for_loo_batch(
        model,
        physical_batch,
        torch.zeros((1, 8), dtype=torch.float32),
        choice_token_ids,
        args.semantic_layer_start,
        device,
    )
    physical_logits = physical_logits.cpu()

    variants: list[tuple[int, float]] = [
        (document_index, factor)
        for document_index in range(8)
        for factor in args.gate_factors
    ]
    variant_logits: list[torch.Tensor] = []
    for start in range(0, len(variants), args.gate_batch_size):
        chunk = variants[start : start + args.gate_batch_size]
        biases = torch.zeros((len(chunk), 8), dtype=torch.float32)
        for row, (document_index, factor) in enumerate(chunk):
            biases[row, document_index] = args.zero_log_bias if factor == 0 else math.log(factor)
        variant_logits.append(
            choice_logits_for_gate_variants(
                model,
                input_ids,
                mapping,
                assistant_start,
                biases,
                choice_token_ids,
                attention_scope=args.attention_scope,
                semantic_layer_start=args.semantic_layer_start,
                device=device,
            )
        )
    gate_logits = torch.cat(variant_logits, dim=0)
    by_variant = {variant: gate_logits[index] for index, variant in enumerate(variants)}
    gold_index = int(payload["gold_options"][row_index])
    full_margin = gold_margin(physical_logits[0], gold_index)
    full_probability = torch.softmax(physical_logits[0], dim=-1)
    physical_contributions: list[float] = []
    suppression_contributions: list[float] = []
    amplification_effects: list[float] = []
    zero_prediction_agreements: list[bool] = []
    zero_margin_errors: list[float] = []
    zero_jsds: list[float] = []
    monotonic_sign_agreements: list[bool] = []
    document_rows: list[dict[str, Any]] = []
    ordered_factors = sorted(set(args.gate_factors + [1.0]))
    for document_index in range(8):
        removed = physical_logits[document_index + 1]
        removed_margin = gold_margin(removed, gold_index)
        zero_logits = by_variant[(document_index, 0.0)]
        zero_margin = gold_margin(zero_logits, gold_index)
        factor_margins = {
            str(factor): (
                full_margin if factor == 1.0 else gold_margin(by_variant[(document_index, factor)], gold_index)
            )
            for factor in ordered_factors
        }
        physical = full_margin - removed_margin
        suppression = full_margin - zero_margin
        amplification = factor_margins.get("2.0", full_margin) - full_margin
        physical_contributions.append(physical)
        suppression_contributions.append(suppression)
        amplification_effects.append(amplification)
        zero_prediction_agreements.append(int(removed.argmax()) == int(zero_logits.argmax()))
        zero_margin_errors.append(abs(removed_margin - zero_margin))
        zero_jsds.append(
            jensen_shannon_divergence(
                torch.softmax(removed, dim=-1), torch.softmax(zero_logits, dim=-1)
            )
        )
        factor_rho = spearman_correlation(
            ordered_factors, [factor_margins[str(factor)] for factor in ordered_factors]
        )
        monotonic_sign_agreements.append(
            factor_rho is not None and (physical == 0 or math.copysign(1.0, factor_rho) == math.copysign(1.0, physical))
        )
        document_rows.append(
            {
                "document_index": document_index,
                "pair_id": str(payload["pair_ids"][row_index][document_index]),
                "semantic_class_id": int(payload["semantic_class_ids"][row_index, document_index]),
                "physical_loo_contribution": physical,
                "zero_gate_contribution": suppression,
                "amplification_effect": amplification,
                "zero_vs_delete_prediction_agreement": zero_prediction_agreements[-1],
                "zero_vs_delete_margin_absolute_error": zero_margin_errors[-1],
                "zero_vs_delete_jsd": zero_jsds[-1],
                "factor_margin_spearman": factor_rho,
                "factor_margins": factor_margins,
            }
        )
    return {
        "run_version": RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "sample_id": str(payload["sample_ids"][row_index]),
        "full_gold_margin": full_margin,
        "full_prediction": CHOICES[int(physical_logits[0].argmax())],
        "physical_vs_zero_gate_spearman": spearman_correlation(
            physical_contributions, suppression_contributions
        ),
        "physical_vs_amplification_spearman": spearman_correlation(
            physical_contributions, amplification_effects
        ),
        "zero_vs_delete_prediction_agreement": statistics.fmean(zero_prediction_agreements),
        "zero_vs_delete_margin_mae": statistics.fmean(zero_margin_errors),
        "zero_vs_delete_mean_jsd": statistics.fmean(zero_jsds),
        "monotonic_sign_agreement": statistics.fmean(monotonic_sign_agreements),
        "full_probability": [float(value) for value in full_probability.tolist()],
        "documents": document_rows,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    documents = [document for row in rows for document in row["documents"]]
    metrics = {
        "questions": len(rows),
        "documents": len(documents),
        "median_per_question_physical_vs_zero_gate_spearman": median(
            row["physical_vs_zero_gate_spearman"] for row in rows
        ),
        "median_per_question_physical_vs_amplification_spearman": median(
            row["physical_vs_amplification_spearman"] for row in rows
        ),
        "zero_vs_delete_prediction_agreement": mean(
            document["zero_vs_delete_prediction_agreement"] for document in documents
        ),
        "zero_vs_delete_margin_mae": mean(
            document["zero_vs_delete_margin_absolute_error"] for document in documents
        ),
        "zero_vs_delete_mean_jsd": mean(document["zero_vs_delete_jsd"] for document in documents),
        "monotonic_direction_agreement": mean(row["monotonic_sign_agreement"] for row in rows),
    }
    criteria = {
        "zero_gate_tracks_physical_removal": bool(
            (metrics["median_per_question_physical_vs_zero_gate_spearman"] or -1) >= 0.50
            and (metrics["zero_vs_delete_prediction_agreement"] or 0) >= 0.90
        ),
        "amplification_tracks_physical_value": bool(
            (metrics["median_per_question_physical_vs_amplification_spearman"] or -1) >= 0.30
            and (metrics["monotonic_direction_agreement"] or 0) >= 0.70
        ),
    }
    return {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "attention_scope": args.attention_scope,
        "semantic_layer_start": args.semantic_layer_start,
        "gate_factors": sorted(set(args.gate_factors + [1.0])),
        "metrics": metrics,
        "go_criteria": criteria,
        "gate_contract_pass": all(criteria.values()),
        "interpretation": (
            "Gate intervention is faithful enough for a bounded controller pilot."
            if all(criteria.values())
            else "Do not train a continuous gate yet; the intervention itself does not reliably represent document use."
        ),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.max_samples < 0 or args.gate_batch_size <= 0:
        raise ValueError("Invalid sample or gate batch size")
    if any(factor < 0 for factor in args.gate_factors) or 0.0 not in args.gate_factors:
        raise ValueError("Gate factors must be non-negative and include zero")
    if not 0 <= args.semantic_layer_start < 32:
        raise ValueError("semantic-layer-start must be in [0, 31]")
    manifest_path = args.feature_dir / "preparation_manifest.json"
    if not manifest_path.is_file() or not args.llm_model.exists():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else args.llm_model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != args.dataset:
        raise ValueError("Feature dataset does not match --dataset")
    if manifest.get("llm_model_bundle") != model_bundle_identity(args.llm_model):
        raise RuntimeError("Prepared features and requested Llama model differ")
    hidden_size = int(manifest["feature_hidden_size"])
    fingerprint = str(manifest["contract_fingerprint"])
    shard_paths = list_feature_shards(args.feature_dir, args.split, manifest)
    selected_ids = select_sample_ids(
        shard_paths,
        dataset=args.dataset,
        split=args.split,
        fingerprint=fingerprint,
        hidden_size=hidden_size,
        max_samples=args.max_samples,
        seed=args.sample_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "gate_contract_details.jsonl"
    summary_path = args.output_dir / "gate_contract_summary.json"
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "feature_contract_fingerprint": fingerprint,
        "llm_model_bundle": model_bundle_identity(args.llm_model),
        "selected_ids": selected_ids,
        "gate_factors": args.gate_factors,
        "zero_log_bias": args.zero_log_bias,
        "attention_scope": args.attention_scope,
        "semantic_layer_start": args.semantic_layer_start,
        "dtype": args.dtype,
    }
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Gate-contract resume mismatch; use a new output directory")
    else:
        if detail_path.exists() and not args.resume:
            detail_path.unlink()
        atomic_json(contract_path, contract)
    existing = load_details(detail_path) if args.resume else {}
    rows = {sample_id: row for sample_id, row in existing.items() if sample_id in set(selected_ids)}
    remaining = set(selected_ids) - set(rows)
    variants_per_question = 9 + 8 * len(args.gate_factors)
    logging.info(
        "Gate-contract plan: dataset=%s split=%s questions=%d cached=%d remaining=%d variants/question=%d",
        args.dataset,
        args.split,
        len(selected_ids),
        len(rows),
        len(remaining),
        variants_per_question,
    )
    if args.plan_only:
        return
    progress = PipelineProgress(
        overall_total=2 * len(selected_ids),
        overall_initial=len(rows),
        desc=f"GateContract:{args.dataset}",
    )
    try:
        if remaining:
            attention_name = register_semantic_attention()
            tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            choice_ids = []
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
                f"1/2 physical deletion vs continuous gates ({args.attention_scope})",
                total=len(selected_ids),
                initial=len(rows),
            )
            pending = []
            for shard_index, path in enumerate(shard_paths, start=1):
                payload = load_feature_shard(
                    path,
                    dataset=args.dataset,
                    split=args.split,
                    fingerprint=fingerprint,
                    hidden_size=hidden_size,
                )
                if "assistant_query_starts" not in payload:
                    raise RuntimeError("Rationale-wide gate audit requires assistant_query_starts")
                for row_index, sample_id_value in enumerate(payload["sample_ids"]):
                    sample_id = str(sample_id_value)
                    if sample_id not in remaining:
                        continue
                    row = evaluate_sample(args, payload, row_index, model, tokenizer, choice_token_ids)
                    rows[sample_id] = row
                    pending.append(row)
                    progress.update(1)
                    progress.set_detail(
                        f"shard={shard_index}/{len(shard_paths)} sample={sample_id} variants={variants_per_question}"
                    )
                    if len(pending) >= 4:
                        append_jsonl(detail_path, pending)
                        pending.clear()
                if pending:
                    append_jsonl(detail_path, pending)
                    pending.clear()
            del model
            torch.cuda.empty_cache()
        progress.set_stage("2/2 aggregate mechanism criteria", total=len(selected_ids))
        ordered_rows = [rows[sample_id] for sample_id in selected_ids]
        for _ in ordered_rows:
            progress.update(1)
        summary = summarize(ordered_rows, args)
        atomic_json(summary_path, summary)
    finally:
        progress.close()
    logging.info("Gate-contract preflight complete: %s", summary_path)


if __name__ == "__main__":
    main()

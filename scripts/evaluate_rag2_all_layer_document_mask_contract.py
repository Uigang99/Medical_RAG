#!/usr/bin/env python3
"""Compare physical document deletion with exact all-layer attention masks.

This is a bounded mechanism test, not controller training.  For every selected
Top-8 MedQA question and every document it compares four interventions using
the final direct-choice A/B/C/D logits:

1. physically remove the mapped document tokens;
2. keep the tokens but block their key/value positions at every Llama layer;
3. apply the same exact block and compact later position IDs as if deleted;
4. reproduce the legacy layer-16 assistant-only ``-20`` logit bias.

The compact-position condition should match physical deletion up to numerical
error.  The original-position condition isolates the residual effect of token
positions after document content flow has been blocked.
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

from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from scripts.evaluate_rag2_attention_gate_contract import (  # noqa: E402
    choice_logits_for_gate_variants,
)
from scripts.evaluate_rag2_semantic_gate_fidelity import (  # noqa: E402
    build_physical_loo_batch,
    gold_margin,
    jensen_shannon_divergence,
    pearson_correlation,
    select_sample_ids,
    spearman_correlation,
)
from scripts.train_rag2_semantic_attention_controller import (  # noqa: E402
    list_feature_shards,
    load_feature_shard,
    model_bundle_identity,
)


RUN_VERSION = "rag2_all_layer_document_mask_contract_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
CONDITIONS = (
    "all_layer_original_position",
    "all_layer_compact_position",
    "legacy_layer16_assistant_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("medqa", "medmcqa"), default="medqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64, help="0 uses the complete split")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--document-count", type=int, default=8)
    parser.add_argument("--legacy-layer-start", type=int, default=16)
    parser.add_argument("--legacy-zero-log-bias", type=float, default=-20.0)
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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_details(path: Path) -> dict[str, dict[str, Any]]:
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


def finite_mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(kept) if kept else None


def finite_median(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(kept) if kept else None


def build_exact_document_mask_batch(
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    *,
    compact_positions: bool,
    document_count: int = 8,
) -> dict[str, torch.Tensor]:
    """Build one fixed-length all-layer mask variant per document.

    Document tokens remain in the sequence.  ``blocked_document_ids`` tells
    the custom attention implementation which key/value positions must receive
    exactly zero attention at every layer.  When ``compact_positions`` is true,
    the remaining tokens receive the same position IDs they would have after
    physical deletion.
    """

    ids = input_ids.long().cpu()
    mapping = token_document_ids.long().cpu()
    if ids.ndim != 1 or mapping.shape != ids.shape:
        raise ValueError("input_ids and token_document_ids must be aligned vectors")
    blocked = torch.arange(document_count, dtype=torch.long)
    expanded_ids = ids.unsqueeze(0).expand(document_count, -1).clone()
    expanded_mapping = mapping.unsqueeze(0).expand(document_count, -1).clone()
    blocked_keys = expanded_mapping.eq(blocked.unsqueeze(1))
    if bool((blocked_keys.sum(dim=1) <= 0).any()):
        raise RuntimeError("Every document slot must map to at least one token")
    attention_mask = torch.ones_like(expanded_ids, dtype=torch.long)
    if compact_positions:
        kept = ~blocked_keys
        position_ids = kept.long().cumsum(dim=1) - 1
        position_ids.clamp_min_(0)
    else:
        position_ids = torch.arange(ids.numel(), dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(document_count, -1).clone()
    return {
        "input_ids": expanded_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_document_ids": expanded_mapping,
        "blocked_document_ids": blocked,
    }


def choice_logits_for_plain_batch(
    model: Any,
    batch: dict[str, torch.Tensor],
    choice_token_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids).cpu()
    del outputs
    return logits


def choice_logits_for_exact_document_masks(
    model: Any,
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    choice_token_ids: torch.Tensor,
    *,
    compact_positions: bool,
    document_count: int,
    device: torch.device,
) -> torch.Tensor:
    batch = build_exact_document_mask_batch(
        input_ids,
        token_document_ids,
        compact_positions=compact_positions,
        document_count=document_count,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            semantic_token_document_ids=batch["token_document_ids"].to(device),
            semantic_blocked_document_ids=batch["blocked_document_ids"].to(device),
            semantic_document_block_layer_start=0,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids).cpu()
    del outputs
    return logits


def condition_record(
    logits: torch.Tensor,
    gold_index: int,
    full_margin: float,
) -> dict[str, Any]:
    margin = gold_margin(logits, gold_index)
    return {
        "choice_logits": [float(value) for value in logits.tolist()],
        "prediction": CHOICES[int(logits.argmax())],
        "gold_margin": margin,
        "contribution": full_margin - margin,
    }


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
        attention_scope="rationale_wide",
        document_count=args.document_count,
    )
    physical_logits = choice_logits_for_plain_batch(
        model,
        physical_batch,
        choice_token_ids,
        device,
    )
    original_mask_logits = choice_logits_for_exact_document_masks(
        model,
        input_ids,
        mapping,
        choice_token_ids,
        compact_positions=False,
        document_count=args.document_count,
        device=device,
    )
    compact_mask_logits = choice_logits_for_exact_document_masks(
        model,
        input_ids,
        mapping,
        choice_token_ids,
        compact_positions=True,
        document_count=args.document_count,
        device=device,
    )
    legacy_biases = torch.zeros((args.document_count, args.document_count), dtype=torch.float32)
    legacy_biases[torch.arange(args.document_count), torch.arange(args.document_count)] = (
        args.legacy_zero_log_bias
    )
    legacy_logits = choice_logits_for_gate_variants(
        model,
        input_ids,
        mapping,
        assistant_start,
        legacy_biases,
        choice_token_ids,
        attention_scope="rationale_wide",
        semantic_layer_start=args.legacy_layer_start,
        device=device,
    )

    gold_index = int(payload["gold_options"][row_index])
    full_logits = physical_logits[0]
    full_margin = gold_margin(full_logits, gold_index)
    documents: list[dict[str, Any]] = []
    per_question_effects: dict[str, list[float]] = {
        "physical_delete": [],
        **{condition: [] for condition in CONDITIONS},
    }
    for document_index in range(args.document_count):
        physical = condition_record(
            physical_logits[document_index + 1],
            gold_index,
            full_margin,
        )
        conditions = {
            "all_layer_original_position": condition_record(
                original_mask_logits[document_index], gold_index, full_margin
            ),
            "all_layer_compact_position": condition_record(
                compact_mask_logits[document_index], gold_index, full_margin
            ),
            "legacy_layer16_assistant_bias": condition_record(
                legacy_logits[document_index], gold_index, full_margin
            ),
        }
        per_question_effects["physical_delete"].append(float(physical["contribution"]))
        comparisons: dict[str, dict[str, Any]] = {}
        physical_probability = torch.softmax(physical_logits[document_index + 1], dim=-1)
        for condition, record in conditions.items():
            condition_logits = {
                "all_layer_original_position": original_mask_logits,
                "all_layer_compact_position": compact_mask_logits,
                "legacy_layer16_assistant_bias": legacy_logits,
            }[condition][document_index]
            per_question_effects[condition].append(float(record["contribution"]))
            comparisons[condition] = {
                "prediction_agreement": record["prediction"] == physical["prediction"],
                "margin_absolute_error": abs(
                    float(record["gold_margin"]) - float(physical["gold_margin"])
                ),
                "choice_distribution_jsd": jensen_shannon_divergence(
                    physical_probability,
                    torch.softmax(condition_logits, dim=-1),
                ),
            }
        documents.append(
            {
                "document_index": document_index,
                "pair_id": str(payload["pair_ids"][row_index][document_index]),
                "semantic_class_id": int(
                    payload["semantic_class_ids"][row_index, document_index]
                ),
                "physical_delete": physical,
                "conditions": conditions,
                "comparisons_to_physical_delete": comparisons,
            }
        )
    per_question_spearman = {
        condition: spearman_correlation(
            per_question_effects["physical_delete"],
            per_question_effects[condition],
        )
        for condition in CONDITIONS
    }
    return {
        "run_version": RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "sample_id": str(payload["sample_ids"][row_index]),
        "full_choice_logits": [float(value) for value in full_logits.tolist()],
        "full_prediction": CHOICES[int(full_logits.argmax())],
        "full_gold_margin": full_margin,
        "per_question_contribution_spearman": per_question_spearman,
        "documents": documents,
    }


def summarize_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    documents = [document for row in rows for document in row["documents"]]
    physical_effects = [
        float(document["physical_delete"]["contribution"]) for document in documents
    ]
    condition_effects = [
        float(document["conditions"][condition]["contribution"]) for document in documents
    ]
    nonzero_pairs = [
        (physical, alternative)
        for physical, alternative in zip(physical_effects, condition_effects, strict=True)
        if abs(physical) > 1e-12 and abs(alternative) > 1e-12
    ]
    sign_agreement = (
        statistics.fmean((left > 0) == (right > 0) for left, right in nonzero_pairs)
        if nonzero_pairs
        else None
    )
    return {
        "prediction_agreement": finite_mean(
            float(document["comparisons_to_physical_delete"][condition]["prediction_agreement"])
            for document in documents
        ),
        "gold_margin_mae": finite_mean(
            document["comparisons_to_physical_delete"][condition]["margin_absolute_error"]
            for document in documents
        ),
        "choice_distribution_mean_jsd": finite_mean(
            document["comparisons_to_physical_delete"][condition]["choice_distribution_jsd"]
            for document in documents
        ),
        "median_per_question_contribution_spearman": finite_median(
            row["per_question_contribution_spearman"][condition] for row in rows
        ),
        "global_contribution_pearson": pearson_correlation(
            physical_effects,
            condition_effects,
        ),
        "global_contribution_spearman": spearman_correlation(
            physical_effects,
            condition_effects,
        ),
        "nonzero_contribution_sign_agreement": sign_agreement,
        "condition_exact_zero_effect_rate": statistics.fmean(
            abs(value) <= 1e-12 for value in condition_effects
        ),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    conditions = {condition: summarize_condition(rows, condition) for condition in CONDITIONS}
    compact = conditions["all_layer_compact_position"]
    original = conditions["all_layer_original_position"]
    criteria = {
        "compact_mask_is_deletion_equivalent": bool(
            (compact["prediction_agreement"] or 0.0) >= 0.99
            and (compact["gold_margin_mae"] or math.inf) <= 0.15
            and (compact["choice_distribution_mean_jsd"] or math.inf) <= 0.001
            and (compact["median_per_question_contribution_spearman"] or -1.0) >= 0.90
        ),
        "original_position_mask_is_faithful_content_block": bool(
            (original["prediction_agreement"] or 0.0) >= 0.95
            and (original["choice_distribution_mean_jsd"] or math.inf) <= 0.01
            and (original["median_per_question_contribution_spearman"] or -1.0) >= 0.70
        ),
    }
    return {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "questions": len(rows),
        "documents": sum(len(row["documents"]) for row in rows),
        "conditions": conditions,
        "go_criteria": criteria,
        "interpretation": (
            "Exact all-layer masking with compact positions reproduces physical deletion; "
            "a separate continuous-gate monotonicity pilot is now justified."
            if criteria["compact_mask_is_deletion_equivalent"]
            else "Exact all-layer masking did not reproduce physical deletion closely enough; "
            "debug the masking/position contract before any continuous-gate training."
        ),
        "scope_limit": (
            "This test validates only g=0 document exclusion. It does not establish that "
            "intermediate gate values represent proportional document use."
        ),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    def number(value: Any, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    labels = {
        "all_layer_original_position": "All-layer exact mask, original positions",
        "all_layer_compact_position": "All-layer exact mask, compact positions",
        "legacy_layer16_assistant_bias": "Legacy layer-16 assistant-only -20 bias",
    }
    lines = [
        "# All-layer document-mask contract",
        "",
        f"- Questions: {summary['questions']}",
        f"- Documents: {summary['documents']}",
        "- Reference: physical removal of the mapped document tokens",
        "",
        "| Condition | Answer agreement | Margin MAE | Mean JSD | "
        "Median within-question Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary["conditions"][condition]
        lines.append(
            f"| {labels[condition]} | {number(row['prediction_agreement'])} | "
            f"{number(row['gold_margin_mae'])} | "
            f"{number(row['choice_distribution_mean_jsd'], 6)} | "
            f"{number(row['median_per_question_contribution_spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Compact mask deletion-equivalent: "
            f"{summary['go_criteria']['compact_mask_is_deletion_equivalent']}",
            f"- Original-position content block faithful: "
            f"{summary['go_criteria']['original_position_mask_is_faithful_content_block']}",
            f"- Interpretation: {summary['interpretation']}",
            f"- Scope: {summary['scope_limit']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.max_samples < 0 or args.document_count <= 0:
        raise ValueError("Invalid sample or document count")
    if not 0 <= args.legacy_layer_start < 32:
        raise ValueError("legacy-layer-start must be in [0, 31]")
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
    detail_path = args.output_dir / "mask_contract_details.jsonl"
    summary_path = args.output_dir / "mask_contract_summary.json"
    report_path = args.output_dir / "mask_contract_report.md"
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "feature_contract_fingerprint": fingerprint,
        "llm_model_bundle": model_bundle_identity(args.llm_model),
        "selected_ids": selected_ids,
        "document_count": args.document_count,
        "legacy_layer_start": args.legacy_layer_start,
        "legacy_zero_log_bias": args.legacy_zero_log_bias,
        "dtype": args.dtype,
    }
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Mask-contract resume mismatch; use a new output directory")
    else:
        if detail_path.exists() and not args.resume:
            detail_path.unlink()
        atomic_json(contract_path, contract)
    existing = load_details(detail_path) if args.resume else {}
    selected_set = set(selected_ids)
    rows = {sample_id: row for sample_id, row in existing.items() if sample_id in selected_set}
    remaining = selected_set - set(rows)
    variants_per_question = 1 + 4 * args.document_count
    logging.info(
        "All-layer mask plan: dataset=%s split=%s questions=%d cached=%d remaining=%d "
        "documents/question=%d forward-variants/question=%d conditions=%s",
        args.dataset,
        args.split,
        len(selected_ids),
        len(rows),
        len(remaining),
        args.document_count,
        variants_per_question,
        ["physical_delete", *CONDITIONS],
    )
    logging.info(
        "Resume contract: durable JSONL every 4 questions; "
        "rerunning the same command resumes safely"
    )
    if args.plan_only:
        return

    progress = PipelineProgress(
        overall_total=len(selected_ids),
        overall_initial=len(rows),
        desc=f"AllLayerMask:{args.dataset}",
    )
    try:
        if remaining:
            attention_name = register_semantic_attention()
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model,
                local_files_only=True,
                use_fast=True,
            )
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
                "1/1 physical deletion vs exact all-layer masks",
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
                if "assistant_query_starts" not in payload:
                    raise RuntimeError("Mask audit requires assistant_query_starts")
                for row_index, sample_id_value in enumerate(payload["sample_ids"]):
                    sample_id = str(sample_id_value)
                    if sample_id not in remaining:
                        continue
                    row = evaluate_sample(
                        args,
                        payload,
                        row_index,
                        model,
                        tokenizer,
                        choice_token_ids,
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
        ordered_rows = [rows[sample_id] for sample_id in selected_ids]
        summary = summarize(ordered_rows, args)
        atomic_json(summary_path, summary)
        atomic_text(report_path, report_markdown(summary))
    except BaseException:
        logging.exception(
            "All-layer mask audit failed: completed=%d remaining=%d durable_cache=%s "
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
        "All-layer mask audit complete: questions=%d summary=%s report=%s",
        len(selected_ids),
        summary_path,
        report_path,
    )


if __name__ == "__main__":
    main()

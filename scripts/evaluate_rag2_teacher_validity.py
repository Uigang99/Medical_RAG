#!/usr/bin/env python3
"""Compare fixed, regenerated, direct-choice, and one-forward document influence.

The regenerated-rationale leave-one-document-out (LOO) run is treated as the
end-to-end reference for the anchored rationale pipeline.  The audit compares:

* the existing fixed-rationale LOO teacher;
* direct-choice LOO with no generated rationale;
* direct-choice final-query attention mass;
* direct-choice attention-weighted value-vector norm.

All comparisons use the same Top-8 document identities and order.  The script
does not train a predictor.  It is a bounded construct-validity test with
atomic caches, active-stage progress, and ETA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
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

from medrag.core import BenchmarkSample  # noqa: E402
from medrag.filtering.rag2_preanswer_text_hidden import (  # noqa: E402
    FINAL_ANSWER_PREFILL,
    build_preanswer_user_prompt,
)
from medrag.generation.semantic_attention import (  # noqa: E402
    DocumentAttentionCollector,
    register_semantic_attention,
)
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    END_REASONING_MARKER,
    FINAL_ANSWER_PREFIX,
    RATIONALE_HEADER,
    build_anchored_user_prompt,
    render_chat_prompt,
)
from medrag.training.rag2_semantic_attention_data import (  # noqa: E402
    RAG2SemanticAttentionDataset,
    SemanticAttentionQuestion,
)
from scripts.evaluate_rag2_semantic_gate_fidelity import (  # noqa: E402
    jensen_shannon_divergence,
    normalize_positive,
    pearson_correlation,
    spearman_correlation,
)


RUN_VERSION = "rag2_teacher_construct_validity_v1"
DIRECT_ROW_VERSION = "rag2_teacher_validity_direct_choice_proxy_v1"
E2E_SCORE_ROW_VERSION = "rag2_teacher_validity_regenerated_rationale_exact_choice_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
SIGNAL_THRESHOLDS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
MIN_REFERENCE_COVERAGE = 0.50
MIN_FIXED_MEDIAN_SPEARMAN = 0.60
MIN_FIXED_TOP1_AGREEMENT = 0.60
MAX_REFERENCE_REPEAT_NOISE_JSD = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medqa",), default="medqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--fixed-teacher-dir", type=Path, required=True)
    parser.add_argument("--end-to-end-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-reference-jsd", type=float, default=1e-4)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def hashed_path(root: Path, sample_id: str, suffix: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return root / f"{digest}.{suffix}"


def logprobs_to_probabilities(values: dict[str, float | None]) -> list[float]:
    logits: list[float] = []
    for choice in CHOICES:
        value = values.get(choice)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Missing finite choice logprob for {choice}: {values}")
        logits.append(float(value))
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def distribution_metrics(
    full: list[float],
    repeat: list[float],
    removals: list[list[float]],
) -> dict[str, Any]:
    if len(full) != 4 or len(repeat) != 4 or len(removals) != 8:
        raise ValueError("Expected full/repeat four-choice distributions and eight removals")
    full_tensor = torch.tensor(full)
    jsd = [
        jensen_shannon_divergence(full_tensor, torch.tensor(probabilities))
        for probabilities in removals
    ]
    full_prediction = max(range(4), key=full.__getitem__)
    removal_predictions = [max(range(4), key=value.__getitem__) for value in removals]
    return {
        "jsd": jsd,
        "total_jsd": sum(jsd),
        "repeat_noise_jsd": jensen_shannon_divergence(full_tensor, torch.tensor(repeat)),
        "full_probabilities": full,
        "removal_probabilities": removals,
        "full_prediction": full_prediction,
        "removal_predictions": removal_predictions,
        "flips": [value != full_prediction for value in removal_predictions],
    }


def e2e_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Parse complete vLLM choice rows; retained for cache/unit diagnostics.

    The construct-validity report uses exact Hugging Face rescoring instead,
    because vLLM top-logprobs can omit a very-low-probability allowed choice.
    """

    variants = {str(value["variant"]): value for value in row["variants"]}
    expected = {"full", "repeat", *(f"remove_{index}" for index in range(8))}
    if set(variants) != expected:
        raise ValueError(f"Invalid regenerated variants for {row.get('sample_id')}")
    full = logprobs_to_probabilities(variants["full"]["choice_logprobs"])
    repeat = logprobs_to_probabilities(variants["repeat"]["choice_logprobs"])
    removals = [
        logprobs_to_probabilities(variants[f"remove_{index}"]["choice_logprobs"])
        for index in range(8)
    ]
    return distribution_metrics(full, repeat, removals)


def benchmark_sample(question: SemanticAttentionQuestion) -> BenchmarkSample:
    answer = question.gold_answers[0] if question.gold_answers else None
    raw = {
        "question": question.question,
        "options": dict(question.options),
        "answer": answer,
    }
    return BenchmarkSample(
        row_idx=int(question.row_idx or 0),
        id=question.sample_id,
        task="mcq",
        collection=question.dataset,
        dataset=question.dataset,
        split=question.split,
        question=question.question,
        options=dict(question.options),
        answer=answer,
        answers=[answer] if answer else [],
        raw=raw,
    )


def document_texts(question: SemanticAttentionQuestion) -> list[str]:
    return [" ".join(document.text.split()) for document in question.documents]


def encode_direct_variant(
    tokenizer: Any,
    sample: BenchmarkSample,
    texts: list[str],
    active_indices: list[int],
    marker_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    active_texts = [texts[index] for index in active_indices]
    context = "\n\n".join(active_texts)
    user_prompt = build_preanswer_user_prompt(sample, context)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]] + marker_ids
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    mapping = [-1] * len(offsets)
    context_start = rendered.find(context)
    if context_start < 0:
        raise RuntimeError(f"Direct-choice context is absent from rendered prompt: {sample.id}")
    cursor = context_start
    for original_index, text in zip(active_indices, active_texts, strict=True):
        start = rendered.find(text, cursor)
        if start < 0:
            raise RuntimeError(f"Document text span is absent from rendered prompt: {sample.id}")
        end = start + len(text)
        for token_index, (left, right) in enumerate(offsets):
            if right > start and left < end:
                mapping[token_index] = original_index
        cursor = end
    mapping.extend([-1] * len(marker_ids))
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mapping, dtype=torch.long)


def build_direct_batch(
    tokenizer: Any,
    question: SemanticAttentionQuestion,
    marker_ids: list[int],
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    sample = benchmark_sample(question)
    texts = document_texts(question)
    variants: list[tuple[torch.Tensor, torch.Tensor]] = []
    complete = list(range(8))
    variants.append(encode_direct_variant(tokenizer, sample, texts, complete, marker_ids))
    variants.append(encode_direct_variant(tokenizer, sample, texts, complete, marker_ids))
    for removed in range(8):
        active = [index for index in range(8) if index != removed]
        variants.append(encode_direct_variant(tokenizer, sample, texts, active, marker_ids))
    maximum = max(int(ids.numel()) for ids, _ in variants)
    if maximum > max_input_tokens:
        raise RuntimeError(
            f"Direct-choice validity prompt exceeds {max_input_tokens} tokens for "
            f"{question.sample_id}: {maximum}"
        )
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((10, maximum), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((10, maximum), dtype=torch.long)
    token_document_ids = torch.full((10, maximum), -1, dtype=torch.long)
    query_mask = torch.zeros((10, maximum), dtype=torch.float32)
    for row_index, (ids, mapping) in enumerate(variants):
        length = int(ids.numel())
        left = maximum - length
        input_ids[row_index, left:] = ids
        attention_mask[row_index, left:] = 1
        token_document_ids[row_index, left:] = mapping
        query_mask[row_index, -1] = 1.0
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_document_ids": token_document_ids,
        "query_mask": query_mask,
    }


def build_e2e_choice_batch(
    tokenizer: Any,
    question: SemanticAttentionQuestion,
    e2e_row: dict[str, Any],
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    variants = {str(value["variant"]): value for value in e2e_row["variants"]}
    variant_names = ["full", "repeat"] + [f"remove_{index}" for index in range(8)]
    if set(variants) != set(variant_names):
        raise ValueError(f"Invalid regenerated variants for {question.sample_id}")
    texts = document_texts(question)
    prompt_row = {
        "question": question.question,
        "options": dict(question.options),
        "answer": question.gold_answers[0] if question.gold_answers else None,
    }
    encoded_variants: list[torch.Tensor] = []
    for name in variant_names:
        if name in {"full", "repeat"}:
            active = list(range(8))
        else:
            removed = int(name.split("_", 1)[1])
            active = [index for index in range(8) if index != removed]
        context = "\n\n".join(texts[index] for index in active)
        user_prompt = build_anchored_user_prompt(prompt_row, context)
        rationale = str(variants[name]["rationale"]).strip()
        decision_prompt = (
            render_chat_prompt(tokenizer, user_prompt)
            + RATIONALE_HEADER
            + rationale
            + "\n"
            + END_REASONING_MARKER
            + "\n"
            + FINAL_ANSWER_PREFIX
        )
        token_ids = tokenizer.encode(decision_prompt, add_special_tokens=False)
        encoded_variants.append(torch.tensor(token_ids, dtype=torch.long))
    maximum = max(int(ids.numel()) for ids in encoded_variants)
    if maximum > max_input_tokens:
        raise RuntimeError(
            f"Regenerated-rationale validity prompt exceeds {max_input_tokens} tokens for "
            f"{question.sample_id}: {maximum}"
        )
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((10, maximum), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((10, maximum), dtype=torch.long)
    for row_index, ids in enumerate(encoded_variants):
        length = int(ids.numel())
        input_ids[row_index, maximum - length :] = ids
        attention_mask[row_index, maximum - length :] = 1
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def direct_row_path(output_dir: Path, sample_id: str) -> Path:
    return hashed_path(output_dir / "direct_rows", sample_id, "pt")


def e2e_score_row_path(output_dir: Path, sample_id: str) -> Path:
    return hashed_path(output_dir / "e2e_exact_choice_rows", sample_id, "pt")


def valid_direct_row(path: Path, sample_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        row.get("run_version") == DIRECT_ROW_VERSION
        and row.get("sample_id") == sample_id
        and row.get("contract_fingerprint") == fingerprint
    )


def valid_e2e_score_row(path: Path, sample_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        row.get("run_version") == E2E_SCORE_ROW_VERSION
        and row.get("sample_id") == sample_id
        and row.get("contract_fingerprint") == fingerprint
    )


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def evaluate_direct_one(
    question: SemanticAttentionQuestion,
    model: Any,
    tokenizer: Any,
    marker_ids: list[int],
    choice_ids: torch.Tensor,
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, Any]:
    batch = build_direct_batch(tokenizer, question, marker_ids, args.max_input_tokens)
    collector = DocumentAttentionCollector(document_count=8, collect_value_norm=True)
    device = torch.device(args.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            semantic_query_mask=batch["query_mask"].to(device),
            semantic_token_document_ids=batch["token_document_ids"].to(device),
            semantic_attention_collector=collector,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_ids).cpu()
    del outputs
    probabilities = torch.softmax(logits, dim=-1)
    full = probabilities[0]
    repeat = probabilities[1]
    removals = probabilities[2:]
    jsd = [jensen_shannon_divergence(full, removals[index]) for index in range(8)]
    full_prediction = int(full.argmax().item())
    removal_predictions = [int(value) for value in removals.argmax(dim=-1).tolist()]
    all_layers = collector.summarize()
    late_layers = collector.summarize(layer_start=20)
    return {
        "run_version": DIRECT_ROW_VERSION,
        "contract_fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": question.sample_id,
        "pair_ids": [document.pair_id for document in question.documents],
        "direct_jsd": torch.tensor(jsd, dtype=torch.float32),
        "direct_total_jsd": float(sum(jsd)),
        "direct_repeat_noise_jsd": jensen_shannon_divergence(full, repeat),
        "direct_full_probabilities": full,
        "direct_loo_probabilities": removals,
        "direct_full_prediction": full_prediction,
        "direct_removal_predictions": torch.tensor(removal_predictions, dtype=torch.int8),
        "direct_flips": torch.tensor(
            [value != full_prediction for value in removal_predictions], dtype=torch.bool
        ),
        "attention_share_all": all_layers["document_share"][0],
        "attention_share_late": late_layers["document_share"][0],
        "value_share_all": all_layers["document_value_share"][0],
        "value_share_late": late_layers["document_value_share"][0],
        "attention_layers_all": all_layers["layers"],
        "attention_layers_late": late_layers["layers"],
    }


def evaluate_e2e_exact_one(
    question: SemanticAttentionQuestion,
    e2e_row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    choice_ids: torch.Tensor,
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, Any]:
    batch = build_e2e_choice_batch(tokenizer, question, e2e_row, args.max_input_tokens)
    device = torch.device(args.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_ids).cpu()
    probabilities = torch.softmax(logits, dim=-1)
    metrics = distribution_metrics(
        probabilities[0].tolist(),
        probabilities[1].tolist(),
        probabilities[2:].tolist(),
    )
    return {
        "run_version": E2E_SCORE_ROW_VERSION,
        "contract_fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": question.sample_id,
        "pair_ids": [document.pair_id for document in question.documents],
        "exact_choice_probabilities": probabilities,
        "exact_jsd": torch.tensor(metrics["jsd"], dtype=torch.float32),
        "exact_total_jsd": float(metrics["total_jsd"]),
        "exact_repeat_noise_jsd": float(metrics["repeat_noise_jsd"]),
        "exact_full_prediction": int(metrics["full_prediction"]),
        "exact_removal_predictions": torch.tensor(
            metrics["removal_predictions"], dtype=torch.int8
        ),
        "exact_flips": torch.tensor(metrics["flips"], dtype=torch.bool),
    }


def as_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    return [float(item) for item in value]


def fixed_metrics(row: dict[str, Any]) -> dict[str, Any]:
    full = row["full_choice_probabilities"]
    removals = row["loo_choice_probabilities"]
    full_prediction = int(full.argmax().item())
    removal_predictions = [int(value) for value in removals.argmax(dim=-1).tolist()]
    return {
        "jsd": as_float_list(row["loo_jsd"]),
        "total_jsd": float(row["total_loo_jsd"]),
        "repeat_noise_jsd": float(row["repeat_noise_jsd"]),
        "full_prediction": full_prediction,
        "removal_predictions": removal_predictions,
        "flips": [value != full_prediction for value in removal_predictions],
    }


def binary_metrics(predicted: list[bool], reference: list[bool]) -> dict[str, float | int]:
    tp = sum(pred and ref for pred, ref in zip(predicted, reference, strict=True))
    fp = sum(pred and not ref for pred, ref in zip(predicted, reference, strict=True))
    fn = sum(not pred and ref for pred, ref in zip(predicted, reference, strict=True))
    tn = sum(not pred and not ref for pred, ref in zip(predicted, reference, strict=True))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def comparison_metrics(
    rows: list[dict[str, Any]],
    candidate_key: str,
    reference_key: str,
    threshold: float,
) -> dict[str, Any]:
    per_spearman: list[float] = []
    per_pearson: list[float] = []
    top1 = 0
    top1_questions = 0
    share_errors: list[float] = []
    selected = 0
    for row in rows:
        reference = [float(value) for value in row[reference_key]]
        candidate = [float(value) for value in row[candidate_key]]
        if len(reference) != len(candidate) or not reference:
            raise ValueError(
                f"Candidate/reference length mismatch for {candidate_key} vs {reference_key}"
            )
        if sum(reference) <= threshold:
            continue
        selected += 1
        spearman = spearman_correlation(candidate, reference)
        pearson = pearson_correlation(candidate, reference)
        if spearman is not None:
            per_spearman.append(float(spearman))
        if pearson is not None:
            per_pearson.append(float(pearson))
        reference_share = normalize_positive(reference)
        candidate_share = normalize_positive(candidate)
        if reference_share is not None and candidate_share is not None:
            ordered_reference = sorted(reference_share, reverse=True)
            if ordered_reference[0] - ordered_reference[1] >= 0.05:
                top1_questions += 1
                candidate_has_order = max(candidate_share) - min(candidate_share) > 1e-12
                top1 += int(
                    candidate_has_order
                    and max(range(len(candidate)), key=candidate.__getitem__)
                    == max(range(len(reference)), key=reference.__getitem__)
                )
            share_errors.extend(
                abs(left - right)
                for left, right in zip(candidate_share, reference_share, strict=True)
            )
    return {
        "questions": selected,
        "spearman_questions": len(per_spearman),
        "mean_spearman": statistics.fmean(per_spearman) if per_spearman else None,
        "median_spearman": statistics.median(per_spearman) if per_spearman else None,
        "mean_pearson": statistics.fmean(per_pearson) if per_pearson else None,
        "top1_questions": top1_questions,
        "top1_agreement": top1 / top1_questions if top1_questions else None,
        "share_mae": statistics.fmean(share_errors) if share_errors else None,
    }


def build_summary(rows: list[dict[str, Any]], minimum_reference_jsd: float) -> dict[str, Any]:
    comparisons = {
        "fixed_vs_end_to_end": ("fixed_jsd", "end_to_end_jsd"),
        "direct_vs_end_to_end": ("direct_jsd", "end_to_end_jsd"),
        "attention_all_vs_end_to_end": ("attention_share_all", "end_to_end_jsd"),
        "attention_late_vs_end_to_end": ("attention_share_late", "end_to_end_jsd"),
        "value_all_vs_end_to_end": ("value_share_all", "end_to_end_jsd"),
        "value_late_vs_end_to_end": ("value_share_late", "end_to_end_jsd"),
        "attention_late_vs_direct": ("attention_share_late", "direct_jsd"),
        "value_late_vs_direct": ("value_share_late", "direct_jsd"),
    }
    by_threshold: dict[str, Any] = {}
    for threshold in SIGNAL_THRESHOLDS:
        by_threshold[f"{threshold:.0e}"] = {
            name: comparison_metrics(rows, candidate, reference, threshold)
            for name, (candidate, reference) in comparisons.items()
        }
    e2e_flips = [bool(value) for row in rows for value in row["end_to_end_flips"]]
    fixed_flips = [bool(value) for row in rows for value in row["fixed_flips"]]
    direct_flips = [bool(value) for row in rows for value in row["direct_flips"]]
    signal = {
        name: [float(row[f"{name}_total_jsd"]) for row in rows]
        for name in ("end_to_end", "fixed", "direct")
    }
    primary = {
        name: comparison_metrics(rows, candidate, reference, minimum_reference_jsd)
        for name, (candidate, reference) in comparisons.items()
    }
    reference_coverage = sum(
        float(row["end_to_end_total_jsd"]) > minimum_reference_jsd for row in rows
    ) / max(1, len(rows))
    reference_repeat_max = max(
        float(row["end_to_end_repeat_noise_jsd"]) for row in rows
    )
    fixed_primary = primary["fixed_vs_end_to_end"]
    validity_checks = {
        "reference_coverage": reference_coverage >= MIN_REFERENCE_COVERAGE,
        "reference_repeat_noise": reference_repeat_max <= MAX_REFERENCE_REPEAT_NOISE_JSD,
        "fixed_median_spearman": (
            fixed_primary["median_spearman"] is not None
            and float(fixed_primary["median_spearman"]) >= MIN_FIXED_MEDIAN_SPEARMAN
        ),
        "fixed_top1_agreement": (
            fixed_primary["top1_agreement"] is not None
            and float(fixed_primary["top1_agreement"]) >= MIN_FIXED_TOP1_AGREEMENT
        ),
    }
    return {
        "run_version": RUN_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(rows),
        "documents": len(rows) * 8,
        "primary_minimum_reference_jsd": minimum_reference_jsd,
        "signal": {
            name: {
                "median_total_jsd": statistics.median(values),
                "mean_total_jsd": statistics.fmean(values),
                "coverage_above_threshold": {
                    f"{threshold:.0e}": sum(value > threshold for value in values) / len(values)
                    for threshold in SIGNAL_THRESHOLDS
                },
            }
            for name, values in signal.items()
        },
        "repeat_noise": {
            name: {
                "median_jsd": statistics.median(
                    [float(row[f"{name}_repeat_noise_jsd"]) for row in rows]
                ),
                "maximum_jsd": max(float(row[f"{name}_repeat_noise_jsd"]) for row in rows),
            }
            for name in ("end_to_end", "fixed", "direct")
        },
        "comparisons_by_reference_threshold": by_threshold,
        "primary_comparisons": primary,
        "fixed_teacher_exploratory_verdict": {
            "decision": "PROCEED_TO_FRESH_CONFIRMATION" if all(validity_checks.values()) else "STOP",
            "checks": validity_checks,
            "criteria": {
                "minimum_reference_coverage": MIN_REFERENCE_COVERAGE,
                "maximum_reference_repeat_noise_jsd": MAX_REFERENCE_REPEAT_NOISE_JSD,
                "minimum_fixed_median_spearman": MIN_FIXED_MEDIAN_SPEARMAN,
                "minimum_fixed_top1_agreement": MIN_FIXED_TOP1_AGREEMENT,
            },
            "note": (
                "Passing this exploratory 128-question audit permits only a fresh held-out "
                "confirmation; it does not by itself validate predictor training."
            ),
        },
        "answer_flip_against_end_to_end": {
            "fixed": binary_metrics(fixed_flips, e2e_flips),
            "direct": binary_metrics(direct_flips, e2e_flips),
            "end_to_end_flip_rate": sum(e2e_flips) / max(1, len(e2e_flips)),
        },
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    primary = summary["primary_comparisons"]
    verdict = summary["fixed_teacher_exploratory_verdict"]
    lines = [
        "# Teacher construct-validity audit",
        "",
        f"- Questions: {summary['questions']}",
        f"- Primary end-to-end signal threshold: `{summary['primary_minimum_reference_jsd']:.1e}`",
        "- End-to-end reference: rationale and answer regenerated after each physical document removal",
        f"- Exploratory fixed-teacher verdict: **{verdict['decision']}**",
        "- A pass requires a separate run on the untouched confirmation cohort.",
        "",
        "| Candidate | Reference | N | Top-1 N | Median Spearman | Top-1 | Share MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    display = {
        "fixed_vs_end_to_end": ("Fixed-rationale LOO", "End-to-end LOO"),
        "direct_vs_end_to_end": ("Direct-choice LOO", "End-to-end LOO"),
        "attention_all_vs_end_to_end": ("Attention mass (all)", "End-to-end LOO"),
        "attention_late_vs_end_to_end": ("Attention mass (L20+)", "End-to-end LOO"),
        "value_all_vs_end_to_end": ("Attention×value norm (all)", "End-to-end LOO"),
        "value_late_vs_end_to_end": ("Attention×value norm (L20+)", "End-to-end LOO"),
        "attention_late_vs_direct": ("Attention mass (L20+)", "Direct-choice LOO"),
        "value_late_vs_direct": ("Attention×value norm (L20+)", "Direct-choice LOO"),
    }
    for key, (candidate, reference) in display.items():
        values = primary[key]
        lines.append(
            f"| {candidate} | {reference} | {values['questions']} | "
            f"{values['top1_questions']} | "
            f"{number(values['median_spearman'])} | {percent(values['top1_agreement'])} | "
            f"{number(values['share_mae'])} |"
        )
    flips = summary["answer_flip_against_end_to_end"]
    checks = verdict["checks"]
    lines.extend(
        [
            "",
            "## Pre-declared exploratory checks",
            "",
            f"- Reference signal coverage >= 50%: {'PASS' if checks['reference_coverage'] else 'FAIL'}",
            f"- Reference repeat-noise JSD <= 1e-8: {'PASS' if checks['reference_repeat_noise'] else 'FAIL'}",
            f"- Fixed-rationale median Spearman >= 0.60: {'PASS' if checks['fixed_median_spearman'] else 'FAIL'}",
            f"- Fixed-rationale Top-1 agreement >= 0.60: {'PASS' if checks['fixed_top1_agreement'] else 'FAIL'}",
            "",
            "## Answer-flip agreement with regenerated-rationale LOO",
            "",
            f"- End-to-end removal flip rate: {percent(flips['end_to_end_flip_rate'])}",
            f"- Fixed-rationale: accuracy={percent(flips['fixed']['accuracy'])}, "
            f"precision={percent(flips['fixed']['precision'])}, recall={percent(flips['fixed']['recall'])}",
            f"- Direct-choice: accuracy={percent(flips['direct']['accuracy'])}, "
            f"precision={percent(flips['direct']['precision'])}, recall={percent(flips['direct']['recall'])}",
            "",
            "Attention and attention×value are diagnostic proxies, not causal explanations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.minimum_reference_jsd < 0 or args.max_input_tokens <= 0:
        raise ValueError("Signal threshold must be non-negative and token limit positive")
    for path in (args.fixed_teacher_dir, args.end_to_end_dir, args.index_path, args.llm_model):
        if not path.exists():
            raise FileNotFoundError(path)
    e2e_manifest_path = args.end_to_end_dir / "generation_manifest.json"
    fixed_manifest_path = args.fixed_teacher_dir / "preparation_manifest.json"
    if not e2e_manifest_path.is_file() or not fixed_manifest_path.is_file():
        raise FileNotFoundError("Completed end-to-end and fixed teacher manifests are required")
    e2e_manifest = json.loads(e2e_manifest_path.read_text(encoding="utf-8"))
    selected_ids = [str(value) for value in e2e_manifest["selected_sample_ids"]]
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("End-to-end manifest contains invalid selected IDs")

    dataset = RAG2SemanticAttentionDataset(args.index_path, args.split)
    indexed: dict[str, SemanticAttentionQuestion] = {}
    try:
        for index in range(len(dataset)):
            question = dataset[index]
            if question.sample_id in selected_ids:
                indexed[question.sample_id] = question
    finally:
        dataset.close()
    missing = sorted(set(selected_ids) - set(indexed))
    if missing:
        raise RuntimeError(f"Selected IDs are missing from grouped index: {missing[:5]}")

    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "fixed_teacher_dir": str(args.fixed_teacher_dir.resolve()),
        "fixed_teacher_manifest_sha256": sha256_file(fixed_manifest_path),
        "end_to_end_dir": str(args.end_to_end_dir.resolve()),
        "end_to_end_manifest_sha256": sha256_file(e2e_manifest_path),
        "index_path": str(args.index_path.resolve()),
        "llm_model": str(args.llm_model.resolve()),
        "selected_sample_ids": selected_ids,
        "direct_prompt": "rag2_fixed_direct_choice_context_v1",
        "max_input_tokens": args.max_input_tokens,
        "dtype": args.dtype,
        "minimum_reference_jsd": args.minimum_reference_jsd,
    }
    fingerprint = canonical_hash(run_contract)
    run_contract["contract_fingerprint"] = fingerprint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError("Teacher-validity resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(contract_path, run_contract)

    direct_cached_ids = {
        sample_id
        for sample_id in selected_ids
        if valid_direct_row(direct_row_path(args.output_dir, sample_id), sample_id, fingerprint)
    }
    e2e_cached_ids = {
        sample_id
        for sample_id in selected_ids
        if valid_e2e_score_row(
            e2e_score_row_path(args.output_dir, sample_id), sample_id, fingerprint
        )
    }
    fully_cached_ids = direct_cached_ids & e2e_cached_ids
    pending_ids = [sample_id for sample_id in selected_ids if sample_id not in fully_cached_ids]
    logging.info(
        "Teacher-validity evaluation plan: questions=%d direct_cached=%d "
        "e2e_exact_cached=%d remaining_questions=%d exact_forward_batches=%d",
        len(selected_ids),
        len(direct_cached_ids),
        len(e2e_cached_ids),
        len(pending_ids),
        sum(sample_id not in direct_cached_ids for sample_id in selected_ids)
        + sum(sample_id not in e2e_cached_ids for sample_id in selected_ids),
    )
    if args.plan_only:
        return

    progress = PipelineProgress(
        overall_total=2 * len(selected_ids),
        overall_initial=len(fully_cached_ids),
        desc="TeacherValidityAudit:medqa",
    )
    try:
        progress.set_stage(
            "1/2 exact regenerated-rationale/direct LOO + one-forward proxies",
            total=len(selected_ids),
            initial=len(fully_cached_ids),
        )
        if pending_ids:
            attention_name = register_semantic_attention()
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model, local_files_only=True, use_fast=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            marker_ids = tokenizer.encode(FINAL_ANSWER_PREFILL, add_special_tokens=False)
            choice_token_values: list[int] = []
            for label in CHOICES:
                token_ids = tokenizer.encode(label, add_special_tokens=False)
                if len(token_ids) != 1:
                    raise RuntimeError(f"Choice {label} is not one token: {token_ids}")
                choice_token_values.append(int(token_ids[0]))
            device = torch.device(args.device)
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            logging.info("Loading frozen target Llama for direct-choice validity audit: %s", args.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                local_files_only=True,
                dtype=dtype,
                attn_implementation=attention_name,
            ).to(device)
            model.eval()
            model.requires_grad_(False)
            choice_ids = torch.tensor(choice_token_values, dtype=torch.long, device=device)
            for sample_id in pending_ids:
                if sample_id not in direct_cached_ids:
                    direct = evaluate_direct_one(
                        indexed[sample_id],
                        model,
                        tokenizer,
                        marker_ids,
                        choice_ids,
                        args,
                        fingerprint,
                    )
                    atomic_torch_save(direct_row_path(args.output_dir, sample_id), direct)
                if sample_id not in e2e_cached_ids:
                    raw_e2e_path = hashed_path(args.end_to_end_dir / "rows", sample_id, "json")
                    raw_e2e = json.loads(raw_e2e_path.read_text(encoding="utf-8"))
                    exact_e2e = evaluate_e2e_exact_one(
                        indexed[sample_id],
                        raw_e2e,
                        model,
                        tokenizer,
                        choice_ids,
                        args,
                        fingerprint,
                    )
                    atomic_torch_save(e2e_score_row_path(args.output_dir, sample_id), exact_e2e)
                progress.update(1)
                progress.set_detail(f"sample={sample_id}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        progress.set_stage("2/2 join teachers, aggregate validity metrics, and write report", total=len(selected_ids))
        detail_rows: list[dict[str, Any]] = []
        for sample_id in selected_ids:
            fixed_path = hashed_path(args.fixed_teacher_dir / "rows" / args.split, sample_id, "pt")
            e2e_path = hashed_path(args.end_to_end_dir / "rows", sample_id, "json")
            direct_path = direct_row_path(args.output_dir, sample_id)
            e2e_score_path = e2e_score_row_path(args.output_dir, sample_id)
            if (
                not fixed_path.is_file()
                or not e2e_path.is_file()
                or not direct_path.is_file()
                or not e2e_score_path.is_file()
            ):
                raise FileNotFoundError(f"Incomplete teacher tuple for {sample_id}")
            fixed = torch.load(fixed_path, map_location="cpu", weights_only=False)
            direct = torch.load(direct_path, map_location="cpu", weights_only=False)
            e2e_score = torch.load(e2e_score_path, map_location="cpu", weights_only=False)
            e2e = {
                "jsd": as_float_list(e2e_score["exact_jsd"]),
                "total_jsd": float(e2e_score["exact_total_jsd"]),
                "repeat_noise_jsd": float(e2e_score["exact_repeat_noise_jsd"]),
                "flips": [bool(value) for value in e2e_score["exact_flips"].tolist()],
            }
            fixed_value = fixed_metrics(fixed)
            expected_pairs = [document.pair_id for document in indexed[sample_id].documents]
            if [str(value) for value in fixed["pair_ids"]] != expected_pairs:
                raise RuntimeError(f"Fixed teacher document mismatch for {sample_id}")
            if [str(value) for value in direct["pair_ids"]] != expected_pairs:
                raise RuntimeError(f"Direct teacher document mismatch for {sample_id}")
            if [str(value) for value in e2e_score["pair_ids"]] != expected_pairs:
                raise RuntimeError(f"End-to-end teacher document mismatch for {sample_id}")
            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "pair_ids": expected_pairs,
                    "end_to_end_jsd": e2e["jsd"],
                    "end_to_end_total_jsd": e2e["total_jsd"],
                    "end_to_end_repeat_noise_jsd": e2e["repeat_noise_jsd"],
                    "end_to_end_flips": e2e["flips"],
                    "fixed_jsd": fixed_value["jsd"],
                    "fixed_total_jsd": fixed_value["total_jsd"],
                    "fixed_repeat_noise_jsd": fixed_value["repeat_noise_jsd"],
                    "fixed_flips": fixed_value["flips"],
                    "direct_jsd": as_float_list(direct["direct_jsd"]),
                    "direct_total_jsd": float(direct["direct_total_jsd"]),
                    "direct_repeat_noise_jsd": float(direct["direct_repeat_noise_jsd"]),
                    "direct_flips": [bool(value) for value in direct["direct_flips"].tolist()],
                    "attention_share_all": as_float_list(direct["attention_share_all"]),
                    "attention_share_late": as_float_list(direct["attention_share_late"]),
                    "value_share_all": as_float_list(direct["value_share_all"]),
                    "value_share_late": as_float_list(direct["value_share_late"]),
                }
            )
            progress.update(1)
        summary = build_summary(detail_rows, args.minimum_reference_jsd)
        summary.update(
            {
                "dataset": args.dataset,
                "split": args.split,
                "fixed_teacher_dir": str(args.fixed_teacher_dir.resolve()),
                "end_to_end_dir": str(args.end_to_end_dir.resolve()),
                "llm_model": str(args.llm_model.resolve()),
            }
        )
        atomic_write_jsonl(args.output_dir / "details.jsonl", detail_rows)
        atomic_write_json(args.output_dir / "summary.json", summary)
        write_markdown(summary, args.output_dir / "summary.md")
        logging.info("Teacher construct-validity audit complete: %s", args.output_dir)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

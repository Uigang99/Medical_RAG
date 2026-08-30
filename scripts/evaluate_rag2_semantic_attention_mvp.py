#!/usr/bin/env python3
"""Evaluate suppression-only semantic attention on cached RAG2 MCQ candidates.

The MVP keeps every reranked Top-k document in the prompt.  A frozen semantic
binary Flan-T5 model assigns each question-document pair a Helpful-vs-Not-
Helpful margin.  During anchored rationale and answer generation, negative
margins reduce attention to the corresponding document tokens without removing
those tokens from the context.

The workflow is resumable and reports both overall and active-stage progress
and ETA.  Lambda zero is always evaluated through the same Transformers path
and is therefore the exact engine-matched unfiltered baseline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path  # noqa: E402
from medrag.core import BenchmarkSample, RetrievedDocument  # noqa: E402
from medrag.evaluation import evaluate_prediction  # noqa: E402
from medrag.filtering.rag2_filter import Rag2FlanT5Filter  # noqa: E402
from medrag.generation.semantic_attention import (  # noqa: E402
    register_semantic_attention,
    suppression_bias,
)
from medrag.io_utils import iter_jsonl  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    END_REASONING_MARKER,
    FINAL_ANSWER_PREFIX,
    RATIONALE_HEADER,
    build_anchored_user_prompt,
    canonical_response,
    normalize_rationale,
    render_chat_prompt,
)


RUN_VERSION = "rag2_semantic_attention_suppression_mvp_v1"
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1"
    / "all_mcq_paper_balanced_max32_rationale_answer_rerank128"
    / "candidates/521e23c599352822/candidates.jsonl"
)
DEFAULT_FILTER = (
    WORKSPACE_ROOT
    / "models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa"
    / "medqa_semantic_top8_binary_support_epoch8_len1280_fullpair"
    / "20260830_170945/final_model"
)
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/rag2_semantic_attention_suppression_mvp_v1"
PAPER_SOURCES = ("pubmed", "pmc", "cpg", "textbooks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), default="medqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-root", type=Path, default=PROJECT_ROOT / "datasets/benchmark")
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--semantic-filter-model", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--max-samples", type=int, default=256, help="0 uses the complete dataset")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--filter-question-batch-size", type=int, default=32)
    parser.add_argument("--filter-batch-size", type=int, default=64)
    parser.add_argument("--filter-max-input-length", type=int, default=1280)
    parser.add_argument("--semantic-temperature", type=float, default=1.0)
    parser.add_argument("--max-suppression-factor", type=float, default=4.0)
    parser.add_argument("--semantic-layer-start", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def stable_lambda_name(value: float) -> str:
    return (f"{value:g}").replace("-", "m").replace(".", "p")


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rows[str(row["sample_id"])] = row
    return rows


def document_from_row(row: dict[str, Any]) -> RetrievedDocument:
    metadata = {
        "retrieval_bucket": row.get("retrieval_bucket"),
        "source_retrieval_rank": row.get("source_retrieval_rank"),
    }
    return RetrievedDocument(
        source=str(row.get("source") or ""),
        local_id=int(row.get("local_id") or 0),
        db_id=str(row.get("db_id") or ""),
        corpus_id=row.get("corpus_id"),
        chunk_id=row.get("chunk_id"),
        doc_id=row.get("doc_id"),
        title=row.get("title"),
        text=str(row.get("text") or ""),
        retrieval_score=float(row.get("retrieval_score") or 0.0),
        retrieval_rank=row.get("retrieval_rank"),
        rerank_score=row.get("rerank_score"),
        rerank_rank=row.get("rerank_rank"),
        metadata=metadata,
    )


def project_paper_top_k(row: dict[str, Any], top_k: int) -> list[RetrievedDocument]:
    source_ranks: Counter[str] = Counter()
    rank_by_id: dict[str, int] = {}
    for raw in row.get("initial_documents") or []:
        source = str(raw.get("retrieval_bucket") or raw.get("source"))
        source_ranks[source] += 1
        stable_id = str(raw.get("stable_id") or raw.get("corpus_id") or raw.get("db_id"))
        rank_by_id[stable_id] = int(source_ranks[source])
    expected = {source: top_k for source in PAPER_SOURCES}
    actual = {source: min(int(source_ranks[source]), top_k) for source in PAPER_SOURCES}
    if actual != expected:
        raise RuntimeError(f"Incomplete paper-balanced candidate row {row.get('sample_id')}: {actual} != {expected}")

    eligible: list[dict[str, Any]] = []
    for raw in row.get("reranked_documents") or []:
        stable_id = str(raw.get("stable_id") or raw.get("corpus_id") or raw.get("db_id"))
        if rank_by_id.get(stable_id, top_k + 1) <= top_k:
            eligible.append(raw)
    expected_pool = len(PAPER_SOURCES) * top_k
    if len(eligible) != expected_pool:
        raise RuntimeError(
            f"Incomplete rerank pool {row.get('sample_id')}: {len(eligible)} != {expected_pool}"
        )
    eligible.sort(key=lambda item: float(item.get("rerank_score") or float("-inf")), reverse=True)
    return [document_from_row(item) for item in eligible[:top_k]]


def select_samples(args: argparse.Namespace) -> list[BenchmarkSample]:
    path = resolve_benchmark_path(args.benchmark_root, "mcq", "unified", args.dataset, args.split)
    samples = load_benchmark_samples(path, "mcq", "unified", args.dataset, args.split)
    if args.max_samples > 0 and args.max_samples < len(samples):
        rng = random.Random(args.sample_seed)
        samples = sorted(rng.sample(samples, args.max_samples), key=lambda sample: sample.row_idx)
    return samples


def load_candidates(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    progress: PipelineProgress,
) -> dict[str, list[RetrievedDocument]]:
    requested = {sample.id for sample in samples}
    selected: dict[str, list[RetrievedDocument]] = {}
    progress.set_stage("1/3 load cached paper-balanced candidates", total=len(samples))
    for row in iter_jsonl(args.candidate_cache):
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in requested:
            continue
        if str(row.get("dataset")) != args.dataset:
            raise RuntimeError(f"Candidate dataset mismatch for {sample_id}")
        selected[sample_id] = project_paper_top_k(row, args.top_k)
        progress.update(1)
        if len(selected) == len(samples):
            break
    missing = sorted(requested - set(selected))
    if missing:
        raise RuntimeError(f"Candidate cache is missing {len(missing)} selected samples: {missing[:10]}")
    return selected


def score_documents(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    candidates: dict[str, list[RetrievedDocument]],
    cache_path: Path,
    progress: PipelineProgress,
) -> dict[str, dict[str, Any]]:
    cached = load_rows_by_id(cache_path) if args.resume else {}
    valid_cached: dict[str, dict[str, Any]] = {}
    for sample in samples:
        row = cached.get(sample.id)
        expected_ids = [doc.stable_id for doc in candidates[sample.id]]
        if row and row.get("document_ids") == expected_ids:
            valid_cached[sample.id] = row
    total_documents = len(samples) * args.top_k
    completed_documents = len(valid_cached) * args.top_k
    progress.set_stage(
        "2/3 semantic document scoring",
        total=total_documents,
        initial=completed_documents,
    )
    remaining = [sample for sample in samples if sample.id not in valid_cached]
    if not remaining:
        return valid_cached

    filterer = Rag2FlanT5Filter(
        model_path=args.semantic_filter_model,
        batch_size=args.filter_batch_size,
        max_input_length=args.filter_max_input_length,
        max_doc_chars=0,
        device=args.device,
        bf16=args.dtype == "bfloat16",
        scoring_method="special_token",
        input_format="official",
    )
    try:
        for start in range(0, len(remaining), args.filter_question_batch_size):
            question_batch = remaining[start : start + args.filter_question_batch_size]
            flat_samples: list[BenchmarkSample] = []
            flat_evidence: list[str] = []
            for sample in question_batch:
                docs = candidates[sample.id]
                flat_samples.extend([sample] * len(docs))
                flat_evidence.extend(doc.text for doc in docs)
            scores = filterer.score_evidences(flat_samples, flat_evidence)
            offset = 0
            output_rows: list[dict[str, Any]] = []
            for sample in question_batch:
                docs = candidates[sample.id]
                sample_scores = scores[offset : offset + len(docs)]
                offset += len(docs)
                row = {
                    "run_version": RUN_VERSION,
                    "sample_id": sample.id,
                    "dataset": sample.dataset,
                    "top_k": args.top_k,
                    "document_ids": [doc.stable_id for doc in docs],
                    "scores": sample_scores,
                }
                output_rows.append(row)
                valid_cached[sample.id] = row
            append_jsonl(cache_path, output_rows)
            progress.update(len(flat_samples))
    finally:
        filterer.close()
        del filterer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return valid_cached


def greedy_generate_with_semantic_attention(
    model: Any,
    input_ids: torch.Tensor,
    token_bias: torch.Tensor,
    query_mask: torch.Tensor,
    layer_start: int,
    max_new_tokens: int,
    stop_sequences: list[list[int]],
    eos_token_id: int | None,
) -> torch.Tensor:
    """Greedy decode with a DynamicCache while retaining custom model kwargs.

    Transformers ``generate`` rejects custom kwargs before calling models whose
    forward signatures expose a typed generic kwargs object.  This loop is the
    equivalent batch-one greedy path and deliberately forwards the semantic
    tensors on the prefill and every cached decoding step.
    """

    generated: list[int] = []
    current_ids = input_ids
    attention_mask = torch.ones_like(input_ids)
    past_key_values = None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                semantic_token_bias=token_bias,
                semantic_query_mask=query_mask,
                semantic_layer_start=layer_start,
            )
            next_token = int(outputs.logits[0, -1].argmax().item())
            generated.append(next_token)
            past_key_values = outputs.past_key_values
            if eos_token_id is not None and next_token == int(eos_token_id):
                break
            if any(
                len(generated) >= len(stop)
                and generated[-len(stop) :] == stop
                for stop in stop_sequences
                if stop
            ):
                break
            current_ids = torch.tensor([[next_token]], dtype=input_ids.dtype, device=input_ids.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=input_ids.device)],
                dim=1,
            )
    return torch.tensor([generated], dtype=input_ids.dtype, device=input_ids.device)


def token_overlap_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end]


def build_semantic_tensors(
    tokenizer: Any,
    full_prompt: str,
    user_prompt: str,
    document_texts: list[str],
    document_biases: list[float],
    assistant_start: int,
    reserved_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"][0].tolist()]
    total_length = int(input_ids.shape[1]) + int(reserved_length)
    token_bias = torch.zeros((1, total_length), dtype=torch.float32, device=device)
    query_mask = torch.zeros((1, total_length), dtype=torch.float32, device=device)

    user_start = full_prompt.find(user_prompt)
    if user_start < 0:
        raise RuntimeError("Rendered chat prompt does not contain the original user prompt")
    documents_marker = "Documents:\n"
    evidence_start_in_user = user_prompt.find(documents_marker)
    if evidence_start_in_user < 0:
        raise RuntimeError("Anchored user prompt has no Documents marker")
    evidence_start_in_user += len(documents_marker)
    cursor = evidence_start_in_user
    for document_text, bias in zip(document_texts, document_biases, strict=True):
        local_start = user_prompt.find(document_text, cursor)
        if local_start < 0:
            raise RuntimeError("Unable to align a document to its prompt character span")
        local_end = local_start + len(document_text)
        absolute_start = user_start + local_start
        absolute_end = user_start + local_end
        indices = token_overlap_indices(offsets, absolute_start, absolute_end)
        if not indices:
            raise RuntimeError("A document character span did not overlap any prompt token")
        token_bias[0, indices] = float(bias)
        cursor = local_end

    assistant_indices = [
        index for index, (_, right) in enumerate(offsets) if right > assistant_start
    ]
    if assistant_indices:
        query_mask[0, assistant_indices] = 1.0
    query_mask[0, int(input_ids.shape[1]) :] = 1.0
    return input_ids, token_bias, query_mask


def generate_one(
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    sample: BenchmarkSample,
    documents: list[RetrievedDocument],
    score_row: dict[str, Any],
    strength: float,
    choice_token_ids: dict[str, int],
) -> dict[str, Any]:
    temperature = float(args.semantic_temperature)
    if temperature <= 0:
        raise ValueError("semantic_temperature must be positive")
    margins = [float(item["margin"]) / temperature for item in score_row["scores"]]
    biases = [
        suppression_bias(margin, strength, args.max_suppression_factor)
        for margin in margins
    ]
    document_texts = [" ".join(doc.text.split()) for doc in documents]
    evidence = "\n\n".join(document_texts)
    user_prompt = build_anchored_user_prompt(sample.raw, evidence)
    chat_prompt = render_chat_prompt(tokenizer, user_prompt)
    rationale_prompt = chat_prompt + RATIONALE_HEADER
    input_ids, token_bias, query_mask = build_semantic_tensors(
        tokenizer,
        rationale_prompt,
        user_prompt,
        document_texts,
        biases,
        assistant_start=len(chat_prompt),
        reserved_length=args.max_new_tokens,
        device=model.device,
    )
    if int(input_ids.shape[1]) + args.max_new_tokens > args.max_model_length:
        raise RuntimeError(
            f"Prompt exceeds model budget for {sample.id}: prompt={input_ids.shape[1]} "
            f"new={args.max_new_tokens} max={args.max_model_length}"
        )
    stop_texts = [END_REASONING_MARKER, "\nFinal answer:", "\nTherefore, the answer"]
    stop_ids = [tokenizer.encode(text, add_special_tokens=False) for text in stop_texts]
    generated = greedy_generate_with_semantic_attention(
        model,
        input_ids,
        token_bias,
        query_mask,
        args.semantic_layer_start,
        args.max_new_tokens,
        stop_ids,
        tokenizer.eos_token_id,
    )
    raw_rationale = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    for stop_text in stop_texts:
        if raw_rationale.endswith(stop_text.strip()):
            raw_rationale = raw_rationale[: -len(stop_text.strip())].rstrip()
            break
    rationale, quality_flags = normalize_rationale(raw_rationale)
    generated_ids = generated[0].tolist()
    ended_with_eos = bool(
        generated_ids
        and tokenizer.eos_token_id is not None
        and generated_ids[-1] == int(tokenizer.eos_token_id)
    )
    ended_with_stop = any(
        len(generated_ids) >= len(stop) and generated_ids[-len(stop) :] == stop
        for stop in stop_ids
        if stop
    )
    if int(generated.shape[1]) >= args.max_new_tokens and not ended_with_eos and not ended_with_stop:
        quality_flags.append("rationale_length_exhausted")

    decision_prompt = (
        chat_prompt
        + RATIONALE_HEADER
        + rationale
        + "\n"
        + END_REASONING_MARKER
        + "\n"
        + FINAL_ANSWER_PREFIX
    )
    decision_ids, decision_bias, decision_query_mask = build_semantic_tensors(
        tokenizer,
        decision_prompt,
        user_prompt,
        document_texts,
        biases,
        assistant_start=len(chat_prompt),
        reserved_length=0,
        device=model.device,
    )
    if int(decision_ids.shape[1]) > args.max_model_length:
        raise RuntimeError(f"Decision prompt exceeds model budget for {sample.id}")
    with torch.inference_mode():
        outputs = model(
            input_ids=decision_ids,
            attention_mask=torch.ones_like(decision_ids),
            use_cache=False,
            semantic_token_bias=decision_bias,
            semantic_query_mask=decision_query_mask,
            semantic_layer_start=args.semantic_layer_start,
        )
    option_logits = {
        label: float(outputs.logits[0, -1, token_id].float().cpu())
        for label, token_id in choice_token_ids.items()
    }
    answer = max(option_logits, key=option_logits.get)
    response = canonical_response(rationale, answer, sample.options or {})
    evaluation = evaluate_prediction(sample, answer)
    return {
        "run_version": RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample.id,
        "dataset": sample.dataset,
        "row_idx": sample.row_idx,
        "lambda": strength,
        "top_k": args.top_k,
        "document_ids": [doc.stable_id for doc in documents],
        "semantic_margins": margins,
        "attention_biases": biases,
        "semantic_prob_helpful": [float(item["prob_helpful"]) for item in score_row["scores"]],
        "prediction": answer,
        "gold_answer": sample.answer,
        "correct": bool(evaluation["correct"]),
        "rationale": rationale,
        "raw_rationale": raw_rationale,
        "quality_flags": quality_flags,
        "canonical_response": response,
        "choice_logits": option_logits,
        "prompt_tokens": int(input_ids.shape[1]),
        "rationale_tokens": int(generated.shape[1]),
    }


def summarize(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    run_dir: Path,
) -> dict[str, Any]:
    sample_ids = {sample.id for sample in samples}
    conditions: list[dict[str, Any]] = []
    rows_by_lambda: dict[float, dict[str, dict[str, Any]]] = {}
    for strength in args.lambdas:
        path = run_dir / f"lambda_{stable_lambda_name(strength)}.jsonl"
        rows = {key: value for key, value in load_rows_by_id(path).items() if key in sample_ids}
        if len(rows) != len(samples):
            raise RuntimeError(f"Incomplete condition lambda={strength}: {len(rows)}/{len(samples)}")
        rows_by_lambda[float(strength)] = rows
    baseline_strength = 0.0
    if baseline_strength not in rows_by_lambda:
        raise ValueError("The MVP requires lambda=0 as the engine-matched baseline")
    baseline = rows_by_lambda[baseline_strength]
    baseline_correct = sum(bool(row["correct"]) for row in baseline.values())
    for strength in args.lambdas:
        rows = rows_by_lambda[float(strength)]
        correct = sum(bool(row["correct"]) for row in rows.values())
        wrong_to_correct = sum(
            not bool(baseline[sample_id]["correct"]) and bool(row["correct"])
            for sample_id, row in rows.items()
        )
        correct_to_wrong = sum(
            bool(baseline[sample_id]["correct"]) and not bool(row["correct"])
            for sample_id, row in rows.items()
        )
        conditions.append(
            {
                "lambda": float(strength),
                "samples": len(rows),
                "documents_per_question": args.top_k,
                "correct": correct,
                "accuracy": correct / len(rows),
                "delta_accuracy_vs_lambda_0": (correct - baseline_correct) / len(rows),
                "wrong_to_correct": wrong_to_correct,
                "correct_to_wrong": correct_to_wrong,
                "net_answer_gain": wrong_to_correct - correct_to_wrong,
                "mean_attention_bias": sum(
                    sum(float(value) for value in row["attention_biases"])
                    for row in rows.values()
                )
                / (len(rows) * args.top_k),
            }
        )
    summary = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "top_k": args.top_k,
        "sample_seed": args.sample_seed,
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest(),
        "semantic_filter_model": str(args.semantic_filter_model.resolve()),
        "llm_model": str(args.llm_model.resolve()),
        "semantic_layer_start": args.semantic_layer_start,
        "semantic_temperature": args.semantic_temperature,
        "max_suppression_factor": args.max_suppression_factor,
        "conditions": conditions,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    lines = [
        "| Lambda | N | Docs | Correct | Accuracy | Delta vs 0 | W->C | C->W | Net | Mean bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(
            f"| {row['lambda']:g} | {row['samples']} | {row['documents_per_question']:.2f} | "
            f"{row['correct']} | {100*row['accuracy']:.2f}% | "
            f"{100*row['delta_accuracy_vs_lambda_0']:+.2f}%p | {row['wrong_to_correct']} | "
            f"{row['correct_to_wrong']} | {row['net_answer_gain']:+d} | {row['mean_attention_bias']:.4f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0 or args.top_k > 32:
        raise ValueError("top_k must be in [1, 32]")
    if args.filter_question_batch_size <= 0 or args.filter_batch_size <= 0:
        raise ValueError("Filter batch sizes must be positive")
    if args.max_new_tokens <= 0 or args.max_model_length <= args.max_new_tokens:
        raise ValueError("Invalid model/generation token budget")
    if args.semantic_temperature <= 0:
        raise ValueError("semantic_temperature must be positive")
    if args.semantic_layer_start < 0 or args.semantic_layer_start >= 32:
        raise ValueError("semantic_layer_start must be in [0, 31] for Llama-3-8B")
    if 0.0 not in args.lambdas:
        raise ValueError("lambdas must include 0 for the engine-matched baseline")
    for path in (args.candidate_cache, args.semantic_filter_model, args.llm_model):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    samples = select_samples(args)
    conditions = list(dict.fromkeys(float(value) for value in args.lambdas))
    args.lambdas = conditions
    run_dir = args.output_dir / f"{args.dataset}_top{args.top_k}_n{len(samples)}_seed{args.sample_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "sample_count": len(samples),
        "sample_seed": args.sample_seed,
        "top_k": args.top_k,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "semantic_filter_model": str(args.semantic_filter_model.resolve()),
        "llm_model": str(args.llm_model.resolve()),
        "semantic_temperature": args.semantic_temperature,
        "max_suppression_factor": args.max_suppression_factor,
        "semantic_layer_start": args.semantic_layer_start,
        "max_new_tokens": args.max_new_tokens,
        "max_model_length": args.max_model_length,
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file() and args.resume:
        previous_contract = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_contract != run_contract:
            raise RuntimeError(
                "Resume contract mismatch. Use a different --output-dir or --no-resume. "
                f"existing={previous_contract} requested={run_contract}"
            )
    else:
        if not args.resume:
            for path in run_dir.glob("lambda_*.jsonl"):
                path.unlink()
            score_path = run_dir / "semantic_scores.jsonl"
            if score_path.exists():
                score_path.unlink()
        atomic_write_json(manifest_path, run_contract)
    score_cache = run_dir / "semantic_scores.jsonl"
    score_cached = len(load_rows_by_id(score_cache)) if args.resume else 0
    completed_results = 0
    if args.resume:
        for strength in conditions:
            completed_results += 2 * len(
                load_rows_by_id(run_dir / f"lambda_{stable_lambda_name(strength)}.jsonl")
            )
    overall_total = len(samples) + len(samples) * args.top_k + 2 * len(samples) * len(conditions)
    progress = PipelineProgress(
        overall_total=overall_total,
        overall_initial=min(score_cached, len(samples)) * args.top_k + completed_results,
        desc="SemanticAttentionMVP",
    )
    try:
        candidates = load_candidates(args, samples, progress)
        if args.dry_run:
            logging.info(
                "Dry-run complete: dataset=%s samples=%d top_k=%d conditions=%s",
                args.dataset,
                len(samples),
                args.top_k,
                conditions,
            )
            return
        scores = score_documents(args, samples, candidates, score_cache, progress)

        selected_ids = {sample.id for sample in samples}
        completed_condition_rows = {
            strength: {
                sample_id: row
                for sample_id, row in load_rows_by_id(
                    run_dir / f"lambda_{stable_lambda_name(strength)}.jsonl"
                ).items()
                if sample_id in selected_ids
            }
            for strength in conditions
        }
        if args.resume and all(
            len(rows) == len(samples) for rows in completed_condition_rows.values()
        ):
            summary = summarize(args, samples, run_dir)
            logging.info("All generation conditions are cached; summary refreshed at %s", run_dir / "summary.md")
            return

        attention_name = register_semantic_attention()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        logging.info("Loading target LLM with custom attention: %s", args.llm_model)
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=attention_name,
        ).to(torch.device(args.device))
        model.eval()
        choice_token_ids: dict[str, int] = {}
        for label in CHOICES:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"Choice label {label} is not one token after '(': {token_ids}")
            choice_token_ids[label] = int(token_ids[0])

        for condition_index, strength in enumerate(conditions, start=1):
            result_path = run_dir / f"lambda_{stable_lambda_name(strength)}.jsonl"
            existing = completed_condition_rows[strength] if args.resume else {}
            valid_existing = {sample_id: row for sample_id, row in existing.items() if sample_id in scores}
            progress.set_stage(
                f"3/3 generate rationale+answer lambda={strength:g} ({condition_index}/{len(conditions)})",
                total=2 * len(samples),
                initial=2 * len(valid_existing),
            )
            for sample in samples:
                if sample.id in valid_existing:
                    continue
                progress.set_detail(f"sample={sample.id}")
                row = generate_one(
                    args,
                    tokenizer,
                    model,
                    sample,
                    candidates[sample.id],
                    scores[sample.id],
                    strength,
                    choice_token_ids,
                )
                append_jsonl(result_path, [row])
                progress.update(2)
        summary = summarize(args, samples, run_dir)
        logging.info("Semantic-attention MVP complete: %s", run_dir / "summary.md")
        for row in summary["conditions"]:
            logging.info(
                "lambda=%g accuracy=%.4f delta=%+.4f W->C=%d C->W=%d net=%+d",
                row["lambda"],
                row["accuracy"],
                row["delta_accuracy_vs_lambda_0"],
                row["wrong_to_correct"],
                row["correct_to_wrong"],
                row["net_answer_gain"],
            )
    finally:
        progress.close()


if __name__ == "__main__":
    main()

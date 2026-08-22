#!/usr/bin/env python3
"""Generate one resumable three-anchor rationale pilot for layer selection.

The pilot deliberately reuses already reranked training candidates only to
sample representative question-document pairs.  It does not claim that those
candidates were retrieved with the new rationale format.  Full training data
must rerun retrieval after the anchor/layer contract is frozen.

Each trace is generated in two constrained parts:

1. free rationale generation, stopped at the fixed end-of-reasoning marker;
2. one A/B/C/D token after ``Final answer: (``.

The exact option text is appended deterministically.  Consequently every
saved trace has one valid answer without a post-hoc answer parser.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_rag2_preanswer_hidden_pilot import load_or_create_plan  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    END_REASONING_MARKER,
    FINAL_ANSWER_PREFIX,
    GENERATION_POLICY_VERSION,
    PROMPT_VERSION,
    RATIONALE_HEADER,
    TRACE_VERSION,
    build_anchored_user_prompt,
    canonical_response,
    normalize_rationale,
    normalized_mcq_row,
    rationale_generation_prompt,
    render_chat_prompt,
    semantic_retrieval_queries,
    sha256_text,
)
from medrag.rag2_generation import selected_logprob, span_stats  # noqa: E402


DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2"
    / "candidates/quality_selected_source_balanced40_rerank32_v1"
)
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
PILOT_VERSION = "rag2_three_anchor_layer_pilot_generation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medmcqa-candidates-path",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT / "medmcqa/train/candidates_top32.jsonl",
    )
    parser.add_argument(
        "--medqa-candidates-path",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT / "medqa/train/candidates_top32.jsonl",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=10_000)
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--questions-per-shard", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--llm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=80)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument(
        "--vllm-performance-mode",
        choices=["balanced", "interactivity", "throughput"],
        default="throughput",
    )
    parser.add_argument("--max-doc-chars", type=int, default=0)
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    lines = []
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False))
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def truncate_document(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars > 0 and len(value) > max_chars:
        return value[: max(0, max_chars - 3)].rstrip() + "..."
    return value


def token_logprobs(output: Any) -> list[float]:
    values: list[float] = []
    token_ids = list(getattr(output, "token_ids", None) or [])
    rows = list(getattr(output, "logprobs", None) or [])
    for token_id, row in zip(token_ids, rows):
        value = selected_logprob(row, int(token_id))
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def choice_logprob_map(choice_output: Any, choice_token_ids: dict[str, int]) -> dict[str, float | None]:
    rows = list(getattr(choice_output, "logprobs", None) or [])
    first = rows[0] if rows else None
    values: dict[str, float | None] = {}
    for label, token_id in choice_token_ids.items():
        item = None
        if isinstance(first, dict):
            item = first.get(token_id, first.get(str(token_id)))
        try:
            values[label] = float(getattr(item, "logprob", item)) if item is not None else None
        except (TypeError, ValueError):
            values[label] = None
    return values


def init_llm(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, dict[str, int]]:
    from vllm import LLM, SamplingParams

    if args.max_new_tokens <= 0 or args.retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("Retry token limit must be at least the primary token limit")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path), local_files_only=True, trust_remote_code=True
    )
    model_config = AutoConfig.from_pretrained(
        str(args.model_name_or_path), local_files_only=True, trust_remote_code=True
    )
    kwargs: dict[str, Any] = {
        "model": str(args.model_name_or_path),
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "disable_log_stats": True,
        "runner": "generate",
        "performance_mode": args.vllm_performance_mode,
        "max_num_seqs": args.vllm_max_num_seqs,
        "max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "enable_prefix_caching": True,
        "max_model_len": args.llm_max_model_len,
    }
    model_type = str(getattr(model_config, "model_type", "")).lower()
    logging.info("Loading %s through vLLM: %s", model_type or "local model", args.model_name_or_path)
    llm = LLM(**kwargs)

    rationale_stop = [END_REASONING_MARKER, "\nFinal answer:", "\nTherefore, the answer"]
    rationale_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=rationale_stop,
        include_stop_str_in_output=False,
        logprobs=1,
    )
    retry_sampling = SamplingParams(
        max_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=rationale_stop,
        include_stop_str_in_output=False,
        logprobs=1,
    )
    choice_token_ids: dict[str, int] = {}
    for label in CHOICES:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice {label!r} is not exactly one token after '(': {ids}")
        choice_token_ids[label] = int(ids[0])
    choice_sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        allowed_token_ids=list(choice_token_ids.values()),
        logprobs=len(choice_token_ids),
    )
    return tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, choice_token_ids


def trace_specs(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalized_mcq_row(row)
        questions.append(
            {
                "kind": "no_document",
                "dataset": normalized["dataset"],
                "sample_id": normalized["sample_id"],
                "row": normalized,
                "document": None,
            }
        )
        for document in normalized["documents"]:
            pairs.append(
                {
                    "kind": "with_document",
                    "dataset": normalized["dataset"],
                    "sample_id": normalized["sample_id"],
                    "pair_id": document["pair_id"],
                    "row": normalized,
                    "document": document,
                }
            )
    return questions, pairs


def generate_specs(
    args: argparse.Namespace,
    tokenizer: Any,
    llm: Any,
    rationale_sampling: Any,
    retry_sampling: Any,
    choice_sampling: Any,
    choice_token_ids: dict[str, int],
    specs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for batch_specs in chunks(specs, args.generation_batch_size):
        prompts: list[str] = []
        document_texts: list[str | None] = []
        for spec in batch_specs:
            document = spec.get("document")
            document_text = (
                truncate_document(document.get("text"), args.max_doc_chars)
                if isinstance(document, dict)
                else None
            )
            document_texts.append(document_text)
            prompts.append(rationale_generation_prompt(tokenizer, spec["row"], document_text))
        generated = list(llm.generate(prompts, rationale_sampling, use_tqdm=False))
        if len(generated) != len(batch_specs):
            raise RuntimeError("Rationale generation count mismatch")

        retry_indices = [
            index
            for index, item in enumerate(generated)
            if getattr(item.outputs[0], "finish_reason", None) == "length"
        ]
        if retry_indices:
            retry_outputs = list(
                llm.generate([prompts[index] for index in retry_indices], retry_sampling, use_tqdm=False)
            )
            if len(retry_outputs) != len(retry_indices):
                raise RuntimeError("Rationale retry count mismatch")
            for index, retry_output in zip(retry_indices, retry_outputs):
                generated[index] = retry_output

        rationales: list[str] = []
        raw_rationales: list[str] = []
        flags_by_trace: list[list[str]] = []
        decision_prompts: list[str] = []
        for spec, document_text, prompt, generation in zip(
            batch_specs, document_texts, prompts, generated
        ):
            choice = generation.outputs[0]
            raw = str(choice.text or "").strip()
            rationale, flags = normalize_rationale(raw)
            if getattr(choice, "finish_reason", None) == "length":
                flags.append("rationale_length_exhausted")
            raw_rationales.append(raw)
            rationales.append(rationale)
            flags_by_trace.append(sorted(set(flags)))
            user_prompt = build_anchored_user_prompt(spec["row"], document_text)
            decision_prompts.append(
                render_chat_prompt(tokenizer, user_prompt)
                + RATIONALE_HEADER
                + rationale
                + "\n"
                + END_REASONING_MARKER
                + "\n"
                + FINAL_ANSWER_PREFIX
            )

        choice_generations = list(llm.generate(decision_prompts, choice_sampling, use_tqdm=False))
        if len(choice_generations) != len(batch_specs):
            raise RuntimeError("Constrained choice generation count mismatch")

        for spec, document_text, prompt, generation, raw, rationale, flags, choice_generation in zip(
            batch_specs,
            document_texts,
            prompts,
            generated,
            raw_rationales,
            rationales,
            flags_by_trace,
            choice_generations,
        ):
            rationale_output = generation.outputs[0]
            choice_output = choice_generation.outputs[0]
            answer = str(choice_output.text or "").strip().upper()
            if answer not in CHOICES:
                raise RuntimeError(
                    f"Constrained decoder returned {answer!r} for {spec['sample_id']}"
                )
            options = spec["row"]["options"]
            query_views = semantic_retrieval_queries(rationale, answer, options)
            query_views["question_only"] = spec["row"]["question"]
            row = {
                "pilot_version": PILOT_VERSION,
                "trace_version": TRACE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "generation_policy_version": GENERATION_POLICY_VERSION,
                "kind": spec["kind"],
                "dataset": spec["dataset"],
                "sample_id": spec["sample_id"],
                "pair_id": spec.get("pair_id"),
                "question": spec["row"]["question"],
                "options": options,
                "gold_answer": spec["row"]["answer"],
                "document": spec.get("document"),
                "document_text_used": document_text,
                "model_raw_rationale": raw,
                "rationale": rationale,
                "answer": answer,
                "answer_text": options[answer],
                "answer_correct": answer == spec["row"]["answer"],
                "canonical_response": canonical_response(rationale, answer, options),
                "retrieval_queries": query_views,
                "quality_flags": flags,
                "valid_for_layer_analysis": not flags,
                "rationale_finish_reason": getattr(rationale_output, "finish_reason", None),
                "rationale_stop_reason": getattr(rationale_output, "stop_reason", None),
                "rationale_token_ids": [int(value) for value in rationale_output.token_ids],
                "rationale_stats": span_stats(token_logprobs(rationale_output)),
                "choice_token_id": int(choice_output.token_ids[0]),
                "choice_logprobs": choice_logprob_map(choice_output, choice_token_ids),
                "user_prompt_sha256": sha256_text(
                    build_anchored_user_prompt(spec["row"], document_text)
                ),
                "rendered_rationale_prompt_sha256": sha256_text(prompt),
            }
            output_rows.append(row)
    return output_rows


def shard_paths(output_dir: Path, shard_index: int) -> dict[str, Path]:
    root = output_dir / "trace_shards" / f"shard_{shard_index:05d}"
    return {
        "root": root,
        "questions": root / "questions.jsonl",
        "pairs": root / "pairs.jsonl",
        "complete": root / "COMPLETE.json",
    }


def valid_complete(paths: dict[str, Path], question_count: int, pair_count: int) -> bool:
    if not paths["complete"].is_file() or not paths["questions"].is_file() or not paths["pairs"].is_file():
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("pilot_version") == PILOT_VERSION
        and marker.get("question_count") == question_count
        and marker.get("pair_count") == pair_count
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # This environment does not ship the ``ninja`` executable required by
    # FlashInfer's sampling JIT.  The native vLLM sampler is deterministic for
    # the greedy settings used here and matches the rest of this project's
    # stable generation pipelines.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.max_pairs <= 0 or args.docs_per_question <= 0:
        raise ValueError("--max-pairs and --docs-per-question must be positive")
    if args.max_pairs % 2:
        raise ValueError("--max-pairs must be even so MedMCQA and MedQA receive equal pair counts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_or_create_plan(args)
    pair_count = sum(len(row["documents"]) for row in plan)
    logging.info("Pilot plan ready: questions=%d pairs=%d", len(plan), pair_count)
    if args.dry_run:
        counts = Counter(row["dataset"] for row in plan)
        logging.info("Dry run dataset questions: %s", dict(counts))
        return

    tokenizer, llm, rationale_sampling, retry_sampling, choice_sampling, choice_token_ids = init_llm(args)
    total_shards = math.ceil(len(plan) / args.questions_per_shard)
    progress = tqdm(total=pair_count, desc="AnchoredTracePilot", unit="pair", dynamic_ncols=True)
    valid_counts: Counter[str] = Counter()
    for shard_index, shard_rows in enumerate(chunks(plan, args.questions_per_shard)):
        expected_pairs = sum(len(row["documents"]) for row in shard_rows)
        paths = shard_paths(args.output_dir, shard_index)
        if args.resume and valid_complete(paths, len(shard_rows), expected_pairs):
            progress.update(expected_pairs)
            continue
        paths["root"].mkdir(parents=True, exist_ok=True)
        question_specs, pair_specs = trace_specs(shard_rows)
        questions = generate_specs(
            args,
            tokenizer,
            llm,
            rationale_sampling,
            retry_sampling,
            choice_sampling,
            choice_token_ids,
            question_specs,
        )
        pairs = generate_specs(
            args,
            tokenizer,
            llm,
            rationale_sampling,
            retry_sampling,
            choice_sampling,
            choice_token_ids,
            pair_specs,
        )
        atomic_write_jsonl(paths["questions"], questions)
        atomic_write_jsonl(paths["pairs"], pairs)
        atomic_write_json(
            paths["complete"],
            {
                "pilot_version": PILOT_VERSION,
                "completed_at": utc_now(),
                "shard_index": shard_index,
                "question_count": len(questions),
                "pair_count": len(pairs),
                "valid_questions": sum(bool(row["valid_for_layer_analysis"]) for row in questions),
                "valid_pairs": sum(bool(row["valid_for_layer_analysis"]) for row in pairs),
            },
        )
        valid_counts.update(
            "question_valid" if row["valid_for_layer_analysis"] else "question_invalid"
            for row in questions
        )
        valid_counts.update(
            "pair_valid" if row["valid_for_layer_analysis"] else "pair_invalid"
            for row in pairs
        )
        progress.update(expected_pairs)
        progress.set_postfix(shard=f"{shard_index + 1}/{total_shards}")
    progress.close()

    manifest = {
        "pilot_version": PILOT_VERSION,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "created_at": utc_now(),
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "candidate_paths": {
            "medmcqa": str(args.medmcqa_candidates_path.resolve()),
            "medqa": str(args.medqa_candidates_path.resolve()),
        },
        "selection_plan": str((args.output_dir / "selection_plan.jsonl").resolve()),
        "questions": len(plan),
        "pairs": pair_count,
        "docs_per_question": args.docs_per_question,
        "selection_seed": args.selection_seed,
        "shards": total_shards,
        "choice_token_ids": choice_token_ids,
        "rationale_max_new_tokens": args.max_new_tokens,
        "rationale_retry_max_new_tokens": args.retry_max_new_tokens,
        "max_doc_chars": args.max_doc_chars,
        "validity_observed_in_new_shards": dict(valid_counts),
        "candidate_usage_note": (
            "Existing candidates are used only for the layer-selection pilot. Full data generation must rerun "
            "rationale-query embedding, retrieval, and reranking under the frozen trace contract."
        ),
    }
    atomic_write_json(args.output_dir / "generation_manifest.json", manifest)
    logging.info("Anchored pilot generation complete: %s", args.output_dir)


if __name__ == "__main__":
    main()

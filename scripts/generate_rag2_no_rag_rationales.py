from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.environment import collect_environment, write_environment_files
from medrag.io_utils import iter_jsonl, write_json
from medrag.progress import StageProgress
from medrag.rag2_generation import (
    GENERATION_POLICY_VERSION,
    PPL_SCOPE_VERSION,
    generation_stats,
    render_prompt,
)
from medrag.rag2_mcq import (
    PAPER_ANSWER_FORMAT_PROMPT_VERSION,
    PAPER_EXACT_PROMPT_VERSION,
    PAPER_EXACT_TERMINAL_PROMPT_VERSION,
    PROMPT_VERSION,
    append_paper_exact_terminal_answer,
    build_choice_selection_messages,
    build_no_rag_messages,
    build_paper_answer_format_no_rag_messages,
    build_paper_exact_no_rag_messages,
    build_paper_exact_terminal_no_rag_messages,
    clean_text,
    gold_answers,
    is_correct,
    normalized_options,
    parse_mcq_output,
    parse_mcq_output_for_prompt_profile,
    parse_paper_exact_mcq_output,
    paper_exact_terminal_regex,
)


DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "datasets" / "benchmark"
DEFAULT_LLM_PATH = WORKSPACE_ROOT / "models" / "Qwen3.5-9B"
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT / "datasets" / "filtering" / "rag2" / "paper4_qwen35_9b_rationale_paper_focused_v4"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rag2" / "paper4_qwen35_9b_rationale_paper_focused_v4"
PAPER_EXACT_GENERATION_POLICY_VERSION = "rag2_llama3_paper_exact_greedy_v1"
PAPER_EXACT_TERMINAL_GENERATION_POLICY_VERSION = "rag2_llama3_paper_exact_terminal_structured_v1"
PAPER_ANSWER_FORMAT_GENERATION_POLICY_VERSION = "rag2_llama3_paper_answer_format_greedy_v2"
PAPER_EXACT_ANSWER_EXTRACTION_VERSION = "terminal_decision_sentence_no_rewrite_v4"
PAPER_EXACT_RETRIEVAL_QUERY_POLICY = "complete_visible_response_including_expressed_answer_no_rewrite_v1"
PAPER_EXACT_TERMINAL_ANSWER_EXTRACTION_VERSION = "exact_canonical_terminal_line_v1"
PAPER_EXACT_TERMINAL_RETRIEVAL_QUERY_POLICY = "complete_visible_response_including_exact_terminal_answer_v1"
MCQ_EVAL_DATASETS = [
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reusable no-RAG rationale queries with embedded answers for RAG2-style MCQ pseudo-labeling. "
            "Each output row contains a paper-style rationale query and both rationale-only and rationale+answer PPL."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=MCQ_EVAL_DATASETS, default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--llm-model-path", type=Path, default=DEFAULT_LLM_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="train_no_rag_rationale")
    parser.add_argument("--output-file", default="no_rag_generations.jsonl")
    parser.add_argument(
        "--prompt-profile",
        choices=["focused_v4", "paper_exact", "paper_exact_terminal", "paper_answer_format"],
        default="focused_v4",
        help=(
            "Use the legacy focused prompt, the prompt reported verbatim in RAG2, or the paper prompt with only a "
            "fixed final-answer sentence."
        ),
    )
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--length-retry-attempts",
        type=int,
        default=1,
        help="Retry only generations that terminate with finish_reason=length.",
    )
    parser.add_argument(
        "--length-retry-max-new-tokens",
        type=int,
        default=384,
        help="Generation cap for a compact rewrite when the primary generation reaches its length limit.",
    )
    parser.add_argument("--invalid-retry-attempts", type=int, default=1)
    parser.add_argument("--invalid-retry-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--quality-retry-max-new-tokens",
        type=int,
        default=384,
        help="Generation cap used when --retry-quality is enabled.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", nargs="+", default=["<|im_end|>", "<|eot_id|>"])
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--llm-max-model-len", type=int, default=8192)
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=["none", "auto", "flashinfer", "triton", "cutedsl"],
        default="auto",
        help="Qwen GDN backend. It is ignored for model families without GDN attention.",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-performance-mode", choices=["balanced", "interactivity", "throughput"], default="throughput")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", type=int, default=0, help="Per-dataset inclusive row offset.")
    parser.add_argument("--end", type=int, default=None, help="Per-dataset exclusive row offset.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-invalid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--retry-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regenerate only malformed outputs or rows without a usable rationale PPL trace."
        ),
    )
    parser.add_argument(
        "--choice-anchored-retry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For samples still failing after retries, select one option letter first and regenerate a rationale anchored to it.",
    )
    parser.add_argument("--choice-selection-max-new-tokens", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report selected rows without loading vLLM.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def active_prompt_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_exact_terminal":
        return PAPER_EXACT_TERMINAL_PROMPT_VERSION
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_PROMPT_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_PROMPT_VERSION
    return PROMPT_VERSION


def active_generation_policy_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_exact_terminal":
        return PAPER_EXACT_TERMINAL_GENERATION_POLICY_VERSION
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_GENERATION_POLICY_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_GENERATION_POLICY_VERSION
    return GENERATION_POLICY_VERSION


def build_generation_messages(
    args: argparse.Namespace,
    row: dict[str, Any],
    *,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    builders = {
        "focused_v4": build_no_rag_messages,
        "paper_exact": build_paper_exact_no_rag_messages,
        "paper_exact_terminal": build_paper_exact_terminal_no_rag_messages,
        "paper_answer_format": build_paper_answer_format_no_rag_messages,
    }
    builder = builders[args.prompt_profile]
    return builder(
        row,
        format_retry=format_retry,
        selected_answer=selected_answer,
        compact_retry=compact_retry,
    )


def benchmark_path(args: argparse.Namespace, dataset: str) -> Path:
    return args.benchmark_root / "mcq" / args.collection / dataset / f"{args.split}.jsonl"


def artifact_dir(args: argparse.Namespace, dataset: str) -> Path:
    return args.artifact_root / "no_rag" / dataset / args.split


def iter_selected_rows(path: Path, args: argparse.Namespace) -> Iterator[tuple[int, dict[str, Any]]]:
    start = max(0, int(args.start))
    end = None if args.end is None else max(start, int(args.end))
    selected = 0
    for row_idx, row in enumerate(iter_jsonl(path)):
        if row_idx < start:
            continue
        if end is not None and row_idx >= end:
            break
        if args.limit_per_dataset is not None and selected >= args.limit_per_dataset:
            break
        selected += 1
        yield row_idx, row


def count_selected_rows(path: Path, args: argparse.Namespace) -> int:
    return sum(1 for _ in iter_selected_rows(path, args))


def sample_id(row_idx: int, row: dict[str, Any], dataset: str, split: str) -> str:
    return str(row.get("id") or f"{dataset}:{split}:{row_idx:06d}")


def has_rationale_ppl(row: dict[str, Any]) -> bool:
    rationale_stats = ((row.get("generation_stats") or {}).get("rationale") or {})
    try:
        token_count = int(rationale_stats.get("token_count") or 0)
        ppl = float(rationale_stats.get("ppl"))
    except (TypeError, ValueError):
        return False
    return token_count > 0 and math.isfinite(ppl) and ppl > 0.0


def has_rationale_only_ppl(row: dict[str, Any]) -> bool:
    rationale_stats = ((row.get("generation_stats") or {}).get("rationale_only") or {})
    try:
        token_count = int(rationale_stats.get("token_count") or 0)
        ppl = float(rationale_stats.get("ppl"))
    except (TypeError, ValueError):
        return False
    return token_count > 0 and math.isfinite(ppl) and ppl > 0.0


def hard_generation_issues(row: dict[str, Any]) -> list[str]:
    parsed = row.get("parsed") or {}
    issues: list[str] = []
    if parsed.get("parse_errors"):
        issues.append("parse_errors")
    if not parsed.get("rationale"):
        issues.append("missing_rationale")
    if not parsed.get("final_answer"):
        issues.append("missing_final_answer")
    if not has_rationale_ppl(row):
        issues.append("missing_rationale_ppl")
    if not has_rationale_only_ppl(row):
        issues.append("missing_rationale_only_ppl")
    if row.get("truncated_by_max_tokens") or row.get("finish_reason") == "length":
        issues.append("max_tokens_exhausted")
    return issues


def retry_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.retry_quality or args.retry_invalid:
        return hard_generation_issues(row)
    return []


def load_existing(path: Path, args: argparse.Namespace) -> tuple[set[str], Counter[str]]:
    done: set[str] = set()
    retry_counts: Counter[str] = Counter()
    if not path.exists():
        return done, retry_counts
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Ignoring malformed JSONL line while resuming: %s:%s", path, line_no)
                continue
            identifier = str(row.get("sample_id") or "")
            if not identifier:
                continue
            latest[identifier] = row
    for identifier, row in latest.items():
        reasons = []
        if row.get("prompt_version") != active_prompt_version(args):
            reasons.append("prompt_version_mismatch")
        if row.get("ppl_scope_version") != PPL_SCOPE_VERSION:
            reasons.append("ppl_scope_version_mismatch")
        if row.get("generation_policy_version") != active_generation_policy_version(args):
            reasons.append("generation_policy_version_mismatch")
        reasons.extend(retry_reasons(row, args))
        if reasons:
            retry_counts.update(reasons)
        else:
            done.add(identifier)
    return done, retry_counts


def init_llm(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any]:
    from vllm import LLM, SamplingParams

    model_config = AutoConfig.from_pretrained(args.llm_model_path, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, local_files_only=True, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "model": str(args.llm_model_path),
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "disable_log_stats": True,
        "runner": "generate",
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "performance_mode": args.vllm_performance_mode,
        "max_num_seqs": args.vllm_max_num_seqs,
        "max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "max_model_len": args.llm_max_model_len,
    }
    model_type = str(getattr(model_config, "model_type", "")).lower()
    uses_gdn_attention = "qwen3_5" in model_type or "qwen3.5" in model_type
    if uses_gdn_attention and args.gdn_prefill_backend != "none":
        backend = "triton" if args.gdn_prefill_backend == "auto" else args.gdn_prefill_backend
        kwargs["additional_config"] = {"gdn_prefill_backend": backend}
    logging.info("Loading %s through vLLM: %s", model_type or "local model", args.llm_model_path)
    llm = LLM(**kwargs)
    quality_max_tokens = int(args.quality_retry_max_new_tokens)
    if quality_max_tokens <= 0:
        raise ValueError("--quality-retry-max-new-tokens must be positive.")
    if args.choice_selection_max_new_tokens <= 0:
        raise ValueError("--choice-selection-max-new-tokens must be positive.")
    initial_max_tokens = args.max_new_tokens
    retry_max_tokens = max(args.max_new_tokens, args.invalid_retry_max_new_tokens, quality_max_tokens)
    length_retry_max_tokens = max(args.max_new_tokens, int(args.length_retry_max_new_tokens))
    if args.length_retry_attempts < 0:
        raise ValueError("--length-retry-attempts cannot be negative.")
    if args.length_retry_attempts > 0 and length_retry_max_tokens < initial_max_tokens:
        raise ValueError("--length-retry-max-new-tokens must be at least --max-new-tokens.")
    sampling = SamplingParams(
        max_tokens=initial_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=1,
    )
    retry_sampling = SamplingParams(
        max_tokens=retry_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=1,
    )
    length_retry_sampling = SamplingParams(
        max_tokens=length_retry_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=1,
    )
    allowed_choice_ids = None
    choice_max_tokens = args.choice_selection_max_new_tokens
    if args.prompt_profile == "paper_exact_terminal":
        allowed_choice_ids = []
        for label in ("A", "B", "C", "D"):
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"Terminal fallback choice {label!r} is not one token: {token_ids}")
            allowed_choice_ids.append(int(token_ids[0]))
        choice_max_tokens = 1
    choice_sampling = SamplingParams(
        max_tokens=choice_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=[*args.stop, "\n"],
        logprobs=1,
        allowed_token_ids=allowed_choice_ids,
    )
    logging.info("Model ready for no-RAG rationale generation.")
    return tokenizer, llm, sampling, length_retry_sampling, retry_sampling, choice_sampling


def structured_sampling_for_rows(
    args: argparse.Namespace,
    base_sampling: Any,
    rows: list[tuple[int, dict[str, Any]]],
) -> Any:
    if args.prompt_profile != "paper_exact_terminal":
        return base_sampling
    from vllm.sampling_params import StructuredOutputsParams

    params = []
    for _, row in rows:
        item = base_sampling.clone()
        item.structured_outputs = StructuredOutputsParams(
            regex=paper_exact_terminal_regex(normalized_options(row))
        )
        params.append(item)
    return params


def output_row(
    row_idx: int,
    row: dict[str, Any],
    dataset: str,
    split: str,
    generated: Any,
    tokenizer: Any,
    generation_attempts: int,
    generation_prompt_variant: str = "standard",
    prompt_version: str = PROMPT_VERSION,
    generation_policy_version: str = GENERATION_POLICY_VERSION,
    preserve_raw_generation: bool = False,
    prompt_profile: str = "focused_v4",
    response_text: str | None = None,
    terminal_repair_source: str | None = None,
) -> dict[str, Any]:
    options = normalized_options(row)
    choice = generated.outputs[0] if getattr(generated, "outputs", None) else None
    model_raw_text = choice.text.strip() if choice is not None else ""
    raw_text = str(response_text).strip() if response_text is not None else model_raw_text
    parsed = parse_mcq_output_for_prompt_profile(raw_text, options, prompt_profile)
    answer = parsed.final_answer
    rationale_query = (
        parsed.visible_text
        if prompt_profile in {"paper_exact", "paper_exact_terminal"}
        else (clean_text(parsed.rationale) if preserve_raw_generation else parsed.rationale_query)
    )
    stats = (
        generation_stats(choice, tokenizer, model_raw_text, options)
        if choice is not None
        else {}
    )
    normalized_generation = (
        raw_text
        if preserve_raw_generation
        else (f"Rationale: {rationale_query}" if rationale_query and answer else raw_text)
    )
    finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
    stop_reason = getattr(choice, "stop_reason", None) if choice is not None else None
    return {
        "schema_version": 2,
        "stage": "rag2_no_rag_rationale",
        "prompt_version": prompt_version,
        "ppl_scope_version": PPL_SCOPE_VERSION,
        "generation_policy_version": generation_policy_version,
        "sample_id": sample_id(row_idx, row, dataset, split),
        "row_idx": row_idx,
        "dataset": str(row.get("dataset") or dataset),
        "split": str(row.get("split") or split),
        "subject": row.get("subject"),
        "question": clean_text(row.get("question")),
        "options": options,
        "gold_answer": row.get("answer"),
        "gold_answers": sorted(gold_answers(row)),
        "answer_text": row.get("answer_text"),
        "no_rag_generation": normalized_generation,
        "model_raw_generation": model_raw_text,
        "canonical_generation": raw_text,
        "terminal_repair_source": terminal_repair_source,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "truncated_by_max_tokens": finish_reason == "length",
        "generation_attempts": generation_attempts,
        "generation_prompt_variant": generation_prompt_variant,
        "generation_was_normalized": normalized_generation != parsed.visible_text,
        "answer_extraction_method": (
            PAPER_EXACT_TERMINAL_ANSWER_EXTRACTION_VERSION
            if prompt_profile == "paper_exact_terminal"
            else
            PAPER_EXACT_ANSWER_EXTRACTION_VERSION
            if prompt_profile == "paper_exact"
            else "legacy_contract_parser_v1"
        ),
        "retrieval_query_policy": (
            PAPER_EXACT_TERMINAL_RETRIEVAL_QUERY_POLICY
            if prompt_profile == "paper_exact_terminal"
            else
            PAPER_EXACT_RETRIEVAL_QUERY_POLICY
            if prompt_profile == "paper_exact"
            else "parsed_rationale_query_v1"
        ),
        "parsed": {
            "visible_text": parsed.visible_text,
            "rationale": parsed.rationale,
            "rationale_only": parsed.rationale_only,
            "rationale_query": rationale_query,
            "answer_conclusion": parsed.answer_conclusion,
            "rationale_query_normalized": False if preserve_raw_generation else parsed.rationale_query_normalized,
            "final_answer": answer,
            "final_answer_correct": is_correct(row, answer),
            "parse_errors": parsed.parse_errors,
        },
        "ppl": {
            "rationale_only": stats.get("rationale_only") or {},
            "rationale_plus_answer": stats.get("rationale_plus_answer") or stats.get("rationale") or {},
        },
        "generation_stats": stats,
    }


def extract_selected_option(output: Any, row: dict[str, Any]) -> str | None:
    choice = output.outputs[0] if getattr(output, "outputs", None) else None
    text = choice.text.strip().upper() if choice is not None else ""
    valid_options = set(normalized_options(row))
    for match in re.finditer(r"\b([A-Z])\b", text):
        candidate = match.group(1)
        if candidate in valid_options:
            return candidate
    return None


def summarize_output(path: Path) -> dict[str, Any]:
    summary = {
        "rows": 0,
        "raw_rows": 0,
        "superseded_rows": 0,
        "valid_rationale": 0,
        "valid_answer": 0,
        "valid_both": 0,
        "valid_rationale_ppl": 0,
        "valid_rationale_only_ppl": 0,
        "correct": 0,
        "invalid": 0,
        "quality_retry_eligible": 0,
        "malformed_rows": 0,
    }
    if not path.exists():
        return summary
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                summary["malformed_rows"] += 1
                continue
            summary["raw_rows"] += 1
            identifier = str(row.get("sample_id") or f"__line_{line_no}")
            latest[identifier] = row
    summary["rows"] = len(latest)
    summary["superseded_rows"] = summary["raw_rows"] - summary["rows"]
    for row in latest.values():
        parsed = row.get("parsed") or {}
        has_rationale = bool(parsed.get("rationale"))
        has_answer = bool(parsed.get("final_answer"))
        has_valid_output = bool(
            has_rationale
            and has_answer
            and not parsed.get("parse_errors")
            and not row.get("truncated_by_max_tokens")
            and has_rationale_ppl(row)
            and has_rationale_only_ppl(row)
        )
        summary["valid_rationale"] += int(has_rationale)
        summary["valid_answer"] += int(has_answer)
        summary["valid_both"] += int(has_valid_output)
        summary["valid_rationale_ppl"] += int(has_rationale_ppl(row))
        summary["valid_rationale_only_ppl"] += int(has_rationale_only_ppl(row))
        summary["correct"] += int(bool(parsed.get("final_answer_correct")))
        summary["invalid"] += int(not has_valid_output)
        summary["quality_retry_eligible"] += int(bool(hard_generation_issues(row)))
    if summary["rows"]:
        summary["accuracy"] = summary["correct"] / summary["rows"]
        summary["valid_both_rate"] = summary["valid_both"] / summary["rows"]
    else:
        summary["accuracy"] = None
        summary["valid_both_rate"] = None
    return summary


def write_summary_markdown(path: Path, run_config: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# RAG2 No-RAG Rationale Generation",
        "",
        f"- Run: `{run_config['run_name']}`",
        f"- Model: `{run_config['llm_model_path']}`",
        f"- Prompt version: `{run_config['prompt_version']}`",
        f"- Temperature: `{run_config['temperature']}`",
        f"- Max new tokens: `{run_config['max_new_tokens']}`",
        "",
        "| Dataset | Requested | Saved | Valid rationale + answer | Accuracy | Artifact |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, info in datasets.items():
        summary = info["summary"]
        accuracy = summary.get("accuracy")
        valid_rate = summary.get("valid_both_rate")
        lines.append(
            f"| {dataset} | {info['requested']} | {summary['rows']} | "
            f"{summary['valid_both']} ({'n/a' if valid_rate is None else f'{valid_rate:.2%}'}) | "
            f"{'n/a' if accuracy is None else f'{accuracy:.2%}'} | `{info['output_path']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.max_new_tokens <= 0 or args.length_retry_max_new_tokens <= 0:
        raise ValueError("Initial and length-retry token limits must be positive.")
    if args.length_retry_attempts < 0:
        raise ValueError("--length-retry-attempts cannot be negative.")
    if args.length_retry_attempts > 0 and args.length_retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("--length-retry-max-new-tokens must be at least --max-new-tokens.")
    prompt_version = active_prompt_version(args)
    generation_policy_version = active_generation_policy_version(args)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / "no_rag" / args.run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    selected: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        source_path = benchmark_path(args, dataset)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing benchmark split: {source_path}")
        target_dir = artifact_dir(args, dataset)
        output_path = target_dir / args.output_file
        count = count_selected_rows(source_path, args)
        selected[dataset] = {
            "source_path": source_path,
            "target_dir": target_dir,
            "output_path": output_path,
            "requested": count,
        }
        logging.info("[%s] selected %s rows from %s", dataset, count, source_path)

    run_config = {
        "stage": "rag2_no_rag_rationale",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "datasets": args.datasets,
        "split": args.split,
        "collection": args.collection,
        "benchmark_root": str(args.benchmark_root),
        "llm_model_path": str(args.llm_model_path),
        "artifact_root": str(args.artifact_root),
        "output_file": args.output_file,
        "prompt_profile": args.prompt_profile,
        "prompt_version": prompt_version,
        "generation_policy_version": generation_policy_version,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "length_retry_attempts": args.length_retry_attempts,
        "length_retry_max_new_tokens": args.length_retry_max_new_tokens,
        "invalid_retry_attempts": args.invalid_retry_attempts,
        "invalid_retry_max_new_tokens": args.invalid_retry_max_new_tokens,
        "quality_retry_max_new_tokens": args.quality_retry_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": args.stop,
        "use_chat_template": args.use_chat_template,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "llm_max_model_len": args.llm_max_model_len,
        "gdn_prefill_backend": args.gdn_prefill_backend,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "vllm_performance_mode": args.vllm_performance_mode,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "start": args.start,
        "end": args.end,
        "limit_per_dataset": args.limit_per_dataset,
        "resume": args.resume,
        "retry_invalid": args.retry_invalid,
        "retry_quality": args.retry_quality,
        "choice_anchored_retry": args.choice_anchored_retry,
        "choice_selection_max_new_tokens": args.choice_selection_max_new_tokens,
        "quality_policy": {
            "length_limit": None,
            "style_rewrites": False,
            "missing_rationale_ppl": "retry",
        },
        "benchmark_path": ",".join(str(selected[name]["source_path"]) for name in args.datasets),
    }
    write_json(run_dir / "run_config.json", run_config)
    environment = collect_environment(
        command=[sys.executable, *sys.argv],
        project_root=PROJECT_ROOT,
        workspace_root=WORKSPACE_ROOT,
        run_config=run_config,
    )
    write_environment_files(run_dir, environment)

    if args.dry_run:
        result_info = {}
        for dataset, info in selected.items():
            result_info[dataset] = {**info, "summary": summarize_output(info["output_path"])}
        write_json(run_dir / "artifact_locations.json", {name: {key: str(value) if isinstance(value, Path) else value for key, value in info.items() if key != "summary"} for name, info in result_info.items()})
        write_summary_markdown(run_dir / "summary.md", run_config, result_info)
        logging.info("Dry run complete: %s", run_dir)
        return

    if not args.llm_model_path.exists():
        raise FileNotFoundError(f"Missing local LLM model: {args.llm_model_path}")

    tokenizer, llm, sampling, length_retry_sampling, retry_sampling, choice_sampling = init_llm(args)
    result_info: dict[str, dict[str, Any]] = {}
    try:
        for dataset, info in selected.items():
            output_path: Path = info["output_path"]
            target_dir: Path = info["target_dir"]
            target_dir.mkdir(parents=True, exist_ok=True)
            done, retry_counts = load_existing(output_path, args) if args.resume else (set(), Counter())
            total = int(info["requested"])
            logging.info(
                "[%s] no-RAG generation: requested=%s existing_output_ids=%s retry_candidates=%s reasons=%s",
                dataset,
                total,
                len(done),
                total - len(done),
                dict(retry_counts),
            )

            artifact_manifest = {
                "type": "rag2_no_rag_rationale_artifact",
                "created_or_updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "dataset": dataset,
                "split": args.split,
                "source_path": str(info["source_path"]),
                "output_path": str(output_path),
                "llm_model_path": str(args.llm_model_path),
                "prompt_profile": args.prompt_profile,
                "prompt_version": prompt_version,
                "ppl_scope_version": PPL_SCOPE_VERSION,
                "generation_policy_version": generation_policy_version,
                "prompt_contract": (
                    "Rationale-generation prompt reported in RAG2 section 3.3 with one added constraint: the response "
                    "must end on a separate line with an uppercase option letter in mandatory parentheses and the exact "
                    "option text copied verbatim. No other response-style or length constraint is added."
                    if args.prompt_profile in {"paper_answer_format", "paper_exact_terminal"}
                    else (
                        "Verbatim rationale-generation prompt reported in RAG2 section 3.3; no response-style or length "
                        "constraint is added to the standard pass."
                        if args.prompt_profile == "paper_exact"
                        else (
                            "Legacy focused prompt requesting one concise paragraph centered on the decisive evidence. "
                            "The response ends as 'Therefore, the answer is (<option letter>) <option text>.'"
                        )
                    )
                ),
                "rationale_query": (
                    "The complete stored response, including the exact terminal selected option, but excluding the original "
                    "question and option list. Normally this is the structured primary generation; any exceptional terminal "
                    "repair is recorded per row."
                    if args.prompt_profile == "paper_exact_terminal"
                    else "The complete visible model response, including its final selected option, but excluding the original "
                    "question and option list. The paper-exact path preserves the model wording without canonical rewrite."
                    if args.prompt_profile in {"paper_exact", "paper_exact_terminal", "paper_answer_format"}
                    else (
                        "parsed.rationale_query including the answer conclusion; minor punctuation deviations are "
                        "normalized to the focused prompt's canonical conclusion."
                    )
                ),
                "answer_extraction": (
                    "Conservative post-generation extraction: accept exactly one option only when a terminal "
                    "answer/conclusion sentence explicitly identifies it, or when a terminal decision sentence "
                    "contains one unambiguous full option text. The generated response is never rewritten or repaired."
                    if args.prompt_profile in {"paper_exact", "paper_exact_terminal"}
                    else "Profile-specific MCQ output parser."
                ),
                "answer_extraction_version": (
                    PAPER_EXACT_TERMINAL_ANSWER_EXTRACTION_VERSION
                    if args.prompt_profile == "paper_exact_terminal"
                    else
                    PAPER_EXACT_ANSWER_EXTRACTION_VERSION
                    if args.prompt_profile == "paper_exact"
                    else "legacy_contract_parser_v1"
                ),
                "retrieval_query_policy": (
                    PAPER_EXACT_TERMINAL_RETRIEVAL_QUERY_POLICY
                    if args.prompt_profile == "paper_exact_terminal"
                    else
                    PAPER_EXACT_RETRIEVAL_QUERY_POLICY
                    if args.prompt_profile == "paper_exact"
                    else "parsed_rationale_query_v1"
                ),
                "raw_generation": "model_raw_generation preserves the unmodified model output used for token logprobs and PPL",
                "rationale_ppl": (
                    "primary length-normalized PPL over the complete paper-style rationale query, including its "
                    "answer conclusion"
                ),
                "rationale_only_ppl": (
                    "diagnostic PPL over the reasoning span before the detected final option; computed from the "
                    "same generation and token logprobs"
                ),
                "answer_conclusion_ppl": (
                    "diagnostic PPL over the detected final-answer span from the same generation"
                ),
                "invalid_retry": {
                    "attempts": args.invalid_retry_attempts,
                    "max_new_tokens": args.invalid_retry_max_new_tokens,
                },
                "length_retry": {
                    "attempts": args.length_retry_attempts,
                    "max_new_tokens": args.length_retry_max_new_tokens,
                    "trigger": "vLLM finish_reason=length",
                    "prompt": "same paper prompt with a compact-rewrite instruction; token cap is not increased",
                },
                "quality_retry": {
                    "enabled": args.retry_quality,
                    "max_rationale_words": None,
                    "checks": [
                        "parse_errors",
                        "missing_rationale",
                        "missing_final_answer",
                        "missing_rationale_ppl",
                        "missing_rationale_only_ppl",
                        "max_tokens_exhausted",
                    ],
                },
                "choice_anchored_retry": {
                    "enabled": args.choice_anchored_retry,
                    "selection_max_new_tokens": args.choice_selection_max_new_tokens,
                    "selection_source": "same local LLM; no gold answer is provided",
                },
                "run_dir": str(run_dir),
            }
            write_json(target_dir / "manifest.json", artifact_manifest)

            progress = StageProgress(total=total, desc=f"NoRAG:{dataset}", enabled=True)
            generated_count = 0
            with output_path.open("a", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
                batch_rows: list[tuple[int, dict[str, Any]]] = []
                batch_prompts: list[str] = []

                def flush_batch() -> None:
                    nonlocal generated_count, batch_rows, batch_prompts
                    if not batch_prompts:
                        return
                    outputs = llm.generate(
                        batch_prompts,
                        structured_sampling_for_rows(args, sampling, batch_rows),
                        use_tqdm=False,
                    )
                    if len(outputs) != len(batch_rows):
                        raise RuntimeError(f"vLLM output count mismatch: expected={len(batch_rows)} got={len(outputs)}")
                    outputs = list(outputs)
                    attempt_counts = [1] * len(outputs)
                    prompt_variants = ["standard"] * len(outputs)
                    length_retry_attempted: set[int] = set()
                    for length_retry_round in range(args.length_retry_attempts):
                        length_indices = [
                            index
                            for index, output in enumerate(outputs)
                            if (
                                getattr(
                                    output.outputs[0] if getattr(output, "outputs", None) else None,
                                    "finish_reason",
                                    None,
                                )
                                == "length"
                            )
                        ]
                        if not length_indices:
                            break
                        logging.info(
                            "[%s] length retry %s/%s for %s sample(s): max_tokens=%s",
                            dataset,
                            length_retry_round + 1,
                            args.length_retry_attempts,
                            len(length_indices),
                            args.length_retry_max_new_tokens,
                        )
                        length_prompts = [
                            render_prompt(
                                tokenizer,
                                build_generation_messages(
                                    args,
                                    batch_rows[index][1],
                                    format_retry=False,
                                    compact_retry=True,
                                ),
                                args.use_chat_template,
                            )
                            for index in length_indices
                        ]
                        length_rows = [batch_rows[index] for index in length_indices]
                        length_outputs = list(
                            llm.generate(
                                length_prompts,
                                structured_sampling_for_rows(args, length_retry_sampling, length_rows),
                                use_tqdm=False,
                            )
                        )
                        if len(length_outputs) != len(length_indices):
                            raise RuntimeError(
                                "vLLM length-retry output count mismatch: "
                                f"expected={len(length_indices)} got={len(length_outputs)}"
                            )
                        for index, length_output in zip(length_indices, length_outputs):
                            outputs[index] = length_output
                            attempt_counts[index] += 1
                            prompt_variants[index] = "decisive_compact_retry"
                            length_retry_attempted.add(index)

                    for retry_round in range(args.invalid_retry_attempts):
                        invalid_indices: list[int] = []
                        retry_reason_counts: Counter[str] = Counter()
                        for index, ((_, row), output) in enumerate(zip(batch_rows, outputs)):
                            choice = output.outputs[0] if getattr(output, "outputs", None) else None
                            if index in length_retry_attempted and getattr(choice, "finish_reason", None) == "length":
                                retry_reason_counts.update(["max_tokens_exhausted_after_length_retry"])
                                continue
                            if args.retry_invalid or args.retry_quality:
                                candidate_row = output_row(
                                    batch_rows[index][0],
                                    row,
                                    dataset,
                                    args.split,
                                    output,
                                    tokenizer,
                                    attempt_counts[index],
                                    prompt_version=prompt_version,
                                    generation_policy_version=generation_policy_version,
                                    preserve_raw_generation=args.prompt_profile in {
                                        "paper_exact",
                                        "paper_exact_terminal",
                                        "paper_answer_format",
                                    },
                                    prompt_profile=args.prompt_profile,
                                )
                                reasons = retry_reasons(candidate_row, args)
                            else:
                                choice = output.outputs[0] if getattr(output, "outputs", None) else None
                                raw_text = choice.text.strip() if choice is not None else ""
                                parsed = parse_mcq_output_for_prompt_profile(
                                    raw_text,
                                    normalized_options(row),
                                    args.prompt_profile,
                                )
                                reasons = ["parse_errors"] if (
                                    parsed.parse_errors or not parsed.rationale or not parsed.final_answer
                                ) else []
                            if reasons:
                                invalid_indices.append(index)
                                retry_reason_counts.update(reasons)
                        if not invalid_indices:
                            break
                        logging.info(
                            "[%s] quality retry %s/%s for %s sample(s): %s",
                            dataset,
                            retry_round + 1,
                            args.invalid_retry_attempts,
                            len(invalid_indices),
                            dict(retry_reason_counts),
                        )
                        retry_prompts = [
                            render_prompt(
                                tokenizer,
                                build_generation_messages(args, batch_rows[index][1], format_retry=True),
                                args.use_chat_template,
                            )
                            for index in invalid_indices
                        ]
                        retry_rows = [batch_rows[index] for index in invalid_indices]
                        retry_outputs = llm.generate(
                            retry_prompts,
                            structured_sampling_for_rows(args, retry_sampling, retry_rows),
                            use_tqdm=False,
                        )
                        if len(retry_outputs) != len(invalid_indices):
                            raise RuntimeError(
                                f"vLLM retry output count mismatch: expected={len(invalid_indices)} got={len(retry_outputs)}"
                            )
                        for index, retry_output in zip(invalid_indices, retry_outputs):
                            outputs[index] = retry_output
                            attempt_counts[index] += 1
                            prompt_variants[index] = "format_retry"

                    if args.choice_anchored_retry and args.prompt_profile != "paper_exact_terminal":
                        unresolved_indices: list[int] = []
                        for index, ((row_idx, row), output) in enumerate(zip(batch_rows, outputs)):
                            candidate_row = output_row(
                                row_idx,
                                row,
                                dataset,
                                args.split,
                                output,
                                tokenizer,
                                attempt_counts[index],
                                prompt_version=prompt_version,
                                generation_policy_version=generation_policy_version,
                                preserve_raw_generation=args.prompt_profile in {
                                    "paper_exact",
                                    "paper_exact_terminal",
                                    "paper_answer_format",
                                },
                                prompt_profile=args.prompt_profile,
                            )
                            if retry_reasons(candidate_row, args):
                                unresolved_indices.append(index)

                        if unresolved_indices:
                            logging.info(
                                "[%s] choice-anchored repair for %s unresolved sample(s)",
                                dataset,
                                len(unresolved_indices),
                            )
                            selection_prompts = [
                                render_prompt(
                                    tokenizer,
                                    build_choice_selection_messages(batch_rows[index][1]),
                                    args.use_chat_template,
                                )
                                for index in unresolved_indices
                            ]
                            selection_outputs = list(llm.generate(selection_prompts, choice_sampling, use_tqdm=False))
                            if len(selection_outputs) != len(unresolved_indices):
                                raise RuntimeError(
                                    "vLLM choice-selection count mismatch: "
                                    f"expected={len(unresolved_indices)} got={len(selection_outputs)}"
                                )
                            anchored_indices: list[int] = []
                            anchored_answers: list[str] = []
                            for index, selection_output in zip(unresolved_indices, selection_outputs):
                                selected_answer = extract_selected_option(selection_output, batch_rows[index][1])
                                if selected_answer is None:
                                    logging.warning(
                                        "[%s] choice-selection returned no valid option: row_idx=%s",
                                        dataset,
                                        batch_rows[index][0],
                                    )
                                    continue
                                anchored_indices.append(index)
                                anchored_answers.append(selected_answer)

                            if anchored_indices:
                                anchored_prompts = [
                                    render_prompt(
                                        tokenizer,
                                        build_generation_messages(
                                            args,
                                            batch_rows[index][1],
                                            format_retry=True,
                                            selected_answer=selected_answer,
                                            compact_retry=True,
                                        ),
                                        args.use_chat_template,
                                    )
                                    for index, selected_answer in zip(anchored_indices, anchored_answers)
                                ]
                                anchored_outputs = list(llm.generate(anchored_prompts, retry_sampling, use_tqdm=False))
                                if len(anchored_outputs) != len(anchored_indices):
                                    raise RuntimeError(
                                        "vLLM choice-anchored generation count mismatch: "
                                        f"expected={len(anchored_indices)} got={len(anchored_outputs)}"
                                    )
                                for index, anchored_output in zip(anchored_indices, anchored_outputs):
                                    outputs[index] = anchored_output
                                    attempt_counts[index] += 2
                                    prompt_variants[index] = "choice_anchored_decisive_repair"

                    response_texts: list[str | None] = [None] * len(outputs)
                    terminal_repair_sources: list[str | None] = [None] * len(outputs)
                    if args.prompt_profile == "paper_exact_terminal":
                        needs_choice: list[int] = []
                        for index, ((_, row), output) in enumerate(zip(batch_rows, outputs)):
                            choice = output.outputs[0] if getattr(output, "outputs", None) else None
                            raw_text = choice.text.strip() if choice is not None else ""
                            strict = parse_mcq_output_for_prompt_profile(
                                raw_text,
                                normalized_options(row),
                                args.prompt_profile,
                            )
                            if not strict.parse_errors and strict.final_answer:
                                response_texts[index] = raw_text
                                terminal_repair_sources[index] = "structured_primary"
                                continue
                            recovered = parse_paper_exact_mcq_output(raw_text, normalized_options(row))
                            if recovered.final_answer is not None:
                                response_texts[index] = append_paper_exact_terminal_answer(
                                    raw_text,
                                    normalized_options(row),
                                    recovered.final_answer,
                                )
                                terminal_repair_sources[index] = "canonicalized_primary_answer"
                            else:
                                needs_choice.append(index)

                        if needs_choice:
                            logging.info(
                                "[%s] constrained one-token terminal fallback for %s sample(s)",
                                dataset,
                                len(needs_choice),
                            )
                            continuation_prompts = []
                            for index in needs_choice:
                                output = outputs[index]
                                choice = output.outputs[0] if getattr(output, "outputs", None) else None
                                raw_text = choice.text.strip() if choice is not None else ""
                                rendered_prompt = str(getattr(output, "prompt", "") or batch_prompts[index])
                                continuation_prompts.append(
                                    f"{rendered_prompt}{raw_text.rstrip()}\nTherefore, the answer is ("
                                )
                            selection_outputs = list(
                                llm.generate(continuation_prompts, choice_sampling, use_tqdm=False)
                            )
                            if len(selection_outputs) != len(needs_choice):
                                raise RuntimeError(
                                    "Terminal fallback output count mismatch: "
                                    f"expected={len(needs_choice)} got={len(selection_outputs)}"
                                )
                            for index, selection_output in zip(needs_choice, selection_outputs):
                                selected = extract_selected_option(selection_output, batch_rows[index][1])
                                if selected is None:
                                    raise RuntimeError(
                                        f"Constrained terminal fallback returned no valid option: row_idx={batch_rows[index][0]}"
                                    )
                                choice = outputs[index].outputs[0] if getattr(outputs[index], "outputs", None) else None
                                raw_text = choice.text.strip() if choice is not None else ""
                                response_texts[index] = append_paper_exact_terminal_answer(
                                    raw_text,
                                    normalized_options(batch_rows[index][1]),
                                    selected,
                                )
                                terminal_repair_sources[index] = "constrained_one_token_fallback"
                                attempt_counts[index] += 1

                    for index, ((row_idx, row), output) in enumerate(zip(batch_rows, outputs)):
                        generated_row = output_row(
                            row_idx,
                            row,
                            dataset,
                            args.split,
                            output,
                            tokenizer,
                            attempt_counts[index],
                            prompt_variants[index],
                            prompt_version=prompt_version,
                            generation_policy_version=generation_policy_version,
                            preserve_raw_generation=args.prompt_profile in {
                                "paper_exact",
                                "paper_exact_terminal",
                                "paper_answer_format",
                            },
                            prompt_profile=args.prompt_profile,
                            response_text=response_texts[index],
                            terminal_repair_source=terminal_repair_sources[index],
                        )
                        handle.write(json.dumps(generated_row, ensure_ascii=False) + "\n")
                    handle.flush()
                    generated_count += len(batch_rows)
                    progress.update(len(batch_rows))
                    batch_rows = []
                    batch_prompts = []

                for row_idx, row in iter_selected_rows(info["source_path"], args):
                    identifier = sample_id(row_idx, row, dataset, args.split)
                    if identifier in done:
                        progress.update(1)
                        continue
                    options = normalized_options(row)
                    if not options:
                        logging.warning("[%s] skipping row without options: row_idx=%s sample_id=%s", dataset, row_idx, identifier)
                        progress.update(1)
                        continue
                    prompt = render_prompt(
                        tokenizer,
                        build_generation_messages(
                            args,
                            row,
                            format_retry=False,
                        ),
                        args.use_chat_template,
                    )
                    batch_rows.append((row_idx, row))
                    batch_prompts.append(prompt)
                    if len(batch_prompts) >= args.generation_batch_size:
                        flush_batch()
                flush_batch()
            progress.close()
            logging.info("[%s] generated %s new rows", dataset, generated_count)
            result_info[dataset] = {**info, "summary": summarize_output(output_path)}
    finally:
        del llm

    artifact_locations = {
        dataset: {
            "source_path": str(info["source_path"]),
            "output_path": str(info["output_path"]),
            "manifest_path": str(Path(info["target_dir"]) / "manifest.json"),
            "requested": info["requested"],
        }
        for dataset, info in result_info.items()
    }
    write_json(run_dir / "artifact_locations.json", artifact_locations)
    write_json(run_dir / "summary.json", {dataset: info["summary"] for dataset, info in result_info.items()})
    write_summary_markdown(run_dir / "summary.md", run_config, result_info)
    logging.info("No-RAG rationale generation complete. Run report: %s", run_dir)


if __name__ == "__main__":
    main()

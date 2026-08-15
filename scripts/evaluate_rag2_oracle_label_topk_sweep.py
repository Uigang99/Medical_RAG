#!/usr/bin/env python3
from __future__ import annotations

"""Compare RAG2 and hidden-state gold-label oracle filters on held-out questions.

The evaluator never runs a learned filter.  It deterministically samples the
question-level 8:1:1 test split, reuses the stored rerank list, applies each
gold-label policy to Top-k prefixes, and generates all answers with one common
paper-exact fixed-terminal prompt.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.core import BenchmarkSample, GenerationOutput, PromptRequest, RetrievedDocument
from medrag.evaluation import evaluate_prediction
from medrag.generation.transformers_generator import VLLMChatGenerator
from medrag.filtering.rag2_preanswer_text_hidden import (
    FINAL_ANSWER_PREFILL,
    PREANSWER_PROMPT_VERSION,
    build_preanswer_user_prompt,
)
from medrag.progress import StageProgress
from medrag.rag2_mcq import (
    PAPER_EXACT_TERMINAL_DOCUMENT_PROMPT_VERSION,
    PAPER_EXACT_TERMINAL_PROMPT_VERSION,
    append_paper_exact_terminal_answer,
    build_paper_exact_terminal_documents_messages,
    build_paper_exact_terminal_no_rag_messages,
    normalized_options,
    parse_mcq_output_for_prompt_profile,
    parse_paper_exact_mcq_output,
)
from medrag.rag2_oracle import (
    canonicalize_rag2_labels,
    deterministic_question_sample,
    hidden_policy_name,
    load_hidden_projection_labels,
    oracle_document_is_helpful,
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2"
)
DEFAULT_CANDIDATE_ROOT = ARTIFACT_ROOT / "candidates/quality_selected_source_balanced40_rerank32_v1"
DEFAULT_RAG2_LABEL_ROOT = ARTIFACT_ROOT / "filter_training_inputs_top10_independent_ppl_v2_corrected_nodoc"
DEFAULT_HIDDEN_LABEL_ROOT = ARTIFACT_ROOT / "preanswer_hidden_labels_full_top8_tau0_v1"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/rag2_oracle_label_topk_4994_v1"
DATASETS = ("medmcqa", "medqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle Top-k evaluation on a deterministic held-out subset: RAG2 Helpful, "
            "hidden projection > 0, and hidden projection > 0.2."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--label-split", default="test")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-file", default="candidates_top32.jsonl")
    parser.add_argument("--rag2-label-root", type=Path, default=DEFAULT_RAG2_LABEL_ROOT)
    parser.add_argument("--hidden-label-root", type=Path, default=DEFAULT_HIDDEN_LABEL_ROOT)
    parser.add_argument("--medmcqa-question-limit", type=int, default=4000)
    parser.add_argument("--medqa-question-limit", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--hidden-thresholds", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--include-rag2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--answer-decision-mode",
        choices=("paper_exact_terminal", "constrained_choice"),
        default="paper_exact_terminal",
        help=(
            "paper_exact_terminal generates a rationale and fixed terminal answer; "
            "constrained_choice uses the same no-rationale prompt and A/B/C/D decision "
            "space as hidden-state label extraction."
        ),
    )
    parser.add_argument("--llm-model-path", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--reuse-no-rag-path",
        type=Path,
        default=None,
        help="Reuse a complete no-RAG JSONL generated with the same answer-decision mode.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-doc-chars", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--llm-max-model-len", type=int, default=8192)
    parser.add_argument("--gdn-prefill-backend", choices=("auto", "flashinfer", "triton", "cutedsl"), default="triton")
    parser.add_argument("--vllm-performance-mode", choices=("balanced", "interactivity", "throughput"), default="throughput")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=80)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def document_from_dict(row: dict[str, Any]) -> RetrievedDocument:
    return RetrievedDocument(
        source=str(row.get("source") or ""),
        local_id=int(row.get("local_id", -1)),
        db_id=str(row.get("db_id") or ""),
        corpus_id=row.get("corpus_id"),
        chunk_id=row.get("chunk_id"),
        doc_id=row.get("doc_id"),
        title=row.get("title"),
        text=str(row.get("text") or ""),
        retrieval_score=float(row.get("retrieval_score", 0.0)),
        retrieval_rank=row.get("retrieval_rank"),
        rerank_score=row.get("rerank_score"),
        rerank_rank=row.get("rerank_rank"),
    )


def sample_from_record(row: dict[str, Any]) -> BenchmarkSample:
    options = {str(key).upper(): str(value) for key, value in (row.get("options") or {}).items()}
    answer = str(row.get("answer") or row.get("gold_answer") or "").upper() or None
    answers = [
        str(value).upper()
        for value in (row.get("answers") or row.get("gold_answers") or [answer])
        if value
    ]
    raw = {
        "question": str(row.get("question") or ""),
        "options": options,
        "answer": answer,
        "answers": answers,
    }
    return BenchmarkSample(
        row_idx=int(row.get("row_idx", -1)),
        id=str(row["sample_id"]),
        task="mcq",
        collection="rag2_filter_training_test_oracle",
        dataset=str(row["dataset"]),
        split="test",
        question=raw["question"],
        options=options,
        answer=answer,
        answers=answers,
        raw=raw,
    )


def selected_ids_for_dataset(args: argparse.Namespace, dataset: str) -> list[str]:
    path = args.rag2_label_root / dataset / "sample_ids" / f"{args.label_split}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = path.read_text(encoding="utf-8").splitlines()
    limit = args.medmcqa_question_limit if dataset == "medmcqa" else args.medqa_question_limit
    return deterministic_question_sample(values, dataset=dataset, limit=limit, seed=args.sample_seed)


def load_artifacts(args: argparse.Namespace) -> tuple[
    list[BenchmarkSample],
    dict[str, list[RetrievedDocument]],
    dict[str, dict[str, str]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    maximum_rank = max(args.top_k_values)
    samples: list[BenchmarkSample] = []
    candidates: dict[str, list[RetrievedDocument]] = {}
    rag2_labels: dict[str, dict[str, str]] = {}
    hidden_scores: dict[str, dict[str, float]] = {}
    audit: dict[str, Any] = {"datasets": {}}

    for dataset in args.datasets:
        selected = selected_ids_for_dataset(args, dataset)
        selected_set = set(selected)
        candidate_path = args.candidate_root / dataset / args.source_split / args.candidate_file
        rag2_path = args.rag2_label_root / dataset / "labels_all.jsonl"
        hidden_path = args.hidden_label_root / dataset / "hidden_labels.jsonl"
        for path in (candidate_path, rag2_path, hidden_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        candidate_rows = {
            str(row["sample_id"]): row
            for row in iter_jsonl(candidate_path)
            if str(row.get("sample_id") or "") in selected_set
        }
        missing = selected_set - set(candidate_rows)
        if missing:
            raise RuntimeError(f"[{dataset}] candidate rows missing: {len(missing)} first={sorted(missing)[:3]}")

        dataset_rag2, rag2_audit = canonicalize_rag2_labels(
            iter_jsonl(rag2_path), selected_sample_ids=selected_set, max_rank=maximum_rank
        )
        dataset_hidden, hidden_audit = load_hidden_projection_labels(
            iter_jsonl(hidden_path), selected_sample_ids=selected_set, max_rank=maximum_rank
        )

        for sample_id in selected:
            row = candidate_rows[sample_id]
            documents = [document_from_dict(value) for value in (row.get("candidate_documents") or [])]
            documents.sort(key=lambda value: int(value.rerank_rank or 10**9))
            documents = documents[:maximum_rank]
            if len(documents) != maximum_rank:
                raise RuntimeError(
                    f"[{dataset}] {sample_id} expected Top-{maximum_rank}, got {len(documents)}"
                )
            actual_ranks = [int(document.rerank_rank or 0) for document in documents]
            if actual_ranks != list(range(1, maximum_rank + 1)):
                raise RuntimeError(f"[{dataset}] non-canonical rerank prefix for {sample_id}: {actual_ranks}")
            for document in documents:
                if document.stable_id not in dataset_rag2.get(sample_id, {}):
                    raise RuntimeError(f"Missing RAG2 label: {sample_id} {document.stable_id}")
                if document.stable_id not in dataset_hidden.get(sample_id, {}):
                    raise RuntimeError(f"Missing hidden score: {sample_id} {document.stable_id}")
            samples.append(sample_from_record(row))
            candidates[sample_id] = documents

        rag2_labels.update(dataset_rag2)
        hidden_scores.update(dataset_hidden)
        audit["datasets"][dataset] = {
            "available_test_questions": len(
                (args.rag2_label_root / dataset / "sample_ids" / f"{args.label_split}.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            "selected_questions": len(selected),
            "selected_pair_count": len(selected) * maximum_rank,
            "rag2": rag2_audit,
            "hidden": hidden_audit,
        }

    samples.sort(key=lambda value: (DATASETS.index(value.dataset), value.row_idx, value.id))
    audit["total_questions"] = len(samples)
    audit["total_pairs"] = sum(len(candidates[sample.id]) for sample in samples)
    return samples, candidates, rag2_labels, hidden_scores, audit


def policies(args: argparse.Namespace) -> list[tuple[str, float | None]]:
    result: list[tuple[str, float | None]] = []
    if args.include_rag2:
        result.append(("rag2", None))
    seen = set()
    for threshold in args.hidden_thresholds:
        value = float(threshold)
        if value < 0:
            raise ValueError("Hidden thresholds must be non-negative")
        if value in seen:
            continue
        seen.add(value)
        result.append((hidden_policy_name(value), value))
    if not result:
        raise ValueError("At least one oracle policy must be requested")
    return result


def selected_documents(
    sample_id: str,
    prefix: list[RetrievedDocument],
    *,
    policy: str,
    hidden_threshold: float | None,
    rag2_labels: dict[str, dict[str, str]],
    hidden_scores: dict[str, dict[str, float]],
) -> list[RetrievedDocument]:
    return [
        document
        for document in prefix
        if oracle_document_is_helpful(
            policy=policy,
            rag2_label=rag2_labels[sample_id].get(document.stable_id),
            hidden_projection=hidden_scores[sample_id].get(document.stable_id),
            hidden_threshold=hidden_threshold,
        )
    ]


def prompt_versions(args: argparse.Namespace) -> dict[str, str]:
    if args.answer_decision_mode == "constrained_choice":
        return {
            "no_rag": PREANSWER_PROMPT_VERSION,
            "documents": PREANSWER_PROMPT_VERSION,
        }
    return {
        "no_rag": PAPER_EXACT_TERMINAL_PROMPT_VERSION,
        "documents": PAPER_EXACT_TERMINAL_DOCUMENT_PROMPT_VERSION,
    }


def prompt_request(
    args: argparse.Namespace,
    sample: BenchmarkSample,
    documents: list[RetrievedDocument],
    *,
    case_id: str,
    max_doc_chars: int,
) -> PromptRequest:
    document_rows = [document.to_dict(include_text=True) for document in documents]
    if args.answer_decision_mode == "constrained_choice":
        rendered_documents: list[str] = []
        for row in document_rows:
            text = " ".join(str(row.get("text") or row.get("title") or "").split())
            if max_doc_chars > 0 and len(text) > max_doc_chars:
                text = text[: max_doc_chars - 3].rstrip() + "..."
            if text:
                rendered_documents.append(text)
        context = "\n\n".join(rendered_documents) or None
        return PromptRequest(
            sample_id=sample.id,
            case_id=case_id,
            messages=[{"role": "user", "content": build_preanswer_user_prompt(sample, context)}],
            metadata={"structured_regex": r" (A|B|C|D)"},
        )
    messages = (
        build_paper_exact_terminal_documents_messages(
            sample.raw,
            document_rows,
            max_doc_chars=max_doc_chars,
        )
        if document_rows
        else build_paper_exact_terminal_no_rag_messages(sample.raw)
    )
    return PromptRequest(
        sample_id=sample.id,
        case_id=case_id,
        messages=messages,
    )


def build_generator(args: argparse.Namespace) -> VLLMChatGenerator:
    direct_choice = args.answer_decision_mode == "constrained_choice"
    return VLLMChatGenerator(
        model_path=args.llm_model_path,
        max_new_tokens=1 if direct_choice else args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["<|im_end|>", "<|eot_id|>"],
        bad_words=["```", "```python", "```text"],
        use_chat_template=True,
        use_tqdm=False,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.llm_max_model_len,
        gdn_prefill_backend=args.gdn_prefill_backend,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        performance_mode=args.vllm_performance_mode,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        assistant_prefill=FINAL_ANSWER_PREFILL if direct_choice else None,
    )


def result_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["policy"]), int(row["top_k"]), str(row["sample_id"])


def load_result_rows(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    return {result_key(row): row for row in iter_jsonl(path)}


def load_no_rag_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in iter_jsonl(path)}


def load_reused_no_rag_rows(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_no_rag_rows(path)
    expected_ids = {sample.id for sample in samples}
    actual_ids = set(rows)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Reused no-RAG cache does not match cohort: expected={len(expected_ids)} "
            f"actual={len(actual_ids)} missing={len(expected_ids - actual_ids)} "
            f"extra={len(actual_ids - expected_ids)} path={path}"
        )
    expected_prompt = prompt_versions(args)["no_rag"]
    invalid_prompt = [
        sample_id
        for sample_id, row in rows.items()
        if str(row.get("prompt_version") or "") != expected_prompt
    ]
    if invalid_prompt:
        raise RuntimeError(
            f"Reused no-RAG cache has the wrong prompt contract for "
            f"{len(invalid_prompt)} row(s): expected={expected_prompt} "
            f"first={invalid_prompt[:3]} path={path}"
        )
    invalid_mode = [
        sample_id
        for sample_id, row in rows.items()
        if row.get("answer_decision_mode") not in (None, args.answer_decision_mode)
    ]
    if invalid_mode:
        raise RuntimeError(
            f"Reused no-RAG cache has the wrong answer-decision mode for "
            f"{len(invalid_mode)} row(s): first={invalid_mode[:3]} path={path}"
        )
    logging.info("Reusing complete no-RAG cache: %s (%s rows)", path, len(rows))
    return rows


def parse_generation(
    args: argparse.Namespace,
    sample: BenchmarkSample,
    text: str,
) -> tuple[str, list[str]]:
    if args.answer_decision_mode == "constrained_choice":
        answer = extract_constrained_option(text, sample)
        if answer is None:
            raise RuntimeError(
                f"Constrained direct-choice generation failed for {sample.id}: {text!r}"
            )
        return answer, []
    parsed = parse_mcq_output_for_prompt_profile(text, normalized_options(sample.raw), "paper_exact_terminal")
    if not parsed.final_answer or parsed.parse_errors:
        raise RuntimeError(
            f"Structured terminal generation failed for {sample.id}: answer={parsed.final_answer} "
            f"errors={parsed.parse_errors} text={text[-500:]!r}"
        )
    return parsed.final_answer, parsed.parse_errors


def finalize_generations(
    args: argparse.Namespace,
    generator: VLLMChatGenerator,
    samples: list[BenchmarkSample],
    generations: list[GenerationOutput],
) -> list[tuple[GenerationOutput, str, str]]:
    if args.answer_decision_mode == "paper_exact_terminal":
        return repair_terminal_generations(generator, samples, generations)
    if len(samples) != len(generations):
        raise RuntimeError(
            f"Direct-choice output mismatch: samples={len(samples)} generations={len(generations)}"
        )
    finalized: list[tuple[GenerationOutput, str, str]] = []
    for sample, generation in zip(samples, generations):
        prediction, _ = parse_generation(args, sample, generation.text)
        finalized.append((generation, prediction, "constrained_choice"))
    return finalized


def extract_constrained_option(text: str, sample: BenchmarkSample) -> str | None:
    valid = set(sample.options or {})
    for match in re.finditer(r"\b([A-Z])\b", str(text or "").upper()):
        if match.group(1) in valid:
            return match.group(1)
    return None


def repair_terminal_generations(
    generator: VLLMChatGenerator,
    samples: list[BenchmarkSample],
    generations: list[GenerationOutput],
) -> list[tuple[GenerationOutput, str, str]]:
    """Guarantee one canonical terminal answer without changing reasoning.

    Constraining an arbitrary 768-token rationale with one regex can enter a
    valid-but-nonterminating loop in XGrammar.  Generate the rationale freely,
    canonicalize an answer already present in it, and only when no answer is
    recoverable ask the same model for one constrained A/B/C/D continuation.
    """

    if len(samples) != len(generations):
        raise RuntimeError(
            f"Terminal repair input mismatch: samples={len(samples)} generations={len(generations)}"
        )

    repaired: list[tuple[GenerationOutput, str, str] | None] = [None] * len(generations)
    unresolved: list[int] = []
    for index, (sample, generation) in enumerate(zip(samples, generations)):
        options = normalized_options(sample.raw)
        strict = parse_mcq_output_for_prompt_profile(
            generation.text, options, "paper_exact_terminal"
        )
        if strict.final_answer and not strict.parse_errors:
            repaired[index] = (generation, strict.final_answer, "exact_primary")
            continue
        recovered = parse_paper_exact_mcq_output(generation.text, options)
        if recovered.final_answer is not None:
            canonical = append_paper_exact_terminal_answer(
                generation.text, options, recovered.final_answer
            )
            repaired[index] = (
                GenerationOutput(
                    text=canonical,
                    prompt=generation.prompt,
                    raw_text=generation.text,
                    finish_reason=generation.finish_reason,
                    stop_reason=generation.stop_reason,
                ),
                recovered.final_answer,
                "canonicalized_primary_answer",
            )
            continue
        unresolved.append(index)

    if unresolved:
        logging.info(
            "Constrained one-token terminal fallback for %s/%s oracle answer(s).",
            len(unresolved),
            len(generations),
        )
        prefixes = [
            f"{generations[index].prompt}{generations[index].text.rstrip()}\nTherefore, the answer is ("
            for index in unresolved
        ]
        choices = generator.generate_allowed_single_token_continuations(prefixes)
        if len(choices) != len(unresolved):
            raise RuntimeError(
                f"Terminal fallback output mismatch: expected={len(unresolved)} got={len(choices)}"
            )
        for index, choice in zip(unresolved, choices):
            sample = samples[index]
            selected = extract_constrained_option(choice.text, sample)
            if selected is None:
                raise RuntimeError(
                    f"Constrained terminal fallback returned no option for {sample.id}: {choice.text!r}"
                )
            primary = generations[index]
            canonical = append_paper_exact_terminal_answer(
                primary.text, normalized_options(sample.raw), selected
            )
            repaired[index] = (
                GenerationOutput(
                    text=canonical,
                    prompt=primary.prompt,
                    raw_text=primary.text,
                    finish_reason=primary.finish_reason,
                    stop_reason=primary.stop_reason,
                ),
                selected,
                "constrained_one_token_fallback",
            )

    if any(value is None for value in repaired):
        raise RuntimeError("Internal error: missing repaired terminal generation")
    return [value for value in repaired if value is not None]


def generate_no_rag(
    args: argparse.Namespace,
    generator: VLLMChatGenerator,
    samples: list[BenchmarkSample],
    path: Path,
) -> dict[str, dict[str, Any]]:
    completed = load_no_rag_rows(path) if args.resume else {}
    pending = [sample for sample in samples if sample.id not in completed]
    progress = StageProgress(total=len(samples), desc="OracleNoRAG", enabled=True)
    progress.update(len(samples) - len(pending))
    try:
        mode = "a" if args.resume else "w"
        with path.open(mode, encoding="utf-8", buffering=16 * 1024 * 1024) as output:
            for start in range(0, len(pending), args.generation_batch_size):
                batch = pending[start : start + args.generation_batch_size]
                requests = [
                    prompt_request(
                        args,
                        sample,
                        [],
                        case_id="no_rag",
                        max_doc_chars=args.max_doc_chars,
                    )
                    for sample in batch
                ]
                primary_generations = generator.generate_batch(requests)
                generations = finalize_generations(args, generator, batch, primary_generations)
                for sample, (generation, prediction, repair_source) in zip(batch, generations):
                    _, errors = parse_generation(args, sample, generation.text)
                    row = {
                        "sample_id": sample.id,
                        "dataset": sample.dataset,
                        "row_idx": sample.row_idx,
                        "answers": sample.answers,
                        "prediction": prediction,
                        "correct": bool(evaluate_prediction(sample, prediction)["correct"]),
                        "raw_prediction": generation.text,
                        "terminal_primary_generation": generation.raw_text,
                        "terminal_repair_source": repair_source,
                        "generation_attempts": 1 + int(
                            repair_source == "constrained_one_token_fallback"
                        ),
                        "parse_errors": errors,
                        "prompt_version": prompt_versions(args)["no_rag"],
                        "answer_decision_mode": args.answer_decision_mode,
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed[sample.id] = row
                output.flush()
                progress.update(len(batch))
    finally:
        progress.close()
    if len(completed) != len(samples):
        raise RuntimeError(f"Incomplete no-RAG cache: expected={len(samples)} got={len(completed)}")
    return completed


def generate_oracle_conditions(
    args: argparse.Namespace,
    generator: VLLMChatGenerator,
    samples: list[BenchmarkSample],
    candidates: dict[str, list[RetrievedDocument]],
    rag2_labels: dict[str, dict[str, str]],
    hidden_scores: dict[str, dict[str, float]],
    no_rag_rows: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    conditions = [(policy, threshold, top_k) for policy, threshold in policies(args) for top_k in args.top_k_values]
    expected_keys = {
        (policy, top_k, sample.id)
        for policy, _, top_k in conditions
        for sample in samples
    }
    loaded = load_result_rows(output_path) if args.resume else {}
    completed = {key: row for key, row in loaded.items() if key in expected_keys}
    total = len(samples) * len(conditions)
    done = sum((policy, top_k, sample.id) in completed for policy, _, top_k in conditions for sample in samples)
    progress = StageProgress(total=total, desc="OracleLabelSweep", enabled=True)
    progress.update(done)
    try:
        mode = "a" if args.resume else "w"
        with output_path.open(mode, encoding="utf-8", buffering=16 * 1024 * 1024) as output:
            for policy, threshold, top_k in conditions:
                pending = [sample for sample in samples if (policy, top_k, sample.id) not in completed]
                logging.info(
                    "Oracle condition policy=%s threshold=%s top_k=%s pending=%s cached=%s",
                    policy,
                    threshold,
                    top_k,
                    len(pending),
                    len(samples) - len(pending),
                )
                for start in range(0, len(pending), args.generation_batch_size):
                    batch = pending[start : start + args.generation_batch_size]
                    requests: list[PromptRequest] = []
                    generated: list[tuple[BenchmarkSample, list[RetrievedDocument], list[RetrievedDocument]]] = []
                    immediate: list[dict[str, Any]] = []
                    for sample in batch:
                        prefix = candidates[sample.id][:top_k]
                        context = selected_documents(
                            sample.id,
                            prefix,
                            policy=policy,
                            hidden_threshold=threshold,
                            rag2_labels=rag2_labels,
                            hidden_scores=hidden_scores,
                        )
                        if not context:
                            baseline = no_rag_rows[sample.id]
                            immediate.append(
                                {
                                    "policy": policy,
                                    "hidden_threshold": threshold,
                                    "top_k": top_k,
                                    "sample_id": sample.id,
                                    "dataset": sample.dataset,
                                    "row_idx": sample.row_idx,
                                    "answers": sample.answers,
                                    "prediction": baseline["prediction"],
                                    "correct": baseline["correct"],
                                    "raw_prediction": baseline["raw_prediction"],
                                    "parse_errors": baseline["parse_errors"],
                                    "rerank_prefix_doc_ids": [document.stable_id for document in prefix],
                                    "context_doc_ids": [],
                                    "context_document_count": 0,
                                    "zero_context_fallback": True,
                                    "prompt_version": prompt_versions(args)["no_rag"],
                                    "answer_decision_mode": args.answer_decision_mode,
                                }
                            )
                            continue
                        requests.append(
                            prompt_request(
                                args,
                                sample,
                                context,
                                case_id=f"{policy}_top{top_k}",
                                max_doc_chars=args.max_doc_chars,
                            )
                        )
                        generated.append((sample, prefix, context))

                    for row in immediate:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed[result_key(row)] = row

                    primary_generations = generator.generate_batch(requests) if requests else []
                    generations = finalize_generations(
                        args,
                        generator,
                        [sample for sample, _, _ in generated],
                        primary_generations,
                    )
                    for (sample, prefix, context), (generation, prediction, repair_source) in zip(
                        generated, generations
                    ):
                        _, errors = parse_generation(args, sample, generation.text)
                        row = {
                            "policy": policy,
                            "hidden_threshold": threshold,
                            "top_k": top_k,
                            "sample_id": sample.id,
                            "dataset": sample.dataset,
                            "row_idx": sample.row_idx,
                            "answers": sample.answers,
                            "prediction": prediction,
                            "correct": bool(evaluate_prediction(sample, prediction)["correct"]),
                            "raw_prediction": generation.text,
                            "terminal_primary_generation": generation.raw_text,
                            "terminal_repair_source": repair_source,
                            "generation_attempts": 1 + int(
                                repair_source == "constrained_one_token_fallback"
                            ),
                            "parse_errors": errors,
                            "rerank_prefix_doc_ids": [document.stable_id for document in prefix],
                            "context_doc_ids": [document.stable_id for document in context],
                            "context_document_count": len(context),
                            "zero_context_fallback": False,
                            "prompt_version": prompt_versions(args)["documents"],
                            "answer_decision_mode": args.answer_decision_mode,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed[result_key(row)] = row
                    output.flush()
                    progress.update(len(batch))
    finally:
        progress.close()
    if len(completed) != total:
        raise RuntimeError(f"Incomplete oracle output: expected={total} got={len(completed)}")
    return completed


def metric(rows: list[dict[str, Any]], no_rag: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    docs = [int(row["context_document_count"]) for row in rows]
    gains = sum(not bool(no_rag[row["sample_id"]]["correct"]) and bool(row["correct"]) for row in rows)
    losses = sum(bool(no_rag[row["sample_id"]]["correct"]) and not bool(row["correct"]) for row in rows)
    return {
        "questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "mean_context_documents": sum(docs) / total if total else 0.0,
        "zero_context_questions": sum(value == 0 for value in docs),
        "zero_context_rate": sum(value == 0 for value in docs) / total if total else 0.0,
        "gains_vs_no_rag": gains,
        "losses_vs_no_rag": losses,
        "net_gain_vs_no_rag": gains - losses,
        "invalid": sum(bool(row.get("parse_errors")) for row in rows),
    }


def summarize(
    args: argparse.Namespace,
    rows: dict[tuple[str, int, str], dict[str, Any]],
    no_rag: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in no_rag.values():
        baseline_by_dataset[str(row["dataset"])].append(row)
    baseline = {
        dataset: {
            "questions": len(values),
            "correct": sum(bool(row["correct"]) for row in values),
            "accuracy": sum(bool(row["correct"]) for row in values) / len(values),
        }
        for dataset, values in sorted(baseline_by_dataset.items())
    }
    all_baseline = list(no_rag.values())
    baseline["overall"] = {
        "questions": len(all_baseline),
        "correct": sum(bool(row["correct"]) for row in all_baseline),
        "accuracy": sum(bool(row["correct"]) for row in all_baseline) / len(all_baseline),
    }

    conditions: dict[str, Any] = {}
    for policy, threshold in policies(args):
        conditions[policy] = {"hidden_threshold": threshold, "top_k": {}}
        for top_k in sorted(set(args.top_k_values)):
            selected = [row for (row_policy, row_k, _), row in rows.items() if row_policy == policy and row_k == top_k]
            by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                by_dataset[str(row["dataset"])].append(row)
            conditions[policy]["top_k"][str(top_k)] = {
                "overall": metric(selected, no_rag),
                "datasets": {
                    dataset: metric(values, no_rag) for dataset, values in sorted(by_dataset.items())
                },
            }
    return {
        "answer_decision_mode": args.answer_decision_mode,
        "prompt_versions": prompt_versions(args),
        "baseline": baseline,
        "conditions": conditions,
    }


def write_pretty_summary(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    answer_mode = str(summary["answer_decision_mode"])
    lines = [
        "RAG2 vs Hidden-State Gold-Label Oracle Top-k Sweep",
        "",
        f"Common final answer protocol: {answer_mode}",
        "",
        "| Policy | Top-k | Dataset | Questions | Correct | Accuracy | Avg docs | Zero docs | Gains | Losses | Net |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, policy_values in summary["conditions"].items():
        for top_k, values in sorted(policy_values["top_k"].items(), key=lambda item: int(item[0])):
            for dataset, metrics in [("overall", values["overall"]), *values["datasets"].items()]:
                lines.append(
                    f"| {policy} | {top_k} | {dataset} | {metrics['questions']:,} | "
                    f"{metrics['correct']:,} | {metrics['accuracy'] * 100:.2f}% | "
                    f"{metrics['mean_context_documents']:.2f} | {metrics['zero_context_questions']:,} | "
                    f"{metrics['gains_vs_no_rag']:,} | {metrics['losses_vs_no_rag']:,} | "
                    f"{metrics['net_gain_vs_no_rag']:+,} |"
                )
    lines.extend(
        [
            "",
            "No-RAG baseline",
            "",
            "| Dataset | Questions | Correct | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset, values in [("overall", baseline["overall"]), *[(key, baseline[key]) for key in sorted(set(baseline) - {"overall"})]]:
        lines.append(
            f"| {dataset} | {values['questions']:,} | {values['correct']:,} | {values['accuracy'] * 100:.2f}% |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if not args.top_k_values or any(value not in {1, 2, 4, 8} for value in args.top_k_values):
        raise ValueError("--top-k-values must be selected from 1 2 4 8")
    requested_policies = policies(args)
    if not args.llm_model_path.exists() and not args.dry_run:
        raise FileNotFoundError(args.llm_model_path)

    samples, candidates, rag2_labels, hidden_scores, audit = load_artifacts(args)
    if len({sample.id for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate questions in selected oracle cohort")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    no_rag_cache_path = args.reuse_no_rag_path or (args.run_dir / "no_rag_results.jsonl")
    manifest = {
        "version": "rag2_gold_label_oracle_topk_v2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets": args.datasets,
        "question_limits": {"medmcqa": args.medmcqa_question_limit, "medqa": args.medqa_question_limit},
        "sample_seed": args.sample_seed,
        "selected_sample_ids": [sample.id for sample in samples],
        "top_k_values": sorted(set(args.top_k_values)),
        "policies": [
            {"name": name, "hidden_threshold": threshold}
            for name, threshold in requested_policies
        ],
        "answer_decision_mode": args.answer_decision_mode,
        "prompt_versions": prompt_versions(args),
        "no_rag_cache_path": str(no_rag_cache_path.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "rag2_label_root": str(args.rag2_label_root.resolve()),
        "hidden_label_root": str(args.hidden_label_root.resolve()),
        "audit": audit,
    }
    manifest_path = args.run_dir / "selection_manifest.json"
    if manifest_path.exists() and args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_keys = [
            "version", "datasets", "question_limits", "sample_seed", "selected_sample_ids",
            "top_k_values", "policies", "answer_decision_mode", "prompt_versions",
            "no_rag_cache_path", "candidate_root", "rag2_label_root", "hidden_label_root",
        ]
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise RuntimeError(f"Existing run manifest is incompatible with requested run: {manifest_path}")
    else:
        atomic_json(manifest_path, manifest)

    logging.info(
        "Oracle cohort ready: questions=%s pairs=%s policies=%s top_k=%s",
        len(samples),
        audit["total_pairs"],
        [name for name, _ in requested_policies],
        sorted(set(args.top_k_values)),
    )
    for dataset, values in audit["datasets"].items():
        logging.info(
            "[%s] selected=%s/%s pairs=%s RAG2=%s hidden=%s",
            dataset,
            values["selected_questions"],
            values["available_test_questions"],
            values["selected_pair_count"],
            values["rag2"],
            values["hidden"],
        )
    if args.dry_run:
        logging.info("Dry-run contract validation complete: %s", manifest_path)
        return

    generator = build_generator(args)
    try:
        if args.reuse_no_rag_path is not None:
            no_rag = load_reused_no_rag_rows(args, samples, args.reuse_no_rag_path)
        else:
            no_rag = generate_no_rag(
                args,
                generator,
                samples,
                args.run_dir / "no_rag_results.jsonl",
            )
        rows = generate_oracle_conditions(
            args,
            generator,
            samples,
            candidates,
            rag2_labels,
            hidden_scores,
            no_rag,
            args.run_dir / "oracle_results.jsonl",
        )
    finally:
        generator.close()

    summary = summarize(args, rows, no_rag)
    atomic_json(args.run_dir / "summary.json", summary)
    write_pretty_summary(args.run_dir / "summary_table_pretty.txt", summary)
    logging.info("Oracle Top-k evaluation complete: %s", args.run_dir)


if __name__ == "__main__":
    main()

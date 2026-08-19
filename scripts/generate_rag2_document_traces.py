from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.io_utils import write_json
from medrag.progress import StageProgress
from medrag.rag2_generation import (
    GENERATION_POLICY_VERSION,
    PPL_SCOPE_VERSION,
    flatten_generation_stats,
    generation_stats,
    render_prompt,
)
from medrag.rag2_mcq import (
    DOCUMENT_PROMPT_VERSION,
    PAPER_ANSWER_FORMAT_DOCUMENT_PROMPT_VERSION,
    PAPER_ANSWER_FORMAT_PROMPT_VERSION,
    PAPER_EXACT_DOCUMENT_PROMPT_VERSION,
    PAPER_EXACT_PROMPT_VERSION,
    PROMPT_VERSION,
    build_document_choice_selection_messages,
    build_document_messages,
    build_paper_answer_format_document_messages,
    build_paper_exact_documents_messages,
    clean_text,
    normalized_options,
    parse_mcq_output_for_prompt_profile,
)


PAPER_ANSWER_FORMAT_GENERATION_POLICY_VERSION = "rag2_llama3_paper_answer_format_greedy_v2"
PAPER_EXACT_GENERATION_POLICY_VERSION = "rag2_llama3_paper_exact_greedy_v1"
WINDOW_ANNOTATION_SELECTION_VERSION = "rag2_direct_window_candidates_attr_top2_counter_medcpt_v1"
SENTENCE_ANNOTATION_SELECTION_VERSION = "rag2_direct_sentence_candidates_attr_top2_counter_medcpt_v1"


DEFAULT_LLM = WORKSPACE_ROOT / "models" / "Qwen3.5-9B"
ANSWER_CONCLUSION_PATTERN = re.compile(r"(?is)\btherefore\s*,?\s+the\s+answer\s+is\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RAG2 single-document rationale traces from reranked candidates and a fixed no-RAG baseline."
    )
    parser.add_argument("--candidates-path", type=Path, required=True)
    parser.add_argument(
        "--candidate-input-format",
        choices=["retrieval_documents", "window_annotation_candidates", "sentence_annotation_candidates"],
        default="retrieval_documents",
        help=(
            "retrieval_documents reads the historical question-level candidate_documents schema. "
            "window_annotation_candidates reads rows produced by select_rag2_window_annotation_candidates.py; "
            "sentence_annotation_candidates reads rows produced by select_rag2_sentence_annotation_candidates.py. "
            "Both independently generate one trace per selected evidence unit."
        ),
    )
    parser.add_argument(
        "--expected-window-selection-version",
        default=WINDOW_ANNOTATION_SELECTION_VERSION,
        help="Reject candidate-window rows from a different selection protocol; empty disables the check.",
    )
    parser.add_argument(
        "--expected-sentence-selection-version",
        default=SENTENCE_ANNOTATION_SELECTION_VERSION,
        help="Reject sentence-candidate rows from a different selection protocol; empty disables the check.",
    )
    parser.add_argument("--no-rag-path", type=Path, required=True)
    parser.add_argument(
        "--quality-selection-path",
        type=Path,
        default=None,
        help=(
            "Optional usable_rows.jsonl from audit_rag2_no_rag_quality_selection.py. "
            "For paper_exact, this is the authoritative no-RAG question/answer selection used to build candidates."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None, help="Label used in logs and the trace manifest.")
    parser.add_argument("--output-file", default="pseudo_label_traces.jsonl")
    parser.add_argument("--llm-model-path", type=Path, default=DEFAULT_LLM)
    parser.add_argument(
        "--prompt-profile",
        choices=["legacy_document", "paper_exact", "paper_answer_format"],
        default="legacy_document",
        help=(
            "Use the legacy Qwen document prompt, the verbatim RAG2 paper prompt with a raw evidence chunk appended, "
            "or the paper prompt with a fixed final-answer contract."
        ),
    )
    parser.add_argument("--docs-per-question", type=int, default=10)
    parser.add_argument(
        "--candidate-query-alignment",
        choices=["strict", "identity_only"],
        default="strict",
        help=(
            "strict requires candidates to have been retrieved with this no-RAG response. "
            "identity_only is for frozen-candidate oracle audits: it validates sample/question/options "
            "but preserves the different query that originally selected the documents."
        ),
    )
    parser.add_argument("--max-doc-chars", type=int, default=2600)
    parser.add_argument("--question-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--length-retry-attempts",
        type=int,
        default=0,
        help="Retry only generations that terminate with finish_reason=length.",
    )
    parser.add_argument(
        "--length-retry-max-new-tokens",
        type=int,
        default=384,
        help="Generation cap for a compact rewrite when the primary generation reaches its length limit.",
    )
    parser.add_argument("--invalid-retry-attempts", type=int, default=0)
    parser.add_argument("--invalid-retry-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--retry-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retry only malformed outputs or rows without a usable rationale PPL trace.",
    )
    parser.add_argument("--quality-retry-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--choice-anchored-retry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For unresolved format-quality cases, select an option from the question and document before regenerating.",
    )
    parser.add_argument("--choice-selection-max-new-tokens", type=int, default=4)
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
        default="triton",
        help="Qwen GDN backend; ignored by Llama and other non-GDN model families.",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-performance-mode", choices=["balanced", "interactivity", "throughput"], default="throughput")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", type=int, default=0, help="Inclusive candidate question row.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive candidate question row.")
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate candidate/no-RAG alignment and write the manifest without loading vLLM or generating outputs.",
    )
    parser.add_argument(
        "--force-pairs-path",
        type=Path,
        default=None,
        help="JSONL target list with pair_id values. Only these pairs are regenerated, regardless of --resume state.",
    )
    parser.add_argument(
        "--resume-forced-pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When forcing pairs, reuse valid pair IDs already present in the output instead of regenerating them.",
    )
    parser.add_argument(
        "--exception-output-path",
        type=Path,
        default=None,
        help="Write pairs still failing quality checks after all retries to this JSONL file.",
    )
    parser.add_argument(
        "--prefill-rationale-marker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally prefill 'Rationale:' for compatibility with older artifacts; disabled for the paper prompt.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def active_document_prompt_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_DOCUMENT_PROMPT_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_DOCUMENT_PROMPT_VERSION
    return DOCUMENT_PROMPT_VERSION


def active_no_doc_prompt_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_PROMPT_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_PROMPT_VERSION
    return PROMPT_VERSION


def active_generation_policy_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_GENERATION_POLICY_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_GENERATION_POLICY_VERSION
    return GENERATION_POLICY_VERSION


def trace_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset_name:
        return str(args.dataset_name)
    try:
        return args.candidates_path.parent.parent.name
    except IndexError:
        return "dataset"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL: {path}:{line_no}") from exc


def window_candidate_document(
    row: dict[str, Any],
    window: dict[str, Any],
    *,
    fallback_selection_rank: int,
) -> dict[str, Any]:
    """Materialize one selected window as the existing generator's evidence object.

    Only ``text`` is visible to the prompt builder.  Parent-document and
    selection fields remain metadata in the saved trace, which prevents a
    role marker or corpus label from changing the generation condition.
    """

    candidate_id = str(window.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError(f"Window candidate without candidate_id: {row.get('pair_id')}")
    text = str(window.get("text") or "").strip()
    if not text:
        raise ValueError(f"Window candidate without text: {candidate_id}")
    selection_rank = int(window.get("selection_rank") or fallback_selection_rank)
    if selection_rank <= 0:
        raise ValueError(f"Invalid window selection_rank: {candidate_id}")
    parent_doc_rank = int(row.get("doc_rank") or 0)
    if parent_doc_rank <= 0:
        raise ValueError(f"Invalid parent doc_rank: {row.get('pair_id')}")
    # Three fixed roles are requested per parent document. A composite rank
    # preserves the original rerank order while making ranks unique within a
    # question even when a short document supplies fewer than three windows.
    composite_rank = (parent_doc_rank - 1) * 3 + selection_rank
    document = row.get("document") if isinstance(row.get("document"), dict) else {}
    return {
        "source": row.get("doc_source") or document.get("source"),
        "local_id": composite_rank,
        "db_id": candidate_id,
        "corpus_id": candidate_id,
        "chunk_id": candidate_id,
        "doc_id": row.get("doc_stable_id") or document.get("stable_id"),
        # Do not expose the parent title or [TARGET]/selection markers to the
        # LLM. This makes raw evidence text the only changed prompt content.
        "title": None,
        "text": text,
        "retrieval_score": None,
        "retrieval_rank": parent_doc_rank,
        "rerank_score": None,
        "rerank_rank": composite_rank,
        "filter_score": None,
        "filter_rank": None,
        "filter_prediction": None,
        "filter_prob_helpful": None,
        "stable_id": candidate_id,
        "evidence_unit": "sentence_context_window",
        "window_candidate_id": candidate_id,
        "window_selection_version": row.get("selection_version"),
        "window_selection_role": window.get("selection_role"),
        "window_selection_rank": selection_rank,
        "window_id": window.get("window_id"),
        "window_sha256": window.get("sha256"),
        "window_sentence_ids": window.get("sentence_ids"),
        "window_sentence_count": window.get("sentence_count"),
        "window_centre_sentence_id": window.get("centre_sentence_id"),
        "window_centre_sentence_index": window.get("centre_sentence_index"),
        "window_centre_positive_mass_ratio": window.get("centre_positive_mass_ratio"),
        "window_positive_mass_ratio": window.get("window_positive_mass_ratio"),
        "window_question_relevance_score": window.get("question_window_relevance_score"),
        "parent_document_pair_id": row.get("pair_id"),
        "parent_document_rank": parent_doc_rank,
        "parent_document_source": row.get("doc_source") or document.get("source"),
        "parent_document_stable_id": row.get("doc_stable_id") or document.get("stable_id"),
        # Audit-only. It is never included in the generation prompt or used to
        # select/parse the generated response.
        "parent_document_pseudo_label": row.get("document_pseudo_label"),
    }


def sentence_candidate_document(
    row: dict[str, Any],
    sentence: dict[str, Any],
    *,
    fallback_selection_rank: int,
) -> dict[str, Any]:
    """Materialize one selected single sentence as a prompt evidence object.

    As in the direct-window protocol, only raw evidence text is exposed to
    the LLM. Selection roles, attribution, the parent-document label, and all
    source metadata remain trace-only audit fields.
    """

    candidate_id = str(sentence.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError(f"Sentence candidate without candidate_id: {row.get('pair_id')}")
    text = str(sentence.get("text") or "").strip()
    if not text:
        raise ValueError(f"Sentence candidate without text: {candidate_id}")
    selection_rank = int(sentence.get("selection_rank") or fallback_selection_rank)
    if selection_rank <= 0:
        raise ValueError(f"Invalid sentence selection_rank: {candidate_id}")
    parent_doc_rank = int(row.get("doc_rank") or 0)
    if parent_doc_rank <= 0:
        raise ValueError(f"Invalid parent doc_rank: {row.get('pair_id')}")
    composite_rank = (parent_doc_rank - 1) * 3 + selection_rank
    document = row.get("document") if isinstance(row.get("document"), dict) else {}
    return {
        "source": row.get("doc_source") or document.get("source"),
        "local_id": composite_rank,
        "db_id": candidate_id,
        "corpus_id": candidate_id,
        "chunk_id": candidate_id,
        "doc_id": row.get("doc_stable_id") or document.get("stable_id"),
        "title": None,
        "text": text,
        "retrieval_score": None,
        "retrieval_rank": parent_doc_rank,
        "rerank_score": None,
        "rerank_rank": composite_rank,
        "filter_score": None,
        "filter_rank": None,
        "filter_prediction": None,
        "filter_prob_helpful": None,
        "stable_id": candidate_id,
        "evidence_unit": "single_sentence",
        "sentence_candidate_id": candidate_id,
        "sentence_selection_version": row.get("selection_version"),
        "sentence_selection_role": sentence.get("selection_role"),
        "sentence_selection_rank": selection_rank,
        "sentence_id": sentence.get("sentence_id"),
        "sentence_index": sentence.get("sentence_index"),
        "sentence_sha256": sentence.get("sha256"),
        "sentence_token_count": sentence.get("token_count"),
        "sentence_positive_attribution": sentence.get("positive_attribution"),
        "sentence_positive_mass_ratio": sentence.get("positive_mass_ratio"),
        "sentence_positive_mass_rank": sentence.get("positive_mass_rank"),
        "sentence_question_relevance_score": sentence.get("question_sentence_relevance_score"),
        "parent_document_pair_id": row.get("pair_id"),
        "parent_document_rank": parent_doc_rank,
        "parent_document_source": row.get("doc_source") or document.get("source"),
        "parent_document_stable_id": row.get("doc_stable_id") or document.get("stable_id"),
        "parent_document_pseudo_label": row.get("document_pseudo_label"),
    }


def selected_evidence_input(args: argparse.Namespace) -> bool:
    return args.candidate_input_format in {
        "window_annotation_candidates",
        "sentence_annotation_candidates",
    }


def selected_evidence_unit(args: argparse.Namespace) -> str:
    if args.candidate_input_format == "window_annotation_candidates":
        return "sentence_context_window"
    if args.candidate_input_format == "sentence_annotation_candidates":
        return "single_sentence"
    return "retrieved_document"


def normalize_candidate_input_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_input_format == "retrieval_documents":
        normalized = dict(row)
        normalized["candidate_documents"] = list(row.get("candidate_documents") or [])[
            : args.docs_per_question
        ]
        return normalized

    expected_version = str(
        args.expected_window_selection_version
        if args.candidate_input_format == "window_annotation_candidates"
        else args.expected_sentence_selection_version
        or ""
    )
    actual_version = str(row.get("selection_version") or "")
    if expected_version and actual_version != expected_version:
        raise ValueError(
            f"Unexpected evidence selection version for {row.get('pair_id')}: "
            f"expected={expected_version} actual={actual_version}"
        )
    if args.candidate_input_format == "window_annotation_candidates":
        evidence = row.get("window_candidates")
        unit_name = "windows"
    else:
        evidence = row.get("sentence_candidates")
        unit_name = "sentences"
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"Candidate row has no selected {unit_name}: {row.get('pair_id')}")
    if len(evidence) > 3:
        raise ValueError(f"Candidate row has more than three {unit_name}: {row.get('pair_id')}")
    if args.candidate_input_format == "window_annotation_candidates":
        documents = [
            window_candidate_document(row, item, fallback_selection_rank=index)
            for index, item in enumerate(evidence, start=1)
        ]
        candidate_ids = [str(document["window_candidate_id"]) for document in documents]
    else:
        documents = [
            sentence_candidate_document(row, item, fallback_selection_rank=index)
            for index, item in enumerate(evidence, start=1)
        ]
        candidate_ids = [str(document["sentence_candidate_id"]) for document in documents]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"Duplicate evidence candidate IDs: {row.get('pair_id')}")
    normalized = dict(row)
    normalized.update(
        {
            "split": row.get("source_split"),
            "answer": (row.get("answers") or [None])[0] if isinstance(row.get("answers"), list) else None,
            "candidate_documents": documents,
            "candidate_evidence_unit": selected_evidence_unit(args),
            "parent_document_pair_id": row.get("pair_id"),
        }
    )
    return normalized


def iter_candidate_chunks(path: Path, args: argparse.Namespace) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    selected = 0
    for row_number, row in enumerate(iter_jsonl(path)):
        if row_number < args.start:
            continue
        if args.end is not None and row_number >= args.end:
            break
        if args.limit_questions is not None and selected >= args.limit_questions:
            break
        chunk.append(normalize_candidate_input_row(row, args))
        selected += 1
        if len(chunk) >= args.question_batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def selected_question_count(candidates_path: Path, args: argparse.Namespace) -> int:
    """Count selected candidate rows, not all valid no-RAG cache rows.

    The strict quality-selection artifact may exclude additional no-RAG rows.
    The candidate JSONL is therefore the authoritative question set for this
    job and for the progress denominator.
    """
    return sum(len(chunk) for chunk in iter_candidate_chunks(candidates_path, args))


def selected_input_counts(candidates_path: Path, args: argparse.Namespace) -> tuple[int, int]:
    rows = 0
    pairs = 0
    for chunk in iter_candidate_chunks(candidates_path, args):
        rows += len(chunk)
        pairs += sum(len(row.get("candidate_documents") or []) for row in chunk)
    return rows, pairs


def load_forced_pair_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing force-pair target list: {path}")
    pair_ids: set[str] = set()
    for row in iter_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError(f"Target row without pair_id in {path}")
        pair_ids.add(pair_id)
    if not pair_ids:
        raise ValueError(f"No pair IDs found in {path}")
    return pair_ids


def load_quality_selection(path: Path | None, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Load the exact audited no-RAG selection used for retrieval, if supplied.

    The paper's free response prompt does not prescribe a literal terminal
    marker.  ``usable_rows.jsonl`` therefore records deterministic recovery of
    an explicitly expressed option without rewriting the model response.  It
    must be reused here; otherwise a later, narrower parser can silently drop
    questions that were already embedded and retrieved.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Missing --quality-selection-path: {path}")
    selected: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        answer = str(row.get("selected_no_rag_answer") or "").upper()
        if not sample_id or not answer:
            raise ValueError(f"Malformed quality-selection row in {path}: {row}")
        if sample_id in selected:
            raise ValueError(f"Duplicate quality-selection sample_id in {path}: {sample_id}")
        selected[sample_id] = row
    if not selected:
        raise ValueError(f"No rows in --quality-selection-path: {path}")
    return selected


def use_full_response_as_paper_exact_rationale_only(
    parsed: Any,
    nested_stats: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Use the complete visible free response as the comparable PPL span.

    The verbatim RAG2 prompt separates neither a rationale marker nor an
    answer marker.  For such a free response, a post-hoc rationale-only
    boundary is not reproducible for every valid answer style.  The entire
    independently generated visible response (reasoning plus expressed
    answer) is the stable PPL unit for both no-RAG and one-document outputs.
    """
    if not parsed.rationale:
        return parsed, nested_stats
    rationale = (nested_stats or {}).get("rationale") or {}
    if rationale.get("ppl") is None or not int(rationale.get("token_count") or 0):
        return parsed, nested_stats
    normalized_stats = copy.deepcopy(nested_stats)
    full_response_scope = dict(normalized_stats["rationale"])
    full_response_scope["source"] = "paper_exact_complete_visible_response_ppl"
    normalized_stats["rationale_only"] = full_response_scope
    if not parsed.rationale_only:
        parsed = replace(
            parsed,
            rationale_only=parsed.rationale,
            rationale_only_span=parsed.rationale_span,
        )
    return parsed, normalized_stats


def load_no_doc_cache(path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Load raw no-RAG rows or normalized ``search_ready`` rows.

    ``search_ready`` preserves the original no-RAG generation/PPL fields and
    adds the exact canonical query used by the candidate builder. Using it here
    makes the candidate-to-baseline alignment check unambiguous for the Llama
    reproduction pipeline.
    """
    cache: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    selection = load_quality_selection(args.quality_selection_path, args)
    expected_prompt = active_no_doc_prompt_version(args)
    expected_policy = active_generation_policy_version(args)
    for row in iter_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        parsed = row.get("parsed") or {}
        rationale_stats = ((row.get("generation_stats") or {}).get("rationale") or {})
        retrieval_query = row.get("retrieval_query") or {}
        query_text = str(retrieval_query.get("query_text") or parsed.get("rationale_query") or "").strip()
        selected = selection.get(sample_id)
        common_valid = (
            row.get("prompt_version") == expected_prompt
            and row.get("ppl_scope_version") == PPL_SCOPE_VERSION
            and row.get("generation_policy_version") == expected_policy
            and query_text
            and rationale_stats.get("ppl") is not None
            and int(rationale_stats.get("token_count") or 0) > 0
            and not row.get("truncated_by_max_tokens")
            and row.get("finish_reason") != "length"
        )
        if selected is not None:
            # Candidates were built from this exact external selection.  Its
            # recovered answer is authoritative while the raw response and
            # retrieval query remain unchanged.
            prediction = str(selected["selected_no_rag_answer"]).upper()
            valid = common_valid and prediction in normalized_options(row)
            rationale = parsed.get("rationale") or str(
                row.get("model_raw_generation") or row.get("no_rag_generation") or ""
            ).strip()
            rationale_only = rationale
            nested_stats = copy.deepcopy(row.get("generation_stats") or {})
            full_response_scope = dict(nested_stats.get("rationale") or {})
            full_response_scope["source"] = "paper_exact_complete_visible_response_ppl"
            nested_stats["rationale_only"] = full_response_scope
            parse_errors: list[str] = []
            answer_conclusion = parsed.get("answer_conclusion")
            answer_source = str(selected.get("answer_source") or "external_quality_selection")
        else:
            valid = (
                common_valid
                and parsed.get("final_answer")
                and not parsed.get("parse_errors")
                and ((row.get("generation_stats") or {}).get("rationale_only") or {}).get("ppl") is not None
                and int(((row.get("generation_stats") or {}).get("rationale_only") or {}).get("token_count") or 0) > 0
            )
            prediction = parsed.get("final_answer")
            rationale = parsed.get("rationale")
            rationale_only = parsed.get("rationale_only")
            nested_stats = row.get("generation_stats") or {}
            parse_errors = parsed.get("parse_errors") or []
            answer_conclusion = parsed.get("answer_conclusion")
            answer_source = "stored_parser"
        cache[sample_id] = {
            "valid": bool(valid),
            "question": row.get("question"),
            "options": normalized_options(row),
            "gold_answers": sorted(gold_answers(row)),
            "generation": row.get("no_rag_generation"),
            "raw_generation": row.get("model_raw_generation") or row.get("no_rag_generation"),
            "rationale": rationale,
            "rationale_only": rationale_only,
            "rationale_query": query_text,
            "rationale_query_source": retrieval_query.get("query_field") or "parsed.rationale_query",
            "answer_conclusion": answer_conclusion,
            "prediction": prediction,
            "correct": prediction in gold_answers(row),
            "parse_errors": parse_errors,
            "source_parse_errors": parsed.get("parse_errors") or [],
            "answer_source": answer_source,
            "generation_prompt_variant": row.get("generation_prompt_variant", "standard"),
            "nested_stats": nested_stats,
            "stats": flatten_generation_stats(nested_stats),
        }
    excluded = [sample_id for sample_id, row in cache.items() if not row["valid"]]
    for sample_id in excluded:
        del cache[sample_id]
    if excluded:
        logging.warning(
            "Excluded %s invalid no-RAG rows from single-document generation; first sample IDs=%s",
            len(excluded),
            excluded[:10],
        )
    if not cache:
        raise RuntimeError(f"No valid no-RAG baselines found: {path}")
    return cache


def raw_format_issues(
    raw_text: str,
    options: dict[str, Any] | None = None,
    *,
    prompt_profile: str,
) -> list[str]:
    # The verbatim paper prompt requires a final option but does not prescribe
    # a literal answer marker. Its terminal-choice parser is the only format
    # check; applying the legacy ``Therefore ...`` checks would manufacture
    # cosmetic failures without changing the model output.
    if prompt_profile == "paper_exact":
        return []
    visible_text = str(raw_text or "").strip()
    if not visible_text:
        return ["empty_raw_generation"]
    issues: list[str] = []
    if len(ANSWER_CONCLUSION_PATTERN.findall(visible_text)) != 1:
        issues.append("unexpected_answer_conclusion_count")
        return issues

    nonempty_lines = [line.strip() for line in visible_text.splitlines() if line.strip()]
    terminal = nonempty_lines[-1] if nonempty_lines else ""
    strict_match = re.fullmatch(r"Therefore, the answer is \(([A-Z])\) (.+)\.", terminal)
    if strict_match is None:
        issues.append("noncanonical_terminal_answer_format")
        return issues
    option_label, option_text = strict_match.groups()
    normalized = normalized_options({"options": options or {}})
    if option_label not in normalized or option_text != normalized[option_label]:
        issues.append("terminal_answer_option_text_mismatch")
    return issues


def hard_quality_issues(
    parsed: Any,
    nested_stats: dict[str, Any],
    finish_reason: str | None = None,
) -> list[str]:
    """Return only failures that make a trace unusable for pseudo-labeling.

    The terminal answer sentence is deliberately checked separately. Llama can
    express a valid option with a small wording variation, and treating that as
    a failed sample would bias the training pool toward a stylistic artifact.
    """
    issues: list[str] = []
    if parsed.parse_errors:
        issues.append("parse_errors")
    if not parsed.rationale:
        issues.append("missing_rationale")
    if not parsed.final_answer:
        issues.append("missing_final_answer")
    if rationale_ppl(nested_stats) is None:
        issues.append("missing_rationale_ppl")
    if span_ppl(nested_stats, "rationale_only") is None:
        issues.append("missing_rationale_only_ppl")
    if finish_reason == "length":
        issues.append("max_tokens_exhausted")
    return issues


def evaluate_output(
    output: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    prefilled_rationale_marker: bool,
    prompt_profile: str,
) -> dict[str, Any]:
    options = normalized_options(row)
    choice = output.outputs[0] if getattr(output, "outputs", None) else None
    generated_text = choice.text.strip() if choice is not None else ""
    raw_text = f"Rationale: {generated_text}" if prefilled_rationale_marker else generated_text
    parsed = parse_mcq_output_for_prompt_profile(raw_text, options, prompt_profile)
    nested_stats = (
        generation_stats(choice, tokenizer, raw_text, options, parsed_output=parsed)
        if choice is not None
        else {}
    )
    if prompt_profile == "paper_exact":
        parsed, nested_stats = use_full_response_as_paper_exact_rationale_only(parsed, nested_stats)
    finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
    return {
        "choice": choice,
        "raw_text": raw_text,
        "parsed": parsed,
        "nested_stats": nested_stats,
        "quality_issues": hard_quality_issues(parsed, nested_stats, finish_reason),
        "format_issues": raw_format_issues(raw_text, options, prompt_profile=prompt_profile),
    }


def load_done_pairs(path: Path, args: argparse.Namespace) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            latest[pair_id] = row
    done: set[str] = set()
    for pair_id, row in latest.items():
        if row.get("prompt_version") != active_document_prompt_version(args):
            continue
        if row.get("ppl_scope_version") != PPL_SCOPE_VERSION:
            continue
        if row.get("generation_policy_version") != active_generation_policy_version(args):
            continue
        parsed = row.get("with_doc_parse_errors") or []
        rationale = row.get("with_doc_rationale")
        prediction = row.get("with_doc_prediction")
        stats = row.get("with_doc_generation_stats") or {}
        pseudo_parsed = type(
            "StoredParse",
            (),
            {"parse_errors": parsed, "rationale": rationale, "final_answer": prediction},
        )()
        if not hard_quality_issues(
            pseudo_parsed,
            stats,
            str(row.get("with_doc_finish_reason") or ""),
        ):
            done.add(pair_id)
    return done


def document_pair_id(sample_id: str, doc: dict[str, Any], rank: int) -> str:
    stable_id = doc.get("stable_id") or doc.get("corpus_id") or doc.get("db_id")
    if not stable_id:
        stable_id = f"{doc.get('source')}:{doc.get('local_id')}"
    return f"{sample_id}::{rank}::{stable_id}"


def gold_answers(row: dict[str, Any]) -> set[str]:
    options = {str(key).upper() for key in (row.get("options") or {})}
    values = row.get("answers")
    if not isinstance(values, list):
        # Raw no-RAG generations use ``gold_answers``/``gold_answer`` while
        # retrieval candidates use ``answers``/``answer``.  Accept both
        # schemas so the cached no-RAG correctness flag is trustworthy.
        values = row.get("gold_answers")
    if not isinstance(values, list):
        values = [row.get("answer") or row.get("gold_answer")]
    return {str(value).upper() for value in values if str(value).upper() in options}


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
    if uses_gdn_attention and args.gdn_prefill_backend not in {None, "none"}:
        kwargs["additional_config"] = {"gdn_prefill_backend": args.gdn_prefill_backend}
    logging.info("Loading %s through vLLM: %s", model_type or "local model", args.llm_model_path)
    llm = LLM(**kwargs)
    initial_max_tokens = args.max_new_tokens
    retry_max_tokens = max(
        args.max_new_tokens, args.invalid_retry_max_new_tokens, args.quality_retry_max_new_tokens
    )
    length_retry_max_tokens = int(args.length_retry_max_new_tokens)
    if args.length_retry_attempts < 0:
        raise ValueError("--length-retry-attempts cannot be negative.")
    # A paper-exact run deliberately disables repair generation.  In that
    # case the unused retry cap must not constrain the primary free response.
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
    choice_sampling = SamplingParams(
        max_tokens=args.choice_selection_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=[*args.stop, "\n"],
        logprobs=1,
    )
    return tokenizer, llm, sampling, length_retry_sampling, retry_sampling, choice_sampling


def render_document_generation_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    doc: dict[str, Any],
    args: argparse.Namespace,
    *,
    format_retry: bool,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> str:
    if args.prompt_profile == "paper_exact":
        messages = build_paper_exact_documents_messages(
            row,
            [doc],
            max_doc_chars=args.max_doc_chars,
            format_retry=format_retry,
            selected_answer=selected_answer,
            compact_retry=compact_retry,
        )
    elif args.prompt_profile == "paper_answer_format":
        messages = build_paper_answer_format_document_messages(
            row,
            doc,
            max_doc_chars=args.max_doc_chars,
            format_retry=format_retry,
            selected_answer=selected_answer,
            compact_retry=compact_retry,
        )
    else:
        messages = build_document_messages(
            row,
            doc,
            max_doc_chars=args.max_doc_chars,
            format_retry=format_retry,
            selected_answer=selected_answer,
            compact_retry=compact_retry,
        )
    prompt = render_prompt(
        tokenizer,
        messages,
        args.use_chat_template,
    )
    return f"{prompt}Rationale: " if args.prefill_rationale_marker else prompt


def generate_batch(
    llm: Any,
    tokenizer: Any,
    sampling: Any,
    length_retry_sampling: Any,
    retry_sampling: Any,
    choice_sampling: Any,
    prompts: list[str],
    items: list[tuple[dict[str, Any], dict[str, Any], int, str]],
    args: argparse.Namespace,
) -> tuple[list[Any], list[int], list[dict[str, Any]], list[str]]:
    outputs = list(llm.generate(prompts, sampling, use_tqdm=False))
    if len(outputs) != len(items):
        raise RuntimeError(f"vLLM output count mismatch: expected={len(items)} got={len(outputs)}")
    attempts = [1] * len(outputs)
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
            "Single-doc length retry %s/%s for %s pair(s): max_tokens=%s",
            length_retry_round + 1,
            args.length_retry_attempts,
            len(length_indices),
            args.length_retry_max_new_tokens,
        )
        length_prompts = [
            render_document_generation_prompt(
                tokenizer,
                items[index][0],
                items[index][1],
                args,
                format_retry=False,
                compact_retry=True,
            )
            for index in length_indices
        ]
        length_outputs = list(llm.generate(length_prompts, length_retry_sampling, use_tqdm=False))
        if len(length_outputs) != len(length_indices):
            raise RuntimeError(
                "vLLM length-retry output count mismatch: "
                f"expected={len(length_indices)} got={len(length_outputs)}"
            )
        for index, length_output in zip(length_indices, length_outputs):
            outputs[index] = length_output
            attempts[index] += 1
            prompt_variants[index] = "decisive_compact_retry"
            length_retry_attempted.add(index)

    for retry_round in range(args.invalid_retry_attempts):
        invalid_indices = []
        retry_reason_counts: Counter[str] = Counter()
        for index, ((row, _, _, _), output) in enumerate(zip(items, outputs)):
            choice = output.outputs[0] if getattr(output, "outputs", None) else None
            if index in length_retry_attempted and getattr(choice, "finish_reason", None) == "length":
                retry_reason_counts.update(["max_tokens_exhausted_after_length_retry"])
                continue
            evaluation = evaluate_output(
                output,
                tokenizer,
                row,
                prefilled_rationale_marker=args.prefill_rationale_marker,
                prompt_profile=args.prompt_profile,
            )
            reasons = evaluation["quality_issues"] if args.retry_quality else [
                reason for reason in evaluation["quality_issues"] if reason in {
                    "parse_errors", "missing_rationale", "missing_final_answer"
                }
            ]
            if reasons:
                invalid_indices.append(index)
                retry_reason_counts.update(reasons)
        if not invalid_indices:
            break
        logging.info(
            "Single-doc quality retry %s/%s for %s pair(s): %s",
            retry_round + 1,
            args.invalid_retry_attempts,
            len(invalid_indices),
            dict(retry_reason_counts),
        )
        retry_prompts = [
            render_document_generation_prompt(
                tokenizer,
                items[index][0],
                items[index][1],
                args,
                format_retry=True,
            )
            for index in invalid_indices
        ]
        retry_outputs = list(llm.generate(retry_prompts, retry_sampling, use_tqdm=False))
        if len(retry_outputs) != len(invalid_indices):
            raise RuntimeError(
                f"vLLM retry output count mismatch: expected={len(invalid_indices)} got={len(retry_outputs)}"
            )
        for index, retry_output in zip(invalid_indices, retry_outputs):
            outputs[index] = retry_output
            attempts[index] += 1
            prompt_variants[index] = "format_retry"

    evaluations = [
        evaluate_output(
            output,
            tokenizer,
            row,
            prefilled_rationale_marker=args.prefill_rationale_marker,
            prompt_profile=args.prompt_profile,
        )
        for (row, _, _, _), output in zip(items, outputs)
    ]
    if args.choice_anchored_retry:
        unresolved_indices = [
            index
            for index, evaluation in enumerate(evaluations)
            if evaluation["quality_issues"]
        ]
        if unresolved_indices:
            logging.info("Single-doc choice-anchored repair for %s unresolved pair(s)", len(unresolved_indices))
            selection_prompts = [
                render_prompt(
                    tokenizer,
                    build_document_choice_selection_messages(
                        items[index][0], items[index][1], max_doc_chars=args.max_doc_chars
                    ),
                    args.use_chat_template,
                )
                for index in unresolved_indices
            ]
            selection_outputs = list(llm.generate(selection_prompts, choice_sampling, use_tqdm=False))
            if len(selection_outputs) != len(unresolved_indices):
                raise RuntimeError(
                    f"vLLM choice-selection count mismatch: expected={len(unresolved_indices)} got={len(selection_outputs)}"
                )
            anchored: list[tuple[int, str]] = []
            for index, selection_output in zip(unresolved_indices, selection_outputs):
                choice = selection_output.outputs[0] if getattr(selection_output, "outputs", None) else None
                choice_text = choice.text.strip().upper() if choice is not None else ""
                valid_options = set(normalized_options(items[index][0]))
                selected_answer = next(
                    (match.group(1) for match in re.finditer(r"\b([A-Z])\b", choice_text) if match.group(1) in valid_options),
                    None,
                )
                if selected_answer is not None:
                    anchored.append((index, selected_answer))
            if anchored:
                anchored_prompts = [
                    render_document_generation_prompt(
                        tokenizer,
                        items[index][0],
                        items[index][1],
                        args,
                        format_retry=True,
                        selected_answer=selected_answer,
                        compact_retry=True,
                    )
                    for index, selected_answer in anchored
                ]
                anchored_outputs = list(llm.generate(anchored_prompts, retry_sampling, use_tqdm=False))
                if len(anchored_outputs) != len(anchored):
                    raise RuntimeError(
                        f"vLLM choice-anchored generation count mismatch: expected={len(anchored)} got={len(anchored_outputs)}"
                    )
                for (index, _), anchored_output in zip(anchored, anchored_outputs):
                    outputs[index] = anchored_output
                    attempts[index] += 2
                    prompt_variants[index] = "choice_anchored_decisive_repair"
                evaluations = [
                    evaluate_output(
                        output,
                        tokenizer,
                        row,
                        prefilled_rationale_marker=args.prefill_rationale_marker,
                        prompt_profile=args.prompt_profile,
                    )
                    for (row, _, _, _), output in zip(items, outputs)
                ]
    return outputs, attempts, evaluations, prompt_variants


def span_ppl(nested_stats: dict[str, Any], scope: str) -> float | None:
    value = ((nested_stats.get(scope) or {}).get("ppl"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rationale_ppl(nested_stats: dict[str, Any]) -> float | None:
    return span_ppl(nested_stats, "rationale")


def validate_trace_job_args(args: argparse.Namespace) -> None:
    if args.docs_per_question <= 0 or args.generation_batch_size <= 0 or args.question_batch_size <= 0:
        raise ValueError("Document and batch sizes must be positive.")
    if (
        args.quality_retry_max_new_tokens <= 0
        or args.choice_selection_max_new_tokens <= 0
        or args.length_retry_max_new_tokens <= 0
    ):
        raise ValueError("Quality-retry and choice-selection token limits must be positive.")
    if not args.candidates_path.exists():
        raise FileNotFoundError(f"Missing candidate dataset: {args.candidates_path}")
    if not args.no_rag_path.exists():
        raise FileNotFoundError(f"Missing no-RAG baseline: {args.no_rag_path}")
    if args.quality_selection_path is not None and not args.quality_selection_path.exists():
        raise FileNotFoundError(f"Missing --quality-selection-path: {args.quality_selection_path}")
    if args.quality_selection_path is not None and args.prompt_profile != "paper_exact":
        raise ValueError("--quality-selection-path is currently supported only with --prompt-profile paper_exact.")
    if not args.llm_model_path.exists():
        raise FileNotFoundError(f"Missing local LLM model: {args.llm_model_path}")
    if args.prompt_profile == "paper_exact" and any(
        [
            args.length_retry_attempts,
            args.invalid_retry_attempts,
            args.retry_quality,
            args.choice_anchored_retry,
            args.prefill_rationale_marker,
        ]
    ):
        raise ValueError(
            "paper_exact preserves the published prompt without repair instructions or response prefills. "
            "Use zero retries, --no-retry-quality, --no-choice-anchored-retry, and "
            "--no-prefill-rationale-marker."
        )


def align_candidate_row_with_baseline(
    row: dict[str, Any],
    baseline: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Validate no-RAG identity and attach its retrieval-query metadata.

    Historical retrieval candidates store the rationale query explicitly.
    Window-selection rows intentionally omit that large repeated field, so
    their identity is checked with sample ID, question, and options, then the
    exact cached no-RAG query is copied into the in-memory row for trace
    metadata. It is not part of the evidence-conditioned prompt.
    """

    sample_id = str(row.get("sample_id") or "")
    baseline_query = str(baseline.get("rationale_query") or "").strip()
    if args.candidate_input_format == "retrieval_documents" and args.candidate_query_alignment == "strict":
        candidate_query = str(row.get("query_text") or "").strip()
        if candidate_query != baseline_query:
            raise ValueError(
                f"Candidate query is not the normalized no-RAG rationale query for {sample_id}. "
                "Rebuild candidates from the matching search-ready cache."
            )
        return

    if clean_text(row.get("question")) != clean_text(baseline.get("question")):
        raise ValueError(f"Evidence candidate question does not match no-RAG baseline: {sample_id}")
    if normalized_options(row) != dict(baseline.get("options") or {}):
        raise ValueError(f"Evidence candidate options do not match no-RAG baseline: {sample_id}")
    if args.candidate_input_format != "retrieval_documents":
        row["query_text"] = baseline_query
        row["retrieval_query_type"] = "reused_no_rag_rationale_query"
        row["rerank_query_type"] = "parent_document_selection_metadata"
    else:
        row["labeling_no_rag_query_text"] = baseline_query
        row["retrieval_query_type"] = row.get("retrieval_query_type") or "frozen_external_candidate_query"


def validate_trace_inputs(args: argparse.Namespace) -> dict[str, int]:
    """Validate the exact no-RAG cache/candidate join without loading the LLM."""
    no_doc_cache = load_no_doc_cache(args.no_rag_path, args)
    questions = 0
    pairs = 0
    for question_rows in iter_candidate_chunks(args.candidates_path, args):
        for row in question_rows:
            sample_id = str(row.get("sample_id") or "")
            baseline = no_doc_cache.get(sample_id)
            if baseline is None:
                raise KeyError(f"Candidate sample is missing from no-RAG baseline: {sample_id}")
            align_candidate_row_with_baseline(row, baseline, args)
            docs = list(row.get("candidate_documents") or [])
            if args.candidate_input_format == "retrieval_documents" and len(docs) != args.docs_per_question:
                raise ValueError(
                    f"Expected {args.docs_per_question} reranked docs for {sample_id}, found {len(docs)}"
                )
            if selected_evidence_input(args) and not 1 <= len(docs) <= 3:
                raise ValueError(
                    f"Expected 1-3 selected evidence units for {row.get('pair_id')}, found {len(docs)}"
                )
            questions += 1
            pairs += len(docs)
    if not questions:
        raise RuntimeError("No candidate questions selected.")
    return {"questions": questions, "pairs": pairs, "valid_no_rag_rows": len(no_doc_cache)}


def run_trace_job(
    args: argparse.Namespace,
    tokenizer: Any,
    llm: Any,
    sampling: Any,
    length_retry_sampling: Any,
    retry_sampling: Any,
    choice_sampling: Any,
) -> dict[str, Any]:
    validate_trace_job_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_file
    forced_pair_ids = load_forced_pair_ids(args.force_pairs_path)
    exception_output_path = args.exception_output_path or (
        args.output_dir / "generation_quality_exceptions.jsonl"
    )
    no_doc_cache = load_no_doc_cache(args.no_rag_path, args)
    resume_forced_pairs = bool(getattr(args, "resume_forced_pairs", False))
    should_load_done = bool(args.resume and (forced_pair_ids is None or resume_forced_pairs))
    done_pairs = load_done_pairs(output_path, args) if should_load_done else set()
    total_questions, selected_pairs = selected_input_counts(args.candidates_path, args)
    total_pairs = len(forced_pair_ids) if forced_pair_ids is not None else selected_pairs
    logging.info(
        "Single-doc trace input ready: no_doc=%s selected_questions=%s expected_pairs=%s existing_pairs=%s forced_pairs=%s",
        len(no_doc_cache),
        total_questions,
        total_pairs,
        len(done_pairs),
        0 if forced_pair_ids is None else len(forced_pair_ids),
    )

    write_json(
        args.output_dir / "pseudo_label_trace_manifest.json",
        {
            "type": "rag2_single_document_trace_dataset",
            "created_or_updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "candidates_path": str(args.candidates_path),
            "no_rag_path": str(args.no_rag_path),
            "quality_selection_path": (
                None if args.quality_selection_path is None else str(args.quality_selection_path)
            ),
            "dataset_name": trace_dataset_name(args),
            "output_path": str(output_path),
            "llm_model_path": str(args.llm_model_path),
            "prompt_profile": args.prompt_profile,
            "prompt_version": active_document_prompt_version(args),
            "no_doc_prompt_version": active_no_doc_prompt_version(args),
            "ppl_scope_version": PPL_SCOPE_VERSION,
            "generation_policy_version": active_generation_policy_version(args),
            "candidate_input_format": args.candidate_input_format,
            "evidence_unit": selected_evidence_unit(args),
            "expected_window_selection_version": (
                args.expected_window_selection_version
                if args.candidate_input_format == "window_annotation_candidates"
                else None
            ),
            "expected_sentence_selection_version": (
                args.expected_sentence_selection_version
                if args.candidate_input_format == "sentence_annotation_candidates"
                else None
            ),
            "docs_per_question": args.docs_per_question,
            "candidate_query_alignment": args.candidate_query_alignment,
            "selected_input_rows": total_questions,
            "expected_pairs": total_pairs,
            "max_doc_chars": args.max_doc_chars,
            "ppl_source": (
                "vLLM cumulative logprobs over the complete independently generated visible response "
                "(reasoning plus expressed answer); the verbatim paper prompt does not define a stable boundary "
                "between those two pieces."
                if args.prompt_profile == "paper_exact"
                else "vLLM cumulative logprobs over independently generated no-doc and with-doc rationales"
            ),
            "generated_output_delta_ppl": (
                "no_doc_rationale_with_answer_ppl - with_doc_rationale_with_answer_ppl; positive means the document "
                "lowered independently generated output PPL. The comparison is stored for the RAG2-style pseudo-label "
                "protocol and does not teacher-force either output."
            ),
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "length_retry_attempts": args.length_retry_attempts,
                "length_retry_max_new_tokens": args.length_retry_max_new_tokens,
                "invalid_retry_attempts": args.invalid_retry_attempts,
                "invalid_retry_max_new_tokens": args.invalid_retry_max_new_tokens,
                "retry_quality": args.retry_quality,
                "quality_retry_max_new_tokens": args.quality_retry_max_new_tokens,
                "choice_anchored_retry": args.choice_anchored_retry,
                "choice_selection_max_new_tokens": args.choice_selection_max_new_tokens,
            },
            "prompt_contract": (
                "The RAG2 section 3.3 instruction is retained verbatim. The original question and options are followed "
                "only by a 'Documents:' header and the raw selected evidence text. Parent document "
                "labels, source/title/rank metadata, attribution values, and selection-role markers are saved only as "
                "trace metadata and are not shown to the LLM. No evidence usage instruction, output-format instruction, "
                "prefix, or retry instruction is supplied."
                if args.prompt_profile == "paper_exact"
                and selected_evidence_input(args)
                else "The RAG2 section 3.3 instruction is retained verbatim. The original question and options are followed "
                "only by a 'Documents:' header and the raw retrieved chunk text. No source/title/rank metadata, evidence "
                "usage instruction, output-format instruction, prefix, or retry instruction is supplied."
                if args.prompt_profile == "paper_exact"
                else (
                    "The no-RAG rationale-generation instruction and exact terminal-answer contract are retained. One "
                    "document is appended after the original question without any added instruction to use, ignore, cite, "
                    "summarize, or assess it. This preserves the no-RAG answer format while allowing the document's actual "
                    "effect on the independently generated rationale and answer to determine its pseudo-label."
                )
            ),
            "quality_policy": {
                "max_rationale_words": None,
                "hard_exclusion_checks": [
                    "parse_errors",
                    "missing_rationale",
                    "missing_final_answer",
                    "missing_rationale_ppl",
                    "missing_rationale_only_ppl",
                    "max_tokens_exhausted",
                ],
                "format_checks_recorded_only": [
                    "unexpected_answer_conclusion_count",
                    "noncanonical_terminal_answer_format",
                    "terminal_answer_option_text_mismatch",
                ],
            },
            "force_pairs_path": None if args.force_pairs_path is None else str(args.force_pairs_path),
            "resume_forced_pairs": resume_forced_pairs,
            "exception_output_path": str(exception_output_path),
            "prefill_rationale_marker": args.prefill_rationale_marker,
        },
    )

    progress = StageProgress(
        total=total_pairs,
        desc=f"SingleDocGeneration:{trace_dataset_name(args)}",
        enabled=True,
    )
    unresolved_pairs = 0
    regenerated_pair_ids: set[str] = set()
    exception_rows: list[dict[str, Any]] = []
    try:
        with output_path.open("a", encoding="utf-8", buffering=16 * 1024 * 1024) as output_handle:
            for question_rows in iter_candidate_chunks(args.candidates_path, args):
                prompts: list[str] = []
                items: list[tuple[dict[str, Any], dict[str, Any], int, str]] = []
                for row in question_rows:
                    sample_id = str(row.get("sample_id") or "")
                    if sample_id not in no_doc_cache:
                        raise KeyError(f"Candidate sample is missing from no-RAG baseline: {sample_id}")
                    align_candidate_row_with_baseline(row, no_doc_cache[sample_id], args)
                    docs = list(row.get("candidate_documents") or [])
                    if (
                        forced_pair_ids is None
                        and args.candidate_input_format == "retrieval_documents"
                        and len(docs) != args.docs_per_question
                    ):
                        raise ValueError(
                            f"Expected {args.docs_per_question} reranked docs for {sample_id}, found {len(docs)}"
                        )
                    if (
                        forced_pair_ids is None
                        and selected_evidence_input(args)
                        and not 1 <= len(docs) <= 3
                    ):
                        raise ValueError(
                            f"Expected 1-3 selected evidence units for {row.get('parent_document_pair_id')}, "
                            f"found {len(docs)}"
                        )
                    if forced_pair_ids is not None and not docs:
                        continue
                    for fallback_rank, doc in enumerate(docs, start=1):
                        doc_rank = int(doc.get("rerank_rank") or fallback_rank)
                        pair_id = document_pair_id(sample_id, doc, doc_rank)
                        is_forced = forced_pair_ids is not None and pair_id in forced_pair_ids
                        if forced_pair_ids is not None and not is_forced:
                            continue
                        if pair_id in done_pairs and (not is_forced or resume_forced_pairs):
                            progress.update(1)
                            continue
                        prompts.append(
                            render_document_generation_prompt(
                                tokenizer,
                                row,
                                doc,
                                args,
                                format_retry=False,
                            )
                        )
                        items.append((row, doc, doc_rank, pair_id))

                for batch_start in range(0, len(prompts), args.generation_batch_size):
                    batch_prompts = prompts[batch_start : batch_start + args.generation_batch_size]
                    batch_items = items[batch_start : batch_start + args.generation_batch_size]
                    outputs, attempts, evaluations, prompt_variants = generate_batch(
                        llm,
                        tokenizer,
                        sampling,
                        length_retry_sampling,
                        retry_sampling,
                        choice_sampling,
                        batch_prompts,
                        batch_items,
                        args,
                    )
                    for (row, doc, doc_rank, pair_id), output, generation_attempts, evaluation, prompt_variant in zip(
                        batch_items,
                        outputs,
                        attempts,
                        evaluations,
                        prompt_variants,
                    ):
                        sample_id = str(row["sample_id"])
                        options = row.get("options") or {}
                        choice = evaluation["choice"]
                        raw_text = evaluation["raw_text"]
                        parsed = evaluation["parsed"]
                        normalized_generation = (
                            raw_text
                            if args.prompt_profile in {"paper_exact", "paper_answer_format"}
                            else (
                                f"Rationale: {parsed.rationale_query}"
                                if parsed.rationale_query and parsed.final_answer
                                else raw_text
                            )
                        )
                        nested_stats = evaluation["nested_stats"]
                        flat_stats = flatten_generation_stats(nested_stats)
                        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
                        stop_reason = getattr(choice, "stop_reason", None) if choice is not None else None
                        no_doc = no_doc_cache[sample_id]
                        no_ppl = rationale_ppl(no_doc["nested_stats"])
                        with_ppl = rationale_ppl(nested_stats)
                        delta_ppl = no_ppl - with_ppl if no_ppl is not None and with_ppl is not None else None
                        no_rationale_only_ppl = span_ppl(no_doc["nested_stats"], "rationale_only")
                        with_rationale_only_ppl = span_ppl(nested_stats, "rationale_only")
                        delta_ppl_rationale_only = (
                            no_rationale_only_ppl - with_rationale_only_ppl
                            if no_rationale_only_ppl is not None and with_rationale_only_ppl is not None
                            else None
                        )
                        prediction = parsed.final_answer
                        correct = prediction in gold_answers(row) if prediction else False
                        trace = {
                            "schema_version": (
                                6
                                if args.candidate_input_format == "sentence_annotation_candidates"
                                else 5
                                if args.candidate_input_format == "window_annotation_candidates"
                                else 4
                            ),
                            "candidate_input_format": args.candidate_input_format,
                            "evidence_unit": doc.get("evidence_unit") or "retrieved_document",
                            "prompt_profile": args.prompt_profile,
                            "prompt_version": active_document_prompt_version(args),
                            "no_doc_prompt_version": active_no_doc_prompt_version(args),
                            "ppl_scope_version": PPL_SCOPE_VERSION,
                            "generation_policy_version": active_generation_policy_version(args),
                            "pair_id": pair_id,
                            "sample_id": sample_id,
                            "row_idx": row.get("row_idx"),
                            "dataset": row.get("dataset"),
                            "split": row.get("split"),
                            "question": row.get("question"),
                            "options": options,
                            "answers": row.get("answers"),
                            "answer": row.get("answer"),
                            "retrieval_query_text": row.get("query_text"),
                            "retrieval_query_type": row.get("retrieval_query_type"),
                            "rerank_query_type": row.get("rerank_query_type"),
                            "doc_rank": doc_rank,
                            "doc": doc,
                            "window_candidate_id": doc.get("window_candidate_id"),
                            "window_selection_version": doc.get("window_selection_version"),
                            "window_selection_role": doc.get("window_selection_role"),
                            "window_selection_rank": doc.get("window_selection_rank"),
                            "sentence_candidate_id": doc.get("sentence_candidate_id"),
                            "sentence_selection_version": doc.get("sentence_selection_version"),
                            "sentence_selection_role": doc.get("sentence_selection_role"),
                            "sentence_selection_rank": doc.get("sentence_selection_rank"),
                            "parent_document_pair_id": doc.get("parent_document_pair_id"),
                            "parent_document_rank": doc.get("parent_document_rank"),
                            "parent_document_source": doc.get("parent_document_source"),
                            "parent_document_stable_id": doc.get("parent_document_stable_id"),
                            "parent_document_pseudo_label": doc.get("parent_document_pseudo_label"),
                            "no_doc_generation": no_doc["generation"],
                            "no_doc_raw_generation": no_doc["raw_generation"],
                            "no_doc_rationale": no_doc["rationale"],
                            "no_doc_rationale_only": no_doc["rationale_only"],
                            "no_doc_rationale_query": no_doc["rationale_query"],
                            "no_doc_answer_conclusion": no_doc["answer_conclusion"],
                            "no_doc_prediction": no_doc["prediction"],
                            "no_doc_correct": no_doc["correct"],
                            "no_doc_parse_errors": no_doc["parse_errors"],
                            "no_doc_source_parse_errors": no_doc["source_parse_errors"],
                            "no_doc_answer_source": no_doc["answer_source"],
                            "no_doc_generation_prompt_variant": no_doc["generation_prompt_variant"],
                            "no_doc_stats": no_doc["stats"],
                            "no_doc_generation_stats": no_doc["nested_stats"],
                            "with_doc_generation": normalized_generation,
                            "with_doc_raw_generation": raw_text,
                            "with_doc_finish_reason": finish_reason,
                            "with_doc_stop_reason": stop_reason,
                            "with_doc_truncated_by_max_tokens": finish_reason == "length",
                            "with_doc_rationale": parsed.rationale,
                            "with_doc_rationale_only": parsed.rationale_only,
                            "with_doc_rationale_query": parsed.rationale_query,
                            "with_doc_answer_conclusion": parsed.answer_conclusion,
                            "with_doc_prediction": prediction,
                            "with_doc_correct": bool(correct),
                            "with_doc_parse_errors": parsed.parse_errors,
                            "with_doc_quality_issues": evaluation["quality_issues"],
                            "with_doc_format_issues": evaluation["format_issues"],
                            "with_doc_generation_attempts": generation_attempts,
                            "with_doc_generation_prompt_variant": prompt_variant,
                            "with_doc_generation_prefilled_rationale_marker": args.prefill_rationale_marker,
                            "with_doc_generation_was_normalized": normalized_generation != parsed.visible_text,
                            "with_doc_stats": flat_stats,
                            "with_doc_generation_stats": nested_stats,
                            "ppl_comparison_version": (
                                "rag2_paper_exact_independent_complete_visible_response_ppl_v1"
                                if args.prompt_profile == "paper_exact"
                                else "rag2_independent_generated_outputs_v2"
                            ),
                            "generated_output_delta_ppl": delta_ppl,
                            "generated_output_delta_ppl_rationale_with_answer": delta_ppl,
                            "generated_output_delta_ppl_rationale_only": delta_ppl_rationale_only,
                            "generated_output_delta_ppl_source": (
                                (
                                    "independent_generated_complete_visible_response_ppl"
                                    if args.prompt_profile == "paper_exact"
                                    else "independent_generated_rationale_with_answer_ppl"
                                )
                                if delta_ppl is not None
                                else None
                            ),
                            "delta_ppl": delta_ppl,
                            "delta_ppl_rationale_with_answer": delta_ppl,
                            "delta_ppl_rationale_only": delta_ppl_rationale_only,
                            "delta_ppl_source": "independent_generated_outputs",
                        }
                        unresolved_pairs += int(bool(evaluation["quality_issues"]))
                        regenerated_pair_ids.add(pair_id)
                        if evaluation["quality_issues"]:
                            exception_rows.append(
                                {
                                    "schema_version": 1,
                                    "pair_id": pair_id,
                                    "sample_id": sample_id,
                                    "row_idx": row.get("row_idx"),
                                    "dataset": row.get("dataset"),
                                    "split": row.get("split"),
                                    "doc_rank": doc_rank,
                                    "document": {
                                        "source": (doc or {}).get("source"),
                                        "stable_id": (doc or {}).get("stable_id"),
                                    },
                                    "quality_issues": evaluation["quality_issues"],
                                    "format_issues": evaluation["format_issues"],
                                    "parse_errors": parsed.parse_errors,
                                    "generation_attempts": generation_attempts,
                                    "raw_generation": raw_text,
                                }
                            )
                        output_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        done_pairs.add(pair_id)
                        progress.update(1)
                    output_handle.flush()
    finally:
        progress.close()

    if forced_pair_ids is not None:
        missing_forced = forced_pair_ids - regenerated_pair_ids - done_pairs
        if missing_forced:
            raise RuntimeError(
                f"Could not find {len(missing_forced)} forced pair(s) in candidates. "
                f"First IDs: {sorted(missing_forced)[:5]}"
            )
    exception_output_path.parent.mkdir(parents=True, exist_ok=True)
    with exception_output_path.open("w", encoding="utf-8") as handle:
        for row in exception_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info(
        "Single-document pseudo-label traces complete: %s (regenerated=%s unresolved=%s exceptions=%s)",
        output_path,
        len(regenerated_pair_ids),
        unresolved_pairs,
        exception_output_path,
    )
    return {
        "output_path": str(output_path),
        "manifest_path": str(args.output_dir / "pseudo_label_trace_manifest.json"),
        "exception_output_path": str(exception_output_path),
        "selected_questions": total_questions,
        "expected_pairs": total_pairs,
        "generated_pairs": len(regenerated_pair_ids),
        "unresolved_pairs": unresolved_pairs,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    validate_trace_job_args(args)
    if args.dry_run:
        result = validate_trace_inputs(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.update(
            {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "prompt_profile": args.prompt_profile,
                "document_prompt_version": active_document_prompt_version(args),
                "no_doc_prompt_version": active_no_doc_prompt_version(args),
                "generation_policy_version": active_generation_policy_version(args),
                "candidates_path": str(args.candidates_path),
                "candidate_input_format": args.candidate_input_format,
                "evidence_unit": selected_evidence_unit(args),
                "no_rag_path": str(args.no_rag_path),
                "dataset_name": trace_dataset_name(args),
            }
        )
        write_json(args.output_dir / "preflight.json", result)
        logging.info("Preflight complete: %s", result)
        return
    tokenizer, llm, sampling, length_retry_sampling, retry_sampling, choice_sampling = init_llm(args)
    try:
        run_trace_job(
            args,
            tokenizer,
            llm,
            sampling,
            length_retry_sampling,
            retry_sampling,
            choice_sampling,
        )
    finally:
        del llm


if __name__ == "__main__":
    main()

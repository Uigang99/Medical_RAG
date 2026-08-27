from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from build_rag2_filter_candidates import (
    build_source_sequential_hits,
    expected_candidate_pool_size,
    materialize_initial_docs,
)
from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path
from medrag.core import BenchmarkSample, CaseResult, GenerationOutput, PromptRequest, RetrievedDocument
from medrag.environment import collect_environment, write_environment_files
from medrag.evaluation import evaluate_prediction
from medrag.filtering.rag2_filter import DatasetRoutedRag2Filter, Rag2FlanT5Filter
from medrag.filtering.rag2_hierarchical import HierarchicalRag2DocumentFilter
from medrag.filtering.rag2_preanswer_text_hidden import (
    FINAL_ANSWER_PREFILL,
    PREANSWER_PROMPT_VERSION,
    PreAnswerLayerExtractor,
    TextHiddenRag2Filter,
    build_preanswer_user_prompt,
    preanswer_evidence,
)
from medrag.filtering.rag2_windowing import sentence_context_windows, windowing_contract
from medrag.generation.transformers_generator import VLLMChatGenerator
from medrag.io_utils import write_json, write_jsonl
from medrag.progress import StageProgress
from medrag.rag2_anchored_trace import (
    END_REASONING_MARKER as ANCHORED_END_REASONING_MARKER,
    GENERATION_POLICY_VERSION as ANCHORED_GENERATION_POLICY_VERSION,
    PROMPT_VERSION as ANCHORED_PROMPT_VERSION,
    RATIONALE_HEADER as ANCHORED_RATIONALE_HEADER,
    build_anchored_user_prompt,
    canonical_response as anchored_canonical_response,
    normalize_rationale as normalize_anchored_rationale,
    semantic_retrieval_queries as anchored_retrieval_queries,
)
from medrag.rag2_mcq import (
    DOCUMENT_PROMPT_VERSION,
    PAPER_ANSWER_FORMAT_DOCUMENT_PROMPT_VERSION,
    PAPER_ANSWER_FORMAT_PROMPT_VERSION,
    PAPER_EXACT_DOCUMENT_PROMPT_VERSION,
    PAPER_EXACT_PROMPT_VERSION,
    PAPER_EXACT_TERMINAL_DOCUMENT_PROMPT_VERSION,
    PAPER_EXACT_TERMINAL_PROMPT_VERSION,
    PROMPT_VERSION,
    append_paper_exact_terminal_answer,
    build_choice_selection_messages,
    build_documents_choice_selection_messages,
    build_documents_messages,
    build_no_rag_messages,
    build_paper_answer_format_documents_messages,
    build_paper_answer_format_no_rag_messages,
    build_paper_exact_documents_messages,
    build_paper_exact_no_rag_messages,
    build_paper_exact_terminal_documents_messages,
    build_paper_exact_terminal_no_rag_messages,
    format_question,
    normalized_options,
    parse_mcq_output,
    parse_mcq_output_for_prompt_profile,
    parse_paper_exact_mcq_output,
    paper_exact_terminal_regex,
)
from medrag.report import write_markdown_report, write_pretty_summary_table, write_text_report
from medrag.reranking.medcpt_cross_encoder import MedCPTCrossEncoderReranker
from medrag.retrieval.faiss_retriever import FaissMedCPTRetriever
from medrag.runner import results_to_jsonable


MCQ_DATASETS = [
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]
PAPER_SOURCES = ["pubmed", "pmc", "cpg", "textbooks"]
MMLU_DATASETS = [name for name in MCQ_DATASETS if name.startswith("mmlu_")]

DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "datasets" / "benchmark"
DEFAULT_VECTOR_DB_ROOT = PROJECT_ROOT / "databases" / "vector_db" / "medcpt_article_encoder"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "databases" / "run_cache" / "rag2_mcq_eval"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rag2_mcq_eval"
DEFAULT_LLM = WORKSPACE_ROOT / "models" / "Qwen3.5-9B"
DEFAULT_QUERY_ENCODER = WORKSPACE_ROOT / "models" / "MedCPT-Query-Encoder"
DEFAULT_CROSS_ENCODER = WORKSPACE_ROOT / "models" / "MedCPT-Cross-Encoder"
DEFAULT_MEDMCQA_FILTER = (
    WORKSPACE_ROOT
    / "models"
    / "RAG2-Filter-FlanT5-large"
    / "medmcqa"
    / "paper4_qwen35_9b_top10_epoch5_h200_b64"
    / "20260716_092458"
    / "final_model"
)
DEFAULT_MEDQA_FILTER = (
    WORKSPACE_ROOT
    / "models"
    / "RAG2-Filter-FlanT5-large"
    / "medqa"
    / "paper4_qwen35_9b_top10_epoch5_h200_b64"
    / "20260715_142343"
    / "final_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG2-style MCQ cases with a shared no-RAG rationale query, source-balanced retrieval, "
            "MedCPT reranking, and dataset-routed Flan-T5 filtering."
        )
    )
    parser.add_argument("--case", choices=["no_rag", "rerank_rag", "filter_rag", "oracle_rag"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=MCQ_DATASETS, default=MCQ_DATASETS)
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--vector-db-root", type=Path, default=DEFAULT_VECTOR_DB_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--candidate-cache-source-path",
        type=Path,
        default=None,
        help=(
            "Optional completed candidates.jsonl from a superset run. Exact sample keys, dense queries, "
            "retrieval/rerank settings, and Top-k documents are validated before a subset cache is materialized. "
            "This permits a one-dataset ablation to reuse precisely the candidates from an earlier multi-dataset run."
        ),
    )
    parser.add_argument(
        "--rationale-artifact-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing no_rag/<dataset>/<split>/no_rag_generations.jsonl. "
            "Use this to reuse a separately generated no-RAG artifact without copying it into --cache-root."
        ),
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--sources", nargs="+", default=PAPER_SOURCES)

    parser.add_argument("--llm-model-path", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--query-encoder-path", type=Path, default=DEFAULT_QUERY_ENCODER)
    parser.add_argument("--cross-encoder-path", type=Path, default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--medmcqa-filter-model-path", type=Path, default=DEFAULT_MEDMCQA_FILTER)
    parser.add_argument("--medqa-filter-model-path", type=Path, default=DEFAULT_MEDQA_FILTER)
    parser.add_argument(
        "--oracle-labels-path",
        type=Path,
        default=None,
        help="Gold document-label JSONL used only by --case oracle_rag.",
    )
    parser.add_argument(
        "--oracle-policy",
        choices=[
            "rag2",
            "hidden_tau_0",
            "hidden_tau_0p4",
            "hidden_three_class",
            "margin_utility",
        ],
        default=None,
        help="Label/score field to materialize as Helpful for --case oracle_rag.",
    )
    parser.add_argument(
        "--hidden-filter-backbone-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/Flan-T5-large",
        help="Original Flan-T5 backbone used to reconstruct a text+hidden filter checkpoint.",
    )
    parser.add_argument("--medmcqa-document-transformer-checkpoint", type=Path, default=None)
    parser.add_argument("--medqa-document-transformer-checkpoint", type=Path, default=None)

    parser.add_argument(
        "--prompt-profile",
        choices=[
            "focused_v4",
            "paper_exact",
            "paper_exact_terminal",
            "paper_answer_format",
            "paper_compatible_three_anchor",
        ],
        default="focused_v4",
        help=(
            "Generation contract for both no-RAG and multi-document answers. "
            "paper_exact uses only the prompt text reported in RAG2; paper_exact_terminal adds only one "
            "fixed terminal-line serialization rule; paper_answer_format is the legacy strict contract; "
            "paper_compatible_three_anchor reproduces the new training-data contract: free rationale to a fixed "
            "reasoning boundary followed by one constrained A/B/C/D token."
        ),
    )
    parser.add_argument(
        "--answer-decision-mode",
        choices=["free_generation", "constrained_choice"],
        default="free_generation",
        help=(
            "Final MCQ decision protocol. free_generation preserves the rationale+answer generation and parser. "
            "constrained_choice uses the same fixed direct-choice prompt as pre-answer hidden extraction, ends "
            "at 'Final answer:', and greedily emits exactly one token restricted to A/B/C/D."
        ),
    )

    parser.add_argument("--per-source-top-k", type=int, default=10)
    parser.add_argument("--candidate-pool-top-k", type=int, default=40)
    parser.add_argument(
        "--candidate-layout",
        choices=["source_balanced", "released_pubmed_groups"],
        default="source_balanced",
        help=(
            "source_balanced keeps equal candidates from the four logical corpora. "
            "released_pubmed_groups reproduces the public RAG2 README literal layout: "
            "top-k from each PubMed physical-shard group plus top-k from each other corpus."
        ),
    )
    parser.add_argument(
        "--pubmed-shards-per-group",
        type=int,
        default=10,
        help=(
            "Physical PubMed FAISS shards per release-style group. The RAG_Square PubMed DB has 40 shards; "
            "the default therefore creates four PubMed groups and, with top-k=10, 70 candidates."
        ),
    )
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=None,
        help=(
            "For --case rerank_rag, give only this reranked Top-k prefix to the answer LLM. "
            "Keep --rerank-top-k at the largest sweep value (for example 32) so all prefix runs reuse "
            "the same cached retrieval and reranking result. Defaults to --rerank-top-k."
        ),
    )
    parser.add_argument(
        "--filter-rerank-top-k",
        type=int,
        default=None,
        help=(
            "For --case filter_rag or oracle_rag, define the reranked Top-k prefix eligible for filtering. "
            "All --rerank-top-k documents are scored and cached once, but only helpful documents "
            "inside this prefix are augmented; the final document count is therefore between zero and k. "
            "Defaults to --rerank-top-k."
        ),
    )
    parser.add_argument(
        "--paper-balanced-top-k",
        type=int,
        default=None,
        help=(
            "Paper-text balanced-retrieval cutoff for an efficient Top-k sweep. Build one source-balanced "
            "master cache with --per-source-top-k N, --candidate-pool-top-k 4N, and --rerank-top-k 4N. "
            "For a requested k, retain dense ranks 1..k independently inside each logical corpus (4k total), "
            "then select the best k by the already-cached MedCPT cross-encoder scores. Filtering is applied "
            "only after this exact 4k-to-k projection."
        ),
    )
    parser.add_argument(
        "--dense-query-mode",
        choices=["initial", "rationale"],
        default="rationale",
        help=(
            "Query used only by the MedCPT dense retriever. initial uses the original MCQ question "
            "with all options; rationale uses the cached no-RAG rationale_query. The MedCPT "
            "cross-encoder reranker always receives the initial MCQ query, following RAG2."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--query-max-length", type=int, default=256)
    parser.add_argument("--retrieval-batch-size", type=int, default=2048)
    parser.add_argument("--rerank-batch-size", type=int, default=1024)
    parser.add_argument("--cross-encoder-max-length", type=int, default=512)
    parser.add_argument(
        "--cross-encoder-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-gpu-use-float16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faiss-gpu-add-batch-size", type=int, default=500_000)
    parser.add_argument("--faiss-gpu-temp-memory-mb", type=int, default=2048)
    parser.add_argument("--gpu-search-chunk-size", type=int, default=500_000)
    parser.add_argument("--gpu-search-device", default="auto")
    parser.add_argument("--gpu-search-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--metadata-row-cache-size", type=int, default=50_000)
    parser.add_argument(
        "--keep-faiss-indexes-in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache loaded CPU FAISS indexes across shards. Disabled by default because a complete "
            "sharded PubMed IndexFlat can exceed host RAM; enable only for corpus roots known to fit."
        ),
    )
    parser.add_argument("--faiss-mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faiss-shard-threaded", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--retrieval-search-mode",
        choices=["faiss_gpu_source_sequential"],
        default="faiss_gpu_source_sequential",
    )

    parser.add_argument("--filter-batch-size", type=int, default=2048)
    parser.add_argument("--filter-max-input-length", type=int, default=512)
    parser.add_argument("--filter-max-new-tokens", type=int, default=4)
    parser.add_argument("--filter-max-doc-chars", type=int, default=2600)
    parser.add_argument("--filter-device", default="auto")
    parser.add_argument("--filter-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--filter-scoring-method",
        choices=["generate", "log_likelihood", "special_token"],
        default="generate",
    )
    parser.add_argument("--filter-input-format", choices=["auto", "legacy", "official"], default="auto")
    parser.add_argument("--filter-score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument(
        "--filter-evidence-unit",
        choices=["document", "sentence_window", "document_transformer", "preanswer_text_hidden"],
        default="document",
        help=(
            "document scores each reranked chunk once (legacy baseline). sentence_window enumerates "
            "all centred sentence-context windows, aggregates their Helpful probabilities to the document level, "
            "and, by default, passes the original document to the answer LLM. document_transformer extracts the "
            "same features for all windows and predicts the original document label with a learned sequence model."
        ),
    )
    parser.add_argument(
        "--hidden-feature-layer",
        type=int,
        default=28,
        help="Llama hidden-state index used by the trained text+hidden filter.",
    )
    parser.add_argument(
        "--hidden-feature-batch-size",
        type=int,
        default=64,
        help="Batch size for no-gold h0/hD extraction with the target Llama.",
    )
    parser.add_argument(
        "--hidden-filter-question-batch-size",
        type=int,
        default=32,
        help=(
            "Number of questions committed per resumable text+hidden filter-score batch. "
            "Completed batches are retained when --filter-cache-only is interrupted."
        ),
    )
    parser.add_argument("--hidden-feature-max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--hidden-feature-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--hidden-feature-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument(
        "--hidden-filter-helpful-threshold",
        type=float,
        default=0.5,
        help="Helpful probability threshold for the text+hidden filter; 0.5 is its two-label argmax.",
    )
    parser.add_argument(
        "--filter-generation-context-unit",
        choices=["document", "best_helpful_window"],
        default="document",
        help=(
            "For --case filter_rag, document supplies each passing reranked document intact (the controlled "
            "baseline). best_helpful_window instead supplies exactly one centred sentence-context window per "
            "passing document: the highest-P(Helpful) window among its Helpful-scored windows. This changes "
            "only final LLM context, not retrieval, reranking, filtering, or the filter-score cache."
        ),
    )
    parser.add_argument(
        "--filter-window-context-sentences",
        type=int,
        default=1,
        help=(
            "Sentence context on each side of a window centre. Use 1 to match the attribution-window "
            "filter training inputs (at most three sentences)."
        ),
    )
    parser.add_argument(
        "--filter-window-question-batch-size",
        type=int,
        default=32,
        help=(
            "Number of MCQ questions whose document windows are materialised together while one filter model "
            "is resident. This bounds RAM while retaining large GPU scoring batches."
        ),
    )
    parser.add_argument(
        "--filter-window-helpful-threshold",
        type=float,
        default=0.5,
        help=(
            "Fallback document-pass threshold on max window Helpful probability. Prefer "
            "--filter-window-thresholds-path calibrated on the filter validation split."
        ),
    )
    parser.add_argument(
        "--filter-window-thresholds-path",
        type=Path,
        default=None,
        help=(
            "JSON output of calibrate_rag2_window_filter_threshold.py. It supplies separate MedMCQA and MedQA "
            "document thresholds and is recorded in the cache fingerprint."
        ),
    )
    parser.add_argument(
        "--document-transformer-helpful-threshold",
        type=float,
        default=0.5,
        help=(
            "Fallback Helpful threshold for learned Document Transformer outputs. Route-specific "
            "thresholds below override this value."
        ),
    )
    parser.add_argument(
        "--medmcqa-document-transformer-helpful-threshold",
        type=float,
        default=None,
        help="Validation-selected threshold for the MedMCQA model (also routes all MMLU datasets).",
    )
    parser.add_argument(
        "--medqa-document-transformer-helpful-threshold",
        type=float,
        default=None,
        help="Validation-selected threshold for the MedQA model.",
    )
    parser.add_argument(
        "--document-transformer-batch-size",
        type=int,
        default=512,
        help="Number of variable-length document sequences scored per Document Transformer batch.",
    )
    parser.add_argument(
        "--filter-cache-only",
        action="store_true",
        help=(
            "For --case filter_rag, score all reranked documents and persist the filter decisions, "
            "then exit before answer generation. Subsequent Top-k runs reuse this cache."
        ),
    )
    parser.add_argument(
        "--rebuild-filter-cache",
        action="store_true",
        help="Ignore an otherwise compatible completed filter-score cache and score all documents again.",
    )

    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--rationale-max-new-tokens",
        type=int,
        default=384,
        help="Token cap used only when creating reusable no-RAG rationale artifacts.",
    )
    parser.add_argument(
        "--rationale-length-retry-attempts",
        type=int,
        default=1,
        help="Number of compact retries for no-RAG rationales that hit the token cap.",
    )
    parser.add_argument(
        "--rationale-length-retry-max-new-tokens",
        type=int,
        default=384,
        help="Token cap for a no-RAG rationale length retry.",
    )
    parser.add_argument(
        "--rationale-invalid-retry-attempts",
        type=int,
        default=1,
        help="Number of malformed no-RAG rationale retries.",
    )
    parser.add_argument(
        "--rationale-invalid-retry-max-new-tokens",
        type=int,
        default=384,
        help="Token cap for a malformed no-RAG rationale retry.",
    )
    parser.add_argument(
        "--rationale-retry-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry no-RAG rationale rows that fail the quality contract.",
    )
    parser.add_argument(
        "--rationale-retry-invalid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate cached no-RAG rationale rows with an invalid or missing answer.",
    )
    parser.add_argument(
        "--rationale-choice-anchored-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a choice-selection repair pass for unresolved no-RAG rationale rows.",
    )
    parser.add_argument("--max-doc-chars", type=int, default=2200)
    parser.add_argument(
        "--document-packing",
        choices=["fixed_chars", "dynamic_token_budget"],
        default="fixed_chars",
        help=(
            "fixed_chars applies --max-doc-chars independently to every document. "
            "dynamic_token_budget keeps every selected document while allocating its text against the actual "
            "Llama chat-prompt token budget."
        ),
    )
    parser.add_argument(
        "--document-token-safety-margin",
        type=int,
        default=128,
        help=(
            "Tokens kept free beyond --max-new-tokens when --document-packing=dynamic_token_budget. "
            "This absorbs chat-template/version overhead and prevents context-length failures."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--llm-max-model-len", type=int, default=8192)
    parser.add_argument("--gdn-prefill-backend", choices=["auto", "flashinfer", "triton", "cutedsl"], default="triton")
    parser.add_argument("--vllm-performance-mode", choices=["balanced", "interactivity", "throughput"], default="throughput")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=131_072)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--format-retry-attempts", type=int, default=1)
    parser.add_argument("--include-doc-text-in-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--candidate-cache-only",
        action="store_true",
        help=(
            "Build/validate the dense-retrieval plus MedCPT-reranking cache, then exit before loading the "
            "answer LLM. Use this once before a --generation-top-k sweep."
        ),
    )
    parser.add_argument("--regenerate-rationales", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rationale-artifact-policy",
        choices=["reuse_only", "repair_invalid"],
        default="repair_invalid",
        help=(
            "reuse_only reads cached no-RAG rationales without writing or retrying them; invalid rows are "
            "excluded consistently. repair_invalid retains the legacy repair behavior."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def active_no_rag_prompt_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_compatible_three_anchor":
        return ANCHORED_PROMPT_VERSION
    if args.prompt_profile == "paper_exact_terminal":
        return PAPER_EXACT_TERMINAL_PROMPT_VERSION
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_PROMPT_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_PROMPT_VERSION
    return PROMPT_VERSION


def active_document_prompt_version(args: argparse.Namespace) -> str:
    if args.prompt_profile == "paper_compatible_three_anchor":
        return ANCHORED_PROMPT_VERSION
    if args.prompt_profile == "paper_exact_terminal":
        return PAPER_EXACT_TERMINAL_DOCUMENT_PROMPT_VERSION
    if args.prompt_profile == "paper_answer_format":
        return PAPER_ANSWER_FORMAT_DOCUMENT_PROMPT_VERSION
    if args.prompt_profile == "paper_exact":
        return PAPER_EXACT_DOCUMENT_PROMPT_VERSION
    return DOCUMENT_PROMPT_VERSION


def sample_key(sample: BenchmarkSample) -> str:
    return f"{sample.dataset}::{sample.split}::{sample.id}::{sample.row_idx}"


def artifact_path(args: argparse.Namespace, dataset: str) -> Path:
    root = args.rationale_artifact_root or (args.cache_root / "no_rag_rationales")
    return root / "no_rag" / dataset / args.split / "no_rag_generations.jsonl"


def rationale_artifact_root(args: argparse.Namespace) -> Path:
    return args.rationale_artifact_root or (args.cache_root / "no_rag_rationales")


def load_latest_artifacts_with_issues(
    path: Path,
    samples: list[BenchmarkSample],
    *,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed rationale JSONL: {path}:{line_no}") from exc
            latest[int(row.get("row_idx", -1))] = row
    missing = [idx for idx in range(len(samples)) if idx not in latest]
    if missing:
        raise RuntimeError(f"Incomplete rationale artifact: {path} missing={len(missing)} first={missing[:10]}")
    rows = [latest[idx] for idx in range(len(samples))]
    failures: dict[int, str] = {}
    for idx, (sample, row) in enumerate(zip(samples, rows)):
        parsed = row.get("parsed") or {}
        if parsed.get("parse_errors") or not parsed.get("rationale_query") or not parsed.get("final_answer"):
            raw_generation = str(row.get("model_raw_generation") or row.get("no_rag_generation") or "")
            reparsed = None
            if prompt_version != ANCHORED_PROMPT_VERSION:
                reparsed = parse_mcq_output_for_prompt_profile(
                    raw_generation,
                    normalized_options(sample.raw),
                    (
                        "paper_exact_terminal"
                        if prompt_version == PAPER_EXACT_TERMINAL_PROMPT_VERSION
                        else "paper_exact"
                        if prompt_version == PAPER_EXACT_PROMPT_VERSION
                        else "focused_v4"
                    ),
                )
            if reparsed is not None and not reparsed.parse_errors and reparsed.rationale_query and reparsed.final_answer:
                recovered_query = (
                    reparsed.visible_text
                    if prompt_version in {PAPER_EXACT_PROMPT_VERSION, PAPER_EXACT_TERMINAL_PROMPT_VERSION}
                    else reparsed.rationale_query
                )
                row = dict(row)
                row["parsed"] = {
                    **parsed,
                    "visible_text": reparsed.visible_text,
                    "rationale": reparsed.rationale,
                    "rationale_query": recovered_query,
                    "rationale_query_normalized": reparsed.rationale_query_normalized,
                    "final_answer": reparsed.final_answer,
                    "final_answer_correct": reparsed.final_answer in set(sample.answers),
                    "parse_errors": [],
                    "parser_recovery": (
                        "paper_exact_final_line_conservative_no_rewrite"
                        if prompt_version in {PAPER_EXACT_PROMPT_VERSION, PAPER_EXACT_TERMINAL_PROMPT_VERSION}
                        else "accepted_unmarked_paper_style_rationale"
                    ),
                }
                rows[idx] = row
                parsed = row["parsed"]
        reasons = []
        if str(row.get("sample_id")) != sample.id:
            reasons.append("sample_id_mismatch")
        if row.get("prompt_version") != prompt_version:
            reasons.append("prompt_version_mismatch")
        if not parsed.get("rationale_query"):
            reasons.append("missing_rationale_query")
        if not parsed.get("final_answer"):
            reasons.append("missing_final_answer")
        if parsed.get("parse_errors"):
            reasons.append("parse_errors")
        if reasons:
            failures[idx] = f"{sample.dataset}:{idx}:{sample.id}:{','.join(reasons)}"
    return rows, failures


def load_latest_artifacts(
    path: Path,
    samples: list[BenchmarkSample],
    *,
    prompt_version: str,
) -> list[dict[str, Any]]:
    rows, failures = load_latest_artifacts_with_issues(
        path,
        samples,
        prompt_version=prompt_version,
    )
    if failures:
        examples = list(failures.values())[:20]
        raise RuntimeError("Invalid no-RAG rationale rows remain:\n" + "\n".join(examples))
    return rows


def load_benchmarks(args: argparse.Namespace) -> tuple[list[BenchmarkSample], dict[str, list[BenchmarkSample]]]:
    grouped: dict[str, list[BenchmarkSample]] = {}
    combined: list[BenchmarkSample] = []
    for dataset in args.datasets:
        path = resolve_benchmark_path(args.benchmark_root, "mcq", args.collection, dataset, args.split)
        samples = load_benchmark_samples(path, "mcq", args.collection, dataset, args.split)
        grouped[dataset] = samples
        combined.extend(samples)
        logging.info("[%s] loaded %s benchmark samples", dataset, len(samples))
    return combined, grouped


def rationales_are_ready(args: argparse.Namespace, grouped: dict[str, list[BenchmarkSample]]) -> bool:
    try:
        for dataset, samples in grouped.items():
            load_latest_artifacts(
                artifact_path(args, dataset),
                samples,
                prompt_version=active_no_rag_prompt_version(args),
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.info("Rationale cache is not ready: %s", exc)
        return False
    return True


def generate_missing_rationales(args: argparse.Namespace) -> None:
    if args.prompt_profile == "paper_compatible_three_anchor":
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_rag2_anchored_no_rag_train.py"),
            "--datasets",
            *args.datasets,
            "--split",
            args.split,
            "--benchmark-root",
            str(args.benchmark_root / "mcq" / args.collection),
            "--model-name-or-path",
            str(args.llm_model_path),
            "--output-root",
            str(rationale_artifact_root(args)),
            "--generation-batch-size",
            str(args.generation_batch_size),
            "--max-new-tokens",
            str(args.rationale_max_new_tokens),
            "--retry-max-new-tokens",
            str(max(args.rationale_max_new_tokens, args.rationale_length_retry_max_new_tokens)),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--llm-max-model-len",
            str(args.llm_max_model_len),
            "--vllm-max-num-seqs",
            str(args.vllm_max_num_seqs),
            "--vllm-max-num-batched-tokens",
            str(args.vllm_max_num_batched_tokens),
            "--vllm-performance-mode",
            args.vllm_performance_mode,
        ]
        if args.regenerate_rationales:
            command.append("--no-resume")
        logging.info(
            "Generating shared anchored no-RAG rationale+answer queries for %s datasets.",
            len(args.datasets),
        )
        subprocess.run(command, check=True, env=os.environ.copy())
        return

    paper_exact = args.prompt_profile in {"paper_exact", "paper_exact_terminal"}
    length_retry_attempts = 0 if paper_exact else args.rationale_length_retry_attempts
    invalid_retry_attempts = 0 if paper_exact else args.rationale_invalid_retry_attempts
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_rag2_no_rag_rationales.py"),
        "--datasets",
        *args.datasets,
        "--split",
        args.split,
        "--collection",
        args.collection,
        "--benchmark-root",
        str(args.benchmark_root),
        "--llm-model-path",
        str(args.llm_model_path),
        "--artifact-root",
        str(rationale_artifact_root(args)),
        "--results-root",
        str(args.results_root / "_rationale_generation"),
        "--run-name",
        "rag2_mcq_eval_no_rag_rationale",
        "--prompt-profile",
        args.prompt_profile,
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--max-new-tokens",
        str(args.rationale_max_new_tokens),
        "--length-retry-attempts",
        str(length_retry_attempts),
        "--length-retry-max-new-tokens",
        str(args.rationale_length_retry_max_new_tokens),
        "--invalid-retry-attempts",
        str(invalid_retry_attempts),
        "--invalid-retry-max-new-tokens",
        str(args.rationale_invalid_retry_max_new_tokens),
        "--quality-retry-max-new-tokens",
        str(args.rationale_invalid_retry_max_new_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--llm-max-model-len",
        str(args.llm_max_model_len),
        "--gdn-prefill-backend",
        args.gdn_prefill_backend,
        "--vllm-performance-mode",
        args.vllm_performance_mode,
        "--vllm-max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--vllm-max-num-batched-tokens",
        str(args.vllm_max_num_batched_tokens),
    ]
    if args.regenerate_rationales:
        command.append("--no-resume")
    if args.rationale_retry_quality and not paper_exact:
        command.append("--retry-quality")
    else:
        command.append("--no-retry-quality")
    if args.rationale_retry_invalid and not paper_exact:
        command.append("--retry-invalid")
    else:
        command.append("--no-retry-invalid")
    if args.rationale_choice_anchored_retry and not paper_exact:
        command.append("--choice-anchored-retry")
    else:
        command.append("--no-choice-anchored-retry")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if not args.enable_prefix_caching:
        command.append("--no-enable-prefix-caching")
    logging.info("Generating shared no-RAG rationale queries for %s datasets.", len(args.datasets))
    subprocess.run(command, check=True, env=os.environ.copy())


def ensure_rationale_artifacts(
    args: argparse.Namespace,
    grouped: dict[str, list[BenchmarkSample]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    def load_all() -> dict[str, list[dict[str, Any]]]:
        return {
            dataset: load_latest_artifacts(
                artifact_path(args, dataset),
                samples,
                prompt_version=active_no_rag_prompt_version(args),
            )
            for dataset, samples in grouped.items()
        }

    def load_with_explicit_exclusions() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        """Read existing artifacts without modifying their append-only JSONL files."""
        artifacts: dict[str, list[dict[str, Any]]] = {}
        exclusions: list[dict[str, Any]] = []
        for dataset, dataset_samples in grouped.items():
            rows, issues = load_latest_artifacts_with_issues(
                artifact_path(args, dataset),
                dataset_samples,
                prompt_version=active_no_rag_prompt_version(args),
            )
            valid_indices = [idx for idx in range(len(dataset_samples)) if idx not in issues]
            artifacts[dataset] = [rows[idx] for idx in valid_indices]
            exclusions.extend(
                {
                    "dataset": dataset_samples[idx].dataset,
                    "row_idx": idx,
                    "sample_id": dataset_samples[idx].id,
                    "reason": issues[idx],
                }
                for idx in sorted(issues)
            )
        return artifacts, exclusions

    if args.rationale_artifact_policy == "reuse_only":
        if args.regenerate_rationales:
            raise ValueError("--regenerate-rationales cannot be combined with --rationale-artifact-policy reuse_only.")
        artifacts, exclusions = load_with_explicit_exclusions()
        if exclusions:
            logging.warning(
                "Reusing existing no-RAG rationales: excluding %s malformed row(s) without retrying or "
                "modifying the artifacts.",
                len(exclusions),
            )
        else:
            logging.info("Reusing existing no-RAG rationale artifacts without modification.")
        return artifacts, exclusions

    if args.regenerate_rationales:
        generate_missing_rationales(args)

    try:
        return load_all(), []
    except FileNotFoundError:
        # No cache exists yet: run the normal resumable generation path.
        generate_missing_rationales(args)
        return load_all(), []
    except RuntimeError as exc:
        # Keep all valid cached rows.  The generator's --retry-invalid mode only
        # appends repaired versions of malformed rows; load_latest_artifacts then
        # uses the latest row for each sample id/index.
        repair_args = copy.copy(args)
        repair_args.rationale_retry_invalid = True
        repair_args.rationale_choice_anchored_retry = True
        repair_args.rationale_invalid_retry_attempts = max(
            2, args.rationale_invalid_retry_attempts
        )
        repair_args.rationale_invalid_retry_max_new_tokens = max(
            1024, args.rationale_invalid_retry_max_new_tokens
        )
        logging.warning(
            "Malformed no-RAG rationale rows detected; repairing only those rows: %s",
            exc,
        )
        generate_missing_rationales(repair_args)
        try:
            return load_all(), []
        except RuntimeError as repair_exc:
            # A result table is still useful if an exceptionally small number of
            # rows remain malformed after all repairs.  Do not invent a choice or
            # rationale: remove only those rows from *every* evaluation condition
            # and make the exclusion explicit in the run artifacts.
            artifacts, exclusions = load_with_explicit_exclusions()
            if not exclusions:
                raise repair_exc
            logging.error(
                "No-RAG artifact repair left %s malformed row(s); excluding them "
                "from every evaluation condition and recording the details.",
                len(exclusions),
            )
            return artifacts, exclusions


def embedding_cache_current(
    path: Path,
    source: Path,
    rows: int,
    *,
    dense_query_mode: str,
    prompt_version: str,
    max_length: int,
    model_path: Path,
) -> bool:
    manifest_path = path / "manifest.json"
    embedding_path = path / "embeddings.npy"
    if not manifest_path.exists() or not embedding_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_path, mmap_mode="r")
    except Exception:
        return False
    stat = source.stat()
    return (
        tuple(embeddings.shape) == (rows, 768)
        and int(manifest.get("source_size_bytes", -1)) == stat.st_size
        and int(manifest.get("source_mtime_ns", -1)) == stat.st_mtime_ns
        and manifest.get("dense_query_mode") == dense_query_mode
        and manifest.get("prompt_version") == prompt_version
        and int(manifest.get("max_length", -1)) == max_length
        and manifest.get("model_path") == str(model_path)
    )


def dense_query_text(args: argparse.Namespace, sample: BenchmarkSample, artifact: dict[str, Any]) -> str:
    """Return the sole query variant used by the first-stage dense retriever."""
    if args.dense_query_mode == "initial":
        return format_question(sample.raw)
    rationale = str((artifact.get("parsed") or {}).get("rationale_query") or "").strip()
    if not rationale:
        raise RuntimeError(f"Missing cached rationale query for {sample.dataset}:{sample.id}")
    return rationale


def dense_query_texts(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    artifacts: list[dict[str, Any]],
) -> list[str]:
    if len(samples) != len(artifacts):
        raise RuntimeError(f"Dense-query alignment mismatch: samples={len(samples)} artifacts={len(artifacts)}")
    return [dense_query_text(args, sample, artifact) for sample, artifact in zip(samples, artifacts)]


def ensure_dense_query_embeddings(
    args: argparse.Namespace,
    grouped: dict[str, list[BenchmarkSample]],
    artifacts: dict[str, list[dict[str, Any]]],
) -> dict[str, np.ndarray]:
    cache_dirs = {
        dataset: args.cache_root / "dense_query_embeddings" / args.dense_query_mode / dataset / args.split
        for dataset in args.datasets
    }
    pending = [
        dataset
        for dataset in args.datasets
        if not embedding_cache_current(
            cache_dirs[dataset],
            artifact_path(args, dataset),
            len(grouped[dataset]),
            dense_query_mode=args.dense_query_mode,
            prompt_version=active_no_rag_prompt_version(args),
            max_length=args.query_max_length,
            model_path=args.query_encoder_path,
        )
    ]
    if pending:
        logging.info("Embedding %s dense queries for: %s", args.dense_query_mode, ", ".join(pending))
        tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path, local_files_only=True, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
        model_kwargs: dict[str, Any] = {"local_files_only": True, "attn_implementation": "eager"}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        model = AutoModel.from_pretrained(args.query_encoder_path, **model_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        try:
            for dataset in pending:
                texts = dense_query_texts(args, grouped[dataset], artifacts[dataset])
                output_dir = cache_dirs[dataset]
                output_dir.mkdir(parents=True, exist_ok=True)
                temp_path = output_dir / "embeddings.npy.tmp"
                output_path = output_dir / "embeddings.npy"
                array = np.lib.format.open_memmap(
                    temp_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(texts), int(model.config.hidden_size)),
                )
                progress = StageProgress(
                    total=len(texts),
                    desc=f"Embed{args.dense_query_mode.title()}:{dataset}",
                    enabled=True,
                )
                try:
                    for start in range(0, len(texts), args.embedding_batch_size):
                        end = min(start + args.embedding_batch_size, len(texts))
                        encoded = tokenizer(
                            texts[start:end],
                            padding=True,
                            truncation=True,
                            max_length=args.query_max_length,
                            return_tensors="pt",
                        )
                        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
                        autocast = (
                            torch.autocast(device_type="cuda", dtype=dtype)
                            if device.type == "cuda" and dtype is not None
                            else torch.autocast(device_type="cpu", enabled=False)
                        )
                        with torch.inference_mode(), autocast:
                            output = model(**encoded).last_hidden_state[:, 0, :]
                        array[start:end] = output.float().cpu().numpy()
                        progress.update(end - start)
                finally:
                    progress.close()
                array.flush()
                del array
                temp_path.replace(output_path)
                stat = artifact_path(args, dataset).stat()
                write_json(
                    output_dir / "manifest.json",
                    {
                        "type": "rag2_mcq_eval_dense_query_embeddings",
                        "dataset": dataset,
                        "split": args.split,
                        "rows": len(texts),
                        "dimension": int(model.config.hidden_size),
                        "dense_query_mode": args.dense_query_mode,
                        "prompt_version": active_no_rag_prompt_version(args),
                        "source_path": str(artifact_path(args, dataset)),
                        "source_size_bytes": stat.st_size,
                        "source_mtime_ns": stat.st_mtime_ns,
                        "model_path": str(args.query_encoder_path),
                        "max_length": args.query_max_length,
                    },
                )
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        dataset: np.asarray(np.load(cache_dirs[dataset] / "embeddings.npy", mmap_mode="r"), dtype="float32")
        for dataset in args.datasets
    }


def document_to_dict(doc: RetrievedDocument, include_text: bool) -> dict[str, Any]:
    row = doc.to_dict(include_text=include_text)
    row["stable_id"] = doc.stable_id
    if "source_retrieval_rank" in doc.metadata:
        row["source_retrieval_rank"] = doc.metadata["source_retrieval_rank"]
    if "retrieval_bucket" in doc.metadata:
        row["retrieval_bucket"] = doc.metadata["retrieval_bucket"]
    return row


def document_from_dict(row: dict[str, Any]) -> RetrievedDocument:
    metadata = dict(row.get("metadata") or {})
    if row.get("source_retrieval_rank") is not None:
        metadata["source_retrieval_rank"] = int(row["source_retrieval_rank"])
    if row.get("retrieval_bucket") is not None:
        metadata["retrieval_bucket"] = str(row["retrieval_bucket"])
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
        filter_score=row.get("filter_score"),
        filter_rank=row.get("filter_rank"),
        filter_prediction=row.get("filter_prediction"),
        filter_prob_helpful=row.get("filter_prob_helpful"),
        metadata=metadata,
    )


def project_paper_balanced_candidates(
    initial_document_lists: list[list[RetrievedDocument]],
    fully_reranked_document_lists: list[list[RetrievedDocument]],
    *,
    sources: list[str],
    top_k: int,
) -> tuple[list[list[RetrievedDocument]], list[list[RetrievedDocument]]]:
    """Materialize the paper-described ``4 corpora x k -> rerank Top-k`` contract.

    The master cache contains the dense Top-N from every logical corpus and
    cross-encoder scores for all 4N documents.  Cross-encoder scores are
    document-local, so slicing each corpus by its stored dense rank and sorting
    the surviving 4k documents is equivalent to rerunning the reranker for
    every k.  All sweep conditions can therefore reuse one expensive pass.
    """
    if top_k <= 0:
        raise ValueError("paper-balanced top-k must be positive")
    if len(initial_document_lists) != len(fully_reranked_document_lists):
        raise RuntimeError(
            "Paper-balanced projection row mismatch: "
            f"initial={len(initial_document_lists)} reranked={len(fully_reranked_document_lists)}"
        )

    expected_source_counts = {source: top_k for source in sources}
    expected_pool_size = top_k * len(sources)
    projected_initial: list[list[RetrievedDocument]] = []
    projected_reranked: list[list[RetrievedDocument]] = []

    for row_index, (initial_docs, reranked_docs) in enumerate(
        zip(initial_document_lists, fully_reranked_document_lists, strict=True)
    ):
        dense_prefix: list[RetrievedDocument] = []
        inferred_source_ranks: Counter[str] = Counter()
        source_rank_by_stable_id: dict[str, int] = {}
        for document in initial_docs:
            # ``initial_docs`` is the globally score-sorted merge of complete
            # source-local Top-N lists. Therefore the occurrence index within
            # each source is its exact dense rank (stable sorting preserves
            # source order on score ties). Derive it from this authoritative
            # order instead of trusting cached metadata: candidate caches made
            # before 2026-08-25 could share the metadata dict returned by the
            # metadata-store row cache, allowing later questions to overwrite
            # ``source_retrieval_rank`` without changing document identities,
            # retrieval scores, rerank scores, or filter decisions.
            logical_source = str(document.metadata.get("retrieval_bucket") or document.source)
            inferred_source_ranks[logical_source] += 1
            source_rank = inferred_source_ranks[logical_source]
            source_rank_by_stable_id[document.stable_id] = int(source_rank)
            if int(source_rank) <= top_k:
                projected_document = copy.copy(document)
                projected_document.metadata = dict(document.metadata)
                projected_document.metadata["retrieval_bucket"] = logical_source
                projected_document.metadata["source_retrieval_rank"] = int(source_rank)
                dense_prefix.append(projected_document)

        actual_source_counts = Counter(
            str(document.metadata.get("retrieval_bucket") or document.source)
            for document in dense_prefix
        )
        if dict(actual_source_counts) != expected_source_counts or len(dense_prefix) != expected_pool_size:
            raise RuntimeError(
                "Paper-balanced dense-prefix invariant failed: "
                f"row={row_index} expected={expected_source_counts} "
                f"actual={dict(actual_source_counts)} total={len(dense_prefix)}"
            )

        eligible_reranked: list[RetrievedDocument] = []
        for document in reranked_docs:
            source_rank = source_rank_by_stable_id.get(document.stable_id)
            if source_rank is None:
                raise RuntimeError(
                    "Paper-balanced projection cannot align a reranked document to the dense pool: "
                    f"row={row_index} document={document.stable_id}"
                )
            if int(source_rank) <= top_k:
                projected_document = copy.copy(document)
                projected_document.metadata = dict(document.metadata)
                projected_document.metadata["source_retrieval_rank"] = int(source_rank)
                eligible_reranked.append(projected_document)
        if len(eligible_reranked) != expected_pool_size:
            raise RuntimeError(
                "Paper-balanced rerank pool is incomplete. The master cache must retain scores/text for "
                f"all candidates: row={row_index} expected={expected_pool_size} "
                f"actual={len(eligible_reranked)}"
            )
        eligible_reranked.sort(
            key=lambda document: (
                document.rerank_score
                if document.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )
        selected = eligible_reranked[:top_k]
        for rank, document in enumerate(selected, start=1):
            document.rerank_rank = rank
        projected_initial.append(dense_prefix)
        projected_reranked.append(selected)

    return projected_initial, projected_reranked


def candidate_query_fingerprint(
    samples: list[BenchmarkSample],
    query_texts: list[str],
) -> str:
    """Stable identity for the actual first-stage queries used in a cache.

    The no-RAG artifact files are append-only: retrying an invalid row changes
    their mtime and size even when every *valid* rationale query is unchanged.
    File stats therefore must not decide whether a 70-candidate retrieval cache
    can be reused.  Bind the cache to the ordered sample keys and query texts
    instead.
    """
    if len(samples) != len(query_texts):
        raise RuntimeError(
            f"Candidate-query alignment mismatch: samples={len(samples)} queries={len(query_texts)}"
        )
    digest = hashlib.sha256()
    for sample, query_text in zip(samples, query_texts):
        payload = {"key": sample_key(sample), "query_text": str(query_text)}
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def candidate_cache_dir(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    query_texts: list[str],
) -> Path:
    source_stats = {}
    for source in args.sources:
        manifest = args.vector_db_root / source / "manifest.json"
        source_stats[source] = manifest.stat().st_mtime_ns if manifest.exists() else None
    payload = {
        "datasets": args.datasets,
        "split": args.split,
        # Candidate identities depend on the actual corpus, not just on the
        # manifests' timestamps.  Keep caches from separately-built corpus
        # roots (e.g. the legacy MedCorp DB and RAG_Square) disjoint even if
        # their manifests happen to share the same metadata timestamps.
        "vector_db_root": str(args.vector_db_root.resolve()),
        "sources": args.sources,
        "source_stats": source_stats,
        # The query-content hash, unlike append-only artifact file statistics,
        # remains unchanged when an already-known malformed row is retried.
        "candidate_query_fingerprint": candidate_query_fingerprint(samples, query_texts),
        "per_source_top_k": args.per_source_top_k,
        "candidate_pool_top_k": args.candidate_pool_top_k,
        "candidate_layout": args.candidate_layout,
        "pubmed_shards_per_group": args.pubmed_shards_per_group,
        "rerank_top_k": args.rerank_top_k,
        "dense_query_mode": args.dense_query_mode,
        "prompt_profile": args.prompt_profile,
        "rationale_prompt_version": active_no_rag_prompt_version(args),
        "query_max_length": args.query_max_length,
        "query_encoder_path": str(args.query_encoder_path.resolve()),
        "cross_encoder_path": str(args.cross_encoder_path.resolve()),
        "cross_encoder_max_length": args.cross_encoder_max_length,
        "cross_encoder_attn_implementation": args.cross_encoder_attn_implementation,
        "retrieval_search_mode": args.retrieval_search_mode,
        "faiss_gpu_use_float16": args.faiss_gpu_use_float16,
    }
    cache_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return args.cache_root / "candidates" / cache_id


def rerank_query_text(
    args: argparse.Namespace,
    sample: BenchmarkSample,
    rationale_query: str,
) -> str:
    """RAG2 reranks every dense-retrieved candidate with the original MCQ query.

    Rationale-guided query formulation applies to first-stage dense retrieval;
    Figure 1 of the target paper specifies cross-encoding the *initial query*
    with each candidate snippet for reranking.  ``rationale_query`` remains an
    argument to keep the caller interface explicit and to avoid accidentally
    coupling the reranker to the selected dense-query mode.
    """
    del args, rationale_query
    return format_question(sample.raw)


def load_candidate_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Ignoring malformed candidate row: %s:%s", path, line_no)
                continue
            rows[str(row.get("key") or "")] = row
    return rows


def candidate_cache_manifest_matches(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    *,
    rows: int,
    query_fingerprint: str,
) -> bool:
    """Check static retrieval/reranking settings before reading a cache file."""
    if manifest.get("type") != "rag2_mcq_eval_candidates":
        return False
    if int(manifest.get("rows", -1)) != rows:
        return False
    if list(manifest.get("sources") or []) != list(args.sources):
        return False
    if int(manifest.get("per_source_top_k", -1)) != args.per_source_top_k:
        return False
    if int(manifest.get("candidate_pool_top_k", -1)) != args.candidate_pool_top_k:
        return False
    if int(manifest.get("rerank_top_k", -1)) != args.rerank_top_k:
        return False
    if manifest.get("dense_query_mode") != args.dense_query_mode:
        return False
    if manifest.get("prompt_profile") != args.prompt_profile:
        return False
    if manifest.get("rationale_prompt_version") != active_no_rag_prompt_version(args):
        return False
    if Path(str(manifest.get("cross_encoder_path") or "")).resolve() != args.cross_encoder_path.resolve():
        return False
    # Older, completed caches predate these fields.  They are still validated
    # exactly against all stored query texts below before being reused.
    if manifest.get("candidate_layout", "source_balanced") != args.candidate_layout:
        return False
    if int(manifest.get("pubmed_shards_per_group", 1)) != args.pubmed_shards_per_group:
        return False
    if "vector_db_root" in manifest and Path(str(manifest["vector_db_root"])).resolve() != args.vector_db_root.resolve():
        return False
    if "candidate_query_fingerprint" in manifest and manifest["candidate_query_fingerprint"] != query_fingerprint:
        return False
    return True


def find_compatible_candidate_cache(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    query_texts: list[str],
    preferred_cache_dir: Path,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Reuse a completed cache even if an old artifact-stat key differed.

    This is deliberately strict: all sample keys, dense queries and reranked
    document counts must match.  It fixes interrupted/multi-prefix sweeps
    without allowing a cache made from a different rationale set to leak in.
    """
    preferred_path = preferred_cache_dir / "candidates.jsonl"
    if args.rebuild_candidates:
        return preferred_cache_dir, {}

    root = args.cache_root / "candidates"
    candidate_dirs = [preferred_cache_dir]
    if root.exists():
        candidate_dirs.extend(
            path
            for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime_ns, reverse=True)
            if path.is_dir() and path != preferred_cache_dir
        )

    expected_queries = {sample_key(sample): str(query) for sample, query in zip(samples, query_texts)}
    fingerprint = candidate_query_fingerprint(samples, query_texts)
    for cache_dir in candidate_dirs:
        manifest_path = cache_dir / "manifest.json"
        output_path = cache_dir / "candidates.jsonl"
        if not manifest_path.exists() or not output_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not candidate_cache_manifest_matches(
            manifest, args, rows=len(samples), query_fingerprint=fingerprint
        ):
            continue
        existing = load_candidate_rows(output_path)
        if any(
            key not in existing
            or str(existing[key].get("query_text") or "") != query_text
            or len(existing[key].get("reranked_documents") or []) < args.rerank_top_k
            for key, query_text in expected_queries.items()
        ):
            continue
        if cache_dir != preferred_cache_dir:
            logging.info(
                "Reusing compatible completed candidate cache: %s (preferred key was %s)",
                cache_dir,
                preferred_cache_dir,
            )
        return cache_dir, existing

    if args.candidate_cache_source_path is not None:
        source_path = args.candidate_cache_source_path.resolve()
        source_manifest_path = source_path.parent / "manifest.json"
        if not source_path.is_file() or not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"Explicit candidate-cache source or manifest is missing: {source_path}"
            )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        static_checks = {
            "type": "rag2_mcq_eval_candidates",
            "sources": list(args.sources),
            "per_source_top_k": args.per_source_top_k,
            "candidate_pool_top_k": args.candidate_pool_top_k,
            "candidate_layout": args.candidate_layout,
            "pubmed_shards_per_group": args.pubmed_shards_per_group,
            "dense_query_mode": args.dense_query_mode,
            "prompt_profile": args.prompt_profile,
            "rationale_prompt_version": active_no_rag_prompt_version(args),
        }
        for name, expected in static_checks.items():
            actual = source_manifest.get(name)
            if actual != expected:
                raise RuntimeError(
                    f"Explicit candidate cache setting mismatch for {name}: expected={expected!r} actual={actual!r}"
                )
        source_rerank_top_k = int(source_manifest.get("rerank_top_k", -1))
        if source_rerank_top_k < args.rerank_top_k:
            raise RuntimeError(
                "Explicit candidate cache has fewer reranked documents than requested: "
                f"source={source_rerank_top_k} requested={args.rerank_top_k}"
            )
        if Path(str(source_manifest.get("cross_encoder_path") or "")).resolve() != args.cross_encoder_path.resolve():
            raise RuntimeError("Explicit candidate cache used a different cross encoder")
        if Path(str(source_manifest.get("vector_db_root") or "")).resolve() != args.vector_db_root.resolve():
            raise RuntimeError("Explicit candidate cache used a different vector DB root")
        source_rows = load_candidate_rows(source_path)
        subset_rows: list[dict[str, Any]] = []
        for sample, query_text in zip(samples, query_texts, strict=True):
            key = sample_key(sample)
            row = source_rows.get(key)
            if row is None:
                raise RuntimeError(f"Explicit candidate cache lacks sample: {key}")
            if str(row.get("query_text") or "") != str(query_text):
                raise RuntimeError(f"Explicit candidate cache dense query differs for sample: {key}")
            if len(row.get("initial_documents") or []) != args.candidate_pool_top_k:
                raise RuntimeError(f"Explicit candidate cache initial candidate count differs for sample: {key}")
            if len(row.get("reranked_documents") or []) < args.rerank_top_k:
                raise RuntimeError(f"Explicit candidate cache reranked candidate count is short for sample: {key}")
            # A completed Top-32 cache is a strict superset of Top-10.  Keep
            # the identical rerank prefix while materialising an honest
            # requested-Top-k cache/manifest so downstream filtering scores
            # only those eligible documents.
            subset_rows.append(
                {
                    **row,
                    "reranked_documents": list(row.get("reranked_documents") or [])[: args.rerank_top_k],
                }
            )
        preferred_cache_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(preferred_cache_dir / "candidates.jsonl", subset_rows)
        logging.info(
            "Materialized exact candidate subset: rows=%s source_top_k=%s requested_top_k=%s "
            "source=%s destination=%s",
            len(subset_rows),
            source_rerank_top_k,
            args.rerank_top_k,
            source_path,
            preferred_cache_dir / "candidates.jsonl",
        )
        return preferred_cache_dir, {str(row["key"]): row for row in subset_rows}
    return preferred_cache_dir, load_candidate_rows(preferred_path)


def ensure_candidates(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    query_vectors: np.ndarray,
    query_texts: list[str],
) -> tuple[list[list[RetrievedDocument]], list[list[RetrievedDocument]], Path]:
    preferred_cache_dir = candidate_cache_dir(args, samples, query_texts)
    cache_dir, existing = find_compatible_candidate_cache(
        args, samples, query_texts, preferred_cache_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "candidates.jsonl"
    missing_indices = [idx for idx, sample in enumerate(samples) if sample_key(sample) not in existing]
    logging.info("Candidate cache: ready=%s missing=%s path=%s", len(samples) - len(missing_indices), len(missing_indices), output_path)

    if missing_indices:
        retriever = FaissMedCPTRetriever(
            vector_db_root=args.vector_db_root,
            sources=args.sources,
            per_source_top_k=args.per_source_top_k,
            keep_indexes_in_memory=args.keep_faiss_indexes_in_memory,
            mmap_indexes=args.faiss_mmap,
            metadata_row_cache_size=args.metadata_row_cache_size,
            search_mode=args.retrieval_search_mode,
            shard_threaded=args.faiss_shard_threaded,
            gpu_search_chunk_size=args.gpu_search_chunk_size,
            gpu_search_device=args.gpu_search_device,
            gpu_search_dtype=args.gpu_search_dtype,
            faiss_gpu_device=args.faiss_gpu_device,
            faiss_gpu_use_float16=args.faiss_gpu_use_float16,
            faiss_gpu_add_batch_size=args.faiss_gpu_add_batch_size,
            faiss_gpu_temp_memory_mb=args.faiss_gpu_temp_memory_mb,
        )
        reranker: MedCPTCrossEncoderReranker | None = None
        progress = StageProgress(total=len(missing_indices), desc="CandidateBuild", enabled=True)
        rerank_progress = StageProgress(
            total=len(missing_indices) * args.candidate_pool_top_k,
            desc="RerankingPairs",
            enabled=True,
        )
        try:
            source_hits, bucket_sources, bucket_order = build_source_sequential_hits(
                retriever, query_vectors, missing_indices, args
            )
            reranker = MedCPTCrossEncoderReranker(
                model_path=args.cross_encoder_path,
                batch_size=args.rerank_batch_size,
                max_length=args.cross_encoder_max_length,
                attn_implementation=args.cross_encoder_attn_implementation,
            )
            mode = "w" if args.rebuild_candidates else "a"
            with output_path.open(mode, encoding="utf-8", buffering=16 * 1024 * 1024) as out:
                for offset in range(0, len(missing_indices), args.retrieval_batch_size):
                    end = min(offset + args.retrieval_batch_size, len(missing_indices))
                    hit_positions = list(range(offset, end))
                    batch_indices = missing_indices[offset:end]
                    initial_docs = materialize_initial_docs(
                        retriever=retriever,
                        source_hits=source_hits,
                        bucket_sources=bucket_sources,
                        bucket_order=bucket_order,
                        hit_positions=hit_positions,
                        top_k=args.candidate_pool_top_k,
                    )
                    rerank_queries = [
                        rerank_query_text(args, samples[idx], query_texts[idx]) for idx in batch_indices
                    ]
                    reranked_docs = reranker.rerank_batch(
                        rerank_queries,
                        initial_docs,
                        top_k=args.rerank_top_k,
                        progress_callback=rerank_progress.update,
                    )
                    for idx, docs, final_docs, rerank_query in zip(
                        batch_indices,
                        initial_docs,
                        reranked_docs,
                        rerank_queries,
                    ):
                        row = {
                            "key": sample_key(samples[idx]),
                            "dataset": samples[idx].dataset,
                            "sample_id": samples[idx].id,
                            "row_idx": samples[idx].row_idx,
                            "dense_query_mode": args.dense_query_mode,
                            "query_text": query_texts[idx],
                            "retrieval_query_text": query_texts[idx],
                            "rerank_query_text": rerank_query,
                            "initial_documents": [document_to_dict(doc, include_text=False) for doc in docs],
                            "reranked_documents": [document_to_dict(doc, include_text=True) for doc in final_docs],
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        existing[row["key"]] = row
                    progress.update(len(batch_indices))
        finally:
            progress.close()
            rerank_progress.close()
            if reranker is not None:
                reranker.close()
            retriever.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing_after = [sample_key(sample) for sample in samples if sample_key(sample) not in existing]
    if missing_after:
        raise RuntimeError(f"Candidate cache remains incomplete: missing={len(missing_after)} first={missing_after[:10]}")
    write_json(
        cache_dir / "manifest.json",
        {
            "type": "rag2_mcq_eval_candidates",
            "rows": len(samples),
            "sources": args.sources,
            "per_source_top_k": args.per_source_top_k,
            "candidate_pool_top_k": args.candidate_pool_top_k,
            "candidate_layout": args.candidate_layout,
            "pubmed_shards_per_group": args.pubmed_shards_per_group,
            "rerank_top_k": args.rerank_top_k,
            "dense_query_mode": args.dense_query_mode,
            "query": (
                "original MCQ question with all options"
                if args.dense_query_mode == "initial"
                else "no-RAG rationale_query including the generated answer conclusion"
            ),
            "prompt_profile": args.prompt_profile,
            "rationale_prompt_version": active_no_rag_prompt_version(args),
            "rerank_query": "original MCQ question with all options",
            "cross_encoder_path": str(args.cross_encoder_path),
            "vector_db_root": str(args.vector_db_root.resolve()),
            "candidate_query_fingerprint": candidate_query_fingerprint(samples, query_texts),
            "output_path": str(output_path),
        },
    )
    initial = [
        [document_from_dict(row) for row in existing[sample_key(sample)]["initial_documents"]]
        for sample in samples
    ]
    reranked = [
        [document_from_dict(row) for row in existing[sample_key(sample)]["reranked_documents"]]
        for sample in samples
    ]
    return initial, reranked, cache_dir


def extract_selected_option(text: str, sample: BenchmarkSample) -> str | None:
    valid = set(sample.options or {})
    for match in re.finditer(r"\b([A-Z])\b", str(text or "").upper()):
        if match.group(1) in valid:
            return match.group(1)
    return None


_PROMPT_TOKENIZER_CACHE: dict[str, Any] = {}


def build_document_messages_for_request(
    args: argparse.Namespace,
    sample: BenchmarkSample,
    document_rows: list[dict[str, Any]],
    *,
    max_doc_chars: int,
    format_retry: bool,
    selected_answer: str | None,
    choice_only: bool,
) -> list[dict[str, str]]:
    """Build the exact document-conditioned message list used for generation."""
    if args.prompt_profile == "paper_compatible_three_anchor":
        rendered_documents: list[str] = []
        for row in document_rows:
            # Match the anchored training traces: chunk body only, with blank
            # lines as neutral boundaries and no source/title/rank metadata.
            document_text = " ".join(str(row.get("text") or "").split())
            if max_doc_chars > 0 and len(document_text) > max_doc_chars:
                document_text = document_text[: max(0, max_doc_chars - 3)].rstrip() + "..."
            if document_text:
                rendered_documents.append(document_text)
        evidence = "\n\n".join(rendered_documents) or None
        return [{"role": "user", "content": build_anchored_user_prompt(sample.raw, evidence)}]
    if args.answer_decision_mode == "constrained_choice":
        rendered_documents: list[str] = []
        for row in document_rows:
            document_text = " ".join(str(row.get("text") or row.get("title") or "").split())
            if max_doc_chars > 0 and len(document_text) > max_doc_chars:
                document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
            if document_text:
                rendered_documents.append(document_text)
        context = "\n\n".join(rendered_documents) or None
        return [{"role": "user", "content": build_preanswer_user_prompt(sample, context)}]
    if choice_only:
        return (
            build_documents_choice_selection_messages(sample.raw, document_rows, max_doc_chars=max_doc_chars)
            if document_rows
            else build_choice_selection_messages(sample.raw)
        )
    if args.prompt_profile == "paper_answer_format":
        return (
            build_paper_answer_format_documents_messages(
                sample.raw,
                document_rows,
                max_doc_chars=max_doc_chars,
                format_retry=format_retry,
                selected_answer=selected_answer,
            )
            if document_rows
            else build_paper_answer_format_no_rag_messages(
                sample.raw,
                format_retry=format_retry,
                selected_answer=selected_answer,
            )
        )
    if args.prompt_profile == "paper_exact_terminal":
        return (
            build_paper_exact_terminal_documents_messages(
                sample.raw,
                document_rows,
                max_doc_chars=max_doc_chars,
            )
            if document_rows
            else build_paper_exact_terminal_no_rag_messages(sample.raw)
        )
    if args.prompt_profile == "paper_exact":
        return (
            build_paper_exact_documents_messages(
                sample.raw,
                document_rows,
                max_doc_chars=max_doc_chars,
                format_retry=format_retry,
                selected_answer=selected_answer,
            )
            if document_rows
            else build_paper_exact_no_rag_messages(
                sample.raw,
                format_retry=format_retry,
                selected_answer=selected_answer,
            )
        )
    return (
        build_documents_messages(
            sample.raw,
            document_rows,
            max_doc_chars=max_doc_chars,
            format_retry=format_retry,
            selected_answer=selected_answer,
        )
        if document_rows
        else build_no_rag_messages(
            sample.raw,
            format_retry=format_retry,
            selected_answer=selected_answer,
        )
    )


def prompt_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    key = str(args.llm_model_path.resolve())
    tokenizer = _PROMPT_TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        logging.info("Loading Llama tokenizer for dynamic document packing: %s", args.llm_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        _PROMPT_TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def rendered_chat_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    """Mirror the VLLMChatGenerator chat-template rendering for token budgeting."""
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return re.sub(r"(?is)(<\|im_start\|>assistant\s*)<think>\s*$", r"\1", str(rendered))


def rendered_token_count(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    assistant_prefill: str = "",
) -> int:
    rendered = rendered_chat_prompt(tokenizer, messages) + assistant_prefill
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def equal_token_allocation(capacities: list[int], budget: int) -> list[int]:
    """Water-fill a token budget across documents, preserving every document when possible."""
    allocations = [0] * len(capacities)
    remaining = max(0, int(budget))
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active: list[int] = []
        for position, index in enumerate(active):
            grant = min(share, capacities[index] - allocations[index], remaining)
            allocations[index] += grant
            remaining -= grant
            if allocations[index] < capacities[index]:
                next_active.append(index)
            if remaining <= 0:
                next_active.extend(
                    other for other in active[position + 1 :] if allocations[other] < capacities[other]
                )
                break
        active = next_active
    return allocations


def document_text_for_budget(document: dict[str, Any], *, include_title: bool = True) -> str:
    title = " ".join(str(document.get("title") or "").split())
    text = " ".join(str(document.get("text") or "").split())
    parts = (title, text) if include_title else (text,)
    return "\n".join(part for part in parts if part)


def dynamic_document_rows(
    args: argparse.Namespace,
    sample: BenchmarkSample,
    document_rows: list[dict[str, Any]],
    *,
    format_retry: bool,
    selected_answer: str | None,
    choice_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit all selected documents into one Llama context by token, not characters."""
    tokenizer = prompt_tokenizer(args)
    input_budget = int(args.llm_max_model_len) - int(args.max_new_tokens) - int(args.document_token_safety_margin)
    if input_budget <= 0:
        raise ValueError(
            "Dynamic document packing requires llm_max_model_len > max_new_tokens + document_token_safety_margin."
        )

    shell_rows: list[dict[str, Any]] = []
    token_ids: list[list[int]] = []
    # ``paper_exact`` intentionally exposes chunk body text only.  In
    # particular, do not let the token-packing path reintroduce a document
    # title after ``build_paper_exact_documents_messages`` removed metadata
    # from the final prompt.  The legacy prompt profiles retain their prior
    # title-plus-text budget behavior.
    include_title_in_budget = args.prompt_profile not in {
        "paper_exact",
        "paper_exact_terminal",
        "paper_compatible_three_anchor",
    }
    for row in document_rows:
        shell = dict(row)
        shell["title"] = ""
        shell["text"] = ""
        shell_rows.append(shell)
        token_ids.append(
            tokenizer(
                document_text_for_budget(row, include_title=include_title_in_budget),
                add_special_tokens=False,
            )["input_ids"]
        )

    shell_messages = build_document_messages_for_request(
        args,
        sample,
        shell_rows,
        max_doc_chars=0,
        format_retry=format_retry,
        selected_answer=selected_answer,
        choice_only=choice_only,
    )
    assistant_prefill = (
        FINAL_ANSWER_PREFILL
        if args.answer_decision_mode == "constrained_choice"
        else ANCHORED_RATIONALE_HEADER
        if args.prompt_profile == "paper_compatible_three_anchor"
        else ""
    )
    fixed_prompt_tokens = rendered_token_count(
        tokenizer,
        shell_messages,
        assistant_prefill=assistant_prefill,
    )
    document_budget = input_budget - fixed_prompt_tokens
    if document_budget < 0:
        raise RuntimeError(
            f"Question and document headers already exceed the input budget for {sample.id}: "
            f"fixed={fixed_prompt_tokens} budget={input_budget}."
        )

    allocations = equal_token_allocation([len(ids) for ids in token_ids], document_budget)

    def rows_for(current_allocations: list[int]) -> list[dict[str, Any]]:
        packed: list[dict[str, Any]] = []
        for row, ids, allocation in zip(document_rows, token_ids, current_allocations):
            packed_row = dict(row)
            packed_row["title"] = ""
            packed_row["text"] = tokenizer.decode(ids[:allocation], skip_special_tokens=True).strip()
            packed.append(packed_row)
        return packed

    packed_rows = rows_for(allocations)
    messages = build_document_messages_for_request(
        args,
        sample,
        packed_rows,
        max_doc_chars=0,
        format_retry=format_retry,
        selected_answer=selected_answer,
        choice_only=choice_only,
    )
    prompt_tokens = rendered_token_count(
        tokenizer,
        messages,
        assistant_prefill=assistant_prefill,
    )

    # Tokenizer decode/re-encode can differ slightly around whitespace.  Make
    # the final, rendered prompt satisfy the conservative budget exactly.
    for _ in range(16):
        overflow = prompt_tokens - input_budget
        if overflow <= 0:
            break
        adjustable = [index for index, value in enumerate(allocations) if value > 0]
        if not adjustable:
            raise RuntimeError(f"Unable to fit document prompt within the token budget for {sample.id}.")
        for index in sorted(adjustable, key=lambda item: allocations[item], reverse=True):
            if overflow <= 0:
                break
            reduction = min(allocations[index], max(1, overflow))
            allocations[index] -= reduction
            overflow -= reduction
        packed_rows = rows_for(allocations)
        messages = build_document_messages_for_request(
            args,
            sample,
            packed_rows,
            max_doc_chars=0,
            format_retry=format_retry,
            selected_answer=selected_answer,
            choice_only=choice_only,
        )
        prompt_tokens = rendered_token_count(
            tokenizer,
            messages,
            assistant_prefill=assistant_prefill,
        )
    else:
        raise RuntimeError(f"Dynamic document packing did not converge for {sample.id}.")

    return packed_rows, {
        "mode": "dynamic_token_budget",
        "input_token_budget": input_budget,
        "prompt_tokens": prompt_tokens,
        "document_text_token_budget": document_budget,
        "document_text_tokens": sum(allocations),
        "document_count": len(document_rows),
        "truncated_document_count": sum(allocation < len(ids) for allocation, ids in zip(allocations, token_ids)),
        "per_document_text_tokens": allocations,
    }


def prompt_request(
    args: argparse.Namespace,
    sample: BenchmarkSample,
    docs: list[RetrievedDocument],
    case_id: str,
    max_doc_chars: int,
    *,
    format_retry: bool = False,
    selected_answer: str | None = None,
    choice_only: bool = False,
) -> PromptRequest:
    doc_rows = [document_to_dict(doc, include_text=True) for doc in docs]
    packing: dict[str, Any] = {"mode": "fixed_chars", "max_doc_chars": max_doc_chars, "document_count": len(doc_rows)}
    if doc_rows and args.document_packing == "dynamic_token_budget":
        doc_rows, packing = dynamic_document_rows(
            args,
            sample,
            doc_rows,
            format_retry=format_retry,
            selected_answer=selected_answer,
            choice_only=choice_only,
        )
        max_doc_chars = 0
    messages = build_document_messages_for_request(
        args,
        sample,
        doc_rows,
        max_doc_chars=max_doc_chars,
        format_retry=format_retry,
        selected_answer=selected_answer,
        choice_only=choice_only,
    )
    metadata: dict[str, Any] = {"document_packing": packing}
    if not choice_only:
        if args.answer_decision_mode == "constrained_choice":
            # Hidden-state label extraction scores the four leading-space
            # choice tokens immediately after ``Final answer:``.  This
            # one-token grammar exposes that exact same decision space while
            # avoiding the current vLLM allowed_token_ids remapping bug.
            metadata["structured_regex"] = r" (A|B|C|D)"
        elif uses_free_terminal_generation(args):
            metadata["structured_regex"] = paper_exact_terminal_regex(
                normalized_options(sample.raw)
            )
    return PromptRequest(sample_id=sample.id, case_id=case_id, messages=messages, metadata=metadata)


def uses_free_terminal_generation(args: argparse.Namespace) -> bool:
    """Whether generation needs the rationale-plus-terminal repair contract.

    Constrained choice already appends ``Final answer:`` and emits exactly one
    allowed A/B/C/D token.  Applying the paper-terminal grammar or its repair
    path on top of that would turn a one-pass direct decision into a second,
    parser-mediated decision.
    """
    return (
        args.answer_decision_mode == "free_generation"
        and args.prompt_profile == "paper_exact_terminal"
    )


def build_generator(args: argparse.Namespace) -> VLLMChatGenerator:
    assistant_prefill = None
    if args.answer_decision_mode == "constrained_choice":
        assistant_prefill = FINAL_ANSWER_PREFILL
    elif args.prompt_profile == "paper_compatible_three_anchor":
        assistant_prefill = ANCHORED_RATIONALE_HEADER
    stop = ["<|im_end|>", "<|eot_id|>"]
    if args.prompt_profile == "paper_compatible_three_anchor":
        stop.extend(
            [
                ANCHORED_END_REASONING_MARKER,
                "\nFinal answer:",
                "\nTherefore, the answer",
            ]
        )
    return VLLMChatGenerator(
        model_path=args.llm_model_path,
        max_new_tokens=1 if args.answer_decision_mode == "constrained_choice" else args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop,
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
        assistant_prefill=assistant_prefill,
    )


def generate_rag_answers(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    context_docs: list[list[RetrievedDocument]],
    initial_docs: list[list[RetrievedDocument]],
    reranked_docs: list[list[RetrievedDocument]],
    query_texts: list[str],
    no_rag_by_key: dict[str, dict[str, Any]],
) -> tuple[list[CaseResult], dict[str, dict[str, Any]]]:
    generator = build_generator(args)
    results: list[CaseResult] = []
    details: dict[str, dict[str, Any]] = {}
    progress = StageProgress(total=len(samples), desc="Generation", enabled=True)
    try:
        for start in range(0, len(samples), args.generation_batch_size):
            end = min(start + args.generation_batch_size, len(samples))
            batch_samples = samples[start:end]
            batch_docs = context_docs[start:end]
            batch_initial = initial_docs[start:end]
            batch_reranked = reranked_docs[start:end]
            batch_queries = query_texts[start:end]
            records: list[tuple[CaseResult, dict[str, Any]] | None] = [None] * len(batch_samples)
            pending: list[tuple[int, BenchmarkSample, list[RetrievedDocument]]] = []
            for local_idx, (sample, docs, initial, reranked, query_text) in enumerate(
                zip(batch_samples, batch_docs, batch_initial, batch_reranked, batch_queries)
            ):
                # A constrained decision is a fresh forward pass even when no
                # document survived filtering.  Reusing the cached free-form
                # no-RAG answer here would mix two answer contracts inside one
                # sweep and reintroduce parser-dependent predictions.
                if docs or args.answer_decision_mode == "constrained_choice":
                    pending.append((local_idx, sample, docs))
                    continue
                artifact = no_rag_by_key[sample_key(sample)]
                parsed = artifact.get("parsed") or {}
                prediction = str(parsed.get("final_answer") or artifact.get("model_raw_generation") or "")
                request = prompt_request(args, sample, [], args.case, args.max_doc_chars)
                records[local_idx] = (
                    CaseResult(
                        case_id=args.case,
                        sample=sample,
                        prediction=prediction,
                        prompt=request.rendered,
                        initial_documents=initial,
                        reranked_documents=reranked,
                        final_documents=[],
                        evaluation=evaluate_prediction(sample, prediction),
                        raw_prediction=str(
                            artifact.get("canonical_generation")
                            or artifact.get("no_rag_generation")
                            or artifact.get("model_raw_generation")
                            or ""
                        ),
                    ),
                    {
                        "dense_query_mode": args.dense_query_mode,
                        "retrieval_query": query_text,
                        "rationale": parsed.get("rationale"),
                        "rationale_query": parsed.get("rationale_query"),
                        "final_answer": parsed.get("final_answer"),
                        "parse_errors": parsed.get("parse_errors") or [],
                        "generation_attempts": 0,
                        "context_document_count": 0,
                        "reused_cached_no_rag": True,
                        "document_packing": {"mode": "no_documents", "document_count": 0},
                    },
                )

            requests = [
                prompt_request(args, sample, docs, args.case, args.max_doc_chars)
                for _, sample, docs in pending
            ]
            outputs = generator.generate_batch(requests) if requests else []
            attempts = [1] * len(outputs)
            terminal_repair_sources: list[str | None] = [None] * len(outputs)
            terminal_primary_texts: list[str | None] = [None] * len(outputs)
            anchored_decisions: list[dict[str, Any] | None] = [None] * len(outputs)

            if args.prompt_profile == "paper_compatible_three_anchor":
                rationales: list[str] = []
                quality_flags: list[list[str]] = []
                decision_prefixes: list[str] = []
                for idx, output in enumerate(outputs):
                    raw_rationale = str(output.text or "").strip()
                    rationale, flags = normalize_anchored_rationale(raw_rationale)
                    if output.finish_reason == "length":
                        flags.append("rationale_length_exhausted")
                    rationales.append(rationale)
                    quality_flags.append(sorted(set(flags)))
                    terminal_primary_texts[idx] = raw_rationale
                    decision_prefixes.append(
                        f"{output.prompt}{rationale}\n"
                        f"{ANCHORED_END_REASONING_MARKER}\n"
                        "Final answer: ("
                    )
                choice_outputs = generator.generate_allowed_single_token_continuations(
                    decision_prefixes
                )
                if len(choice_outputs) != len(outputs):
                    raise RuntimeError(
                        "Anchored constrained-choice generation count mismatch: "
                        f"{len(choice_outputs)} != {len(outputs)}"
                    )
                for idx, (sample_tuple, primary, choice_output, rationale, flags) in enumerate(
                    zip(pending, outputs, choice_outputs, rationales, quality_flags)
                ):
                    sample = sample_tuple[1]
                    selected = str(choice_output.text or "").strip().upper()
                    if selected not in {"A", "B", "C", "D"}:
                        raise RuntimeError(
                            "Anchored constrained decoder returned no valid option for "
                            f"{sample_key(sample)}: {choice_output.text!r}"
                        )
                    options = normalized_options(sample.raw)
                    canonical = anchored_canonical_response(rationale, selected, options)
                    query_views = anchored_retrieval_queries(rationale, selected, options)
                    outputs[idx] = GenerationOutput(
                        text=canonical,
                        prompt=primary.prompt,
                        raw_text=primary.text,
                        finish_reason=primary.finish_reason,
                        stop_reason=primary.stop_reason,
                    )
                    anchored_decisions[idx] = {
                        "rationale": rationale,
                        "rationale_query": query_views["rationale_answer"],
                        "final_answer": selected,
                        "parse_errors": flags,
                    }
                    terminal_repair_sources[idx] = "anchored_constrained_single_token"
                    attempts[idx] += 1

            if uses_free_terminal_generation(args):
                needs_choice: list[int] = []
                for idx, ((_, sample, _), output) in enumerate(zip(pending, outputs)):
                    terminal_primary_texts[idx] = output.text
                    strict = parse_mcq_output_for_prompt_profile(
                        output.text,
                        normalized_options(sample.raw),
                        args.prompt_profile,
                    )
                    if not strict.parse_errors and strict.final_answer:
                        terminal_repair_sources[idx] = "structured_primary"
                        continue
                    recovered = parse_paper_exact_mcq_output(
                        output.text,
                        normalized_options(sample.raw),
                    )
                    if recovered.final_answer is not None:
                        canonical = append_paper_exact_terminal_answer(
                            output.text,
                            normalized_options(sample.raw),
                            recovered.final_answer,
                        )
                        outputs[idx] = GenerationOutput(
                            text=canonical,
                            prompt=output.prompt,
                            raw_text=output.text,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,
                        )
                        terminal_repair_sources[idx] = "canonicalized_primary_answer"
                    else:
                        needs_choice.append(idx)

                if needs_choice:
                    logging.info(
                        "Constrained one-token terminal fallback for %s/%s generated answer(s).",
                        len(needs_choice),
                        len(outputs),
                    )
                    prefixes = [
                        f"{outputs[idx].prompt}{outputs[idx].text.rstrip()}\nTherefore, the answer is ("
                        for idx in needs_choice
                    ]
                    selected_outputs = generator.generate_allowed_single_token_continuations(prefixes)
                    for idx, selected_output in zip(needs_choice, selected_outputs):
                        sample = pending[idx][1]
                        selected = extract_selected_option(selected_output.text, sample)
                        if selected is None:
                            raise RuntimeError(
                                "Constrained terminal fallback returned no valid option for "
                                f"{sample_key(sample)}: {selected_output.text!r}"
                            )
                        primary = outputs[idx]
                        canonical = append_paper_exact_terminal_answer(
                            primary.text,
                            normalized_options(sample.raw),
                            selected,
                        )
                        outputs[idx] = GenerationOutput(
                            text=canonical,
                            prompt=primary.prompt,
                            raw_text=primary.text,
                            finish_reason=primary.finish_reason,
                            stop_reason=primary.stop_reason,
                        )
                        terminal_repair_sources[idx] = "constrained_one_token_fallback"
                        attempts[idx] += 1

            format_retry_attempts = (
                0
                if args.answer_decision_mode == "constrained_choice"
                or args.prompt_profile in {
                    "paper_exact",
                    "paper_exact_terminal",
                    "paper_compatible_three_anchor",
                }
                else args.format_retry_attempts
            )
            for _ in range(format_retry_attempts):
                invalid = []
                for idx, ((_, sample, _), output) in enumerate(zip(pending, outputs)):
                    parsed = parse_mcq_output_for_prompt_profile(
                        output.text, normalized_options(sample.raw), args.prompt_profile
                    )
                    if parsed.parse_errors or not parsed.rationale_query or not parsed.final_answer:
                        invalid.append(idx)
                if not invalid:
                    break
                retry_outputs = generator.generate_batch(
                    [
                        prompt_request(
                            args,
                            pending[idx][1],
                            pending[idx][2],
                            args.case,
                            args.max_doc_chars,
                            format_retry=True,
                        )
                        for idx in invalid
                    ]
                )
                for idx, output in zip(invalid, retry_outputs):
                    outputs[idx] = output
                    attempts[idx] += 1

            unresolved = []
            if (
                args.answer_decision_mode == "free_generation"
                and args.prompt_profile != "paper_compatible_three_anchor"
            ):
                for idx, ((_, sample, _), output) in enumerate(zip(pending, outputs)):
                    parsed = parse_mcq_output_for_prompt_profile(
                        output.text, normalized_options(sample.raw), args.prompt_profile
                    )
                    if parsed.parse_errors or not parsed.rationale_query or not parsed.final_answer:
                        unresolved.append(idx)
            if unresolved and args.prompt_profile != "paper_exact":
                selected_outputs = generator.generate_batch(
                    [
                        prompt_request(
                            args,
                            pending[idx][1],
                            pending[idx][2],
                            args.case,
                            args.max_doc_chars,
                            choice_only=True,
                        )
                        for idx in unresolved
                    ]
                )
                anchored: list[tuple[int, str]] = []
                for idx, output in zip(unresolved, selected_outputs):
                    selected = extract_selected_option(output.text, pending[idx][1])
                    if selected is not None:
                        anchored.append((idx, selected))
                anchored_outputs = generator.generate_batch(
                    [
                        prompt_request(
                            args,
                            pending[idx][1],
                            pending[idx][2],
                            args.case,
                            args.max_doc_chars,
                            format_retry=True,
                            selected_answer=selected,
                        )
                        for idx, selected in anchored
                    ]
                )
                for (idx, _), output in zip(anchored, anchored_outputs):
                    outputs[idx] = output
                    attempts[idx] += 2

            for pending_idx, ((local_idx, sample, docs), output) in enumerate(zip(pending, outputs)):
                if args.prompt_profile == "paper_compatible_three_anchor":
                    decision = anchored_decisions[pending_idx]
                    if decision is None:
                        raise RuntimeError(
                            f"Missing anchored decision for {sample_key(sample)}"
                        )
                    final_answer = str(decision["final_answer"])
                    rationale = str(decision["rationale"])
                    rationale_query = str(decision["rationale_query"])
                    parse_errors = list(decision["parse_errors"])
                    prediction = final_answer
                elif args.answer_decision_mode == "constrained_choice":
                    final_answer = extract_selected_option(output.text, sample)
                    if final_answer is None:
                        raise RuntimeError(
                            f"Constrained decoder returned no valid option for {sample_key(sample)}: "
                            f"{output.text!r}"
                        )
                    rationale = None
                    parse_errors: list[str] = []
                    prediction = final_answer
                    artifact_parsed = (no_rag_by_key[sample_key(sample)].get("parsed") or {})
                    rationale_query = artifact_parsed.get("rationale_query")
                else:
                    parsed = parse_mcq_output_for_prompt_profile(
                        output.text, normalized_options(sample.raw), args.prompt_profile
                    )
                    final_answer = parsed.final_answer
                    rationale = parsed.rationale
                    rationale_query = parsed.rationale_query
                    parse_errors = parsed.parse_errors
                    prediction = final_answer or output.text
                records[local_idx] = (
                    CaseResult(
                        case_id=args.case,
                        sample=sample,
                        prediction=prediction,
                        prompt=output.prompt,
                        initial_documents=batch_initial[local_idx],
                        reranked_documents=batch_reranked[local_idx],
                        final_documents=docs,
                        evaluation=evaluate_prediction(sample, prediction),
                        raw_prediction=output.text,
                    ),
                    {
                        "dense_query_mode": args.dense_query_mode,
                        "retrieval_query": batch_queries[local_idx],
                        "rationale": rationale,
                        "rationale_query": rationale_query,
                        "final_answer": final_answer,
                        "parse_errors": parse_errors,
                        "answer_decision_mode": args.answer_decision_mode,
                        "terminal_repair_source": terminal_repair_sources[pending_idx],
                        "terminal_primary_generation": terminal_primary_texts[pending_idx],
                        "answer_prompt_version": (
                            PREANSWER_PROMPT_VERSION
                            if args.answer_decision_mode == "constrained_choice"
                            else active_document_prompt_version(args)
                        ),
                        "generation_attempts": attempts[pending_idx],
                        "context_document_count": len(docs),
                        "reused_cached_no_rag": False,
                        "document_packing": requests[pending_idx].metadata.get("document_packing"),
                    },
                )
            for record in records:
                if record is None:
                    raise RuntimeError("Internal error: missing evaluation record after generation.")
                result, detail = record
                results.append(result)
                details[sample_key(result.sample)] = detail
            progress.update(len(batch_samples))
    finally:
        progress.close()
        generator.close()
    return results, details


def no_rag_results(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    artifacts: dict[str, list[dict[str, Any]]],
) -> tuple[list[CaseResult], dict[str, dict[str, Any]]]:
    counters = {dataset: 0 for dataset in artifacts}
    results: list[CaseResult] = []
    details: dict[str, dict[str, Any]] = {}
    for sample in samples:
        idx = counters[sample.dataset]
        counters[sample.dataset] += 1
        artifact = artifacts[sample.dataset][idx]
        parsed = artifact.get("parsed") or {}
        prediction = str(parsed.get("final_answer") or artifact.get("model_raw_generation") or "")
        messages = (
            [{"role": "user", "content": build_anchored_user_prompt(sample.raw, None)}]
            if args.prompt_profile == "paper_compatible_three_anchor"
            else build_paper_answer_format_no_rag_messages(sample.raw)
            if args.prompt_profile == "paper_answer_format"
            else (
                build_paper_exact_terminal_no_rag_messages(sample.raw)
                if args.prompt_profile == "paper_exact_terminal"
                else (
                    build_paper_exact_no_rag_messages(sample.raw)
                    if args.prompt_profile == "paper_exact"
                    else build_no_rag_messages(sample.raw)
                )
            )
        )
        request = PromptRequest(sample_id=sample.id, case_id="no_rag", messages=messages)
        results.append(
            CaseResult(
                case_id="no_rag",
                sample=sample,
                prediction=prediction,
                prompt=request.rendered,
                initial_documents=[],
                reranked_documents=[],
                final_documents=[],
                evaluation=evaluate_prediction(sample, prediction),
                raw_prediction=str(
                    artifact.get("canonical_generation")
                    or artifact.get("no_rag_generation")
                    or artifact.get("model_raw_generation")
                    or ""
                ),
            )
        )
        details[sample_key(sample)] = {
            "dense_query_mode": None,
            "retrieval_query": parsed.get("rationale_query"),
            "rationale": parsed.get("rationale"),
            "rationale_query": parsed.get("rationale_query"),
            "final_answer": parsed.get("final_answer"),
            "parse_errors": parsed.get("parse_errors") or [],
            "parser_recovery": parsed.get("parser_recovery"),
            "generation_attempts": artifact.get("generation_attempts"),
            "context_document_count": 0,
            "rationale_ppl": ((artifact.get("generation_stats") or {}).get("rationale") or {}).get("ppl"),
        }
    return results, details


def align_no_rag_artifacts(
    samples: list[BenchmarkSample],
    artifacts: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Associate each combined evaluation sample with its cached no-RAG generation."""
    counters = {dataset: 0 for dataset in artifacts}
    aligned: dict[str, dict[str, Any]] = {}
    for sample in samples:
        index = counters[sample.dataset]
        counters[sample.dataset] += 1
        try:
            artifact = artifacts[sample.dataset][index]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Missing no-RAG artifact for {sample_key(sample)}") from exc
        aligned[sample_key(sample)] = artifact
    return aligned


def resolve_filter_window_thresholds(args: argparse.Namespace) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Load per-route document thresholds for multiple-instance window scores."""

    default = float(args.filter_window_helpful_threshold)
    if not 0.0 <= default <= 1.0:
        raise ValueError("--filter-window-helpful-threshold must be in [0, 1].")
    thresholds = {"medmcqa": default, "medqa": default}
    if args.filter_window_thresholds_path is None:
        return thresholds, None
    path = args.filter_window_thresholds_path
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed window-threshold JSON: {path}") from exc
    values = artifact.get("thresholds", artifact) if isinstance(artifact, dict) else {}
    if not isinstance(values, dict):
        raise ValueError(f"Window-threshold artifact has no thresholds mapping: {path}")
    for route in thresholds:
        value = values.get(route)
        if isinstance(value, dict):
            value = value.get("threshold")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Window-threshold artifact lacks a numeric {route} threshold: {path}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Window threshold outside [0, 1] for {route}: {threshold}")
        thresholds[route] = threshold
    return thresholds, artifact if isinstance(artifact, dict) else None


def resolve_document_transformer_thresholds(args: argparse.Namespace) -> dict[str, float]:
    """Resolve validation-selected thresholds independently for each dataset route."""

    fallback = float(args.document_transformer_helpful_threshold)
    values = {
        "medmcqa": args.medmcqa_document_transformer_helpful_threshold,
        "medqa": args.medqa_document_transformer_helpful_threshold,
    }
    thresholds: dict[str, float] = {}
    for route, value in values.items():
        threshold = fallback if value is None else float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Document Transformer threshold outside [0, 1] for {route}: {threshold}")
        thresholds[route] = threshold
    return thresholds


def _filter_routes() -> dict[str, str]:
    return {"medqa": "medqa", "medmcqa": "medmcqa", **{name: "medmcqa" for name in MMLU_DATASETS}}


def _filter_evidence_from_document(document: RetrievedDocument, max_doc_chars: int) -> str:
    """Mirror the original filter-data evidence fallback: body text, then title."""

    evidence = " ".join(str(document.text or document.title or "").split())
    if max_doc_chars > 0 and len(evidence) > max_doc_chars:
        return evidence[: max_doc_chars - 3].rstrip() + "..."
    return evidence


def _set_window_document_decision(
    document: RetrievedDocument,
    windows: list[dict[str, Any]],
    scores: list[dict[str, float | str]],
    *,
    threshold: float,
    context_sentences: int,
) -> None:
    """Aggregate window predictions with max P(Helpful) into one document decision."""

    decisions: list[dict[str, Any]] = []
    for window, score in zip(windows, scores):
        decisions.append(
            {
                "window_id": window["window_id"],
                "centre_sentence_id": window["centre_sentence_id"],
                "sentence_ids": window["sentence_ids"],
                "sentence_count": window["sentence_count"],
                "char_start": window["char_start"],
                "char_end": window["char_end"],
                "window_sha256": window["sha256"],
                "prediction": str(score["prediction"]),
                "score_helpful": float(score["score_helpful"]),
                "score_not_helpful": float(score["score_not_helpful"]),
                "margin_helpful_minus_not_helpful": float(score["margin"]),
                "prob_helpful_over_candidates": float(score["prob_helpful"]),
            }
        )
    if decisions:
        # Stable tie-breaks make resumed caches and analyses deterministic.
        best_index = max(
            range(len(decisions)),
            key=lambda index: (
                float(decisions[index]["prob_helpful_over_candidates"]),
                float(decisions[index]["margin_helpful_minus_not_helpful"]),
                -index,
            ),
        )
        best = decisions[best_index]
        probability = float(best["prob_helpful_over_candidates"])
        margin = float(best["margin_helpful_minus_not_helpful"])
    else:
        best_index = None
        best = None
        probability = 0.0
        margin = float("-inf")
    document.filter_prob_helpful = probability
    document.filter_score = margin
    document.filter_prediction = "helpful" if probability >= threshold else "not helpful"
    document.metadata["window_filter"] = {
        "aggregation": "max_probability",
        "threshold": threshold,
        "windowing": windowing_contract(context_sentences),
        "scored_window_count": len(decisions),
        "argmax_window_index": best_index,
        "argmax_window_id": best.get("window_id") if best else None,
        "argmax_window_probability": probability if best else None,
        "raw_helpful_window_count": sum(item["prediction"] == "helpful" for item in decisions),
        "window_decisions": decisions,
    }


def _best_helpful_window_context_document(
    args: argparse.Namespace,
    document: RetrievedDocument,
) -> RetrievedDocument:
    """Replace one eligible document's visible context by its strongest Helpful window.

    This deliberately happens *after* prefix-k eligibility has been fixed at
    the document level. The returned object keeps the same retrieval/rerank
    identity and scores, so whole-document and window-context generations can
    be compared with exactly the same candidate and filter decisions.
    """

    if args.filter_evidence_unit != "sentence_window":
        raise ValueError(
            "--filter-generation-context-unit=best_helpful_window requires "
            "--filter-evidence-unit=sentence_window."
        )
    if document.filter_prediction != "helpful":
        raise ValueError("Only filter-eligible documents can be materialized as Helpful windows.")
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    window_filter = metadata.get("window_filter")
    if not isinstance(window_filter, dict):
        raise ValueError(
            f"Missing cached window-filter metadata for selected document: {document.stable_id}"
        )
    decisions = window_filter.get("window_decisions")
    if not isinstance(decisions, list):
        raise ValueError(
            f"Missing cached per-window decisions for selected document: {document.stable_id}"
        )
    helpful_decisions = [
        item
        for item in decisions
        if isinstance(item, dict) and str(item.get("prediction") or "") == "helpful"
    ]
    if not helpful_decisions:
        raise ValueError(
            "A document passed the window filter without any raw Helpful window; "
            "best_helpful_window requires a raw Helpful window. "
            f"document={document.stable_id} threshold={window_filter.get('threshold')}"
        )
    selected = max(
        helpful_decisions,
        key=lambda item: (
            float(item.get("prob_helpful_over_candidates") or float("-inf")),
            float(item.get("margin_helpful_minus_not_helpful") or float("-inf")),
            str(item.get("window_id") or ""),
        ),
    )
    selected_window_id = str(selected.get("window_id") or "")
    evidence = _filter_evidence_from_document(document, args.filter_max_doc_chars)
    windows = sentence_context_windows(
        evidence,
        context_sentences=args.filter_window_context_sentences,
    )
    window_by_id = {str(window["window_id"]): window for window in windows}
    window = window_by_id.get(selected_window_id)
    if window is None:
        raise ValueError(
            "Could not deterministically reconstruct selected Helpful window from cached metadata: "
            f"document={document.stable_id} window={selected_window_id}"
        )
    expected_sha = str(selected.get("window_sha256") or "")
    if expected_sha and expected_sha != str(window["sha256"]):
        raise ValueError(
            "Selected Helpful window changed since filter scoring; refusing an uncontrolled context comparison: "
            f"document={document.stable_id} window={selected_window_id}"
        )

    contextual_document = copy.copy(document)
    # The paper-exact prompt receives only the selected medical evidence, not
    # a chunk title or corpus label that was absent from filter training.
    contextual_document.title = None
    contextual_document.text = str(window["text"])
    contextual_document.metadata = dict(metadata)
    contextual_document.metadata["generation_context"] = {
        "unit": "best_helpful_window",
        "parent_document_stable_id": document.stable_id,
        "window_id": selected_window_id,
        "centre_sentence_id": window["centre_sentence_id"],
        "sentence_ids": list(window["sentence_ids"]),
        "sentence_count": int(window["sentence_count"]),
        "char_start": int(window["char_start"]),
        "char_end": int(window["char_end"]),
        "window_sha256": str(window["sha256"]),
        "prediction": str(selected.get("prediction") or "helpful"),
        "prob_helpful_over_candidates": float(selected["prob_helpful_over_candidates"]),
        "margin_helpful_minus_not_helpful": float(selected["margin_helpful_minus_not_helpful"]),
    }
    return contextual_document


def materialize_filter_generation_context(
    args: argparse.Namespace,
    documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    """Keep document-level selection fixed and materialize the requested LLM context."""

    selected = [copy.copy(document) for document in documents if document.filter_prediction == "helpful"]
    if args.filter_generation_context_unit == "document":
        return selected
    return [_best_helpful_window_context_document(args, document) for document in selected]


def _score_filter_document_windows(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
) -> list[list[RetrievedDocument]]:
    """Score every centred sentence window while retaining bounded host memory.

    Loading every window for 6k MCQs at once would unnecessarily duplicate
    hundreds of MB of text.  One dataset-routed Flan model remains resident,
    while a modest question batch is materialised and scored at a time.
    """

    routes = _filter_routes()
    thresholds, _ = resolve_filter_window_thresholds(args)
    model_paths = {"medmcqa": args.medmcqa_filter_model_path, "medqa": args.medqa_filter_model_path}
    progress = StageProgress(
        total=sum(len(docs) for docs in reranked_docs),
        desc="FilteringSentenceWindows",
        enabled=True,
    )
    try:
        for route in ("medqa", "medmcqa"):
            indices = [index for index, sample in enumerate(samples) if routes[sample.dataset] == route]
            if not indices:
                continue
            logging.info(
                "Window filtering route=%s datasets=%s questions=%s model=%s threshold=%.6f",
                route,
                sorted({samples[index].dataset for index in indices}),
                len(indices),
                model_paths[route],
                thresholds[route],
            )
            filterer = Rag2FlanT5Filter(
                model_path=model_paths[route],
                batch_size=args.filter_batch_size,
                max_input_length=args.filter_max_input_length,
                max_new_tokens=args.filter_max_new_tokens,
                # Windows are already derived from the evidence passed to the
                # filter.  Never truncate an individual window a second time.
                max_doc_chars=0,
                device=args.filter_device,
                bf16=args.filter_bf16,
                scoring_method=args.filter_scoring_method,
                score_normalization=args.filter_score_normalization,
                input_format=args.filter_input_format,
            )
            try:
                for batch_start in range(0, len(indices), args.filter_window_question_batch_size):
                    batch_indices = indices[batch_start : batch_start + args.filter_window_question_batch_size]
                    flat_samples: list[BenchmarkSample] = []
                    flat_evidences: list[str] = []
                    assignments: list[tuple[RetrievedDocument, dict[str, Any]]] = []
                    document_windows: list[tuple[RetrievedDocument, list[dict[str, Any]]]] = []
                    for sample_index in batch_indices:
                        sample = samples[sample_index]
                        for document in reranked_docs[sample_index]:
                            evidence = _filter_evidence_from_document(document, args.filter_max_doc_chars)
                            windows = sentence_context_windows(
                                evidence,
                                context_sentences=args.filter_window_context_sentences,
                            )
                            document_windows.append((document, windows))
                            for window in windows:
                                flat_samples.append(sample)
                                flat_evidences.append(str(window["text"]))
                                assignments.append((document, window))
                    scores = filterer.score_evidences(flat_samples, flat_evidences) if flat_samples else []
                    by_document: dict[int, list[dict[str, float | str]]] = {}
                    for (document, _), score in zip(assignments, scores):
                        by_document.setdefault(id(document), []).append(score)
                    for document, windows in document_windows:
                        _set_window_document_decision(
                            document,
                            windows,
                            by_document.get(id(document), []),
                            threshold=thresholds[route],
                            context_sentences=args.filter_window_context_sentences,
                        )
                    progress.update(sum(len(reranked_docs[index]) for index in batch_indices))
            finally:
                filterer.close()
        return reranked_docs
    finally:
        progress.close()


def _score_filter_document_transformer(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
) -> list[list[RetrievedDocument]]:
    """Score all windows once, then infer each original document with the learned sequence model."""

    routes = _filter_routes()
    window_model_paths = {
        "medmcqa": args.medmcqa_filter_model_path,
        "medqa": args.medqa_filter_model_path,
    }
    document_checkpoint_paths = {
        "medmcqa": args.medmcqa_document_transformer_checkpoint,
        "medqa": args.medqa_document_transformer_checkpoint,
    }
    document_thresholds = resolve_document_transformer_thresholds(args)
    progress = StageProgress(
        total=sum(len(docs) for docs in reranked_docs),
        desc="FilteringDocumentTransformer",
        enabled=True,
    )
    try:
        for route in ("medqa", "medmcqa"):
            indices = [index for index, sample in enumerate(samples) if routes[sample.dataset] == route]
            if not indices:
                continue
            checkpoint_path = document_checkpoint_paths[route]
            if checkpoint_path is None:
                raise ValueError(f"Missing {route} Document Transformer checkpoint")
            logging.info(
                "Hierarchical filtering route=%s datasets=%s questions=%s window_model=%s "
                "document_checkpoint=%s threshold=%.6f",
                route,
                sorted({samples[index].dataset for index in indices}),
                len(indices),
                window_model_paths[route],
                checkpoint_path,
                document_thresholds[route],
            )
            filterer = HierarchicalRag2DocumentFilter(
                window_model_path=window_model_paths[route],
                document_checkpoint_path=checkpoint_path,
                max_input_length=args.filter_max_input_length,
                window_batch_size=args.filter_batch_size,
                document_batch_size=args.document_transformer_batch_size,
                context_sentences=args.filter_window_context_sentences,
                document_threshold=document_thresholds[route],
                device=args.filter_device,
                bf16=args.filter_bf16,
            )
            try:
                for batch_start in range(0, len(indices), args.filter_window_question_batch_size):
                    batch_indices = indices[batch_start : batch_start + args.filter_window_question_batch_size]
                    filterer.score_question_batch(
                        [samples[index] for index in batch_indices],
                        [reranked_docs[index] for index in batch_indices],
                        max_doc_chars=args.filter_max_doc_chars,
                    )
                    progress.update(sum(len(reranked_docs[index]) for index in batch_indices))
            finally:
                filterer.close()
        return reranked_docs
    finally:
        progress.close()


def _score_filter_preanswer_text_hidden(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
) -> list[list[RetrievedDocument]]:
    """Extract h0/hD without gold and score documents with text+hidden filters.

    One route-specific filter is loaded at a time.  h0 is computed once per
    question and hD once per cached reranked document.  This function is used
    during ``--filter-cache-only``; all later Top-k answer-generation runs
    reuse the persisted decisions and do not repeat hidden extraction.
    """

    routes = _filter_routes()
    model_paths = {
        "medmcqa": args.medmcqa_filter_model_path,
        "medqa": args.medqa_filter_model_path,
    }
    progress = StageProgress(
        total=sum(len(docs) for docs in reranked_docs),
        desc="FilteringPreAnswerTextHidden",
        enabled=True,
    )
    try:
        for route in ("medqa", "medmcqa"):
            indices = [index for index, sample in enumerate(samples) if routes[sample.dataset] == route]
            if not indices:
                continue
            logging.info(
                "Pre-answer text+hidden filtering route=%s datasets=%s questions=%s documents=%s "
                "checkpoint=%s state_model=%s layer=%s threshold=%.6f",
                route,
                sorted({samples[index].dataset for index in indices}),
                len(indices),
                sum(len(reranked_docs[index]) for index in indices),
                model_paths[route],
                args.llm_model_path,
                args.hidden_feature_layer,
                args.hidden_filter_helpful_threshold,
            )
            filterer = TextHiddenRag2Filter(
                checkpoint_path=model_paths[route],
                backbone_path=args.hidden_filter_backbone_path,
                state_model_path=args.llm_model_path,
                layer=args.hidden_feature_layer,
                hidden_batch_size=args.hidden_feature_batch_size,
                filter_batch_size=args.filter_batch_size,
                max_hidden_input_tokens=args.hidden_feature_max_input_tokens,
                max_filter_input_length=args.filter_max_input_length,
                max_doc_chars=args.filter_max_doc_chars,
                helpful_threshold=args.hidden_filter_helpful_threshold,
                device=args.filter_device,
                bf16=args.filter_bf16,
                hidden_dtype=args.hidden_feature_dtype,
                hidden_attn_implementation=args.hidden_feature_attn_implementation,
            )
            try:
                filterer.score_documents(
                    [samples[index] for index in indices],
                    [reranked_docs[index] for index in indices],
                    progress_callback=progress.update,
                )
            finally:
                filterer.close()
        return reranked_docs
    finally:
        progress.close()


def score_filter_documents(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
) -> list[list[RetrievedDocument]]:
    """Score every cached reranked document exactly once.

    The returned lists retain all documents in MedCPT rerank order.  Prefix-k
    eligibility and Helpful-only selection happen later so k means the
    reranking cutoff, never a post-filter document-count target.
    """
    if args.filter_evidence_unit == "preanswer_text_hidden":
        scored = _score_filter_preanswer_text_hidden(args, samples, reranked_docs)
    elif args.filter_evidence_unit == "sentence_window":
        scored = _score_filter_document_windows(args, samples, reranked_docs)
    elif args.filter_evidence_unit == "document_transformer":
        scored = _score_filter_document_transformer(args, samples, reranked_docs)
    else:
        routes = _filter_routes()
        filterer = DatasetRoutedRag2Filter(
            model_paths={"medmcqa": args.medmcqa_filter_model_path, "medqa": args.medqa_filter_model_path},
            dataset_routes=routes,
            batch_size=args.filter_batch_size,
            max_input_length=args.filter_max_input_length,
            max_new_tokens=args.filter_max_new_tokens,
            max_doc_chars=args.filter_max_doc_chars,
            device=args.filter_device,
            bf16=args.filter_bf16,
            scoring_method=args.filter_scoring_method,
            score_normalization=args.filter_score_normalization,
            input_format=args.filter_input_format,
        )
        progress = StageProgress(
            total=sum(len(docs) for docs in reranked_docs),
            desc="FilteringPairs",
            enabled=True,
        )
        try:
            # Rag2FlanT5Filter records every decision on the original candidate
            # objects before returning its Helpful-only view.  We intentionally
            # ignore that view and preserve all scored candidates for Top-k sweeps.
            filterer.filter_batch(
                samples=samples,
                candidate_lists=reranked_docs,
                top_k=args.rerank_top_k,
                fill_to_top_k=False,
                progress_callback=progress.update,
            )
            scored = reranked_docs
        finally:
            progress.close()
            filterer.close()
    for docs in scored:
        helpful_rank = 0
        for doc in docs:
            if doc.filter_prediction == "helpful":
                helpful_rank += 1
                doc.filter_rank = helpful_rank
            else:
                doc.filter_rank = None
    return scored


def _path_cache_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "file": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        }
    files = []
    for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "rag2_hidden_filter_architecture.json",
    ):
        candidate = resolved / name
        if candidate.exists():
            stat = candidate.stat()
            files.append({"name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"path": str(resolved), "files": files}


def filter_cache_settings(args: argparse.Namespace, candidate_dir: Path) -> dict[str, Any]:
    candidate_path = candidate_dir / "candidates.jsonl"

    def stat_or_none(path: Path) -> dict[str, int] | None:
        if not path.exists():
            return None
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    thresholds, threshold_artifact = (
        resolve_filter_window_thresholds(args)
        if args.filter_evidence_unit == "sentence_window"
        else ({"medmcqa": None, "medqa": None}, None)
    )
    document_checkpoints = {
        "medmcqa": (
            _path_cache_identity(args.medmcqa_document_transformer_checkpoint)
            if args.medmcqa_document_transformer_checkpoint is not None
            else None
        ),
        "medqa": (
            _path_cache_identity(args.medqa_document_transformer_checkpoint)
            if args.medqa_document_transformer_checkpoint is not None
            else None
        ),
    }
    document_thresholds = resolve_document_transformer_thresholds(args)
    return {
        "candidate_cache_dir": str(candidate_dir.resolve()),
        "candidate_file": stat_or_none(candidate_path),
        "rerank_top_k_scored": args.rerank_top_k,
        "routes": {"medqa": "medqa", "medmcqa_and_mmlu": "medmcqa"},
        "models": {
            "medmcqa": _path_cache_identity(args.medmcqa_filter_model_path),
            "medqa": _path_cache_identity(args.medqa_filter_model_path),
        },
        "document_transformer_checkpoints": (
            document_checkpoints if args.filter_evidence_unit == "document_transformer" else None
        ),
        "document_transformer_helpful_thresholds": (
            document_thresholds
            if args.filter_evidence_unit == "document_transformer"
            else None
        ),
        "document_transformer_batch_size": (
            args.document_transformer_batch_size
            if args.filter_evidence_unit == "document_transformer"
            else None
        ),
        "filter_batch_size": args.filter_batch_size,
        "filter_max_input_length": args.filter_max_input_length,
        "filter_max_new_tokens": args.filter_max_new_tokens,
        "filter_max_doc_chars": args.filter_max_doc_chars,
        "filter_bf16": args.filter_bf16,
        "filter_scoring_method": args.filter_scoring_method,
        "filter_score_normalization": args.filter_score_normalization,
        "filter_input_format": args.filter_input_format,
        "filter_evidence_unit": args.filter_evidence_unit,
        "filter_window_context_sentences": args.filter_window_context_sentences,
        "filter_window_question_batch_size": args.filter_window_question_batch_size,
        "filter_window_aggregation": (
            "max_probability"
            if args.filter_evidence_unit == "sentence_window"
            else "document_transformer"
            if args.filter_evidence_unit == "document_transformer"
            else None
        ),
        "filter_window_thresholds": thresholds if args.filter_evidence_unit == "sentence_window" else None,
        "filter_window_thresholds_path": (
            str(args.filter_window_thresholds_path.resolve())
            if args.filter_evidence_unit == "sentence_window" and args.filter_window_thresholds_path
            else None
        ),
        "filter_window_threshold_artifact_created_at": (
            threshold_artifact.get("created_at")
            if args.filter_evidence_unit == "sentence_window" and threshold_artifact
            else None
        ),
        "filter_windowing_contract": (
            windowing_contract(args.filter_window_context_sentences)
            if args.filter_evidence_unit in {"sentence_window", "document_transformer"}
            else None
        ),
        "preanswer_text_hidden": (
            {
                "prompt_version": PREANSWER_PROMPT_VERSION,
                "backbone": _path_cache_identity(args.hidden_filter_backbone_path),
                "state_model": _path_cache_identity(args.llm_model_path),
                "hidden_layer": args.hidden_feature_layer,
                "hidden_batch_size": args.hidden_feature_batch_size,
                "question_commit_batch_size": args.hidden_filter_question_batch_size,
                "hidden_max_input_tokens": args.hidden_feature_max_input_tokens,
                "hidden_dtype": args.hidden_feature_dtype,
                "hidden_attn_implementation": args.hidden_feature_attn_implementation,
                "helpful_threshold": args.hidden_filter_helpful_threshold,
                "hidden_inputs": ["h0", "delta_h=hD-h0"],
                "forbidden_inputs": ["gold_answer", "c", "projection_score", "answer_transition"],
            }
            if args.filter_evidence_unit == "preanswer_text_hidden"
            else None
        ),
    }


def filter_score_cache_dir(args: argparse.Namespace, candidate_dir: Path) -> tuple[Path, dict[str, Any], str]:
    settings = filter_cache_settings(args, candidate_dir)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    return candidate_dir / "filter_scores" / fingerprint[:16], settings, fingerprint


PREANSWER_HIDDEN_FEATURE_CACHE_VERSION = "rag2_preanswer_hidden_states_v1"


def preanswer_hidden_feature_cache_settings(
    args: argparse.Namespace,
    candidate_dir: Path,
) -> dict[str, Any]:
    """Return only inputs that can change h0/hD, never filter-model settings."""

    candidate_path = candidate_dir / "candidates.jsonl"
    candidate_stat = candidate_path.stat()
    state_model = args.llm_model_path.resolve()
    state_model_identity = _path_cache_identity(state_model)
    if state_model.is_dir():
        state_model_identity["weight_shards"] = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(state_model.glob("*.safetensors"))
        ]
    return {
        "version": PREANSWER_HIDDEN_FEATURE_CACHE_VERSION,
        "candidate_cache_dir": str(candidate_dir.resolve()),
        "candidate_file": {
            "size": candidate_stat.st_size,
            "mtime_ns": candidate_stat.st_mtime_ns,
        },
        "state_model": state_model_identity,
        "prompt_version": PREANSWER_PROMPT_VERSION,
        "hidden_layer": args.hidden_feature_layer,
        "hidden_max_input_tokens": args.hidden_feature_max_input_tokens,
        "hidden_dtype": args.hidden_feature_dtype,
        "hidden_attn_implementation": args.hidden_feature_attn_implementation,
        "max_doc_chars": args.filter_max_doc_chars,
        # Sharding does not change values, but fixing it in the contract makes
        # interrupted-run validation and deterministic shard names trivial.
        "question_shard_size": args.hidden_filter_question_batch_size,
        "routes": {"medqa": "medqa", "medmcqa_and_mmlu": "medmcqa"},
        "stored_tensors": ["h0", "hD", "document_offsets"],
        "forbidden_inputs": ["gold_answer", "c", "projection_score", "answer_transition"],
    }


def preanswer_hidden_feature_cache_dir(
    args: argparse.Namespace,
    candidate_dir: Path,
) -> tuple[Path, dict[str, Any], str]:
    settings = preanswer_hidden_feature_cache_settings(args, candidate_dir)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    return candidate_dir / "preanswer_hidden_features" / fingerprint[:16], settings, fingerprint


def _preanswer_feature_batch_specs(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
) -> list[dict[str, Any]]:
    routes = _filter_routes()
    specs: list[dict[str, Any]] = []
    shard_size = max(1, int(args.hidden_filter_question_batch_size))
    for route in ("medqa", "medmcqa"):
        route_indices = [
            index for index, sample in enumerate(samples) if routes[sample.dataset] == route
        ]
        for start in range(0, len(route_indices), shard_size):
            indices = route_indices[start : start + shard_size]
            specs.append(
                {
                    "name": f"{route}_{start:07d}",
                    "route": route,
                    "indices": indices,
                    "questions": len(indices),
                    "documents": sum(len(reranked_docs[index]) for index in indices),
                }
            )
    return specs


def _preanswer_feature_expected_metadata(
    spec: dict[str, Any],
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
    fingerprint: str,
) -> dict[str, Any]:
    indices = list(spec["indices"])
    return {
        "type": "rag2_mcq_eval_preanswer_hidden_feature_shard",
        "version": PREANSWER_HIDDEN_FEATURE_CACHE_VERSION,
        "settings_fingerprint": fingerprint,
        "name": spec["name"],
        "route": spec["route"],
        "sample_indices": indices,
        "sample_keys": [sample_key(samples[index]) for index in indices],
        "document_keys": [
            [[document.db_id, document.local_id] for document in reranked_docs[index]]
            for index in indices
        ],
        "questions": int(spec["questions"]),
        "documents": int(spec["documents"]),
    }


def _preanswer_feature_shard_paths(cache_dir: Path, name: str) -> tuple[Path, Path]:
    return cache_dir / "shards" / f"{name}.safetensors", cache_dir / "shards" / f"{name}.json"


def _valid_preanswer_feature_shard(
    cache_dir: Path,
    expected: dict[str, Any],
    hidden_size: int | None = None,
) -> bool:
    tensor_path, metadata_path = _preanswer_feature_shard_paths(cache_dir, expected["name"])
    if not tensor_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "type",
            "version",
            "settings_fingerprint",
            "name",
            "route",
            "sample_indices",
            "sample_keys",
            "document_keys",
            "questions",
            "documents",
        ):
            if metadata.get(key) != expected.get(key):
                return False
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"document_offsets", "h0", "hD"}:
                return False
            h0_shape = tuple(handle.get_slice("h0").get_shape())
            hD_shape = tuple(handle.get_slice("hD").get_shape())
            offsets_shape = tuple(handle.get_slice("document_offsets").get_shape())
        if h0_shape[0] != expected["questions"] or hD_shape[0] != expected["documents"]:
            return False
        if offsets_shape != (expected["questions"] + 1,):
            return False
        if len(h0_shape) != 2 or len(hD_shape) != 2 or h0_shape[1] != hD_shape[1]:
            return False
        return hidden_size is None or h0_shape[1] == hidden_size
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _write_preanswer_feature_shard(
    cache_dir: Path,
    expected: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> None:
    tensor_path, metadata_path = _preanswer_feature_shard_paths(cache_dir, expected["name"])
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tensor = tensor_path.with_name(f".{tensor_path.stem}.tmp.safetensors")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    save_safetensors(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(temporary_tensor),
    )
    temporary_tensor.replace(tensor_path)
    write_json(
        temporary_metadata,
        {
            **expected,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tensor_path": str(tensor_path),
            "h0_shape": list(tensors["h0"].shape),
            "hD_shape": list(tensors["hD"].shape),
            "storage_dtype": str(tensors["h0"].dtype).replace("torch.", ""),
        },
    )
    temporary_metadata.replace(metadata_path)


def _load_preanswer_feature_shard(
    cache_dir: Path,
    expected: dict[str, Any],
    hidden_size: int | None = None,
) -> dict[str, torch.Tensor]:
    if not _valid_preanswer_feature_shard(cache_dir, expected, hidden_size=hidden_size):
        raise RuntimeError(f"Invalid pre-answer hidden feature shard: {expected['name']}")
    tensor_path, _ = _preanswer_feature_shard_paths(cache_dir, expected["name"])
    tensors = load_safetensors(str(tensor_path), device="cpu")
    offsets = tensors["document_offsets"].to(dtype=torch.int64).tolist()
    expected_counts = [len(documents) for documents in expected["document_keys"]]
    if offsets != [0, *list(np.cumsum(expected_counts, dtype=np.int64))]:
        raise RuntimeError(f"Invalid document offsets in pre-answer shard: {expected['name']}")
    return tensors


def ensure_preanswer_hidden_feature_cache(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
    candidate_dir: Path,
) -> tuple[Path, list[dict[str, Any]], str]:
    """Materialize h0/hD once, independently of any trained filter checkpoint."""

    cache_dir, settings, fingerprint = preanswer_hidden_feature_cache_dir(args, candidate_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    specs = _preanswer_feature_batch_specs(args, samples, reranked_docs)
    expected_rows = [
        _preanswer_feature_expected_metadata(spec, samples, reranked_docs, fingerprint)
        for spec in specs
    ]
    manifest_path = cache_dir / "manifest.json"
    valid = [
        _valid_preanswer_feature_shard(cache_dir, expected) for expected in expected_rows
    ]
    if all(valid):
        write_json(
            manifest_path,
            {
                "type": "rag2_mcq_eval_preanswer_hidden_features",
                "version": PREANSWER_HIDDEN_FEATURE_CACHE_VERSION,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "settings_fingerprint": fingerprint,
                "settings": settings,
                "questions": len(samples),
                "documents": sum(len(documents) for documents in reranked_docs),
                "shards": len(specs),
            },
        )
        logging.info(
            "Reusing pre-answer hidden feature cache: questions=%s documents=%s shards=%s path=%s",
            len(samples),
            sum(len(documents) for documents in reranked_docs),
            len(specs),
            cache_dir,
        )
        return cache_dir, expected_rows, fingerprint

    storage_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.hidden_feature_dtype]
    progress = StageProgress(
        total=sum(len(documents) for documents in reranked_docs),
        desc="CachingPreAnswerHidden",
        enabled=True,
    )
    for is_valid, expected in zip(valid, expected_rows):
        if is_valid:
            progress.update(int(expected["documents"]))

    logging.info(
        "Building model-independent pre-answer hidden cache: missing_shards=%s/%s "
        "state_model=%s layer=%s storage_dtype=%s path=%s",
        sum(not value for value in valid),
        len(valid),
        args.llm_model_path,
        args.hidden_feature_layer,
        args.hidden_feature_dtype,
        cache_dir,
    )
    extractor = PreAnswerLayerExtractor(
        args.llm_model_path,
        layer=args.hidden_feature_layer,
        batch_size=args.hidden_feature_batch_size,
        max_input_tokens=args.hidden_feature_max_input_tokens,
        device=args.filter_device,
        dtype=args.hidden_feature_dtype,
        attn_implementation=args.hidden_feature_attn_implementation,
    )
    try:
        for spec, expected, is_valid in zip(specs, expected_rows, valid):
            if is_valid:
                continue
            indices = list(spec["indices"])
            batch_samples = [samples[index] for index in indices]
            batch_docs = [reranked_docs[index] for index in indices]
            h0 = extractor.states(batch_samples, [None] * len(batch_samples))
            flat_samples = [
                sample
                for sample, documents in zip(batch_samples, batch_docs)
                for _ in documents
            ]
            flat_evidences = [
                preanswer_evidence(document, args.filter_max_doc_chars)
                for documents in batch_docs
                for document in documents
            ]
            hD = extractor.states(flat_samples, flat_evidences)
            offsets = [0]
            for documents in batch_docs:
                offsets.append(offsets[-1] + len(documents))
            _write_preanswer_feature_shard(
                cache_dir,
                expected,
                {
                    "h0": h0.to(dtype=storage_dtype),
                    "hD": hD.to(dtype=storage_dtype),
                    "document_offsets": torch.tensor(offsets, dtype=torch.int64),
                },
            )
            progress.update(len(flat_samples))
    finally:
        extractor.close()
        progress.close()

    if not all(
        _valid_preanswer_feature_shard(cache_dir, expected, hidden_size=extractor.hidden_size)
        for expected in expected_rows
    ):
        raise RuntimeError(f"Pre-answer hidden feature cache incomplete: {cache_dir}")
    write_json(
        manifest_path,
        {
            "type": "rag2_mcq_eval_preanswer_hidden_features",
            "version": PREANSWER_HIDDEN_FEATURE_CACHE_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings_fingerprint": fingerprint,
            "settings": settings,
            "questions": len(samples),
            "documents": sum(len(documents) for documents in reranked_docs),
            "shards": len(specs),
            "hidden_size": extractor.hidden_size,
        },
    )
    logging.info("Pre-answer hidden feature cache complete: %s", cache_dir)
    return cache_dir, expected_rows, fingerprint


def _apply_cached_filter_rows(
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
    rows: dict[str, dict[str, Any]],
) -> bool:
    if len(rows) < len(samples):
        return False
    for sample, docs in zip(samples, reranked_docs):
        row = rows.get(sample_key(sample))
        decisions = list((row or {}).get("filter_decisions") or [])
        if len(decisions) != len(docs):
            return False
        for doc, decision in zip(docs, decisions):
            if str(decision.get("db_id") or "") != doc.db_id:
                return False
            if int(decision.get("local_id", -1)) != doc.local_id:
                return False
    for sample, docs in zip(samples, reranked_docs):
        decisions = rows[sample_key(sample)]["filter_decisions"]
        for doc, decision in zip(docs, decisions):
            doc.filter_prediction = decision.get("filter_prediction")
            doc.filter_score = decision.get("filter_score")
            doc.filter_prob_helpful = decision.get("filter_prob_helpful")
            doc.filter_rank = decision.get("filter_rank")
            window_filter = decision.get("window_filter")
            if window_filter is not None:
                if not isinstance(window_filter, dict):
                    return False
                doc.metadata["window_filter"] = window_filter
            hidden_filter = decision.get("preanswer_text_hidden_filter")
            if hidden_filter is not None:
                if not isinstance(hidden_filter, dict):
                    return False
                doc.metadata["preanswer_text_hidden_filter"] = hidden_filter
    return True


def _cached_filter_row_matches(
    sample: BenchmarkSample,
    docs: list[RetrievedDocument],
    row: dict[str, Any] | None,
) -> bool:
    if not isinstance(row, dict) or str(row.get("key") or "") != sample_key(sample):
        return False
    decisions = list(row.get("filter_decisions") or [])
    if len(decisions) != len(docs):
        return False
    return all(
        str(decision.get("db_id") or "") == doc.db_id
        and int(decision.get("local_id", -1)) == doc.local_id
        for doc, decision in zip(docs, decisions)
    )


def _apply_cached_filter_row(
    docs: list[RetrievedDocument],
    row: dict[str, Any],
) -> None:
    for doc, decision in zip(docs, row["filter_decisions"]):
        doc.filter_prediction = decision.get("filter_prediction")
        doc.filter_score = decision.get("filter_score")
        doc.filter_prob_helpful = decision.get("filter_prob_helpful")
        doc.filter_rank = decision.get("filter_rank")
        window_filter = decision.get("window_filter")
        if isinstance(window_filter, dict):
            doc.metadata["window_filter"] = window_filter
        hidden_filter = decision.get("preanswer_text_hidden_filter")
        if isinstance(hidden_filter, dict):
            doc.metadata["preanswer_text_hidden_filter"] = hidden_filter


def _rank_scored_documents(docs: list[RetrievedDocument]) -> None:
    helpful_rank = 0
    for doc in docs:
        if doc.filter_prediction == "helpful":
            helpful_rank += 1
            doc.filter_rank = helpful_rank
        else:
            doc.filter_rank = None


def apply_oracle_labels(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
) -> list[list[RetrievedDocument]]:
    """Apply externally materialized gold labels without invoking a learned filter."""

    assert args.oracle_labels_path is not None
    labels: dict[tuple[str, int, str], dict[str, Any]] = {}
    labels_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    with args.oracle_labels_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed oracle JSONL: {args.oracle_labels_path}:{line_number}") from error
            key = (str(row.get("sample_key") or ""), int(row.get("doc_rank") or 0), str(row.get("doc_stable_id") or ""))
            if key in labels:
                raise ValueError(f"Duplicate oracle decision: {key}")
            labels[key] = row
            identity_key = (key[0], key[2])
            if not identity_key[0] or not identity_key[1]:
                raise ValueError(f"Oracle decision has an incomplete identity: {key}")
            if identity_key in labels_by_identity:
                raise ValueError(f"Duplicate oracle document identity: {identity_key}")
            labels_by_identity[identity_key] = row
    used: set[tuple[str, int, str]] = set()
    for sample, documents in zip(samples, reranked_docs):
        for rank, document in enumerate(documents, 1):
            key = (sample_key(sample), rank, document.stable_id)
            row = labels.get(key)
            if row is None:
                # Dynamic paper-balanced Top-k conditions rerank a different
                # 4k pool for every k. The document's rank in the compact
                # annotation union is therefore not its rank in a projected
                # condition; sample key + stable document identity is the
                # authoritative oracle join.
                row = labels_by_identity.get((key[0], key[2]))
            if row is None:
                raise KeyError(f"Missing oracle decision: {key}")
            used.add(key)
            if args.oracle_policy == "rag2":
                helpful = str(row.get("pseudo_label") or "") == "Helpful" and bool(row.get("quality_pass"))
                score = row.get("delta_ppl")
            elif args.oracle_policy == "margin_utility":
                helpful = str(row.get("pseudo_label") or "") == "Helpful" and bool(
                    row.get("quality_pass")
                )
                score = row.get("utility_score")
            elif args.oracle_policy == "hidden_three_class":
                helpful = str(row.get("hidden_label") or "") == "Helpful" and bool(
                    row.get("hidden_quality_pass")
                )
                score = row.get("projection_score")
            else:
                threshold = 0.0 if args.oracle_policy == "hidden_tau_0" else 0.4
                score = float(row["projection_score"])
                helpful = score > threshold
            document.filter_prediction = "helpful" if helpful else "not helpful"
            document.filter_score = None if score is None else float(score)
            document.filter_prob_helpful = None
            document.metadata["oracle_filter"] = {
                "policy": args.oracle_policy,
                "gold_label": (
                    row.get("pseudo_label")
                    if args.oracle_policy in {"rag2", "margin_utility"}
                    else row.get("hidden_label")
                ),
                "utility_score": row.get("utility_score"),
                "projection_score": row.get("projection_score"),
                "hidden_threshold": row.get("hidden_threshold"),
            }
        _rank_scored_documents(documents)
    if len(used) != sum(len(value) for value in reranked_docs):
        raise RuntimeError("Oracle label application did not cover every reranked document")
    logging.info("Applied %s oracle decisions from %s", len(used), args.oracle_labels_path)
    return reranked_docs


def _filter_score_row(
    sample: BenchmarkSample,
    docs: list[RetrievedDocument],
) -> dict[str, Any]:
    route = "medqa" if sample.dataset == "medqa" else "medmcqa"
    return {
        "key": sample_key(sample),
        "dataset": sample.dataset,
        "sample_id": sample.id,
        "row_idx": sample.row_idx,
        "filter_model_route": route,
        "filter_decisions": [
            {
                "db_id": doc.db_id,
                "local_id": doc.local_id,
                "rerank_rank": doc.rerank_rank,
                "filter_prediction": doc.filter_prediction,
                "filter_score": doc.filter_score,
                "filter_prob_helpful": doc.filter_prob_helpful,
                "filter_rank": doc.filter_rank,
                "window_filter": doc.metadata.get("window_filter"),
                "preanswer_text_hidden_filter": doc.metadata.get(
                    "preanswer_text_hidden_filter"
                ),
            }
            for doc in docs
        ],
    }


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _resume_preanswer_hidden_filter_scores(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
    candidate_dir: Path,
    cache_dir: Path,
    settings: dict[str, Any],
    fingerprint: str,
) -> list[list[RetrievedDocument]]:
    """Score text+hidden pairs in durable question batches.

    h0/hD are first materialized in a filter-checkpoint-independent cache.  The
    route-specific classifier then consumes those tensors.  After every
    question shard, complete decisions are flushed and fsynced; a later run can
    therefore reuse both hidden features and all completed classifier rows.
    """

    feature_cache_dir, feature_metadata, _ = ensure_preanswer_hidden_feature_cache(
        args,
        samples,
        reranked_docs,
        candidate_dir,
    )

    output_path = cache_dir / "filter_scores.jsonl"
    progress_path = cache_dir / "in_progress.json"
    cached_rows: dict[str, dict[str, Any]] = {}
    if not args.rebuild_filter_cache and output_path.exists() and progress_path.exists():
        try:
            progress_state = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_state.get("settings_fingerprint") == fingerprint:
                cached_rows = load_candidate_rows(output_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logging.warning("Ignoring invalid resumable hidden-filter cache: %s", output_path)

    valid_rows: dict[str, dict[str, Any]] = {}
    for sample, docs in zip(samples, reranked_docs):
        row = cached_rows.get(sample_key(sample))
        if _cached_filter_row_matches(sample, docs, row):
            assert row is not None
            _apply_cached_filter_row(docs, row)
            valid_rows[sample_key(sample)] = row

    # Canonicalise a possibly interrupted trailing JSONL line before appending.
    _write_jsonl_atomic(
        output_path,
        [valid_rows[sample_key(sample)] for sample in samples if sample_key(sample) in valid_rows],
    )
    write_json(
        progress_path,
        {
            "type": "rag2_mcq_eval_filter_scores_in_progress",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings_fingerprint": fingerprint,
            "settings": settings,
            "rows_total": len(samples),
            "rows_completed": len(valid_rows),
            "documents_completed": sum(
                len(docs)
                for sample, docs in zip(samples, reranked_docs)
                if sample_key(sample) in valid_rows
            ),
            "output_path": str(output_path),
        },
    )

    model_paths = {
        "medmcqa": args.medmcqa_filter_model_path,
        "medqa": args.medqa_filter_model_path,
    }
    completed_documents = sum(
        len(docs)
        for sample, docs in zip(samples, reranked_docs)
        if sample_key(sample) in valid_rows
    )
    progress = StageProgress(
        total=sum(len(docs) for docs in reranked_docs),
        desc="FilteringPreAnswerTextHidden",
        enabled=True,
    )
    if completed_documents:
        progress.update(completed_documents)

    try:
        with output_path.open("a", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
            for route in ("medqa", "medmcqa"):
                indices = [
                    index for index, sample in enumerate(samples)
                    if ("medqa" if sample.dataset == "medqa" else "medmcqa") == route
                    and sample_key(sample) not in valid_rows
                ]
                if not indices:
                    continue
                logging.info(
                    "Resumable pre-answer text+hidden route=%s datasets=%s missing_questions=%s "
                    "missing_documents=%s checkpoint=%s state_model=%s layer=%s threshold=%.6f",
                    route,
                    sorted({samples[index].dataset for index in indices}),
                    len(indices),
                    sum(len(reranked_docs[index]) for index in indices),
                    model_paths[route],
                    args.llm_model_path,
                    args.hidden_feature_layer,
                    args.hidden_filter_helpful_threshold,
                )
                filterer = TextHiddenRag2Filter(
                    checkpoint_path=model_paths[route],
                    backbone_path=args.hidden_filter_backbone_path,
                    state_model_path=args.llm_model_path,
                    layer=args.hidden_feature_layer,
                    hidden_batch_size=args.hidden_feature_batch_size,
                    filter_batch_size=args.filter_batch_size,
                    max_hidden_input_tokens=args.hidden_feature_max_input_tokens,
                    max_filter_input_length=args.filter_max_input_length,
                    max_doc_chars=args.filter_max_doc_chars,
                    helpful_threshold=args.hidden_filter_helpful_threshold,
                    device=args.filter_device,
                    bf16=args.filter_bf16,
                    hidden_dtype=args.hidden_feature_dtype,
                    hidden_attn_implementation=args.hidden_feature_attn_implementation,
                    load_state_extractor=False,
                )
                try:
                    route_shards = [
                        metadata for metadata in feature_metadata if metadata["route"] == route
                    ]
                    for metadata in route_shards:
                        shard_indices = list(metadata["sample_indices"])
                        missing_positions = [
                            position
                            for position, index in enumerate(shard_indices)
                            if sample_key(samples[index]) not in valid_rows
                        ]
                        if not missing_positions:
                            continue
                        tensors = _load_preanswer_feature_shard(
                            feature_cache_dir,
                            metadata,
                            hidden_size=filterer.hidden_size,
                        )
                        original_offsets = tensors["document_offsets"].to(torch.int64).tolist()
                        hD_parts = [
                            tensors["hD"][original_offsets[position] : original_offsets[position + 1]]
                            for position in missing_positions
                        ]
                        subset_offsets = [0]
                        for part in hD_parts:
                            subset_offsets.append(subset_offsets[-1] + int(part.shape[0]))
                        batch_indices = [shard_indices[position] for position in missing_positions]
                        filterer.score_documents_from_features(
                            [samples[index] for index in batch_indices],
                            [reranked_docs[index] for index in batch_indices],
                            tensors["h0"][missing_positions],
                            torch.cat(hD_parts, dim=0),
                            torch.tensor(subset_offsets, dtype=torch.int64),
                            progress_callback=progress.update,
                        )
                        del tensors, hD_parts
                        for index in batch_indices:
                            _rank_scored_documents(reranked_docs[index])
                            row = _filter_score_row(samples[index], reranked_docs[index])
                            valid_rows[sample_key(samples[index])] = row
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        write_json(
                            progress_path,
                            {
                                "type": "rag2_mcq_eval_filter_scores_in_progress",
                                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "settings_fingerprint": fingerprint,
                                "settings": settings,
                                "rows_total": len(samples),
                                "rows_completed": len(valid_rows),
                                "documents_completed": sum(
                                    len(row["filter_decisions"]) for row in valid_rows.values()
                                ),
                                "output_path": str(output_path),
                            },
                        )
                finally:
                    filterer.close()
    finally:
        progress.close()

    if len(valid_rows) != len(samples):
        raise RuntimeError(
            f"Hidden-filter cache incomplete: rows={len(valid_rows)}/{len(samples)} path={output_path}"
        )
    # Stable sample order makes the completed artifact easy to audit.
    _write_jsonl_atomic(output_path, [valid_rows[sample_key(sample)] for sample in samples])
    return reranked_docs


def ensure_filter_scores(
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    reranked_docs: list[list[RetrievedDocument]],
    candidate_dir: Path,
) -> tuple[list[list[RetrievedDocument]], Path]:
    cache_dir, settings, fingerprint = filter_score_cache_dir(args, candidate_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "filter_scores.jsonl"
    manifest_path = cache_dir / "manifest.json"

    if not args.rebuild_filter_cache and output_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached_rows = load_candidate_rows(output_path)
            compatible = (
                manifest.get("type") == "rag2_mcq_eval_filter_scores"
                and manifest.get("settings_fingerprint") == fingerprint
                and int(manifest.get("rows", -1)) == len(samples)
                and int(manifest.get("documents", -1)) == sum(len(docs) for docs in reranked_docs)
                and _apply_cached_filter_rows(samples, reranked_docs, cached_rows)
            )
            if compatible:
                logging.info(
                    "Reusing completed filter-score cache: rows=%s documents=%s path=%s",
                    len(samples),
                    sum(len(docs) for docs in reranked_docs),
                    output_path,
                )
                return reranked_docs, cache_dir
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logging.warning("Ignoring invalid filter-score cache: %s", output_path)

    if args.filter_evidence_unit == "preanswer_text_hidden":
        scored_docs = _resume_preanswer_hidden_filter_scores(
            args,
            samples,
            reranked_docs,
            candidate_dir,
            cache_dir,
            settings,
            fingerprint,
        )
    else:
        scored_docs = score_filter_documents(args, samples, reranked_docs)
    rows = [_filter_score_row(sample, docs) for sample, docs in zip(samples, scored_docs)]
    write_jsonl(output_path, rows)
    write_json(
        manifest_path,
        {
            "type": "rag2_mcq_eval_filter_scores",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "rows": len(samples),
            "documents": sum(len(docs) for docs in scored_docs),
            "settings_fingerprint": fingerprint,
            "settings": settings,
            "output_path": str(output_path),
        },
    )
    logging.info(
        "Filter-score cache complete: rows=%s documents=%s path=%s",
        len(samples),
        sum(len(docs) for docs in scored_docs),
        output_path,
    )
    return scored_docs, cache_dir


def write_outputs(
    args: argparse.Namespace,
    results: list[CaseResult],
    details: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_case = args.case
    if args.paper_balanced_top_k is not None and args.case in {"rerank_rag", "filter_rag", "oracle_rag"}:
        suffix = f"_{args.oracle_policy}" if args.case == "oracle_rag" else ""
        output_case = f"{args.case}{suffix}_top{args.paper_balanced_top_k}"
    elif args.case == "rerank_rag" and args.generation_top_k is not None:
        output_case = f"{args.case}_top{args.generation_top_k}"
    elif args.case in {"filter_rag", "oracle_rag"} and args.filter_rerank_top_k is not None:
        suffix = f"_{args.oracle_policy}" if args.case == "oracle_rag" else ""
        output_case = f"{args.case}{suffix}_top{args.filter_rerank_top_k}"
    output_dir = args.results_root / output_case / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {**config, "output_dir": str(output_dir), "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    write_json(output_dir / "run_config.json", config)
    environment = collect_environment(
        command=[sys.executable, *sys.argv],
        project_root=PROJECT_ROOT,
        workspace_root=WORKSPACE_ROOT,
        run_config=config,
    )
    write_environment_files(output_dir, environment)

    rows = results_to_jsonable(results, include_doc_text=args.include_doc_text_in_jsonl)
    for row, result in zip(rows, results):
        row.update(details[sample_key(result.sample)])
        row["filter_model_route"] = (
            "medqa" if args.case == "filter_rag" and result.sample.dataset == "medqa"
            else "medmcqa" if args.case == "filter_rag"
            else "gold_oracle" if args.case == "oracle_rag"
            else None
        )
    write_jsonl(output_dir / "results.jsonl", rows)
    write_markdown_report(output_dir / "summary.md", results, config=config)
    write_text_report(output_dir / "summary.txt", results)
    write_pretty_summary_table(output_dir / "summary_table_pretty.txt", results)
    exclusions = config.get("no_rag_artifact_exclusions") or []
    if exclusions:
        write_json(output_dir / "rationale_exclusions.json", exclusions)

    if args.case in {"filter_rag", "oracle_rag"}:
        by_dataset: dict[str, dict[str, Any]] = {}
        eligible_top_k = (
            args.paper_balanced_top_k
            or args.filter_rerank_top_k
            or args.rerank_top_k
        )
        for result in results:
            bucket = by_dataset.setdefault(
                result.sample.dataset,
                {
                    "samples": 0,
                    "eligible_reranked_documents": 0,
                    "context_documents": 0,
                    "zero_document_samples": 0,
                    "predictions_all_scored": Counter(),
                    "predictions_inside_rerank_top_k": Counter(),
                    "all_scored_documents": 0,
                    "scored_windows": 0,
                    "documents_with_raw_helpful_window": 0,
                },
            )
            bucket["samples"] += 1
            eligible_docs = result.reranked_documents[:eligible_top_k]
            bucket["eligible_reranked_documents"] += len(eligible_docs)
            bucket["context_documents"] += len(result.final_documents)
            bucket["zero_document_samples"] += int(not result.final_documents)
            bucket["predictions_all_scored"].update(
                str(doc.filter_prediction or "missing") for doc in result.reranked_documents
            )
            bucket["all_scored_documents"] += len(result.reranked_documents)
            bucket["predictions_inside_rerank_top_k"].update(
                str(doc.filter_prediction or "missing") for doc in eligible_docs
            )
            for doc in result.reranked_documents:
                window_filter = doc.metadata.get("window_filter") if isinstance(doc.metadata, dict) else None
                if isinstance(window_filter, dict):
                    bucket["scored_windows"] += int(window_filter.get("scored_window_count") or 0)
                    bucket["documents_with_raw_helpful_window"] += int(
                        int(window_filter.get("raw_helpful_window_count") or 0) > 0
                    )
        serializable = {}
        for dataset, bucket in by_dataset.items():
            serializable[dataset] = {
                **bucket,
                "filter_rerank_top_k": eligible_top_k,
                "predictions_all_scored": dict(bucket["predictions_all_scored"]),
                "predictions_inside_rerank_top_k": dict(bucket["predictions_inside_rerank_top_k"]),
                "mean_context_documents": bucket["context_documents"] / bucket["samples"] if bucket["samples"] else 0.0,
                "zero_document_rate": bucket["zero_document_samples"] / bucket["samples"] if bucket["samples"] else 0.0,
                "mean_scored_windows_per_reranked_document": (
                    bucket["scored_windows"] / bucket["all_scored_documents"]
                    if bucket["all_scored_documents"]
                    else 0.0
                ),
                "documents_with_raw_helpful_window": bucket["documents_with_raw_helpful_window"],
            }
        write_json(output_dir / "filter_summary.json", serializable)
    logging.info("Run complete: %s", output_dir)
    return output_dir


def validate_paths(args: argparse.Namespace) -> None:
    required = [] if args.candidate_cache_only or args.filter_cache_only else [args.llm_model_path]
    if args.case in {"rerank_rag", "filter_rag", "oracle_rag"}:
        required.extend([args.query_encoder_path, args.cross_encoder_path])
        required.extend(args.vector_db_root / source for source in args.sources)
    if args.case == "filter_rag":
        routes = {_filter_routes()[dataset] for dataset in args.datasets}
        if "medqa" in routes:
            required.append(args.medqa_filter_model_path)
        if "medmcqa" in routes:
            required.append(args.medmcqa_filter_model_path)
        if args.filter_evidence_unit == "preanswer_text_hidden":
            required.extend([args.hidden_filter_backbone_path, args.llm_model_path])
        if args.filter_evidence_unit == "document_transformer":
            if "medqa" in routes:
                if args.medqa_document_transformer_checkpoint is None:
                    raise ValueError("--medqa-document-transformer-checkpoint is required for MedQA")
                required.append(args.medqa_document_transformer_checkpoint)
            if "medmcqa" in routes:
                if args.medmcqa_document_transformer_checkpoint is None:
                    raise ValueError(
                        "--medmcqa-document-transformer-checkpoint is required for MedMCQA/MMLU"
                    )
                required.append(args.medmcqa_document_transformer_checkpoint)
    if args.case == "oracle_rag":
        if args.oracle_labels_path is None or args.oracle_policy is None:
            raise ValueError("--case oracle_rag requires --oracle-labels-path and --oracle-policy")
        required.append(args.oracle_labels_path)
    elif args.oracle_labels_path is not None or args.oracle_policy is not None:
        raise ValueError("--oracle-labels-path/--oracle-policy are supported only with --case oracle_rag")
    if args.candidate_cache_source_path is not None:
        required.extend(
            [
                args.candidate_cache_source_path,
                args.candidate_cache_source_path.parent / "manifest.json",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(str(path) for path in missing))
    if args.pubmed_shards_per_group <= 0:
        raise ValueError("--pubmed-shards-per-group must be positive.")
    if args.filter_window_context_sentences < 0:
        raise ValueError("--filter-window-context-sentences must be non-negative.")
    if args.filter_window_question_batch_size <= 0:
        raise ValueError("--filter-window-question-batch-size must be positive.")
    if not 0.0 <= args.filter_window_helpful_threshold <= 1.0:
        raise ValueError("--filter-window-helpful-threshold must be in [0, 1].")
    if not 0.0 <= args.document_transformer_helpful_threshold <= 1.0:
        raise ValueError("--document-transformer-helpful-threshold must be in [0, 1].")
    resolve_document_transformer_thresholds(args)
    if args.document_transformer_batch_size <= 0:
        raise ValueError("--document-transformer-batch-size must be positive.")
    if args.hidden_feature_batch_size <= 0:
        raise ValueError("--hidden-feature-batch-size must be positive.")
    if args.hidden_filter_question_batch_size <= 0:
        raise ValueError("--hidden-filter-question-batch-size must be positive.")
    if args.hidden_feature_max_input_tokens <= 0:
        raise ValueError("--hidden-feature-max-input-tokens must be positive.")
    if not 0.0 <= args.hidden_filter_helpful_threshold <= 1.0:
        raise ValueError("--hidden-filter-helpful-threshold must be in [0, 1].")
    if args.filter_evidence_unit == "sentence_window":
        resolve_filter_window_thresholds(args)
    if args.filter_generation_context_unit == "best_helpful_window" and args.filter_evidence_unit != "sentence_window":
        raise ValueError(
            "--filter-generation-context-unit=best_helpful_window requires "
            "--filter-evidence-unit=sentence_window."
        )
    if args.document_token_safety_margin < 0:
        raise ValueError("--document-token-safety-margin must be non-negative.")
    if args.document_packing == "dynamic_token_budget" and (
        args.llm_max_model_len <= args.max_new_tokens + args.document_token_safety_margin
    ):
        raise ValueError(
            "--llm-max-model-len must exceed --max-new-tokens plus --document-token-safety-margin "
            "when using dynamic_token_budget."
        )
    expected_pool = expected_candidate_pool_size(args.vector_db_root, args)
    if args.candidate_pool_top_k != expected_pool:
        raise ValueError(
            "Candidate-pool size does not match the selected retrieval layout: "
            f"candidate_pool_top_k={args.candidate_pool_top_k}, expected={expected_pool} "
            f"(layout={args.candidate_layout}, per_source_top_k={args.per_source_top_k})."
        )
    if args.rerank_top_k > args.candidate_pool_top_k:
        raise ValueError("--rerank-top-k cannot exceed --candidate-pool-top-k")
    if args.generation_top_k is not None:
        if args.case != "rerank_rag":
            raise ValueError("--generation-top-k is supported only with --case rerank_rag.")
        if args.generation_top_k <= 0 or args.generation_top_k > args.rerank_top_k:
            raise ValueError(
                f"--generation-top-k must be in [1, {args.rerank_top_k}], got {args.generation_top_k}."
            )
    if args.filter_rerank_top_k is not None:
        if args.case not in {"filter_rag", "oracle_rag"}:
            raise ValueError("--filter-rerank-top-k is supported only with --case filter_rag or oracle_rag.")
        if args.filter_rerank_top_k <= 0 or args.filter_rerank_top_k > args.rerank_top_k:
            raise ValueError(
                f"--filter-rerank-top-k must be in [1, {args.rerank_top_k}], "
                f"got {args.filter_rerank_top_k}."
            )
    if args.paper_balanced_top_k is not None:
        if args.case not in {"rerank_rag", "filter_rag", "oracle_rag"}:
            raise ValueError(
                "--paper-balanced-top-k is supported only with rerank_rag, filter_rag, or oracle_rag."
            )
        if args.candidate_layout != "source_balanced":
            raise ValueError("--paper-balanced-top-k requires --candidate-layout source_balanced.")
        if args.sources != PAPER_SOURCES:
            raise ValueError(
                "--paper-balanced-top-k requires exactly the four logical paper corpora in canonical order: "
                f"{PAPER_SOURCES}; got {args.sources}."
            )
        if args.paper_balanced_top_k <= 0 or args.paper_balanced_top_k > args.per_source_top_k:
            raise ValueError(
                "--paper-balanced-top-k must be in "
                f"[1, {args.per_source_top_k}], got {args.paper_balanced_top_k}."
            )
        if args.rerank_top_k != args.candidate_pool_top_k:
            raise ValueError(
                "--paper-balanced-top-k requires the master cache to retain a rerank score and document text "
                "for every dense candidate: --rerank-top-k must equal --candidate-pool-top-k."
            )
        if args.generation_top_k is not None or args.filter_rerank_top_k is not None:
            raise ValueError(
                "--paper-balanced-top-k cannot be combined with the legacy global-prefix "
                "--generation-top-k/--filter-rerank-top-k options."
            )
    if args.candidate_cache_only and args.case != "rerank_rag":
        raise ValueError("--candidate-cache-only is supported only with --case rerank_rag.")
    if args.filter_cache_only and args.case != "filter_rag":
        raise ValueError("--filter-cache-only is supported only with --case filter_rag.")
    if args.candidate_cache_only and args.filter_cache_only:
        raise ValueError("--candidate-cache-only and --filter-cache-only are mutually exclusive.")
    if args.answer_decision_mode == "constrained_choice":
        if args.max_new_tokens != 1:
            raise ValueError("--answer-decision-mode=constrained_choice requires --max-new-tokens 1.")
        if args.temperature != 0.0:
            raise ValueError("--answer-decision-mode=constrained_choice requires --temperature 0.")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    validate_paths(args)
    samples, grouped = load_benchmarks(args)
    if args.answer_decision_mode == "constrained_choice":
        invalid_options = [
            sample.id
            for sample in samples
            if not isinstance(sample.options, dict) or set(sample.options) != {"A", "B", "C", "D"}
        ]
        if invalid_options:
            preview = ", ".join(invalid_options[:5])
            raise ValueError(
                "Constrained-choice evaluation currently requires exactly A/B/C/D options; "
                f"invalid samples={len(invalid_options)} examples={preview}"
            )
    logging.info("Loaded %s total MCQ test samples across %s datasets.", len(samples), len(grouped))
    if args.dry_run:
        logging.info(
            "Dry run valid: case=%s samples=%s sources=%s candidate_layout=%s candidates=%s "
            "rerank_top_k_scored=%s effective_rerank_top_k=%s",
            args.case,
            len(samples),
            ",".join(args.sources),
            args.candidate_layout,
            args.candidate_pool_top_k,
            args.rerank_top_k,
            (
                args.paper_balanced_top_k
                or args.filter_rerank_top_k
                or args.rerank_top_k
                if args.case in {"filter_rag", "oracle_rag"}
                else args.paper_balanced_top_k
                or args.generation_top_k
                or args.rerank_top_k
            ),
        )
        return

    requested_total_samples = len(samples)
    artifacts, artifact_exclusions = ensure_rationale_artifacts(args, grouped)
    if artifact_exclusions:
        excluded_indices = {
            (str(item["dataset"]), int(item["row_idx"]))
            for item in artifact_exclusions
        }
        grouped = {
            dataset: [
                sample
                for idx, sample in enumerate(dataset_samples)
                if (dataset, idx) not in excluded_indices
            ]
            for dataset, dataset_samples in grouped.items()
        }
        samples = [sample for dataset in args.datasets for sample in grouped[dataset]]
        logging.warning(
            "Proceeding with %s/%s samples after excluding %s malformed no-RAG artifacts.",
            len(samples),
            requested_total_samples,
            len(artifact_exclusions),
        )
    document_transformer_thresholds = resolve_document_transformer_thresholds(args)
    config: dict[str, Any] = {
        "case": args.case,
        "datasets": args.datasets,
        "total_samples": len(samples),
        "requested_total_samples": requested_total_samples,
        "excluded_no_rag_artifact_count": len(artifact_exclusions),
        "no_rag_artifact_exclusions": artifact_exclusions,
        "collection": args.collection,
        "split": args.split,
        "prompt_profile": args.prompt_profile,
        "answer_decision_mode": args.answer_decision_mode,
        "answer_decision_contract": (
            {
                "prompt_version": ANCHORED_PROMPT_VERSION,
                "generation_policy_version": ANCHORED_GENERATION_POLICY_VERSION,
                "rationale_decoding": "greedy_free_generation_to_fixed_reasoning_boundary",
                "choice_decoding": "greedy_argmax_over_allowed_single_token_A_B_C_D",
                "terminal_format": "Final answer: (<OPTION LETTER>) <EXACT OPTION TEXT>",
                "rationale_generated": True,
                "parser_used": False,
                "retrieval_query": "generated rationale plus fixed terminal answer; control markers excluded",
            }
            if args.prompt_profile == "paper_compatible_three_anchor"
            else
            {
                "prompt_version": PREANSWER_PROMPT_VERSION,
                "assistant_prefill": FINAL_ANSWER_PREFILL,
                "valid_choices": ["A", "B", "C", "D"],
                "decoding": "greedy_argmax_over_allowed_single_token_choices",
                "rationale_generated": False,
                "parser_used": False,
            }
            if args.answer_decision_mode == "constrained_choice"
            else {
                "prompt_version": active_document_prompt_version(args),
                "decoding": (
                    "free_rationale_with_structured_exact_terminal_line_and_constrained_exception_fallback"
                    if args.prompt_profile == "paper_exact_terminal"
                    else "free_generation_then_parse"
                ),
                "rationale_generated": True,
                "parser_used": True,
                "retrieval_query_includes_terminal_answer": args.prompt_profile == "paper_exact_terminal",
            }
        ),
        "rationale_prompt_version": active_no_rag_prompt_version(args),
        "rationale_artifact_policy": args.rationale_artifact_policy,
        "multi_document_prompt_version": active_document_prompt_version(args),
        "rationale_artifact_root": str(rationale_artifact_root(args)),
        "candidate_cache_source_path": (
            str(args.candidate_cache_source_path.resolve())
            if args.candidate_cache_source_path is not None
            else None
        ),
        "sources": args.sources,
        "per_source_top_k": args.per_source_top_k,
        "candidate_pool_top_k": args.candidate_pool_top_k,
        "candidate_layout": args.candidate_layout,
        "pubmed_shards_per_group": args.pubmed_shards_per_group,
        "rerank_top_k": args.rerank_top_k,
        "rerank_top_k_scored": args.rerank_top_k,
        "paper_balanced_top_k": args.paper_balanced_top_k,
        "paper_balanced_candidate_pool_top_k": (
            args.paper_balanced_top_k * len(args.sources)
            if args.paper_balanced_top_k is not None
            else None
        ),
        "generation_top_k": (
            args.paper_balanced_top_k
            or args.generation_top_k
            or args.rerank_top_k
            if args.case == "rerank_rag"
            else None
        ),
        "filter_rerank_top_k": (
            args.paper_balanced_top_k
            or args.filter_rerank_top_k
            or args.rerank_top_k
            if args.case in {"filter_rag", "oracle_rag"} else None
        ),
        "oracle_policy": args.oracle_policy,
        "oracle_labels_path": str(args.oracle_labels_path.resolve()) if args.oracle_labels_path else None,
        "filter_policy": (
            "gold-label oracle: inside each rerank Top-k prefix, retain only documents whose externally "
            "materialized policy label is Helpful; preserve rerank order, no fill, no learned filter"
            if args.case == "oracle_rag"
            else
            f"split every reranked document into all unique "
            f"{'single sentences' if args.filter_window_context_sentences == 0 else 'centred sentence-context windows'}; "
            "use the route-specific frozen Direct Sentence/Window Filter to extract each evidence unit's "
            "Evidence-pooled encoder vector, utility probability/margin, and position; classify the original "
            "document with the learned Document Transformer using the route-specific validation threshold; inside "
            "the selected MedCPT rerank Top-k prefix, keep predicted-Helpful original documents in rerank order; "
            "no fill and no post-filter Top-k. Final LLM context unit: document"
            if args.filter_evidence_unit == "document_transformer"
            else
            (
                "for each reranked document, extract the target Llama's fixed-prompt pre-answer h0 and hD "
                f"at layer {args.hidden_feature_layer} without gold labels; score the official Question+Evidence "
                "text together with h0 and delta_h=hD-h0 using the trained text+hidden filter; inside the "
                "selected MedCPT rerank Top-k prefix, retain only documents above the Helpful threshold in "
                "rerank order; no fill and no post-filter Top-k"
            )
            if args.filter_evidence_unit == "preanswer_text_hidden"
            else
            (
                "split every reranked document into all unique centred sentence-context windows; score/cache every "
                "window, aggregate each document as max P(Helpful), and keep a document iff that score clears its "
                "dataset-routed validation-calibrated threshold. Inside the selected MedCPT rerank Top-k prefix, keep "
                "passing documents in rerank order; no fill and no post-filter Top-k. Final LLM context unit: "
                f"{args.filter_generation_context_unit}"
                if args.filter_window_thresholds_path
                else "split every reranked document into all unique centred sentence-context windows; score/cache every "
                "window, aggregate each document as max P(Helpful), and keep a document iff that score clears the "
                "explicit fixed threshold (0.5 means at least one window has Helpful two-label argmax). Inside the "
                "selected MedCPT rerank Top-k prefix, keep passing documents in rerank order; no fill and no "
                f"post-filter Top-k. Final LLM context unit: {args.filter_generation_context_unit}"
            )
            if args.filter_evidence_unit == "sentence_window"
            else "score/cache every reranked document once; inside the selected MedCPT rerank Top-k prefix, "
            "keep only documents predicted helpful in original rerank order; no fill and no post-filter Top-k"
        ),
        "filter_routes": {"medqa": "medqa", "medmcqa_and_mmlu": "medmcqa"},
        "llm_model_path": str(args.llm_model_path),
        "query_encoder_path": str(args.query_encoder_path),
        "cross_encoder_path": str(args.cross_encoder_path),
        "medmcqa_filter_model_path": str(args.medmcqa_filter_model_path),
        "medqa_filter_model_path": str(args.medqa_filter_model_path),
        "medmcqa_document_transformer_checkpoint": (
            str(args.medmcqa_document_transformer_checkpoint)
            if args.medmcqa_document_transformer_checkpoint is not None
            else None
        ),
        "medqa_document_transformer_checkpoint": (
            str(args.medqa_document_transformer_checkpoint)
            if args.medqa_document_transformer_checkpoint is not None
            else None
        ),
        "document_transformer_helpful_thresholds": (
            document_transformer_thresholds
            if args.filter_evidence_unit == "document_transformer"
            else None
        ),
        "document_transformer_batch_size": (
            args.document_transformer_batch_size
            if args.filter_evidence_unit == "document_transformer"
            else None
        ),
        "filter_scoring_method": args.filter_scoring_method,
        "filter_input_format": args.filter_input_format,
        "filter_score_normalization": args.filter_score_normalization,
        "filter_max_input_length": args.filter_max_input_length,
        "filter_max_doc_chars": args.filter_max_doc_chars,
        "filter_evidence_unit": args.filter_evidence_unit,
        "filter_generation_context_unit": args.filter_generation_context_unit,
        "filter_window_context_sentences": args.filter_window_context_sentences,
        "filter_window_question_batch_size": args.filter_window_question_batch_size,
        "filter_window_aggregation": (
            "max_probability"
            if args.filter_evidence_unit == "sentence_window"
            else "document_transformer"
            if args.filter_evidence_unit == "document_transformer"
            else None
        ),
        "filter_window_thresholds": (
            resolve_filter_window_thresholds(args)[0] if args.filter_evidence_unit == "sentence_window" else None
        ),
        "filter_window_thresholds_path": (
            str(args.filter_window_thresholds_path) if args.filter_evidence_unit == "sentence_window" and args.filter_window_thresholds_path else None
        ),
        "filter_windowing_contract": (
            windowing_contract(args.filter_window_context_sentences)
            if args.filter_evidence_unit in {"sentence_window", "document_transformer"}
            else None
        ),
        "preanswer_text_hidden": (
            {
                "prompt_version": PREANSWER_PROMPT_VERSION,
                "backbone_path": str(args.hidden_filter_backbone_path),
                "state_model_path": str(args.llm_model_path),
                "hidden_layer": args.hidden_feature_layer,
                "hidden_batch_size": args.hidden_feature_batch_size,
                "hidden_max_input_tokens": args.hidden_feature_max_input_tokens,
                "hidden_dtype": args.hidden_feature_dtype,
                "hidden_attn_implementation": args.hidden_feature_attn_implementation,
                "helpful_threshold": args.hidden_filter_helpful_threshold,
                "hidden_inputs": ["h0", "delta_h=hD-h0"],
                "forbidden_inputs": ["gold_answer", "c", "projection_score", "answer_transition"],
            }
            if args.filter_evidence_unit == "preanswer_text_hidden"
            else None
        ),
        "max_doc_chars": args.max_doc_chars,
        "document_packing": args.document_packing,
        "document_token_safety_margin": args.document_token_safety_margin,
        "query_max_length": args.query_max_length,
        "dense_query_mode": args.dense_query_mode,
        "retrieval_query": (
            "original MCQ question with all options"
            if args.dense_query_mode == "initial"
            else "no-RAG rationale query including its answer conclusion"
        ),
        "rerank_query": "original MCQ question with all options",
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    if args.case == "no_rag":
        if args.answer_decision_mode == "constrained_choice":
            empty_docs: list[list[RetrievedDocument]] = [[] for _ in samples]
            results, details = generate_rag_answers(
                args,
                samples,
                empty_docs,
                empty_docs,
                empty_docs,
                ["" for _ in samples],
                align_no_rag_artifacts(samples, artifacts),
            )
        else:
            results, details = no_rag_results(args, samples, artifacts)
        write_outputs(args, results, details, config)
        return

    query_texts = [
        dense_query_text(args, sample, row)
        for dataset in args.datasets
        for sample, row in zip(grouped[dataset], artifacts[dataset])
    ]
    if args.candidate_cache_source_path is not None:
        # An explicit completed candidate cache is validated below against
        # every sample key and exact dense-query text before its requested
        # subset is materialised. Query vectors are never consumed on this
        # path, so recomputing MedCPT embeddings first wastes time/VRAM and can
        # OOM even though neither retrieval nor reranking is needed.
        query_vectors = np.empty((len(samples), 0), dtype="float32")
        logging.info(
            "Explicit candidate-cache source supplied; skipping dense-query "
            "embedding, retrieval, and reranking before exact cache validation."
        )
    else:
        embeddings = ensure_dense_query_embeddings(args, grouped, artifacts)
        query_vectors = np.concatenate([embeddings[dataset] for dataset in args.datasets], axis=0)
    initial_docs, reranked_docs, cache_dir = ensure_candidates(
        args,
        samples,
        np.ascontiguousarray(query_vectors, dtype="float32"),
        query_texts,
    )
    config["candidate_cache_dir"] = str(cache_dir)
    if args.candidate_cache_only:
        write_json(
            cache_dir / "candidate_cache_build_report.json",
            {
                **config,
                "candidate_cache_only": True,
                "candidate_rows": len(samples),
                "reranked_documents_per_question": args.rerank_top_k,
                "status": "complete",
            },
        )
        logging.info(
            "Candidate cache complete: %s rows, %s reranked documents/question. "
            "No answer LLM generation was run. Cache: %s",
            len(samples),
            args.rerank_top_k,
            cache_dir,
        )
        return

    no_rag_by_key = align_no_rag_artifacts(samples, artifacts)
    if args.case == "rerank_rag":
        if args.paper_balanced_top_k is not None:
            initial_docs, reranked_docs = project_paper_balanced_candidates(
                initial_docs,
                reranked_docs,
                sources=args.sources,
                top_k=args.paper_balanced_top_k,
            )
            context_docs = reranked_docs
        else:
            generation_top_k = args.generation_top_k or args.rerank_top_k
            context_docs = [docs[:generation_top_k] for docs in reranked_docs]
    elif args.case == "filter_rag":
        reranked_docs, filter_cache_dir = ensure_filter_scores(
            args,
            samples,
            reranked_docs,
            cache_dir,
        )
        config["filter_score_cache_dir"] = str(filter_cache_dir)
        if args.filter_cache_only:
            write_json(
                filter_cache_dir / "filter_cache_build_report.json",
                {
                    **config,
                    "filter_cache_only": True,
                    "filter_rows": len(samples),
                    "filter_documents": sum(len(docs) for docs in reranked_docs),
                    "status": "complete",
                },
            )
            logging.info(
                "Filter cache complete. No answer LLM generation was run. Cache: %s",
                filter_cache_dir,
            )
            return
        if args.paper_balanced_top_k is not None:
            initial_docs, reranked_docs = project_paper_balanced_candidates(
                initial_docs,
                reranked_docs,
                sources=args.sources,
                top_k=args.paper_balanced_top_k,
            )
            context_docs = [
                materialize_filter_generation_context(args, docs)
                for docs in reranked_docs
            ]
        else:
            eligible_top_k = args.filter_rerank_top_k or args.rerank_top_k
            context_docs = [
                materialize_filter_generation_context(args, docs[:eligible_top_k])
                for docs in reranked_docs
            ]
    else:
        if args.paper_balanced_top_k is not None:
            initial_docs, reranked_docs = project_paper_balanced_candidates(
                initial_docs,
                reranked_docs,
                sources=args.sources,
                top_k=args.paper_balanced_top_k,
            )
            # Apply labels only after reconstructing the exact 4k -> Top-k
            # condition. This lets the oracle annotate the union of documents
            # that can actually enter generation instead of wasting one-doc
            # generations on all 128 master-cache candidates.
            reranked_docs = apply_oracle_labels(args, samples, reranked_docs)
            context_docs = [
                [copy.copy(document) for document in docs if document.filter_prediction == "helpful"]
                for docs in reranked_docs
            ]
        else:
            reranked_docs = apply_oracle_labels(args, samples, reranked_docs)
            eligible_top_k = args.filter_rerank_top_k or args.rerank_top_k
            context_docs = [
                [copy.copy(document) for document in docs[:eligible_top_k] if document.filter_prediction == "helpful"]
                for docs in reranked_docs
            ]
    results, details = generate_rag_answers(
        args,
        samples,
        context_docs,
        initial_docs,
        reranked_docs,
        query_texts,
        no_rag_by_key,
    )
    write_outputs(args, results, details, config)


if __name__ == "__main__":
    main()

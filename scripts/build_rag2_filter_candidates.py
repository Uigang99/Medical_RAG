from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path
from medrag.core import BenchmarkSample, RetrievedDocument
from medrag.io_utils import read_json, write_json
from medrag.progress import StageProgress
from medrag.query_embeddings import QueryEmbeddingStore
from medrag.retrieval.faiss_retriever import FaissMedCPTRetriever
from medrag.reranking.medcpt_cross_encoder import MedCPTCrossEncoderReranker
from medrag.rag2_mcq import format_question


DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "datasets" / "benchmark"
DEFAULT_QUERY_CACHE_ROOT = (
    PROJECT_ROOT
    / "databases"
    / "query_embeddings"
    / "medcpt_query_encoder"
    / "rag2_llama3_8b_paper_answer_format_v2"
)
DEFAULT_VECTOR_DB_ROOT = PROJECT_ROOT / "databases" / "vector_db" / "medcpt_article_encoder"
DEFAULT_CROSS_ENCODER = WORKSPACE_ROOT / "models" / "MedCPT-Cross-Encoder"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "filtering"
    / "rag2"
    / "llama3_8b_paper_answer_format_v2"
    / "candidates"
)
PAPER4_SOURCES = ["pubmed", "pmc", "cpg", "textbooks"]
QUERY_ENCODING_PROTOCOL_VERSION = "rag2_released_medcpt_query_cls_truncate512_v1"
CANONICAL_RATIONALE_QUERY_FIELD = "reparsed(no_rag_generation).rationale_query"
RETRIEVAL_QUERY_CANONICALIZATION_VERSION = "rationale_only_plus_single_canonical_answer_v3"
PAPER_EXACT_RATIONALE_QUERY_FIELD = "parsed.rationale_query(raw_visible_response)"
PAPER_EXACT_RETRIEVAL_QUERY_CANONICALIZATION_VERSION = "raw_visible_response_no_rewrite_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG2-style filtering candidates with balanced retrieval and MedCPT reranking."
    )
    parser.add_argument("--dataset", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--query-cache-root", type=Path, default=DEFAULT_QUERY_CACHE_ROOT)
    parser.add_argument(
        "--query-cache-dir",
        type=Path,
        default=None,
        help="Optional explicit query cache. RAG2 rationale caches must include query_text in metadata.jsonl.",
    )
    parser.add_argument("--vector-db-root", type=Path, default=DEFAULT_VECTOR_DB_ROOT)
    parser.add_argument("--cross-encoder-path", type=Path, default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-file", default="candidates_top10.jsonl")
    parser.add_argument("--sources", nargs="+", default=PAPER4_SOURCES)
    parser.add_argument("--per-source-top-k", type=int, default=10)
    parser.add_argument("--candidate-pool-top-k", type=int, default=40)
    parser.add_argument(
        "--candidate-layout",
        choices=["source_balanced", "released_pubmed_groups"],
        default="source_balanced",
        help=(
            "source_balanced: retain top-k from each logical corpus before reranking. "
            "released_pubmed_groups: reproduce the public RAG2 README literal candidate layout: "
            "retrieve top-k from every PubMed physical-shard group, plus top-k from PMC/CPG/Textbook."
        ),
    )
    parser.add_argument(
        "--pubmed-shards-per-group",
        type=int,
        default=10,
        help=(
            "Physical PubMed FAISS shards in one release-style group. With the RAG_Square 40-shard "
            "PubMed index and the default 10, this yields four PubMed groups and 70 total candidates."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--retrieval-search-mode",
        choices=[
            "faiss_gpu_source_loop",
            "faiss_gpu_source_sequential",
            "source_loop",
            "faiss_gpu",
            "logical_shards",
            "gpu_stream",
        ],
        default="faiss_gpu_source_sequential",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=2048)
    parser.add_argument(
        "--retrieval-progress-chunk-size",
        type=int,
        default=0,
        help="Use 0 for one FAISS-GPU search per retrieval batch. Small values give smoother progress but slow exact search.",
    )
    parser.add_argument("--rerank-batch-size", type=int, default=1024)
    parser.add_argument("--cross-encoder-max-length", type=int, default=512)
    parser.add_argument("--cross-encoder-attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--keep-faiss-indexes-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faiss-mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faiss-shard-threaded", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-gpu-use-float16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faiss-gpu-add-batch-size", type=int, default=1000000)
    parser.add_argument("--faiss-gpu-temp-memory-mb", type=int, default=2048)
    parser.add_argument("--metadata-row-cache-size", type=int, default=50000)
    parser.add_argument("--gpu-search-chunk-size", type=int, default=500000)
    parser.add_argument("--gpu-search-device", default="auto")
    parser.add_argument("--gpu-search-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--sample-id-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited sample IDs.  Restrict candidate construction to these questions "
            "before applying --start/--end/--limit; useful for a held-out system-selection split."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-initial-doc-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def expected_candidate_pool_size(vector_db_root: Path, args: argparse.Namespace) -> int:
    """Return the exact pre-rerank candidate count for the selected layout.

    ``released_pubmed_groups`` deliberately follows the public RAG2 README's
    physical PubMed grouping.  It is kept separate from ``source_balanced``:
    the latter is the paper-text interpretation (equal quota per logical
    corpus), whereas the former gives PubMed one quota per physical group.
    """
    per_bucket_top_k = int(args.per_source_top_k)
    if getattr(args, "candidate_layout", "source_balanced") == "source_balanced":
        return per_bucket_top_k * len(args.sources)

    if "pubmed" not in args.sources:
        raise ValueError("released_pubmed_groups requires 'pubmed' in --sources.")
    shards_per_group = int(getattr(args, "pubmed_shards_per_group", 10))
    if shards_per_group <= 0:
        raise ValueError("--pubmed-shards-per-group must be positive.")

    manifest_path = vector_db_root / "pubmed" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing PubMed vector DB manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    physical_shard_count = len((manifest.get("index") or {}).get("shards") or [])
    if physical_shard_count <= 0:
        physical_shard_count = 1
    pubmed_group_count = (physical_shard_count + shards_per_group - 1) // shards_per_group
    return per_bucket_top_k * (len(args.sources) - 1 + pubmed_group_count)


def validate_configuration(args: argparse.Namespace) -> None:
    if len(set(args.sources)) != len(args.sources):
        raise ValueError(f"Duplicate corpus sources are not allowed: {args.sources}")
    if args.per_source_top_k <= 0 or args.candidate_pool_top_k <= 0 or args.top_k <= 0:
        raise ValueError("Retrieval and reranking top-k values must be positive.")
    if args.pubmed_shards_per_group <= 0:
        raise ValueError("--pubmed-shards-per-group must be positive.")
    expected_pool = expected_candidate_pool_size(args.vector_db_root, args)
    if args.retrieval_search_mode == "faiss_gpu_source_sequential" and args.candidate_pool_top_k != expected_pool:
        raise ValueError(
            "Candidate-pool size does not match the selected retrieval layout: "
            f"candidate_pool_top_k={args.candidate_pool_top_k}, expected={expected_pool} "
            f"(layout={args.candidate_layout}, per_source_top_k={args.per_source_top_k})."
        )
    if args.top_k > args.candidate_pool_top_k:
        raise ValueError("--top-k cannot exceed --candidate-pool-top-k.")
    if args.retrieval_batch_size <= 0 or args.rerank_batch_size <= 0:
        raise ValueError("Retrieval and reranking batch sizes must be positive.")


def validate_query_cache_manifest(manifest: dict[str, Any], dataset: str, split: str) -> None:
    paper_exact = manifest.get("prompt_profile") == "paper_exact"
    expected_query_field = (
        PAPER_EXACT_RATIONALE_QUERY_FIELD if paper_exact else CANONICAL_RATIONALE_QUERY_FIELD
    )
    expected_canonicalization = (
        PAPER_EXACT_RETRIEVAL_QUERY_CANONICALIZATION_VERSION
        if paper_exact
        else RETRIEVAL_QUERY_CANONICALIZATION_VERSION
    )
    expected = {
        "dataset": dataset,
        "split": split,
        "query_field": expected_query_field,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if manifest.get("query_includes_answer_conclusion") is not True:
        mismatches["query_includes_answer_conclusion"] = {
            "expected": True,
            "actual": manifest.get("query_includes_answer_conclusion"),
        }
    allowed_quality_policies = {"technical", "conservative"} if paper_exact else {"conservative"}
    if manifest.get("quality_policy") not in allowed_quality_policies:
        mismatches["quality_policy"] = {
            "expected": sorted(allowed_quality_policies),
            "actual": manifest.get("quality_policy"),
        }
    if manifest.get("retrieval_query_canonicalization_version") != expected_canonicalization:
        mismatches["retrieval_query_canonicalization_version"] = {
            "expected": expected_canonicalization,
            "actual": manifest.get("retrieval_query_canonicalization_version"),
        }
    if manifest.get("query_encoding_protocol_version") != QUERY_ENCODING_PROTOCOL_VERSION:
        mismatches["query_encoding_protocol_version"] = {
            "expected": QUERY_ENCODING_PROTOCOL_VERSION,
            "actual": manifest.get("query_encoding_protocol_version"),
        }
    for key in ("prompt_version", "ppl_scope_version", "generation_policy_version"):
        if not str(manifest.get(key) or "").strip():
            mismatches[key] = {
                "expected": "non-empty value inherited from no-RAG manifest",
                "actual": manifest.get(key),
            }
    if bool((manifest.get("model") or {}).get("normalize")):
        mismatches["model.normalize"] = {
            "expected": False,
            "actual": (manifest.get("model") or {}).get("normalize"),
        }
    query_model = manifest.get("model") or {}
    expected_model = {
        "name": "MedCPT-Query-Encoder",
        "embedding_field": "last_hidden_state[:, 0, :]",
        "max_length": 512,
    }
    for key, value in expected_model.items():
        if query_model.get(key) != value:
            mismatches[f"model.{key}"] = {"expected": value, "actual": query_model.get(key)}
    if int(manifest.get("dimension", -1)) != 768:
        mismatches["dimension"] = {"expected": 768, "actual": manifest.get("dimension")}
    if mismatches:
        raise ValueError(f"Incompatible RAG2 rationale query cache: {mismatches}")


def validate_vector_db_contract(vector_db_root: Path, sources: list[str]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for source in sources:
        manifest_path = vector_db_root / source / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing vector DB manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        model = manifest.get("model") or {}
        index = manifest.get("index") or {}
        expected = {
            "source": (manifest.get("source"), source),
            "model.name": (model.get("name"), "MedCPT-Article-Encoder"),
            "model.embedding_field": (model.get("embedding_field"), "last_hidden_state[:, 0, :]"),
            "model.max_length": (model.get("max_length"), 512),
            "model.normalize": (model.get("normalize"), False),
            "index.index_type": (index.get("index_type"), "IndexFlatIP"),
            "index.metric": (index.get("metric"), "inner_product"),
            "index.dimension": (index.get("dimension"), 768),
        }
        mismatches = {
            key: {"expected": expected_value, "actual": actual}
            for key, (actual, expected_value) in expected.items()
            if actual != expected_value
        }
        if manifest.get("type") not in {"source_vector_db_flat", "source_vector_db"}:
            mismatches["type"] = {
                "expected": "source_vector_db_flat or source_vector_db",
                "actual": manifest.get("type"),
            }
        if mismatches:
            raise ValueError(f"Incompatible vector DB contract for source={source}: {mismatches}")
        contracts[source] = {
            "manifest_path": str(manifest_path),
            "rows": int(manifest.get("rows", -1)),
            "encoder": model.get("name"),
            "embedding_field": model.get("embedding_field"),
            "max_length": model.get("max_length"),
            "normalize": model.get("normalize"),
            "index_type": index.get("index_type"),
            "metric": index.get("metric"),
            "dimension": index.get("dimension"),
        }
    return contracts


def source_counts(docs: list[RetrievedDocument]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc.source] = counts.get(doc.source, 0) + 1
    return counts


def validate_balanced_docs(
    document_lists: list[list[RetrievedDocument]],
    sources: list[str],
    per_source_top_k: int,
) -> None:
    expected = {source: per_source_top_k for source in sources}
    for batch_index, docs in enumerate(document_lists):
        actual = source_counts(docs)
        if actual != expected:
            raise RuntimeError(
                f"Balanced retrieval invariant failed at batch row {batch_index}: "
                f"expected={expected} actual={actual}"
            )


def validate_layout_docs(
    document_lists: list[list[RetrievedDocument]],
    args: argparse.Namespace,
) -> None:
    """Validate the pre-rerank candidate quota without relabeling the layout."""
    layout = getattr(args, "candidate_layout", "source_balanced")
    if layout == "source_balanced":
        validate_balanced_docs(document_lists, args.sources, args.per_source_top_k)
        return

    expected_total = int(args.candidate_pool_top_k)
    per_bucket = int(args.per_source_top_k)
    non_pubmed_sources = [source for source in args.sources if source != "pubmed"]
    for batch_index, docs in enumerate(document_lists):
        actual_sources = source_counts(docs)
        if len(docs) != expected_total:
            raise RuntimeError(
                f"Release-style candidate count failed at batch row {batch_index}: "
                f"expected={expected_total} actual={len(docs)}"
            )
        for source in non_pubmed_sources:
            if actual_sources.get(source, 0) != per_bucket:
                raise RuntimeError(
                    f"Release-style non-PubMed quota failed at batch row {batch_index}: "
                    f"source={source} expected={per_bucket} actual={actual_sources.get(source, 0)}"
                )
        bucket_counts: dict[str, int] = {}
        for doc in docs:
            bucket = str(doc.metadata.get("retrieval_bucket") or "")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        expected_buckets = [bucket for bucket in bucket_counts if bucket.startswith("pubmed_group_")]
        if not expected_buckets or any(bucket_counts[bucket] != per_bucket for bucket in expected_buckets):
            raise RuntimeError(
                f"Release-style PubMed group quota failed at batch row {batch_index}: "
                f"counts={bucket_counts} per_bucket={per_bucket}"
            )


def original_mcq_query(sample: BenchmarkSample) -> str:
    """Return the exact initial MCQ text used by the rationale-generation prompt."""
    query = format_question(sample.raw).strip()
    if not query:
        raise ValueError(f"Empty original MCQ query: sample_id={sample.id}")
    if not sample.options:
        raise ValueError(f"Original MCQ query has no options: sample_id={sample.id}")
    return query


def doc_to_dict(doc: RetrievedDocument, include_text: bool = True) -> dict[str, Any]:
    row = doc.to_dict(include_text=include_text)
    row["stable_id"] = doc.stable_id
    row["source_retrieval_rank"] = doc.metadata.get("source_retrieval_rank")
    return row


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = str(row.get("sample_id") or "")
            if key:
                done.add(key)
    return done


def select_range(
    samples: list[BenchmarkSample],
    query_vectors: np.ndarray,
    query_texts: list[str],
    args: argparse.Namespace,
) -> tuple[list[BenchmarkSample], np.ndarray, list[str]]:
    if args.sample_id_file is not None:
        if not args.sample_id_file.is_file():
            raise FileNotFoundError(f"Missing --sample-id-file: {args.sample_id_file}")
        requested_ids = {
            line.strip()
            for line in args.sample_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not requested_ids:
            raise ValueError(f"--sample-id-file is empty: {args.sample_id_file}")
        selected_indices = [
            index for index, sample in enumerate(samples) if sample.id in requested_ids
        ]
        found_ids = {samples[index].id for index in selected_indices}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise RuntimeError(
                "Some requested sample IDs are absent from the usable rationale-query cache: "
                f"missing={len(missing_ids)} first={sorted(missing_ids)[:5]}"
            )
        samples = [samples[index] for index in selected_indices]
        query_vectors = query_vectors[selected_indices]
        query_texts = [query_texts[index] for index in selected_indices]

    start = max(0, int(args.start))
    end = len(samples) if args.end is None else min(len(samples), int(args.end))
    if start >= end:
        return [], np.empty((0, query_vectors.shape[1]), dtype="float32"), []
    samples = samples[start:end]
    query_vectors = query_vectors[start:end]
    query_texts = query_texts[start:end]
    if args.limit is not None:
        samples = samples[: args.limit]
        query_vectors = query_vectors[: args.limit]
        query_texts = query_texts[: args.limit]
    return samples, np.asarray(query_vectors, dtype="float32"), query_texts


def load_cached_queries(
    cache_dir: Path,
    benchmark_samples: list[BenchmarkSample],
    manifest: dict[str, Any],
) -> tuple[list[BenchmarkSample], list[str]]:
    metadata_path = cache_dir / str(manifest.get("metadata_path") or "metadata.jsonl")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing query cache metadata: {metadata_path}")
    metadata: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                metadata.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed query metadata: {metadata_path}:{line_no}") from exc
    expected_cache_rows = int(manifest.get("rows", -1))
    if len(metadata) != expected_cache_rows:
        raise ValueError(f"Query metadata rows mismatch: metadata={len(metadata)} manifest={expected_cache_rows}")

    samples_by_row_idx = {sample.row_idx: sample for sample in benchmark_samples}
    if len(samples_by_row_idx) != len(benchmark_samples):
        raise ValueError("Benchmark contains duplicate row_idx values.")

    selected_samples: list[BenchmarkSample] = []
    query_texts: list[str] = []
    seen_row_indices: set[int] = set()
    for expected_index, item in enumerate(metadata):
        cache_index = int(item.get("cache_index", expected_index))
        row_idx = int(item.get("row_idx", -1))
        sample_id = str(item.get("sample_id") or "")
        query_text = str(item.get("query_text") or "").strip()
        sample = samples_by_row_idx.get(row_idx)
        if sample is None:
            raise ValueError(f"Query cache points to missing benchmark row_idx={row_idx} at cache_index={expected_index}")
        if row_idx in seen_row_indices:
            raise ValueError(f"Duplicate benchmark row_idx={row_idx} in query cache metadata")
        if cache_index != expected_index or sample_id != sample.id:
            raise ValueError(
                f"Query cache alignment error at index={expected_index}: "
                f"cache_index={cache_index} row_idx={row_idx} sample_id={sample_id}/{sample.id}"
            )
        if not query_text:
            raise ValueError(f"Empty cached query at index={expected_index} sample_id={sample.id}")
        seen_row_indices.add(row_idx)
        selected_samples.append(sample)
        query_texts.append(query_text)
    return selected_samples, query_texts


def retrieval_bucket_specs(
    retriever: FaissMedCPTRetriever,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Describe the independently retrieved candidate buckets.

    For the paper-text ``source_balanced`` layout, a logical corpus is one
    bucket even when it is physically sharded.  For the public-release layout,
    PubMed is intentionally split into consecutive groups of physical shards;
    this reproduces the README's 10/10/10/8-style candidate expansion without
    changing any other corpus's quota.
    """
    layout = getattr(args, "candidate_layout", "source_balanced")
    specs: list[dict[str, Any]] = []
    for source in args.sources:
        source_store = retriever._indexes[source]
        physical_shard_count = len(source_store._shards) if source_store._is_sharded else 1
        if layout != "released_pubmed_groups" or source != "pubmed":
            specs.append(
                {
                    "bucket_id": source,
                    "source": source,
                    "physical_shard_ids": set(range(physical_shard_count)),
                }
            )
            continue

        shards_per_group = int(getattr(args, "pubmed_shards_per_group", 10))
        for group_start in range(0, physical_shard_count, shards_per_group):
            group_end = min(group_start + shards_per_group, physical_shard_count)
            group_idx = group_start // shards_per_group
            specs.append(
                {
                    "bucket_id": f"pubmed_group_{group_idx:02d}",
                    "source": source,
                    "physical_shard_ids": set(range(group_start, group_end)),
                }
            )
    return specs


def build_source_sequential_hits(
    retriever: FaissMedCPTRetriever,
    query_vectors: np.ndarray,
    sample_indices: list[int],
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], list[str]]:
    per_source_top_k = int(args.per_source_top_k)
    if per_source_top_k <= 0:
        raise ValueError("--per-source-top-k must be positive for faiss_gpu_source_sequential.")
    if not sample_indices:
        return {}, {}, []

    bucket_specs = retrieval_bucket_specs(retriever, args)
    bucket_sources = {str(spec["bucket_id"]): str(spec["source"]) for spec in bucket_specs}
    bucket_order = [str(spec["bucket_id"]) for spec in bucket_specs]
    hits_by_source: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    progress = StageProgress(total=len(sample_indices) * len(bucket_specs), desc="DenseRetrieval", enabled=True)
    selected_vectors = np.ascontiguousarray(query_vectors[sample_indices], dtype="float32")

    try:
        for source in args.sources:
            source_store = retriever._indexes[source]
            rows = int(source_store.rows)
            local_top_k = min(per_source_top_k, rows)
            source_specs = [spec for spec in bucket_specs if spec["source"] == source]
            bucket_state: dict[str, tuple[np.ndarray, np.ndarray]] = {
                str(spec["bucket_id"]): (
                    np.full((len(sample_indices), local_top_k), float("-inf"), dtype="float32"),
                    np.full((len(sample_indices), local_top_k), -1, dtype="int64"),
                )
                for spec in source_specs
            }
            bucket_for_physical_shard: dict[int, str] = {}
            for spec in source_specs:
                for physical_shard_id in spec["physical_shard_ids"]:
                    bucket_for_physical_shard[int(physical_shard_id)] = str(spec["bucket_id"])
            logging.info(
                "[%s] GPU-sequential retrieval: rows=%s top_k=%s buckets=%s layout=%s",
                source,
                rows,
                local_top_k,
                [str(spec["bucket_id"]) for spec in source_specs],
                getattr(args, "candidate_layout", "source_balanced"),
            )
            completed_rows = 0

            # RAG_Square keeps large sources in physical FAISS shards.  Search
            # one shard at a time on GPU.  A source-balanced bucket merges all
            # of its physical shards; a release-style PubMed bucket merges only
            # the consecutive shards assigned to that public-README group.
            for physical_shard_id, (shard_start, cpu_index) in enumerate(source_store.iter_physical_indexes()):
                shard_rows = int(cpu_index.ntotal)
                shard_top_k = min(local_top_k, shard_rows)
                bucket_id = bucket_for_physical_shard.get(physical_shard_id)
                if bucket_id is None:
                    raise RuntimeError(
                        f"No retrieval bucket assigned to {source} physical shard {physical_shard_id}."
                    )
                logging.info(
                    "[%s] physical shard %s: rows=%s; transferring to FAISS-GPU "
                    "(add_batch=%s, search_batch=%s)",
                    source,
                    physical_shard_id + 1,
                    shard_rows,
                    retriever.faiss_gpu_add_batch_size,
                    args.retrieval_batch_size,
                )
                resource = retriever._configure_gpu_resource()
                gpu_index = retriever._copy_cpu_index_to_gpu_flat(
                    f"{source}:start={shard_start}",
                    cpu_index,
                    resource,
                )
                try:
                    logging.info(
                        "[%s] physical shard %s: GPU index ready; exact-searching %s query vectors.",
                        source,
                        physical_shard_id + 1,
                        len(sample_indices),
                    )
                    for offset in range(0, len(sample_indices), args.retrieval_batch_size):
                        end = min(offset + args.retrieval_batch_size, len(sample_indices))
                        chunk = np.ascontiguousarray(selected_vectors[offset:end], dtype="float32")
                        shard_scores, shard_ids = gpu_index.search(chunk, shard_top_k)
                        shard_ids = shard_ids.astype("int64", copy=False)
                        valid = shard_ids >= 0
                        shard_ids = shard_ids.copy()
                        shard_ids[valid] += int(shard_start)

                        bucket_scores, bucket_ids = bucket_state[bucket_id]
                        combined_scores = np.concatenate([bucket_scores[offset:end], shard_scores], axis=1)
                        combined_ids = np.concatenate([bucket_ids[offset:end], shard_ids], axis=1)
                        keep = np.argsort(-combined_scores, axis=1, kind="stable")[:, :local_top_k]
                        bucket_scores[offset:end] = np.take_along_axis(combined_scores, keep, axis=1)
                        bucket_ids[offset:end] = np.take_along_axis(combined_ids, keep, axis=1)
                finally:
                    del gpu_index, resource
                    # ``cpu_index`` is the loop-local owner of a ~6 GiB PubMed
                    # shard.  Release it before clearing the source store so a
                    # no-cache run truly holds at most one CPU shard at once.
                    del cpu_index
                    gc.collect()
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if not retriever.keep_indexes_in_memory:
                        # Drop this CPU FAISS shard before opening the next
                        # one. This bounds host RAM for the 78M-chunk PubMed
                        # source even when FAISS cannot fully mmap IndexFlat.
                        source_store.unload_index()

                completed_rows += shard_rows
            for spec in source_specs:
                bucket_id = str(spec["bucket_id"])
                hits_by_source[bucket_id] = bucket_state[bucket_id]
                progress.update(len(sample_indices))
            logging.info("[%s] retrieval complete across physical shard(s): buckets=%s", source, len(source_specs))
    finally:
        progress.close()
    return hits_by_source, bucket_sources, bucket_order


def materialize_initial_docs(
    retriever: FaissMedCPTRetriever,
    source_hits: dict[str, tuple[np.ndarray, np.ndarray]],
    bucket_sources: dict[str, str],
    bucket_order: list[str],
    hit_positions: list[int],
    top_k: int,
) -> list[list[RetrievedDocument]]:
    batches: list[list[RetrievedDocument]] = []
    for hit_pos in hit_positions:
        hits: list[tuple[float, str, int, int]] = []
        for bucket_id in bucket_order:
            source = bucket_sources[bucket_id]
            scores, local_ids = source_hits[bucket_id]
            for bucket_rank, (score, local_id) in enumerate(
                zip(scores[hit_pos].tolist(), local_ids[hit_pos].tolist()), start=1
            ):
                if int(local_id) < 0:
                    continue
                hits.append((float(score), source, int(local_id), bucket_rank, bucket_id))
        hits.sort(key=lambda item: item[0], reverse=True)
        docs: list[RetrievedDocument] = []
        for score, source, local_id, bucket_rank, bucket_id in hits[:top_k]:
            doc = retriever._indexes[source].get_document(local_id=local_id, score=score)
            doc.metadata["source_retrieval_rank"] = bucket_rank
            doc.metadata["retrieval_bucket"] = bucket_id
            docs.append(doc)
        for rank, doc in enumerate(docs, start=1):
            doc.retrieval_rank = rank
        batches.append(docs)
    return batches


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    validate_configuration(args)

    benchmark_path = resolve_benchmark_path(args.benchmark_root, "mcq", args.collection, args.dataset, args.split)
    query_cache_dir = args.query_cache_dir or (args.query_cache_root / args.dataset / args.split)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset / args.split)
    benchmark_samples = load_benchmark_samples(benchmark_path, "mcq", args.collection, args.dataset, args.split)
    query_store = QueryEmbeddingStore(query_cache_dir)
    validate_query_cache_manifest(query_store.manifest, args.dataset, args.split)
    vector_db_contracts = validate_vector_db_contract(args.vector_db_root, args.sources)
    benchmark_rows = int(query_store.manifest.get("benchmark_rows", len(benchmark_samples)))
    if benchmark_rows != len(benchmark_samples):
        raise ValueError(
            f"Rationale query cache benchmark mismatch: cache={benchmark_rows} benchmark={len(benchmark_samples)}"
        )
    samples, query_texts = load_cached_queries(query_cache_dir, benchmark_samples, query_store.manifest)
    query_vectors = query_store.get_batch(list(range(len(samples))))
    excluded_query_rows = len(benchmark_samples) - len(samples)
    logging.info(
        "Rationale query cache: benchmark=%s usable=%s excluded=%s",
        len(benchmark_samples),
        len(samples),
        excluded_query_rows,
    )
    samples, query_vectors, query_texts = select_range(samples, query_vectors, query_texts, args)
    if not samples:
        raise RuntimeError("No samples selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file
    done = load_done(output_path) if args.resume else set()
    missing_indices = [idx for idx, sample in enumerate(samples) if sample.id not in done]
    logging.info(
        "Selected %s samples from %s. Done=%s missing=%s output=%s",
        len(samples),
        benchmark_path,
        len(samples) - len(missing_indices),
        len(missing_indices),
        output_path,
    )

    manifest = {
        "type": "rag2_filter_candidate_dataset",
        "created_or_updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split": args.split,
        "collection": args.collection,
        "benchmark_path": str(benchmark_path),
        "benchmark_rows": len(benchmark_samples),
        "query_rows": int(query_store.manifest.get("rows", len(samples))),
        "excluded_query_rows": excluded_query_rows,
        "query_excluded_rows_path": query_store.manifest.get("excluded_rows_path"),
        "query_cache_dir": str(query_cache_dir),
        "query_mode": query_store.manifest.get("query_field") or query_store.manifest.get("mcq_query_mode"),
        "retrieval_query_mode": "no_rag_rationale_with_answer_conclusion",
        "rerank_query_mode": "original_mcq_question_with_all_options",
        "query_prompt_version": query_store.manifest.get("prompt_version"),
        "query_generation_policy_version": query_store.manifest.get("generation_policy_version"),
        "query_encoding_protocol_version": query_store.manifest.get("query_encoding_protocol_version"),
        "query_includes_answer_conclusion": query_store.manifest.get("query_includes_answer_conclusion"),
        "output_path": str(output_path),
        "sources": args.sources,
        "per_source_top_k": args.per_source_top_k,
        "candidate_pool_top_k": args.candidate_pool_top_k,
        "candidate_layout": args.candidate_layout,
        "pubmed_shards_per_group": args.pubmed_shards_per_group,
        "top_k": args.top_k,
        "retrieval_search_mode": args.retrieval_search_mode,
        "sample_id_file": str(args.sample_id_file) if args.sample_id_file is not None else None,
        "selected_question_count": len(samples),
        "candidate_pool_policy": {
            "mode": args.candidate_layout,
            "source_top_k": {source: args.per_source_top_k for source in args.sources},
            "pool_size": args.candidate_pool_top_k,
            "pubmed_shards_per_group": args.pubmed_shards_per_group,
            "description": (
                "Equal Top-k from each logical corpus."
                if args.candidate_layout == "source_balanced"
                else "Public RAG2 README literal layout: Top-k from each consecutive PubMed physical-shard "
                "group, plus Top-k from each remaining corpus."
            ),
        },
        "query_embedding": query_store.manifest.get("model"),
        "vector_db_contracts": vector_db_contracts,
        "reranker": {
            "path": str(args.cross_encoder_path),
            "max_length": args.cross_encoder_max_length,
            "query": "original MCQ question with all options; no rationale and no gold answer",
        },
        "notes": (
            "Dense retrieval embeds only the complete no-RAG rationale response, including its answer conclusion. "
            "The MedCPT cross-encoder reranks each document against the original MCQ question and all options. "
            "initial_documents contain the complete balanced dense pool; candidate_documents contain the "
            "cross-encoder Top-k with full text for pseudo labeling."
        ),
    }
    manifest_path = output_dir / "candidate_manifest.json"
    if args.resume and output_path.exists() and manifest_path.exists():
        previous = read_json(manifest_path)
        immutable_fields = [
            "dataset",
            "split",
            "query_cache_dir",
            "query_prompt_version",
            "query_generation_policy_version",
            "query_encoding_protocol_version",
            "retrieval_query_mode",
            "rerank_query_mode",
            "sources",
            "per_source_top_k",
            "candidate_pool_top_k",
            "candidate_layout",
            "pubmed_shards_per_group",
            "top_k",
            "retrieval_search_mode",
            "sample_id_file",
        ]
        conflicts = {
            key: {"previous": previous.get(key), "current": manifest.get(key)}
            for key in immutable_fields
            if previous.get(key) != manifest.get(key)
        }
        if conflicts:
            raise RuntimeError(f"Cannot resume an incompatible candidate build: {conflicts}")
    write_json(manifest_path, manifest)

    if not missing_indices:
        logging.info("All selected samples already have candidates.")
        return

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
    try:
        with output_path.open("a", encoding="utf-8", buffering=16 * 1024 * 1024) as out:
            if args.retrieval_search_mode == "faiss_gpu_source_sequential":
                source_hits, bucket_sources, bucket_order = build_source_sequential_hits(
                    retriever, query_vectors, missing_indices, args
                )
                reranker = MedCPTCrossEncoderReranker(
                    model_path=args.cross_encoder_path,
                    batch_size=args.rerank_batch_size,
                    max_length=args.cross_encoder_max_length,
                    attn_implementation=args.cross_encoder_attn_implementation,
                )
                for offset in range(0, len(missing_indices), args.retrieval_batch_size):
                    hit_positions = list(range(offset, min(offset + args.retrieval_batch_size, len(missing_indices))))
                    batch_indices = missing_indices[offset : offset + len(hit_positions)]
                    initial_docs = materialize_initial_docs(
                        retriever=retriever,
                        source_hits=source_hits,
                        bucket_sources=bucket_sources,
                        bucket_order=bucket_order,
                        hit_positions=hit_positions,
                        top_k=args.candidate_pool_top_k,
                    )
                    validate_layout_docs(initial_docs, args)
                    batch_samples = [samples[idx] for idx in batch_indices]
                    retrieval_queries = [query_texts[idx] for idx in batch_indices]
                    rerank_queries = [original_mcq_query(sample) for sample in batch_samples]
                    reranked_docs = reranker.rerank_batch(rerank_queries, initial_docs, top_k=args.top_k)

                    for sample, retrieval_query, rerank_query, docs, final_docs in zip(
                        batch_samples, retrieval_queries, rerank_queries, initial_docs, reranked_docs
                    ):
                        row = {
                            "sample_id": sample.id,
                            "row_idx": sample.row_idx,
                            "dataset": sample.dataset,
                            "split": sample.split,
                            "question": sample.question,
                            "options": sample.options,
                            "answers": sample.answers,
                            "answer": sample.answer,
                            "query_text": retrieval_query,
                            "retrieval_query_text": retrieval_query,
                            "rerank_query_text": rerank_query,
                            "retrieval_query_mode": "no_rag_rationale_with_answer_conclusion",
                            "rerank_query_mode": "original_mcq_question_with_all_options",
                            "query_prompt_version": query_store.manifest.get("prompt_version"),
                            "initial_source_counts": source_counts(docs),
                            "initial_documents": [
                                doc_to_dict(doc, include_text=args.include_initial_doc_text) for doc in docs
                            ],
                            "candidate_documents": [doc_to_dict(doc, include_text=True) for doc in final_docs],
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.update(len(batch_indices))
            else:
                reranker = MedCPTCrossEncoderReranker(
                    model_path=args.cross_encoder_path,
                    batch_size=args.rerank_batch_size,
                    max_length=args.cross_encoder_max_length,
                    attn_implementation=args.cross_encoder_attn_implementation,
                )
                for offset in range(0, len(missing_indices), args.retrieval_batch_size):
                    batch_indices = missing_indices[offset : offset + args.retrieval_batch_size]
                    batch_vectors = query_vectors[batch_indices]
                    completed = 0

                    def on_retrieved(n: int) -> None:
                        nonlocal completed
                        completed += max(0, int(n))

                    initial_docs = retriever.retrieve_batch(
                        batch_vectors,
                        top_k=args.candidate_pool_top_k,
                        progress_callback=on_retrieved,
                        progress_chunk_size=args.retrieval_progress_chunk_size,
                    )
                    if args.candidate_pool_top_k == args.per_source_top_k * len(args.sources):
                        validate_balanced_docs(initial_docs, args.sources, args.per_source_top_k)
                    batch_samples = [samples[idx] for idx in batch_indices]
                    retrieval_queries = [query_texts[idx] for idx in batch_indices]
                    rerank_queries = [original_mcq_query(sample) for sample in batch_samples]
                    reranked_docs = reranker.rerank_batch(rerank_queries, initial_docs, top_k=args.top_k)

                    for sample, retrieval_query, rerank_query, docs, final_docs in zip(
                        batch_samples, retrieval_queries, rerank_queries, initial_docs, reranked_docs
                    ):
                        row = {
                            "sample_id": sample.id,
                            "row_idx": sample.row_idx,
                            "dataset": sample.dataset,
                            "split": sample.split,
                            "question": sample.question,
                            "options": sample.options,
                            "answers": sample.answers,
                            "answer": sample.answer,
                            "query_text": retrieval_query,
                            "retrieval_query_text": retrieval_query,
                            "rerank_query_text": rerank_query,
                            "retrieval_query_mode": "no_rag_rationale_with_answer_conclusion",
                            "rerank_query_mode": "original_mcq_question_with_all_options",
                            "query_prompt_version": query_store.manifest.get("prompt_version"),
                            "initial_source_counts": source_counts(docs),
                            "initial_documents": [
                                doc_to_dict(doc, include_text=args.include_initial_doc_text) for doc in docs
                            ],
                            "candidate_documents": [doc_to_dict(doc, include_text=True) for doc in final_docs],
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.update(len(batch_indices))
                    if completed < len(batch_indices):
                        logging.debug("Retriever callback completed %s/%s rows in batch.", completed, len(batch_indices))
    finally:
        progress.close()
        if reranker is not None:
            reranker.close()
        retriever.close()

    logging.info("Candidate build complete: %s", output_path)


if __name__ == "__main__":
    main()

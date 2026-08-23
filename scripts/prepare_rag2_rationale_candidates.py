from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
EMBED_SCRIPT = PROJECT_ROOT / "scripts" / "precompute_rag2_rationale_embeddings.py"
CANDIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "build_rag2_filter_candidates.py"

DEFAULT_NO_RAG_ROOT = (
    PROJECT_ROOT / "datasets" / "filtering" / "rag2" / "llama3_8b_paper_answer_format_v2"
)
DEFAULT_QUERY_CACHE_ROOT = (
    PROJECT_ROOT
    / "databases"
    / "query_embeddings"
    / "medcpt_query_encoder"
    / "rag2_llama3_8b_paper_answer_format_v2"
)
DEFAULT_CANDIDATE_ROOT = DEFAULT_NO_RAG_ROOT / "candidates"
DEFAULT_VECTOR_DB_ROOT = PROJECT_ROOT / "databases" / "vector_db" / "medcpt_article_encoder"
DEFAULT_QUERY_ENCODER = WORKSPACE_ROOT / "models" / "MedCPT-Query-Encoder"
DEFAULT_CROSS_ENCODER = WORKSPACE_ROOT / "models" / "MedCPT-Cross-Encoder"
DEFAULT_SOURCES = ["pubmed", "pmc", "cpg", "textbooks"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare RAG2 rationale candidates end to end: exclude invalid no-RAG rows, embed complete rationale "
            "responses with the MedCPT query encoder, retrieve an equal number from each corpus, and rerank with "
            "the original MCQ question and all options using the MedCPT cross-encoder."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=["medmcqa", "medqa"], default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--stages", nargs="+", choices=["embed", "retrieve"], default=["embed", "retrieve"])
    parser.add_argument("--no-rag-root", type=Path, default=DEFAULT_NO_RAG_ROOT)
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=None,
        help=(
            "Optional root created by audit_rag2_no_rag_quality_selection.py. When set, only "
            "<selection-root>/<dataset>/<split>/usable_rows.jsonl is embedded and retrieved."
        ),
    )
    parser.add_argument("--query-cache-root", type=Path, default=DEFAULT_QUERY_CACHE_ROOT)
    parser.add_argument("--candidate-output-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--vector-db-root", type=Path, default=DEFAULT_VECTOR_DB_ROOT)
    parser.add_argument("--query-encoder-path", type=Path, default=DEFAULT_QUERY_ENCODER)
    parser.add_argument("--cross-encoder-path", type=Path, default=DEFAULT_CROSS_ENCODER)

    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--embedding-attn-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--quality-policy", choices=["technical", "conservative"], default="conservative")
    parser.add_argument("--overwrite-embeddings", action="store_true")

    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--per-source-top-k", type=int, default=10)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--retrieval-batch-size", type=int, default=2048)
    parser.add_argument("--rerank-batch-size", type=int, default=1024)
    parser.add_argument("--cross-encoder-max-length", type=int, default=512)
    parser.add_argument(
        "--cross-encoder-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument(
        "--keep-faiss-indexes-in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep every CPU FAISS shard resident. Disable this for the large RAG_Square PubMed/PMC "
            "indexes so each physical shard is released before the next one is opened."
        ),
    )
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-gpu-add-batch-size", type=int, default=1_000_000)
    parser.add_argument("--faiss-gpu-temp-memory-mb", type=int, default=2048)
    parser.add_argument("--metadata-row-cache-size", type=int, default=50_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError(f"Duplicate datasets are not allowed: {args.datasets}")
    if len(set(args.sources)) != len(args.sources):
        raise ValueError(f"Duplicate corpus sources are not allowed: {args.sources}")
    if args.embedding_batch_size <= 0 or args.embedding_max_length <= 0:
        raise ValueError("Embedding batch size and maximum length must be positive.")
    if args.per_source_top_k <= 0 or args.rerank_top_k <= 0:
        raise ValueError("Retrieval and reranking Top-k values must be positive.")
    candidate_pool_top_k = args.per_source_top_k * len(args.sources)
    if args.rerank_top_k > candidate_pool_top_k:
        raise ValueError("--rerank-top-k cannot exceed the balanced retrieval pool size.")
    if args.retrieval_batch_size <= 0 or args.rerank_batch_size <= 0:
        raise ValueError("Retrieval and reranking batch sizes must be positive.")
    if args.selection_root is not None:
        for dataset in args.datasets:
            selection_path = args.selection_root / dataset / args.split / "usable_rows.jsonl"
            if not selection_path.exists():
                raise FileNotFoundError(f"Missing quality selection for {dataset}: {selection_path}")


def embedding_command(args: argparse.Namespace, dataset: str) -> list[str]:
    command = [
        sys.executable,
        str(EMBED_SCRIPT),
        "--dataset",
        dataset,
        "--split",
        args.split,
        "--collection",
        args.collection,
        "--no-rag-root",
        str(args.no_rag_root),
        "--model-path",
        str(args.query_encoder_path),
        "--output-root",
        str(args.query_cache_root),
        "--batch-size",
        str(args.embedding_batch_size),
        "--max-length",
        str(args.embedding_max_length),
        "--attn-implementation",
        args.embedding_attn_implementation,
        "--invalid-row-policy",
        "exclude",
        "--quality-policy",
        args.quality_policy,
        "--log-level",
        args.log_level,
    ]
    if args.overwrite_embeddings:
        command.append("--overwrite")
    if args.selection_root is not None:
        command.extend(
            [
                "--selection-path",
                str(args.selection_root / dataset / args.split / "usable_rows.jsonl"),
            ]
        )
    return command


def retrieval_command(args: argparse.Namespace, dataset: str) -> list[str]:
    candidate_pool_top_k = args.per_source_top_k * len(args.sources)
    output_dir = args.candidate_output_root / dataset / args.split
    command = [
        sys.executable,
        str(CANDIDATE_SCRIPT),
        "--dataset",
        dataset,
        "--split",
        args.split,
        "--collection",
        args.collection,
        "--query-cache-root",
        str(args.query_cache_root),
        "--vector-db-root",
        str(args.vector_db_root),
        "--cross-encoder-path",
        str(args.cross_encoder_path),
        "--output-dir",
        str(output_dir),
        "--output-file",
        f"candidates_top{args.rerank_top_k}.jsonl",
        "--sources",
        *args.sources,
        "--per-source-top-k",
        str(args.per_source_top_k),
        "--candidate-pool-top-k",
        str(candidate_pool_top_k),
        "--top-k",
        str(args.rerank_top_k),
        "--retrieval-search-mode",
        "faiss_gpu_source_sequential",
        "--retrieval-batch-size",
        str(args.retrieval_batch_size),
        "--rerank-batch-size",
        str(args.rerank_batch_size),
        "--cross-encoder-max-length",
        str(args.cross_encoder_max_length),
        "--cross-encoder-attn-implementation",
        args.cross_encoder_attn_implementation,
        "--keep-faiss-indexes-in-memory" if args.keep_faiss_indexes_in_memory else "--no-keep-faiss-indexes-in-memory",
        "--faiss-mmap",
        "--no-faiss-shard-threaded",
        "--faiss-gpu-device",
        str(args.faiss_gpu_device),
        "--faiss-gpu-use-float16",
        "--faiss-gpu-add-batch-size",
        str(args.faiss_gpu_add_batch_size),
        "--faiss-gpu-temp-memory-mb",
        str(args.faiss_gpu_temp_memory_mb),
        "--metadata-row-cache-size",
        str(args.metadata_row_cache_size),
        "--no-include-initial-doc-text",
        "--log-level",
        args.log_level,
        "--resume" if args.resume else "--no-resume",
    ]
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    logging.info("Command: %s", shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=WORKSPACE_ROOT, env=os.environ.copy(), check=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    commands: list[tuple[str, list[str]]] = []
    if "embed" in args.stages:
        commands.extend(
            (f"embed:{dataset}", embedding_command(args, dataset)) for dataset in args.datasets
        )
    if "retrieve" in args.stages:
        commands.extend(
            (f"retrieve+rerank:{dataset}", retrieval_command(args, dataset)) for dataset in args.datasets
        )

    logging.info(
        "RAG2 candidate preparation: datasets=%s stages=%s jobs=%s",
        ",".join(args.datasets),
        ",".join(args.stages),
        len(commands),
    )
    pipeline_started = time.monotonic()
    for index, (name, command) in enumerate(commands, start=1):
        completed = index - 1
        elapsed = time.monotonic() - pipeline_started
        estimated_remaining = (
            elapsed / completed * (len(commands) - completed)
            if completed > 0
            else None
        )
        logging.info(
            "Pipeline %s/%s (%.1f%%) | current=%s | elapsed=%s | overall ETA=%s",
            completed,
            len(commands),
            100.0 * completed / max(len(commands), 1),
            name,
            format_duration(elapsed),
            format_duration(estimated_remaining),
        )
        run_command(command, args.dry_run)
        elapsed = time.monotonic() - pipeline_started
        estimated_remaining = elapsed / index * (len(commands) - index)
        logging.info(
            "Pipeline %s/%s (%.1f%%) | finished=%s | elapsed=%s | overall ETA=%s",
            index,
            len(commands),
            100.0 * index / max(len(commands), 1),
            name,
            format_duration(elapsed),
            format_duration(estimated_remaining),
        )


if __name__ == "__main__":
    main()

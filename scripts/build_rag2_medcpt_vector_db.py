from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models" / "MedCPT-Article-Encoder"
DEFAULT_VECTOR_ROOT = PROJECT_ROOT / "databases" / "vector_db"
DEFAULT_INDEX_NAME = "medcpt_article_encoder"
RAG2_MANIFEST = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2" / "manifest.json"
RAG2_UNIFIED_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(sources: list[str]) -> None:
    if not RAG2_MANIFEST.exists():
        raise FileNotFoundError(f"Missing RAG2 unified manifest: {RAG2_MANIFEST}")
    manifest = read_json(RAG2_MANIFEST)
    manifest_sources = manifest.get("sources") or {}
    for source in sources:
        input_path = RAG2_UNIFIED_DIR / f"{source}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing unified corpus for {source}: {input_path}")
        if source not in manifest_sources:
            raise RuntimeError(f"Manifest has no sources.{source} entry: {RAG2_MANIFEST}")
        chunks = int(manifest_sources[source].get("chunks", 0))
        if chunks <= 0:
            raise RuntimeError(f"Manifest has invalid chunk count for {source}: {chunks}")
        logging.info("[%s] unified ready: chunks=%s path=%s", source, chunks, input_path)


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    logging.info("Running command:\n%s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT), env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MedCPT Article Encoder vector DBs for RAG2-style PMC/CPG corpora. "
            "The safe sharded build is created first, then optionally flattened into "
            "the existing one-source-folder format: index.faiss + metadata.jsonl."
        )
    )
    parser.add_argument("--sources", nargs="+", choices=["cpg", "pmc"], default=["cpg", "pmc"])
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=2_000_000)
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16", "none"], default="auto")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--flatten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flatten sharded build into existing source DB format under output-root/index-name/<source>.",
    )
    parser.add_argument("--reconstruct-batch-size", type=int, default=100_000)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    sources = list(dict.fromkeys(args.sources))
    validate_inputs(sources)

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    python = sys.executable
    build_cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "build_medcpt_vector_db.py"),
        "--model-path",
        str(args.model_path),
        "--output-root",
        str(args.output_root),
        "--index-name",
        args.index_name,
        "--sources",
        *sources,
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--shard-size",
        str(args.shard_size),
        "--amp-dtype",
        args.amp_dtype,
        "--model-dtype",
        args.model_dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--skip-logical-merge",
        "--torch-num-threads",
        str(args.torch_num_threads),
    ]
    if args.normalize:
        build_cmd.append("--normalize")
    if args.overwrite:
        build_cmd.append("--overwrite")
    if args.allow_count_mismatch:
        build_cmd.append("--allow-count-mismatch")
    run_command(build_cmd, env)

    base_dir = args.output_root / args.index_name
    logging.info("Sharded source DBs are under: %s/sources/{%s}", base_dir, ",".join(sources))

    if args.flatten:
        flatten_cmd = [
            python,
            str(PROJECT_ROOT / "scripts" / "flatten_medcpt_vector_db.py"),
            "--input-root",
            str(base_dir),
            "--output-root",
            str(base_dir),
            "--sources",
            *sources,
            "--reconstruct-batch-size",
            str(args.reconstruct_batch_size),
        ]
        if args.overwrite:
            flatten_cmd.append("--overwrite")
        run_command(flatten_cmd, env)
        logging.info("Flat source DBs are under: %s/{%s}", base_dir, ",".join(sources))


if __name__ == "__main__":
    main()

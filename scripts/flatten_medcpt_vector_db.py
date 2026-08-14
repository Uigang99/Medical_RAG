from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "databases" / "vector_db" / "medcpt_article_encoder"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "databases" / "vector_db" / "medcpt_article_encoder_flat"
SOURCE_ORDER = ["pubmed", "statpearls", "textbooks", "wikipedia", "pmc", "cpg", "bioasq", "covidqa", "mashqa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten an existing sharded MedCPT vector DB into one FAISS file and one metadata JSONL "
            "file per source dataset. This does not recompute embeddings."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_ORDER, default=SOURCE_ORDER)
    parser.add_argument(
        "--reconstruct-batch-size",
        type=int,
        default=100_000,
        help="Vectors reconstructed from each input shard at a time before adding to the flat index.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild an existing flattened source directory.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def progress_bar(*args: Any, **kwargs: Any) -> Any:
    if tqdm is None:
        return None
    return tqdm(*args, **kwargs)


def metadata_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = manifest.get("metadata") or {}
    if metadata.get("layout") != "sharded":
        raise RuntimeError("Expected a sharded metadata layout in the source manifest.")
    return list(metadata["shards"])


def expected_index_size(rows: int, dim: int) -> int:
    # Current DB uses FAISS IndexFlatIP. Its file size is the small FAISS header
    # plus rows * dim * float32 bytes.
    return 45 + rows * dim * 4


def copy_metadata(source_dir: Path, output_path: Path, manifest: dict[str, Any]) -> int:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    rows_written = 0
    shards = metadata_shards(manifest)
    pbar = progress_bar(
        total=int(manifest["rows"]),
        desc=f"metadata:{manifest['source']}",
        unit="row",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    with tmp_path.open("wb", buffering=16 * 1024 * 1024) as out:
        for shard in shards:
            shard_path = source_dir / shard["path"]
            logging.info("[%s] appending metadata %s", manifest["source"], shard_path)
            with shard_path.open("rb", buffering=16 * 1024 * 1024) as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            rows = int(shard["rows"])
            rows_written += rows
            if pbar is not None:
                pbar.update(rows)
    if pbar is not None:
        pbar.close()
    tmp_path.replace(output_path)
    return rows_written


def flatten_index(
    source_dir: Path,
    output_path: Path,
    manifest: dict[str, Any],
    reconstruct_batch_size: int,
) -> int:
    dim = int(manifest["index"]["dimension"])
    index = faiss.IndexFlatIP(dim)
    rows_added = 0
    pbar = progress_bar(
        total=int(manifest["rows"]),
        desc=f"index:{manifest['source']}",
        unit="vec",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    for shard in manifest["index"]["shards"]:
        shard_path = source_dir / shard["path"]
        logging.info("[%s] reading index shard %s", manifest["source"], shard_path)
        shard_index = faiss.read_index(str(shard_path))
        shard_rows = int(shard_index.ntotal)
        if shard_rows != int(shard["rows"]):
            raise RuntimeError(
                f"Shard row mismatch: {shard_path} has {shard_rows}, manifest has {shard['rows']}"
            )
        for start in range(0, shard_rows, reconstruct_batch_size):
            size = min(reconstruct_batch_size, shard_rows - start)
            vectors = shard_index.reconstruct_n(start, size)
            index.add(np.ascontiguousarray(vectors, dtype="float32"))
            rows_added += size
            if pbar is not None:
                pbar.update(size)
        del shard_index
        gc.collect()
    if pbar is not None:
        pbar.close()

    if rows_added != int(manifest["rows"]):
        raise RuntimeError(f"Flattened row mismatch: added {rows_added}, expected {manifest['rows']}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    logging.info("[%s] writing flat FAISS index %s", manifest["source"], output_path)
    faiss.write_index(index, str(tmp_path))
    del index
    gc.collect()
    tmp_path.replace(output_path)

    actual_size = output_path.stat().st_size
    expected_size = expected_index_size(rows_added, dim)
    if actual_size != expected_size:
        raise RuntimeError(
            f"Unexpected index file size for {output_path}: {actual_size} != {expected_size}"
        )
    return rows_added


def valid_flat_source(source_dir: Path, expected_rows: int, dim: int) -> bool:
    manifest_path = source_dir / "manifest.json"
    index_path = source_dir / "index.faiss"
    metadata_path = source_dir / "metadata.jsonl"
    if not manifest_path.exists() or not index_path.exists() or not metadata_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return False
    if int(manifest.get("rows", -1)) != expected_rows:
        return False
    if int(manifest.get("index", {}).get("dimension", -1)) != dim:
        return False
    return index_path.stat().st_size == expected_index_size(expected_rows, dim)


def flatten_source(input_root: Path, output_root: Path, source: str, args: argparse.Namespace) -> dict[str, Any]:
    source_dir = input_root / "sources" / source
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    source_manifest = read_json(source_manifest_path)
    rows = int(source_manifest["rows"])
    dim = int(source_manifest["index"]["dimension"])

    output_dir = output_root / source
    if output_dir.exists() and valid_flat_source(output_dir, rows, dim) and not args.overwrite:
        logging.info("[%s] existing flat DB is valid; skipping.", source)
        return read_json(output_dir / "manifest.json")
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{output_dir} exists but is incomplete. Rerun with --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[%s] flattening %s rows into %s", source, rows, output_dir)
    metadata_rows = copy_metadata(source_dir, output_dir / "metadata.jsonl", source_manifest)
    index_rows = flatten_index(
        source_dir=source_dir,
        output_path=output_dir / "index.faiss",
        manifest=source_manifest,
        reconstruct_batch_size=args.reconstruct_batch_size,
    )
    if metadata_rows != rows or index_rows != rows:
        raise RuntimeError(f"[{source}] row mismatch: metadata={metadata_rows}, index={index_rows}, expected={rows}")

    manifest = {
        "type": "source_vector_db_flat",
        "layout": "single_faiss_index_plus_single_metadata_jsonl",
        "source": source,
        "rows": rows,
        "created_at": now_utc(),
        "source_manifest": rel(source_manifest_path),
        "model": source_manifest["model"],
        "index": {
            "backend": "faiss",
            "index_type": source_manifest["index"]["index_type"],
            "metric": source_manifest["index"]["metric"],
            "dimension": dim,
            "path": "index.faiss",
            "bytes": (output_dir / "index.faiss").stat().st_size,
        },
        "metadata": {
            "format": "jsonl",
            "path": "metadata.jsonl",
            "rows": rows,
            "bytes": (output_dir / "metadata.jsonl").stat().st_size,
        },
        "input_shards": {
            "index_shards": len(source_manifest["index"]["shards"]),
            "metadata_shards": len(metadata_shards(source_manifest)),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    logging.info("[%s] done: index=%s, metadata=%s", source, manifest["index"]["bytes"], manifest["metadata"]["bytes"])
    return manifest


def main() -> None:
    configure_logging()
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifests: list[dict[str, Any]] = []
    for source in args.sources:
        source_manifests.append(flatten_source(input_root, output_root, source, args))

    total_rows = sum(int(manifest["rows"]) for manifest in source_manifests)
    total_index_bytes = sum(int(manifest["index"]["bytes"]) for manifest in source_manifests)
    total_metadata_bytes = sum(int(manifest["metadata"]["bytes"]) for manifest in source_manifests)
    root_manifest = {
        "type": "flat_vector_db_collection",
        "layout": "one_directory_per_source_no_merged_index",
        "created_at": now_utc(),
        "input_root": rel(input_root),
        "output_root": rel(output_root),
        "sources": {
            manifest["source"]: {
                "path": manifest["source"],
                "rows": manifest["rows"],
                "index_path": f"{manifest['source']}/index.faiss",
                "metadata_path": f"{manifest['source']}/metadata.jsonl",
                "index_bytes": manifest["index"]["bytes"],
                "metadata_bytes": manifest["metadata"]["bytes"],
            }
            for manifest in source_manifests
        },
        "totals": {
            "rows": total_rows,
            "index_bytes": total_index_bytes,
            "metadata_bytes": total_metadata_bytes,
        },
    }
    write_json(output_root / "manifest.json", root_manifest)
    logging.info(
        "Flat DB collection complete: rows=%s, index=%.2f GiB, metadata=%.2f GiB",
        total_rows,
        total_index_bytes / 1024**3,
        total_metadata_bytes / 1024**3,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Project one paper-balanced master cache into exact dynamic Top-k caches.

For every requested k, the contract is:
  dense Top-k from each of four logical corpora -> rerank all 4k -> keep Top-k.

The master cache already contains dense Top-32 per corpus and reranker scores
for all 128 candidates, so no retrieval or cross-encoder inference is repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = (
    PROJECT_ROOT
    / "databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1/"
    "all_mcq_paper_balanced_max32_rationale_answer_rerank128/candidates/"
    "521e23c599352822/candidates.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "databases/run_cache/rag2_pced_dynamic_topk_v1"
SOURCES = ("pubmed", "pmc", "cpg", "textbooks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-cache", type=Path, required=True,
        help="Exact 4-corpus x 32 / rerank-128 master; required to prevent silent query-cache substitution.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-top8-cache", type=Path, default=None,
        help="Optional completed Top-8 cache whose selected document IDs/order must be exactly reproduced.",
    )
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--expected-rows", type=int, default=6545)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL: {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("chunk_id") or document.get("db_id")
    if value is None:
        source = str(document.get("source") or "")
        local_id = document.get("local_id")
        value = f"{source}:{local_id}"
    return str(value)


def logical_source(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    return str(metadata.get("retrieval_bucket") or document.get("retrieval_bucket") or document.get("source") or "")


def project(row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    initial = list(row.get("initial_documents") or [])
    reranked = list(row.get("reranked_documents") or [])
    source_ranks: Counter[str] = Counter()
    rank_by_id: dict[str, int] = {}
    source_by_id: dict[str, str] = {}
    eligible: set[str] = set()
    for document in initial:
        source = logical_source(document)
        source_ranks[source] += 1
        identifier = stable_id(document)
        if identifier in rank_by_id:
            raise RuntimeError(f"Duplicate stable_id in dense pool: {row.get('key')} {identifier}")
        rank_by_id[identifier] = int(source_ranks[source])
        source_by_id[identifier] = source
        if source_ranks[source] <= top_k:
            eligible.add(identifier)
    expected_counts = {source: top_k for source in SOURCES}
    actual_counts = Counter(source_by_id[value] for value in eligible)
    if dict(actual_counts) != expected_counts or len(eligible) != 4 * top_k:
        raise RuntimeError(
            f"Dense-prefix invariant failed: key={row.get('key')} k={top_k} "
            f"expected={expected_counts} actual={dict(actual_counts)}"
        )
    selected_pool: list[dict[str, Any]] = []
    for original in reranked:
        identifier = stable_id(original)
        if identifier not in eligible:
            continue
        item = dict(original)
        metadata = dict(item.get("metadata") or {})
        metadata["retrieval_bucket"] = source_by_id[identifier]
        metadata["source_retrieval_rank"] = rank_by_id[identifier]
        item["metadata"] = metadata
        item["retrieval_bucket"] = source_by_id[identifier]
        item["source_retrieval_rank"] = rank_by_id[identifier]
        selected_pool.append(item)
    if len(selected_pool) != 4 * top_k:
        raise RuntimeError(
            f"Incomplete rerank pool: key={row.get('key')} k={top_k} "
            f"expected={4*top_k} actual={len(selected_pool)}"
        )
    selected_pool.sort(
        key=lambda item: float(item.get("rerank_score")) if item.get("rerank_score") is not None else float("-inf"),
        reverse=True,
    )
    selected = selected_pool[:top_k]
    for rank, item in enumerate(selected, 1):
        item["rerank_rank"] = rank
    return selected


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def complete_cache(
    path: Path, top_k: int, expected_rows: int, master_hash: str, reference_hash: str | None,
) -> bool:
    manifest = path.parent / "manifest.json"
    if not path.is_file() or not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        value.get("type") == "rag2_pced_dynamic_topk_projection_v1"
        and int(value.get("rows", -1)) == expected_rows
        and int(value.get("evaluation_top_k", -1)) == top_k
        and value.get("master_sha256") == master_hash
        and value.get("reference_top8_sha256") == reference_hash
        and value.get("output_sha256") == sha256_file(path)
    )


def main() -> None:
    args = parse_args()
    values = sorted(set(args.top_k_values))
    if not values or min(values) <= 0 or max(values) > 32:
        raise ValueError("top-k values must be unique integers in [1, 32]")
    manifest_path = args.master_cache.parent / "manifest.json"
    if not args.master_cache.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing master cache/manifest: {args.master_cache}")
    master_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"rows": args.expected_rows, "per_source_top_k": 32, "candidate_pool_top_k": 128, "rerank_top_k": 128}
    mismatch = {key: (expected, master_manifest.get(key)) for key, expected in required.items() if master_manifest.get(key) != expected}
    if mismatch:
        raise RuntimeError(f"Master candidate contract mismatch: {mismatch}")
    master_hash = sha256_file(args.master_cache)
    reference_hash = None
    if args.reference_top8_cache is not None:
        if not args.reference_top8_cache.is_file():
            raise FileNotFoundError(args.reference_top8_cache)
        reference_hash = sha256_file(args.reference_top8_cache)
    pending = [
        top_k for top_k in values
        if not complete_cache(
            args.output_root / f"top{top_k}" / "candidates.jsonl",
            top_k, args.expected_rows, master_hash, reference_hash,
        )
    ]
    print(
        f"[stage 1/1 | project dynamic candidates] rows={args.expected_rows} k={values} "
        f"cached={len(values)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        print(f"[stage 1/1 complete] all candidate projections ready under {args.output_root}", flush=True)
        return
    handles: dict[int, Any] = {}
    temporary_paths: dict[int, Path] = {}
    started = time.time()
    try:
        for top_k in pending:
            output = args.output_root / f"top{top_k}" / "candidates.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + f".partial.{os.getpid()}")
            temporary_paths[top_k] = temporary
            handles[top_k] = temporary.open("w", encoding="utf-8")
        progress = tqdm(total=args.expected_rows, desc="Stage 1/1 - 4-corpus dynamic Top-k projection", unit="question", dynamic_ncols=True)
        rows = 0
        for row in iter_jsonl(args.master_cache):
            rows += 1
            for top_k in pending:
                value = {
                    "key": row.get("key"),
                    "dataset": row.get("dataset"),
                    "sample_id": row.get("sample_id"),
                    "row_idx": row.get("row_idx"),
                    "dense_query_mode": row.get("dense_query_mode"),
                    "query_text": row.get("query_text"),
                    "reranked_documents": project(row, top_k),
                }
                handles[top_k].write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            progress.update(1)
            elapsed = max(time.time() - started, 1e-6)
            rate = rows / elapsed
            eta = (args.expected_rows - rows) / rate if rate > 0 else float("inf")
            progress.set_postfix_str(f"{rate:.1f} q/s ETA={eta/60:.1f}m", refresh=False)
        progress.close()
        if rows != args.expected_rows:
            raise RuntimeError(f"Master row count mismatch: expected={args.expected_rows} actual={rows}")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for top_k in pending:
            output = args.output_root / f"top{top_k}" / "candidates.jsonl"
            os.replace(temporary_paths[top_k], output)
        if args.reference_top8_cache is not None and 8 in values:
            projected_path = args.output_root / "top8" / "candidates.jsonl"
            checked = 0
            for projected, reference in zip(
                iter_jsonl(projected_path), iter_jsonl(args.reference_top8_cache), strict=True
            ):
                if projected.get("key") != reference.get("key"):
                    raise RuntimeError(f"Top-8 reference key mismatch at row {checked}")
                projected_ids = [stable_id(item) for item in projected["reranked_documents"]]
                reference_ids = [stable_id(item) for item in list(reference["reranked_documents"])[:8]]
                if projected_ids != reference_ids:
                    raise RuntimeError(
                        f"Dynamic Top-8 does not reproduce the prior PCED candidate set: "
                        f"row={checked} key={projected.get('key')}"
                    )
                checked += 1
            if checked != args.expected_rows:
                raise RuntimeError(f"Top-8 reference audit count mismatch: {checked}")
            print(f"[reference audit] exact Top-8 document ID/order match: {checked}/{checked}", flush=True)
        for top_k in pending:
            output = args.output_root / f"top{top_k}" / "candidates.jsonl"
            atomic_json(output.parent / "manifest.json", {
                "type": "rag2_pced_dynamic_topk_projection_v1",
                "rows": args.expected_rows,
                "sources": list(SOURCES),
                "per_source_top_k": top_k,
                "candidate_pool_top_k": 4 * top_k,
                "candidate_layout": "source_balanced",
                "rerank_top_k": top_k,
                "evaluation_top_k": top_k,
                "projection": "dense Top-k per corpus (4k total), then rerank and keep final Top-k",
                "master_cache": str(args.master_cache.resolve()),
                "master_manifest_sha256": sha256_file(manifest_path),
                "master_sha256": master_hash,
                "reference_top8_cache": (
                    str(args.reference_top8_cache.resolve()) if args.reference_top8_cache is not None else None
                ),
                "reference_top8_sha256": reference_hash,
                "output_path": str(output.resolve()),
                "output_sha256": sha256_file(output),
            })
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise
    elapsed = time.time() - started
    print(
        f"[stage 1/1 complete | elapsed={elapsed/60:.1f}m] rows={args.expected_rows} "
        f"written_k={pending} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()

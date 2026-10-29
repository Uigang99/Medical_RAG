"""One-pass, per-source MedCPT inner-product search for four candidate views.

The stored RAG_Square FAISS vectors and query embeddings are not normalized;
scores therefore match the original IndexFlatIP retrieval metric.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def _atomic_state(path: Path, *, contract: str, next_shard: int,
                  scores: np.ndarray, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, contract=np.asarray(contract), next_shard=np.asarray(next_shard),
                 scores=scores, ids=ids)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_state(path: Path, *, contract: str, shape: tuple[int, int],
               shard_count: int) -> tuple[int, np.ndarray, np.ndarray]:
    if not path.exists():
        return (0, np.full(shape, -np.inf, dtype=np.float32),
                np.full(shape, -1, dtype=np.int64))
    with np.load(path, allow_pickle=False) as state:
        if str(state["contract"].item()) != contract:
            raise RuntimeError(f"Inner-product checkpoint contract mismatch: {path}")
        next_shard = int(state["next_shard"].item())
        scores = np.asarray(state["scores"], dtype=np.float32)
        ids = np.asarray(state["ids"], dtype=np.int64)
    if scores.shape != shape or ids.shape != shape or not 0 <= next_shard <= shard_count:
        raise RuntimeError(f"Invalid inner-product checkpoint shape or cursor: {path}")
    return next_shard, scores, ids


def merge_topk(best_scores: torch.Tensor, best_ids: torch.Tensor,
               chunk_scores: torch.Tensor, chunk_ids: torch.Tensor,
               k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the best document IDs for each query after one vector chunk."""
    combined_scores = torch.cat((best_scores, chunk_scores), dim=1)
    combined_ids = torch.cat((best_ids, chunk_ids), dim=1)
    scores, positions = torch.topk(combined_scores, k=k, dim=1)
    return scores, combined_ids.gather(1, positions)


def source_topk_inner_product(
    source_index,
    query_vectors: np.ndarray,
    *,
    k: int,
    vector_chunk_size: int,
    query_batch_size: int,
    device: torch.device,
    checkpoint_path: Path,
    contract: str,
    on_vectors: Callable[[int], None] | None = None,
    on_checkpoint: Callable[[int, int], None] | None = None,
    eligibility: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Search each physical FAISS shard once against unnormalized queries.

    Scores are raw float32 inner products, matching IndexFlatIP. A checkpoint
    is committed after every complete physical shard, so a restart never needs
    to repeat an earlier shard.  Source-local IDs are preserved throughout.
    """
    if k <= 0 or vector_chunk_size <= 0 or query_batch_size <= 0:
        raise ValueError("k, vector_chunk_size, and query_batch_size must be positive")
    vectors = np.asarray(query_vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0 or not np.isfinite(vectors).all():
        raise ValueError("Query vectors must be a non-empty, finite 2-D array")
    if int(source_index.rows) < k:
        raise ValueError(f"Source {source_index.source} has fewer than {k} vectors")
    if eligibility is not None:
        if eligibility.shape != (int(source_index.rows),) or eligibility.dtype != np.bool_:
            raise ValueError("Eligibility must be a boolean mask aligned to source-local IDs")
        if np.count_nonzero(eligibility) < k:
            raise ValueError(f"Source {source_index.source} has fewer than {k} eligible vectors")

    shards = source_index._shards if source_index._is_sharded else [None]
    cursor, saved_scores, saved_ids = load_state(
        checkpoint_path, contract=contract, shape=(len(vectors), k), shard_count=len(shards)
    )
    if cursor == len(shards):
        return saved_scores, saved_ids

    queries = torch.as_tensor(vectors, device=device)
    best_scores = torch.as_tensor(saved_scores, device=device)
    best_ids = torch.as_tensor(saved_ids, device=device)
    for shard_number in range(cursor, len(shards)):
        shard = shards[shard_number]
        index = source_index._shard_index(shard) if shard is not None else source_index.index
        source_offset = int(shard["start"]) if shard is not None else 0
        for offset in range(0, int(index.ntotal), vector_chunk_size):
            size = min(vector_chunk_size, int(index.ntotal) - offset)
            keep = (eligibility[source_offset + offset:source_offset + offset + size]
                    if eligibility is not None else None)
            if keep is not None and not keep.any():
                if on_vectors:
                    on_vectors(size)
                continue
            raw = np.ascontiguousarray(index.reconstruct_n(offset, size), dtype=np.float32)
            selected_ids = None
            if keep is not None:
                selected_ids = torch.as_tensor(
                    np.flatnonzero(keep) + source_offset + offset, device=device)
                raw = np.ascontiguousarray(raw[keep])
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite vector in {source_index.source}:{source_offset + offset}")
            block = torch.as_tensor(raw, device=device)
            local_k = min(k, len(raw))
            for query_start in range(0, len(vectors), query_batch_size):
                query_end = min(query_start + query_batch_size, len(vectors))
                similarity = queries[query_start:query_end] @ block.T
                scores, positions = torch.topk(similarity, k=local_k, dim=1)
                ids = (selected_ids[positions] if selected_ids is not None else
                       positions.to(torch.int64) + source_offset + offset)
                merged = merge_topk(best_scores[query_start:query_end],
                                    best_ids[query_start:query_end], scores, ids, k)
                best_scores[query_start:query_end] = merged[0]
                best_ids[query_start:query_end] = merged[1]
                del similarity, scores, positions, ids, merged
            del raw, block, selected_ids
            if on_vectors:
                on_vectors(size)
        saved_scores = best_scores.cpu().numpy().copy()
        saved_ids = best_ids.cpu().numpy().copy()
        _atomic_state(checkpoint_path, contract=contract, next_shard=shard_number + 1,
                      scores=saved_scores, ids=saved_ids)
        if on_checkpoint:
            on_checkpoint(shard_number + 1, len(shards))
        if shard is not None and not source_index.cache_index:
            shard["index"] = None
    return saved_scores, saved_ids


def derive_candidate_ids(per_source: dict[str, tuple[np.ndarray, np.ndarray]],
                         row: int, *, balanced_per_source: int = 8,
                         balanced_quotas: dict[str, int] | None = None,
                         global_k: int = 32) -> tuple[list[tuple[str, int, float, int]],
                                                           list[tuple[str, int, float, int]]]:
    """Derive both exact sets from per-source Top-32 inner-product hits."""
    balanced: list[tuple[str, int, float, int]] = []
    pool: list[tuple[str, int, float, int]] = []
    quotas = (balanced_quotas if balanced_quotas is not None else
              {source: balanced_per_source for source in per_source})
    if set(quotas) != set(per_source) or any(q < 1 for q in quotas.values()):
        raise ValueError("Positive balanced quota required for every selected source")
    for source, (scores, ids) in per_source.items():
        source_hits = 0
        for source_rank, (score, local_id) in enumerate(zip(scores[row], ids[row]), start=1):
            if int(local_id) < 0 or not np.isfinite(score):
                continue
            hit = (source, int(local_id), float(score), source_rank)
            pool.append(hit)
            if source_rank <= quotas[source]:
                balanced.append(hit)
                source_hits += 1
        if source_hits != quotas[source]:
            raise RuntimeError(f"Incomplete balanced hits for {source} at row {row}")
    if len(balanced) != sum(quotas.values()):
        raise RuntimeError(f"Incomplete balanced inner-product hits at row {row}")
    if len(pool) < global_k:
        raise RuntimeError(f"Fewer than {global_k} global inner-product hits at row {row}")
    balanced.sort(key=lambda hit: (-hit[2], hit[0], hit[1]))
    pool.sort(key=lambda hit: (-hit[2], hit[0], hit[1]))
    return balanced, pool[:global_k]

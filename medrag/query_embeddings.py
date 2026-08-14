from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_utils import read_json


class QueryEmbeddingStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.manifest = read_json(cache_dir / "manifest.json")
        self.embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
        expected = (int(self.manifest["rows"]), int(self.manifest["dimension"]))
        if tuple(self.embeddings.shape) != expected:
            raise ValueError(f"Query embedding shape mismatch: {self.embeddings.shape} != {expected}")

    @property
    def dimension(self) -> int:
        return int(self.manifest["dimension"])

    def get_batch(self, row_indices: list[int]) -> np.ndarray:
        if not row_indices:
            return np.empty((0, self.dimension), dtype="float32")
        return np.asarray(self.embeddings[row_indices], dtype="float32")


def resolve_query_cache_dir(
    query_cache_root: Path,
    task: str,
    collection: str,
    dataset: str,
    split: str,
) -> Path:
    if task == "mcq":
        return query_cache_root / "mcq" / collection / dataset / split
    if task == "open_ended":
        return query_cache_root / "open_ended" / collection / dataset
    raise ValueError(f"Unsupported task: {task}")


from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import faiss
except Exception as exc:  # pragma: no cover
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:
    FAISS_IMPORT_ERROR = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from ..core import RetrievedDocument
from ..io_utils import read_json


DEFAULT_SOURCES = ["pubmed", "statpearls", "textbooks", "wikipedia", "bioasq", "covidqa", "mashqa"]
MCQ_SOURCES = ["pubmed", "statpearls", "textbooks", "wikipedia"]
OPEN_ENDED_SOURCES = ["bioasq", "covidqa", "mashqa", "pubmed"]
_INDEX_CACHE: dict[str, Any] = {}


class MetadataStore:
    """Random-access JSONL metadata for either a flat or one physical shard.

    The optional explicit paths are used by the resumable ``source_vector_db``
    layout.  Omitting them preserves the legacy ``source_vector_db_flat``
    contract exactly.
    """

    def __init__(
        self,
        source_dir: Path,
        row_cache_size: int = 50000,
        metadata_path: Path | None = None,
        rows: int | None = None,
        offset_path: Path | None = None,
    ) -> None:
        self.source_dir = source_dir
        if metadata_path is None:
            self.manifest = read_json(source_dir / "manifest.json")
            metadata = self.manifest["metadata"]
            self.metadata_path = source_dir / metadata["path"]
            self.rows = int(metadata["rows"])
            self.offset_path = source_dir / "metadata.offsets.npy"
        else:
            if rows is None:
                raise ValueError("rows is required when metadata_path is supplied")
            self.manifest = None
            self.metadata_path = metadata_path
            self.rows = int(rows)
            self.offset_path = offset_path or metadata_path.with_suffix(".offsets.npy")
        self._offsets: np.memmap | None = None
        self._handle: Any | None = None
        self.row_cache_size = max(0, int(row_cache_size))
        self._row_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def ensure_offsets(self) -> None:
        if self.offset_path.exists():
            offsets = np.load(self.offset_path, mmap_mode="r")
            if tuple(offsets.shape) == (self.rows,):
                self._offsets = offsets
                return
            logging.warning("Ignoring stale metadata offset index: %s", self.offset_path)

        logging.info("Building metadata offset index: %s", self.offset_path)
        tmp_path = self.offset_path.with_suffix(".npy.tmp")
        offsets = np.lib.format.open_memmap(tmp_path, mode="w+", dtype="int64", shape=(self.rows,))
        row_idx = 0
        pbar = tqdm(total=self.rows, desc=f"offsets:{self.source_dir.name}", unit="row") if tqdm else None
        with self.metadata_path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if row_idx >= self.rows:
                    raise RuntimeError(f"Metadata has more rows than manifest: {self.metadata_path}")
                offsets[row_idx] = pos
                row_idx += 1
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()
        offsets.flush()
        del offsets
        if row_idx != self.rows:
            raise RuntimeError(f"Metadata offset rows mismatch for {self.metadata_path}: {row_idx} != {self.rows}")
        tmp_path.replace(self.offset_path)
        self._offsets = np.load(self.offset_path, mmap_mode="r")

    @property
    def offsets(self) -> np.memmap:
        if self._offsets is None:
            self.ensure_offsets()
        assert self._offsets is not None
        return self._offsets

    @property
    def handle(self) -> Any:
        if self._handle is None:
            self._handle = self.metadata_path.open("rb")
        return self._handle

    def get(self, local_id: int) -> dict[str, Any]:
        if local_id < 0 or local_id >= self.rows:
            raise IndexError(f"local_id out of range for {self.source_dir.name}: {local_id}")
        cached = self._row_cache.get(local_id)
        if cached is not None:
            self._row_cache.move_to_end(local_id)
            return cached
        self.handle.seek(int(self.offsets[local_id]))
        row = json.loads(self.handle.readline().decode("utf-8"))
        if self.row_cache_size > 0:
            self._row_cache[local_id] = row
            if len(self._row_cache) > self.row_cache_size:
                self._row_cache.popitem(last=False)
        return row

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._row_cache.clear()


class FaissSourceIndex:
    def __init__(
        self,
        source_dir: Path,
        cache_index: bool = True,
        mmap_index: bool = True,
        metadata_row_cache_size: int = 50000,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss is required for retrieval.") from FAISS_IMPORT_ERROR
        self.source_dir = source_dir
        self.manifest = read_json(source_dir / "manifest.json")
        self.source = str(self.manifest["source"])
        self.cache_index = bool(cache_index)
        self.mmap_index = bool(mmap_index)
        self.rows = int(self.manifest["rows"])
        self._index: Any | None = None
        self._is_sharded = bool((self.manifest.get("index") or {}).get("shards"))
        self.index_path: Path | None = None
        self.metadata: MetadataStore | None = None
        self._shards: list[dict[str, Any]] = []
        self._shard_starts: list[int] = []

        if not self._is_sharded:
            self.index_path = source_dir / self.manifest["index"]["path"]
            self.metadata = MetadataStore(source_dir, row_cache_size=metadata_row_cache_size)
            return

        index_shards = list((self.manifest.get("index") or {}).get("shards") or [])
        metadata_shards = list((self.manifest.get("metadata") or {}).get("shards") or [])
        metadata_by_id = {int(shard["shard_id"]): shard for shard in metadata_shards}
        if not index_shards or len(index_shards) != len(metadata_shards):
            raise ValueError(f"Invalid sharded source manifest: {source_dir / 'manifest.json'}")
        for index_shard in index_shards:
            shard_id = int(index_shard["shard_id"])
            metadata_shard = metadata_by_id.get(shard_id)
            if metadata_shard is None:
                raise ValueError(f"Missing metadata shard {shard_id} for source {self.source}")
            start = int(index_shard["start_local_id"])
            shard_rows = int(index_shard["rows"])
            if start != int(metadata_shard["start_local_id"]) or shard_rows != int(metadata_shard["rows"]):
                raise ValueError(f"Index/metadata shard range mismatch for {self.source}:{shard_id}")
            metadata_path = source_dir / metadata_shard["path"]
            metadata = MetadataStore(
                source_dir,
                row_cache_size=metadata_row_cache_size,
                metadata_path=metadata_path,
                rows=shard_rows,
                offset_path=metadata_path.with_suffix(".offsets.npy"),
            )
            self._shards.append(
                {
                    "shard_id": shard_id,
                    "start": start,
                    "rows": shard_rows,
                    "index_path": source_dir / index_shard["path"],
                    "metadata": metadata,
                    "index": None,
                }
            )
            self._shard_starts.append(start)
        if sum(int(shard["rows"]) for shard in self._shards) != self.rows:
            raise ValueError(f"Sharded source rows mismatch for {self.source}")

    def _load_index(self, index_path: Path) -> Any:
        cache_key = str(index_path.resolve())
        if self.cache_index and cache_key in _INDEX_CACHE:
            return _INDEX_CACHE[cache_key]
        logging.debug("[%s] loading FAISS index: %s", self.source, index_path)
        flags = getattr(faiss, "IO_FLAG_MMAP", 0) if self.mmap_index else 0
        loaded = faiss.read_index(str(index_path), flags)
        if self.cache_index:
            _INDEX_CACHE[cache_key] = loaded
        logging.debug("[%s] index ready: rows=%s dim=%s", self.source, loaded.ntotal, loaded.d)
        return loaded

    def _shard_index(self, shard: dict[str, Any]) -> Any:
        if shard["index"] is None:
            shard["index"] = self._load_index(shard["index_path"])
        return shard["index"]

    def iter_physical_indexes(self) -> Any:
        """Yield ``(source_local_start, faiss_index)`` without materializing IndexShards.

        This is used by the GPU-sequential balanced retriever.  Keeping every
        physical shard separate prevents a large source such as PubMed from
        being reconstructed into one monolithic GPU flat index.
        """
        if not self._is_sharded:
            yield 0, self.index
            return
        for shard in self._shards:
            yield int(shard["start"]), self._shard_index(shard)

    @property
    def index(self) -> Any:
        if self._index is None:
            if not self._is_sharded:
                assert self.index_path is not None
                self._index = self._load_index(self.index_path)
            else:
                first_index = self._shard_index(self._shards[0])
                combined = faiss.IndexShards(first_index.d, False, True)
                for shard in self._shards:
                    combined.add_shard(self._shard_index(shard))
                self._index = combined
        return self._index

    def unload_index(self) -> None:
        if not self.cache_index:
            self._index = None
            for shard in self._shards:
                shard["index"] = None

    def search_ids(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        if not self._is_sharded:
            return self.index.search(query_vectors, top_k)
        if top_k <= 0:
            rows = len(query_vectors)
            return np.empty((rows, 0), dtype="float32"), np.empty((rows, 0), dtype="int64")
        score_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        for shard in self._shards:
            local_k = min(int(top_k), int(shard["rows"]))
            scores, local_ids = self._shard_index(shard).search(query_vectors, local_k)
            local_ids = local_ids.astype("int64", copy=False)
            valid = local_ids >= 0
            local_ids = local_ids.copy()
            local_ids[valid] += int(shard["start"])
            score_parts.append(scores)
            id_parts.append(local_ids)
        all_scores = np.concatenate(score_parts, axis=1)
        all_ids = np.concatenate(id_parts, axis=1)
        order = np.argsort(-all_scores, axis=1, kind="stable")[:, : min(int(top_k), all_scores.shape[1])]
        return np.take_along_axis(all_scores, order, axis=1), np.take_along_axis(all_ids, order, axis=1)

    def reconstruct(self, start: int, size: int) -> np.ndarray:
        start = int(start)
        size = int(size)
        if start < 0 or size < 0 or start + size > self.rows:
            raise IndexError(f"reconstruct range out of bounds for {self.source}: {start}+{size}")
        if not self._is_sharded:
            return np.ascontiguousarray(self.index.reconstruct_n(start, size), dtype="float32")
        if size == 0:
            return np.empty((0, int(self.index.d)), dtype="float32")
        end = start + size
        pieces: list[np.ndarray] = []
        for shard in self._shards:
            shard_start = int(shard["start"])
            shard_end = shard_start + int(shard["rows"])
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            if overlap_start >= overlap_end:
                continue
            pieces.append(
                np.ascontiguousarray(
                    self._shard_index(shard).reconstruct_n(overlap_start - shard_start, overlap_end - overlap_start),
                    dtype="float32",
                )
            )
        return np.ascontiguousarray(np.concatenate(pieces, axis=0), dtype="float32")

    def _metadata_for_local_id(self, local_id: int) -> tuple[MetadataStore, int]:
        if not self._is_sharded:
            assert self.metadata is not None
            return self.metadata, local_id
        shard_idx = bisect_right(self._shard_starts, local_id) - 1
        if shard_idx < 0 or shard_idx >= len(self._shards):
            raise IndexError(f"local_id out of range for {self.source}: {local_id}")
        shard = self._shards[shard_idx]
        offset = local_id - int(shard["start"])
        if offset < 0 or offset >= int(shard["rows"]):
            raise IndexError(f"local_id out of range for {self.source}: {local_id}")
        return shard["metadata"], offset

    def get_document(self, local_id: int, score: float) -> RetrievedDocument:
        local_id = int(local_id)
        metadata, shard_local_id = self._metadata_for_local_id(local_id)
        row = metadata.get(shard_local_id)
        return RetrievedDocument(
            source=str(row.get("source") or self.source),
            local_id=int(row.get("local_id", local_id)),
            db_id=str(row.get("db_id") or f"{self.source}:{local_id}"),
            corpus_id=row.get("corpus_id"),
            chunk_id=row.get("chunk_id"),
            doc_id=row.get("doc_id"),
            title=row.get("title"),
            text=str(row.get("text") or ""),
            retrieval_score=float(score),
            metadata=row.get("metadata") or {},
        )

    def search(self, query_vectors: np.ndarray, top_k: int) -> list[list[RetrievedDocument]]:
        scores, local_ids = self.search_ids(query_vectors, top_k)
        per_query: list[list[RetrievedDocument]] = []
        for query_scores, query_ids in zip(scores, local_ids):
            docs: list[RetrievedDocument] = []
            for score, local_id in zip(query_scores.tolist(), query_ids.tolist()):
                if int(local_id) < 0:
                    continue
                docs.append(self.get_document(local_id=int(local_id), score=float(score)))
            per_query.append(docs)
        return per_query

    def close(self) -> None:
        self.unload_index()
        if self.metadata is not None:
            self.metadata.close()
        for shard in self._shards:
            shard["metadata"].close()


class FaissMedCPTRetriever:
    def __init__(
        self,
        vector_db_root: Path,
        sources: list[str],
        per_source_top_k: int | None = None,
        keep_indexes_in_memory: bool = True,
        mmap_indexes: bool = True,
        metadata_row_cache_size: int = 50000,
        search_mode: str = "logical_shards",
        shard_threaded: bool = False,
        gpu_search_chunk_size: int = 500_000,
        gpu_search_device: str = "auto",
        gpu_search_dtype: str = "float16",
        faiss_gpu_device: int = 0,
        faiss_gpu_use_float16: bool = True,
        faiss_gpu_add_batch_size: int = 500_000,
        faiss_gpu_temp_memory_mb: int = 1024,
    ) -> None:
        self.vector_db_root = vector_db_root
        self.sources = sources
        self.per_source_top_k = per_source_top_k
        self.keep_indexes_in_memory = keep_indexes_in_memory
        self.mmap_indexes = mmap_indexes
        if search_mode not in {
            "logical_shards",
            "source_loop",
            "gpu_stream",
            "faiss_gpu",
            "faiss_gpu_source_loop",
            "faiss_gpu_source_sequential",
        }:
            raise ValueError(f"Unsupported retrieval search_mode: {search_mode}")
        self.search_mode = search_mode
        self.shard_threaded = bool(shard_threaded)
        self.gpu_search_chunk_size = max(1, int(gpu_search_chunk_size))
        self.gpu_search_device = gpu_search_device
        self.gpu_search_dtype = gpu_search_dtype
        self.faiss_gpu_device = int(faiss_gpu_device)
        self.faiss_gpu_use_float16 = bool(faiss_gpu_use_float16)
        self.faiss_gpu_add_batch_size = max(1, int(faiss_gpu_add_batch_size))
        self.faiss_gpu_temp_memory_mb = int(faiss_gpu_temp_memory_mb)
        self._indexes = {
            source: FaissSourceIndex(
                vector_db_root / source,
                cache_index=keep_indexes_in_memory,
                mmap_index=mmap_indexes,
                metadata_row_cache_size=metadata_row_cache_size,
            )
            for source in sources
        }
        self._sharded_index: Any | None = None
        self._shard_starts: list[int] = []
        self._shard_sources: list[str] = []
        self._gpu_resources: list[Any] = []
        self._gpu_sharded_index: Any | None = None
        self._gpu_shard_starts: list[int] = []
        self._gpu_shard_sources: list[str] = []
        self._gpu_source_indexes: dict[str, Any] = {}

    def _resolve_gpu_device(self) -> Any:
        if torch is None:
            raise ImportError("torch is required for retrieval_search_mode='gpu_stream'.")
        if self.gpu_search_device != "auto":
            device = torch.device(self.gpu_search_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError(
                "retrieval_search_mode='gpu_stream' requires a CUDA device. "
                "Use --retrieval-search-mode logical_shards for CPU FAISS search."
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        return device

    def _resolve_gpu_dtype(self) -> Any:
        if torch is None:
            raise ImportError("torch is required for retrieval_search_mode='gpu_stream'.")
        if self.gpu_search_dtype == "float16":
            return torch.float16
        if self.gpu_search_dtype == "bfloat16":
            return torch.bfloat16
        if self.gpu_search_dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported gpu_search_dtype: {self.gpu_search_dtype}")

    def _build_sharded_index(self) -> Any:
        if self._sharded_index is not None:
            return self._sharded_index
        if not self.sources:
            raise ValueError("No sources configured for FAISS retrieval.")
        first_index = self._indexes[self.sources[0]].index
        sharded = faiss.IndexShards(first_index.d, self.shard_threaded, True)
        starts: list[int] = []
        offset = 0
        for source in self.sources:
            index = self._indexes[source].index
            starts.append(offset)
            sharded.add_shard(index)
            offset += int(index.ntotal)
        self._shard_starts = starts
        self._shard_sources = list(self.sources)
        self._sharded_index = sharded
        logging.debug(
            "Logical FAISS shard index ready: sources=%s rows=%s threaded=%s",
            ",".join(self.sources),
            offset,
            self.shard_threaded,
        )
        return self._sharded_index

    def _require_faiss_gpu(self) -> None:
        if faiss is None:
            raise ImportError(
                "faiss import failed. If faiss-cpu and faiss-gpu are both installed, remove faiss-cpu "
                "and reinstall/import-check faiss-gpu."
            ) from FAISS_IMPORT_ERROR
        required = ["StandardGpuResources", "GpuIndexFlatConfig", "GpuIndexFlatIP"]
        missing = [name for name in required if not hasattr(faiss, name)]
        if missing:
            raise RuntimeError(
                "Current faiss module does not expose GPU APIs: "
                f"{missing}. Install a working faiss-gpu build in this venv."
            )
        if hasattr(faiss, "get_num_gpus") and int(faiss.get_num_gpus()) <= self.faiss_gpu_device:
            raise RuntimeError(
                f"faiss sees {faiss.get_num_gpus()} GPU(s), but faiss_gpu_device={self.faiss_gpu_device}."
            )

    def _configure_gpu_resource(self) -> Any:
        resource = faiss.StandardGpuResources()
        if self.faiss_gpu_temp_memory_mb <= 0:
            if hasattr(resource, "noTempMemory"):
                resource.noTempMemory()
        elif hasattr(resource, "setTempMemory"):
            resource.setTempMemory(int(self.faiss_gpu_temp_memory_mb) * 1024 * 1024)
        if hasattr(resource, "setDefaultNullStreamAllDevices"):
            resource.setDefaultNullStreamAllDevices()
        return resource

    def _gpu_flat_config(self) -> Any:
        config = faiss.GpuIndexFlatConfig()
        if hasattr(config, "device"):
            config.device = self.faiss_gpu_device
        if hasattr(config, "useFloat16"):
            config.useFloat16 = self.faiss_gpu_use_float16
        if hasattr(config, "storeTransposed"):
            config.storeTransposed = True
        return config

    def _copy_cpu_index_to_gpu_flat(self, source: str, source_index: Any, resource: Any) -> Any:
        gpu_index = faiss.GpuIndexFlatIP(resource, int(source_index.d), self._gpu_flat_config())
        rows = int(source_index.ntotal)
        logging.debug(
            "[%s] adding vectors to FAISS-GPU flat index: rows=%s batch_size=%s",
            source,
            rows,
            self.faiss_gpu_add_batch_size,
        )
        for start in range(0, rows, self.faiss_gpu_add_batch_size):
            size = min(self.faiss_gpu_add_batch_size, rows - start)
            vectors = np.ascontiguousarray(source_index.reconstruct_n(int(start), int(size)), dtype="float32")
            gpu_index.add(vectors)
        return gpu_index

    def _build_faiss_gpu_index(self) -> Any:
        if self._gpu_sharded_index is not None:
            return self._gpu_sharded_index
        self._require_faiss_gpu()
        if not self.sources:
            raise ValueError("No sources configured for FAISS retrieval.")

        first_index = self._indexes[self.sources[0]].index
        sharded = faiss.IndexShards(first_index.d, self.shard_threaded, True)
        starts: list[int] = []
        offset = 0
        self._gpu_resources = []
        logging.debug(
            "Preparing FAISS-GPU retrieval index: sources=%s device=%s float16=%s rows=loading...",
            len(self.sources),
            self.faiss_gpu_device,
            self.faiss_gpu_use_float16,
        )
        shared_resource = None if self.shard_threaded else self._configure_gpu_resource()
        if shared_resource is not None:
            self._gpu_resources.append(shared_resource)
        for source in self.sources:
            source_index = self._indexes[source].index
            resource = shared_resource or self._configure_gpu_resource()
            gpu_index = self._copy_cpu_index_to_gpu_flat(source, source_index, resource)
            self._gpu_source_indexes[source] = gpu_index
            starts.append(offset)
            sharded.add_shard(gpu_index)
            if shared_resource is None:
                self._gpu_resources.append(resource)
            offset += int(source_index.ntotal)
            logging.debug("[%s] FAISS-GPU shard ready: rows=%s cumulative=%s", source, source_index.ntotal, offset)

        self._gpu_shard_starts = starts
        self._gpu_shard_sources = list(self.sources)
        self._gpu_sharded_index = sharded
        logging.debug("FAISS-GPU retrieval index ready: sources=%s rows=%s dim=%s", len(self.sources), offset, first_index.d)
        return self._gpu_sharded_index

    def _build_faiss_gpu_source_indexes(self) -> dict[str, Any]:
        if self._gpu_source_indexes:
            return self._gpu_source_indexes
        self._require_faiss_gpu()
        if not self.sources:
            raise ValueError("No sources configured for FAISS retrieval.")

        self._gpu_resources = []
        shared_resource = self._configure_gpu_resource()
        self._gpu_resources.append(shared_resource)
        for source in self.sources:
            source_index = self._indexes[source].index
            self._gpu_source_indexes[source] = self._copy_cpu_index_to_gpu_flat(source, source_index, shared_resource)
        logging.debug("FAISS-GPU per-source indexes ready: sources=%s", len(self._gpu_source_indexes))
        return self._gpu_source_indexes

    def _resolve_gpu_global_id(self, global_id: int) -> tuple[str, int]:
        shard_idx = bisect_right(self._gpu_shard_starts, int(global_id)) - 1
        if shard_idx < 0 or shard_idx >= len(self._gpu_shard_sources):
            raise IndexError(f"global FAISS-GPU id out of range: {global_id}")
        source = self._gpu_shard_sources[shard_idx]
        local_id = int(global_id) - int(self._gpu_shard_starts[shard_idx])
        return source, local_id

    def _documents_from_gpu_search_rows(self, scores: np.ndarray, global_ids: np.ndarray) -> list[list[RetrievedDocument]]:
        merged: list[list[RetrievedDocument]] = []
        for query_scores, query_ids in zip(scores, global_ids):
            docs: list[RetrievedDocument] = []
            for score, global_id in zip(query_scores.tolist(), query_ids.tolist()):
                if int(global_id) < 0:
                    continue
                source, local_id = self._resolve_gpu_global_id(int(global_id))
                docs.append(self._indexes[source].get_document(local_id=local_id, score=float(score)))
            for rank, doc in enumerate(docs, start=1):
                doc.retrieval_rank = rank
            merged.append(docs)
        return merged

    def _documents_from_logical_search_rows(self, scores: np.ndarray, global_ids: np.ndarray) -> list[list[RetrievedDocument]]:
        merged: list[list[RetrievedDocument]] = []
        for query_scores, query_ids in zip(scores, global_ids):
            docs: list[RetrievedDocument] = []
            for score, global_id in zip(query_scores.tolist(), query_ids.tolist()):
                if int(global_id) < 0:
                    continue
                source, local_id = self._resolve_global_id(int(global_id))
                docs.append(self._indexes[source].get_document(local_id=local_id, score=float(score)))
            for rank, doc in enumerate(docs, start=1):
                doc.retrieval_rank = rank
            merged.append(docs)
        return merged

    def _iter_query_chunks(self, query_vectors: np.ndarray, progress_chunk_size: int | None) -> Any:
        chunk_size = int(progress_chunk_size or 0)
        if chunk_size <= 0 or chunk_size >= len(query_vectors):
            yield 0, len(query_vectors), query_vectors
            return
        for start in range(0, len(query_vectors), chunk_size):
            end = min(start + chunk_size, len(query_vectors))
            yield start, end, query_vectors[start:end]

    def _retrieve_batch_faiss_gpu(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        progress_callback: Callable[[int], None] | None = None,
        progress_chunk_size: int | None = None,
    ) -> list[list[RetrievedDocument]]:
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        index = self._build_faiss_gpu_index()
        merged: list[list[RetrievedDocument]] = []
        for _start, end, chunk in self._iter_query_chunks(query_vectors, progress_chunk_size):
            scores, global_ids = index.search(chunk, top_k)
            merged.extend(self._documents_from_gpu_search_rows(scores, global_ids))
            if progress_callback is not None:
                progress_callback(len(chunk))
        return merged

    def _retrieve_batch_faiss_gpu_source_loop(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        progress_callback: Callable[[int], None] | None = None,
        progress_chunk_size: int | None = None,
    ) -> list[list[RetrievedDocument]]:
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        gpu_indexes = self._build_faiss_gpu_source_indexes()
        per_source_top_k = self.per_source_top_k or top_k
        merged: list[list[RetrievedDocument]] = []
        for _start, _end, chunk in self._iter_query_chunks(query_vectors, progress_chunk_size):
            candidates: list[list[tuple[float, str, int]]] = [[] for _ in range(len(chunk))]
            for source in self.sources:
                scores, local_ids = gpu_indexes[source].search(chunk, per_source_top_k)
                for query_idx, (query_scores, query_ids) in enumerate(zip(scores, local_ids)):
                    for score, local_id in zip(query_scores.tolist(), query_ids.tolist()):
                        if int(local_id) < 0:
                            continue
                        candidates[query_idx].append((float(score), source, int(local_id)))

            for hits in candidates:
                hits.sort(key=lambda item: item[0], reverse=True)
                selected_hits = hits[:top_k]
                selected = [
                    self._indexes[source].get_document(local_id=local_id, score=score)
                    for score, source, local_id in selected_hits
                ]
                for rank, doc in enumerate(selected, start=1):
                    doc.retrieval_rank = rank
                merged.append(selected)
            if progress_callback is not None:
                progress_callback(len(chunk))
        return merged

    def _resolve_global_id(self, global_id: int) -> tuple[str, int]:
        shard_idx = bisect_right(self._shard_starts, int(global_id)) - 1
        if shard_idx < 0 or shard_idx >= len(self._shard_sources):
            raise IndexError(f"global FAISS id out of range: {global_id}")
        source = self._shard_sources[shard_idx]
        local_id = int(global_id) - int(self._shard_starts[shard_idx])
        return source, local_id

    def _retrieve_batch_logical_shards(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        progress_callback: Callable[[int], None] | None = None,
        progress_chunk_size: int | None = None,
    ) -> list[list[RetrievedDocument]]:
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        index = self._build_sharded_index()
        merged: list[list[RetrievedDocument]] = []
        for _start, end, chunk in self._iter_query_chunks(query_vectors, progress_chunk_size):
            scores, global_ids = index.search(chunk, top_k)
            merged.extend(self._documents_from_logical_search_rows(scores, global_ids))
            if progress_callback is not None:
                progress_callback(len(chunk))
        return merged

    def _retrieve_batch_gpu_stream(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        progress_callback: Callable[[int], None] | None = None,
        progress_chunk_size: int | None = None,
    ) -> list[list[RetrievedDocument]]:
        if top_k <= 0:
            return [[] for _ in range(len(query_vectors))]
        if self.per_source_top_k:
            return self._retrieve_batch_gpu_stream_source_loop(
                query_vectors=query_vectors,
                top_k=top_k,
                progress_callback=progress_callback,
                progress_chunk_size=progress_chunk_size,
            )
        device = self._resolve_gpu_device()
        dtype = self._resolve_gpu_dtype()
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        batch_size = int(query_vectors.shape[0])
        q = torch.as_tensor(query_vectors, device=device, dtype=dtype)

        best_scores = torch.full((batch_size, top_k), float("-inf"), device=device, dtype=torch.float32)
        best_local_ids = torch.full((batch_size, top_k), -1, device=device, dtype=torch.long)
        best_source_ids = torch.full((batch_size, top_k), -1, device=device, dtype=torch.long)
        total_rows = sum(int(self._indexes[source].rows) for source in self.sources)
        logging.debug(
            "GPU streaming exact search: queries=%s rows=%s top_k=%s chunk_size=%s dtype=%s device=%s",
            batch_size,
            total_rows,
            top_k,
            self.gpu_search_chunk_size,
            self.gpu_search_dtype,
            device,
        )

        with torch.inference_mode():
            for source_idx, source in enumerate(self.sources):
                source_index = self._indexes[source]
                rows = int(source_index.rows)
                logging.debug("[%s] GPU streaming source rows=%s", source, rows)
                for start in range(0, rows, self.gpu_search_chunk_size):
                    size = min(self.gpu_search_chunk_size, rows - start)
                    vectors_np = source_index.reconstruct(start, size)
                    vectors = torch.as_tensor(vectors_np, device=device, dtype=dtype)
                    scores = torch.matmul(q, vectors.T).float()
                    local_k = min(top_k, int(scores.shape[1]))
                    chunk_scores, chunk_pos = torch.topk(scores, k=local_k, dim=1)
                    chunk_local_ids = chunk_pos.long() + int(start)
                    chunk_source_ids = torch.full_like(chunk_local_ids, int(source_idx))

                    combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
                    combined_local_ids = torch.cat([best_local_ids, chunk_local_ids], dim=1)
                    combined_source_ids = torch.cat([best_source_ids, chunk_source_ids], dim=1)
                    best_scores, keep = torch.topk(combined_scores, k=top_k, dim=1)
                    best_local_ids = torch.gather(combined_local_ids, 1, keep)
                    best_source_ids = torch.gather(combined_source_ids, 1, keep)

                    del vectors_np, vectors, scores, chunk_scores, chunk_pos
                    del chunk_local_ids, chunk_source_ids, combined_scores, combined_local_ids, combined_source_ids
                if not self.keep_indexes_in_memory:
                    source_index.unload_index()

        score_rows = best_scores.cpu().numpy()
        local_id_rows = best_local_ids.cpu().numpy()
        source_id_rows = best_source_ids.cpu().numpy()
        source_names = list(self.sources)
        merged: list[list[RetrievedDocument]] = []
        for query_scores, query_local_ids, query_source_ids in zip(score_rows, local_id_rows, source_id_rows):
            docs: list[RetrievedDocument] = []
            for score, local_id, source_id in zip(query_scores.tolist(), query_local_ids.tolist(), query_source_ids.tolist()):
                if int(local_id) < 0 or int(source_id) < 0:
                    continue
                source = source_names[int(source_id)]
                docs.append(self._indexes[source].get_document(local_id=int(local_id), score=float(score)))
            for rank, doc in enumerate(docs, start=1):
                doc.retrieval_rank = rank
            merged.append(docs)
        del q, best_scores, best_local_ids, best_source_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback(len(query_vectors))
        return merged

    def _retrieve_batch_gpu_stream_source_loop(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        progress_callback: Callable[[int], None] | None = None,
        progress_chunk_size: int | None = None,
    ) -> list[list[RetrievedDocument]]:
        """Exact source-balanced GPU search without storing full source indexes on GPU.

        This mode is intended for very large corpora such as PubMed+PMC where
        faiss_gpu_source_loop would need to materialize all vectors in VRAM.
        It streams CPU FAISS vectors to CUDA in chunks, keeps top-k per source,
        then merges the per-source hits. The retrieval semantics match
        source_loop/faiss_gpu_source_loop when per_source_top_k is set.
        """
        device = self._resolve_gpu_device()
        dtype = self._resolve_gpu_dtype()
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        batch_size = int(query_vectors.shape[0])
        q = torch.as_tensor(query_vectors, device=device, dtype=dtype)
        per_source_top_k = self.per_source_top_k or top_k
        candidates: list[list[tuple[float, str, int]]] = [[] for _ in range(batch_size)]

        with torch.inference_mode():
            for source in self.sources:
                source_index = self._indexes[source]
                rows = int(source_index.rows)
                local_top_k = min(per_source_top_k, rows)
                best_scores = torch.full((batch_size, local_top_k), float("-inf"), device=device, dtype=torch.float32)
                best_local_ids = torch.full((batch_size, local_top_k), -1, device=device, dtype=torch.long)
                logging.debug(
                    "[%s] GPU streaming balanced source search: queries=%s rows=%s per_source_top_k=%s chunk_size=%s",
                    source,
                    batch_size,
                    rows,
                    local_top_k,
                    self.gpu_search_chunk_size,
                )

                for start in range(0, rows, self.gpu_search_chunk_size):
                    size = min(self.gpu_search_chunk_size, rows - start)
                    vectors_np = source_index.reconstruct(start, size)
                    vectors = torch.as_tensor(vectors_np, device=device, dtype=dtype)
                    scores = torch.matmul(q, vectors.T).float()
                    chunk_k = min(local_top_k, int(scores.shape[1]))
                    chunk_scores, chunk_pos = torch.topk(scores, k=chunk_k, dim=1)
                    chunk_local_ids = chunk_pos.long() + int(start)

                    combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
                    combined_local_ids = torch.cat([best_local_ids, chunk_local_ids], dim=1)
                    best_scores, keep = torch.topk(combined_scores, k=local_top_k, dim=1)
                    best_local_ids = torch.gather(combined_local_ids, 1, keep)

                    del vectors_np, vectors, scores, chunk_scores, chunk_pos
                    del chunk_local_ids, combined_scores, combined_local_ids

                score_rows = best_scores.cpu().numpy()
                local_id_rows = best_local_ids.cpu().numpy()
                for query_idx, (query_scores, query_ids) in enumerate(zip(score_rows, local_id_rows)):
                    for score, local_id in zip(query_scores.tolist(), query_ids.tolist()):
                        if int(local_id) < 0:
                            continue
                        candidates[query_idx].append((float(score), source, int(local_id)))

                del best_scores, best_local_ids, score_rows, local_id_rows
                if not self.keep_indexes_in_memory:
                    source_index.unload_index()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        merged: list[list[RetrievedDocument]] = []
        for hits in candidates:
            hits.sort(key=lambda item: item[0], reverse=True)
            selected_hits = hits[:top_k]
            selected = [
                self._indexes[source].get_document(local_id=local_id, score=score)
                for score, source, local_id in selected_hits
            ]
            for rank, doc in enumerate(selected, start=1):
                doc.retrieval_rank = rank
            merged.append(selected)

        del q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback(len(query_vectors))
        return merged

    def retrieve_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        progress_callback: Callable[[int], None] | None = None,
        progress_chunk_size: int | None = None,
    ) -> list[list[RetrievedDocument]]:
        if len(query_vectors) == 0:
            return []
        if self.search_mode == "faiss_gpu_source_loop":
            return self._retrieve_batch_faiss_gpu_source_loop(
                query_vectors,
                top_k=top_k,
                progress_callback=progress_callback,
                progress_chunk_size=progress_chunk_size,
            )
        if self.search_mode == "faiss_gpu":
            return self._retrieve_batch_faiss_gpu(
                query_vectors,
                top_k=top_k,
                progress_callback=progress_callback,
                progress_chunk_size=progress_chunk_size,
            )
        if self.search_mode == "gpu_stream":
            return self._retrieve_batch_gpu_stream(
                query_vectors,
                top_k=top_k,
                progress_callback=progress_callback,
                progress_chunk_size=progress_chunk_size,
            )
        if self.search_mode == "faiss_gpu_source_sequential":
            return self._retrieve_batch_gpu_stream_source_loop(
                query_vectors,
                top_k=top_k,
                progress_callback=progress_callback,
                progress_chunk_size=progress_chunk_size,
            )
        if self.search_mode == "logical_shards":
            return self._retrieve_batch_logical_shards(
                query_vectors,
                top_k=top_k,
                progress_callback=progress_callback,
                progress_chunk_size=progress_chunk_size,
            )

        per_source_top_k = self.per_source_top_k or top_k
        candidates: list[list[tuple[float, str, int]]] = [[] for _ in range(len(query_vectors))]
        for source in self.sources:
            source_index = self._indexes[source]
            scores, local_ids = source_index.search_ids(query_vectors, top_k=per_source_top_k)
            for query_idx, (query_scores, query_ids) in enumerate(zip(scores, local_ids)):
                for score, local_id in zip(query_scores.tolist(), query_ids.tolist()):
                    if int(local_id) < 0:
                        continue
                    candidates[query_idx].append((float(score), source, int(local_id)))
            if not self.keep_indexes_in_memory:
                source_index.unload_index()

        merged: list[list[RetrievedDocument]] = []
        for hits in candidates:
            hits.sort(key=lambda item: item[0], reverse=True)
            selected_hits = hits[:top_k]
            selected = [
                self._indexes[source].get_document(local_id=local_id, score=score)
                for score, source, local_id in selected_hits
            ]
            for rank, doc in enumerate(selected, start=1):
                doc.retrieval_rank = rank
            merged.append(selected)
        if progress_callback is not None:
            progress_callback(len(query_vectors))
        return merged

    def close(self) -> None:
        self._gpu_sharded_index = None
        self._gpu_source_indexes = {}
        self._gpu_resources = []
        self._sharded_index = None
        for source_index in self._indexes.values():
            source_index.close()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_sources(task: str, dataset: str, requested_sources: list[str]) -> list[str]:
    if requested_sources == ["all"]:
        return DEFAULT_SOURCES
    if requested_sources == ["auto"]:
        if task == "mcq":
            return MCQ_SOURCES
        if task == "open_ended":
            return [dataset] if dataset in {"bioasq", "covidqa", "mashqa", "pubmed"} else OPEN_ENDED_SOURCES
    return requested_sources

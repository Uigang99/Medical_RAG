from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError as exc:  # pragma: no cover - exercised only when faiss is absent.
    raise SystemExit(
        "faiss is required to build the vector DB. Install faiss-cpu or faiss-gpu in this environment."
    ) from exc

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback is fine.
    orjson = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in this project env.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models" / "MedCPT-Article-Encoder"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "databases" / "vector_db"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    input_path: Path
    expected_count_path: Path
    expected_count_key: tuple[str, ...]


SOURCE_SPECS: dict[str, SourceSpec] = {
    "pubmed": SourceSpec(
        name="pubmed",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "pubmed.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "manifest.json",
        expected_count_key=("sources", "pubmed", "chunks"),
    ),
    "pubmed26": SourceSpec(
        name="pubmed26",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2_pubmed26" / "pubmed.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2_pubmed26" / "manifest.json",
        expected_count_key=("sources", "pubmed", "chunks"),
    ),
    "statpearls": SourceSpec(
        name="statpearls",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "statpearls.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "manifest.json",
        expected_count_key=("sources", "statpearls", "chunks"),
    ),
    "textbooks": SourceSpec(
        name="textbooks",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "textbooks.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "manifest.json",
        expected_count_key=("sources", "textbooks", "chunks"),
    ),
    "wikipedia": SourceSpec(
        name="wikipedia",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "wikipedia.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "medcorp" / "manifest.json",
        expected_count_key=("sources", "wikipedia", "chunks"),
    ),
    "pmc": SourceSpec(
        name="pmc",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2" / "pmc.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2" / "manifest.json",
        expected_count_key=("sources", "pmc", "chunks"),
    ),
    "cpg": SourceSpec(
        name="cpg",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2" / "cpg.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2" / "manifest.json",
        expected_count_key=("sources", "cpg", "chunks"),
    ),
    "bioasq": SourceSpec(
        name="bioasq",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "open_ended" / "processed" / "bioasq" / "chunks.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "open_ended" / "processed" / "bioasq" / "summary.json",
        expected_count_key=("chunks",),
    ),
    "covidqa": SourceSpec(
        name="covidqa",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "open_ended" / "processed" / "covidqa" / "chunks.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "open_ended" / "processed" / "covidqa" / "summary.json",
        expected_count_key=("chunks",),
    ),
    "mashqa": SourceSpec(
        name="mashqa",
        input_path=PROJECT_ROOT / "datasets" / "corpus" / "open_ended" / "processed" / "mashqa" / "chunks.jsonl",
        expected_count_path=PROJECT_ROOT / "datasets" / "corpus" / "open_ended" / "processed" / "mashqa" / "summary.json",
        expected_count_key=("chunks",),
    ),
}

DEFAULT_SOURCE_ORDER = ["pubmed", "statpearls", "textbooks", "wikipedia", "bioasq", "covidqa", "mashqa"]
SUPPORTED_SOURCE_ORDER = [
    "pubmed",
    "pubmed26",
    "statpearls",
    "textbooks",
    "wikipedia",
    "pmc",
    "cpg",
    "bioasq",
    "covidqa",
    "mashqa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MedCPT Article Encoder FAISS vector DBs for MCQ/open-ended corpora, "
            "then create an ordered concatenated DB manifest."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-name", default="medcpt_article_encoder")
    parser.add_argument(
        "--source-layout",
        choices=["nested", "direct"],
        default="nested",
        help=(
            "Where to place source DB directories below OUTPUT_ROOT/INDEX_NAME. "
            "nested writes sources/<source> (legacy layout); direct writes <source> "
            "and is intended for the isolated RAG_Square corpus root."
        ),
    )
    parser.add_argument("--sources", nargs="+", choices=SUPPORTED_SOURCE_ORDER, default=DEFAULT_SOURCE_ORDER)
    parser.add_argument(
        "--input-path-override",
        nargs="*",
        default=[],
        metavar="SOURCE=JSONL_PATH",
        help=(
            "Use a different unified JSONL for one or more configured sources without changing its logical "
            "source name in the output DB. Example: cpg=/path/cpg.jsonl textbooks=/path/textbooks.jsonl"
        ),
    )
    parser.add_argument(
        "--expected-count-path-override",
        nargs="*",
        default=[],
        metavar="SOURCE=MANIFEST_PATH",
        help=(
            "Use a different manifest/summary when validating an overridden source input. The source's "
            "normal count key is retained (for RAG2 corpora: sources.<source>.chunks)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=256,
        help=(
            "Smallest micro-batch used by CUDA OOM recovery. A failed larger batch is split in half "
            "without losing source-build progress."
        ),
    )
    parser.add_argument(
        "--auto-recover-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On CUDA OOM, empty the cache and recursively split only the current batch instead of aborting.",
    )
    parser.add_argument(
        "--oom-backoff-threshold",
        type=int,
        default=2,
        help=(
            "Number of consecutive recovered OOM batches before lowering the steady-state batch size. "
            "Use 1 to back off immediately."
        ),
    )
    parser.add_argument(
        "--oom-recovery-probe-batches",
        type=int,
        default=128,
        help=(
            "After this many successful batches at a reduced size, retry a larger batch. "
            "Use 0 to keep the reduced size for the rest of the run."
        ),
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=8,
        help="Tokenizer padding multiple for Tensor Core-friendly CUDA shapes; use 0 to disable.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer length for MedCPT input. Longer chunks are truncated.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_000_000,
        help="Vectors per FAISS shard. Use 0 for one shard per source, but that can require huge RAM.",
    )
    parser.add_argument(
        "--show-shard-commits",
        action="store_true",
        help="Print one INFO line whenever a physical shard is committed. Disabled by default to keep tqdm on one line.",
    )
    parser.add_argument("--merge-batch-size", type=int, default=100_000)
    parser.add_argument(
        "--amp-dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="CUDA autocast dtype. auto uses bf16 when supported, otherwise fp16.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Weight dtype for model loading. auto uses bfloat16 on CUDA when available.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
        help=(
            "Attention backend for the BERT encoder. eager is the default because PyTorch SDPA can hit "
            "cuDNN Frontend 'No valid execution plans' errors on some H200/CUDA combinations."
        ),
    )
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before adding to FAISS.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing DB directories and discard resumable checkpoints.",
    )
    parser.add_argument("--skip-source-build", action="store_true", help="Only create merged manifests/indexes from existing sources.")
    parser.add_argument("--skip-logical-merge", action="store_true", help="Do not write the logical concatenated manifest.")
    parser.add_argument(
        "--physical-merge",
        action="store_true",
        help="Also materialize a merged FAISS DB by copying source vectors and metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug/smoke-test limit per source. Use a temp output root when setting this.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Do not fail if rows read differ from the source manifest/summary count.",
    )
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument(
        "--faiss-num-threads",
        type=int,
        default=0,
        help="FAISS OpenMP thread count; 0 leaves the FAISS default unchanged.",
    )
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
        try:
            return str(resolved.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(resolved)


def json_loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def json_dumps_line(obj: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def nested_get(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = obj
    for key in keys:
        current = current[key]
    return current


def expected_rows(spec: SourceSpec, limit: int | None) -> int | None:
    if not spec.expected_count_path.exists():
        return limit
    count = int(nested_get(read_json_file(spec.expected_count_path), spec.expected_count_key))
    return min(count, limit) if limit is not None else count


def parse_source_path_overrides(values: list[str], flag_name: str, allowed_sources: list[str]) -> dict[str, Path]:
    """Parse repeatable SOURCE=PATH CLI values without silently accepting typos."""
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag_name} must use SOURCE=PATH, got: {value!r}")
        source, raw_path = value.split("=", 1)
        source = source.strip()
        raw_path = raw_path.strip()
        if source not in allowed_sources:
            raise ValueError(f"{flag_name} refers to {source!r}, which is not listed in --sources.")
        if not raw_path:
            raise ValueError(f"{flag_name} has an empty path for source {source!r}.")
        if source in parsed:
            raise ValueError(f"{flag_name} repeats source {source!r}.")
        parsed[source] = Path(raw_path).expanduser().resolve()
    return parsed


def resolve_source_specs(args: argparse.Namespace) -> dict[str, SourceSpec]:
    """Apply input/manifest overrides while keeping FAISS source names stable.

    This makes it possible to build a fresh CPG/Textbook reproduction under a
    separate index root as logical sources named ``cpg`` and ``textbooks``.
    Existing databases and downstream balanced-retrieval source names remain
    unchanged.
    """
    input_overrides = parse_source_path_overrides(
        args.input_path_override,
        "--input-path-override",
        args.sources,
    )
    count_overrides = parse_source_path_overrides(
        args.expected_count_path_override,
        "--expected-count-path-override",
        args.sources,
    )
    resolved: dict[str, SourceSpec] = {}
    for source in args.sources:
        spec = SOURCE_SPECS[source]
        resolved[source] = replace(
            spec,
            input_path=input_overrides.get(source, spec.input_path),
            expected_count_path=count_overrides.get(source, spec.expected_count_path),
        )
    return resolved


def progress_bar(*args: Any, **kwargs: Any) -> Any:
    if tqdm is None:
        return nullcontext()
    return tqdm(*args, **kwargs)


def iter_jsonl_batches(
    path: Path,
    batch_size: int,
    limit: int | None,
    skip: int = 0,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    seen = 0
    with path.open("rb", buffering=16 * 1024 * 1024) as f:
        for line in f:
            if not line.strip():
                continue
            if seen < skip:
                seen += 1
                continue
            if limit is not None and seen >= limit:
                break
            batch.append(json_loads(line))
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def row_text(row: dict[str, Any]) -> str:
    text = row.get("text") or row.get("contents") or row.get("content") or ""
    if text is None:
        return ""
    return text if isinstance(text, str) else str(text)


def metadata_row(row: dict[str, Any], source: str, local_id: int) -> dict[str, Any]:
    original_source = row.get("source")
    dataset = row.get("dataset") or source
    chunk_id = row.get("chunk_id") or row.get("corpus_id") or f"{source}::{local_id}"
    text = row_text(row)
    return {
        "db_id": f"medcpt_article::{source}::{local_id:012d}",
        "local_id": local_id,
        "source": source,
        "dataset": dataset,
        "original_source": original_source,
        "corpus_id": row.get("corpus_id"),
        "chunk_id": chunk_id,
        "doc_id": row.get("doc_id"),
        "source_doc_id": row.get("source_doc_id") or row.get("doc_id"),
        "source_chunk_id": row.get("source_chunk_id") or chunk_id,
        "title": row.get("title") or "",
        "text": text,
        "metadata": row.get("metadata") or {},
    }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA available: %s", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return device
    logging.warning("CUDA is not visible. The script will run on CPU unless you launch it in a GPU-enabled session.")
    return torch.device("cpu")


def resolve_amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or requested == "none":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_model_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if requested == "float32" or device.type != "cuda":
        return None
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype | None) -> Any:
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_encoder(
    model_path: Path,
    device: torch.device,
    model_dtype: torch.dtype | None,
    attn_implementation: str,
) -> tuple[Any, torch.nn.Module]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    logging.info("Loading tokenizer: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    logging.info("Loading model: %s", model_path)
    kwargs: dict[str, Any] = {"local_files_only": True}
    if model_dtype is not None:
        kwargs["torch_dtype"] = model_dtype
    kwargs["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    logging.info(
        "Loaded MedCPT article encoder on %s (model_dtype=%s)",
        device,
        str(model_dtype).replace("torch.", "") if model_dtype is not None else "float32",
    )
    logging.info("Attention implementation: %s", attn_implementation)
    return tokenizer, model


def embed_texts(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    amp_dtype: torch.dtype | None,
    pad_to_multiple_of: int,
) -> np.ndarray:
    tokenizer_kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if pad_to_multiple_of > 1:
        tokenizer_kwargs["pad_to_multiple_of"] = pad_to_multiple_of
    encoded = tokenizer(
        texts,
        **tokenizer_kwargs,
    )
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    with torch.inference_mode(), autocast_context(device, amp_dtype):
        output = model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
    return np.ascontiguousarray(embeddings.float().cpu().numpy(), dtype="float32")


def is_cuda_oom(exc: BaseException) -> bool:
    """Recognize PyTorch OOM variants without masking unrelated runtime failures."""
    oom_type = getattr(torch, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def embed_texts_resilient(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    amp_dtype: torch.dtype | None,
    pad_to_multiple_of: int,
    min_batch_size: int,
    auto_recover_oom: bool,
) -> tuple[np.ndarray, int, bool]:
    """Embed one ordered batch, reducing only this batch when CUDA memory is tight.

    Returned ``safe_batch_size`` and ``recovered_from_oom`` allow the caller
    to distinguish a one-off outlier from a persistent capacity limit.  This
    lets an H200 start with an aggressive batch while retaining a deterministic
    no-data-loss fallback if another process temporarily occupies VRAM.
    """
    recover_from_oom = False
    try:
        return (
            embed_texts(
                texts=texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                amp_dtype=amp_dtype,
                pad_to_multiple_of=pad_to_multiple_of,
            ),
            len(texts),
            False,
        )
    except Exception as exc:
        recover_from_oom = auto_recover_oom and device.type == "cuda" and is_cuda_oom(exc)
        if not recover_from_oom:
            raise

    # We deliberately recover *after* leaving the except block.  Otherwise
    # Python's exception traceback can retain failed CUDA tensors while the
    # smaller retry is running, which defeats ``empty_cache`` on an OOM path.
    assert recover_from_oom
    if len(texts) <= min_batch_size:
        raise RuntimeError(
            f"CUDA OOM at the configured minimum batch size ({len(texts)}). "
            "Free GPU memory, reduce --min-batch-size, then rerun to resume from the last committed shard."
        )
    reduced = max(min_batch_size, len(texts) // 2)
    logging.warning("CUDA OOM at batch=%s; retrying current data as batches of at most %s", len(texts), reduced)
    torch.cuda.empty_cache()
    pieces: list[np.ndarray] = []
    safe_batch_size = reduced
    for start in range(0, len(texts), reduced):
        vectors, child_safe_size, _ = embed_texts_resilient(
            texts=texts[start : start + reduced],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            amp_dtype=amp_dtype,
            pad_to_multiple_of=pad_to_multiple_of,
            min_batch_size=min_batch_size,
            auto_recover_oom=auto_recover_oom,
        )
        pieces.append(vectors)
        safe_batch_size = min(safe_batch_size, child_safe_size)
    return np.ascontiguousarray(np.concatenate(pieces, axis=0), dtype="float32"), safe_batch_size, True


def make_index(dim: int) -> Any:
    return faiss.IndexFlatIP(dim)


def write_index(index: Any, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(path)


class ShardWriter:
    def __init__(self, output_dir: Path, dim: int, shard_size: int, log_prefix: str) -> None:
        self.output_dir = output_dir
        self.dim = dim
        self.shard_size = shard_size
        self.log_prefix = log_prefix
        self.shards_dir = output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.index = make_index(dim)
        self.shards: list[dict[str, Any]] = []
        self.current_rows = 0
        self.next_start = 0
        self.shard_id = 0

    def add(self, vectors: np.ndarray) -> None:
        pos = 0
        while pos < len(vectors):
            if self.shard_size > 0 and self.current_rows >= self.shard_size:
                self.flush()

            if self.shard_size > 0:
                capacity = self.shard_size - self.current_rows
                take = min(len(vectors) - pos, capacity)
            else:
                take = len(vectors) - pos

            self.index.add(vectors[pos : pos + take])
            self.current_rows += take
            pos += take

    def flush(self) -> None:
        if self.current_rows == 0:
            return
        shard_name = f"shard_{self.shard_id:05d}.faiss"
        shard_path = self.shards_dir / shard_name
        logging.info(
            "%s writing %s (%s vectors, local_id %s-%s)",
            self.log_prefix,
            shard_path,
            self.current_rows,
            self.next_start,
            self.next_start + self.current_rows - 1,
        )
        write_index(self.index, shard_path)
        self.shards.append(
            {
                "shard_id": self.shard_id,
                "path": str(Path("shards") / shard_name),
                "start_local_id": self.next_start,
                "rows": self.current_rows,
                "index_type": "IndexFlatIP",
            }
        )
        self.next_start += self.current_rows
        self.shard_id += 1
        self.index = make_index(self.dim)
        self.current_rows = 0
        gc.collect()

    @property
    def total_rows(self) -> int:
        return self.next_start + self.current_rows


def build_signature(
    spec: SourceSpec,
    args: argparse.Namespace,
    dim: int,
) -> dict[str, Any]:
    return {
        "source": spec.name,
        "input_path": rel(spec.input_path),
        "model_path": rel(args.model_path),
        "max_length": args.max_length,
        "normalize": bool(args.normalize),
        "dimension": dim,
        "shard_size": args.shard_size,
        "limit": args.limit,
    }


def state_path(source_dir: Path) -> Path:
    return source_dir / "build_state.json"


def cleanup_partial_files(source_dir: Path) -> None:
    for directory in [source_dir / "shards", source_dir / "metadata_shards"]:
        if not directory.exists():
            continue
        for path in directory.glob("*.tmp"):
            path.unlink()


def shard_rows(shards: list[dict[str, Any]]) -> int:
    return sum(int(shard["rows"]) for shard in shards)


def validate_committed_shards(
    source_dir: Path,
    shards: list[dict[str, Any]],
    require_metadata: bool = True,
) -> int:
    expected_start = 0
    for expected_id, shard in enumerate(shards):
        if int(shard["shard_id"]) != expected_id:
            raise RuntimeError(f"Non-contiguous shard id in {source_dir}: {shard}")
        if int(shard["start_local_id"]) != expected_start:
            raise RuntimeError(f"Non-contiguous shard start in {source_dir}: {shard}")
        if int(shard["rows"]) <= 0:
            raise RuntimeError(f"Empty committed shard in {source_dir}: {shard}")
        index_path = source_dir / shard["path"]
        if not index_path.exists():
            raise RuntimeError(f"Committed index shard is missing: {index_path}")
        if shard.get("metadata_path"):
            metadata_path = source_dir / shard["metadata_path"]
            if not metadata_path.exists():
                raise RuntimeError(f"Committed metadata shard is missing: {metadata_path}")
        elif require_metadata:
            raise RuntimeError(f"Committed shard has no metadata_path: {shard}")
        try:
            index = faiss.read_index(str(index_path))
            ntotal = int(index.ntotal)
            del index
        except Exception as exc:
            raise RuntimeError(f"Failed to read committed shard: {index_path}") from exc
        if ntotal != int(shard["rows"]):
            raise RuntimeError(f"Shard row mismatch in {index_path}: index={ntotal}, manifest={shard['rows']}")
        expected_start += int(shard["rows"])
    return expected_start


def write_build_state(
    source_dir: Path,
    spec: SourceSpec,
    args: argparse.Namespace,
    dim: int,
    shards: list[dict[str, Any]],
    status: str,
) -> None:
    committed_rows = shard_rows(shards)
    write_json_file(
        state_path(source_dir),
        {
            "type": "source_vector_db_build_state",
            "status": status,
            "updated_at": now_utc(),
            "signature": build_signature(spec, args, dim),
            "rows_committed": committed_rows,
            "next_local_id": committed_rows,
            "shards": shards,
        },
    )


def load_resumable_state(
    source_dir: Path,
    spec: SourceSpec,
    args: argparse.Namespace,
    dim: int,
) -> tuple[list[dict[str, Any]], int] | None:
    path = state_path(source_dir)
    if not path.exists():
        return None
    state = read_json_file(path)
    expected_signature = build_signature(spec, args, dim)
    if state.get("signature") != expected_signature:
        raise RuntimeError(
            f"{path} exists, but its build settings differ from the current command. "
            "Use --overwrite to rebuild, or rerun with the same --max-length/--normalize/--shard-size/--limit."
        )
    shards = list(state.get("shards", []))
    rows = validate_committed_shards(source_dir, shards)
    if rows != int(state.get("rows_committed", rows)):
        raise RuntimeError(f"{path} has inconsistent committed row counts.")
    return shards, rows


class SourceShardWriter:
    def __init__(
        self,
        output_dir: Path,
        spec: SourceSpec,
        args: argparse.Namespace,
        dim: int,
        existing_shards: list[dict[str, Any]],
    ) -> None:
        self.output_dir = output_dir
        self.spec = spec
        self.args = args
        self.dim = dim
        self.shard_size = args.shard_size
        self.shards_dir = output_dir / "shards"
        self.metadata_dir = output_dir / "metadata_shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.shards = list(existing_shards)
        self.next_start = shard_rows(self.shards)
        self.shard_id = len(self.shards)
        self.index = make_index(dim)
        self.current_rows = 0
        self.meta_out: Any | None = None
        self.meta_tmp_path: Path | None = None
        self._open_metadata_tmp()

    def _open_metadata_tmp(self) -> None:
        self.meta_tmp_path = self.metadata_dir / f"metadata_{self.shard_id:05d}.jsonl.tmp"
        self.meta_out = self.meta_tmp_path.open("wb", buffering=16 * 1024 * 1024)

    def add(self, vectors: np.ndarray, rows: list[dict[str, Any]]) -> None:
        if len(vectors) != len(rows):
            raise ValueError(f"Vector/metadata length mismatch: {len(vectors)} != {len(rows)}")
        pos = 0
        while pos < len(vectors):
            if self.shard_size > 0 and self.current_rows >= self.shard_size:
                self.flush()

            if self.shard_size > 0:
                capacity = self.shard_size - self.current_rows
                take = min(len(vectors) - pos, capacity)
            else:
                take = len(vectors) - pos

            self.index.add(vectors[pos : pos + take])
            assert self.meta_out is not None
            for row in rows[pos : pos + take]:
                self.meta_out.write(json_dumps_line(row))
            self.current_rows += take
            pos += take

    def flush(self) -> None:
        if self.current_rows == 0:
            return
        assert self.meta_out is not None and self.meta_tmp_path is not None
        self.meta_out.close()
        self.meta_out = None

        index_name = f"shard_{self.shard_id:05d}.faiss"
        metadata_name = f"metadata_{self.shard_id:05d}.jsonl"
        index_path = self.shards_dir / index_name
        metadata_path = self.metadata_dir / metadata_name
        shard_commit_log = logging.info if self.args.show_shard_commits else logging.debug
        shard_commit_log(
            "[%s] committing shard %05d (%s vectors, local_id %s-%s)",
            self.spec.name,
            self.shard_id,
            self.current_rows,
            self.next_start,
            self.next_start + self.current_rows - 1,
        )
        write_index(self.index, index_path)
        self.meta_tmp_path.replace(metadata_path)
        shard = {
            "shard_id": self.shard_id,
            "path": str(Path("shards") / index_name),
            "metadata_path": str(Path("metadata_shards") / metadata_name),
            "start_local_id": self.next_start,
            "rows": self.current_rows,
            "index_type": "IndexFlatIP",
        }
        self.shards.append(shard)
        write_build_state(self.output_dir, self.spec, self.args, self.dim, self.shards, status="in_progress")

        self.next_start += self.current_rows
        self.shard_id += 1
        self.index = make_index(self.dim)
        self.current_rows = 0
        gc.collect()
        self._open_metadata_tmp()

    def close(self) -> None:
        if self.meta_out is not None:
            self.meta_out.close()
            self.meta_out = None
        if self.current_rows == 0 and self.meta_tmp_path is not None and self.meta_tmp_path.exists():
            self.meta_tmp_path.unlink()

    @property
    def total_rows(self) -> int:
        return self.next_start + self.current_rows


def valid_source_db(source_dir: Path, expected: int | None) -> bool:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json_file(manifest_path)
    except Exception:
        return False
    rows = manifest.get("rows")
    if expected is not None and rows != expected:
        return False
    shards = manifest.get("index", {}).get("shards", [])
    if not shards and rows:
        return False
    metadata = manifest.get("metadata")
    require_shard_metadata = isinstance(metadata, dict) and metadata.get("layout") == "sharded"
    try:
        validate_committed_shards(source_dir, shards, require_metadata=require_shard_metadata)
    except Exception:
        return False
    if isinstance(metadata, dict) and metadata.get("layout") == "sharded":
        for shard in metadata.get("shards", []):
            if not (source_dir / shard["path"]).exists():
                return False
    elif manifest.get("metadata_path"):
        if not (source_dir / manifest["metadata_path"]).exists():
            return False
    else:
        return False
    return True


def build_source_db(
    spec: SourceSpec,
    source_dir: Path,
    expected: int | None,
    args: argparse.Namespace,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    dim: int,
) -> dict[str, Any]:
    if not spec.input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL for {spec.name}: {spec.input_path}")

    source_dir_had_contents = source_dir.exists() and any(source_dir.iterdir())
    if source_dir.exists():
        if valid_source_db(source_dir, expected) and not args.overwrite:
            logging.info("[%s] existing DB is valid; skipping source build.", spec.name)
            return read_json_file(source_dir / "manifest.json")
        if args.overwrite:
            shutil.rmtree(source_dir)
            source_dir_had_contents = False

    source_dir.mkdir(parents=True, exist_ok=True)
    if source_dir_had_contents and not state_path(source_dir).exists():
        raise RuntimeError(
            f"{source_dir} exists but has no resumable build_state.json. "
            "Rerun with --overwrite to rebuild this source safely."
        )
    cleanup_partial_files(source_dir)
    resume_state = load_resumable_state(source_dir, spec, args, dim)
    if resume_state is None:
        existing_shards: list[dict[str, Any]] = []
        rows_written = 0
        write_build_state(source_dir, spec, args, dim, existing_shards, status="in_progress")
    else:
        existing_shards, rows_written = resume_state
        logging.info(
            "[%s] resuming from checkpoint: %s committed rows across %s shard(s)",
            spec.name,
            rows_written,
            len(existing_shards),
        )

    shard_writer = SourceShardWriter(
        output_dir=source_dir,
        spec=spec,
        args=args,
        dim=dim,
        existing_shards=existing_shards,
    )

    logging.info(
        "[%s] building source DB from %s (expected rows=%s, resume_from=%s, batch_size=%s, min_batch_size=%s, shard_size=%s)",
        spec.name,
        spec.input_path,
        expected if expected is not None else "unknown",
        rows_written,
        args.batch_size,
        args.min_batch_size,
        args.shard_size if args.shard_size > 0 else "single",
    )

    pbar = progress_bar(
        total=expected,
        initial=rows_written,
        desc=f"embed:{spec.name}",
        unit="chunk",
        dynamic_ncols=True,
        smoothing=0.05,
        mininterval=1.0,
        maxinterval=10.0,
    )
    target_batch_size = args.batch_size
    active_batch_size = target_batch_size
    recovered_oom_streak = 0
    successful_reduced_batches = 0
    pending_probe_previous_batch_size: int | None = None
    try:
        for super_batch in iter_jsonl_batches(spec.input_path, args.batch_size, args.limit, skip=rows_written):
            # Keep the source reader at the requested large batch size, but
            # process it as adaptive micro-batches after a recoverable OOM.
            # This avoids rereading or reordering any source rows.
            start = 0
            while start < len(super_batch):
                batch = super_batch[start : start + active_batch_size]
                texts = [row_text(row) for row in batch]
                vectors, discovered_safe_batch_size, recovered_from_oom = embed_texts_resilient(
                    texts=texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=args.max_length,
                    amp_dtype=amp_dtype,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                    min_batch_size=args.min_batch_size,
                    auto_recover_oom=args.auto_recover_oom,
                )
                start += len(batch)

                if recovered_from_oom:
                    successful_reduced_batches = 0
                    if pending_probe_previous_batch_size is not None:
                        # A recovery probe failed.  The batch itself was still
                        # completed by the resilient splitter, but do not make
                        # the next normal batch fail again just to reconfirm a
                        # known safe size.
                        active_batch_size = pending_probe_previous_batch_size
                        pending_probe_previous_batch_size = None
                        recovered_oom_streak = 0
                        logging.info(
                            "[%s] larger-batch probe did not fit; returning to stable batch size=%s",
                            spec.name,
                            active_batch_size,
                        )
                    else:
                        recovered_oom_streak += 1
                        if (
                            recovered_oom_streak >= args.oom_backoff_threshold
                            and discovered_safe_batch_size < active_batch_size
                        ):
                            active_batch_size = discovered_safe_batch_size
                            recovered_oom_streak = 0
                            logging.info(
                                "[%s] persistent OOM detected; lowering steady batch size to %s",
                                spec.name,
                                active_batch_size,
                            )
                else:
                    recovered_oom_streak = 0
                    if pending_probe_previous_batch_size is not None and len(batch) == active_batch_size:
                        logging.info("[%s] larger-batch probe succeeded; retaining batch size=%s", spec.name, active_batch_size)
                        pending_probe_previous_batch_size = None
                    if active_batch_size < target_batch_size and args.oom_recovery_probe_batches > 0:
                        successful_reduced_batches += 1
                        if successful_reduced_batches >= args.oom_recovery_probe_batches:
                            previous_batch_size = active_batch_size
                            active_batch_size = min(target_batch_size, active_batch_size * 2)
                            pending_probe_previous_batch_size = previous_batch_size
                            successful_reduced_batches = 0
                            logging.info(
                                "[%s] %s stable reduced batches; probing batch size %s -> %s",
                                spec.name,
                                args.oom_recovery_probe_batches,
                                previous_batch_size,
                                active_batch_size,
                            )
                if args.normalize:
                    faiss.normalize_L2(vectors)

                rows = []
                for row in batch:
                    rows.append(metadata_row(row, spec.name, rows_written))
                    rows_written += 1
                shard_writer.add(vectors, rows)

                if tqdm is not None:
                    pbar.update(len(batch))
                    pbar.set_postfix(batch=active_batch_size, shards=len(shard_writer.shards))
                elif rows_written % 100_000 == 0:
                    logging.info("[%s] embedded %s chunks (batch=%s)", spec.name, rows_written, active_batch_size)
    except Exception:
        shard_writer.close()
        raise
    finally:
        if tqdm is not None:
            pbar.close()

    shard_writer.flush()
    shard_writer.close()

    if expected is not None and rows_written != expected and not args.allow_count_mismatch:
        raise RuntimeError(
            f"[{spec.name}] row-count mismatch: read {rows_written}, expected {expected}. "
            "Use --allow-count-mismatch only if this is intentional."
        )

    manifest = {
        "type": "source_vector_db",
        "source": spec.name,
        "input_path": rel(spec.input_path),
        "rows": rows_written,
        "expected_rows": expected,
        "created_at": now_utc(),
        "model": {
            "name": "MedCPT-Article-Encoder",
            "path": rel(args.model_path),
            "embedding_field": "last_hidden_state[:, 0, :]",
            "max_length": args.max_length,
            "normalize": bool(args.normalize),
        },
        "index": {
            "backend": "faiss",
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "dimension": dim,
            "shard_size": args.shard_size,
            "shards": shard_writer.shards,
        },
        "metadata": {
            "layout": "sharded",
            "shards": [
                {
                    "shard_id": shard["shard_id"],
                    "path": shard["metadata_path"],
                    "start_local_id": shard["start_local_id"],
                    "rows": shard["rows"],
                }
                for shard in shard_writer.shards
            ],
        },
        "build": {
            "batch_size": args.batch_size,
            "minimum_batch_size": args.min_batch_size,
            "auto_recover_oom": bool(args.auto_recover_oom),
            "oom_backoff_threshold": args.oom_backoff_threshold,
            "oom_recovery_probe_batches": args.oom_recovery_probe_batches,
            "pad_to_multiple_of": args.pad_to_multiple_of,
            "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
            "attn_implementation": args.attn_implementation,
            "device": str(device),
            "limit": args.limit,
        },
    }
    write_json_file(source_dir / "manifest.json", manifest)
    write_build_state(source_dir, spec, args, dim, shard_writer.shards, status="complete")
    logging.info("[%s] done: %s rows, %s shard(s)", spec.name, rows_written, len(shard_writer.shards))
    return manifest


def source_db_dir(base_dir: Path, source: str, source_layout: str = "nested") -> Path:
    if source_layout == "direct":
        return base_dir / source
    if source_layout == "nested":
        return base_dir / "sources" / source
    raise ValueError(f"Unsupported source layout: {source_layout}")


def metadata_shards_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = manifest.get("metadata")
    if isinstance(metadata, dict) and metadata.get("layout") == "sharded":
        return list(metadata.get("shards", []))
    metadata_path = manifest.get("metadata_path")
    if metadata_path:
        return [
            {
                "shard_id": 0,
                "path": metadata_path,
                "start_local_id": 0,
                "rows": int(manifest["rows"]),
            }
        ]
    raise RuntimeError("Manifest does not define a readable metadata layout.")


def iter_metadata_rows(source_dir: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for shard in metadata_shards_from_manifest(manifest):
        metadata_path = source_dir / shard["path"]
        with metadata_path.open("rb", buffering=16 * 1024 * 1024) as meta_in:
            for line in meta_in:
                if line.strip():
                    yield json_loads(line)


def create_logical_merged_manifest(
    base_dir: Path,
    sources: list[str],
    source_layout: str = "nested",
) -> dict[str, Any]:
    merged_dir = base_dir / "merged" / "all_sources"
    merged_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    entries: list[dict[str, Any]] = []
    dim: int | None = None
    normalize: bool | None = None
    for order, source in enumerate(sources):
        src_dir = source_db_dir(base_dir, source, source_layout)
        manifest = read_json_file(src_dir / "manifest.json")
        source_dim = manifest["index"]["dimension"]
        source_normalize = bool(manifest["model"]["normalize"])
        if dim is None:
            dim = source_dim
            normalize = source_normalize
        elif dim != source_dim or normalize != source_normalize:
            raise RuntimeError(f"Cannot merge {source}: index dimension/normalization differs.")
        rows = int(manifest["rows"])
        entries.append(
            {
                "order": order,
                "source": source,
                "global_start_id": offset,
                "global_end_id": offset + rows - 1,
                "rows": rows,
                "source_db_dir": rel(src_dir),
                "source_manifest": rel(src_dir / "manifest.json"),
                "source_metadata_layout": (manifest.get("metadata") or {}).get("layout", "single_jsonl"),
                "source_metadata_shards": metadata_shards_from_manifest(manifest),
                "source_shards": manifest["index"]["shards"],
            }
        )
        offset += rows

    merged_manifest = {
        "type": "merged_vector_db",
        "merge_mode": "logical_ordered_concatenation",
        "description": (
            "This merged DB keeps source FAISS shards in place and assigns global IDs by source order. "
            "Searching all listed shards and merging scores is equivalent to searching a physically "
            "concatenated IndexFlatIP over the same vectors."
        ),
        "created_at": now_utc(),
        "rows": offset,
        "dimension": dim,
        "sources": entries,
        "metadata": {
            "global_id_rule": "global_id = source.global_start_id + source-local local_id",
            "source_filter_field": "source",
            "metadata_file_per_source": "metadata.jsonl",
        },
    }
    write_json_file(merged_dir / "manifest.json", merged_manifest)
    write_json_file(
        merged_dir / "source_offsets.json",
        {
            entry["source"]: {
                "global_start_id": entry["global_start_id"],
                "global_end_id": entry["global_end_id"],
                "rows": entry["rows"],
                "source_db_dir": entry["source_db_dir"],
            }
            for entry in entries
        },
    )
    logging.info("[merged:logical] done: %s rows across %s sources", offset, len(entries))
    return merged_manifest


def add_vectors_to_physical_shards(
    writer: ShardWriter,
    input_index: Any,
    merge_batch_size: int,
    pbar: Any,
) -> None:
    total = int(input_index.ntotal)
    for start in range(0, total, merge_batch_size):
        size = min(merge_batch_size, total - start)
        vectors = input_index.reconstruct_n(start, size)
        writer.add(np.ascontiguousarray(vectors, dtype="float32"))
        if tqdm is not None:
            pbar.update(size)


def create_physical_merged_db(
    base_dir: Path,
    sources: list[str],
    merge_batch_size: int,
    shard_size: int,
    overwrite: bool,
    source_layout: str = "nested",
) -> dict[str, Any]:
    merged_dir = base_dir / "merged" / "all_sources_physical"
    if merged_dir.exists():
        if not overwrite:
            raise RuntimeError(f"{merged_dir} already exists. Rerun with --overwrite to rebuild it.")
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    source_manifests = [
        (source, read_json_file(source_db_dir(base_dir, source, source_layout) / "manifest.json"))
        for source in sources
    ]
    if not source_manifests:
        raise RuntimeError("No source manifests found for physical merge.")
    dim = int(source_manifests[0][1]["index"]["dimension"])
    total_rows = sum(int(manifest["rows"]) for _, manifest in source_manifests)
    writer = ShardWriter(merged_dir, dim=dim, shard_size=shard_size, log_prefix="[merged:physical]")

    logging.info("[merged:physical] copying vectors into merged shards (rows=%s)", total_rows)
    pbar = progress_bar(
        total=total_rows,
        desc="merge:vectors",
        unit="vec",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    try:
        for source, manifest in source_manifests:
            src_dir = source_db_dir(base_dir, source, source_layout)
            for shard in manifest["index"]["shards"]:
                shard_path = src_dir / shard["path"]
                logging.info("[merged:physical] reading %s", shard_path)
                input_index = faiss.read_index(str(shard_path))
                add_vectors_to_physical_shards(writer, input_index, merge_batch_size, pbar)
                del input_index
                gc.collect()
    finally:
        if tqdm is not None:
            pbar.close()
    writer.flush()

    metadata_tmp = merged_dir / "metadata.jsonl.tmp"
    metadata_final = merged_dir / "metadata.jsonl"
    logging.info("[merged:physical] copying metadata into %s", metadata_final)
    global_id = 0
    source_entries: list[dict[str, Any]] = []
    pbar_meta = progress_bar(
        total=total_rows,
        desc="merge:metadata",
        unit="row",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    with metadata_tmp.open("wb", buffering=16 * 1024 * 1024) as out:
        try:
            for order, (source, manifest) in enumerate(source_manifests):
                source_start = global_id
                src_dir = source_db_dir(base_dir, source, source_layout)
                for row in iter_metadata_rows(src_dir, manifest):
                    row["global_id"] = global_id
                    row["source_local_id"] = row.get("local_id")
                    row["source_global_start_id"] = source_start
                    out.write(json_dumps_line(row))
                    global_id += 1
                    if tqdm is not None:
                        pbar_meta.update(1)
                source_entries.append(
                    {
                        "order": order,
                        "source": source,
                        "global_start_id": source_start,
                        "global_end_id": global_id - 1,
                        "rows": int(manifest["rows"]),
                    }
                )
        finally:
            if tqdm is not None:
                pbar_meta.close()
    metadata_tmp.replace(metadata_final)

    manifest = {
        "type": "merged_vector_db",
        "merge_mode": "physical_sharded_concatenation",
        "created_at": now_utc(),
        "rows": global_id,
        "metadata_path": "metadata.jsonl",
        "dimension": dim,
        "sources": source_entries,
        "index": {
            "backend": "faiss",
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "dimension": dim,
            "shard_size": shard_size,
            "shards": writer.shards,
        },
    }
    write_json_file(merged_dir / "manifest.json", manifest)
    logging.info("[merged:physical] done: %s rows, %s shard(s)", global_id, len(writer.shards))
    return manifest


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_batch_size <= 0:
        raise ValueError("--min-batch-size must be positive")
    if args.min_batch_size > args.batch_size:
        raise ValueError("--min-batch-size must be less than or equal to --batch-size")
    if args.pad_to_multiple_of < 0:
        raise ValueError("--pad-to-multiple-of must be non-negative")
    if args.oom_backoff_threshold <= 0:
        raise ValueError("--oom-backoff-threshold must be positive")
    if args.oom_recovery_probe_batches < 0:
        raise ValueError("--oom-recovery-probe-batches must be non-negative")
    if args.faiss_num_threads < 0:
        raise ValueError("--faiss-num-threads must be non-negative")

    source_specs = resolve_source_specs(args)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        logging.info("PyTorch CPU threads: %s", args.torch_num_threads)
    if args.faiss_num_threads > 0:
        faiss.omp_set_num_threads(args.faiss_num_threads)
        logging.info("FAISS OpenMP threads: %s", args.faiss_num_threads)

    base_dir = args.output_root.resolve() / args.index_name
    base_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output base: %s", base_dir)

    device = choose_device()
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    model_dtype = resolve_model_dtype(args.model_dtype, device)

    tokenizer = None
    model = None
    dim: int | None = None
    if not args.skip_source_build:
        tokenizer, model = load_encoder(
            args.model_path.resolve(),
            device=device,
            model_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
        )
        dim = int(getattr(model.config, "hidden_size"))
        logging.info("Embedding dimension: %s", dim)

    for source in args.sources:
        spec = source_specs[source]
        expected = expected_rows(spec, args.limit)
        src_dir = source_db_dir(base_dir, source, args.source_layout)
        if args.skip_source_build:
            if not valid_source_db(src_dir, expected):
                raise RuntimeError(f"Source DB is missing or invalid: {src_dir}")
            logging.info("[%s] source build skipped; existing DB is valid.", source)
            continue
        assert tokenizer is not None and model is not None and dim is not None
        build_source_db(
            spec=spec,
            source_dir=src_dir,
            expected=expected,
            args=args,
            tokenizer=tokenizer,
            model=model,
            device=device,
            amp_dtype=amp_dtype,
            dim=dim,
        )

    if not args.skip_logical_merge:
        create_logical_merged_manifest(base_dir, args.sources, source_layout=args.source_layout)

    if args.physical_merge:
        create_physical_merged_db(
            base_dir=base_dir,
            sources=args.sources,
            merge_batch_size=args.merge_batch_size,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
            source_layout=args.source_layout,
        )

    logging.info("Vector DB build script finished.")


if __name__ == "__main__":
    main()

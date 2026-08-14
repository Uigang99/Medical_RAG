from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "raw" / "rag2" / "pmc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2"
SPACE_RE = re.compile(r"\s+")
MARKER_RE = re.compile(r"^====\s*(.*?)\s*$", re.MULTILINE)
REFERENCE_HEADING_RE = re.compile(r"\n(?:references|bibliography|literature cited)\s*\n", re.IGNORECASE)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return SPACE_RE.sub(" ", str(text or "")).strip()


def token_count(text: str) -> int:
    text = text.strip()
    return 0 if not text else len(text.split())


def json_dumps(row: dict[str, Any]) -> str:
    if orjson is not None:
        return orjson.dumps(row).decode("utf-8")
    return json.dumps(row, ensure_ascii=False)


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json_dumps(row))
    handle.write("\n")


class TokenStats:
    def __init__(self) -> None:
        self.rows = 0
        self.docs = 0
        self.skipped_docs = 0
        self.total_tokens = 0
        self.min_tokens: int | None = None
        self.max_tokens: int | None = None
        self.hist: Counter[int] = Counter()

    def add_doc(self) -> None:
        self.docs += 1

    def add_skipped_doc(self) -> None:
        self.skipped_docs += 1

    def add_chunk(self, n_tokens: int) -> None:
        self.rows += 1
        self.total_tokens += n_tokens
        self.hist[n_tokens] += 1
        if self.min_tokens is None or n_tokens < self.min_tokens:
            self.min_tokens = n_tokens
        if self.max_tokens is None or n_tokens > self.max_tokens:
            self.max_tokens = n_tokens

    def median(self) -> float:
        if self.rows == 0:
            return 0.0

        def kth_value(k: int) -> int:
            seen = 0
            for value in sorted(self.hist):
                seen += self.hist[value]
                if seen >= k:
                    return value
            raise RuntimeError("median histogram exhausted")

        if self.rows % 2 == 1:
            return float(kth_value(self.rows // 2 + 1))
        return (kth_value(self.rows // 2) + kth_value(self.rows // 2 + 1)) / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_docs": self.docs,
            "skipped_docs": self.skipped_docs,
            "chunks": self.rows,
            "total_tokens": self.total_tokens,
            "min_tokens": self.min_tokens or 0,
            "max_tokens": self.max_tokens or 0,
            "mean_tokens": self.total_tokens / self.rows if self.rows else 0.0,
            "median_tokens": self.median(),
        }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def iter_word_chunks(text: str, max_words: int, overlap_words: int) -> Iterator[tuple[int, str]]:
    words = text.split()
    if not words:
        return
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    step = max_words - overlap_words
    for idx, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            yield idx, " ".join(chunk_words)
        if start + max_words >= len(words):
            break


def extract_pmc_body(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if "==== Body" in text:
        text = text.split("==== Body", 1)[1]
    for marker in ["==== Refs", "==== Back", "==== Floats", "==== Supplementary"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    match = REFERENCE_HEADING_RE.search(text)
    if match:
        text = text[: match.start()]
    text = MARKER_RE.sub("\n", text)
    return clean_text(text)


def infer_title(row: dict[str, Any]) -> str:
    citation = clean_text(row.get("citation"))
    if not citation:
        return ""
    # The PMC legacy TXT files do not expose a clean title in the file list.
    # Keep citation as metadata, but avoid injecting noisy journal metadata into the text.
    return ""


def update_manifest(output_dir: Path, source_summary: dict[str, Any], max_words: int, overlap_words: int, raw_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    existing_sources: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_sources = dict(json.loads(manifest_path.read_text(encoding="utf-8")).get("sources") or {})
        except json.JSONDecodeError:
            logging.warning("Existing manifest is invalid and will be replaced: %s", manifest_path)
    existing_sources["pmc"] = source_summary
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": rel(raw_dir.parent),
        "output_dir": rel(output_dir),
        "dataset": "rag2",
        "schema": {
            "id_fields": ["corpus_id", "chunk_id", "source_chunk_id", "doc_id", "source_doc_id"],
            "text_field": "text",
            "source_field": "source",
            "metadata": "Original source identifiers and source-specific fields are preserved here.",
        },
        "token_count": {
            "method": "whitespace",
            "field": "text",
        },
        "chunking": {
            "unit": "word",
            "max_words": max_words,
            "overlap_words": overlap_words,
            "note": "Matches the Self-BioRAG/RAG2 description: 128-word chunks with 32-word overlap by default.",
        },
        "sources": existing_sources,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_unified(
    raw_dir: Path,
    output_dir: Path,
    max_words: int,
    overlap_words: int,
    fail_on_missing: bool,
    limit_docs: int | None,
) -> dict[str, Any]:
    selected_manifest = raw_dir / "selected_pmc_docs.jsonl"
    if not selected_manifest.exists():
        raise FileNotFoundError(f"Missing selected PMC manifest: {selected_manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pmc.jsonl"
    missing_path = output_dir / "pmc_missing_raw_docs.jsonl"
    stats = TokenStats()
    missing_rows: list[dict[str, Any]] = []
    total_docs = count_lines(selected_manifest)
    if limit_docs is not None:
        total_docs = min(total_docs, limit_docs)

    iterator = iter_jsonl(selected_manifest)
    progress = tqdm(total=total_docs, desc="unify:pmc", unit="doc") if tqdm else None

    with output_path.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(iterator):
            if limit_docs is not None and row_idx >= limit_docs:
                break
            local_path = PROJECT_ROOT / row["local_path"]
            if not local_path.exists() or local_path.stat().st_size == 0:
                missing_rows.append(row)
                stats.add_skipped_doc()
                if progress:
                    progress.update(1)
                continue

            raw_text = local_path.read_text(encoding="utf-8", errors="replace")
            body = extract_pmc_body(raw_text)
            if not body:
                stats.add_skipped_doc()
                if progress:
                    progress.update(1)
                continue

            accession_id = clean_text(row.get("accession_id"))
            doc_id = f"pmc:{accession_id}"
            title = infer_title(row)
            stats.add_doc()
            for local_idx, chunk in iter_word_chunks(body, max_words=max_words, overlap_words=overlap_words):
                source_chunk_id = f"{accession_id}_{local_idx}"
                corpus_id = f"rag2::pmc::{source_chunk_id}"
                text = clean_text(f"{title}. {chunk}") if title else chunk
                n_tokens = token_count(text)
                stats.add_chunk(n_tokens)
                write_jsonl_row(
                    out,
                    {
                        "corpus_id": corpus_id,
                        "chunk_id": corpus_id,
                        "dataset": "rag2",
                        "source": "pmc",
                        "doc_id": doc_id,
                        "source_doc_id": doc_id,
                        "source_chunk_id": source_chunk_id,
                        "title": title,
                        "text": text,
                        "metadata": {
                            "source": "pmc",
                            "source_file": row["local_path"],
                            "source_key": row.get("key", ""),
                            "subset": row.get("subset", ""),
                            "pmcid": accession_id,
                            "pmid": row.get("pmid", ""),
                            "license": row.get("license", ""),
                            "citation": row.get("citation", ""),
                            "etag": row.get("etag", ""),
                            "last_updated_utc": row.get("last_updated_utc", ""),
                            "chunk_index": local_idx,
                            "token_count": n_tokens,
                            "chunking": {
                                "unit": "word",
                                "max_words": max_words,
                                "overlap_words": overlap_words,
                            },
                        },
                    },
                )

            if progress:
                progress.update(1)
                progress.set_postfix(docs=stats.docs, chunks=stats.rows, missing=len(missing_rows))

    if progress:
        progress.close()

    with missing_path.open("w", encoding="utf-8") as f:
        for row in missing_rows:
            write_jsonl_row(f, row)
    if missing_rows and fail_on_missing:
        raise RuntimeError(f"Missing {len(missing_rows)} raw PMC docs. See {missing_path}")

    summary = {
        "source": "pmc",
        "path": rel(output_path),
        "raw_manifest": rel(selected_manifest),
        "missing_raw_docs_path": rel(missing_path),
        **stats.as_dict(),
    }
    update_manifest(output_dir=output_dir, source_summary=summary, max_words=max_words, overlap_words=overlap_words, raw_dir=raw_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert downloaded raw PMC TXT files to RAG2/Self-BioRAG-style unified JSONL chunks.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--overlap-words", type=int, default=32)
    parser.add_argument("--skip-missing", action="store_true", help="Do not fail if some selected raw TXT files are missing.")
    parser.add_argument("--limit-docs", type=int, default=None, help="Smoke-test cap. Omit for full 1,060,173-doc conversion.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    summary = build_unified(
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        fail_on_missing=not args.skip_missing,
        limit_docs=args.limit_docs,
    )
    logging.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

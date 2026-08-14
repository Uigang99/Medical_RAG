#!/usr/bin/env python3
"""Download and convert the PubMed 2026 baseline into the RAG2 unified JSONL schema.

This is a *Self-BioRAG-style* reconstruction, not the unavailable original
Self-BioRAG PubMed archive.  It freezes the official NLM 2026 production
baseline (``pubmed26n0001`` through ``pubmed26n1334``), extracts title plus
abstract text, and applies the published 128-word window / 32-word overlap.

The implementation deliberately has three resumable stages:

* ``download``: stores and MD5-verifies each NLM XML file;
* ``build``: writes one atomic JSONL part plus a statistics sidecar per XML;
* ``merge``: atomically concatenates those parts into ``pubmed.jsonl``.

The part files make a multi-day job restartable without duplicating rows.  All
long stages display tqdm progress bars with rate and estimated remaining time.
No CUDA, torch, or model is used by this script.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import orjson
except ImportError:  # pragma: no cover - the stdlib fallback is sufficient.
    orjson = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - only affects progress display.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "raw" / "rag2_pubmed26"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2_pubmed26"
NLM_BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
BASELINE_PREFIX = "pubmed26n"
BASELINE_FIRST_FILE = 1
BASELINE_LAST_FILE = 1334
MAX_WORDS = 128
OVERLAP_WORDS = 32
SPACE_RE = re.compile(r"\s+")
XML_NAME_RE = re.compile(r"^pubmed26n\d{4}\.xml\.gz$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def json_line(row: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a small JSON manifest or part-statistics sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def baseline_filename(number: int) -> str:
    return f"{BASELINE_PREFIX}{number:04d}.xml.gz"


def baseline_numbers(max_files: int | None) -> list[int]:
    numbers = list(range(BASELINE_FIRST_FILE, BASELINE_LAST_FILE + 1))
    return numbers if max_files is None else numbers[:max_files]


def read_remote_text(url: str, timeout_seconds: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Medical-RAG-RAG2-PubMed26/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def expected_md5(url: str, timeout_seconds: int) -> str:
    content = read_remote_text(url + ".md5", timeout_seconds)
    match = re.search(r"\b([0-9a-fA-F]{32})\b", content)
    if not match:
        raise RuntimeError(f"Could not parse an MD5 checksum from {url}.md5")
    return match.group(1).lower()


def file_md5(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb", buffering=block_bytes) as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def verified_existing_file(path: Path, url: str, *, verify_md5: bool, timeout_seconds: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not verify_md5:
        return True
    expected = expected_md5(url, timeout_seconds)
    actual = file_md5(path)
    if actual == expected:
        return True
    logging.warning("Existing file failed MD5 and will be redownloaded: %s", path)
    path.unlink()
    return False


def download_one(
    url: str,
    destination: Path,
    *,
    retries: int,
    timeout_seconds: int,
    verify_md5: bool,
) -> tuple[Path, bool]:
    """Download one file atomically.  Returns ``(path, reused_existing)``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified_existing_file(destination, url, verify_md5=verify_md5, timeout_seconds=timeout_seconds):
        return destination, True

    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, retries + 1):
        start_at = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "Medical-RAG-RAG2-PubMed26/1.0"})
        if start_at:
            request.add_header("Range", f"bytes={start_at}-")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                partial_response = getattr(response, "status", None) == 206
                mode = "ab" if start_at and partial_response else "wb"
                if mode == "wb":
                    start_at = 0
                content_length = int(response.headers.get("Content-Length") or 0)
                total = start_at + content_length if content_length else None
                bar = (
                    tqdm(
                        total=total,
                        initial=start_at,
                        desc=f"Download {destination.name}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    )
                    if tqdm
                    else None
                )
                with partial.open(mode, buffering=8 * 1024 * 1024) as handle:
                    while block := response.read(8 * 1024 * 1024):
                        handle.write(block)
                        if bar:
                            bar.update(len(block))
                if bar:
                    bar.close()
            partial.replace(destination)
            if verify_md5:
                expected = expected_md5(url, timeout_seconds)
                actual = file_md5(destination)
                if actual != expected:
                    destination.unlink(missing_ok=True)
                    raise RuntimeError(f"MD5 mismatch for {destination.name}: {actual} != {expected}")
            return destination, False
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {url}") from exc
            delay = min(60, 2**attempt)
            logging.warning("Download attempt %d/%d failed for %s: %s; retrying in %ss", attempt, retries, url, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def download_baseline(args: argparse.Namespace) -> list[Path]:
    numbers = baseline_numbers(args.max_files)
    xml_dir = args.raw_dir / "xml"
    urls = [f"{args.baseline_url.rstrip('/')}/{baseline_filename(number)}" for number in numbers]
    raw_manifest = {
        "kind": "official_nlm_pubmed_2026_baseline",
        "created_at": utc_now(),
        "baseline_url": args.baseline_url,
        "filename_prefix": BASELINE_PREFIX,
        "first_file": BASELINE_FIRST_FILE,
        "last_file": BASELINE_LAST_FILE,
        "selected_files": len(numbers),
        "full_baseline_files": BASELINE_LAST_FILE,
        "verify_md5": args.verify_md5,
        "files": [baseline_filename(number) for number in numbers],
    }
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.raw_dir / "download_manifest.json", raw_manifest)

    overall = tqdm(total=len(urls), desc="PubMed26 download", unit="file") if tqdm else None
    paths: list[Path] = []
    reused = 0
    try:
        for ordinal, url in enumerate(urls, start=1):
            filename = Path(url).name
            if overall:
                overall.set_postfix(file=filename, reused=reused)
            path, used_existing = download_one(
                url,
                xml_dir / filename,
                retries=args.retries,
                timeout_seconds=args.timeout_seconds,
                verify_md5=args.verify_md5,
            )
            paths.append(path)
            reused += int(used_existing)
            if overall:
                overall.update(1)
            logging.info("Downloaded %d/%d: %s%s", ordinal, len(urls), filename, " (existing)" if used_existing else "")
    finally:
        if overall:
            overall.close()
    return paths


def xml_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return clean_text(" ".join(piece for piece in element.itertext() if piece))


def pubmed_fields(record: ET.Element) -> tuple[str, str, str]:
    """Extract PubMedArticle and PubmedBookArticle title/abstract fields."""
    if record.tag == "PubmedArticle":
        root = "./MedlineCitation"
        article = "./MedlineCitation/Article"
        pmid = xml_text(record.find(f"{root}/PMID"))
        title = xml_text(record.find(f"{article}/ArticleTitle"))
        abstract_nodes = record.findall(f"{article}/Abstract/AbstractText")
    elif record.tag == "PubmedBookArticle":
        root = "./BookDocument"
        pmid = xml_text(record.find(f"{root}/PMID"))
        title = xml_text(record.find(f"{root}/ArticleTitle"))
        abstract_nodes = record.findall(f"{root}/Abstract/AbstractText")
    else:  # Defensive guard for callers using another parser event.
        return "", "", ""
    abstract = clean_text(" ".join(xml_text(node) for node in abstract_nodes if xml_text(node)))
    return pmid, title, abstract


def iter_pubmed_records(path: Path) -> Iterator[ET.Element]:
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag in {"PubmedArticle", "PubmedBookArticle"}:
                yield element
                element.clear()


def iter_word_chunks(text: str, max_words: int, overlap_words: int) -> Iterator[tuple[int, str]]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap < max_words")
    words = clean_text(text).split()
    step = max_words - overlap_words
    for local_index, start in enumerate(range(0, len(words), step)):
        chunk = words[start : start + max_words]
        if not chunk:
            return
        yield local_index, " ".join(chunk)
        if start + max_words >= len(words):
            return


@dataclass
class PartStats:
    source_file: str
    source_docs: int = 0
    source_records: int = 0
    skipped_no_pmid: int = 0
    skipped_empty_text: int = 0
    chunks: int = 0
    total_words: int = 0
    part_bytes: int = 0
    min_words: int = 0
    max_words: int = 0
    chunk_histogram: Counter[int] = field(default_factory=Counter, repr=False)

    def add_chunk(self, words: int, bytes_written: int) -> None:
        self.chunks += 1
        self.total_words += words
        self.part_bytes += bytes_written
        self.min_words = words if self.min_words == 0 else min(self.min_words, words)
        self.max_words = max(self.max_words, words)
        self.chunk_histogram[words] += 1

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("chunk_histogram", None)
        value["mean_words"] = self.total_words / self.chunks if self.chunks else 0.0
        return value


@dataclass
class AggregateStats:
    source_files: int = 0
    source_docs: int = 0
    source_records: int = 0
    skipped_no_pmid: int = 0
    skipped_empty_text: int = 0
    chunks: int = 0
    total_words: int = 0
    part_bytes: int = 0
    min_words: int = 0
    max_words: int = 0

    def add(self, part: dict[str, Any]) -> None:
        self.source_files += 1
        for name in ("source_docs", "source_records", "skipped_no_pmid", "skipped_empty_text", "chunks", "total_words", "part_bytes"):
            setattr(self, name, getattr(self, name) + int(part.get(name, 0)))
        minimum = int(part.get("min_words", 0))
        maximum = int(part.get("max_words", 0))
        if minimum:
            self.min_words = minimum if self.min_words == 0 else min(self.min_words, minimum)
        self.max_words = max(self.max_words, maximum)

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["mean_words"] = self.total_words / self.chunks if self.chunks else 0.0
        return value


def part_path(parts_dir: Path, xml_path: Path) -> Path:
    return parts_dir / f"{xml_path.name.removesuffix('.xml.gz')}.jsonl"


def part_stats_path(part: Path) -> Path:
    return part.with_suffix(part.suffix + ".stats.json")


def build_one_part(xml_path: Path, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Convert a compressed NLM XML file to one atomic unified-schema JSONL part."""
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    source_stem = xml_path.name.removesuffix(".xml.gz")
    stats = PartStats(source_file=rel(xml_path))
    global_chunk_index = 0
    with temporary.open("wb", buffering=16 * 1024 * 1024) as output:
        for record in iter_pubmed_records(xml_path):
            stats.source_records += 1
            pmid, title, abstract = pubmed_fields(record)
            if not pmid:
                stats.skipped_no_pmid += 1
                continue
            document_text = clean_text(" ".join(piece for piece in (title, abstract) if piece))
            if not document_text:
                stats.skipped_empty_text += 1
                continue
            stats.source_docs += 1
            for local_chunk_index, chunk in iter_word_chunks(document_text, args.max_words, args.overlap_words):
                source_chunk_id = f"{source_stem}_{global_chunk_index}"
                corpus_id = f"rag2::pubmed::{source_chunk_id}"
                words = len(chunk.split())
                row = {
                    "corpus_id": corpus_id,
                    "chunk_id": corpus_id,
                    "dataset": "rag2",
                    "source": "pubmed",
                    "doc_id": f"pubmed:{pmid}",
                    "source_doc_id": f"pubmed:{pmid}",
                    "source_chunk_id": source_chunk_id,
                    "title": title,
                    "text": chunk,
                    "metadata": {
                        "source": "pubmed",
                        "source_file": rel(xml_path),
                        "original_id": source_chunk_id,
                        "original_doc_id": f"pubmed:{pmid}",
                        "pmid": pmid,
                        "chunk_index": local_chunk_index,
                        "token_count": words,
                        "chunking": {
                            "unit": "whitespace_word",
                            "max_words": args.max_words,
                            "overlap_words": args.overlap_words,
                        },
                        "snapshot": "pubmed_2026_baseline",
                    },
                }
                encoded = json_line(row)
                output.write(encoded)
                stats.add_chunk(words, len(encoded))
                global_chunk_index += 1
    temporary.replace(output_path)
    result = stats.serializable()
    write_json(part_stats_path(output_path), result)
    return result


def expected_xml_paths(args: argparse.Namespace) -> list[Path]:
    xml_dir = args.raw_dir / "xml"
    expected = [xml_dir / baseline_filename(number) for number in baseline_numbers(args.max_files)]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        sample = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} PubMed26 XML archives under {xml_dir}; run --mode download first. Example: {sample}"
        )
    return expected


def load_part_stats(part: Path) -> dict[str, Any] | None:
    stats_path = part_stats_path(part)
    if not part.exists() or part.stat().st_size == 0 or not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def update_manifest(args: argparse.Namespace, aggregate: AggregateStats, xml_paths: list[Path]) -> None:
    manifest_path = args.output_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Replacing invalid manifest: %s", manifest_path)
    existing.update(
        {
            "created_at": utc_now(),
            "raw_dir": rel(args.raw_dir),
            "output_dir": rel(args.output_dir),
            "dataset": "rag2",
            "schema": {
                "id_fields": ["corpus_id", "chunk_id", "source_chunk_id", "doc_id", "source_doc_id"],
                "text_field": "text",
                "source_field": "source",
                "metadata": "Original source identifiers and source-specific fields are preserved here.",
            },
            "token_count": {"method": "whitespace", "field": "text"},
            "chunking": {
                "unit": "word",
                "max_words": args.max_words,
                "overlap_words": args.overlap_words,
                "note": "Self-BioRAG/RAG2-style 128-word chunks with 32-word overlap.",
            },
            "pubmed_snapshot": {
                "name": "NLM PubMed 2026 baseline",
                "source_url": args.baseline_url,
                "xml_files": len(xml_paths),
                "first_file": baseline_filename(BASELINE_FIRST_FILE),
                "last_file": baseline_filename(BASELINE_FIRST_FILE + len(xml_paths) - 1),
                "reconstruction": True,
                "note": "Not the original unavailable Self-BioRAG PubMed_128 archive.",
            },
        }
    )
    sources = existing.setdefault("sources", {})
    sources["pubmed"] = {
        "source": "pubmed",
        "path": rel(args.output_dir / "pubmed.jsonl"),
        "parts_dir": rel(args.output_dir / "pubmed_parts"),
        "raw_dir": rel(args.raw_dir / "xml"),
        **aggregate.serializable(),
    }
    write_json(manifest_path, existing)


def build_parts(args: argparse.Namespace) -> AggregateStats:
    xml_paths = expected_xml_paths(args)
    parts_dir = args.output_dir / "pubmed_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    aggregate = AggregateStats()
    overall = tqdm(total=len(xml_paths), desc="PubMed26 chunking", unit="file") if tqdm else None
    try:
        for ordinal, xml_path in enumerate(xml_paths, start=1):
            output_path = part_path(parts_dir, xml_path)
            part = None if args.overwrite else load_part_stats(output_path)
            if part is None:
                logging.info("Chunking %d/%d: %s", ordinal, len(xml_paths), xml_path.name)
                part = build_one_part(xml_path, output_path, args)
            else:
                logging.info("Using completed part %d/%d: %s", ordinal, len(xml_paths), output_path.name)
            aggregate.add(part)
            if overall:
                overall.update(1)
                overall.set_postfix(docs=f"{aggregate.source_docs:,}", chunks=f"{aggregate.chunks:,}")
    finally:
        if overall:
            overall.close()
    update_manifest(args, aggregate, xml_paths)
    return aggregate


def merge_parts(args: argparse.Namespace) -> Path:
    xml_paths = expected_xml_paths(args)
    parts_dir = args.output_dir / "pubmed_parts"
    parts = [part_path(parts_dir, xml_path) for xml_path in xml_paths]
    missing = [path.name for path in parts if load_part_stats(path) is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} completed JSONL parts; run --mode build first. Example: {missing[0]}")
    destination = args.output_dir / "pubmed.jsonl"
    if destination.exists() and destination.stat().st_size > 0 and not args.overwrite:
        logging.info("Using existing merged JSONL: %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    total_bytes = sum(path.stat().st_size for path in parts)
    overall = (
        tqdm(total=total_bytes, desc="PubMed26 JSONL merge", unit="B", unit_scale=True, unit_divisor=1024)
        if tqdm
        else None
    )
    try:
        with temporary.open("wb", buffering=16 * 1024 * 1024) as output:
            for ordinal, part in enumerate(parts, start=1):
                logging.info("Merging %d/%d: %s", ordinal, len(parts), part.name)
                with part.open("rb", buffering=16 * 1024 * 1024) as source:
                    while block := source.read(16 * 1024 * 1024):
                        output.write(block)
                        if overall:
                            overall.update(len(block))
        temporary.replace(destination)
    finally:
        if overall:
            overall.close()
    logging.info("Merged PubMed26 JSONL: %s (%.2f GiB)", destination, destination.stat().st_size / 1024**3)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("download", "build", "merge", "all"), required=True)
    parser.add_argument("--baseline-url", default=NLM_BASELINE_URL)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=None, help="First N XML files only; use only for a smoke test.")
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--verify-md5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Recreate completed parts or final JSONL.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.max_words <= 0 or args.overlap_words < 0 or args.overlap_words >= args.max_words:
        raise ValueError("Require --max-words > 0 and 0 <= --overlap-words < --max-words")
    if args.retries <= 0 or args.timeout_seconds <= 0:
        raise ValueError("--retries and --timeout-seconds must be positive")
    if args.mode in {"download", "all"}:
        download_baseline(args)
    if args.mode in {"build", "all"}:
        stats = build_parts(args)
        logging.info("Chunking complete: %s", json.dumps(stats.serializable(), ensure_ascii=False))
    if args.mode in {"merge", "all"}:
        merge_parts(args)


if __name__ == "__main__":
    main()

"""Prepare strict RAG²-style CPG and textbook corpora.

The target RAG²/Self-BioRAG description uses a 128-word sliding window with
32-word overlap.  This script produces final retrieval text that obeys that
limit exactly:

* CPG: removes the title that the legacy local build prepended to every
  already-windowed body snippet.  The underlying body windows are retained,
  so their original 128/32 boundaries are preserved.
* Textbooks: downloads the official MedQA ``data_clean.zip`` archive (or uses
  an already extracted copy), reads the 18 complete English textbook files,
  and applies the 128/32 window directly to the source text.

The legacy corpora and databases are never overwritten.  The resulting
``manifest.json`` makes the raw-source and chunking provenance explicit.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CPG_INPUT = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2" / "cpg.jsonl"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "raw" / "rag2_textbooks_medqa"
DEFAULT_TEXTBOOK_RAW_DIR = DEFAULT_RAW_ROOT / "textbooks" / "en"
DEFAULT_ARCHIVE_PATH = DEFAULT_RAW_ROOT / "archives" / "data_clean.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "unified" / "rag2_paper128"

# Official MedQA release linked by jind11/MedQA.  It includes the 18 complete
# English medical textbook files used by the MedRAG textbook corpus.
MEDQA_ARCHIVE_FILE_ID = "1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw"
MEDQA_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


@dataclass
class LengthStats:
    rows: int = 0
    total_words: int = 0
    min_words: int | None = None
    max_words: int = 0
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, words: int) -> None:
        self.rows += 1
        self.total_words += words
        self.min_words = words if self.min_words is None else min(self.min_words, words)
        self.max_words = max(self.max_words, words)
        self.histogram[words] += 1

    def serialize(self) -> dict[str, Any]:
        return {
            "chunks": self.rows,
            "total_words": self.total_words,
            "min_words": self.min_words or 0,
            "max_words": self.max_words,
            "mean_words": self.total_words / self.rows if self.rows else 0.0,
            "median_words": self.median(),
        }

    def median(self) -> float:
        if not self.rows:
            return 0.0
        left_rank = (self.rows + 1) // 2
        right_rank = (self.rows + 2) // 2
        seen = 0
        left = right = 0
        for value in sorted(self.histogram):
            seen += self.histogram[value]
            if not left and seen >= left_rank:
                left = value
            if seen >= right_rank:
                right = value
                break
        return (left + right) / 2


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def iter_windows(words: list[str], max_words: int, overlap_words: int) -> Iterator[tuple[int, list[str]]]:
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than max_words")
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + max_words]
        if window:
            yield index, window
        if start + max_words >= len(words):
            break


def title_prefix(title: str) -> str:
    return f"{title}. " if title else ""


def request_opener() -> urllib.request.OpenerDirector:
    cookies = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))


def content_length(response: Any) -> int | None:
    value = response.headers.get("Content-Length")
    try:
        return int(value) if value else None
    except ValueError:
        return None


def download_medqa_archive(
    archive_path: Path,
    *,
    retries: int,
    timeout_seconds: int,
    overwrite: bool,
) -> Path:
    """Download the official Google Drive archive with resumable `.part` data.

    Google Drive currently accepts the usercontent URL with ``confirm=t`` for
    this public file.  The cookie-aware opener and explicit HTML guard produce
    a useful failure instead of saving an access-warning web page as a ZIP.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and archive_path.stat().st_size > 0 and not overwrite:
        logging.info("Using existing MedQA archive: %s", archive_path)
        return archive_path

    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if overwrite:
        for path in (archive_path, part_path):
            if path.exists():
                path.unlink()

    opener = request_opener()
    query = urllib.parse.urlencode({"id": MEDQA_ARCHIVE_FILE_ID, "export": "download", "confirm": "t"})
    url = f"{MEDQA_DOWNLOAD_URL}?{query}"
    for attempt in range(1, retries + 1):
        resume_bytes = part_path.stat().st_size if part_path.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if resume_bytes:
            request.add_header("Range", f"bytes={resume_bytes}-")
        try:
            logging.info("Downloading official MedQA archive attempt %s/%s (resume=%s bytes)", attempt, retries, resume_bytes)
            with opener.open(request, timeout=timeout_seconds) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "text/html" in content_type:
                    preview = response.read(512).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        "Google Drive returned HTML instead of the archive. "
                        f"The public link may require manual confirmation. Response preview: {preview[:180]!r}"
                    )
                status = getattr(response, "status", None)
                mode = "ab" if resume_bytes and status == 206 else "wb"
                total = content_length(response)
                if total is not None and mode == "ab":
                    total += resume_bytes
                progress = tqdm(
                    total=total,
                    initial=resume_bytes if mode == "ab" else 0,
                    desc="download:medqa-textbooks",
                    unit="B",
                    unit_scale=True,
                    dynamic_ncols=True,
                ) if tqdm else None
                try:
                    with part_path.open(mode) as out:
                        while True:
                            block = response.read(8 * 1024 * 1024)
                            if not block:
                                break
                            out.write(block)
                            if progress is not None:
                                progress.update(len(block))
                finally:
                    if progress is not None:
                        progress.close()
            if not zipfile.is_zipfile(part_path):
                raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {part_path}")
            part_path.replace(archive_path)
            logging.info("Official MedQA archive ready: %s (%.2f GiB)", archive_path, archive_path.stat().st_size / 2**30)
            return archive_path
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to download official MedQA archive after {retries} attempts.") from exc
            wait_seconds = min(60, 2**attempt)
            logging.warning("Download attempt %s failed: %s; retrying in %ss", attempt, exc, wait_seconds)
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def is_real_english_textbook_member(name: str) -> bool:
    """Accept only the 18 data files, excluding macOS ``._`` sidecars.

    The official MedQA ZIP contains ``__MACOSX/.../textbooks/en/._*.txt``
    resource-fork files.  A substring match on ``textbooks/en/`` accidentally
    treated those as textbooks, doubling the local count to 36.
    """
    parts = name.replace("\\", "/").split("/")
    return (
        len(parts) == 4
        and parts[:3] == ["data_clean", "textbooks", "en"]
        and parts[3].lower().endswith(".txt")
        and not parts[3].startswith("._")
    )


def extracted_textbook_paths(raw_dir: Path) -> list[Path]:
    """Return only real textbook files and clean failed-run macOS sidecars."""
    stale_sidecars = sorted(raw_dir.glob("._*.txt"))
    for path in stale_sidecars:
        path.unlink()
    if stale_sidecars:
        logging.info("Removed %s macOS metadata sidecar(s) from %s", len(stale_sidecars), raw_dir)
    return sorted(path for path in raw_dir.glob("*.txt") if not path.name.startswith("._"))


def extract_official_textbooks(archive_path: Path, raw_dir: Path, *, overwrite: bool) -> list[Path]:
    """Extract only `data_clean/textbooks/en/*.txt` without extracting QAs."""
    if not archive_path.exists():
        raise FileNotFoundError(f"MedQA archive does not exist: {archive_path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected: list[zipfile.ZipInfo] = []
    marker = "data_clean/textbooks/en/"
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            normalized = info.filename.replace("\\", "/")
            if is_real_english_textbook_member(normalized) and not info.is_dir():
                selected.append(info)
        if not selected:
            raise RuntimeError(f"Could not find English textbook files under {marker!r} in {archive_path}")
        progress = tqdm(selected, desc="extract:medqa-textbooks", unit="file", dynamic_ncols=True) if tqdm else selected
        for info in progress:
            normalized = info.filename.replace("\\", "/")
            relative_name = normalized.split(marker, 1)[1]
            if "/" in relative_name or not relative_name.endswith(".txt"):
                raise RuntimeError(f"Unexpected nested textbook member: {info.filename}")
            output_path = raw_dir / relative_name
            if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
                continue
            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            with archive.open(info, "r") as src, temp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            temp_path.replace(output_path)
    paths = extracted_textbook_paths(raw_dir)
    if len(paths) != 18:
        raise RuntimeError(f"Expected exactly 18 English MedQA textbooks, found {len(paths)} in {raw_dir}")
    logging.info("Extracted/validated %s official English textbooks in %s", len(paths), raw_dir)
    return paths


def ensure_textbook_raw(args: argparse.Namespace) -> list[Path]:
    archive_path = args.textbook_archive.resolve()
    raw_dir = args.textbook_raw_dir.resolve()
    if args.mode in {"download", "all"}:
        download_medqa_archive(
            archive_path,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            overwrite=args.overwrite_raw,
        )
        return extract_official_textbooks(archive_path, raw_dir, overwrite=args.overwrite_raw)
    paths = extracted_textbook_paths(raw_dir)
    if len(paths) != 18:
        raise FileNotFoundError(
            f"--mode build requires 18 extracted .txt files in {raw_dir}, found {len(paths)}. "
            "Run with --mode download or --mode all first."
        )
    return paths


def cpg_body_from_legacy(row: dict[str, Any]) -> tuple[str, bool]:
    """Remove only the title prefix inserted by the earlier local CPG build."""
    text = clean_text(row.get("text") or row.get("contents") or row.get("content"))
    title = clean_text(row.get("title"))
    prefix = title_prefix(title)
    if prefix and text.startswith(prefix):
        return text[len(prefix) :].strip(), True
    if prefix:
        raise RuntimeError(
            "CPG title-bearing row does not match the legacy injected-title format: "
            f"chunk_id={row.get('chunk_id')!r}, title={title!r}, text_prefix={text[:120]!r}"
        )
    return text, False


def build_cpg(
    input_path: Path,
    output_path: Path,
    *,
    max_words: int,
    overlap_words: int,
    limit_rows: int | None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing CPG input: {input_path}")
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    stats = LengthStats()
    input_rows = 0
    source_docs: set[str] = set()
    titles_removed = 0
    progress = tqdm(total=limit_rows or 721_353, desc="prepare:cpg", unit="chunk", dynamic_ncols=True) if tqdm else None
    try:
        with input_path.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as out:
            for line in src:
                if limit_rows is not None and input_rows >= limit_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                body, removed = cpg_body_from_legacy(row)
                words = body.split()
                if not words:
                    raise RuntimeError(f"Empty CPG snippet after title normalization: {row.get('chunk_id')!r}")
                if len(words) > max_words:
                    raise RuntimeError(
                        f"CPG body violates its recorded {max_words}-word window: {row.get('chunk_id')!r} has {len(words)} words"
                    )
                input_rows += 1
                titles_removed += int(removed)
                doc_id = clean_text(row.get("source_doc_id") or row.get("doc_id"))
                source_docs.add(doc_id)
                source_chunk_id = clean_text(row.get("source_chunk_id") or row.get("chunk_id") or row.get("corpus_id"))
                corpus_id = f"rag2_paper128::cpg::{source_chunk_id}"
                out.write(
                    json_line(
                        {
                            "corpus_id": corpus_id,
                            "chunk_id": corpus_id,
                            "dataset": "rag2_paper128",
                            "source": "cpg",
                            "doc_id": doc_id,
                            "source_doc_id": doc_id,
                            "source_chunk_id": source_chunk_id,
                            "title": clean_text(row.get("title")),
                            "text": body,
                            "metadata": {
                                "source": "cpg",
                                "legacy_input_path": rel(input_path),
                                "legacy_chunk_id": row.get("chunk_id") or row.get("corpus_id"),
                                "title_removed_from_each_window": removed,
                                "token_count": len(words),
                                "chunking": {
                                    "unit": "whitespace_word",
                                    "max_words": max_words,
                                    "overlap_words": overlap_words,
                                },
                            },
                        }
                    )
                )
                stats.add(len(words))
                if progress is not None:
                    progress.update(1)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        if progress is not None:
            progress.close()
    temp_path.replace(output_path)
    return {
        "source": "cpg",
        "input_path": rel(input_path),
        "output_path": rel(output_path),
        "source_docs": len(source_docs),
        "input_rows": input_rows,
        "titles_removed_from_windows": titles_removed,
        "method": "retain_existing_128_32_body_windows_and_remove_repeated_title",
        **stats.serialize(),
    }


def build_textbooks(
    raw_paths: list[Path],
    output_path: Path,
    *,
    raw_dir: Path,
    max_words: int,
    overlap_words: int,
    include_title_once: bool,
    limit_docs: int | None,
) -> dict[str, Any]:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    selected = raw_paths[:limit_docs] if limit_docs is not None else raw_paths
    stats = LengthStats()
    progress = tqdm(selected, desc="chunk:textbooks", unit="book", dynamic_ncols=True) if tqdm else selected
    try:
        with temp_path.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as out:
            for raw_path in progress:
                title = raw_path.stem
                doc_id = f"textbooks:{title}"
                body = clean_text(raw_path.read_text(encoding="utf-8", errors="replace"))
                if not body:
                    raise RuntimeError(f"Textbook is empty: {raw_path}")
                document_words = (f"{title}. {body}" if include_title_once else body).split()
                for chunk_index, window in iter_windows(document_words, max_words=max_words, overlap_words=overlap_words):
                    text = " ".join(window)
                    if len(window) > max_words:
                        raise AssertionError("strict textbook window invariant failed")
                    source_chunk_id = f"{title}_{chunk_index}"
                    corpus_id = f"rag2_paper128::textbooks::{source_chunk_id}"
                    out.write(
                        json_line(
                            {
                                "corpus_id": corpus_id,
                                "chunk_id": corpus_id,
                                "dataset": "rag2_paper128",
                                "source": "textbooks",
                                "doc_id": doc_id,
                                "source_doc_id": doc_id,
                                "source_chunk_id": source_chunk_id,
                                "title": title,
                                "text": text,
                                "metadata": {
                                    "source": "textbooks",
                                    "raw_source_path": rel(raw_path),
                                    "raw_source_dataset": "MedQA official data_clean.zip English textbook corpus",
                                    "title_included_once_before_windowing": include_title_once,
                                    "chunk_index": chunk_index,
                                    "token_count": len(window),
                                    "chunking": {
                                        "unit": "whitespace_word",
                                        "max_words": max_words,
                                        "overlap_words": overlap_words,
                                    },
                                },
                            }
                        )
                    )
                    stats.add(len(window))
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    temp_path.replace(output_path)
    return {
        "source": "textbooks",
        "raw_dir": rel(raw_dir),
        "output_path": rel(output_path),
        "source_docs": len(selected),
        "raw_files": [path.name for path in selected],
        "method": "official_medqa_english_textbooks_direct_128_32_window",
        "title_included_once_before_windowing": include_title_once,
        **stats.serialize(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare strict RAG² 128/32 CPG and textbook unified corpora.")
    parser.add_argument("--mode", choices=["download", "build", "all"], default="all")
    parser.add_argument("--sources", nargs="+", choices=["cpg", "textbooks"], default=["cpg", "textbooks"])
    parser.add_argument("--cpg-input", type=Path, default=DEFAULT_CPG_INPUT)
    parser.add_argument("--textbook-archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--textbook-raw-dir", type=Path, default=DEFAULT_TEXTBOOK_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--overlap-words", type=int, default=32)
    parser.add_argument(
        "--include-textbook-title-once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert the filename-derived textbook title once before the source document is windowed.",
    )
    parser.add_argument("--limit-cpg-rows", type=int, default=None, help="Small CPG smoke-test cap.")
    parser.add_argument("--limit-textbook-docs", type=int, default=None, help="Small textbook smoke-test cap.")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--overwrite-raw", action="store_true", help="Redownload/re-extract the official textbook source.")
    parser.add_argument("--overwrite", action="store_true", help="Replace strict unified JSONLs/manifest.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.max_words <= 0 or not 0 <= args.overlap_words < args.max_words:
        raise ValueError("Require max-words > 0 and 0 <= overlap-words < max-words.")
    if args.retries <= 0 or args.timeout_seconds <= 0:
        raise ValueError("retries and timeout-seconds must be positive.")
    if args.limit_cpg_rows is not None and args.limit_cpg_rows <= 0:
        raise ValueError("limit-cpg-rows must be positive.")
    if args.limit_textbook_docs is not None and args.limit_textbook_docs <= 0:
        raise ValueError("limit-textbook-docs must be positive.")

    raw_paths: list[Path] = []
    if "textbooks" in args.sources:
        raw_paths = ensure_textbook_raw(args)
    elif args.mode == "download":
        # Permit a download-only invocation that prepares all textbook files.
        ensure_textbook_raw(args)

    if args.mode == "download":
        logging.info("Download/extraction complete; no unified JSONL requested in --mode download.")
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {source: output_dir / f"{source}.jsonl" for source in args.sources}
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in [*output_paths.values(), manifest_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output already exists: " + ", ".join(map(str, existing)) + ". Use --overwrite to replace it.")

    summaries: dict[str, Any] = {}
    if "cpg" in args.sources:
        summaries["cpg"] = build_cpg(
            args.cpg_input.resolve(),
            output_paths["cpg"],
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            limit_rows=args.limit_cpg_rows,
        )
        logging.info("CPG strict corpus: %s chunks, max=%s words", summaries["cpg"]["chunks"], summaries["cpg"]["max_words"])
    if "textbooks" in args.sources:
        summaries["textbooks"] = build_textbooks(
            raw_paths,
            output_paths["textbooks"],
            raw_dir=args.textbook_raw_dir.resolve(),
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            include_title_once=args.include_textbook_title_once,
            limit_docs=args.limit_textbook_docs,
        )
        logging.info(
            "Textbook strict corpus: %s chunks, max=%s words",
            summaries["textbooks"]["chunks"],
            summaries["textbooks"]["max_words"],
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "rag2_paper128",
        "purpose": "RAG²/Self-BioRAG-style final 128-word windows with 32-word overlap.",
        "chunking": {"unit": "whitespace_word", "max_words": args.max_words, "overlap_words": args.overlap_words},
        "sources": summaries,
    }
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_manifest.replace(manifest_path)
    logging.info("Prepared corpus manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import csv
import heapq
import json
import logging
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "corpus" / "mcq" / "raw" / "rag2" / "pmc"
AWS_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
FILELIST_URLS = {
    "oa_comm": AWS_BASE_URL + "deprecated/oa_comm/txt/metadata/csv/oa_comm.filelist.csv",
    "oa_noncomm": AWS_BASE_URL + "deprecated/oa_noncomm/txt/metadata/csv/oa_noncomm.filelist.csv",
}
PMCID_RE = re.compile(r"PMC(\d+)")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def pmcid_number(accession_id: str) -> int | None:
    match = PMCID_RE.search(accession_id)
    return int(match.group(1)) if match else None


def raw_txt_path(raw_dir: Path, accession_id: str) -> Path:
    number = pmcid_number(accession_id) or 0
    shard = f"PMC{number // 100000:05d}"
    return raw_dir / "txt" / shard / f"{accession_id}.txt"


def download_file(url: str, output_path: Path, overwrite: bool = False, timeout: int = 120) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    resume_from = tmp_path.stat().st_size if tmp_path.exists() and not overwrite else 0
    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        mode = "ab" if resume_from and getattr(response, "status", None) == 206 else "wb"
        with tmp_path.open(mode) as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    tmp_path.replace(output_path)
    return output_path


def download_filelist(raw_dir: Path, subset: str, overwrite: bool) -> Path:
    url = FILELIST_URLS[subset]
    output_path = raw_dir / "filelists" / f"{subset}.filelist.csv"
    logging.info("Downloading PMC filelist [%s]: %s", subset, url)
    return download_file(url, output_path, overwrite=overwrite, timeout=300)


def iter_filelist_rows(path: Path, subset: str) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if clean_text(row.get("Retracted")).lower() not in {"", "no"}:
                continue
            key = clean_text(row.get("Key"))
            accession_id = clean_text(row.get("AccessionID"))
            if not key or not accession_id:
                continue
            number = pmcid_number(accession_id)
            if number is None:
                continue
            yield {
                "subset": subset,
                "key": key,
                "accession_id": accession_id,
                "pmcid_number": str(number),
                "pmid": clean_text(row.get("PMID")),
                "license": clean_text(row.get("License")),
                "citation": clean_text(row.get("Article Citation")),
                "etag": clean_text(row.get("ETag")),
                "last_updated_utc": clean_text(row.get("Last Updated UTC (YYYY-MM-DD HH:MM:SS)")),
            }


def select_target_docs(
    raw_dir: Path,
    subsets: list[str],
    target_docs: int,
    overwrite_filelists: bool,
) -> list[dict[str, str]]:
    heap: list[tuple[int, str, dict[str, str]]] = []
    scanned = 0
    accepted = 0
    filelist_paths = {subset: download_filelist(raw_dir, subset, overwrite=overwrite_filelists) for subset in subsets}

    for subset, path in filelist_paths.items():
        logging.info("Scanning filelist [%s]: %s", subset, path)
        iterator = iter_filelist_rows(path, subset)
        progress = tqdm(iterator, desc=f"scan:{subset}", unit="row") if tqdm else iterator
        for row in progress:
            scanned += 1
            accepted += 1
            number = int(row["pmcid_number"])
            heap_item = (-number, row["accession_id"], row)
            if len(heap) < target_docs:
                heapq.heappush(heap, heap_item)
            elif number < -heap[0][0]:
                heapq.heapreplace(heap, heap_item)

    selected = [item[2] for item in heap]
    selected.sort(key=lambda item: int(item["pmcid_number"]))
    logging.info("Scanned rows=%s usable=%s selected=%s", scanned, accepted, len(selected))
    if len(selected) < target_docs:
        raise RuntimeError(f"Only selected {len(selected)} PMC docs, expected {target_docs}")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_selected_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_manifest_rows(raw_dir: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    normalized: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        row = dict(row)
        accession_id = clean_text(row.get("accession_id") or row.get("AccessionID"))
        subset = clean_text(row.get("subset"))
        key = clean_text(row.get("key") or row.get("Key"))

        # Older manifests from this script accidentally dropped the "deprecated/"
        # prefix from the PMC OpenData TXT object key, which turns every request
        # into a 404. Keep existing manifests resumable, but repair the URL.
        if key and not key.startswith("deprecated/") and subset and key.startswith(f"{subset}/"):
            key = f"deprecated/{key}"
            changed = True

        if not accession_id:
            raise ValueError(f"Selected PMC manifest row is missing accession_id: {row}")
        if not key:
            raise ValueError(f"Selected PMC manifest row is missing key: {row}")

        local_path = raw_txt_path(raw_dir, accession_id)
        expected_url = AWS_BASE_URL + key
        expected_local_path = rel(local_path)

        if row.get("key") != key:
            row["key"] = key
            changed = True
        if row.get("url") != expected_url:
            row["url"] = expected_url
            changed = True
        if row.get("local_path") != expected_local_path:
            row["local_path"] = expected_local_path
            changed = True
        row["accession_id"] = accession_id
        normalized.append(row)
    return normalized, changed


def prepare_manifest_rows(raw_dir: Path, selected: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected:
        accession_id = row["accession_id"]
        local_path = raw_txt_path(raw_dir, accession_id)
        rows.append(
            {
                **row,
                "url": AWS_BASE_URL + row["key"],
                "local_path": rel(local_path),
            }
        )
    return rows


def download_one(row: dict[str, Any], overwrite: bool, timeout: int, retries: int) -> dict[str, Any]:
    output_path = PROJECT_ROOT / row["local_path"]
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {"status": "exists", "accession_id": row["accession_id"], "bytes": output_path.stat().st_size}

    last_error = ""
    for attempt in range(retries + 1):
        try:
            path = download_file(row["url"], output_path, overwrite=overwrite, timeout=timeout)
            return {"status": "downloaded", "accession_id": row["accession_id"], "bytes": path.stat().st_size}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = repr(exc)
            time.sleep(min(2**attempt, 30))
    return {"status": "failed", "accession_id": row["accession_id"], "error": last_error, "url": row["url"]}


async def download_one_async(
    session: "aiohttp.ClientSession",
    row: dict[str, Any],
    overwrite: bool,
    retries: int,
) -> dict[str, Any]:
    output_path = PROJECT_ROOT / row["local_path"]
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {"status": "exists", "accession_id": row["accession_id"], "bytes": output_path.stat().st_size}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    last_error = ""
    for attempt in range(retries + 1):
        try:
            resume_from = tmp_path.stat().st_size if tmp_path.exists() and not overwrite else 0
            headers = {"User-Agent": "Medical_RAG PMC downloader"}
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"

            async with session.get(row["url"], headers=headers) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {text[:200]}")

                mode = "ab" if resume_from and response.status == 206 else "wb"
                with tmp_path.open(mode) as out:
                    async for chunk in response.content.iter_chunked(1024 * 256):
                        if chunk:
                            out.write(chunk)
            tmp_path.replace(output_path)
            return {"status": "downloaded", "accession_id": row["accession_id"], "bytes": output_path.stat().st_size}
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as exc:
            last_error = repr(exc)
            await asyncio.sleep(min(2**attempt, 30))

    return {"status": "failed", "accession_id": row["accession_id"], "error": last_error, "url": row["url"]}


async def bounded_async_download(
    rows: list[dict[str, Any]],
    workers: int,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    if aiohttp is None:
        raise RuntimeError("aiohttp is not installed. Use --backend threaded or install aiohttp.")

    downloaded_or_existing = 0
    downloaded_bytes = 0
    failed: list[dict[str, Any]] = []
    row_iter = iter(rows)
    row_lock = asyncio.Lock()
    progress = tqdm(total=len(rows), desc="download:pmc", unit="doc", mininterval=1.0, miniters=100) if tqdm else None
    postfix_update_every = max(100, len(rows) // 1000)

    async def next_row() -> dict[str, Any] | None:
        async with row_lock:
            try:
                return next(row_iter)
            except StopIteration:
                return None

    async def worker() -> None:
        nonlocal downloaded_or_existing, downloaded_bytes
        while True:
            row = await next_row()
            if row is None:
                return
            result = await download_one_async(session, row, overwrite, retries)
            if result["status"] == "failed":
                failed.append(result)
            else:
                downloaded_or_existing += 1
                downloaded_bytes += int(result.get("bytes") or 0)
            if progress:
                progress.update(1)
                if progress.n % postfix_update_every == 0 or progress.n == len(rows):
                    elapsed = max(time.time() - start_time, 1e-6)
                    progress.set_postfix(
                        ok=downloaded_or_existing,
                        failed=len(failed),
                        mb_s=f"{downloaded_bytes / 1024 / 1024 / elapsed:.1f}",
                        refresh=False,
                    )

    connector = aiohttp.TCPConnector(
        limit=max(workers, 1),
        limit_per_host=max(workers, 1),
        ttl_dns_cache=600,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
    )
    client_timeout = aiohttp.ClientTimeout(total=None, sock_connect=timeout, sock_read=timeout)
    start_time = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        tasks = [asyncio.create_task(worker()) for _ in range(max(workers, 1))]
        await asyncio.gather(*tasks)

    if progress:
        progress.close()
    return downloaded_or_existing, len(rows), failed


def bounded_parallel_download(
    rows: list[dict[str, Any]],
    workers: int,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    downloaded_or_existing = 0
    failed: list[dict[str, Any]] = []
    row_iter = iter(rows)
    pending: set[Future[dict[str, Any]]] = set()
    max_pending = max(workers * 4, workers)
    progress = tqdm(total=len(rows), desc="download:pmc", unit="doc", mininterval=1.0, miniters=100) if tqdm else None
    postfix_update_every = max(100, len(rows) // 1000)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            row = next(row_iter)
        except StopIteration:
            return False
        pending.add(executor.submit(download_one, row, overwrite, timeout, retries))
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(max_pending):
            if not submit_next(executor):
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result["status"] == "failed":
                    failed.append(result)
                else:
                    downloaded_or_existing += 1
                if progress:
                    progress.update(1)
                    if progress.n % postfix_update_every == 0 or progress.n == len(rows):
                        progress.set_postfix(ok=downloaded_or_existing, failed=len(failed), refresh=False)
                submit_next(executor)

    if progress:
        progress.close()
    return downloaded_or_existing, len(rows), failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw PMC full-text TXT files for RAG2/Self-BioRAG-style corpus reconstruction.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--target-docs", type=int, default=1_060_173)
    parser.add_argument("--subsets", nargs="+", choices=sorted(FILELIST_URLS), default=["oa_comm", "oa_noncomm"])
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--backend", choices=["aiohttp", "threaded"], default="aiohttp")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit-docs", type=int, default=None, help="Download only the first N selected docs for a quick resumable check.")
    parser.add_argument("--overwrite-filelists", action="store_true")
    parser.add_argument("--overwrite-text", action="store_true")
    parser.add_argument("--selection-only", action="store_true", help="Build selected_docs.jsonl but do not download article text.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected_path = raw_dir / "selected_pmc_docs.jsonl"
    if selected_path.exists() and not args.overwrite_filelists:
        logging.info("Using existing selected manifest: %s", selected_path)
        rows = load_selected_manifest(selected_path)
        rows, changed = normalize_manifest_rows(raw_dir, rows)
        if changed:
            write_jsonl(selected_path, rows)
            logging.info("Repaired selected manifest URLs/paths: %s", selected_path)
    else:
        selected = select_target_docs(
            raw_dir=raw_dir,
            subsets=args.subsets,
            target_docs=args.target_docs,
            overwrite_filelists=args.overwrite_filelists,
        )
        rows = prepare_manifest_rows(raw_dir, selected)
        rows, _ = normalize_manifest_rows(raw_dir, rows)
        write_jsonl(selected_path, rows)
        logging.info("Wrote selected manifest: %s", selected_path)

    selected_count = len(rows)
    if args.limit_docs is not None:
        rows = rows[: args.limit_docs]
        logging.info("Using first %s selected docs for this run (--limit-docs); selected manifest remains %s docs", len(rows), selected_count)

    summary_path = raw_dir / "download_summary.json"
    if args.selection_only:
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_docs": args.target_docs,
            "selected_docs": selected_count,
            "run_docs": len(rows),
            "selected_manifest": rel(selected_path),
            "downloaded": False,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.backend == "aiohttp":
        ok, total, failed = asyncio.run(
            bounded_async_download(
                rows=rows,
                workers=args.workers,
                overwrite=args.overwrite_text,
                timeout=args.timeout,
                retries=args.retries,
            )
        )
    else:
        ok, total, failed = bounded_parallel_download(
            rows=rows,
            workers=args.workers,
            overwrite=args.overwrite_text,
            timeout=args.timeout,
            retries=args.retries,
        )
    failed_path = raw_dir / "failed_downloads.jsonl"
    write_jsonl(failed_path, failed)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_docs": args.target_docs,
        "selected_docs": selected_count,
        "run_docs": len(rows),
        "attempted_docs": total,
        "available_raw_docs": ok,
        "failed_docs": len(failed),
        "selected_manifest": rel(selected_path),
        "failed_manifest": rel(failed_path),
        "subsets": args.subsets,
        "workers": args.workers,
        "backend": args.backend,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(f"PMC download finished with {len(failed)} failures. Re-run the same command to retry.")


if __name__ == "__main__":
    main()

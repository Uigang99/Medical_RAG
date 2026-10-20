#!/usr/bin/env python3
"""Cheap 21-topic metadata. Default is an offline estimate, never an API call."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import medical_topic_classification as m
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, sha_file
from medrag.training.graph_input_progress import Progress


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("estimate", "run"), default="estimate")
    p.add_argument("--source", type=Path, action="append")
    p.add_argument("--text-field", default="restored_text")
    p.add_argument("--limit", type=int, default=64, help="0 = all unique document IDs")
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-api-usd", type=float, default=0.10,
                   help="Cumulative local estimate cap for this cache, not provider billing limit")
    p.add_argument("--retry-uncertain", action="store_true",
                   help="Explicitly allow rebilling a request whose outcome was uncertain")
    p.add_argument("--cache-root", type=Path, default=ROOT / "cache/jev_medical_topics21")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_jev_topics21_train_pilot64_v1")
    p.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    a = p.parse_args(argv)
    if a.limit < 0 or not 1 <= a.workers <= 16 or not 0 < a.max_api_usd < float("inf"):
        p.error("Require limit >= 0, workers 1..16 and finite positive budget")
    if a.source is None:
        a.source = [ROOT / "datasets/kg/rag2_official_train10000_top8_h200_throughput_v2/restored/restored_chunks.jsonl"]
    return a


def run(a, rows, cache):
    import httpx
    contract = {"version": m.VERSION, "model": m.MODEL, "endpoint": m.ENDPOINT,
                "questions": m.questions(), "sources": [
                    {"path": str(p.resolve()), "sha256": sha_file(p)} for p in a.source],
                "text_field": a.text_field, "seed": a.seed, "limit": a.limit,
                "cohort_sha256": digest(rows), "sampling": "source-balanced round-robin, fixed hash order",
                "classification": "independent substantive-topic scores; ranking only",
                "threshold_selection": "none; no QA answers or choices in API input",
                "price_per_million": m.USD_PER_MTOK, "context_policy": "no truncation; reject oversized input",
                "code_sha256": sha_file(Path(m.__file__))}
    path = a.output_root / "manifest.json"
    manifest = {"contract": contract, "contract_sha256": digest(contract)}
    if path.exists():
        previous = json.loads(path.read_text())
        # Code provenance alone does not invalidate unchanged prompts or data.
        old, new = dict(previous["contract"]), dict(contract)
        old.pop("code_sha256", None); new.pop("code_sha256", None)
        if old != new:
            raise RuntimeError("Changed input/prompt contract: choose a new output-root; cache remains reusable")
    else:
        atomic_json(path, manifest)
        atomic_jsonl(a.output_root / "cohort.jsonl", rows)
    unique = {r["state_sha256"]: r for r in rows}
    jobs = []
    for row in unique.values():
        missing = sorted(set(m.TOPICS) - set(cache.scores(row)))
        if missing:
            body = m.payload(row, missing)
            try:
                m.check_context(body)
            except ValueError:
                raise RuntimeError(f"Oversized request: {row['document_id']}; not truncated or sent") from None
            jobs.append(body)
    key = m.api_key(a.env_file) if jobs else ""
    progress = Progress(["input/cache", "JEV document topics", "metadata export"])
    try:
        progress.stage(0, 1)
        progress.complete(f"{len(rows)} documents; {len(jobs)} fresh calls; 21 scores/call; no GPU")
        progress.stage(1, len(unique), initial=len(unique) - len(jobs), unit="doc", kind="api")
        progress.weights[2] = max(0.1, len(rows) / 5000)  # CPU export estimate only.
        waiting = iter(jobs)
        futures = {}
        error = None
        with httpx.Client(timeout=90, follow_redirects=False, trust_env=False) as client, \
                cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            while True:
                while len(futures) < a.workers and error is None:
                    body = next(waiting, None)
                    if body is None:
                        break
                    try:
                        attempt = cache.reserve(body, a.max_api_usd, a.retry_uncertain)
                    except Exception as exc:
                        error = exc
                        break
                    futures[pool.submit(m.send, client, key, body)] = (attempt, body)
                if not futures:
                    break
                done, _ = cf.wait(futures, timeout=1, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    attempt, body = futures.pop(f)
                    try:
                        data, seconds = f.result()
                        cache.finish(attempt, body, data, seconds)
                        progress.update()
                    except Exception as exc:
                        cache.fail(attempt, "failed_charge_unknown")
                        reason = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__
                        error = RuntimeError(f"Request {attempt} failed ({reason}); "
                                             "response, if received, retained in cache; no automatic retry")
            if error:
                raise error
        progress.complete(f"all category scores cached; {a.cache_root}")
        progress.stage(2, len(rows), unit="doc")
        def outputs():
            for row in rows:
                out = m.export_row(row, cache.scores(row))
                if out["status"] == "incomplete":
                    raise RuntimeError("Incomplete category set at export")
                progress.update()
                yield out
        atomic_jsonl(a.output_root / "labels.jsonl", outputs())
        summary = {"estimate": m.estimate(rows, cache), "cache_usage": cache.usage(),
                   "quality_evaluated": False, "clinical_edges_changed": False,
                   "taxonomy": {k: {"name_ko": v[0], "definition": v[1]} for k, v in m.TOPICS.items()}}
        atomic_json(a.output_root / "summary.json", summary)
        progress.complete(f"{a.output_root / 'labels.jsonl'}; QA/clinical quality not evaluated")
        progress.log(f"[done] {len(rows)} document records; {json.dumps(cache.usage())}")
    except BaseException:
        progress.log(f"[stopped] completed request count={cache.usage()['completed_requests']}; "
                     f"durable cache={a.cache_root}; completed calls reused; uncertain calls require explicit retry")
        raise
    finally:
        progress.close()


def main(argv=None):
    a = parse(argv)
    start = time.monotonic()
    rows = m.select_documents(a.source, a.text_field, a.limit, a.seed)
    if not rows:
        raise RuntimeError("Empty cohort")
    if a.mode == "estimate":
        # Pure offline inspection; don't create even an empty cache.
        cache = m.Cache(a.cache_root / "cache.sqlite3") if (a.cache_root / "cache.sqlite3").exists() else None
        try:
            print(json.dumps(m.estimate(rows, cache), ensure_ascii=False, indent=2))
            print(f"[offline] no API call; elapsed={time.monotonic()-start:.1f}s; "
                  "ETA calibrates after fresh API requests, existing pilot was a different input contract")
        finally:
            if cache:
                cache.close()
        return
    a.cache_root.mkdir(parents=True, exist_ok=True)
    with (a.cache_root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this cache; no duplicate paid requests")
        cache = m.Cache(a.cache_root / "cache.sqlite3")
        try:
            print("[plan] source/cohort -> shared JEV cache -> one request/document, 21 questions -> metadata; "
                  f"workers={a.workers}; cumulative local cap=${a.max_api_usd}; no KG mutation", flush=True)
            run(a, rows, cache)
        finally:
            cache.close()


if __name__ == "__main__":
    main()

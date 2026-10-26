#!/usr/bin/env python3
"""Prepare, preflight, and run a bounded 24-question original-chunk KEV pilot."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import kev_chunk_relations as schema
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions, KEV_MAX_STATE, KEV_MAX_ROW
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity
from medrag.training.graph_input_progress import Progress


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "preflight", "run"), required=True)
    p.add_argument("--input", type=Path, default=ROOT / "databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2/top8/candidates.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "results/rag2_kev_chunk_relations_test24_v1")
    p.add_argument("--seed", type=int, default=2026102607)
    p.add_argument("--batch-rows", type=int, default=8)
    p.add_argument("--source", type=Path, default=ROOT.parent / "tools/kev-runtime")
    p.add_argument("--adapter", type=Path, default=ROOT.parent / "models/kev-9b")
    p.add_argument("--base", type=Path, default=ROOT.parent / "models/Qwen3.5-9B-Base")
    a = p.parse_args()
    if not 1 <= a.batch_rows <= 64:
        p.error("batch-rows must be 1..64")
    return a


def prepare(a):
    cohort = schema.select_cohort(a.input, a.seed)
    jobs = [schema.make_job(p, task) for p in cohort["pairs"] for task in schema.TASKS]
    files = {"schema": ROOT / "medrag/kev_chunk_relations.py", "runner": Path(__file__),
             "runtime": ROOT / "medrag/evidence_graph_runtime.py",
             "adapter_config": a.adapter / "adapter_config.json", "head": a.adapter / "head.pt",
             "adapter_weights": a.adapter / "adapter_model.safetensors", "base_config": a.base / "config.json"}
    shards = sorted(a.base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("Base model weights missing")
    files.update({f"base_{i}": p for i, p in enumerate(shards)})
    contract = {
        "version": schema.VERSION, "hypothesis": "KEV can identify four partial-content relations from full original chunk pairs",
        "scope": "TEST-retrieved annotation quality pilot only; no QA, training, retrieval changes, or prompt tuning",
        "input": identity(a.input, full_hash=True), "seed": a.seed,
        "excluded_previously_read_cache_lines": sorted(schema.EXCLUDED_LINES),
        "cohort_hash": digest(cohort), "jobs_hash": digest(jobs),
        "questions": len(cohort["questions"]), "pairs": len(cohort["pairs"]), "decisions": len(jobs),
        "input_policy": "exact top8 original text; no QA questions/options/gold; no extracted propositions",
        "decision_policy": "independent 3-way argmax; all probabilities saved; NO/UNCERTAIN retained but no edge",
        "review_policy": {"blind_pairs": 96, "positive_audit_per_relation": 30,
                          "precision_gate": .85, "recall_gate": .70, "minimum_predicted_positive_audit": 20,
                          "reference": "assistant source-text review, not independent medical gold",
                          "stop": "Do not scale or tune this version on TEST; insufficient positives are inconclusive"},
        "files": {k: identity(v) for k, v in files.items()}, "gpu": 1, "dtype": "bfloat16",
        "batch_rows": a.batch_rows, "max_state": KEV_MAX_STATE, "max_row": KEV_MAX_ROW,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    h = ensure_contract(a.output / "manifest.json", contract)
    atomic_json(a.output / "cohort.json", cohort)
    atomic_jsonl(a.output / "jobs.jsonl", jobs)
    requested = {pid for q in cohort["questions"] for pid in q["blind_review_pairs"]}
    atomic_jsonl(a.output / "blind_review_inputs.jsonl", [p for p in cohort["pairs"] if p["pair_id"] in requested])
    print(json.dumps({"stage": "prepare", "questions": len(cohort["questions"]), "pairs": len(cohort["pairs"]),
                      "decisions": len(jobs), "blind_review_pairs": len(requested), "output": str(a.output)}))
    return cohort, jobs, h


def preflight(a, jobs):
    sys.path.insert(0, str(a.source))
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode, rows_of
    from kev.checkpoint import Checkpoint
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.adapter, local_files_only=True)
    isolation = Checkpoint(a.adapter).meta.option_isolation
    maximum = {"state_tokens": 0, "row_tokens": 0}
    for job in jobs:
        rec, _ = to_record(SystemOneRequest.model_validate(job["body"]))
        enc = encode(tokenizer, rec, max_state=KEV_MAX_STATE, max_branch=KEV_MAX_ROW,
                     strict=True, option_isolation=isolation)
        if enc["state_truncated"]:
            raise RuntimeError(f"Truncated job: {job['id']}")
        state, _, rows = rows_of(enc)
        maximum["state_tokens"] = max(maximum["state_tokens"], len(state))
        maximum["row_tokens"] = max(maximum["row_tokens"], max(len(state)+len(r["ids"]) for r in rows))
    atomic_json(a.output / "preflight.json", {"jobs": len(jobs), "missing": 0, "truncated": 0,
                                             "maxima": maximum, "model_loaded": False})
    return maximum


def run(a, cohort, jobs, h):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1; GPU0 must not be used")
    refs_path = a.output / "blind_references.jsonl"
    if not refs_path.exists():
        raise RuntimeError("Write the 96 blind source-text references BEFORE KEV execution")
    refs = [json.loads(x) for x in refs_path.read_text().splitlines() if x]
    wanted = {p for q in cohort["questions"] for p in q["blind_review_pairs"]}
    if len(refs) != len(wanted) or {x["pair_id"] for x in refs} != wanted:
        raise RuntimeError("Incomplete/duplicate blind review references")
    for ref in refs:
        if set(ref["labels"]) != set(schema.TASKS) or not set(ref["labels"].values()) <= set(schema.CRITERIA):
            raise ValueError("Invalid blind reference labels")
    reference_contract = {"experiment_hash": h, "reference_identity": identity(refs_path, full_hash=True)}
    ensure_contract(a.output / "frozen_reference_manifest.json", reference_contract)
    cache = DecisionCache(a.output / "decisions", h)
    progress = Progress(("contract / context", "load KEV GPU1", "six relations TEST24", "graphs / summary"))
    # Final stage contains only aggregation/JSON I/O. Its explicit initial cost model
    # is 2 s plus 1 ms per decision, replaced by actual measured duration at completion.
    progress.weights[3] = 2 + len(jobs) * .001
    started = time.monotonic()
    with (a.output / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            cached = sum(cache.get(j) is not None for j in jobs)
            progress.log(f"[plan] frozen 24 questions/Top8 -> context -> KEV -> 6 decisions/pair -> graphs; cached={cached}/{len(jobs)}; GPU1 only")
            progress.log("[ETA] New original-chunk task: loading/decision rates uncalibrated; shared GPU, batch rows="+str(a.batch_rows))
            progress.stage(0, 1)
            limits = preflight(a, jobs)
            progress.update(); progress.complete(f"no truncation; maxima={limits}; frozen references={len(refs)}")
            progress.stage(1, 1)
            judge = None
            if cached < len(jobs):
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
                judge = LocalDecisions(a.source, a.adapter, a.base, cache, a.batch_rows)
            progress.update(); progress.complete("KEV loaded" if judge else "all decisions cached; no GPU load")
            score_start = time.monotonic()
            def emit(event, **data):
                if event == "stage": progress.stage(**data)
                elif event == "update": progress.update(data["amount"])
                elif event == "complete": progress.complete(data["message"])
            if judge:
                decisions = judge.score(jobs, 2, emit)
            else:
                decisions = {j["id"]: cache.get(j) for j in jobs}
                progress.stage(2, len(jobs), initial=len(jobs)); progress.complete("all cached")
            score_seconds = time.monotonic()-score_start
            progress.stage(3, 1)
            predictions, graphs, distribution = schema.materialize(cohort, jobs, decisions)
            atomic_jsonl(a.output / "predictions.jsonl", predictions)
            atomic_jsonl(a.output / "graphs.jsonl", graphs)
            summary = {"questions": len(cohort["questions"]), "pairs": len(predictions), "decisions": len(jobs),
                       "distribution": distribution, "missing": 0, "truncated": 0,
                       "fresh_decisions": len(jobs)-cached, "decision_seconds": score_seconds,
                       "elapsed_seconds": time.monotonic()-started, "batch_rows": a.batch_rows,
                       "batch_parity": judge.parity if judge else None,
                       "batch_disabled": judge.batch_disabled if judge else None,
                       "oom_splits": judge.oom_splits if judge else None,
                       "quality_review_pending": True, "QA_evaluated": False,
                       "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft")}}
            atomic_json(a.output / "summary.json", summary)
            progress.update(); progress.complete(str(a.output / "summary.json"))
        except Exception as exc:
            durable = sum(cache.get(j) is not None for j in jobs)
            progress.log(f"[FAILED] stage={progress.index+1}/4 completed={durable}/{len(jobs)} remaining={len(jobs)-durable}; {type(exc).__name__}: {exc}; durable={a.output}; identical command resumes")
            raise
        finally:
            progress.close()
    print(f"[complete] {len(predictions)} pairs/{len(jobs)} decisions; graphs saved; quality review pending; {a.output}")


if __name__ == "__main__":
    a = arguments()
    cohort, jobs, h = prepare(a)
    if a.stage == "preflight":
        print(json.dumps(preflight(a, jobs)))
    elif a.stage == "run":
        run(a, cohort, jobs, h)

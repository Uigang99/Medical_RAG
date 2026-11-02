#!/usr/bin/env python3
"""Run two independent dataset workers on physical GPU 1.

Only dataset jobs are concurrent. Each child still executes the original
prepare/evaluate loop and the official decoder, one question at a time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.training.graph_input_progress import Progress
from scripts.run_loft_pced_paper import (arguments, atomic_json, preflight,
                                         result_file, sha)

GPU = 1
WORKERS = {"A": ("musique", "nq"), "B": ("hotpotqa", "quest", "qampari")}
DATASETS = ("hotpotqa", "nq", "musique", "qampari", "quest")
# Observed 5-question pilot, seconds for prepare/5 and generation/question.
# Used only for scheduling/progress estimates, never to change predictions.
PILOT = {"hotpotqa": (5.80, 3.32), "nq": (6.10, 4.88),
         "musique": (6.02, 5.73), "qampari": (2.96, 1.40),
         "quest": (3.84, 3.95)}
PYTHON = ROOT / ".venv_pced/bin/python"
WRAPPER = ROOT / "scripts/run_loft_pced_paper.sh"
CORE = ROOT / "scripts/run_loft_pced_paper.py"
PILOT_ROOT = ROOT / "results/loft_pced_paper_v1_smoke5"


def arguments_parallel(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--dry-run", action="store_true", help="CPU-only contract check; no jobs")
    a = p.parse_args(argv)
    a.output_root = a.output_root.resolve()
    if a.output_root == PILOT_ROOT.resolve():
        p.error("Full 500-question run must not reuse the 5-question pilot root")
    return a


def count_json(folder):
    return sum(1 for _ in folder.glob("*.json")) if folder.is_dir() else 0


def counts(root):
    result = {}
    for dataset in DATASETS:
        folder = root / dataset
        result[dataset] = (count_json(folder / "rankings"),
                           count_json(folder / "expert_scores"),
                           count_json(folder / "predictions"))
        if any(n > 100 for n in result[dataset]):
            raise RuntimeError(f"Unexpected >100 cached files: {dataset}")
    return result


def weighted_progress(records):
    total = sum(100 * (prepare + generate) for prepare, generate in PILOT.values())
    done = 0.0
    for dataset, (ranked, scored, answered) in records.items():
        prepare, generate = PILOT[dataset]
        done += ranked * prepare * .85 + scored * prepare * .15 + answered * generate
    return min(10000, round(10000 * done / total))


def verify_pilot_pair(full_root, references, checked_rank, checked_answer):
    """Stop early if a full-run pilot question differs from the sequential run."""
    for dataset, qids in references.items():
        for qid in qids:
            key = (dataset, qid)
            old = PILOT_ROOT / dataset
            new = full_root / dataset
            if key not in checked_rank:
                current = result_file(new / "rankings", qid)
                if current.is_file():
                    prior = json.loads(result_file(old / "rankings", qid).read_text())
                    got = json.loads(current.read_text())
                    if [x["pid"] for x in prior["top90"]] != [x["pid"] for x in got["top90"]]:
                        raise RuntimeError(f"Top-90 changed versus sequential pilot: {dataset}:{qid}")
                    checked_rank.add(key)
            if key not in checked_answer:
                current = result_file(new / "predictions", qid)
                if current.is_file():
                    prior = json.loads(result_file(old / "predictions", qid).read_text())
                    got = json.loads(current.read_text())
                    if (prior["generated_token_ids"] != got["generated_token_ids"] or
                            prior["expert_winners"] != got["expert_winners"]):
                        raise RuntimeError(f"PCED tokens/routing changed versus sequential pilot: {dataset}:{qid}")
                    checked_answer.add(key)


def gpu_preflight():
    result = subprocess.check_output(["nvidia-smi", "-i", str(GPU), "--query-gpu=memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    free = int(result.strip())
    if free < 80 * 1024:
        raise RuntimeError(f"GPU{GPU} has only {free/1024:.1f} GiB free; require 80 GiB for two workers")
    return free


def launch(root, worker, dataset):
    log_path = root / "_parallel_logs" / f"{dataset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    log.write(f"\n[parallel] start worker={worker} dataset={dataset} GPU={GPU} time={time.strftime('%F %T')}\n")
    log.flush()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPU), TOKENIZERS_PARALLELISM="false",
               TF_CPP_MIN_LOG_LEVEL="3")
    command = ["bash", str(WRAPPER), "all", dataset, "--output-root", str(root)]
    try:
        process = subprocess.Popen(command, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return process, log, log_path


def stop_children(active):
    for process, _, _, _ in active.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process, log, _, _ in active.values():
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()


def main(argv=None):
    a = arguments_parallel(argv)
    progress = Progress(["CPU contract", "parallel prepare + PCED 500q", "paired summary"])
    progress.weights[:] = [1., None, 1.]
    active = {}
    try:
        progress.stage(0, 1)
        if not PYTHON.is_file() or not WRAPPER.is_file():
            raise FileNotFoundError(f"Missing {PYTHON} or {WRAPPER}")
        references = {}
        for dataset in DATASETS:
            child_args = arguments(["--phase", "preflight", "--dataset", dataset,
                                    "--output-root", str(a.output_root)])
            _, queries, _ = preflight(child_args, write_manifest=False)
            if len(queries) != 100:
                raise RuntimeError(f"Full LOFT split required: {dataset} has {len(queries)} questions")
            references[dataset] = [str(q["qid"]) for q in queries[:5]]
            for qid in references[dataset]:
                for kind in ("rankings", "predictions"):
                    if not result_file(PILOT_ROOT / dataset / kind, qid).is_file():
                        raise FileNotFoundError(f"Missing sequential parity reference: {dataset}/{kind}/{qid}")
        existing = counts(a.output_root)
        progress.complete(f"5 LOFT test sets x 100=500; cached answers={sum(x[2] for x in existing.values())}")
        progress.log("[plan] GPU1 only: worker A musique,nq; worker B hotpotqa,quest,qampari; original per-question PCED unchanged; first 5 per dataset compared with sequential pilot")
        progress.log("[ETA] Two workers sharing GPU1 have no measured throughput; live progress calibrates ETA. The prior two-GPU 30-50 min estimate does not apply")
        if a.dry_run:
            progress.log(f"[dry-run OK] CPU-only; no manifest or GPU jobs; root={a.output_root}")
            return
        free = gpu_preflight()
        expected = {"version": "loft_pced_parallel_gpu1_v1", "dataset": "LOFT-RAG 5x100 test",
                    "physical_gpu": GPU, "workers": {k: list(v) for k, v in WORKERS.items()},
                    "method": "unchanged official PCED per-question decode; dataset-level data parallelism only",
                    "candidate_mode": "all", "top_k": 90, "score_policy": "paper",
                    "max_new_tokens": 32, "question_order": "original LOFT test order",
                    "core_sha256": sha(CORE), "wrapper_sha256": sha(WRAPPER),
                    "orchestrator_sha256": sha(Path(__file__)),
                    "pilot_parity_root": str(PILOT_ROOT), "output_root": str(a.output_root),
                    "git_commit": subprocess.check_output(["git", "-C", str(ROOT),
                                                              "rev-parse", "HEAD"], text=True).strip()}
        manifest = a.output_root / "parallel_manifest.json"
        if manifest.is_file():
            if json.loads(manifest.read_text()) != expected:
                raise RuntimeError(f"Incompatible parallel resume contract: {manifest}")
        else:
            atomic_json(manifest, expected)
        progress.log(f"[GPU] free GiB: GPU1={free/1024:.1f}; manifest={manifest}")
        initial = weighted_progress(existing)
        progress.stage(1, 10000, initial=initial, unit="work")
        next_index = {worker: 0 for worker in WORKERS}
        checked_rank, checked_answer = set(), set()
        while True:
            for worker, datasets in WORKERS.items():
                if worker not in active and next_index[worker] < len(datasets):
                    dataset = datasets[next_index[worker]]
                    next_index[worker] += 1
                    process, log, path = launch(a.output_root, worker, dataset)
                    active[worker] = (process, log, path, dataset)
                    progress.log(f"[start] GPU{GPU} worker {worker} {dataset}; child log={path}")
            time.sleep(3)
            verify_pilot_pair(a.output_root, references, checked_rank, checked_answer)
            current = counts(a.output_root)
            rank = sum(x[0] for x in current.values())
            prior = sum(x[1] for x in current.values())
            answers = sum(x[2] for x in current.values())
            progress.names[1] = f"GPU1 x2 R{rank}/500 P{prior}/500 A{answers}/500"
            done = weighted_progress(current)
            if done > progress.done:
                progress.update(done - progress.done)
            for worker, (process, log, path, dataset) in list(active.items()):
                status = process.poll()
                if status is not None:
                    log.close()
                    del active[worker]
                    if status:
                        raise RuntimeError(f"GPU{GPU} worker {worker} {dataset} exited {status}; log={path}; cached rank/prior/answer={current[dataset]}")
                    progress.log(f"[complete] GPU{GPU} worker {worker} {dataset}; cached={current[dataset]}; log={path}")
            if not active and all(next_index[worker] == len(WORKERS[worker]) for worker in WORKERS):
                break
        if answers != 500 or len(checked_rank) != 25 or len(checked_answer) != 25:
            raise RuntimeError(f"Incomplete or unverified run: answers={answers}, parity={len(checked_rank)}/{len(checked_answer)}")
        progress.complete(f"500/500 answers; 25/25 pilot Top-90 and PCED token/routing checks passed")
        progress.stage(2, 1)
        datasets = {}
        for dataset in DATASETS:
            path = a.output_root / dataset / "summary.json"
            row = json.loads(path.read_text())
            if row["n"] != 100 or row["missing"] != 0:
                raise RuntimeError(f"Incomplete per-dataset summary: {path}")
            datasets[dataset] = row
        correct = sum(row["correct"] for row in datasets.values())
        overall = {"n": 500, "correct": correct, "micro_subspan_em": correct/500,
                   "macro5_subspan_em": sum(row["subspan_em"] for row in datasets.values())/5,
                   "datasets": datasets, "pilot_parity_questions": 25,
                   "parallel_manifest": str(manifest)}
        atomic_json(a.output_root / "summary.json", overall)
        progress.complete(f"{a.output_root/'summary.json'}")
        progress.log(f"[result] 500/500; Micro Subspan EM={correct}/500; root={a.output_root}")
    except BaseException as exc:
        if active:
            stop_children(active)
        current = counts(a.output_root)
        progress.log(f"[FAILED] stage={progress.index+1}/{len(progress.names)}; {exc}; durable={a.output_root}; cached={current}; identical command resumes compatible shards")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded, source-disjoint KEV proposition role pilot; never QA or full-corpus inference."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity
from medrag.training.graph_input_progress import Progress

ROLES = ("DEFINE", "DESCRIBE", "EXPLAIN", "ASSESS", "METHOD", "NORM")


def cohort(path):
    data = json.loads(path.read_text())
    if data["version"] != "kev_abc_train_input_pilot_v1":
        raise ValueError("Only the frozen official-train proposition cohort is allowed")
    rows, seen, docs = [], {}, {"development": set(), "confirmation": set()}
    for pair in data["pairs"]:
        for side in ("P1", "P2"):
            prop = pair[side]
            phase = pair["split"]
            docs[phase].add(prop["document_id"])
            key = " ".join(prop["claim"].casefold().split())
            if key in seen:
                if seen[key]["split"] != phase:
                    raise ValueError("Exact normalized proposition text crosses the split")
                seen[key]["aliases"].append(pair["case_id"] + side)
                continue
            row = {"id": pair["case_id"] + side, "split": phase,
                   "proposition_id": prop["proposition_id"], "document_id": prop["document_id"],
                   "source": prop["source"], "text": prop["claim"], "aliases": []}
            seen[key] = row
            rows.append(row)
    if docs["development"] & docs["confirmation"]:
        raise ValueError("Source documents cross the split")
    return rows


def jobs_for(rows, spec):
    if set(spec["roles"]) != set(ROLES):
        raise ValueError("Exactly the six preregistered roles are required")
    jobs = []
    for row in rows:
        questions = {role: {"type": "choice", "instructions": spec["roles"][role],
                            "criteria": spec["criteria"]} for role in ROLES}
        body = {"state": {"proposition": row["text"], "instructions": spec["instructions"]},
                "questions": questions}
        jobs.append({"id": digest(body), "body": body, "meta": {"id": row["id"], "version": spec["version"]}})
    return jobs


def wilson(success, total):
    if not total:
        return None
    z = 1.96
    p = success / total
    d = 1 + z*z/total
    c = (p + z*z/(2*total)) / d
    r = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0., c-r), min(1., c+r)]


def metrics(predictions, refs):
    by_role = {}
    exact = 0
    disagreements = []
    for role in ROLES:
        tp = fp = fn = tn = abstain = excluded = 0
        for row in predictions:
            ref = refs[row["id"]]
            if role in ref.get("optional", []):
                excluded += 1
                continue
            expected = role in ref["yes"]
            output = row["answers"][role]["choice"]
            positive = output == "YES"
            abstain += output == "UNCERTAIN"
            tp += positive and expected
            fp += positive and not expected
            fn += not positive and expected
            tn += output == "NO" and not expected
        p = tp/(tp+fp) if tp+fp else None
        r = tp/(tp+fn) if tp+fn else None
        f = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        by_role[role] = dict(tp=tp, fp=fp, fn=fn, tn=tn, uncertain=abstain, ambiguous_excluded=excluded,
                             precision=p, recall=r, f1=f, precision_ci95=wilson(tp,tp+fp),
                             recall_ci95=wilson(tp,tp+fn), positive_reference_n=tp+fn)
    for row in predictions:
        ref = refs[row["id"]]
        errs = []
        for role in ROLES:
            if role in ref.get("optional", []):
                continue
            want = "YES" if role in ref["yes"] else "NO"
            got = row["answers"][role]["choice"]
            if got != want:
                errs.append({"role": role, "reference": want, "prediction": got})
        exact += not errs
        if errs:
            disagreements.append({"id": row["id"], "text": row["text"], "errors": errs,
                                  "reference_note": ref.get("note", "")})
    tp = sum(x["tp"] for x in by_role.values())
    fp = sum(x["fp"] for x in by_role.values())
    fn = sum(x["fn"] for x in by_role.values())
    f1s = [x["f1"] for x in by_role.values() if x["f1"] is not None]
    return {"n": len(predictions), "role_decisions": len(predictions)*6, "by_role": by_role,
            "micro_precision": tp/(tp+fp) if tp+fp else None,
            "micro_recall": tp/(tp+fn) if tp+fn else None,
            "macro_f1": sum(f1s)/len(f1s) if f1s else None, "exact_on_unambiguous_n": exact,
            "exact_on_unambiguous_rate": exact/len(predictions),
            "uncertain_n": sum(v["uncertain"] for v in by_role.values()),
            "all_no_n": sum(all(v["choice"] == "NO" for v in row["answers"].values()) for row in predictions),
            "disagreements": disagreements}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, default=ROOT/"results/rag2_kev_abc_train_input_pilot_v1/cohort.json")
    p.add_argument("--root", type=Path, default=ROOT/"results/rag2_kev_information_roles_pilot_v1")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--phase", choices=("development", "confirmation"), default="development")
    p.add_argument("--selection", type=Path)
    p.add_argument("--batch-rows", type=int, default=48)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not 6 <= a.batch_rows <= 96:
        p.error("batch-rows must be 6..96")
    spec = json.loads(a.spec.read_text())
    reference = json.loads(a.reference.read_text())
    all_rows = cohort(a.cohort)
    rows = [r for r in all_rows if r["split"] == a.phase]
    refs = {}
    for row in all_rows:
        rec = reference["cases"][row["id"]]
        if set(rec["yes"]) & set(rec.get("optional", [])):
            raise ValueError(f"Overlapping reference yes/optional: {row['id']}")
        if not set(rec["yes"] + rec.get("optional", [])).issubset(ROLES):
            raise ValueError("Unknown reference role")
        refs[row["id"]] = rec
    if a.phase == "confirmation":
        if not a.selection:
            p.error("Confirmation requires a frozen development selection")
        selected = json.loads(a.selection.read_text())
        if selected["spec_sha256"] != identity(a.spec)["sha256"]:
            raise ValueError("Selected prompt differs")
    jobs = jobs_for(rows, spec)
    model_root = ROOT.parent / "models"
    runtime = ROOT.parent / "tools/kev-runtime"
    files = {"cohort": a.cohort, "spec": a.spec, "reference": a.reference,
             "runner": Path(__file__), "runtime": ROOT/"medrag/evidence_graph_runtime.py",
             "kev_model": runtime/"kev/model.py", "kev_api": runtime/"kev/api.py",
             "adapter_config": model_root/"kev-9b/adapter_config.json",
             "adapter_weights": model_root/"kev-9b/adapter_model.safetensors",
             "head": model_root/"kev-9b/head.pt", "base_config": model_root/"Qwen3.5-9B-Base/config.json"}
    for i, path in enumerate(sorted((model_root/"Qwen3.5-9B-Base").glob("*.safetensors"))):
        files[f"base_weight_{i}"] = path
    if not any(k.startswith("base_weight_") for k in files):
        raise FileNotFoundError("No base weights")
    contract = {"version": "proposition_information_roles_v1", "phase": a.phase,
                "prompt_version": spec["version"], "changed_variable": spec["change"],
                "scope": "official train only; no QA, options, answers, KG or previous labels in model input",
                "hypothesis": "Fixed coarse independent role decisions can be made reliably from a standalone proposition",
                "rows": [r["id"] for r in rows], "roles": ROLES,
                "files": {k: identity(v) for k,v in files.items()}, "request_hash": digest([j["body"] for j in jobs]),
                "batch_rows": a.batch_rows, "gpu": 1, "dtype": "bfloat16", "generation": "none; argmax decision probabilities",
                "max_prompt_versions": 5, "source_context": False,
                "selection": identity(a.selection) if a.selection else None,
                "reference_status": "assistant provisional pre-output annotation, NOT independently verified gold",
                "success_policy": {"micro_precision_min": .90, "macro_f1_min": .80,
                                   "per_role_precision_min": .85, "per_role_recall_min": .75,
                                   "minimum_confirmation_positives_per_role": 8,
                                   "qualitative_gate": "No recurring mechanism inflation, observed-to-prescriptive error, or metadata contamination; manual review mandatory"},
                "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "packages": {x: importlib.metadata.version(x) for x in ("torch","transformers","peft")}}
    if a.preflight:
        # Exact prompt/API/tokenization check, without loading or allocating the model.
        sys.path.insert(0, str(runtime))
        from kev.api import SystemOneRequest, to_record
        from transformers import AutoTokenizer
        from kev.checkpoint import Checkpoint
        from kev import model as kev_model
        tokenizer = AutoTokenizer.from_pretrained(model_root/"kev-9b",local_files_only=True)
        for j in jobs:
            record, _ = to_record(SystemOneRequest.model_validate(j["body"]))
            if len(tokenizer.encode(json.dumps(record))) > 7000:
                raise ValueError("Conservative token-budget preflight failed")
        print(json.dumps({"status":"preflight_passed","n":len(rows),"decisions":len(jobs)*6,
                          "sources":dict(Counter(r["source"] for r in rows)),"gpu_loaded":False}))
        return
    out = a.root/spec["version"]/a.phase
    out.mkdir(parents=True, exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        h = ensure_contract(out/"manifest.json", contract)
        cache = DecisionCache(out/"cache", h)
        atomic_jsonl(out/"cohort.jsonl", rows)
        atomic_jsonl(out/"requests.jsonl", jobs)
        progress = Progress(("contract official train", "load KEV GPU1", "role decisions", "save review"))
        started = time.monotonic()
        try:
            progress.log(f"[plan] {spec['version']}/{a.phase}: {len(rows)} propositions -> six independent decisions; no QA, no training, no full corpus")
            progress.log("[ETA] several minutes per version estimated from prior KEV pilot; first fresh batch calibrates, review time separate")
            progress.weights = [1., 20., None, 1.]
            progress.stage(0,1);progress.update();progress.complete(f"source-disjoint and exact-text-deduplicated; {out}")
            missing = [j for j in jobs if cache.get(j) is None]
            progress.stage(1,1)
            judge = None
            if missing:
                if a.report_only:
                    raise RuntimeError(f"Missing {len(missing)} cached requests")
                if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
                    raise RuntimeError("Require CUDA_VISIBLE_DEVICES=1")
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error();hf_logging.disable_progress_bar()
                judge = LocalDecisions(runtime, model_root/"kev-9b", model_root/"Qwen3.5-9B-Base", cache, a.batch_rows)
            progress.update();progress.complete("local KEV ready" if judge else "all predictions cached; GPU not loaded")
            def emit(event, **data):
                if event == "stage":progress.stage(**data)
                elif event == "update":progress.update(data["amount"])
                else:progress.complete(data["message"])
            if judge:
                judge.score(jobs,2,emit)
            else:
                progress.stage(2,len(jobs)*6,initial=len(jobs)*6);progress.complete("compatible cache reused")
            progress.stage(3,1)
            predictions = [{**r,"answers":cache.get(j)["answers"]} for r,j in zip(rows,jobs,strict=True)]
            report = metrics(predictions, refs)
            report["runtime_seconds"] = time.monotonic()-started
            report["batch_parity"] = judge.parity if judge else None
            report["batch_fallback"] = judge.batch_disabled if judge else None
            report["OOM_splits"] = judge.oom_splits if judge else None
            report["interpretation"] = "Agreement with frozen assistant labels, not independently measured classification accuracy. Optional boundary labels excluded; source-stratified pilot is not population-uniform."
            baselines = {}
            for policy in ("all_no", "describe_only"):
                fake = [{**r,"answers":{k:{"choice":"YES" if policy == "describe_only" and k == "DESCRIBE" else "NO"} for k in ROLES}} for r in rows]
                bm = metrics(fake,refs);bm.pop("disagreements")
                baselines[policy] = bm
            report["baselines"] = baselines
            atomic_jsonl(out/"predictions.jsonl",predictions)
            atomic_json(out/"summary.json",report)
            atomic_jsonl(out/"review_items.jsonl", [{**r,"reference":refs[r["id"]]} for r in predictions])
            progress.update();progress.complete(f"complete={len(predictions)}/{len(rows)}; missing=0; summary={out/'summary.json'}")
        except BaseException as error:
            done = sum(cache.get(j) is not None for j in jobs)
            progress.log(f"[FAILED] stage={progress.index+1}/4 complete={done}/{len(jobs)} remaining={len(jobs)-done} durable={out}; identical command resumes; {type(error).__name__}: {error}")
            raise
        finally:
            progress.close()
    print(json.dumps({"output":str(out),"n":report["n"],"macro_f1":report["macro_f1"],"micro_precision":report["micro_precision"]}))


if __name__ == "__main__":
    main()

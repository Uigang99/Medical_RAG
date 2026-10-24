#!/usr/bin/env python3
"""48-pair blind-reference ABC decision pilot. Default: development only, no QA."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import kev_abc_quality as q
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions, KEV_SDPA_BACKENDS, KEV_MAX_STATE, KEV_MAX_ROW
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity
from medrag.training.graph_input_progress import Progress

DEFAULT_ROOT = ROOT / "results/rag2_kev_abc_quality_pilot_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true", help="CPU schema/reference/context checks only")
    p.add_argument("--report", action="store_true", help="Recompute reports from complete cached decisions; no models")
    p.add_argument("--phase", choices=("development", "confirmation"), default="development")
    p.add_argument("--cohort", type=Path, default=DEFAULT_ROOT / "cohort.json")
    p.add_argument("--reference", type=Path, default=DEFAULT_ROOT / "reference.json")
    p.add_argument("--candidate-root", type=Path, default=ROOT / "results/rag2_common_pair_candidates_test6545_v1")
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--kev-source", type=Path, default=ROOT.parent / "tools/kev-runtime")
    p.add_argument("--adapter", type=Path, default=ROOT.parent / "models/kev-9b")
    p.add_argument("--base", type=Path, default=ROOT.parent / "models/Qwen3.5-9B-Base")
    p.add_argument("--batch-rows", type=int, default=64)
    p.add_argument("--confidence", type=float, default=.7)
    p.add_argument("--margin", type=float, default=.15)
    a = p.parse_args(argv)
    if not (18 <= a.batch_rows <= 128 and .5 <= a.confidence <= 1 and 0 <= a.margin <= 1):
        p.error("Invalid batch/confidence bounds")
    return a


def check_inputs(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Use the launcher with physical GPU1 only")
    cohort = json.loads(a.cohort.read_text())
    reference = json.loads(a.reference.read_text())
    if reference["type"] != "assistant_adjudicated_provisional" or reference["clinical_gold"] or reference["kev_predictions_seen"]:
        raise ValueError("Expected provisional assistant references written before any KEV predictions")
    pairs, refs = cohort["pairs"], reference["references"]
    if len(pairs) != 48 or Counter(p["split"] for p in pairs) != {"development": 24, "confirmation": 24}:
        raise ValueError("Expected frozen 48-pair / 24+24 pilot")
    if len({p["pair_id"] for p in pairs}) != 48:
        raise ValueError("Duplicate sampled pair")
    q.validate_references(pairs, refs)
    texts = {split: {x["claim"] for p in pairs if p["split"] == split for x in (p["P1"],p["P2"])} for split in ("development", "confirmation")}
    if texts["development"] & texts["confirmation"]:
        raise ValueError("Verbatim proposition leakage between splits")
    all_jobs = [q.make_job(p, g, v) for p in pairs for g in ("A", "B", "C") for v in q.VARIANTS]
    if digest(all_jobs) != cohort["prompt_hash_before_reference"]:
        raise ValueError("The frozen prompt specification changed after reference creation; version a new experiment")
    database = a.candidate_root / "candidates.sqlite3"
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        for p in pairs:
            indices = []
            for node in (p["P1"], p["P2"]):
                r = db.execute("SELECT idx,did,claim FROM proposition WHERE pid=?", (node["id"],)).fetchone()
                if r is None or (r[1], r[2]) != (node["document_id"], node["claim"]):
                    raise ValueError(f"Original proposition changed: {node['id']}")
                indices.append(r[0])
            if db.execute("SELECT 1 FROM pair WHERE a=? AND b=?", sorted(indices)).fetchone() is None:
                raise ValueError("A pilot pair is not in the approved candidate set")
            if digest(sorted((p["P1"]["id"], p["P2"]["id"]))) != p["pair_id"]:
                raise ValueError("Pair ID mismatch")
    files = {"cohort": a.cohort, "reference": a.reference,
             "candidate_manifest": a.candidate_root / "manifest.json",
             "candidate_summary": a.candidate_root / "summary.json",
             "taxonomy_prompts": ROOT / "medrag/kev_abc_quality.py", "runner": Path(__file__),
             "runtime": ROOT / "medrag/evidence_graph_runtime.py",
             "kev_model": a.kev_source / "kev/model.py", "kev_api": a.kev_source / "kev/api.py",
             "kev_checkpoint": a.kev_source / "kev/checkpoint.py",
             "adapter_config": a.adapter / "adapter_config.json", "adapter_weights": a.adapter / "adapter_model.safetensors",
             "head": a.adapter / "head.pt", "tokenizer": a.adapter / "tokenizer.json",
             "tokenizer_config": a.adapter / "tokenizer_config.json", "base_config": a.base / "config.json"}
    shards = sorted(a.base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("Missing local Qwen base weight shards")
    files.update({f"base_weight_{i}": path for i, path in enumerate(shards)})
    contract = dict(version=q.VERSION, files={k: identity(p) for k,p in files.items()},
                    hypothesis="Explicit relation boundaries improve reference agreement without a second reasoning LLM",
                    split="48 original candidate pairs; 24 development / 24 document-disjoint confirmation",
                    reference_type=reference["type"], clinical_gold=False, no_qa_or_training=True,
                    decision_inputs="Two unmodified propositions plus same-document flag; no question/options/gold/KG witnesses or previous decisions",
                    variants=list(q.VARIANTS), confidence=a.confidence, margin=a.margin,
                    selection="Equal graph-family weighting: A categorical macro-F1, B/C YES-predicate micro-F1; graphs without reference positives excluded; ties prefer concise",
                    batch_rows=a.batch_rows, gpu=1, backends=list(KEV_SDPA_BACKENDS), shared_prefix=False,
                    packages={k: importlib.metadata.version(k) for k in ("torch","transformers","peft","numpy")},
                    git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    old = a.output_root / "manifest.json"
    if old.exists() and json.loads(old.read_text()) != {"contract":contract, "contract_sha256":digest(contract)}:
        raise RuntimeError("Output contract changed; use a NEW output root, do not overwrite prior decisions")
    return pairs, refs, contract


def cpu_context_check(a, pairs):
    # Inspect the exact KEV encoder without allocating/loading the 9B model.
    sys.path.insert(0,str(a.kev_source))
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode, rows_of
    from kev.checkpoint import Checkpoint
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.adapter,local_files_only=True)
    isolation = Checkpoint(a.adapter).meta.option_isolation
    maxima = {g: {"state_tokens":0,"row_tokens":0} for g in ("A","B","C")}
    for pair in pairs:
        for g in ("A","B","C"):
            for v in q.VARIANTS:
                job = q.make_job(pair,g,v)
                record, _ = to_record(SystemOneRequest.model_validate(job["body"]))
                enc = encode(tokenizer,record,max_state=KEV_MAX_STATE,max_branch=KEV_MAX_ROW,strict=True,option_isolation=isolation)
                if enc["state_truncated"]:
                    raise RuntimeError("KEV state truncation")
                state, _, rows = rows_of(enc)
                maxima[g]["state_tokens"] = max(maxima[g]["state_tokens"],len(state))
                maxima[g]["row_tokens"] = max(maxima[g]["row_tokens"],max(len(state)+len(r["ids"]) for r in rows))
    return maxima


def collect(a, pairs, variants, cache):
    predictions, records, missing = {}, [], []
    for variant in variants:
        predictions[variant] = {}
        for pair in pairs:
            result = {}
            for graph in ("A","B","C"):
                job = q.make_job(pair,graph,variant)
                raw = cache.get(job)
                if raw is None:
                    missing.append(job["id"])
                    continue
                result[graph] = {k:q.hard_label(v,a.confidence,a.margin) for k,v in raw["answers"].items()}
                edges,status=q.graph_edges(pair,graph,result[graph])
                records.append(dict(case_id=pair["case_id"],pair_id=pair["pair_id"],split=pair["split"],
                                    graph=graph,variant=variant,prediction=result[graph],probabilities=raw["answers"],
                                    edges=edges,status=status,
                                    P1=pair["P1"],P2=pair["P2"]))
            predictions[variant][pair["pair_id"]] = result
    if missing:
        raise RuntimeError(f"{len(missing)} uncached requests; rerun same phase to finish. No incomplete score is reported.")
    return predictions, records


def baseline_predictions(pairs):
    return {p["pair_id"]:{"A":{"TOPIC":"NOT_RELATED"},"B":{k:"NO" for k in q.B_TASKS},
                           "C":{k:"NO" for k in q.c_tasks()}} for p in pairs}


def report(a, pairs, refs, variants, cache, phase, contract_hash):
    predictions, records = collect(a,pairs,variants,cache)
    baseline,_ = q.evaluate(pairs,refs,baseline_predictions(pairs))
    summaries, errors = {}, []
    for v in variants:
        summaries[v], misses = q.evaluate(pairs,refs,predictions[v])
        raw={p["pair_id"]:{} for p in pairs}
        for r in records:
            if r["variant"]==v:
                raw[r["pair_id"]][r["graph"]]={k:answer["choice"] for k,answer in r["probabilities"].items()}
        raw_summary,_=q.evaluate(pairs,refs,raw)
        summaries[v]["raw_argmax_without_confidence_abstention"]=raw_summary
        errors.extend(dict(variant=v,**e) for e in misses)
        for g in ("A","B","C"):
            score = summaries[v][g]["agreement"]
            summaries[v][g]["agreement_delta_vs_no_edge"] = score-baseline[g]["agreement"] if score is not None else None
    if phase == "development":
        selected = max(variants,key=lambda v:(summaries[v]["selection_score"],v=="concise"))
        selection = {"contract_hash":contract_hash,"selected_variant":selected,
                     "scores":{v:summaries[v]["selection_score"] for v in variants},
                     "confirmation_used_for_selection":False,"rule":"Predeclared family-balanced score, never confirmation accuracy"}
        path = a.output_root / "selection.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError("Previously frozen selection differs; do not reuse confirmation")
        atomic_json(path,selection)
    else:
        selected = variants[0]
    positive_counts = Counter()
    for pair in pairs:
        for graph, decisions in q.reference_decisions(refs[pair["case_id"]]).items():
            for k,v in decisions.items():
                if graph == "A" or v == "YES":
                    positive_counts[f"{graph}:{v if graph=='A' else k}"] += 1
    missing_C = [name for name in q.C_DEFINITIONS if not any(k.startswith("C:"+name+":") for k in positive_counts)]
    result = {"phase":phase,"pairs":len(pairs),"selected_variant":selected,"models":summaries,
              "no_edge_baseline":baseline,"reference_positive_counts":dict(positive_counts),
              "C_relations_without_positive_examples":missing_C,"full_annotation_authorized":False,
              "B_predicates_without_positive_examples":[name for name in q.B_TASKS if not positive_counts["B:"+name]],
              "scope":"Assistant-reference agreement only; stratified sample, not a population accuracy estimate",
              "uncertainty":"24 pairs per split, single unvalidated annotator, correlated predicates; rare relations remain unvalidated"}
    atomic_json(a.output_root/f"{phase}_summary.json",result)
    atomic_jsonl(a.output_root/f"{phase}_predictions.jsonl",records)
    atomic_jsonl(a.output_root/f"{phase}_disagreements.jsonl",errors)
    return result


def run(a):
    pairs, refs, contract = check_inputs(a)
    chosen = [p for p in pairs if p["split"]==a.phase]
    if a.preflight:
        context = cpu_context_check(a,pairs)
        print(json.dumps({"preflight":"passed","cohort":48,"development":24,"confirmation":24,
                          "context":context,"development_decision_rows":24*22*2,"confirmation_decision_rows":24*22,
                          "GPU_loaded":False,"KEV_predictions":0},indent=2))
        return
    selection_file = a.output_root / "selection.json"
    if a.phase=="confirmation":
        if not selection_file.exists():
            raise RuntimeError("Run development first; confirmation cannot choose its own prompt")
        selection=json.loads(selection_file.read_text())
        if selection["contract_hash"] != digest(contract):
            raise RuntimeError("Confirmation selection contract mismatch")
        variants=[selection["selected_variant"]]
    else:
        variants=list(q.VARIANTS)
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        h=ensure_contract(a.output_root/"manifest.json",contract)
        cache=DecisionCache(a.output_root/"decisions",h)
        if a.report:
            result=report(a,chosen,refs,variants,cache,a.phase,h)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return
        names=("contracts / contexts","load KEV",f"{a.phase} A",f"{a.phase} B",f"{a.phase} C","agreement report")
        progress=Progress(names)
        timings={}
        started=time.monotonic()
        jobs_by_graph={g:[q.make_job(p,g,v) for p in chosen for v in variants] for g in ("A","B","C")}
        def emit(event,**data):
            if event=="stage": progress.stage(**data)
            elif event=="update": progress.update(data["amount"])
            elif event=="complete":
                timings[progress.names[progress.index]]=time.monotonic()-progress.stage_started
                progress.complete(data["message"])
        try:
            counts={g:sum(len(j["body"]["questions"]) for j in jobs) for g,jobs in jobs_by_graph.items()}
            cached={g:sum(len(j["body"]["questions"]) for j in jobs if cache.get(j) is not None) for g,jobs in jobs_by_graph.items()}
            progress.log(f"[plan] {a.phase}: {len(chosen)} pairs; independent A/B/C; rows={counts}; cached={cached}; GPU1 only; no QA")
            progress.log("[ETA] New ABC tasks: calibrate each family separately. One model load, length-sorted batches, durable request cache.")
            progress.stage(0,1)
            contexts=cpu_context_check(a,chosen)
            atomic_json(a.output_root/f"{a.phase}_contexts.json",contexts)
            progress.update();progress.complete("Unmodified original pairs, document-disjoint split, strict KEV token limits verified")
            progress.stage(1,1,initial=int(counts==cached),unit="model")
            judge=None
            if counts!=cached:
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error();hf_logging.disable_progress_bar()
                judge=LocalDecisions(a.kev_source,a.adapter,a.base,cache,a.batch_rows)
                progress.update()
            progress.complete("KEV loaded once" if judge else "All decisions cached; no model load")
            for index,g in enumerate(("A","B","C"),start=2):
                if judge:
                    judge.score(jobs_by_graph[g],index,emit)
                else:
                    progress.stage(index,counts[g],initial=counts[g],unit="decision")
                    progress.complete("Complete decision cache reused")
            progress.stage(5,1,unit="report")
            result=report(a,chosen,refs,variants,cache,a.phase,h)
            progress.update();progress.complete(f"{a.phase}_summary.json and per-predicate disagreements; selected={result['selected_variant']}")
            atomic_json(a.output_root/f"{a.phase}_runtime.json",{"contract_hash":h,"elapsed_seconds":time.monotonic()-started,
                        "stages_seconds":timings,"weights":progress.weights,
                        "batch_parity":judge.parity if judge else None,
                        "oom_splits":judge.oom_splits if judge else 0,"batch_disabled":judge.batch_disabled if judge else False})
        except Exception as exc:
            progress.log(f"[FAILED] {a.phase} stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; "
                         f"{type(exc).__name__}: {exc}; durable={cache.root}; identical command resumes request caches")
            raise
        finally:
            progress.close()
        print(f"[complete] {a.phase}: pairs={len(chosen)}; selected={result['selected_variant']}; no full annotation or QA; output={a.output_root}")


if __name__=="__main__":
    run(arguments())

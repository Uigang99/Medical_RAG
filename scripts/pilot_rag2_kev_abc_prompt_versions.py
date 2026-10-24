#!/usr/bin/env python3
"""Evaluate one new KEV prompt revision on the frozen development/confirmation pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import kev_abc_quality as q
from medrag import kev_abc_prompt_versions as revisions
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity
from medrag.training.graph_input_progress import Progress
from scripts import pilot_rag2_kev_abc_quality as original


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version",required=True,choices=revisions.VERSIONS)
    p.add_argument("--phase",choices=("development","confirmation"),default="development")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--output-root",type=Path,default=ROOT/"results/rag2_kev_abc_quality_prompt_updates")
    p.add_argument("--batch-rows",type=int,default=64)
    return p.parse_args()


def run(a):
    # The legacy checker validates its own frozen v1 manifest.  The revision
    # manifest is validated separately below; otherwise a confirmation rerun
    # mistakes the revision manifest for an incompatible legacy v1 manifest.
    base=original.arguments(["--output-root",str(original.DEFAULT_ROOT)])
    pairs,refs,contract=original.check_inputs(base)
    chosen=[p for p in pairs if p["split"]==a.phase]
    jobs={g:[revisions.make_job(p,g,a.version) for p in chosen] for g in "ABC"}
    full=dict(contract)
    full.update(version="kev_abc_prompt_revision_1",variants=[a.version],confidence=0.0,margin=0.0,
                files={**contract["files"],"revision_module":identity(ROOT/"medrag/kev_abc_prompt_versions.py"),
                       "revision_runner":identity(Path(__file__))},
                selection="Frozen development prompt revision, confirmation never used to update it")
    out=a.output_root/a.version
    if a.preflight:
        sys.path.insert(0,str(base.kev_source))
        from kev.api import SystemOneRequest, to_record
        from kev.model import encode
        from kev.checkpoint import Checkpoint
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(base.adapter,local_files_only=True)
        isolation=Checkpoint(base.adapter).meta.option_isolation
        maxima={}
        for g,group in jobs.items():
            max_row=0
            for job in group:
                record,_=to_record(SystemOneRequest.model_validate(job["body"]))
                encoded=encode(tok,record,max_state=7552,max_branch=8192,strict=True,option_isolation=isolation)
                if encoded["state_truncated"]:raise RuntimeError("KEV context truncated")
                max_row=max(max_row,len(encoded["ids"]))
            maxima[g]=max_row
        print(json.dumps({"version":a.version,"phase":a.phase,"pairs":len(chosen),"max_tokens":maxima,
                          "gpu_loaded":False},indent=2))
        return
    if a.phase=="confirmation":
        selection=a.output_root/"frozen_selection.json"
        if not selection.exists() or json.loads(selection.read_text()).get("selected_version")!=a.version:
            raise RuntimeError("Freeze the development-selected version before confirmation")
    out.mkdir(parents=True,exist_ok=True)
    h=ensure_contract(out/"manifest.json",full)
    cache=DecisionCache(out/"decisions",h)
    if a.version=="v7_family_composite" and a.phase=="development":
        for graph,source in (("A","v6_minimal_pair_test"),("B","v4_relation_boundaries"),
                             ("C","v4_relation_boundaries")):
            source_root=a.output_root/source
            source_manifest=json.loads((source_root/"manifest.json").read_text())
            previous=DecisionCache(source_root/"decisions",source_manifest["contract_sha256"])
            for pair in chosen:
                old_job=revisions.make_job(pair,graph,source)
                new_job=revisions.make_job(pair,graph,a.version)
                if new_job["body"]!=old_job["body"]:
                    raise RuntimeError("Composite prompt does not exactly match source version")
                saved=previous.get(old_job)
                if saved is None:raise RuntimeError(f"Missing verified source decision: {source}/{pair['case_id']}")
                if cache.get(new_job) is None:
                    cache.put(new_job,saved["answers"],reused_from=source,
                              source_contract_hash=source_manifest["contract_sha256"])
    names=("contract / cohort","load KEV",f"{a.phase} A",f"{a.phase} B",f"{a.phase} C","agreement report")
    progress=Progress(names)
    begin=time.monotonic()
    judge=None
    try:
        progress.log(f"[plan] {a.version}/{a.phase}: 24 frozen pairs; A/B/C independent; GPU1; no QA")
        progress.stage(0,1)
        progress.update();progress.complete("Frozen reference and candidate identities checked")
        all_cached=all(cache.get(j) is not None for group in jobs.values() for j in group)
        progress.stage(1,1)
        if not all_cached:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error();hf_logging.disable_progress_bar()
            judge=LocalDecisions(base.kev_source,base.adapter,base.base,cache,a.batch_rows)
        progress.update();progress.complete("KEV loaded" if judge else "All decisions cached")
        def emit(event,**data):
            if event=="stage":progress.stage(**data)
            elif event=="update":progress.update(data["amount"])
            elif event=="complete":progress.complete(data["message"])
        for i,g in enumerate("ABC",2):
            if judge:judge.score(jobs[g],i,emit)
            else:
                progress.stage(i,sum(len(j["body"]["questions"]) for j in jobs[g]),
                               initial=sum(len(j["body"]["questions"]) for j in jobs[g]),unit="decision")
                progress.complete("Cached decisions")
        progress.stage(5,1)
        preds={};records=[]
        for p in chosen:
            preds[p["pair_id"]]={}
            for g in "ABC":
                job=revisions.make_job(p,g,a.version)
                raw=cache.get(job)["answers"]
                labels={key:value["choice"] for key,value in raw.items()}
                preds[p["pair_id"]][g]=labels
                edges,status=q.graph_edges(p,g,labels)
                records.append({"case_id":p["case_id"],"pair_id":p["pair_id"],"graph":g,"labels":labels,
                                "answers":raw,"edges":edges,"status":status,"P1":p["P1"],"P2":p["P2"]})
        metrics,errors=q.evaluate(chosen,refs,preds)
        result={"version":a.version,"phase":a.phase,"pairs":len(chosen),"reference_type":"assistant_adjudicated_provisional",
                "metrics":metrics,
                "scope":"Single-annotator agreement on 24 stratified pairs; not clinical accuracy or population estimate",
                "full_annotation_authorized":False,
                "runtime_seconds":time.monotonic()-begin,
                "batch_parity":judge.parity if judge else None}
        atomic_json(out/f"{a.phase}_summary.json",result)
        atomic_jsonl(out/f"{a.phase}_predictions.jsonl",records)
        atomic_jsonl(out/f"{a.phase}_disagreements.jsonl",errors)
        progress.update();progress.complete(f"{a.phase}_summary.json + paired predictions/disagreements")
    except Exception as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; "
                     f"{type(exc).__name__}: {exc}; durable={cache.root}; same command resumes compatible decisions")
        raise
    finally:
        progress.close()
    print(f"[complete] {a.version}/{a.phase}: pairs={len(chosen)}; output={out}")


if __name__=="__main__":run(args())

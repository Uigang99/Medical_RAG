#!/usr/bin/env python3
"""One frozen JEV call per pair on the existing 48-pair KEV quality cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import kev_abc_quality as taxonomy
from medrag import jev_abc_quality as design
from medrag.medical_topic_classification import api_key
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity
from medrag.training.graph_input_progress import Progress


SOURCE = ROOT / "results/rag2_kev_abc_quality_pilot_v1"
DEFAULT_OUTPUT = ROOT / "results/rag2_jev_abc_quality_once_v1"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "run", "report"), required=True)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    p.add_argument("--max-cost-usd", type=float, default=.10)
    p.add_argument("--timeout", type=float, default=60)
    a = p.parse_args()
    if not 0 < a.max_cost_usd <= 1 or not 1 <= a.timeout <= 120:
        p.error("Invalid bounded cost/timeout")
    return a


def prepare(a):
    cohort = json.loads((SOURCE / "cohort.json").read_text())
    reference = json.loads((SOURCE / "reference.json").read_text())
    pairs, refs = cohort["pairs"], reference["references"]
    if len(pairs) != 48 or len({p["pair_id"] for p in pairs}) != 48:
        raise ValueError("Expected the exact frozen 48-pair cohort")
    taxonomy.validate_references(pairs, refs)
    requests = [design.make_request(pair) for pair in pairs]
    if len({x["request_id"] for x in requests}) != 48:
        raise ValueError("Repeated JEV request body")
    files = {"cohort": SOURCE / "cohort.json", "reference": SOURCE / "reference.json",
             "taxonomy": ROOT / "medrag/kev_abc_quality.py", "prompt": ROOT / "medrag/jev_abc_quality.py",
             "runner": Path(__file__)}
    contract = {
        "version": "jev_abc_quality_once_v1", "model": design.MODEL, "endpoint": design.ENDPOINT,
        "source": "frozen KEV 48 pairs; no new sampling/retrieval", "files": {k: identity(v) for k,v in files.items()},
        "request_count": 48, "decision_count": 48 * 22,
        "request_hash": digest(requests), "reference_type": reference["type"],
        "label_rule": "A direct choice; B/C noul >= 0.5 is YES, otherwise NO; no tuning after responses",
        "price_usd_per_million_input_tokens": design.USD_PER_MILLION_INPUT_TOKENS,
        "cost_cap_usd": a.max_cost_usd, "timeout_seconds": a.timeout,
        "no_automatic_retry": True,
    }
    return pairs, refs, requests, contract


def reservation(body):
    # Heuristic reserve is deliberately higher than the byte count; actual
    # provider usage replaces it after a validated response.
    size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
    return (2 * size + 4096) * design.USD_PER_MILLION_INPUT_TOKENS / 1e6


def existing(a, requests, contract):
    if not (a.output_root / "manifest.json").exists():
        return None
    manifest = json.loads((a.output_root / "manifest.json").read_text())
    prior_contract = manifest["contract"]
    # A validator-only correction may change source-file hashes; frozen request
    # bodies are the actual paid experiment and must remain byte-for-byte equal.
    def request_contract(value):
        out = json.loads(json.dumps(value))
        out["files"].pop("prompt")
        out["files"].pop("runner")
        return out
    if (manifest.get("contract_sha256") != digest(prior_contract) or
            request_contract(prior_contract) != request_contract(contract) or
            prior_contract["request_hash"] != digest(requests)):
        raise RuntimeError("Frozen JEV request/data contract changed; use a new output directory")
    ledger_path = a.output_root / "cost_ledger.json"
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists() else
              {"contract_hash": digest(contract), "attempts": []})
    if ledger.get("contract_hash") != manifest["contract_sha256"]:
        raise RuntimeError("JEV cost ledger contract mismatch")
    return ledger


def read_complete(a, request):
    path = a.output_root / "raw" / f"{request['request_id']}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("request_id") != request["request_id"]:
        raise RuntimeError("Cached JEV request hash mismatch")
    design.validate_response(request, record["response"])
    return record["response"]


def report(a, pairs, refs, requests, ledger):
    results = {r["pair_id"]: read_complete(a, r) for r in requests}
    if any(v is None for v in results.values()):
        raise RuntimeError("Cannot score an incomplete one-shot JEV pilot")
    predictions = {pair_id: design.labels_from_response(v) for pair_id,v in results.items()}
    by_split = {}
    for split in ("development", "confirmation", "all"):
        subset = [p for p in pairs if p["split"] == split or split == "all"]
        metrics, errors = taxonomy.evaluate(subset, refs, predictions)
        by_split[split] = {"pairs": len(subset), "metrics": metrics, "disagreements": len(errors)}
        atomic_jsonl(a.output_root / f"{split}_disagreements.jsonl", errors)
    rows = []
    for pair in pairs:
        labels = predictions[pair["pair_id"]]
        raw = results[pair["pair_id"]]
        rows.append({"case_id":pair["case_id"],"pair_id":pair["pair_id"],"split":pair["split"],
                     "P1":pair["P1"],"P2":pair["P2"],"labels":labels,
                     "probabilities":raw["answers"],
                     "edges":{g:taxonomy.graph_edges(pair,g,labels[g])[0] for g in "ABC"}})
    atomic_jsonl(a.output_root / "predictions.jsonl", rows)
    result = {"model":design.MODEL,"pairs":48,"decision_count":48*22,"splits":by_split,
              "usage": {"completed_requests":len([x for x in ledger["attempts"] if x["status"]=="complete"]),
                        "input_tokens":sum(x.get("input_tokens",0) for x in ledger["attempts"] if x["status"]=="complete"),
                        "charged_estimate_usd":sum(x["accounted_usd"] for x in ledger["attempts"]),
                        "uncertain_attempts":len([x for x in ledger["attempts"] if x["status"]!="complete"])},
              "scope":"Agreement with one pre-KEV assistant reference, not clinical accuracy; no prompt or threshold tuning"}
    atomic_json(a.output_root / "summary.json",result)
    return result


def run(a):
    pairs, refs, requests, contract = prepare(a)
    total_reserve = sum(reservation(x["body"]) for x in requests)
    if total_reserve > a.max_cost_usd:
        raise RuntimeError(f"Conservative reservation ${total_reserve:.4f} exceeds cap ${a.max_cost_usd:.2f}; no requests sent")
    if a.mode == "preflight":
        print(json.dumps({"requests":48,"independent_decisions":48*22,"model":design.MODEL,
                          "approx_reservation_usd":total_reserve,"cost_cap_usd":a.max_cost_usd,
                          "first_request_bytes":len(json.dumps(requests[0]["body"]).encode()),
                          "existing_output":(a.output_root/"manifest.json").exists(),
                          "api_calls":0},indent=2))
        return
    if a.mode == "report":
        ledger = existing(a,requests,contract)
        if ledger is None:raise RuntimeError("No completed run to report")
        print(json.dumps(report(a,pairs,refs,requests,ledger),ensure_ascii=False,indent=2))
        return
    # Check credentials before creating any result artifact or paid request.
    key = api_key(a.env_file)
    a.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output_root/"manifest.json"
    h = (json.loads(manifest_path.read_text())["contract_sha256"] if manifest_path.exists()
         else ensure_contract(manifest_path,contract))
    ledger = existing(a,requests,contract)
    if ledger is None:ledger={"contract_hash":h,"attempts":[]}
    for entry in ledger["attempts"]:
        if entry["status"] == "pending_charge_unknown":
            matched = next(r for r in requests if r["request_id"] == entry["request_id"])
            recovered = read_complete(a,matched)
            if recovered is not None:
                tokens = design.validate_response(matched,recovered)
                entry.update(status="complete",input_tokens=tokens,
                             accounted_usd=tokens*design.USD_PER_MILLION_INPUT_TOKENS/1e6,
                             recovered_locally=True)
    atomic_json(a.output_root/"cost_ledger.json",ledger)
    atomic_jsonl(a.output_root/"requests.jsonl",requests)
    progress=Progress(["contract / cost", "JEV 48 pair requests", "paired quality report"])
    try:
        cached=sum(read_complete(a,r) is not None for r in requests)
        progress.log(f"[plan] frozen 48 pairs; 22 independent judgments per pair; cached={cached}; cap=${a.max_cost_usd:.2f}; no retries")
        progress.log("[ETA] No matched JEV throughput yet; first paid responses calibrate latency")
        progress.stage(0,1)
        if ledger["contract_hash"]!=h:raise RuntimeError("Ledger contract mismatch")
        progress.update();progress.complete("Request bodies frozen; source and cost cap checked")
        progress.stage(1,48,initial=cached,unit="pair",kind="api")
        with httpx.Client(timeout=a.timeout,follow_redirects=False,trust_env=False) as client:
            for request in requests:
                if read_complete(a,request) is not None:continue
                prior=[x for x in ledger["attempts"] if x["request_id"]==request["request_id"]]
                if prior:raise RuntimeError("A prior request may have been charged; no automatic retry")
                reserve=reservation(request["body"])
                if sum(x["accounted_usd"] for x in ledger["attempts"])+reserve>a.max_cost_usd:
                    raise RuntimeError("Local JEV cost cap reached")
                entry={"request_id":request["request_id"],"case_id":request["case_id"],
                       "status":"pending_charge_unknown","accounted_usd":reserve,"created":time.time()}
                ledger["attempts"].append(entry)
                atomic_json(a.output_root/"cost_ledger.json",ledger)
                start=time.monotonic()
                try:
                    response=client.post(design.ENDPOINT,
                        headers={"Authorization":"Bearer "+key},json=request["body"])
                    if response.status_code!=200:
                        raise RuntimeError(f"JEV HTTP {response.status_code}; charge unknown, no retry")
                    payload=response.json()
                    atomic_json(a.output_root/"raw"/f"{request['request_id']}.json",
                                {"request_id":request["request_id"],"response":payload})
                    tokens=design.validate_response(request,payload)
                    entry.update(status="complete",input_tokens=tokens,
                                 accounted_usd=tokens*design.USD_PER_MILLION_INPUT_TOKENS/1e6,
                                 seconds=time.monotonic()-start)
                    atomic_json(a.output_root/"cost_ledger.json",ledger)
                except Exception:
                    atomic_json(a.output_root/"cost_ledger.json",ledger)
                    raise
                progress.update()
        progress.complete("All 48 single-pass JEV requests stored atomically")
        progress.stage(2,1,unit="report")
        result=report(a,pairs,refs,requests,ledger)
        progress.update();progress.complete(f"All 48 pairs scored; estimated usage ${result['usage']['charged_estimate_usd']:.5f}")
    except Exception as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/3 completed={progress.done}/{progress.total}; "
                     f"{type(exc).__name__}: {exc}; charged/uncertain ledger={a.output_root/'cost_ledger.json'}")
        raise
    finally:
        progress.close()
    print(f"[complete] JEV one-shot 48 pairs; output={a.output_root}")


if __name__=="__main__":run(arguments())

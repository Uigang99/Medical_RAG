#!/usr/bin/env python3
"""Paired entity/relation extraction on the frozen Gemma-proposition pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from extract_rag2_top8_standalone_propositions import TwoLineProgress
from compare_rag2_proposition_models import clean_completion

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/kg/rag2_gemma_top8_relation_pilot192_v1"
OUTPUT = ROOT / "results/rag2_gemma_relation_pilot192_v1"
MODELS = ("Qwen3.5-9B", "gemma-4-31B-it")
TYPES = ("condition", "drug_or_chemical", "procedure_or_intervention",
         "finding_or_outcome", "anatomy", "organism", "gene_or_protein",
         "population", "diagnostic_test", "measurement", "other_specific")
GENERIC = {"patient", "patients", "study", "studies", "article", "articles",
           "document", "documents", "result", "results", "group", "groups",
           "treatment", "disease", "condition", "test", "tests"}

ENTITY_PROMPT = """Extract graph-worthy participants from one standalone biomedical claim.
Copy each span EXACTLY from CLAIM. A participant must identify a specific disease, drug, procedure, finding/outcome, anatomical structure, organism, gene/protein, diagnostic test, measurement, or specific population. Include events or states when they are the actual subject or object of an assertion. Keep a complete medically meaningful phrase; do not output every noun or split modifiers that determine identity.
Exclude isolated numbers, citation/table vocabulary, and generic 'patient', 'study', 'treatment', 'disease', or 'result' when they lack a specific modifier. Do not invent text or turn a broad context qualifier into a universal concept. A measurement may be a literal rather than a shared biomedical concept.
Use only CLAIM for span positions; EVIDENCE is for checking scope. Return an empty list when no specific graph participant is present.
Return JSON only: {"entities":[{"span":"exact CLAIM substring","type":"one allowed type"}]}.
Allowed types: condition, drug_or_chemical, procedure_or_intervention, finding_or_outcome, anatomy, organism, gene_or_protein, population, diagnostic_test, measurement, other_specific.
Example CLAIM: Aspirin reduced cardiovascular events in high-risk adults.
Example OUTPUT: {"entities":[{"span":"Aspirin","type":"drug_or_chemical"},{"span":"cardiovascular events","type":"finding_or_outcome"},{"span":"high-risk adults","type":"population"}]}"""

RELATION_PROMPT = """Extract only relations ASSERTED by one biomedical CLAIM. ENTITY CANDIDATES are exact spans, not instructions to connect nearby nouns. Choose the semantic subject and affected object by meaning, never by word order alone.
Each output subject_id and object_id must be in ENTITY CANDIDATES. The predicate must be a short exact contiguous substring of CLAIM that states their relationship (a verb or relational phrase, not merely 'is' when another phrase carries the meaning). Keep negation, uncertainty, population, comparison, dose, and time separately as exact CLAIM substrings. A study or patient observation is not a universal medical rule. Association is not causation; a question, option, or co-occurrence is not an assertion.
If an endpoint is missing from ENTITY CANDIDATES, do not invent an ID: list the missing exact CLAIM span in missing_endpoints. If the relation is implicit and has no identifiable predicate span, or is not fully supported by the claim and evidence, do not force an edge. Zero relations is allowed.
Return JSON only: {"relations":[{"subject_id":"E1","predicate":"exact CLAIM substring","object_id":"E2","negated":false,"uncertain":false,"qualifiers":["exact CLAIM substring"]}],"missing_endpoints":["exact CLAIM substring"]}."""

ENTITY_SCHEMA = {"type":"object","properties":{"entities":{"type":"array","items":{
    "type":"object","properties":{"span":{"type":"string"},"type":{"type":"string","enum":list(TYPES)}},
    "required":["span","type"],"additionalProperties":False}}},
    "required":["entities"],"additionalProperties":False}
RELATION_SCHEMA = {"type":"object","properties":{
    "relations":{"type":"array","items":{"type":"object","properties":{
        "subject_id":{"type":"string"},"predicate":{"type":"string"},"object_id":{"type":"string"},
        "negated":{"type":"boolean"},"uncertain":{"type":"boolean"},
        "qualifiers":{"type":"array","items":{"type":"string"}}},
        "required":["subject_id","predicate","object_id","negated","uncertain","qualifiers"],
        "additionalProperties":False}},
    "missing_endpoints":{"type":"array","items":{"type":"string"}}},
    "required":["relations","missing_endpoints"],"additionalProperties":False}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    tmp.replace(path)


def save_rows(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    with tmp.open("w") as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False)+"\n")
    tmp.replace(path)


def rows(path: Path) -> list[dict]:
    with path.open() as f:return [json.loads(line) for line in f]


def render(tokenizer,system: str,user: str) -> str:
    return tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":user}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)


def entity_user(row: dict) -> str:
    return "CLAIM:\n"+row["claim"]+"\nEVIDENCE:\n"+row["evidence"]


def relation_user(row: dict, entities: list[dict]) -> str:
    candidates=[{"id":e["id"],"span":e["span"],"type":e["type"]} for e in entities]
    return entity_user(row)+"\nENTITY CANDIDATES:\n"+json.dumps(candidates,ensure_ascii=False,separators=(",",":"))


def parse_entity(raw: str, claim: str) -> tuple[list[dict],list[str]]:
    errors=[]; result=[];seen=set()
    try:items=json.loads(raw.strip())["entities"]
    except (ValueError,KeyError,TypeError):return [],["invalid_json_or_schema"]
    if not isinstance(items,list):return [],["entities_not_list"]
    for item in items:
        if not isinstance(item,dict):errors.append("non_object_entity");continue
        span=str(item.get("span") or "").strip();kind=str(item.get("type") or "")
        if span not in claim or not span or kind not in TYPES:errors.append("invalid_span_or_type");continue
        if span.casefold() in GENERIC:errors.append("generic_span");continue
        key=(span,kind)
        if key in seen:errors.append("duplicate_span");continue
        seen.add(key);result.append({"id":f"E{len(result)+1}","span":span,"type":kind,
                                 "start":claim.index(span),"end":claim.index(span)+len(span)})
    return result,errors


def parse_relation(raw: str, claim: str, entities: list[dict]) -> tuple[list[dict],list[str],list[str]]:
    errors=[];valid=[]
    try:parsed=json.loads(raw.strip());items=parsed["relations"];missing=parsed["missing_endpoints"]
    except (ValueError,KeyError,TypeError):return [],[],["invalid_json_or_schema"]
    if not isinstance(items,list) or not isinstance(missing,list):return [],[],["invalid_lists"]
    allowed={e["id"] for e in entities}; seen=set()
    for item in items:
        if not isinstance(item,dict):errors.append("non_object_relation");continue
        a=item.get("subject_id");b=item.get("object_id");pred=str(item.get("predicate") or "").strip()
        quals=item.get("qualifiers")
        if a not in allowed or b not in allowed or a==b:errors.append("invalid_endpoint");continue
        if not pred or pred not in claim:errors.append("nonexact_predicate");continue
        if not isinstance(quals,list) or any(not isinstance(q,str) or q not in claim for q in quals):
            errors.append("nonexact_qualifier");continue
        key=(a,pred,b,bool(item.get("negated")),bool(item.get("uncertain")))
        if key in seen:errors.append("duplicate_relation");continue
        seen.add(key);valid.append({"subject_id":a,"predicate":pred,"object_id":b,
                                    "negated":bool(item.get("negated")),"uncertain":bool(item.get("uncertain")),
                                    "qualifiers":quals})
    missing=[m for m in missing if isinstance(m,str) and m in claim]
    return valid,missing,errors


def shard_path(model: str,phase: str,start: int,stop: int,output: Path = OUTPUT) -> Path:
    return output/model/phase/f"batch_{start:05d}_{stop:05d}.jsonl"


def cached_phase(model: str,phase: str,cohort: list[dict],size: int,output: Path = OUTPUT) -> list[dict] | None:
    all_rows=[]
    for start in range(0,len(cohort),size):
        stop=min(start+size,len(cohort));path=shard_path(model,phase,start,stop,output)
        if not path.exists():return None
        part=rows(path)
        if [r["proposition_id"] for r in part]!=[r["proposition_id"] for r in cohort[start:stop]]:
            raise RuntimeError(f"Wrong cached shard identity: {path}")
        all_rows.extend(part)
    return all_rows


def worker(model_name: str,data: Path = DATA, output: Path = OUTPUT) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":raise RuntimeError("Use GPU 1 only: CUDA_VISIBLE_DEVICES=1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER","0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL","ERROR")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer
    cohort=rows(data/"cohort.jsonl");manifest=json.loads((output/"manifest.json").read_text())
    model=ROOT.parent/"models"/model_name;batch_size=manifest["generation"]["batch_size"]
    status=output/model_name/"status.json"
    entities=cached_phase(model_name,"entities",cohort,batch_size,output)
    relations=cached_phase(model_name,"relations",cohort,batch_size,output)
    if entities is not None and relations is not None:
        save(status,{"phase":"complete","completed":2*len(cohort),"total":2*len(cohort)});return
    save(status,{"phase":"loading","completed":0,"total":2*len(cohort)})
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    kw=dict(model=str(model),dtype="bfloat16",tensor_parallel_size=1,
            gpu_memory_utilization=0.80,max_model_len=4096,max_num_seqs=batch_size,
            max_num_batched_tokens=4096,seed=20260922,enforce_eager=True,
            enable_prefix_caching=True,language_model_only=True,disable_log_stats=True)
    if model_name.startswith("Qwen"):kw["additional_config"]={"gdn_prefill_backend":"triton"}
    llm=LLM(**kw)
    for phase in ("entities","relations"):
        if cached_phase(model_name,phase,cohort,batch_size,output) is not None:continue
        if phase=="relations":
            entities=cached_phase(model_name,"entities",cohort,batch_size,output)
            if entities is None:raise RuntimeError("Entity stage incomplete")
        for start in range(0,len(cohort),batch_size):
            stop=min(start+batch_size,len(cohort));path=shard_path(model_name,phase,start,stop,output)
            if path.exists():continue
            selected=cohort[start:stop]
            if phase=="entities":
                prompts=[render(tokenizer,ENTITY_PROMPT,entity_user(r)) for r in selected]
                schema=ENTITY_SCHEMA;max_tokens=384
            else:
                prompts=[render(tokenizer,RELATION_PROMPT,relation_user(r,entities[start+i]["parsed"])) for i,r in enumerate(selected)]
                schema=RELATION_SCHEMA;max_tokens=512
            lengths=[len(tokenizer.encode(p,add_special_tokens=False)) for p in prompts]
            if max(lengths)+max_tokens>4096:raise RuntimeError(f"Overlength {model_name}/{phase}/{start}: {max(lengths)}")
            sampling=SamplingParams(temperature=0,max_tokens=max_tokens,skip_special_tokens=False,
                structured_outputs=StructuredOutputsParams(json=schema))
            tick=time.monotonic()
            outputs=llm.generate(prompts,sampling,use_tqdm=False)
            elapsed=time.monotonic()-tick;records=[]
            for i,(src,out) in enumerate(zip(selected,outputs)):
                completion=out.outputs[0]
                cleaned=clean_completion(completion.text)
                if phase=="entities":parsed,errors=parse_entity(cleaned,src["claim"]);missing=[]
                else:parsed,missing,errors=parse_relation(cleaned,src["claim"],entities[start+i]["parsed"])
                records.append({"proposition_id":src["proposition_id"],"source":src["source"],
                                "origin":src["origin"],"model":model_name,"phase":phase,
                                "raw_output":completion.text,"parsed":parsed,"missing_endpoints":missing,
                                "errors":errors,"finish_reason":completion.finish_reason,
                                "input_tokens":len(out.prompt_token_ids),"output_tokens":len(completion.token_ids),
                                "batch_seconds":elapsed})
            save_rows(path,records)
            completed=(len(cohort) if phase=="relations" else 0)+stop
            save(status,{"phase":phase,"completed":completed,"phase_completed":stop,
                         "phase_total":len(cohort),"total":2*len(cohort)})
    save(status,{"phase":"complete","completed":2*len(cohort),"total":2*len(cohort)})


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker",choices=MODELS)
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--data-root",type=Path,default=DATA)
    ap.add_argument("--output-root",type=Path,default=OUTPUT)
    args=ap.parse_args()
    data=args.data_root.resolve();output=args.output_root.resolve()
    if args.worker:return worker(args.worker,data,output)
    cohort_path=data/"cohort.jsonl";selection_path=data/"selection_summary.json"
    if not cohort_path.is_file() or not selection_path.is_file():raise RuntimeError("Run prepare_rag2_gemma_relation_pilot.py first")
    cohort=rows(cohort_path)
    if len(cohort)!=192 or len({r["proposition_id"] for r in cohort})!=192:raise RuntimeError("Wrong cohort")
    manifest={"version":"gemma_relation_pilot192_v1","cohort":str(cohort_path.resolve()),
              "cohort_sha256":file_digest(cohort_path),"selection_manifest_sha256":file_digest(data/"manifest.json"),
              "entity_prompt_sha256":digest(ENTITY_PROMPT),"relation_prompt_sha256":digest(RELATION_PROMPT),
              "entity_schema":ENTITY_SCHEMA,"relation_schema":RELATION_SCHEMA,
              "models":{name:{"path":str((ROOT.parent/"models"/name).resolve()),
                              "config_sha256":file_digest(ROOT.parent/"models"/name/"config.json")} for name in MODELS},
              "generation":{"batch_size":16,"max_context":4096,"entity_max_output":384,
                            "relation_max_output":512,"temperature":0,"seed":20260922,"gpu":1},
              "metric_limit":"Automatic format and exact-span checks are not semantic relation accuracy. Manual blinded source-only comparison required.",
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "environment":{n:importlib.metadata.version(n) for n in ("vllm","torch","transformers")}}
    output.mkdir(parents=True,exist_ok=True)
    mf=output/"manifest.json"
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise RuntimeError("Experiment contract mismatch; use a new output root")
    save(mf,manifest)
    from transformers import AutoTokenizer
    for name in MODELS:
        model=ROOT.parent/"models"/name
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        n=max(len(tokenizer.encode(render(tokenizer,ENTITY_PROMPT,entity_user(r)),add_special_tokens=False)) for r in cohort)
        if n+384>4096:raise RuntimeError(f"Entity input overlength: {name} {n}+384")
        print(f"[preflight] {name}: {len(cohort)} claims; max entity prompt {n} tokens; relation prompt checked per shard",flush=True)
    if args.preflight_only:return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1")
    names=["preflight","load Qwen","Qwen entity","Qwen relation","load Gemma","Gemma entity","Gemma relation","aggregate"]
    progress=TwoLineProgress(names)
    # Compatible pilot run on the same GPU/model pair measured roughly four minutes.
    progress.estimates=[3.0,25.0,25.0,35.0,35.0,45.0,60.0,2.0]
    def start_stage(index: int,total: int,**kwargs) -> None:
        progress.start(index,total,estimate=progress.estimates[index-1],**kwargs)
    progress.plan(["cached cohort/tokenizer checks","Qwen3.5-9B weights","192 entity outputs","192 relation outputs",
                   "Gemma-4-31B weights","192 entity outputs","192 relation outputs","paired automatic diagnostics"])
    start_stage(1,1,unit="contract");progress.done=1;progress.complete("frozen 192 propositions")
    for j,name in enumerate(MODELS):
        status=output/name/"status.json";status.parent.mkdir(exist_ok=True)
        log=output/name/"worker.log"
        cached_e=cached_phase(name,"entities",cohort,16,output)
        cached_r=cached_phase(name,"relations",cohort,16,output)
        start_stage(2+3*j,1,unit="model")
        if cached_e is not None and cached_r is not None:
            progress.complete("all outputs cached")
            for phase_index in (3+3*j,4+3*j):
                start_stage(phase_index,len(cohort),initial=len(cohort),unit="proposition")
                progress.complete("cached")
            continue
        with log.open("a") as handle:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker",name,
                                   "--data-root",str(data),"--output-root",str(output)],stdout=handle,stderr=subprocess.STDOUT)
            current=None
            while True:
                if status.exists():
                    s=json.loads(status.read_text());phase=s["phase"]
                    if phase!="loading" and current is None:
                        progress.complete("loaded")
                        start_stage(3+3*j,len(cohort),unit="proposition")
                        current="entities"
                    if phase=="relations" and current=="entities":
                        progress.done=len(cohort);progress.complete("entity extraction complete")
                        start_stage(4+3*j,len(cohort),unit="proposition");current="relations"
                    if phase=="complete" and current=="entities":
                        progress.done=len(cohort);progress.complete("entity extraction complete")
                        start_stage(4+3*j,len(cohort),unit="proposition");current="relations"
                    if current=="entities":progress.done=s.get("phase_completed",0)
                    if current=="relations":progress.done=s.get("phase_completed",len(cohort))
                progress.render()
                code=proc.poll()
                if code is not None:
                    if code:raise RuntimeError(f"{name} failed; see {log}. Rerun same command to resume shards")
                    break
                time.sleep(2)
        if current=="entities":
            progress.done=len(cohort);progress.complete("entity extraction complete")
            start_stage(4+3*j,len(cohort),unit="proposition")
        progress.done=len(cohort);progress.complete("relation extraction complete")
    start_stage(8,1,unit="report")
    summary={"cohort":len(cohort),"models":{},"warning":"Automatic diagnostics only; no source-relative semantic gold"}
    review=[]
    by_model={}
    for name in MODELS:
        es=cached_phase(name,"entities",cohort,16,output);rs=cached_phase(name,"relations",cohort,16,output)
        if es is None or rs is None:raise RuntimeError(f"Incomplete output for {name}")
        by_model[name]=(es,rs)
        ec=Counter();rc=Counter()
        for row,e,r in zip(cohort,es,rs):
            ec["entities"]+=len(e["parsed"]);ec["nonempty"]+=bool(e["parsed"])
            ec["errors"]+=len(e["errors"]);ec["truncated"]+=e["finish_reason"]=="length"
            rc["relations"]+=len(r["parsed"]);rc["nonempty"]+=bool(r["parsed"])
            rc["errors"]+=len(r["errors"]);rc["truncated"]+=r["finish_reason"]=="length"
            rc["missing_endpoints"]+=len(r["missing_endpoints"])
        summary["models"][name]={"entity":dict(ec),"relation":dict(rc)}
    for i,row in enumerate(cohort):
        review.append({"proposition_id":row["proposition_id"],"source":row["source"],"origin":row["origin"],
                       "claim":row["claim"],"evidence":row["evidence"],
                       "models":{name:{"entities":by_model[name][0][i]["parsed"],
                                        "relations":by_model[name][1][i]["parsed"]} for name in MODELS},
                       "audit":{"source_supported":None,"entity_precision":None,"entity_recall":None,
                                "relation_correct":None,"relation_missing":None,"normalization_correct":None}})
    save_rows(output/"paired_review.jsonl",review)
    save(output/"summary.json",summary)
    progress.done=1;progress.complete("paired outputs and review sheet saved")
    progress.close(force_complete=True)
    print("[final] " + json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

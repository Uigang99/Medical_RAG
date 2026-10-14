#!/usr/bin/env python3
"""Evaluate the frozen six-condition pilot AFTER shared LoRA training finishes.

No training, checkpoint selection, new retrieval, or official TEST-6545 scoring.
No GPU work in --preflight-only. All generated outputs use the same strict parser.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from medrag.text_graph_evaluation import CONDITIONS, GROUPS, generate_tokens, group, pack_evaluation_pair, parse_answer, summarize
from medrag.training.graph_input_data import (SYSTEM, SourceGraph, atomic_json, atomic_jsonl, digest,
    ensure_contract, identity, jsonl, load_population, sha_file)
from medrag.training.graph_input_modules import GraphPrompt, training_attention_backend
from medrag.training.graph_input_progress import Progress
from train_rag2_text_kg_input_pilot import bundle_identity, load_torch, lock_file, set_seed, torch_save


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, default=ROOT / "results/rag2_shared_medical_lora_v1")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_text_graph_six_conditions_v1")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def verified_manifest(path):
    data = json.loads(path.read_text())
    if digest(data["contract"]) != data["contract_sha256"]:
        raise RuntimeError(f"Corrupt manifest: {path}")
    return data


def verify_identity(expected):
    current = identity(expected["path"], full_hash="sha256" in expected)
    if current != expected:
        raise RuntimeError(f"Frozen input changed: {expected['path']}")


def preflight(args):
    """Fail on unfinished training BEFORE tokenizer, source scans, or GPU loading."""
    if min(args.batch_size, args.embedding_batch_size, args.max_new_tokens) < 1:
        raise ValueError("Batch sizes and output budget must be positive")
    completion_path = args.training_root / "COMPLETE.json"
    if not completion_path.is_file():
        raise RuntimeError(f"Shared LoRA training is not complete: {completion_path}. "
                           "Wait for training to finish; do not use last.pt or an intermediate epoch.")
    done = json.loads(completion_path.read_text())
    tm = verified_manifest(args.training_root / "manifest.json")
    tc = tm["contract"]
    if done["contract_sha256"] != tm["contract_sha256"] or tc["purpose"] != "six_condition_training_only":
        raise RuntimeError("Training completion/contract mismatch (smoke models are not evaluable)")
    if done["pilot_evaluation_questions_used"] != 0 or done["epochs"] != tc["training"]["epochs"]:
        raise RuntimeError("Training is incomplete or held-out questions were used")
    best = json.loads((args.training_root / "best.json").read_text())
    if best["adapter"] != done["trained_llm_adapter"] or best["validation_answer_ce"] != done["best_text_validation_ce"]:
        raise RuntimeError("Selected LoRA differs from completed TEXT-validation selection")
    adapter = Path(best["adapter"]).resolve()
    if adapter.parent != args.training_root.resolve():
        raise RuntimeError("Selected LoRA must belong to this training run")
    lora_files = [identity(adapter / n, full_hash=True) for n in ("adapter_config.json", "adapter_model.safetensors")]
    model = Path(done["base_llm"])
    if bundle_identity(model) != tc["model"]:
        raise RuntimeError("Base Llama is not the training checkpoint")
    data_root = Path(done["data_root"])
    dm = verified_manifest(data_root / "manifest.json")
    dc = dm["contract"]
    saved_data = json.loads((data_root / "COMPLETE.json").read_text())
    if dm["contract_sha256"] != tc["data_contract"] or saved_data["contract_sha256"] != tc["data_contract"]:
        raise RuntimeError("Different training/evaluation data provenance")
    for name, value in saved_data["file_sha256"].items():
        if sha_file(data_root / name) != value:
            raise RuntimeError(f"Frozen split/cache changed: {name}")
    assignments = list(jsonl(data_root / "split_assignments.jsonl"))
    if len(assignments) != 6545 or len({r["sample_id"] for r in assignments}) != len(assignments):
        raise RuntimeError("Broken split assignment IDs")
    counts = {s: sum(r["split"] == s for r in assignments) for s in dc["counts"]}
    if counts != dc["counts"] or counts.get("pilot_evaluation") != 512:
        raise RuntimeError("Expected the original fixed 512-question evaluation split")
    seen_groups = {}
    for r in assignments:
        seen_groups.setdefault(r["duplicate_group"], set()).add(r["split"])
    if any(len(s) > 1 for s in seen_groups.values()):
        raise RuntimeError("Question duplicate group crosses splits")
    for entry in dc["benchmark"] + dc["tokenizer"] + [dc[k] for k in ("candidates", "graph", "graph_manifest", "graph_summary", "code")]:
        verify_identity(entry)
    if dc["top_k"] != 8 or dc["extra_neighbors"] or dc["system_prompt"] != SYSTEM:
        raise RuntimeError("Unexpected source/prompt policy")
    graph_path = Path(done["shared_graph_adapter"])
    if sha_file(graph_path) != done["graph_checkpoint_sha256"] or identity(graph_path,full_hash=True) != tc["graph_checkpoint"]:
        raise RuntimeError("The shared frozen graph checkpoint changed")
    gm = verified_manifest(graph_path.parent / "manifest.json")
    gdone = json.loads((graph_path.parent / "COMPLETE.json").read_text())
    if gm["contract_sha256"] != tc["graph_contract"] or gdone["contract_sha256"] != tc["graph_contract"]:
        raise RuntimeError("Graph adapter contract mismatch")
    gc = gm["contract"]
    if gc["data_sha256"] != tc["data_contract"] or gc["model"] != tc["model"]:
        raise RuntimeError("Graph and LoRA training did not use the same paired source/base Llama")
    for rel, expected in gc["code"].items():
        if sha_file(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen graph implementation changed: {rel}")
    feature_contract = gc["features"]
    encoder = Path(next(x["path"] for x in feature_contract["encoder"] if x["path"].endswith("/modules.json"))).parent
    if bundle_identity(encoder) != feature_contract["encoder"]:
        raise RuntimeError("Frozen sentence encoder changed")
    old_features = data_root / "features" / digest(feature_contract)[:20]
    if verified_manifest(old_features / "manifest.json")["contract"] != feature_contract:
        raise RuntimeError("Frozen graph-feature cache contract mismatch")
    if importlib.metadata.version("sentence-transformers") != feature_contract["sentence_transformers"]:
        raise RuntimeError("Sentence encoder library differs from training")
    for lib, version in tc["environment"].items():
        if importlib.metadata.version(lib) != version:
            raise RuntimeError(f"Runtime differs from training: {lib}")
    for root in (args.training_root, data_root, graph_path.parent, model, encoder):
        if args.output_root.resolve() == root.resolve() or root.resolve() in args.output_root.resolve().parents:
            raise ValueError("Evaluation outputs cannot be inside a training/input directory")
    contract = {"version": "text_graph_six_conditions_v1", "hypothesis": "paired TEXT versus graph-vector input under shared Llama adaptation",
        "cohort": "frozen pilot_evaluation split ONLY", "n": 512,
        "training_contract": tm["contract_sha256"], "training_completion": identity(completion_path,full_hash=True),
        "paired_data_contract": dm["contract_sha256"], "assignments_sha256": saved_data["file_sha256"]["split_assignments.jsonl"],
        "base_llm": tc["model"], "lora": lora_files, "checkpoint_selection": "preselected best TEXT validation CE; never reevaluate selection",
        "graph_checkpoint": tc["graph_checkpoint"], "graph_contract": gm["contract_sha256"],
        "sources": {k: dc[k] for k in ("benchmark", "candidates", "graph", "graph_manifest", "graph_summary")},
        "feature_contract": feature_contract,
        "conditions": [{"name":n,"trained_llm":t,"input":m} for n,t,m in CONDITIONS],
        "system_prompt": SYSTEM, "top_k": 8, "extra_neighbors": False,
        "packing": "same ordered source propositions/edges; remove trailing whole propositions jointly; fixed max-new-token reserve, never gold-dependent",
        "max_context": dc["max_context"], "graph_soft_tokens": 1, "graph_position": "after BOS, same as graph training",
        "empty_graph_policy": "retain all questions; GraphPrompt zero pooled representation + frozen projector, as trained",
        "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1,
            "output": "Answer: letter, NO rationale", "batch_size": args.batch_size,
            "oom": "retry failed batch at half micro batch; restore configured size next batch; fail at one",
            "fallback": "none; invalid answers count as wrong without regeneration", "seed": args.seed},
        "embedding_batch_size": args.embedding_batch_size,
        "metrics": {"primary": "trained GRAPH minus trained TEXT Macro-3", "secondary": "micro, three groups, eight datasets, paired transitions",
                    "macro3": "unweighted MedMCQA/MedQA/pooled-MMLU accuracy", "bootstrap": "2000 paired stratified draws, seed=42"},
        "no_training": True, "no_heldout_tuning": True,
        "claim_limit": "TEST-repartition pilot; shared LoRA and matched sources do not isolate pure topology or pure domain learning",
        "code": {str(p.relative_to(ROOT)): sha_file(p) for p in (Path(__file__), ROOT / "medrag/text_graph_evaluation.py",
                  ROOT / "medrag/training/graph_input_data.py", ROOT / "medrag/training/graph_input_modules.py")},
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "sentence-transformers")}}
    return {"contract": contract, "data": dc, "assignments": assignments, "model": model, "adapter": adapter,
            "graph_path": graph_path, "graph_contract": gc, "encoder": encoder, "old_features": old_features}


def prepare_pairs(args, context, tokenizer, progress):
    out, dc = args.output_root, context["data"]
    path = out / "evaluation_pairs.jsonl"
    marker = out / "PAIRS_COMPLETE.json"
    if marker.exists():
        if sha_file(path) != json.loads(marker.read_text())["sha256"]:
            raise RuntimeError("Prepared evaluation pairs changed")
        rows = list(jsonl(path))
        progress.stage(1,512,initial=512,unit="q"); progress.complete(f"cached paired inputs={path}")
        return rows
    benchmark = Path(dc["benchmark"][0]["path"]).parents[3]
    population, identities = load_population(benchmark)
    if identities != dc["benchmark"]:
        raise RuntimeError("Benchmark changed since training")
    selected = {a["sample_id"] for a in context["assignments"] if a["split"] == "pilot_evaluation"}
    questions = {r["sample_id"]: r for r in population if r["sample_id"] in selected}
    if len(questions) != 512:
        raise RuntimeError("Missing evaluation questions")
    candidates = {}
    for r in jsonl(dc["candidates"]["path"]):
        sid = r["sample_id"]
        if sid not in questions:
            continue
        q = questions[sid]
        if sid in candidates or r["dataset"] != q["dataset"] or r["row_idx"] != q["row_idx"] or r["key"] != f"{q['dataset']}::test::{sid}::{q['row_idx']}":
            raise RuntimeError(f"Candidate alignment mismatch: {sid}")
        docs = r["reranked_documents"]
        ids = [d.get("stable_id") or d.get("corpus_id") for d in docs]
        if len(docs) != 8 or [d["rerank_rank"] for d in docs] != list(range(1,9)) or None in ids or len(set(ids)) != 8:
            raise RuntimeError(f"Invalid ordered Top-8: {sid}")
        candidates[sid] = ids
    if set(candidates) != selected:
        raise RuntimeError("Incomplete evaluation retrieval cache")
    ordered = sorted(selected, key=lambda sid: digest([dc["seed"],sid]))
    progress.stage(1,len(ordered),unit="q")
    graph, rows = SourceGraph(dc["graph"]["path"]), []
    try:
        for sid in ordered:
            r = {**questions[sid],"split":"pilot_evaluation","document_ids":candidates[sid]}
            rows.append(pack_evaluation_pair(tokenizer,r,graph.collect(candidates[sid]),dc["max_context"],args.max_new_tokens))
            progress.update()
    finally:
        graph.close()
    atomic_jsonl(path,rows)
    atomic_json(marker,{"sha256":sha_file(path),"n":len(rows),"ids_sha256":digest(ordered),
        "empty_graphs":sum(not r["graph"]["node_ids"] for r in rows),
        "packed_questions":sum(r["packing"]["removed_propositions"]>0 for r in rows),
        "groups":{g:sum(group(r["dataset"])==g for r in rows) for g in GROUPS}})
    progress.complete(f"paired inputs={path}; no labels enter prompts, evidence selection or graph encoding")
    return rows


def prepare_features(args, context, rows, progress):
    import torch
    out = args.output_root / "features"
    out.mkdir(parents=True,exist_ok=True)
    texts = sorted({t for r in rows for t in r["graph"]["node_texts"]+r["graph"]["edge_texts"]})
    atomic_json(out/"texts.json",texts)
    paths = [(i,out/f"{i//512:05d}.pt") for i in range(0,len(texts),512)]
    vectors = {}
    progress.stage(2,max(1,len(paths)),unit="shard")
    for i,path in paths:
        if path.exists():
            item = load_torch(path)
            if item["text_sha256"] != digest(texts[i:i+512]) or len(item["values"]) != len(texts[i:i+512]):
                raise RuntimeError(f"Corrupt evaluation feature shard: {path}")
            vectors.update(zip(texts[i:i+512],item["values"],strict=True))
        progress.update()
    old = context["old_features"]
    old_texts = json.loads((old/"texts.json").read_text())
    if digest(old_texts) != context["graph_contract"]["features"]["texts_sha256"]:
        raise RuntimeError("Frozen cached graph-attribute vocabulary changed")
    wanted = set(texts)-set(vectors)
    indices = {i:t for i,t in enumerate(old_texts) if t in wanted}
    for shard in sorted({i//512 for i in indices}):
        item = load_torch(old/f"{shard:05d}.pt")
        if item["text_sha256"] != digest(old_texts[shard*512:shard*512+512]):
            raise RuntimeError("Frozen training feature shard hash mismatch")
        for i in range(shard*512,min(len(old_texts),shard*512+512)):
            if i in indices: vectors[indices[i]]=item["values"][i%512].clone()
    missing = [t for t in texts if t not in vectors]
    progress.complete(f"attributes={len(texts)}; cached={len(vectors)}; new={len(missing)}; all training caches read-only")
    progress.stage(3,1,initial=0 if missing else 1,unit="model")
    encoder = None
    if missing:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(context["encoder"]),device="cuda:0",local_files_only=True)
        encoder.eval(); encoder.requires_grad_(False)
    progress.complete("frozen SBERT loaded" if missing else "no uncached attributes; SBERT load skipped")
    progress.stage(4,max(1,len(missing)),initial=0 if missing else 1,unit="text")
    clipped = 0
    # Save each full feature shard immediately, including shards entirely reused.
    for i,path in paths:
        if path.exists(): continue
        new = [t for t in texts[i:i+512] if t not in vectors]
        if new:
            clipped += sum(len(encoder.tokenizer.encode(t,add_special_tokens=True,truncation=False)) > encoder.max_seq_length for t in new)
            batch_size = min(args.embedding_batch_size,len(new))
            while True:
                try:
                    with torch.inference_mode():
                        values = encoder.encode(new,batch_size=batch_size,convert_to_tensor=True,
                                                normalize_embeddings=True,show_progress_bar=False).cpu().float()
                    break
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    if batch_size == 1: raise
                    batch_size = max(1,batch_size//2)
                    progress.log(f"[SBERT OOM] retry batch={batch_size}; restore next shard")
            if not torch.isfinite(values).all(): raise FloatingPointError("Non-finite sentence embeddings")
            vectors.update(zip(new,values,strict=True)); progress.update(len(new))
        torch_save(path,{"text_sha256":digest(texts[i:i+512]),"values":torch.stack([vectors[t] for t in texts[i:i+512]])})
    atomic_json(out/"summary.json",{"texts":len(texts),"new_this_run":len(missing),"truncated_new_texts":clipped,
        "truncation_policy":"same frozen sentence encoder max_seq_length as training"})
    progress.complete(f"feature cache={out}; no fitting or vocabulary expansion of the encoder")
    del encoder
    gc.collect(); torch.cuda.empty_cache()
    return vectors


def prepare_graph_prompts(args, context, rows, features, progress):
    import torch
    path = args.output_root / "graph_prompts.pt"
    ids = [r["sample_id"] for r in rows]
    progress.stage(5,1,initial=int(path.exists()),unit="model")
    if path.exists():
        item = load_torch(path)
        if item["ids"] != ids or item["checkpoint_sha256"] != context["contract"]["graph_checkpoint"]["sha256"]:
            raise RuntimeError("Incompatible graph prompt cache")
        result = item["prompts"]
        progress.complete("cached graph soft tokens; graph model load skipped")
        progress.stage(6,len(rows),initial=len(rows),unit="q")
    else:
        saved = load_torch(context["graph_path"])
        a = context["graph_contract"]["architecture"]
        dim = saved["adapter"]["layers.0.query.weight"].shape[1]
        llm_dim = saved["adapter"]["projector.2.weight"].shape[0]
        model = GraphPrompt(dim,llm_dim,a["graph_hidden"],a["graph_layers"],a["graph_heads"]).to("cuda:0")
        model.load_state_dict(saved["adapter"]); model.eval(); model.requires_grad_(False)
        progress.complete("frozen graph encoder/projector loaded; same checkpoint for both GRAPH conditions")
        progress.stage(6,len(rows),unit="q")
        result = []
        with torch.inference_mode():
            for r in rows:
                g = r["graph"]
                x = torch.stack([features[t] for t in g["node_texts"]]).to("cuda:0") if g["node_texts"] else torch.empty(0,dim,device="cuda:0")
                e = torch.stack([features[t] for t in g["edge_texts"]]).to("cuda:0") if g["edge_texts"] else torch.empty(0,dim,device="cuda:0")
                result.append(model(x,torch.tensor(g["edge_index"],device="cuda:0",dtype=torch.long),e).cpu())
                progress.update()
        result = torch.stack(result)
        torch_save(path,{"ids":ids,"prompts":result,"checkpoint_sha256":context["contract"]["graph_checkpoint"]["sha256"]})
        del model, saved
        gc.collect(); torch.cuda.empty_cache()
    if result.shape != (len(rows),1,4096) or not torch.isfinite(result).all():
        raise RuntimeError("Invalid graph prompt vectors")
    progress.complete(f"graph prompts={path}; text evidence is NOT appended in GRAPH conditions")
    return {sid:result[i] for i,sid in enumerate(ids)}


def load_prediction_shard(path, ids, contract_hash, condition):
    value = json.loads(path.read_text())
    if value["contract_sha256"] != contract_hash or value["condition"] != condition or [r["sample_id"] for r in value["rows"]] != ids:
        raise RuntimeError(f"Prediction shard contract/ID mismatch: {path}")
    return value["rows"]


@training_attention_backend()
def evaluate_conditions(args, context, rows, prompts, tokenizer, progress, contract_hash):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    progress.stage(7,1,unit="model")
    base = AutoModelForCausalLM.from_pretrained(context["model"],local_files_only=True,dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(base,context["adapter"],is_trainable=False,local_files_only=True)
    model.eval(); model.requires_grad_(False); model.gradient_checkpointing_disable()
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("Evaluation has trainable weights")
    llm = model.get_base_model()
    pad_id = tokenizer.eos_token_id
    stop_ids = {pad_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    if None in stop_ids: raise RuntimeError("Missing Llama stop token")
    eos = sorted(stop_ids)
    progress.complete("one shared Llama loaded; Base disables LoRA, Trained enables the SAME LoRA")
    all_predictions = {}
    # Explicit approximate cost model only until each condition measures its rate.
    costs = {name:sum(len(r[mode+"_prompt_ids"])+int(mode=="graph")+32*args.max_new_tokens for r in rows)
             for name,_,mode in CONDITIONS}
    progress.weights[-1] = 1.  # small CPU aggregation, later replaced by measured time
    for index,(name,trained,mode) in enumerate(CONDITIONS,8):
        root = args.output_root / "predictions" / name
        root.mkdir(parents=True,exist_ok=True)
        chunks = [(i,root/f"{i//args.batch_size:05d}.json") for i in range(0,len(rows),args.batch_size)]
        cached = {i:load_prediction_shard(path,[r["sample_id"] for r in rows[i:i+args.batch_size]],contract_hash,name)
                  for i,path in chunks if path.exists()}
        progress.stage(index,len(rows),initial=sum(len(v) for v in cached.values()),unit="q",kind="generation")
        predictions = []
        fresh_start, fresh_cost = time.monotonic(), 0
        with (contextlib.nullcontext() if trained else model.disable_adapter()):
            for start,path in chunks:
                batch = rows[start:start+args.batch_size]
                if start in cached:
                    predictions.extend(cached[start]); continue
                size = len(batch)
                while True:
                    try:
                        outputs = []
                        for i in range(0,len(batch),size):
                            sub = batch[i:i+size]
                            soft = [prompts[r["sample_id"]] for r in sub] if mode == "graph" else None
                            outputs.extend(generate_tokens(llm,[r[mode+"_prompt_ids"] for r in sub],pad_id=pad_id,
                                eos_ids=eos,max_new_tokens=args.max_new_tokens,soft_prompts=soft))
                        break
                    except torch.cuda.OutOfMemoryError:
                        gc.collect(); torch.cuda.empty_cache()
                        if size == 1: raise
                        size = max(1,size//2)
                        progress.log(f"[OOM] {name} batch {start}: retry micro={size}; restore {args.batch_size} next batch")
                records = []
                for row,result in zip(batch,outputs,strict=True):
                    text = tokenizer.decode(result["generated_ids"],skip_special_tokens=True)
                    records.append({"sample_id":row["sample_id"],"dataset":row["dataset"],"condition":name,
                        **result,"output_text":text,"prediction":parse_answer(text,row["options"]),
                        "prompt_tokens":len(row[mode+"_prompt_ids"])+int(mode=="graph"),"actual_micro_batch":size})
                atomic_json(path,{"contract_sha256":contract_hash,"condition":name,"rows":records})
                predictions.extend(records)
                progress.update(len(batch))
                fresh_cost += sum(len(r[mode+"_prompt_ids"])+int(mode=="graph")+32*args.max_new_tokens for r in batch)
                rate = (time.monotonic()-fresh_start)/fresh_cost
                for j,(future,_,_) in enumerate(CONDITIONS,8):
                    if j > index: progress.weights[j] = rate*costs[future]
        atomic_jsonl(args.output_root/"predictions"/f"{name}.jsonl",predictions)
        all_predictions[name] = predictions
        progress.complete(f"condition={name}; outputs={len(predictions)}; path={root}; next condition reuses original evidence")
    return all_predictions


def write_summary(args, rows, predictions, contract_hash, progress):
    progress.stage(14,1,unit="report")
    report = summarize(rows,predictions)
    report["contract_sha256"] = contract_hash
    report["evidence_statistics"] = json.loads((args.output_root/"PAIRS_COMPLETE.json").read_text())
    report["subgroups"] = {}
    for label,select in (("empty_graph",lambda r:not r["graph"]["node_ids"]),
                         ("jointly_truncated",lambda r:r["packing"]["removed_propositions"]>0)):
        chosen = {r["sample_id"]:r for r in rows if select(r)}
        statistics = {}
        for name,items in predictions.items():
            subset = [p for p in items if p["sample_id"] in chosen]
            correct = sum(p["prediction"] == chosen[p["sample_id"]]["answer"] for p in subset)
            statistics[name] = {"n":len(subset),"correct":correct,
                "accuracy_pct":100*correct/len(subset) if subset else None,
                "parse_failures":sum(p["prediction"] is None for p in subset)}
        report["subgroups"][label] = {"n":len(chosen),"conditions":statistics}
    atomic_json(args.output_root/"summary.json",report)
    comparisons = {name:{p["sample_id"]:p for p in items} for name,items in predictions.items()}
    atomic_jsonl(args.output_root/"paired_predictions.jsonl",({"sample_id":r["sample_id"],"dataset":r["dataset"],
        "gold":r["answer"],"conditions":{name:{**items[r["sample_id"]],"correct":items[r["sample_id"]]["prediction"]==r["answer"]} for name,items in comparisons.items()}} for r in rows))
    target = args.output_root/"summary.csv"
    temp = target.with_suffix(".csv.tmp")
    with temp.open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition","n","correct","parse_failures","micro_pct","macro3_pct",*GROUPS])
        for name,_,_ in CONDITIONS:
            s=report["conditions"][name]
            writer.writerow([name,s["n"],s["correct"],s["parse_failures"],s["micro_pct"],s["macro3_pct"],*[s["groups"][g]["accuracy_pct"] for g in GROUPS]])
    os.replace(temp,target)
    atomic_json(args.output_root/"COMPLETE.json",{"contract_sha256":contract_hash,"n":len(rows),"conditions":6,
        "summary_sha256":sha_file(args.output_root/"summary.json"),"paired_predictions_sha256":sha_file(args.output_root/"paired_predictions.jsonl")})
    progress.update(); progress.complete(f"summary={target}; paired transitions and confidence intervals saved")
    progress.log(f"[finished] 6 conditions x {len(rows)} questions; report={args.output_root/'summary.json'}")


def main():
    args = arguments()
    names = ["verify completion","pair held-out Top8","read feature caches","load frozen SBERT","encode new attributes",
             "load graph adapter","graph vectors","load Llama + LoRA"]
    names += [name for name,_,_ in CONDITIONS] + ["paired summary"]
    progress = Progress(names)
    try:
        from transformers.utils import logging
        logging.set_verbosity_error(); logging.disable_progress_bar()
        progress.log("[plan] verify completed training -> fixed 512 paired Top-8 inputs -> cached/frozen graph vectors -> six greedy answer-only conditions -> paired report; NO training, no retrieval")
        progress.log("[runtime] GPU 1 only; batch=16 default; unknown ETA until fresh embedding/generation batches calibrate; exact cached counts at each stage")
        progress.stage(0,1)
        context = preflight(args)
        if args.preflight_only:
            progress.complete("all checkpoint/data/split contracts passed; no tokenization/GPU/evaluation")
            return
        with lock_file(args.output_root/".run.lock"):
            contract_hash = ensure_contract(args.output_root/"manifest.json",context["contract"])
            progress.complete("training complete; selected LoRA and frozen graph adapter match; 512 fixed held-out IDs")
            if (args.output_root/"COMPLETE.json").exists():
                done=json.loads((args.output_root/"COMPLETE.json").read_text())
                if (done["contract_sha256"] != contract_hash or
                        done["summary_sha256"] != sha_file(args.output_root/"summary.json") or
                        done["paired_predictions_sha256"] != sha_file(args.output_root/"paired_predictions.jsonl")):
                    raise RuntimeError("Evaluation completion mismatch")
                progress.log(f"[already complete] {args.output_root}; no regeneration")
                return
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(context["model"],local_files_only=True)
            rows = prepare_pairs(args,context,tokenizer,progress)
            # Identical lock to the training script: accidentally launching early
            # never competes with its GPU allocation or touches its checkpoints.
            with lock_file(ROOT/"results/.shared_medical_lora_gpu1.lock"):
                import torch
                if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
                    raise RuntimeError("Use the supplied wrapper with visible physical GPU 1")
                set_seed(args.seed)
                if not (args.output_root/"provenance.json").exists():
                    atomic_json(args.output_root/"provenance.json",{"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"gpu":torch.cuda.get_device_name(0),"visible_physical_gpu":1})
                features = prepare_features(args,context,rows,progress)
                prompts = prepare_graph_prompts(args,context,rows,features,progress)
                del features
                predictions = evaluate_conditions(args,context,rows,prompts,tokenizer,progress,contract_hash)
            write_summary(args,rows,predictions,contract_hash,progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; completed prediction batches are durable under {args.output_root/'predictions'}; rerun the identical command after fixing the cause")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

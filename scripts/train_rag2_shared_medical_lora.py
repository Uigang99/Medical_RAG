#!/usr/bin/env python3
"""Train ONE shared medical-QA LoRA; do not run the six-condition evaluation.

Reuses frozen train/validation examples and the already trained graph adapter.
Only TEXT answer CE selects the LoRA checkpoint. Final graph validation is a
non-selecting compatibility diagnostic; it never updates either model.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, jsonl, sha_file
from medrag.training.graph_input_progress import Progress
from medrag.training.graph_input_modules import GraphPrompt, training_attention_backend
from medrag.training.shared_medical_lora import TARGET_MODULES, adapt_llama, adapter_state, restore_adapter, qa_losses
from train_rag2_text_kg_input_pilot import bundle_identity, lock_file, load_torch, restore_rng, rng_state, set_seed, torch_save


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/pilots/rag2_text_kg_input_test_resplit_v1")
    p.add_argument("--graph-root", type=Path, default=ROOT / "results/rag2_text_kg_input_training_v1_sdpa_fix/graph")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--effective-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-steps", type=int, default=25)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="Stop after this many optimizer updates, in a separate default output root; no full training")
    args = p.parse_args()
    if args.output_root is None:
        suffix = "_smoke" if args.smoke_steps else ""
        args.output_root = ROOT / f"results/rag2_shared_medical_lora_v1{suffix}"
    return args


class LoggedProgress(Progress):
    def __init__(self, names, output):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        super().__init__(names)

    def log(self, message):
        with self.lock:
            with (self.output / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"time": datetime.datetime.now().isoformat(), "message": message}) + "\n")
            super().log(message)


def preflight(args):
    if min(args.epochs, args.micro_batch_size, args.effective_batch_size, args.rank, args.alpha,
           args.checkpoint_steps) < 1 or args.smoke_steps < 0 or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    if args.micro_batch_size > args.effective_batch_size:
        raise ValueError("Micro batch exceeds effective batch")
    if args.output_root.resolve() in (args.model.resolve(), args.data_root.resolve(), args.graph_root.resolve()):
        raise ValueError("Output directory overlaps a required input")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("Need at least 8 GiB free for resumable training artifacts")
    dc = json.loads((args.data_root / "COMPLETE.json").read_text())
    dm = json.loads((args.data_root / "manifest.json").read_text())
    if digest(dm["contract"]) != dc["contract_sha256"] or dm["contract_sha256"] != dc["contract_sha256"]:
        raise RuntimeError("Prepared-data contract mismatch")
    for name, value in dc["file_sha256"].items():
        if sha_file(args.data_root / name) != value:
            raise RuntimeError(f"Prepared data changed: {name}")
    records = list(jsonl(args.data_root / "train_validation.jsonl"))
    assignments = list(jsonl(args.data_root / "split_assignments.jsonl"))
    expected = {r["sample_id"]: r["split"] for r in assignments if r["split"] in ("train", "validation")}
    if len(records) != len(expected) or {r["sample_id"]: r["split"] for r in records} != expected:
        raise RuntimeError("Training input includes missing, duplicated, or held-out questions")
    groups = {}
    for r in assignments:
        groups.setdefault(r["duplicate_group"], set()).add(r["split"])
    if any(len(v) > 1 for v in groups.values()):
        raise RuntimeError("A duplicate question group crosses splits")
    graph_manifest = json.loads((args.graph_root / "manifest.json").read_text())
    graph_contract = graph_manifest["contract"]
    complete = json.loads((args.graph_root / "COMPLETE.json").read_text())
    if (graph_contract["data_sha256"] != dc["contract_sha256"] or
            digest(graph_contract) != graph_manifest["contract_sha256"] or
            complete["contract_sha256"] != graph_manifest["contract_sha256"]):
        raise RuntimeError("Graph adapter was trained on different examples or has a broken contract")
    model_id = bundle_identity(args.model)
    if model_id != graph_contract["model"]:
        raise RuntimeError("Use the exact same base Llama as graph training")
    graph_path = args.graph_root / "best.pt"
    graph = load_torch(graph_path)
    if graph["contract_sha256"] != graph_manifest["contract_sha256"] or graph["validation_loss"] != complete["best_validation_answer_ce"]:
        raise RuntimeError("Graph best checkpoint does not match its completion record")
    import torch
    if not all(torch.isfinite(t).all() for t in graph["adapter"].values()):
        raise RuntimeError("Graph adapter has non-finite weights")
    features = graph_contract["features"]
    feature_root = args.data_root / "features" / digest(features)[:20]
    if json.loads((feature_root / "manifest.json").read_text())["contract"] != features:
        raise RuntimeError("Frozen graph attribute cache mismatch")
    texts = json.loads((feature_root / "texts.json").read_text())
    if digest(texts) != features["texts_sha256"] or len(texts) != features["count"]:
        raise RuntimeError("Graph feature vocabulary mismatch")
    for i in range(0, len(texts), 512):
        if not (feature_root / f"{i//512:05d}.pt").is_file():
            raise FileNotFoundError(f"Missing frozen feature shard {i//512}; no implicit regeneration")
    for r in records:
        for mode in ("text", "graph"):
            ids = r[mode + "_prompt_ids"]
            if not ids or not all(isinstance(x, int) for x in ids):
                raise RuntimeError(f"Invalid token IDs: {r['sample_id']}")
            if len(ids) + len(r["target_ids"]) + (mode == "graph") > dm["contract"]["max_context"]:
                raise RuntimeError(f"Context exceeded: {r['sample_id']}")
    contract = {
        "version": "shared_medical_lora_v1", "purpose": "runtime_smoke" if args.smoke_steps else "six_condition_training_only",
        "hypothesis": "adapt ONE shared Llama to the same medical QA examples used by the graph adapter",
        "data_contract": dc["contract_sha256"], "paired_data": dc["file_sha256"],
        "training_counts": {s: sum(r["split"] == s for r in records) for s in ("train", "validation")},
        "model": model_id, "graph_checkpoint": identity(graph_path, full_hash=True),
        "graph_contract": graph_manifest["contract_sha256"], "graph_best_epoch": graph["epoch"],
        "input": "cached question/options + original source propositions; NO soft prompt; NO graph during LoRA training",
        "target": dm["contract"]["target"], "generation": "none; teacher-forced CE only",
        "loss": "mean per-question answer-token cross entropy; prompt tokens masked",
        "selection": "minimum TEXT validation CE only; graph compatibility does not choose checkpoint",
        "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": 0., "target_modules": list(TARGET_MODULES),
                 "base_weights": "read-only; effective Llama weights change through LoRA; no graph updates"},
        "training": {k: getattr(args, k) for k in ("epochs", "micro_batch_size", "effective_batch_size",
                     "learning_rate", "weight_decay", "seed", "checkpoint_steps")},
        "optimizer": "AdamW; 5% warmup then cosine; gradient norm clip 1; base BF16, LoRA FP32",
        "attention": "same non-cuDNN SDPA context for forward AND backward recomputation",
        "heldout": "512 pilot-evaluation + 3729 reserved IDs unused for training, selection, compatibility",
        "claim_limit": "medical QA adaptation, not isolated medical-fact learning; TEST resplit development only",
        "code": {str(p.relative_to(ROOT)): sha_file(p) for p in (Path(__file__),
                 ROOT / "medrag/training/shared_medical_lora.py", ROOT / "medrag/training/graph_input_modules.py",
                 ROOT / "medrag/training/graph_input_progress.py", ROOT / "scripts/train_rag2_text_kg_input_pilot.py")},
        "environment": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}}
    del graph
    return records, contract, graph_contract, feature_root


def export_adapter(model, output, name):
    target = output / name
    temp = output / (name + ".incomplete")
    temp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp, safe_serialization=True, save_embedding_layers=False)
    # Export dirs are versioned by epoch/step. An interrupted export is reused
    # only if its serialized tensors match the current in-memory LoRA exactly.
    from safetensors.torch import load_file
    if target.exists():
        saved = load_file(str(target / "adapter_model.safetensors"))
        current = adapter_state(model)
        import torch
        if set(saved) != set(current) or any(not torch.equal(saved[k], current[k]) for k in saved):
            raise RuntimeError(f"Existing epoch export differs: {target}")
    else:
        os.replace(temp, target)
    return target


def load_graph_prompts(args, graph_contract, feature_root, rows, device):
    import torch
    texts = json.loads((feature_root / "texts.json").read_text())
    lookup = {t: i for i, t in enumerate(texts)}
    needed = {lookup[t] for r in rows for t in r["graph"]["node_texts"] + r["graph"]["edge_texts"]}
    vectors = {}
    for shard in sorted({i // 512 for i in needed}):
        item = load_torch(feature_root / f"{shard:05d}.pt")
        if item["text_sha256"] != digest(texts[shard*512:shard*512+512]):
            raise RuntimeError("Frozen feature shard hash mismatch")
        for i in range(shard*512, min(len(texts), shard*512+512)):
            if i in needed:
                vectors[i] = item["values"][i % 512].clone()
    saved = load_torch(args.graph_root / "best.pt")
    dim = saved["adapter"]["layers.0.query.weight"].shape[1]
    llm_dim = saved["adapter"]["projector.2.weight"].shape[0]
    a = graph_contract["architecture"]
    graph = GraphPrompt(dim, llm_dim, a["graph_hidden"], a["graph_layers"], a["graph_heads"]).to(device)
    graph.load_state_dict(saved["adapter"])
    graph.requires_grad_(False); graph.eval()
    prompts = {}
    with torch.no_grad():
        for row in rows:
            g = row["graph"]
            x = torch.stack([vectors[lookup[t]] for t in g["node_texts"]]).to(device) if g["node_texts"] else torch.empty(0, dim, device=device)
            e = torch.stack([vectors[lookup[t]] for t in g["edge_texts"]]).to(device) if g["edge_texts"] else torch.empty(0, dim, device=device)
            prompts[row["sample_id"]] = graph(x, torch.tensor(g["edge_index"],device=device), e).detach()
    del graph, saved, vectors
    return prompts


def validate(model, rows, pad_id, micro_batch, progress, prompts=None):
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), micro_batch):
            batch = rows[start:start+micro_batch]
            size = len(batch)
            while True:
                try:
                    values = []
                    for i in range(0, len(batch), size):
                        piece = batch[i:i+size]
                        z = [prompts[r["sample_id"]] for r in piece] if prompts is not None else None
                        values.extend(qa_losses(model,piece,pad_id,graph_prompts=z).cpu().tolist())
                    break
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    if size == 1: raise
                    size = max(1, size//2)
                    progress.log(f"[validation OOM] retry micro={size}; restore next batch")
            if any(not math.isfinite(x) for x in values): raise FloatingPointError("Non-finite validation CE")
            outputs.extend({"sample_id":r["sample_id"],"answer_ce":v} for r,v in zip(batch,values,strict=True))
            progress.update(len(batch))
    return sum(x["answer_ce"] for x in outputs)/len(outputs), outputs


@training_attention_backend()
def train(args, records, contract_hash, graph_contract, feature_root, progress):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output = args.output_root
    train_rows = [r for r in records if r["split"] == "train"]
    val_rows = [r for r in records if r["split"] == "validation"]
    progress.stage(1, 1, unit="model")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda:0")
    model = adapt_llama(base,args.rank,args.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=args.learning_rate,weight_decay=args.weight_decay)
    steps = math.ceil(len(train_rows)/args.effective_batch_size)*args.epochs
    warmup = max(1,int(.05*steps))
    def rate(step):
        return ((step+1)/warmup if step<warmup else
                .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,steps-warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    state = {"epoch":0,"cursor":0,"step":0,"loss_sum":0.,"examples":0,"history":[],"best_loss":float("inf"),"best_adapter":None}
    last_path = output / "last.pt"
    if last_path.exists():
        saved = load_torch(last_path,trusted=True)
        if saved["contract_sha256"] != contract_hash: raise RuntimeError("Resume contract mismatch")
        restore_adapter(model,saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        state=saved["state"]; restore_rng(saved["rng"])
        del saved
        progress.log(f"[resume] epoch={state['epoch']+1}; cursor={state['cursor']}; optimizer step={state['step']}")
    def save():
        begin=time.monotonic()
        torch_save(last_path,{"contract_sha256":contract_hash,"adapter":adapter_state(model),
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"state":state,"rng":rng_state()})
        atomic_json(output / "history.json",state["history"])
        progress.weights[-1]=max(.001,time.monotonic()-begin)
    counts={"trainable_lora":sum(p.numel() for p in params),"trainable_soft_prompt":0,
            "trainable_graph_parameters":0,"trainable_non_lora":0,"shared_llm_for_conditions":[3,5,6]}
    atomic_json(output / "parameter_counts.json",counts)
    progress.complete(f"Llama LoRA trainable={counts['trainable_lora']:,}; shared model, not a soft prompt")
    for epoch in range(state["epoch"]):
        for j,n in ((2+2*epoch,len(train_rows)),(3+2*epoch,len(val_rows))):
            progress.stage(j,n,initial=n,unit="q"); progress.complete("cached epoch")
    if not last_path.exists(): save()
    for epoch in range(state["epoch"],args.epochs):
        order=list(range(len(train_rows))); random.Random(args.seed+epoch).shuffle(order)
        progress.stage(2+2*epoch,len(train_rows),initial=state["cursor"],unit="q",kind="train")
        model.train()
        for start in range(state["cursor"],len(train_rows),args.effective_batch_size):
            batch=[train_rows[i] for i in order[start:start+args.effective_batch_size]]
            micro=min(args.micro_batch_size,len(batch)); before=rng_state()
            while True:
                optimizer.zero_grad(set_to_none=True); total=0.
                try:
                    for i in range(0,len(batch),micro):
                        loss=qa_losses(model,batch[i:i+micro],tokenizer.pad_token_id)
                        if not torch.isfinite(loss).all(): raise FloatingPointError("Non-finite training CE")
                        (loss.sum()/len(batch)).backward(); total+=loss.detach().sum().item(); del loss
                    break
                except torch.cuda.OutOfMemoryError:
                    if "loss" in locals(): del loss
                    optimizer.zero_grad(set_to_none=True); restore_rng(before); gc.collect(); torch.cuda.empty_cache()
                    if micro==1: raise
                    micro=max(1,micro//2)
                    progress.log(f"[OOM] retry optimizer batch micro={micro}; restore configured micro next step")
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("Missing or non-finite LoRA gradients")
            torch.nn.utils.clip_grad_norm_(params,1.)
            optimizer.step(); scheduler.step()
            state.update(epoch=epoch,cursor=start+len(batch),step=state["step"]+1,
                         loss_sum=state["loss_sum"]+total,examples=state["examples"]+len(batch))
            smoke_stop=args.smoke_steps and state["step"]>=args.smoke_steps
            if state["step"]%args.checkpoint_steps==0 or state["cursor"]==len(train_rows) or smoke_stop: save()
            progress.update(len(batch))
            if smoke_stop:
                path=export_adapter(model,output,f"smoke_step_{state['step']:04d}")
                progress.log(f"[smoke stop] {state['step']} optimizer updates; export={path}; no full training/evaluation")
                return model, tokenizer, state
        progress.complete(f"epoch={epoch+1}; mean train CE={state['loss_sum']/state['examples']:.6f}; checkpoint={last_path}")
        progress.stage(3+2*epoch,len(val_rows),unit="q",kind="validation")
        value,items=validate(model,val_rows,tokenizer.pad_token_id,args.micro_batch_size,progress)
        path=export_adapter(model,output,f"epoch_{epoch+1:02d}")
        atomic_jsonl(output / f"validation_loss_epoch{epoch+1}.jsonl",items)
        if value<state["best_loss"]:
            state.update(best_loss=value,best_adapter=str(path))
            atomic_json(output / "best.json",{"epoch":epoch+1,"validation_answer_ce":value,"adapter":str(path)})
        state["history"].append({"epoch":epoch+1,"train_n":len(train_rows),"validation_n":len(val_rows),
            "train_answer_ce":state["loss_sum"]/state["examples"],"validation_answer_ce":value,
            "steps":state["step"],"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated()})
        state.update(epoch=epoch+1,cursor=0,loss_sum=0.,examples=0); save()
        progress.complete(f"validation CE={value:.6f}; best={state['best_loss']:.6f}; checkpoint selection uses TEXT only")
    return model, tokenizer, state


@training_attention_backend()
def graph_diagnostic(args, model, tokenizer, state, records, graph_contract, feature_root, progress):
    import torch
    from safetensors.torch import load_file
    path=(args.output_root/f"smoke_step_{state['step']:04d}" if args.smoke_steps else Path(state["best_adapter"]))
    restore_adapter(model,load_file(str(path/"adapter_model.safetensors")))
    val=[r for r in records if r["split"]=="validation"]
    if args.smoke_steps: val=val[:2]
    progress.stage(2+2*args.epochs,1,unit="cache")
    prompts=load_graph_prompts(args,graph_contract,feature_root,val,torch.device("cuda:0"))
    progress.complete(f"frozen graph vectors prepared for validation only: {len(val)} questions")
    progress.stage(3+2*args.epochs,len(val),unit="q",kind="graph_validation")
    value,items=validate(model,val,tokenizer.pad_token_id,args.micro_batch_size,progress,prompts)
    baseline=json.loads((args.graph_root/"COMPLETE.json").read_text())["best_validation_answer_ce"]
    result={"n":len(val),"trained_llm_graph_answer_ce":value,"selection_criterion":False,
            "graph_updated":False,"finite_forward_pass":True,"accuracy_evaluation":False,
            "interpretation":"interface diagnostic only; no evidence of graph usefulness or QA accuracy"}
    if not args.smoke_steps:
        result.update(base_llm_graph_answer_ce=baseline,delta_answer_ce=value-baseline)
    prefix="smoke_" if args.smoke_steps else ""
    atomic_json(args.output_root/f"{prefix}graph_compatibility.json",result)
    atomic_jsonl(args.output_root/f"{prefix}graph_validation_loss.jsonl",items)
    progress.complete(f"graph validation CE={value:.6f}; graph adapter frozen; no checkpoint selection")
    return result


def main():
    args=arguments()
    names=["verify fixed data","load LoRA Llama"]
    for e in range(1,args.epochs+1): names += [f"train epoch {e}/{args.epochs} mixed",f"validation epoch {e}/{args.epochs} mixed"]
    names += ["load graph cache","graph validation","save completion"]
    progress=LoggedProgress(names,args.output_root)
    try:
        from transformers.utils import logging
        logging.set_verbosity_error(); logging.disable_progress_bar()
        progress.log("[plan] verify cached 2048 train/256 validation -> shared Llama LoRA -> TEXT validation per epoch -> frozen graph validation -> save; NO six-condition evaluation")
        progress.log("[ETA] reuse data/graph caches; optimizer checkpoints resume; uncached LoRA rates calibrate independently of old soft-prompt rates")
        progress.stage(0,1)
        rows,contract,graph_contract,feature_root=preflight(args)
        with lock_file(args.output_root/".run.lock"):
            contract_hash=ensure_contract(args.output_root/"manifest.json",contract)
            progress.complete(f"counts={contract['training_counts']}; model/graph/data verified; held-out unused")
            if args.preflight_only:
                progress.log("[preflight passed] CPU only; no model weights loaded on GPU, no training, no held-out evaluation")
                return
            if (args.output_root/"COMPLETE.json").exists():
                done=json.loads((args.output_root/"COMPLETE.json").read_text())
                if done["contract_sha256"]!=contract_hash: raise RuntimeError("Completion contract mismatch")
                progress.log(f"[already complete] shared LoRA={done['trained_llm_adapter']}; no evaluation started")
                return
            import torch
            if os.environ.get("CUDA_VISIBLE_DEVICES")!="1" or not torch.cuda.is_available():
                raise RuntimeError("Run on GPU 1 using the supplied wrapper; CPU preflight remains available")
            if not (args.output_root/"provenance.json").exists():
                atomic_json(args.output_root/"provenance.json",{"git_commit":subprocess.check_output(
                    ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"gpu":1,"contract_sha256":contract_hash})
            with lock_file(ROOT/"results/.shared_medical_lora_gpu1.lock"):
                model,tokenizer,state=train(args,rows,contract_hash,graph_contract,feature_root,progress)
                graph_diagnostic(args,model,tokenizer,state,rows,graph_contract,feature_root,progress)
                if sha_file(args.graph_root/"best.pt")!=contract["graph_checkpoint"]["sha256"]:
                    raise RuntimeError("Frozen graph checkpoint changed during training")
                if args.smoke_steps:
                    atomic_json(args.output_root/"SMOKE_COMPLETE.json",{"optimizer_steps":state["step"],
                        "training_questions":state["cursor"],"validation_questions":2,"six_condition_evaluation":False})
                    progress.log("[smoke complete] backward, optimizer, resume checkpoint, PEFT export/reload and graph input passed")
                    return
                progress.stage(len(names)-1,1)
                ready={"contract_sha256":contract_hash,"base_llm":str(args.model),
                    "trained_llm_adapter":state["best_adapter"],"same_trained_llm_for":["TEXT","GRAPH","No-RAG"],
                    "shared_graph_adapter":str(args.graph_root/"best.pt"),"graph_checkpoint_sha256":contract["graph_checkpoint"]["sha256"],
                    "data_root":str(args.data_root),"epochs":state["epoch"],"optimizer_steps":state["step"],
                    "best_text_validation_ce":state["best_loss"],"six_condition_evaluation_executed":False,
                    "pilot_evaluation_questions_used":0}
                atomic_json(args.output_root/"SIX_CONDITION_INPUTS.json",ready)
                atomic_json(args.output_root/"COMPLETE.json",ready)
                progress.update(); progress.complete(f"shared trained Llama adapter={state['best_adapter']}; ready for later six-condition evaluation")
                progress.log("[finished] training and validation diagnostics only; held-out QA answers were not generated")
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; {type(exc).__name__}: {exc}; durable checkpoint={args.output_root/'last.pt'}; rerun identical command to resume")
        raise
    finally:
        progress.close()


if __name__=="__main__":
    main()

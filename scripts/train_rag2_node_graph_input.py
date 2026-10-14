#!/usr/bin/env python3
"""Train node-wise KG tokens against one EXACT, frozen Llama checkpoint.

Reuse the immutable TEST6545 development repartition, Top-8 edges, source
propositions, prompts, answer targets and SBERT features. No new retrieval,
held-out examples, answer generation, or final evaluation is performed.
--preflight-only checks the real cache/contract/context path on CPU.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from medrag.training.graph_input_data import (atomic_json, atomic_jsonl, digest,
    ensure_contract, graph_from_edges, identity, jsonl, sha_file)
from medrag.training.graph_input_progress import Progress
from medrag.training.graph_input_modules import answer_losses, training_attention_backend
from medrag.training.node_graph_input import NodeGraphPrompt, node_prompt_batch
from train_rag2_text_kg_input_pilot import (bundle_identity, load_torch, lock_file,
    restore_rng, rng_state, set_seed, torch_save)

VERSION = "node_graph_input_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/pilots/rag2_text_kg_input_test_resplit_v1")
    p.add_argument("--feature-root", type=Path, help="Existing feature cache; auto-resolve ONLY if exactly one matches")
    p.add_argument("--target-llm", choices=("shared", "base"), default="shared")
    p.add_argument("--shared-root", type=Path, default=ROOT / "results/rag2_shared_medical_lora_v1")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--message-direction", choices=("both", "incoming"), default="both",
                   help="incoming is a controlled readout-only ablation, not the default method")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--effective-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--graph-hidden", type=int, default=1024)
    p.add_argument("--graph-layers", type=int, default=4)
    p.add_argument("--graph-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-steps", type=int, default=25)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="Separate-output bounded training smoke; not a quality or overfit test")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.output_root is None:
        suffix = "_smoke" if args.smoke_steps else ""
        args.output_root = ROOT / f"results/rag2_node_graph_input_v1/{args.target_llm}_{args.message_direction}{suffix}"
    return args


def checked_manifest(path):
    value = json.loads(path.read_text())
    if digest(value["contract"]) != value["contract_sha256"]:
        raise RuntimeError(f"Manifest hash mismatch: {path}")
    return value


def validate_rows(rows, assignments, max_context):
    """Reject changed graph identity, accidental held-out use or truncation."""
    assigned = {a["sample_id"]: a["split"] for a in assignments}
    if len(assigned) != len(assignments):
        raise RuntimeError("Duplicated split ID")
    expected = {k: v for k, v in assigned.items() if v in ("train", "validation")}
    actual = {r["sample_id"]: r["split"] for r in rows}
    if len(actual) != len(rows) or actual != expected:
        raise RuntimeError("Missing/duplicate/held-out training examples")
    groups = {}
    for a in assignments:
        groups.setdefault(a["duplicate_group"], set()).add(a["split"])
    if any(len(s) != 1 for s in groups.values()):
        raise RuntimeError("Duplicate question group crosses splits")
    sizes, lengths, counts = [], [], Counter()
    for row in rows:
        g = row["graph"]
        if graph_from_edges(row["edges"]) != g:
            raise RuntimeError(f"Graph no longer matches frozen edges: {row['sample_id']}")
        if len(row["document_ids"]) != 8 or len(set(row["document_ids"])) != 8:
            raise RuntimeError("Expected fixed, distinct Top-8 document IDs")
        if any(e["document_id"] not in row["document_ids"] for e in row["edges"]):
            raise RuntimeError("Graph contains evidence outside Top-8")
        if set(e["proposition_id"] for e in row["edges"]) != set(p["proposition_id"] for p in row["propositions"]):
            raise RuntimeError("Paired source proposition scope differs")
        for key in ("graph_prompt_ids", "text_prompt_ids", "target_ids"):
            if not row[key] or not all(type(x) is int and x >= 0 for x in row[key]):
                raise RuntimeError(f"Invalid cached token IDs: {row['sample_id']}")
        n = len(g["node_ids"])
        length = len(row["graph_prompt_ids"]) + n + len(row["target_ids"])
        if length > max_context:
            raise RuntimeError(f"Node-token context overflow: {row['sample_id']} {length}>{max_context}; no silent node/edge truncation")
        sizes.append(n); lengths.append(length); counts[row["split"]] += 1
    if set(counts) != {"train", "validation"}:
        raise RuntimeError("Both train and validation must be nonempty")
    return {"counts": dict(counts), "empty_graphs": sizes.count(0),
            "node_tokens_mean": sum(sizes) / len(sizes), "node_tokens_max": max(sizes),
            "max_graph_context_with_target": max(lengths)}


def preflight(args):
    if min(args.epochs, args.micro_batch_size, args.effective_batch_size, args.checkpoint_steps,
           args.graph_hidden, args.graph_layers, args.graph_heads, args.cpu_threads) < 1:
        raise ValueError("Sizes must be positive")
    if (args.micro_batch_size > args.effective_batch_size or args.smoke_steps < 0 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.weight_decay) or args.weight_decay < 0 or
            args.graph_hidden % args.graph_heads):
        raise ValueError("Invalid optimizer/architecture/batch configuration")
    output = args.output_root.resolve()
    for protected in (args.model, args.data_root, args.shared_root):
        protected = protected.resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"Output overlaps protected input: {protected}")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("Need 8 GiB free for atomic optimizer checkpoints")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config["model_type"] != "llama":
        raise ValueError("This input adapter targets the existing Llama checkpoint")
    model_id = bundle_identity(args.model)
    dm = checked_manifest(args.data_root / "manifest.json")
    dc = json.loads((args.data_root / "COMPLETE.json").read_text())
    if dc["contract_sha256"] != dm["contract_sha256"]:
        raise RuntimeError("Prepared-data completion mismatch")
    for name, expected in dc["file_sha256"].items():
        if sha_file(args.data_root / name) != expected:
            raise RuntimeError(f"Frozen data modified: {name}")
    for tok in dm["contract"]["tokenizer"]:
        if identity(args.model / Path(tok["path"]).name, full_hash=True) != tok:
            raise RuntimeError("Tokenizer differs from frozen token IDs")
    max_context = dm["contract"]["max_context"]
    if max_context > model_config["max_position_embeddings"]:
        raise RuntimeError("Context contract exceeds Llama capacity")
    rows = list(jsonl(args.data_root / "train_validation.jsonl"))
    assignments = list(jsonl(args.data_root / "split_assignments.jsonl"))
    stats = validate_rows(rows, assignments, max_context)
    if stats["empty_graphs"] == len(rows):
        raise RuntimeError("No nonempty graph to train")
    roots = [args.feature_root] if args.feature_root else sorted((args.data_root / "features").glob("*"))
    candidates = []
    for root in roots:
        if not (root / "manifest.json").is_file():
            continue
        fm = checked_manifest(root / "manifest.json")
        if fm["contract"]["data_contract"] == dm["contract_sha256"]:
            candidates.append((root, fm))
    if len(candidates) != 1:
        raise RuntimeError("Specify exactly one matching existing --feature-root; no implicit embedding generation")
    feature_root, fm = candidates[0]
    if output == feature_root.resolve() or feature_root.resolve() in output.parents or output in feature_root.resolve().parents:
        raise ValueError("Output overlaps feature cache")
    texts = json.loads((feature_root / "texts.json").read_text())
    fc = fm["contract"]
    expected_texts = sorted({t for r in rows for t in r["graph"]["node_texts"] + r["graph"]["edge_texts"]})
    if texts != expected_texts or digest(texts) != fc["texts_sha256"] or len(texts) != fc["count"]:
        raise RuntimeError("Frozen graph feature vocabulary mismatch")
    feature_files = [identity(feature_root / f"{i//512:05d}.pt") for i in range(0, len(texts), 512)]
    limits = json.loads((feature_root / "encoder_limits.json").read_text())
    input_dim = limits["embedding_dim"]
    lora, lora_identity, shared_contract = None, None, None
    if args.target_llm == "shared":
        sm = checked_manifest(args.shared_root / "manifest.json")
        sc = json.loads((args.shared_root / "COMPLETE.json").read_text())
        sb = json.loads((args.shared_root / "best.json").read_text())
        if (sc["contract_sha256"] != sm["contract_sha256"] or
                sm["contract"]["data_contract"] != dm["contract_sha256"] or
                sm["contract"]["model"] != model_id or
                sc["trained_llm_adapter"] != sb["adapter"] or
                sc["best_text_validation_ce"] != sb["validation_answer_ce"]):
            raise RuntimeError("Shared Llama selection/model/data contract mismatch")
        lora = Path(sb["adapter"]).resolve()
        if args.shared_root.resolve() not in lora.parents:
            raise RuntimeError("Selected LoRA escapes shared-root")
        lora_identity = [identity(lora / n, full_hash=True) for n in
                         ("adapter_config.json", "adapter_model.safetensors")]
        shared_contract = sm["contract_sha256"]
    files = (Path(__file__), ROOT / "medrag/training/node_graph_input.py",
             ROOT / "medrag/training/graph_input_modules.py", ROOT / "medrag/training/graph_input_data.py",
             ROOT / "medrag/training/graph_input_progress.py", ROOT / "scripts/train_rag2_text_kg_input_pilot.py")
    contract = {
        "version": VERSION, "purpose": "bounded_training_smoke" if args.smoke_steps else "node_graph_training_only",
        "hypothesis": "Retaining entity-aligned node tokens reduces graph-level pooling loss; QA benefit is unproven",
        "changed_variables": ["global mean readout -> one token per canonical node",
                              "incoming-only -> role-aware two-way messages" if args.message_direction == "both" else "message direction unchanged",
                              f"encoder aligned to frozen {args.target_llm} Llama"],
        "data_contract": dm["contract_sha256"], "data_files": dc["file_sha256"],
        "counts": stats, "split": "existing TEST6545 development repartition; no new split or sample selection",
        "evidence": "same cached Top-8 edges and node identities; cross-document canonical identity retained; no external neighbors",
        "prompt": dm["contract"]["system_prompt"], "target": dm["contract"]["target"],
        "input": "cached graph_prompt_ids; node soft tokens after BOS; NO proposition text",
        "max_context": max_context, "truncation": "none; fail before GPU on overflow",
        "node_order": "frozen graph.node_ids order; one output row per ID; no permutation or node selection",
        "node_cache_scope": "graph identity + node IDs + features + encoder checkpoint + frozen target LLM; NOT entity ID alone",
        "empty_graph": "zero node tokens; retain question, no fabricated entity",
        "model": model_id, "target_llm": args.target_llm, "frozen_lora": lora_identity,
        "shared_lora_contract": shared_contract, "features": fc, "feature_files": feature_files,
        "architecture": {"class": "NodeGraphPrompt", "input_dim": input_dim,
            "llm_dim": model_config["hidden_size"], "hidden_dim": args.graph_hidden,
            "layers": args.graph_layers, "heads": args.graph_heads,
            "bidirectional": args.message_direction == "both", "global_pooling": False,
            "role_semantics": "object receiver vs subject receiver of ORIGINAL edge; no inverse KG facts",
            "normalization_projector": "unchanged BatchNorm/ReLU; Linear-Sigmoid-Linear; no old graph checkpoint reuse"},
        "training": {k: getattr(args, k) for k in ("epochs", "micro_batch_size", "effective_batch_size",
            "learning_rate", "weight_decay", "seed", "checkpoint_steps", "smoke_steps")},
        "loss": "unchanged mean per-question teacher-forced answer CE; only GNN and projector optimized",
        "optimizer": "AdamW; 5% warmup/cosine; clip norm 1; LLM BF16, graph FP32",
        "selection": "minimum validation answer CE; no accuracy or held-out generation; not a quality pass certificate",
        "claim_limit": "TEST repartition pilot; multiple architecture changes, not isolated topology effect; MCQ QA supervision",
        "heldout_policy": "512 pilot-evaluation and reserve questions/labels never loaded; split IDs only",
        "code": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
        "environment": {**{n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft")},
                        "python": platform.python_version()},
    }
    return {"rows": rows, "contract": contract, "feature_root": feature_root,
            "texts": texts, "lora": lora, "input_dim": input_dim, "max_context": max_context}


def load_features(context, progress):
    """Fully validate existing feature shards on CPU before any LLM loading."""
    import torch
    texts, root = context["texts"], context["feature_root"]
    progress.stage(1, len(texts), unit="text")
    features = torch.empty((len(texts), context["input_dim"]), dtype=torch.float32)
    for i in range(0, len(texts), 512):
        item = load_torch(root / f"{i//512:05d}.pt")
        values = item["values"]
        n = min(512, len(texts)-i)
        if (item["text_sha256"] != digest(texts[i:i+n]) or values.shape != (n, context["input_dim"]) or
                values.dtype != torch.float32 or not torch.isfinite(values).all()):
            raise RuntimeError(f"Invalid cached feature shard: {root}/{i//512:05d}.pt")
        features[i:i+n].copy_(values)
        progress.update(n)
    progress.complete(f"read-only attributes={len(texts):,}; cache={root}; no new embeddings")
    return features, {t: i for i, t in enumerate(texts)}


def validate_losses(llm, emb, adapter, rows, features, lookup, pad_id, micro_batch, progress):
    import torch
    results = []
    adapter.eval(); llm.model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), micro_batch):
            batch = rows[start:start+micro_batch]
            micro = len(batch)
            while True:
                try:
                    values = []
                    for i in range(0, len(batch), micro):
                        piece = batch[i:i+micro]
                        z = node_prompt_batch(adapter, piece, features, lookup, emb.weight.device)
                        values.extend(answer_losses(llm, emb, piece, z, "graph", pad_id).cpu().tolist())
                        del z
                    break
                except torch.cuda.OutOfMemoryError:
                    if "z" in locals(): del z
                    gc.collect(); torch.cuda.empty_cache()
                    if micro == 1: raise
                    micro = max(1, micro // 2)
                    progress.log(f"[validation OOM] micro={micro}; restore configured size next batch")
            if any(not math.isfinite(v) for v in values):
                raise FloatingPointError("Non-finite validation CE")
            results.extend({"sample_id": r["sample_id"], "answer_ce": v,
                            "node_tokens": len(r["graph"]["node_ids"])}
                           for r, v in zip(batch, values, strict=True))
            progress.update(len(batch))
    return sum(r["answer_ce"] for r in results) / len(results), results


@training_attention_backend()
def train(args, context, features, lookup, run_hash, progress):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
    output = args.output_root
    progress.stage(2, 1, unit="model")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
              dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    if context["lora"]:
        from peft import PeftModel
        wrapper = PeftModel.from_pretrained(llm, context["lora"], is_trainable=False, local_files_only=True)
        llm = wrapper.get_base_model()  # installed LoRA stays active; no BF16 merge/rounding
    llm.requires_grad_(False); llm.eval(); llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(llm.config, "attention_dropout", 0.) != 0.:
        raise RuntimeError("Frozen target must have zero attention dropout for comparable training/validation")
    llm.model.train()
    for module in llm.modules():
        if isinstance(module, torch.nn.Dropout): module.eval()
    emb = llm.get_input_embeddings()
    adapter = NodeGraphPrompt(context["input_dim"], llm.config.hidden_size, args.graph_hidden,
                 args.graph_layers, args.graph_heads, args.message_direction == "both").to("cuda:0")
    train_rows = [r for r in context["rows"] if r["split"] == "train"]
    val_rows = [r for r in context["rows"] if r["split"] == "validation"]
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    max_steps = math.ceil(len(train_rows)/args.effective_batch_size) * args.epochs
    warmup = max(1, int(.05*max_steps))
    def rate(step):
        return ((step+1)/warmup if step < warmup else
                .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, max_steps-warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    state = {"epoch": 0, "cursor": 0, "step": 0, "best_validation_loss": float("inf"),
             "history": [], "epoch_loss_sum": 0., "epoch_examples": 0}
    last_path = output / "last.pt"
    if last_path.exists():
        saved = load_torch(last_path, trusted=True)
        if saved["contract_sha256"] != run_hash:
            raise RuntimeError("Node checkpoint belongs to a different contract")
        adapter.load_state_dict(saved["adapter"]); opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]; restore_rng(saved["rng"])
        del saved
        progress.log(f"[resume] epoch={state['epoch']+1}, durable={state['cursor']}/{len(train_rows)}, step={state['step']}")
    def save():
        begin = time.monotonic()
        torch_save(last_path, {"contract_sha256": run_hash, "adapter": adapter.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "state": state, "rng": rng_state()})
        atomic_json(output / "history.json", state["history"])
        progress.weights[-1] = max(.001, time.monotonic()-begin)
    if any(p.requires_grad for p in llm.parameters()):
        raise RuntimeError("Target Llama/LoRA must remain frozen")
    atomic_json(output / "parameter_counts.json", {"trainable_graph": sum(p.numel() for p in adapter.parameters()),
               "trainable_llm": 0, "target_llm": args.target_llm, "node_tokens": "one per node; no graph-level pooling"})
    progress.complete(f"target={args.target_llm}; Llama frozen; node encoder initialized; checkpoints={output}")
    for epoch in range(state["epoch"]):
        for j, size in ((3+2*epoch, len(train_rows)), (4+2*epoch, len(val_rows))):
            progress.stage(j, size, initial=size, unit="q"); progress.complete("durable epoch already completed")
    if not last_path.exists(): save()
    for epoch in range(state["epoch"], args.epochs):
        order = list(range(len(train_rows))); random.Random(args.seed+epoch).shuffle(order)
        progress.stage(3+2*epoch, len(train_rows), initial=state["cursor"], unit="q", kind="train")
        adapter.train(); llm.model.train()
        for module in llm.modules():
            if isinstance(module, torch.nn.Dropout): module.eval()
        for start in range(state["cursor"], len(train_rows), args.effective_batch_size):
            batch = [train_rows[i] for i in order[start:start+args.effective_batch_size]]
            micro = min(args.micro_batch_size, len(batch))
            before_rng = rng_state()
            buffers = {k: v.detach().clone() for k, v in adapter.named_buffers()}
            while True:
                opt.zero_grad(set_to_none=True); total_loss = 0.
                try:
                    for i in range(0, len(batch), micro):
                        piece = batch[i:i+micro]
                        z = node_prompt_batch(adapter, piece, features, lookup, "cuda:0")
                        losses = answer_losses(llm, emb, piece, z, "graph", tokenizer.eos_token_id)
                        if not torch.isfinite(losses).all(): raise FloatingPointError("Non-finite training CE")
                        # A microbatch of empty KGs has no adapter gradient, but
                        # its loss still belongs in the per-question mean.
                        if losses.requires_grad: (losses.sum()/len(batch)).backward()
                        total_loss += losses.detach().sum().item()
                        del losses, z
                    break
                except torch.cuda.OutOfMemoryError:
                    if "losses" in locals(): del losses
                    if "z" in locals(): del z
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for name, buf in adapter.named_buffers(): buf.copy_(buffers[name])
                    restore_rng(before_rng); gc.collect(); torch.cuda.empty_cache()
                    if micro == 1: raise
                    micro = max(1, micro//2)
                    progress.log(f"[OOM retry] epoch={epoch+1}, q={start}, micro={micro}; restore {args.micro_batch_size} next step")
            grads = [p.grad for p in adapter.parameters() if p.grad is not None]
            has_nodes = any(r["graph"]["node_ids"] for r in batch)
            if (has_nodes and not grads) or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Missing/non-finite node encoder gradients")
            if grads:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
                opt.step()
            scheduler.step()
            state.update(epoch=epoch, cursor=start+len(batch), step=state["step"]+1,
                         epoch_loss_sum=state["epoch_loss_sum"]+total_loss,
                         epoch_examples=state["epoch_examples"]+len(batch))
            if state["step"] % args.checkpoint_steps == 0 or state["cursor"] == len(train_rows): save()
            progress.update(len(batch))
            if args.smoke_steps and state["step"] >= args.smoke_steps:
                save()
                atomic_json(output / "SMOKE_COMPLETE.json", {"contract_sha256": run_hash, "steps": state["step"],
                    "final_evaluation_executed": False, "quality_validated": False})
                progress.log(f"[smoke complete] {state['step']} optimizer updates; {last_path}; no quality claim")
                return
        progress.complete(f"epoch={epoch+1}; train CE={state['epoch_loss_sum']/state['epoch_examples']:.6f}; checkpoint={last_path}")
        progress.stage(4+2*epoch, len(val_rows), unit="q", kind="validation")
        val_loss, per_question = validate_losses(llm, emb, adapter, val_rows, features, lookup,
                                                tokenizer.eos_token_id, args.micro_batch_size, progress)
        if val_loss < state["best_validation_loss"]:
            state["best_validation_loss"] = val_loss
            torch_save(output / "best.pt", {"contract_sha256": run_hash, "adapter": adapter.state_dict(),
                "architecture": context["contract"]["architecture"], "epoch": epoch+1, "validation_loss": val_loss})
        atomic_jsonl(output / f"validation_loss_epoch{epoch+1}.jsonl", per_question)
        state["history"].append({"epoch": epoch+1, "train_answer_ce": state["epoch_loss_sum"]/state["epoch_examples"],
            "validation_answer_ce": val_loss, "train_n": len(train_rows), "validation_n": len(val_rows),
            "steps": state["step"], "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        state.update(epoch=epoch+1, cursor=0, epoch_loss_sum=0., epoch_examples=0)
        save()
        progress.complete(f"validation CE={val_loss:.6f}; best={state['best_validation_loss']:.6f}; no final evaluation")
    progress.stage(len(progress.names)-1, 1)
    atomic_json(output / "COMPLETE.json", {"contract_sha256": run_hash, "epochs": args.epochs,
        "best_validation_answer_ce": state["best_validation_loss"], "best_checkpoint": str(output / "best.pt"),
        "best_checkpoint_sha256": sha_file(output / "best.pt"), "last_checkpoint": str(last_path),
        "target_llm": args.target_llm, "frozen_lora": str(context["lora"]) if context["lora"] else None,
        "final_evaluation_executed": False, "pilot_evaluation_questions_used": 0, "quality_validated": False})
    progress.update(); progress.complete(f"node-token training complete; {output}; held-out accuracy not measured")
    progress.log(f"[result] training only; selected checkpoint={output / 'best.pt'}")


def main(argv=None):
    args = arguments(argv)
    names = ["CPU contract + node budgets", "validate cached attributes", "load frozen Llama"]
    for i in range(1, args.epochs+1):
        names += [f"train epoch {i}/{args.epochs} mixed", f"validation epoch {i}/{args.epochs} mixed"]
    names += ["save completion"]
    if args.preflight_only: names = names[:2]
    progress = Progress(names)
    try:
        import torch
        torch.set_num_threads(max(1, args.cpu_threads))
        progress.stage(0, 1, unit="contract")
        progress.log(f"[plan] frozen 2,048 train / 256 validation -> cached features -> frozen {args.target_llm} Llama "
                     f"-> node encoder train/validation x{args.epochs} -> checkpoints; no held-out evaluation")
        progress.log("[ETA] node-token training has no measured compatible throughput; loading, train and validation "
                     "calibrate separately. Existing features reused; GPU 1 only. No fixed wall-clock promise.")
        context = preflight(args)
        progress.complete(f"node alignment/context checked: {json.dumps(context['contract']['counts'])}")
        with lock_file(args.output_root / ".training.lock"):
            run_hash = ensure_contract(args.output_root / "manifest.json", context["contract"])
            for name in ("COMPLETE.json", "SMOKE_COMPLETE.json"):
                p = args.output_root / name
                if p.exists():
                    saved = json.loads(p.read_text())
                    if saved["contract_sha256"] != run_hash: raise RuntimeError("Completion contract mismatch")
                    if name == "COMPLETE.json" and sha_file(Path(saved["best_checkpoint"])) != saved["best_checkpoint_sha256"]:
                        raise RuntimeError("Completed node checkpoint modified")
                    progress.log(f"[already complete] {p}; no GPU or held-out evaluation")
                    return
            if not (args.output_root / "provenance.json").exists():
                atomic_json(args.output_root / "provenance.json", {
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "contract_sha256": run_hash})
            features, lookup = load_features(context, progress)
            atomic_json(args.output_root / "PREFLIGHT.json", {"contract_sha256": run_hash,
                "statistics": context["contract"]["counts"], "feature_rows": len(features),
                "gpu_training_executed": False, "final_evaluation_executed": False})
            if args.preflight_only:
                progress.log("[preflight passed] frozen splits, node/edge identity, exact target LLM, ALL feature shards "
                             "and variable node-token budgets validated; no GPU/model loading/training")
                return
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable in this process; run the same wrapper on the GPU server")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
                raise RuntimeError("Restricted to physical GPU 1; use the wrapper")
            train(args, context, features, lookup, run_hash, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)}, item={progress.done}/{progress.total}, "
                     f"remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}. "
                     f"Last durable checkpoint={args.output_root / 'last.pt'}; identical command resumes a matching contract.")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

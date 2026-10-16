#!/usr/bin/env python3
"""Train dynamic KG attention, never run the 512-Q evaluation.

CPU preflight -> frozen MedCPT caches -> real 32-Q learnability gates -> fresh
2048/256 training. Local/graph arms share data, model, initialization and budget.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import run_rag2_kg_attention_pilot as base
from medrag.generation.dynamic_kg_attention import DynamicBias, source_layout
from medrag.generation.kg_structure_attention import answer_losses
from medrag.training.dynamic_kg_features import encode_bank, feature_plan, token_windows
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, jsonl, sha_file
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.text_graph_evaluation import generate_tokens, group
from train_rag2_text_kg_input_pilot import bundle_identity, load_torch, lock_file, restore_rng, rng_state, set_seed, torch_save

VERSION = "dynamic_kg_attention_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_dynamic_kg_attention_training_v1")
    p.add_argument("--variant", choices=("both", "graph", "local"), default="both")
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/pilots/rag2_text_kg_input_test_resplit_v1")
    p.add_argument("--shared-root", type=Path, default=ROOT / "results/rag2_shared_medical_lora_v1")
    p.add_argument("--query-encoder", type=Path, default=ROOT.parent / "models/MedCPT-Query-Encoder")
    p.add_argument("--article-encoder", type=Path, default=ROOT.parent / "models/MedCPT-Article-Encoder")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--effective-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-bias", type=float, default=2.)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--checkpoint-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--gate-only", action="store_true", help="Stop after the real 32-Q gates; normal command can resume")
    a = p.parse_args(argv)
    # Reuse frozen validators, not the static model/training contract.
    a.mode, a.smoke_steps = "train", 0
    a.reference_root = ROOT / "results/rag2_text_graph_six_conditions_v1"
    a.variants = ["local", "graph"] if a.variant == "both" else [a.variant]
    if min(a.hidden, a.rounds, a.embedding_batch_size) < 1: p.error("Positive architecture/batch values required")
    return a


def choose_tiny(rows, n=32, seed=42):
    buckets = defaultdict(list)
    for r in rows:
        if r["split"] == "train": buckets[group(r["dataset"])].append(r)
    total = sum(map(len, buckets.values()))
    quota = {k: int(n*len(v)/total) for k, v in buckets.items()}
    order = sorted(buckets, key=lambda k: (-(n*len(buckets[k])/total-quota[k]), k))
    for k in order[:n-sum(quota.values())]: quota[k] += 1
    result = []
    for k in sorted(buckets):
        items = sorted(buckets[k], key=lambda r: r["sample_id"])
        random.Random(f"{seed}:{k}").shuffle(items)
        result += items[:quota[k]]
    return result


def prepare(a, progress):
    from transformers import AutoTokenizer
    c = base.common_inputs(a)
    if shutil.disk_usage(a.output_root.parent).free < 8*1024**3:
        raise RuntimeError("Need at least 8 GiB free for features and atomic checkpoints")
    encoders = {"query": a.query_encoder, "article": a.article_encoder}
    for path in encoders.values():
        if path.resolve() == a.output_root.resolve() or path.resolve() in a.output_root.resolve().parents:
            raise ValueError("Output overlaps frozen encoder")
        cfg = json.loads((path / "config.json").read_text())
        if cfg["hidden_size"] != 768 or cfg["max_position_embeddings"] != 512:
            raise RuntimeError("Expected the local 768-dim / 512-token MedCPT encoder")
    rows = list(jsonl(a.data_root / "train_validation.jsonl"))
    base.validate_scope(rows, c["assignments"], ("train", "validation"), {"train": 2048, "validation": 256},
                        c["data"]["contract"]["max_context"])
    code = base.code_identity()
    for name in ("scripts/train_rag2_dynamic_kg_attention.py", "medrag/generation/dynamic_kg_attention.py",
                 "medrag/training/dynamic_kg_features.py"):
        code[name] = sha_file(ROOT / name)
    contract = {"version": VERSION, "data_contract": c["data"]["contract_sha256"],
        "input": identity(a.data_root / "train_validation.jsonl", full_hash=True),
        "model": c["model"], "frozen_lora": c["lora_id"], "shared_contract": c["shared_contract"],
        "encoders": {k: bundle_identity(v) for k, v in encoders.items()}, "code": code,
        "environment": base.versions(), "variants": a.variants,
        "architecture": {"layers": c["config"]["num_hidden_layers"], "heads": c["config"]["num_attention_heads"],
                         "hidden": a.hidden, "rounds": a.rounds, "max_bias": a.max_bias, "query_chunk": a.query_chunk},
        "training": {k: getattr(a, k) for k in ("epochs", "micro_batch_size", "effective_batch_size",
                                               "learning_rate", "checkpoint_steps", "seed", "embedding_batch_size")},
        "hypothesis": "Input-dependent coupling improves on static bias; neighborhood states add value beyond local semantics",
        "scope": "TEST repartition development pilot, MCQ answer supervision, not untouched final TEST",
        "heldout": "512 evaluation and 3729 reserved questions/labels never loaded",
        "controller_inputs": "question STEM; all frozen source propositions, canonical endpoints, surface relations; no options, gold or target tokens",
        "target_inputs": "unchanged cached Top-8 text_prompt_ids + teacher-forced target_ids",
        "frozen": "Llama + existing medical LoRA + both MedCPT encoders; only DynamicBias parameters train",
        "loss": "mean per-question CE on unchanged Answer: letter + end token; AdamW wd=0, clip=1",
        "selection": "lowest 256-Q validation CE, including initial zero-bias checkpoint; 3 epochs by default",
        "tiny_gate": {"n": 32, "max_updates": 100, "check_every": 10, "relative_ce_reduction": .20,
                      "selection": "proportional dataset groups, train only, no correctness filtering",
                      "reset_before_full": True, "must_pass_both_before_full": True},
        "encoder_windows": "single-text CLS; non-overlapping 510-content-token windows; all windows cached; learned pooling",
        "positions": "exact static evidence endpoint pairs; causal mask unchanged; same-kind duplicate link biases averaged",
        "message_direction": "both read directions with separate roles; never invert a clinical predicate",
        "local_control": "same message/update network and incident edge semantics, sender state replaced by receiver state; not graph-free",
        "initialization": "only final projection zero; all other parameters same seed in both arms",
        "claim_limits": "question-conditioned prefill only, not live rationale routing; no new retrieval or token positions; extra semantics and parameters vs static are confounds",
        "primary_future_evaluation": "Macro-3 delta over same-LoRA TEXT >=2pp and paired 95% CI lower>0; graph-local separately tests neighbor-state propagation"}
    run_hash = ensure_contract(a.output_root / "manifest.json", contract)
    if not (a.output_root / "provenance.json").exists():
        atomic_json(a.output_root / "provenance.json", {"contract_sha256": run_hash,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "code_sha256": code})
    progress.complete(f"unchanged train=2048 validation=256; no held-out questions; {a.output_root}")
    progress.stage(1, len(rows), unit="q", kind="mapping")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast: raise RuntimeError("Fast target tokenizer required")
    path, marker_path = a.output_root / "layouts.jsonl", a.output_root / "LAYOUTS_COMPLETE.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker["contract_sha256"] != run_hash or marker["sha256"] != sha_file(path):
            raise RuntimeError("Layout cache contract/hash mismatch")
        layouts = list(jsonl(path))
        progress.stage(1, len(rows), initial=len(rows), unit="q")
    else:
        layouts = []
        for row in rows:
            layouts.append(source_layout(tokenizer, row)); progress.update()
        atomic_jsonl(path, layouts)
        atomic_json(marker_path, {"contract_sha256": run_hash, "sha256": sha_file(path)})
    if [x["sample_id"] for x in layouts] != [r["sample_id"] for r in rows]: raise RuntimeError("Layout order mismatch")
    stats = Counter()
    for x in layouts: stats.update(x["statistics"])
    progress.complete(f"all dynamic pair positions match frozen static masks; {path}")
    progress.stage(2, 2, unit="encoder", kind="tokenization")
    plans = feature_plan(layouts)
    window_stats = {}
    for kind in ("query", "article"):
        tok = AutoTokenizer.from_pretrained(encoders[kind], local_files_only=True)
        counts = [len(token_windows(tok, t)) for t in plans[kind]]
        window_stats[kind] = {"texts": len(counts), "windows": sum(counts), "multi_window_texts": sum(x > 1 for x in counts),
                              "max_windows": max(counts)}
        progress.update()
    feature_root = a.output_root / "features"
    feature_hash = ensure_contract(feature_root / "manifest.json", {
        "parent_contract": run_hash, "plan_sha256": digest(plans), "encoders": contract["encoders"],
        "windows": contract["encoder_windows"], "statistics": window_stats})
    atomic_json(feature_root / "plan.json", plans)
    tiny = choose_tiny(rows, seed=a.seed)
    atomic_json(a.output_root / "PREFLIGHT.json", {"contract_sha256": run_hash, "gpu_executed": False,
        "counts": dict(Counter(r["split"] for r in rows)), "mapping": dict(stats), "encoder_windows": window_stats,
        "tiny_sample_ids": [r["sample_id"] for r in tiny], "tiny_datasets": dict(Counter(group(r["dataset"]) for r in tiny)),
        "heldout_loaded": False})
    progress.complete(f"all encoder tokens retained; {window_stats}; no CUDA allocation")
    c.update(rows=rows, layouts={x["sample_id"]: x for x in layouts}, tokenizer=tokenizer, contract=contract,
             run_hash=run_hash, plans=plans, encoders=encoders, feature_hash=feature_hash, feature_root=feature_root, tiny=tiny)
    return c


def new_controller(a, c, variant):
    set_seed(a.seed)
    controller = DynamicBias(**c["contract"]["architecture"], variant=variant).to("cuda:0")
    controller.bind_bank(c["bank"])
    return controller


def layouts_for(c, rows):
    return [c["layouts"][r["sample_id"]] for r in rows]


def calibrate_comparable_stages(progress):
    """Same target/batch/data operation across the two matched controller arms."""
    if not hasattr(progress, "weights"): return  # Minimal CPU test progress.
    operation = progress.names[progress.index].split()[0]
    weight = progress.weights[progress.index]
    if operation not in ("train", "validation") or weight is None: return
    # Never mix encoding, tiny overfit, forward-only and backward throughput.
    for i in range(progress.index+1, len(progress.names)):
        if progress.names[i].split()[0] == operation:
            progress.weights[i] = weight


def validation(model, controller, rows, c, progress=None):
    model.eval(); controller.eval()
    records = []
    with torch.no_grad():
        for row in rows:
            loss = answer_losses(model, controller, [row], layouts_for(c, [row]), c["tokenizer"].eos_token_id).item()
            if not math.isfinite(loss): raise FloatingPointError("Nonfinite validation CE")
            records.append({"sample_id": row["sample_id"], "answer_ce": loss})
            controller.clear()
            if progress is not None:
                progress.update(); calibrate_comparable_stages(progress)
    return sum(x["answer_ce"] for x in records)/len(records), records


def update(model, controller, optimizer, batch, a, c, progress):
    model.train(); controller.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.eval()
    micro, before = min(a.micro_batch_size, len(batch)), rng_state()
    while True:
        optimizer.zero_grad(set_to_none=True); total = 0.
        try:
            for start in range(0, len(batch), micro):
                part = batch[start:start+micro]
                losses = answer_losses(model, controller, part, layouts_for(c, part), c["tokenizer"].eos_token_id)
                if not torch.isfinite(losses).all(): raise FloatingPointError("Nonfinite training CE")
                if losses.requires_grad: (losses.sum()/len(batch)).backward()
                total += losses.detach().sum().item()
                controller.clear(); del losses
            break
        except torch.cuda.OutOfMemoryError:
            if "losses" in locals(): del losses
            controller.clear(); optimizer.zero_grad(set_to_none=True)
            restore_rng(before); gc.collect(); torch.cuda.empty_cache()
            if micro == 1: raise
            micro = max(1, micro//2)
            progress.log(f"[OOM] retry micro={micro}; configured size restored next optimizer step")
    grads = [p.grad for p in controller.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in grads): raise FloatingPointError("Nonfinite controller gradient")
    has_links = any(c["layouts"][r["sample_id"]]["pairs"] for r in batch)
    if has_links and not grads: raise RuntimeError("No controller gradient through frozen Llama")
    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.).item() if grads else 0.
    if grads: optimizer.step()
    if any(p.grad is not None for p in model.parameters()): raise RuntimeError("Llama/LoRA must remain frozen")
    return total, grad_norm


def zero_parity(model, controller, rows, c):
    """Zero-bias generation must match the ordinary SDPA backend, not itself."""
    model.eval(); controller.eval()
    prompts = [r["text_prompt_ids"] for r in rows[:16]]
    layouts = layouts_for(c, rows[:16])
    shifts = [max(map(len, prompts))-len(p) for p in prompts]
    pad = c["tokenizer"].eos_token_id
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        expected = generate_tokens(model, prompts, pad_id=pad, eos_ids=eos, max_new_tokens=16)
        controller.attach(model); controller.set_layouts(layouts, shifts)
        actual = generate_tokens(model, prompts, pad_id=pad, eos_ids=eos, max_new_tokens=16)
    controller.clear()
    if expected != actual: raise RuntimeError("Zero-bias backend changed greedy tokens; stop before training")
    return {"n": len(prompts), "identical_greedy_sequences": len(prompts), "max_new_tokens": 16}


def input_dependence(controller, rows, c):
    """Sensitivity diagnostic, NOT proof gates are correct or graph reasoning."""
    controller.eval()
    qdiff, gdiff = [], []
    with torch.no_grad():
        for i, row in enumerate(rows[:8]):
            spec = c["layouts"][row["sample_id"]]
            if not spec["links"]: continue
            original = controller.link_biases(spec)
            changed = {**spec, "question_text": rows[(i+1) % len(rows)]["question"]}
            qdiff.append((original-controller.link_biases(changed)).abs().max().item())
            old = controller.variant
            controller.variant = "local" if old == "graph" else "graph"
            try: gdiff.append((original-controller.link_biases(spec)).abs().max().item())
            finally: controller.variant = old
    return {"n": len(qdiff), "question_swap_max_abs_delta": max(qdiff, default=0.),
            "neighbor_state_switch_max_abs_delta": max(gdiff, default=0.)}


def tiny_gate(a, c, model, variant, progress, index):
    root = a.output_root / variant / "tiny32"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "GATE.json"
    if marker.exists():
        report = json.loads(marker.read_text())
        if report["contract_sha256"] != c["run_hash"]: raise RuntimeError("Tiny gate contract mismatch")
        progress.stage(index, 100, initial=100, unit="update")
        progress.complete(f"{variant} tiny32 gate reused: passed={report['passed']}")
        return report["passed"]
    controller = new_controller(a, c, variant)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=a.learning_rate, weight_decay=0.)
    rows = c["tiny"]
    state = {"step": 0, "history": [], "max_grad_norm": 0.}
    last = root / "last.pt"
    progress.stage(index, 100, unit="update", kind="tiny")
    if last.exists():
        saved = load_torch(last, trusted=True)
        if saved["contract_sha256"] != c["run_hash"]: raise RuntimeError("Tiny checkpoint contract mismatch")
        controller.load_state_dict(saved["controller"]); optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]; restore_rng(saved["rng"]); controller.attach(model)
        progress.stage(index, 100, initial=state["step"], unit="update", kind="tiny")
    else:
        progress.log(f"[tiny32 {variant}] checking 16 zero-bias outputs, then initial CE; at most 100 updates")
        atomic_json(root / "ZERO_PARITY.json", zero_parity(model, controller, rows, c))
        initial, records = validation(model, controller, rows, c)
        state["history"].append({"step": 0, "mean_answer_ce": initial})
        atomic_jsonl(root / "initial_losses.jsonl", records)
    def save():
        torch_save(last, {"contract_sha256": c["run_hash"], "controller": controller.state_dict(),
                         "optimizer": optimizer.state_dict(), "state": state, "rng": rng_state()})
        atomic_json(root / "history.json", state["history"])
    save()
    while state["step"] < 100:
        if state["history"][-1]["mean_answer_ce"] <= .8*state["history"][0]["mean_answer_ce"]: break
        # Same deterministic sequence and effective batch size in both arms.
        order = list(range(len(rows))); random.Random(a.seed+state["step"]//2).shuffle(order)
        start = (state["step"]*a.effective_batch_size) % len(rows)
        batch = [rows[order[(start+j) % len(rows)]] for j in range(min(a.effective_batch_size, len(rows)))]
        _, grad = update(model, controller, optimizer, batch, a, c, progress)
        state["step"] += 1; state["max_grad_norm"] = max(state["max_grad_norm"], grad)
        progress.update()
        if state["step"] % 10 == 0:
            loss, records = validation(model, controller, rows, c)
            state["history"].append({"step": state["step"], "mean_answer_ce": loss})
            atomic_jsonl(root / "latest_losses.jsonl", records); save()
            progress.log(f"[tiny32 {variant}] update={state['step']} CE={loss:.6f}; target<={.8*state['history'][0]['mean_answer_ce']:.6f}")
        elif state["step"] % a.checkpoint_steps == 0: save()
    sensitivity = input_dependence(controller, rows, c)
    initial, final = state["history"][0]["mean_answer_ce"], state["history"][-1]["mean_answer_ce"]
    passed = bool(final <= .8*initial and state["max_grad_norm"] > 0 and sensitivity["question_swap_max_abs_delta"] > 1e-6)
    if variant == "graph": passed = passed and sensitivity["neighbor_state_switch_max_abs_delta"] > 1e-6
    save()
    atomic_json(marker, {"contract_sha256": c["run_hash"], "variant": variant, "passed": bool(passed),
        "n": len(rows), "updates": state["step"], "initial_answer_ce": initial, "final_answer_ce": final,
        "relative_ce_reduction": 1-final/initial, "required_reduction": .20,
        "max_grad_norm": state["max_grad_norm"], "sensitivity": sensitivity,
        "quality_validated": False, "next": "fresh initialization for full training" if passed else "stop; diagnose actuator/loss/inputs, do not scale"})
    progress.complete(f"{variant} tiny32 passed={passed}; {initial:.6f}->{final:.6f}; {marker}")
    controller.clear(); del optimizer, controller; gc.collect(); torch.cuda.empty_cache()
    return passed


def full_train(a, c, model, variant, progress, first):
    root = a.output_root / variant / "full"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "COMPLETE.json"
    controller = new_controller(a, c, variant)  # NOT the tiny32 checkpoint.
    controller.attach(model)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=a.learning_rate, weight_decay=0.)
    rows = [r for r in c["rows"] if r["split"] == "train"]
    val = [r for r in c["rows"] if r["split"] == "validation"]
    state = {"epoch": 0, "cursor": 0, "step": 0, "sum": 0., "n": 0, "history": [], "best": float("inf")}
    last = root / "last.pt"
    if last.exists():
        saved = load_torch(last, trusted=True)
        if saved["contract_sha256"] != c["run_hash"] or saved["variant"] != variant: raise RuntimeError("Full checkpoint mismatch")
        controller.load_state_dict(saved["controller"]); optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]; restore_rng(saved["rng"])
    def save():
        torch_save(last, {"contract_sha256": c["run_hash"], "variant": variant, "controller": controller.state_dict(),
            "optimizer": optimizer.state_dict(), "state": state, "rng": rng_state()})
        atomic_json(root / "history.json", state["history"])
    def select(epoch, loss):
        if loss < state["best"]:
            state["best"] = loss
            torch_save(root / "best.pt", {"contract_sha256": c["run_hash"], "variant": variant,
                "controller": controller.state_dict(), "epoch": epoch, "validation_ce": loss})
    progress.stage(first, len(val), initial=len(val) if state["history"] else 0, unit="q", kind="validation")
    if not state["history"]:
        loss, records = validation(model, controller, val, c, progress)
        select(0, loss); state["history"].append({"epoch": 0, "validation_ce": loss, "validation_n": len(val)})
        atomic_jsonl(root / "validation_epoch0.jsonl", records); save()
    progress.complete(f"{variant} initial validation CE={state['history'][0]['validation_ce']:.6f}")
    for epoch in range(a.epochs):
        ti = first+1+2*epoch
        if epoch < state["epoch"]:
            for idx, size in ((ti, len(rows)), (ti+1, len(val))):
                progress.stage(idx, size, initial=size); progress.complete("durable epoch reused")
            continue
        order = list(range(len(rows))); random.Random(a.seed+epoch).shuffle(order)
        progress.stage(ti, len(rows), initial=state["cursor"], unit="q", kind="train")
        for start in range(state["cursor"], len(rows), a.effective_batch_size):
            batch = [rows[i] for i in order[start:start+a.effective_batch_size]]
            total, grad = update(model, controller, optimizer, batch, a, c, progress)
            state.update(cursor=start+len(batch), step=state["step"]+1, sum=state["sum"]+total,
                         n=state["n"]+len(batch), last_gradient_norm=grad)
            if state["step"] % a.checkpoint_steps == 0 or state["cursor"] == len(rows): save()
            progress.update(len(batch))
            calibrate_comparable_stages(progress)
        progress.complete(f"{variant} train CE={state['sum']/state['n']:.6f}; last={last}")
        progress.stage(ti+1, len(val), unit="q", kind="validation")
        loss, records = validation(model, controller, val, c, progress)
        select(epoch+1, loss)
        state["history"].append({"epoch": epoch+1, "train_ce": state["sum"]/state["n"], "train_n": state["n"],
            "validation_ce": loss, "validation_n": len(val), "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "last_gradient_norm": state["last_gradient_norm"]})
        atomic_jsonl(root / f"validation_epoch{epoch+1}.jsonl", records)
        state.update(epoch=epoch+1, cursor=0, sum=0., n=0); save()
        progress.complete(f"{variant} validation CE={loss:.6f}; best={state['best']:.6f}")
    best = load_torch(root / "best.pt")
    atomic_json(marker, {"contract_sha256": c["run_hash"], "variant": variant, "epochs": a.epochs,
        "selected_epoch": best["epoch"], "best_validation_ce": best["validation_ce"],
        "best_sha256": sha_file(root / "best.pt"), "zero_baseline_selected": best["epoch"] == 0,
        "train_questions": len(rows), "validation_questions": len(val), "evaluation_questions_used": 0,
        "quality_validated": False, "tiny_weights_reused": False})
    controller.clear(); del optimizer, controller; gc.collect(); torch.cuda.empty_cache()


@training_attention_backend()
def run(a, c, progress):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
        raise RuntimeError("Run wrapper with physical GPU 1; CUDA unavailable/wrong device")
    c["bank"] = encode_bank(c["feature_root"], c["plans"], c["encoders"], c["feature_hash"],
                           progress, {"query": 3, "article": 4}, a.embedding_batch_size, torch_save, load_torch)
    progress.stage(5, 1, unit="model")
    set_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(model, c["lora"], is_trainable=False, local_files_only=True).get_base_model()
    model.requires_grad_(False); model.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("Unexpected trainable Llama weights")
    probe = DynamicBias(**c["contract"]["architecture"])
    parameters = sum(p.numel() for p in probe.parameters()); del probe
    progress.complete(f"frozen Llama/LoRA; dynamic trainable parameters={parameters:,}; CPU feature bank")
    passes = [tiny_gate(a, c, model, v, progress, 6+i) for i, v in enumerate(a.variants)]
    if not all(passes):
        raise RuntimeError("32-Q learnability gate failed. Full training NOT started; inspect <variant>/tiny32/GATE.json")
    if a.gate_only:
        progress.log("[result] real tiny32 gates passed; full training not run; rerun without --gate-only to continue")
        return
    first = 6+len(a.variants)
    for variant in a.variants:
        full_train(a, c, model, variant, progress, first)
        first += 1+2*a.epochs
    progress.stage(len(progress.names)-1, 1, unit="report")
    results = {v: json.loads((a.output_root / v / "full/COMPLETE.json").read_text()) for v in a.variants}
    atomic_json(a.output_root / "COMPLETE.json", {"contract_sha256": c["run_hash"], "variants": results,
        "evaluation_questions_used": 0, "quality_validated": False})
    progress.update(); progress.complete(f"training only; {a.output_root / 'COMPLETE.json'}")
    progress.log("[result] both selected controllers saved; no final 512-Q evaluation performed")


def main(argv=None):
    a = arguments(argv)
    names = ["CPU contracts", "source/token alignment", "encoder token coverage"]
    if not a.preflight_only:
        names += ["MedCPT query windows", "MedCPT article windows", "load frozen Llama"]
        names += [f"tiny32 {v}" for v in a.variants]
        if not a.gate_only:
            for v in a.variants:
                names += [f"validation {v} epoch 0"]
                for epoch in range(1, a.epochs+1):
                    names += [f"train {v} epoch {epoch}/{a.epochs}", f"validation {v} epoch {epoch}/{a.epochs}"]
            names += ["save selections"]
    progress = Progress(names)
    if names[-1] == "save selections": progress.weights[-1] = 1.  # Small JSON I/O cost model.
    try:
        from transformers.utils import logging
        logging.set_verbosity_error(); logging.disable_progress_bar()
        torch.set_num_threads(a.cpu_threads)
        progress.stage(0, 1, unit="contract")
        progress.log(f"[plan] {' -> '.join(names)}; variants={a.variants}; train/val=2048/256; no final evaluation")
        progress.log("[ETA] new controller throughput unmeasured: encoder/tiny/train/validation calibrate separately; no promised completion time")
        with lock_file(a.output_root / ".run.lock"):
            c = prepare(a, progress)
            if a.preflight_only:
                progress.log("[result] CPU preflight passed; no encoder/Llama weights loaded; no GPU execution"); return
            if (a.output_root / "COMPLETE.json").exists():
                done = json.loads((a.output_root / "COMPLETE.json").read_text())
                if done["contract_sha256"] != c["run_hash"]: raise RuntimeError("Completion contract mismatch")
                for v, record in done["variants"].items():
                    if sha_file(a.output_root / v / "full/best.pt") != record["best_sha256"]:
                        raise RuntimeError("Completed selected weights changed")
                progress.log("[result] already complete; no GPU work"); return
            run(a, c, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; "
                     f"remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; "
                     f"durable checkpoints/features={a.output_root}; identical command resumes; failed quality gates do not auto-bypass")
        raise
    finally: progress.close()


if __name__ == "__main__":
    main()

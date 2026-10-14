#!/usr/bin/env python3
"""Learn a KG attention overlay, then compare on the unchanged 512-Q pilot.

train: frozen shared medical LoRA + 2048/256 train/validation, bias only.
evaluate: one new condition; reuse same-LoRA No-RAG and TEXT predictions.
--preflight-only never loads Llama or allocates CUDA.
evaluate --prepare-only checks held-out inputs before a checkpoint exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from medrag.generation.kg_structure_attention import StructureBias, answer_losses, token_layout, KINDS
from medrag.training.graph_input_data import (SYSTEM, atomic_json, atomic_jsonl, digest,
    ensure_contract, identity, jsonl, sha_file, graph_from_edges)
from medrag.training.graph_input_progress import Progress
from medrag.training.graph_input_modules import training_attention_backend
from medrag.text_graph_evaluation import generate_tokens, parse_answer, GROUPS, group, wilson
from train_rag2_text_kg_input_pilot import (bundle_identity, load_torch, lock_file,
    restore_rng, rng_state, set_seed, torch_save)

VERSION = "kg_structure_attention_v1"
SOURCES = {"no_rag": "6_trained_no_rag", "text": "3_trained_text"}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "evaluate"))
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/pilots/rag2_text_kg_input_test_resplit_v1")
    p.add_argument("--shared-root", type=Path, default=ROOT / "results/rag2_shared_medical_lora_v1")
    p.add_argument("--reference-root", type=Path, default=ROOT / "results/rag2_text_graph_six_conditions_v1")
    p.add_argument("--training-root", type=Path, default=ROOT / "results/rag2_kg_attention_training_v1")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--effective-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--max-bias", type=float, default=2.)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--checkpoint-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args(argv)
    if a.output_root is None:
        a.output_root = (a.training_root if a.mode == "train" else ROOT / "results/rag2_kg_attention_evaluation_v1")
        if a.smoke_steps:
            a.output_root = a.output_root.with_name(a.output_root.name + "_smoke")
    if a.prepare_only and a.mode != "evaluate": p.error("--prepare-only is evaluation input checking only")
    if a.smoke_steps and a.mode != "train": p.error("Smoke checkpoints cannot be evaluated")
    return a


def checked(path):
    m = json.loads(path.read_text())
    if digest(m["contract"]) != m["contract_sha256"]:
        raise RuntimeError(f"Manifest checksum mismatch: {path}")
    return m


def versions():
    return {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft")}


def code_identity():
    names = ("scripts/run_rag2_kg_attention_pilot.py", "medrag/generation/kg_structure_attention.py",
             "medrag/text_graph_evaluation.py", "medrag/training/graph_input_data.py",
             "medrag/training/graph_input_progress.py", "medrag/training/graph_input_modules.py",
             "scripts/train_rag2_text_kg_input_pilot.py")
    return {n: sha_file(ROOT / n) for n in names}


def common_inputs(a):
    if min(a.epochs, a.micro_batch_size, a.effective_batch_size, a.query_chunk,
           a.checkpoint_steps, a.cpu_threads) < 1 or a.micro_batch_size > a.effective_batch_size:
        raise ValueError("Invalid batch/epoch/chunk configuration")
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0 or not 0 < a.max_bias <= 10 or a.smoke_steps < 0:
        raise ValueError("Invalid finite optimization parameters")
    protected = [a.model, a.data_root, a.shared_root, a.reference_root]
    if a.mode == "evaluate": protected.append(a.training_root)
    out = a.output_root.resolve()
    for p in protected:
        p = p.resolve()
        if out == p or out in p.parents or p in out.parents:
            raise ValueError(f"Output overlaps protected artifact: {p}")
    dm = checked(a.data_root / "manifest.json")
    dc = json.loads((a.data_root / "COMPLETE.json").read_text())
    if dc["contract_sha256"] != dm["contract_sha256"]: raise RuntimeError("Data completion mismatch")
    # Training verifies only train/validation data; evaluation does not read them.
    names = dc["file_sha256"] if a.mode == "train" else ("split_assignments.jsonl",)
    for name in names:
        if sha_file(a.data_root / name) != dc["file_sha256"][name]:
            raise RuntimeError(f"Frozen split modified: {name}")
    assignments = list(jsonl(a.data_root / "split_assignments.jsonl"))
    if len(assignments) != 6545 or len({r["sample_id"] for r in assignments}) != 6545:
        raise RuntimeError("Expected unchanged TEST6545 partition")
    groups = {}
    for r in assignments: groups.setdefault(r["duplicate_group"], set()).add(r["split"])
    if any(len(s) != 1 for s in groups.values()): raise RuntimeError("Duplicate group crosses split")
    model_id = bundle_identity(a.model)
    for tok in dm["contract"]["tokenizer"]:
        if identity(a.model / Path(tok["path"]).name, full_hash=True) != tok:
            raise RuntimeError("Cached prompt tokenizer changed")
    sm = checked(a.shared_root / "manifest.json")
    sc = json.loads((a.shared_root / "COMPLETE.json").read_text())
    sb = json.loads((a.shared_root / "best.json").read_text())
    if (sm["contract"]["data_contract"] != dm["contract_sha256"] or sm["contract"]["model"] != model_id or
            sc["contract_sha256"] != sm["contract_sha256"] or sc["trained_llm_adapter"] != sb["adapter"] or
            sc["best_text_validation_ce"] != sb["validation_answer_ce"]):
        raise RuntimeError("Frozen shared medical LoRA/data/model mismatch")
    lora = Path(sb["adapter"]).resolve()
    if a.shared_root.resolve() not in lora.parents: raise RuntimeError("LoRA outside shared root")
    lora_id = [identity(lora / n, full_hash=True) for n in ("adapter_config.json", "adapter_model.safetensors")]
    cfg = json.loads((a.model / "config.json").read_text())
    if cfg["model_type"] != "llama" or cfg.get("attention_dropout", 0) != 0:
        raise RuntimeError("Only the existing zero-dropout Llama is supported")
    return {"data": dm, "data_done": dc, "assignments": assignments, "model": model_id,
            "lora": lora, "lora_id": lora_id, "shared_contract": sm["contract_sha256"], "config": cfg}


def validate_scope(rows, assignments, splits, counts, max_context):
    expected = {r["sample_id"]: r["split"] for r in assignments if r["split"] in splits}
    actual = {r["sample_id"]: r["split"] for r in rows}
    if actual != expected or len(actual) != len(rows) or dict(Counter(actual.values())) != counts:
        raise RuntimeError("Missing/duplicate/incorrect split examples")
    for r in rows:
        if graph_from_edges(r["edges"]) != r["graph"]: raise RuntimeError("Frozen graph changed")
        if len(r["document_ids"]) != 8 or len(set(r["document_ids"])) != 8:
            raise RuntimeError("Original Top-8 scope changed")
        if any(e["document_id"] not in r["document_ids"] for e in r["edges"]):
            raise RuntimeError("KG edge outside Top-8")
        if set(e["proposition_id"] for e in r["edges"]) != set(p["proposition_id"] for p in r["propositions"]):
            raise RuntimeError("Edge/proposition provenance mismatch")
        length = len(r["text_prompt_ids"]) + (len(r["target_ids"]) if "target_ids" in r else 16)
        if length > max_context: raise RuntimeError("Text context overflow; no truncation allowed")


def prepare(a, progress):
    c = common_inputs(a)
    dm = c["data"]
    c["max_context"] = dm["contract"]["max_context"]
    if c["max_context"] > c["config"]["max_position_embeddings"]: raise RuntimeError("Context exceeds model")
    if a.mode == "train":
        rows = list(jsonl(a.data_root / "train_validation.jsonl"))
        validate_scope(rows, c["assignments"], ("train", "validation"), {"train": 2048, "validation": 256}, c["max_context"])
        input_id = identity(a.data_root / "train_validation.jsonl", full_hash=True)
    else:
        rm = checked(a.reference_root / "manifest.json"); rc = rm["contract"]
        rd = json.loads((a.reference_root / "COMPLETE.json").read_text())
        for name, key in (("paired_predictions.jsonl", "paired_predictions_sha256"), ("summary.json", "summary_sha256")):
            if sha_file(a.reference_root / name) != rd[key]: raise RuntimeError("Reference output modified")
        if (rd["contract_sha256"] != rm["contract_sha256"] or rd["n"] != 512 or rd["conditions"] != 6 or
                rc["base_llm"] != c["model"] or rc["lora"] != c["lora_id"] or
                rc["training_contract"] != c["shared_contract"] or rc["paired_data_contract"] != dm["contract_sha256"] or
                rc["system_prompt"] != SYSTEM or any(rc["environment"][n] != v for n, v in versions().items())):
            raise RuntimeError("Cached baselines are not the same model/data/prompt/runtime")
        marker = json.loads((a.reference_root / "PAIRS_COMPLETE.json").read_text())
        path = a.reference_root / "evaluation_pairs.jsonl"
        if sha_file(path) != marker["sha256"]: raise RuntimeError("Evaluation inputs modified")
        rows = list(jsonl(path))
        if digest([r["sample_id"] for r in rows]) != marker["ids_sha256"]: raise RuntimeError("Evaluation order changed")
        validate_scope(rows, c["assignments"], ("pilot_evaluation",), {"pilot_evaluation": 512}, c["max_context"])
        old = list(jsonl(a.reference_root / "paired_predictions.jsonl"))
        if [r["sample_id"] for r in old] != [r["sample_id"] for r in rows]: raise RuntimeError("Baseline ID order mismatch")
        baselines = {name: [] for name in SOURCES}
        for r, saved in zip(rows, old, strict=True):
            if r["answer"] != saved["gold"] or r["dataset"] != saved["dataset"]: raise RuntimeError("Reference label mismatch")
            for name, source in SOURCES.items():
                item = saved["conditions"][source]
                if item["prediction"] != parse_answer(item["output_text"], r["options"]): raise RuntimeError("Frozen parser mismatch")
                baselines[name].append({**item, "reused": True, "source_condition": source})
        gen = rc["generation"]
        if gen["do_sample"] or gen["num_beams"] != 1 or gen["max_new_tokens"] != 16 or gen["batch_size"] != 16:
            raise RuntimeError("Unexpected reference generation settings")
        c.update(reference=rm, baselines=baselines, generation=gen)
        input_id = identity(path, full_hash=True)
    c["rows"] = rows
    progress.complete(f"same model/splits checked; rows={len(rows)}; output={a.output_root}")
    progress.stage(1, len(rows), unit="q", kind="mapping")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast: raise RuntimeError("Fast offset tokenizer required")
    layout_contract = {"version": VERSION, "input": input_id, "tokenizer": dm["contract"]["tokenizer"],
                       "code": code_identity(), "scope": "exact evidence-only spans; no gold, options or question mapping"}
    lh = ensure_contract(a.output_root / "layouts_manifest.json", layout_contract)
    layout_path = a.output_root / "layouts.jsonl"
    marker_path = a.output_root / "LAYOUTS_COMPLETE.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker["contract_sha256"] != lh or marker["sha256"] != sha_file(layout_path): raise RuntimeError("Layout cache changed")
        layouts = list(jsonl(layout_path))
        if [x["sample_id"] for x in layouts] != [x["sample_id"] for x in rows]: raise RuntimeError("Layout order mismatch")
        progress.stage(1, len(rows), initial=len(rows), unit="q")
    else:
        layouts = []
        for row in rows:
            layouts.append(token_layout(tokenizer, row)); progress.update()
        atomic_jsonl(layout_path, layouts)
        atomic_json(marker_path, {"contract_sha256": lh, "sha256": sha_file(layout_path)})
    stats = Counter()
    for x in layouts: stats.update(x["statistics"])
    stats["questions"] = len(rows)
    stats["max_prompt_tokens"] = max(len(r["text_prompt_ids"]) for r in rows)
    c.update(layouts={x["sample_id"]: x for x in layouts}, statistics=dict(stats), tokenizer=tokenizer)
    architecture = {"layers": c["config"]["num_hidden_layers"], "heads": c["config"]["num_attention_heads"],
                    "max_bias": a.max_bias, "query_chunk": a.query_chunk}
    contract = {"version": VERSION, "purpose": a.mode, "data_contract": dm["contract_sha256"],
        "data_files": c["data_done"]["file_sha256"], "model": c["model"], "frozen_lora": c["lora_id"],
        "shared_contract": c["shared_contract"], "architecture": architecture, "kinds": list(KINDS),
        "input": input_id, "layout_contract": lh, "layout_sha256": sha_file(layout_path),
        "statistics": dict(stats), "max_context": c["max_context"], "system_prompt": SYSTEM,
        "hypothesis": "Same proposition text plus KG-linked attention improves use of connected evidence",
        "changed_variable": "zero-initialized bounded additive KG attention bias; no new evidence or tokens",
        "initialization": "all 3072 layer/head/type raw scalars zero on the fixed 32x32 Llama",
        "causality": "original causal/padding masks retained; only evidence-to-evidence positions biased; generated tokens have no direct KG bias",
        "mapping": "unique exact source endpoint spans; nested spans allowed; conflicting IDs at same span skip identity links only; ambiguous surface locations skip edge; text retained",
        "loss": "same per-question teacher-forced Answer: letter CE; only attention biases optimized",
        "scope": "existing TEST repartition development pilot; MCQ-supervised, not untouched final TEST",
        "claim_limit": "additional supervised attention parameters; improvement alone does not isolate topology from extra fitting; no shuffled-graph control here",
        "code": code_identity(), "environment": versions()}
    if a.mode == "train":
        contract.update(training={k: getattr(a, k) for k in ("epochs", "micro_batch_size", "effective_batch_size",
            "learning_rate", "checkpoint_steps", "seed", "smoke_steps")}, optimizer="AdamW weight_decay=0, clip=1",
            selection="minimum validation CE INCLUDING initial zero-bias model; no held-out selection",
            heldout_policy="pilot_evaluation and reserve questions/labels never loaded")
    else:
        contract.update(generation=c["generation"], reference_contract=c["reference"]["contract_sha256"],
            reference_predictions=identity(a.reference_root / "paired_predictions.jsonl", full_hash=True), baselines=SOURCES,
            primary_metric="Graph attention minus same-LoRA TEXT Macro-3; +2pp and paired 95% CI lower>0 exploratory gate")
        if not a.prepare_only:
            tm = checked(a.training_root / "manifest.json")
            tc = tm["contract"]
            done = json.loads((a.training_root / "COMPLETE.json").read_text())
            if (done["contract_sha256"] != tm["contract_sha256"] or tc["purpose"] != "train" or
                    tc["training"]["smoke_steps"] or done["epochs"] != tc["training"]["epochs"] or
                    done["evaluation_questions_used"] != 0): raise RuntimeError("Training incomplete or not an eligible pilot")
            for key in ("data_contract", "model", "frozen_lora", "shared_contract", "system_prompt", "environment", "code"):
                if tc[key] != contract[key]: raise RuntimeError(f"Training/evaluation mismatch: {key}")
            best = a.training_root / "best.pt"
            if sha_file(best) != done["best_sha256"]: raise RuntimeError("Selected checkpoint modified")
            saved = load_torch(best)
            if saved["contract_sha256"] != tm["contract_sha256"] or saved["validation_ce"] != done["best_validation_ce"]:
                raise RuntimeError("Best checkpoint/selection mismatch")
            history = json.loads((a.training_root / "history.json").read_text())
            if (len(history) != tc["training"]["epochs"]+1 or min(h["validation_ce"] for h in history) != done["best_validation_ce"] or
                    any(h["validation_n"] != 256 for h in history)):
                raise RuntimeError("Validation selection history incomplete")
            contract.update(architecture=tc["architecture"], training_contract=tm["contract_sha256"],
                selected_checkpoint=identity(best, full_hash=True), selected_epoch=saved["epoch"])
            c["weights"] = saved["controller"]
    c["contract"] = contract
    progress.complete(f"exact text tokens retained; {json.dumps(dict(stats))}; {layout_path}")
    return c


def load_model(a, c, progress, index):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    progress.stage(index, 1, unit="model")
    set_seed(c["generation"]["seed"] if a.mode == "evaluate" else a.seed)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(model, c["lora"], is_trainable=False, local_files_only=True).get_base_model()
    model.requires_grad_(False); model.eval()
    controller = StructureBias(**c["contract"]["architecture"]).to("cuda:0")
    if "weights" in c: controller.load_state_dict(c["weights"])
    controller.attach(model)
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("Frozen Llama unexpectedly trainable")
    progress.complete(f"frozen shared Llama + {controller.raw.numel()} bias scalars; no node vectors")
    return model, controller


def validate(model, bias, rows, c, progress):
    import torch
    model.eval(); bias.eval()
    records = []
    with torch.no_grad():
        for row in rows:
            value = answer_losses(model, bias, [row], [c["layouts"][row["sample_id"]]], c["tokenizer"].eos_token_id).item()
            if not math.isfinite(value): raise FloatingPointError("Non-finite validation loss")
            records.append({"sample_id": row["sample_id"], "answer_ce": value}); progress.update()
    return sum(r["answer_ce"] for r in records)/len(records), records


@training_attention_backend()
def train(a, c, run_hash, progress):
    import torch
    model, bias = load_model(a, c, progress, 2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(bias.parameters(), lr=a.learning_rate, weight_decay=0.)
    rows = [r for r in c["rows"] if r["split"] == "train"]
    val = [r for r in c["rows"] if r["split"] == "validation"]
    state = {"epoch": 0, "cursor": 0, "step": 0, "sum": 0., "n": 0,
             "best": float("inf"), "history": [], "zero_validated": False}
    last = a.output_root / "last.pt"
    if last.exists():
        saved = load_torch(last, trusted=True)
        if saved["contract_sha256"] != run_hash: raise RuntimeError("Resume checkpoint contract mismatch")
        bias.load_state_dict(saved["controller"]); opt.load_state_dict(saved["optimizer"])
        state = saved["state"]; restore_rng(saved["rng"])
    def save():
        torch_save(last, {"contract_sha256": run_hash, "controller": bias.state_dict(),
            "optimizer": opt.state_dict(), "state": state, "rng": rng_state()})
        atomic_json(a.output_root / "history.json", state["history"])
        progress.weights[-1] = 1.  # Small atomic checkpoint I/O cost model, not GPU throughput.
    def selection(epoch, loss):
        if loss < state["best"]:
            state["best"] = loss
            torch_save(a.output_root / "best.pt", {"contract_sha256": run_hash, "controller": bias.state_dict(),
                "epoch": epoch, "validation_ce": loss})
    progress.stage(3, len(val), initial=len(val) if state["zero_validated"] else 0, unit="q", kind="validation")
    if not state["zero_validated"]:
        loss, records = validate(model, bias, val, c, progress)
        selection(0, loss)
        state["history"].append({"epoch": 0, "validation_ce": loss, "validation_n": len(val), "train_n": 0})
        state["zero_validated"] = True
        atomic_jsonl(a.output_root / "validation_epoch0.jsonl", records); save()
    progress.complete(f"zero-bias baseline validation CE={state['history'][0]['validation_ce']:.6f}")
    for epoch in range(a.epochs):
        train_index = 4 + 2*epoch
        if epoch < state["epoch"]:
            for i, n in ((train_index, len(rows)), (train_index+1, len(val))):
                progress.stage(i, n, initial=n); progress.complete("durable epoch reused")
            continue
        order = list(range(len(rows))); random.Random(a.seed+epoch).shuffle(order)
        progress.stage(train_index, len(rows), initial=state["cursor"], unit="q", kind="train")
        model.train(); bias.train()
        # Frozen LoRA can still contain dropout; keep its computation fixed.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.eval()
        for start in range(state["cursor"], len(rows), a.effective_batch_size):
            batch = [rows[i] for i in order[start:start+a.effective_batch_size]]
            micro = min(a.micro_batch_size, len(batch)); before = rng_state()
            while True:
                opt.zero_grad(set_to_none=True); total = 0.
                try:
                    for j in range(0, len(batch), micro):
                        piece = batch[j:j+micro]
                        layouts = [c["layouts"][r["sample_id"]] for r in piece]
                        values = answer_losses(model, bias, piece, layouts, c["tokenizer"].eos_token_id)
                        if not torch.isfinite(values).all(): raise FloatingPointError("Non-finite training loss")
                        if values.requires_grad: (values.sum()/len(batch)).backward()
                        total += values.detach().sum().item(); del values
                    break
                except torch.cuda.OutOfMemoryError:
                    if "values" in locals(): del values
                    opt.zero_grad(set_to_none=True); restore_rng(before); gc.collect(); torch.cuda.empty_cache()
                    if micro == 1: raise
                    micro = max(1, micro//2)
                    progress.log(f"[OOM] retry current optimizer step with micro={micro}; restore configured size next step")
            has_links = any(c["layouts"][r["sample_id"]]["pairs"] for r in batch)
            if has_links and (bias.raw.grad is None or not torch.isfinite(bias.raw.grad).all()):
                raise FloatingPointError("Missing/nonfinite bias gradient; stop before scaling")
            if bias.raw.grad is not None:
                torch.nn.utils.clip_grad_norm_(bias.parameters(), 1.); opt.step()
            state.update(cursor=start+len(batch), step=state["step"]+1,
                         sum=state["sum"]+total, n=state["n"]+len(batch))
            if state["step"] % a.checkpoint_steps == 0 or state["cursor"] == len(rows): save()
            progress.update(len(batch))
            if a.smoke_steps and state["step"] >= a.smoke_steps:
                save(); atomic_json(a.output_root / "SMOKE_COMPLETE.json", {"contract_sha256": run_hash,
                    "step": state["step"], "quality_validated": False})
                progress.log("[result] bounded training smoke only; not eligible for 512-Q evaluation")
                return
        progress.complete(f"train CE={state['sum']/state['n']:.6f}; checkpoint={last}")
        progress.stage(train_index+1, len(val), unit="q", kind="validation")
        loss, records = validate(model, bias, val, c, progress)
        selection(epoch+1, loss)
        state["history"].append({"epoch": epoch+1, "validation_ce": loss, "validation_n": len(val),
            "train_ce": state["sum"]/state["n"], "train_n": state["n"],
            "bias_abs_max": (bias.max_bias*bias.raw.tanh()).abs().max().item(),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        atomic_jsonl(a.output_root / f"validation_epoch{epoch+1}.jsonl", records)
        state.update(epoch=epoch+1, cursor=0, sum=0., n=0); save()
        progress.complete(f"validation CE={loss:.6f}; best={state['best']:.6f}")
    progress.stage(len(progress.names)-1, 1, unit="checkpoint")
    best = load_torch(a.output_root / "best.pt")
    atomic_json(a.output_root / "COMPLETE.json", {"contract_sha256": run_hash, "epochs": a.epochs,
        "best_sha256": sha_file(a.output_root / "best.pt"), "best_validation_ce": state["best"],
        "selected_epoch": best["epoch"], "evaluation_questions_used": 0,
        "quality_validated": False, "zero_baseline_selected": best["epoch"] == 0})
    progress.update(); progress.complete(f"selected epoch={best['epoch']}; {a.output_root / 'best.pt'}")
    progress.log("[result] bias training complete; no 512-Q generation performed")


def summarize(rows, predictions):
    import numpy as np
    ids = [r["sample_id"] for r in rows]
    indices = {g: [i for i, r in enumerate(rows) if group(r["dataset"]) == g] for g in GROUPS}
    rng = np.random.default_rng(42)
    draws = {g: rng.integers(0, len(ix), (2000, len(ix))) for g, ix in indices.items()}
    def ci(values):
        means = {g: values[ix][draws[g]].mean(1) for g, ix in indices.items()}
        return {"macro3_ci95_pp": (100*np.quantile(np.mean(list(means.values()), axis=0), [.025, .975])).tolist(),
                "micro_ci95_pp": (100*np.quantile(sum(means[g]*len(indices[g]) for g in GROUPS)/len(rows), [.025, .975])).tolist()}
    stats, correct = {}, {}
    for name, items in predictions.items():
        if [r["sample_id"] for r in items] != ids: raise RuntimeError("Missing, duplicate or reordered predictions")
        v = np.array([p["prediction"] == r["answer"] for p, r in zip(items, rows)], dtype=float)
        correct[name] = v
        groups = {g: {"n": len(ix), "correct": int(v[ix].sum()), "accuracy_pct": 100*v[ix].mean()} for g, ix in indices.items()}
        stats[name] = {"n": len(rows), "correct": int(v.sum()), "micro_pct": 100*v.mean(),
            "macro3_pct": np.mean([g["accuracy_pct"] for g in groups.values()]), "groups": groups,
            "missing": 0, "parse_failures": sum(p["prediction"] is None for p in items),
            "length_limited": sum(p["finish_reason"] == "length" for p in items),
            "micro_wilson_ci95_pct": wilson(int(v.sum()), len(rows)), **ci(v)}
    pairs = {}
    for baseline in SOURCES:
        a, b = correct[baseline], correct["graph_attention"]
        counts = Counter(("C" if x else "W")+"_"+("C" if y else "W") for x, y in zip(a, b))
        pairs[baseline] = {"n": len(rows), "baseline": baseline, "condition": "graph_attention",
            "baseline_micro_pct": stats[baseline]["micro_pct"], "condition_micro_pct": stats["graph_attention"]["micro_pct"],
            "baseline_macro3_pct": stats[baseline]["macro3_pct"], "condition_macro3_pct": stats["graph_attention"]["macro3_pct"],
            "delta_micro_pp": 100*(b-a).mean(), "delta_macro3_pp": stats["graph_attention"]["macro3_pct"]-stats[baseline]["macro3_pct"],
            "transitions": {k: counts[k] for k in ("C_C", "C_W", "W_C", "W_W")}, **ci(b-a)}
    primary = pairs["text"]
    return {"n": len(rows), "conditions": stats, "paired_comparisons": pairs, "primary": primary,
        "exploratory_gate": {"minimum_delta_macro3_pp": 2, "require_ci95_lower_above_zero": True,
            "passed": primary["delta_macro3_pp"] >= 2 and primary["macro3_ci95_pp"][0] > 0},
        "scope": "previously evaluated TEST-repartition development pilot, not final untouched test"}


@training_attention_backend()
def evaluate(a, c, run_hash, progress):
    import torch
    rows = c["rows"]; batch_size = c["generation"]["batch_size"]
    chunks = [(i, a.output_root / "predictions" / f"{i//batch_size:04d}.json") for i in range(0, len(rows), batch_size)]
    cached = {}
    for i, path in chunks:
        if path.exists():
            v = json.loads(path.read_text())
            if v["contract_sha256"] != run_hash or [r["sample_id"] for r in v["rows"]] != [r["sample_id"] for r in rows[i:i+batch_size]]:
                raise RuntimeError("Prediction shard contract/ID mismatch")
            if v["rows_sha256"] != digest(v["rows"]): raise RuntimeError("Prediction shard checksum mismatch")
            cached[i] = v["rows"]
    all_cached = sum(map(len, cached.values())) == len(rows)
    if not all_cached:
        model, bias = load_model(a, c, progress, 2)
        model.gradient_checkpointing_disable(); model.eval()
        tok = c["tokenizer"]; eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
        if None in eos: raise RuntimeError("Missing frozen Llama stopping token")
        def generate(piece):
            prompts = [r["text_prompt_ids"] for r in piece]; width = max(map(len, prompts))
            bias.set_layouts([c["layouts"][r["sample_id"]] for r in piece], [width-len(p) for p in prompts])
            return generate_tokens(model, prompts, pad_id=tok.eos_token_id, eos_ids=eos, max_new_tokens=16)
    else:
        progress.stage(2, 1, initial=1); progress.complete("all predictions cached; model load skipped")
    parity_path = a.output_root / "ZERO_PARITY.json"
    progress.stage(3, 16, initial=16 if parity_path.exists() else 0, unit="q", kind="generation")
    if parity_path.exists():
        p = json.loads(parity_path.read_text())
        if p["contract_sha256"] != run_hash or not p["passed"]: raise RuntimeError("Parity marker mismatch")
    elif all_cached:
        raise RuntimeError("Cached predictions without zero-bias parity evidence")
    else:
        original = bias.raw.detach().clone()
        with torch.no_grad(): bias.raw.zero_()
        outputs = generate(rows[:16])
        failures = [rows[i]["sample_id"] for i, out in enumerate(outputs)
                    if out["generated_ids"] != c["baselines"]["text"][i]["generated_ids"]]
        with torch.no_grad(): bias.raw.copy_(original)
        atomic_json(parity_path, {"contract_sha256": run_hash, "n": 16, "passed": not failures,
            "mismatch_ids": failures, "purpose": "backend compatibility only; no selection or tuning"})
        if failures:
            raise RuntimeError(f"Zero-bias output differs from cached TEXT on {len(failures)}/16; stop, do not claim a controlled comparison")
        progress.update(16)
    progress.complete(f"zero-bias backend compatibility verified; {parity_path}")
    progress.weights[-1] = 2.  # CPU bootstrap/report cost estimate; replaced on completion.
    progress.stage(4, len(rows), initial=sum(map(len, cached.values())), unit="q", kind="generation")
    predictions = []
    for start, path in chunks:
        if start in cached:
            predictions.extend(cached[start]); continue
        batch = rows[start:start+batch_size]; micro = len(batch)
        while True:
            try:
                outputs = []
                for j in range(0, len(batch), micro): outputs.extend(generate(batch[j:j+micro]))
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if micro == 1: raise
                micro = max(1, micro//2)
                progress.log(f"[OOM] batch={start}; retry micro={micro}; restore 16 next batch")
        items = []
        for r, out in zip(batch, outputs, strict=True):
            text = tok.decode(out["generated_ids"], skip_special_tokens=True)
            items.append({"sample_id": r["sample_id"], "dataset": r["dataset"], **out,
                "output_text": text, "prediction": parse_answer(text, r["options"]), "reused": False,
                "prompt_tokens": len(r["text_prompt_ids"]), "actual_micro_batch": micro})
        atomic_json(path, {"contract_sha256": run_hash, "rows": items, "rows_sha256": digest(items)})
        predictions.extend(items); progress.update(len(batch))
    progress.complete(f"one new condition complete; {a.output_root / 'predictions'}")
    progress.stage(5, 1, unit="report")
    all_predictions = {**c["baselines"], "graph_attention": predictions}
    summary = summarize(rows, all_predictions)
    summary.update(contract_sha256=run_hash, mapping_statistics=c["statistics"])
    atomic_json(a.output_root / "summary.json", summary)
    atomic_jsonl(a.output_root / "paired_predictions.jsonl", ({"sample_id": r["sample_id"], "dataset": r["dataset"],
        "gold": r["answer"], "conditions": {name: {**items[i], "correct": items[i]["prediction"] == r["answer"]}
        for name, items in all_predictions.items()}} for i, r in enumerate(rows)))
    table = [{"condition": name, **{k: s[k] for k in ("n", "correct", "micro_pct", "macro3_pct", "parse_failures")}}
             for name, s in summary["conditions"].items()]
    atomic_json(a.output_root / "summary_table.json", table)
    atomic_json(a.output_root / "COMPLETE.json", {"contract_sha256": run_hash, "n": len(rows),
        "summary_sha256": sha_file(a.output_root / "summary.json"),
        "paired_sha256": sha_file(a.output_root / "paired_predictions.jsonl")})
    progress.update(); progress.complete(f"same-512 paired accuracy, transitions and CIs; {a.output_root / 'summary.json'}")
    progress.log(f"[result] {json.dumps(table)}")


def main(argv=None):
    a = arguments(argv)
    names = ["CPU split/model contracts", "map KG to exact text tokens"]
    if not (a.preflight_only or a.prepare_only):
        names += ["load frozen Llama"]
        if a.mode == "train":
            names += ["validation epoch 0 zero-bias"]
            for i in range(a.epochs): names += [f"train epoch {i+1}/{a.epochs} mixed", f"validation epoch {i+1}/{a.epochs} mixed"]
            names += ["save checkpoint selection"]
        else: names += ["zero-bias parity 16", "graph attention MCQ 512", "paired summary"]
    progress = Progress(names)
    try:
        import torch
        from transformers.utils import logging
        logging.set_verbosity_error(); logging.disable_progress_bar()
        torch.set_num_threads(a.cpu_threads)
        progress.stage(0, 1, unit="contract")
        progress.log(f"[plan] {a.mode}: unchanged 2048/256/512 development split; learned attention ONLY; "
                     "exact source proposition text; reuse same-LoRA No-RAG/TEXT; no new retrieval")
        progress.log("[ETA] no compatible learned-attention throughput yet; train/validation/generation calibrate separately on fresh work; GPU 1 only")
        with lock_file(a.output_root / ".run.lock"):
            c = prepare(a, progress)
            if a.prepare_only:
                atomic_json(a.output_root / "INPUT_PREFLIGHT.json", {"statistics": c["statistics"],
                    "checkpoint_validated": False, "gpu_executed": False, "pending": "completed attention training required"})
                progress.log("[result] evaluation inputs/parity of text IDs verified on CPU; training checkpoint still required")
                return
            run_hash = ensure_contract(a.output_root / "manifest.json", c["contract"])
            provenance = a.output_root / "provenance.json"
            if not provenance.exists():
                atomic_json(provenance, {"contract_sha256": run_hash,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
            done_path = a.output_root / "COMPLETE.json"
            if done_path.exists():
                done = json.loads(done_path.read_text())
                if done["contract_sha256"] != run_hash: raise RuntimeError("Completion mismatch")
                pairs = (("best.pt", "best_sha256"),) if a.mode == "train" else (("summary.json", "summary_sha256"), ("paired_predictions.jsonl", "paired_sha256"))
                if any(sha_file(a.output_root / f) != done[k] for f, k in pairs): raise RuntimeError("Completed output modified")
                progress.log("[result] already complete; no GPU work"); return
            atomic_json(a.output_root / "PREFLIGHT.json", {"contract_sha256": run_hash,
                "statistics": c["statistics"], "gpu_executed": False})
            if a.preflight_only:
                progress.log("[result] CPU preflight passed; no Llama load, training or generation"); return
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
                raise RuntimeError("Use the wrapper on physical GPU 1; CUDA unavailable or wrong device selection")
            if a.mode == "train": train(a, c, run_hash, progress)
            else: evaluate(a, c, run_hash, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total} "
            f"remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; durable state={a.output_root}; "
            "identical command resumes compatible checkpoints/shards, without deleting old results")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

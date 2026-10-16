#!/usr/bin/env python3
"""Evaluate selected dynamic controllers on the SAME 512 development questions.

Reuse cached same-LoRA No-RAG, TEXT and static KG attention. Generate only the
new local/graph conditions, with unchanged greedy answer-only decoding.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import run_rag2_kg_attention_pilot as base
from medrag.dynamic_kg_evaluation import paired_summary, read_shard, selected_controllers
from medrag.generation.dynamic_kg_attention import DynamicBias, source_layout
from medrag.training.dynamic_kg_features import encode_bank, feature_plan
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, jsonl, sha_file
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.text_graph_evaluation import generate_tokens, parse_answer
from train_rag2_text_kg_input_pilot import bundle_identity, load_torch, lock_file, set_seed, torch_save


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, default=ROOT / "results/rag2_dynamic_kg_attention_training_v1")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_dynamic_kg_attention_evaluation_v1")
    p.add_argument("--static-root", type=Path, default=ROOT / "results/rag2_kg_attention_evaluation_v1")
    p.add_argument("--variant", choices=("both", "local", "graph"), default="both")
    p.add_argument("--prepare-only", action="store_true", help="CPU input checks even while training is incomplete")
    p.add_argument("--preflight-only", action="store_true", help="CPU input AND selected-checkpoint checks")
    new = p.parse_args(argv)
    a = base.arguments(["evaluate", "--prepare-only", "--training-root", str(new.training_root),
                        "--output-root", str(new.output_root)])
    a.input_only, a.preflight_only = new.prepare_only, new.preflight_only
    a.static_root = new.static_root
    a.variants = ["local", "graph"] if new.variant == "both" else [new.variant]
    out, prior = a.output_root.resolve(), a.static_root.resolve()
    if out == prior or out in prior.parents or prior in out.parents:
        raise ValueError("Evaluation output must not overlap the frozen static baseline")
    return a


def check_training(a):
    tm = base.checked(a.training_root / "manifest.json")
    tc = tm["contract"]
    if tc["version"] != "dynamic_kg_attention_v1": raise RuntimeError("Unsupported controller version")
    for name, expected in tc["code"].items():
        if sha_file(ROOT / name) != expected: raise RuntimeError(f"Frozen training/inference code changed: {name}")
    if tc["environment"] != base.versions(): raise RuntimeError("Training/evaluation environment differs")
    encoders = {}
    for kind, bundle in tc["encoders"].items():
        path = Path(bundle[0]["path"]).parent
        if bundle_identity(path) != bundle: raise RuntimeError(f"Frozen MedCPT {kind} changed")
        encoders[kind] = path
    selected, weights, pending = selected_controllers(a.training_root, tm, a.variants, load_torch,
                                                     allow_pending=a.input_only)
    return tm, selected, weights, pending, encoders


def prepare(a, progress):
    tm, selected, weights, pending, encoders = check_training(a)
    # Keep the original input/baseline validator; prepare_only prevents it from
    # trying to interpret dynamic checkpoints as static StructureBias weights.
    c = base.prepare(a, progress)
    tc = tm["contract"]
    for key in ("data_contract", "model", "frozen_lora", "shared_contract", "environment"):
        if c["contract"][key] != tc[key]: raise RuntimeError(f"Training/evaluation differs: {key}")
    sm = base.checked(a.static_root / "manifest.json")
    done = json.loads((a.static_root / "COMPLETE.json").read_text())
    if done["contract_sha256"] != sm["contract_sha256"] or done["n"] != 512: raise RuntimeError("Static baseline incomplete")
    for name, key in (("paired_predictions.jsonl", "paired_sha256"), ("summary.json", "summary_sha256")):
        if sha_file(a.static_root / name) != done[key]: raise RuntimeError("Static baseline file changed")
    for key in ("data_contract", "model", "frozen_lora", "shared_contract", "environment", "input",
                "system_prompt", "generation", "reference_contract"):
        if sm["contract"][key] != c["contract"][key]: raise RuntimeError(f"Static comparison mismatch: {key}")
    stored = list(jsonl(a.static_root / "paired_predictions.jsonl"))
    if [r["sample_id"] for r in stored] != [r["sample_id"] for r in c["rows"]]: raise RuntimeError("Static cohort order differs")
    static = []
    for row, prior in zip(c["rows"], stored, strict=True):
        item = prior["conditions"]["graph_attention"]
        if row["answer"] != prior["gold"] or row["dataset"] != prior["dataset"]: raise RuntimeError("Static labels differ")
        if item["prediction"] != parse_answer(item["output_text"], row["options"]): raise RuntimeError("Static parser differs")
        static.append({**item, "reused": True, "source_condition": "static_graph_attention"})
    c["baselines"]["static_attention"] = static
    progress.stage(2, len(c["rows"]), unit="q", kind="mapping")
    code = {name: sha_file(ROOT / name) for name in ("scripts/evaluate_rag2_dynamic_kg_attention.py",
                                                   "medrag/dynamic_kg_evaluation.py")}
    layout_hash = ensure_contract(a.output_root / "dynamic_layout_manifest.json", {
        "version": "dynamic_kg_evaluation_v1", "input": c["contract"]["input"], "code": code,
        "training_contract": tm["contract_sha256"], "training_code": tc["code"]})
    path, marker = a.output_root / "dynamic_layouts.jsonl", a.output_root / "DYNAMIC_LAYOUTS_COMPLETE.json"
    if marker.exists():
        m = json.loads(marker.read_text())
        if m["contract_sha256"] != layout_hash or m["sha256"] != sha_file(path): raise RuntimeError("Dynamic layouts changed")
        layouts = list(jsonl(path)); progress.stage(2, len(c["rows"]), initial=len(c["rows"]), unit="q")
    else:
        layouts = []
        for row in c["rows"]:
            layouts.append(source_layout(c["tokenizer"], row)); progress.update()
        atomic_jsonl(path, layouts)
        atomic_json(marker, {"contract_sha256": layout_hash, "sha256": sha_file(path)})
    if [r["sample_id"] for r in layouts] != [r["sample_id"] for r in c["rows"]]: raise RuntimeError("Dynamic layout order differs")
    plans = feature_plan(layouts)
    feature_root = a.output_root / "features"
    fh = ensure_contract(feature_root / "manifest.json", {"layout_contract": layout_hash,
        "encoders": tc["encoders"], "windows": tc["encoder_windows"], "plan_sha256": digest(plans)})
    atomic_json(feature_root / "plan.json", plans)
    contract = {"version": "dynamic_kg_evaluation_v1", "training_contract": tm["contract_sha256"],
        "selected": selected, "variants": a.variants, "architecture": tc["architecture"],
        "data_contract": c["contract"]["data_contract"], "model": c["model"], "frozen_lora": c["lora_id"],
        "input": c["contract"]["input"], "generation": c["generation"], "system_prompt": c["contract"]["system_prompt"],
        "environment": base.versions(), "code": code, "training_code": tc["code"],
        "feature_contract": fh, "layout_sha256": sha_file(path),
        "reference_contract": c["reference"]["contract_sha256"], "static_contract": sm["contract_sha256"],
        "static_predictions": identity(a.static_root / "paired_predictions.jsonl", full_hash=True),
        "primary": tc["primary_future_evaluation"], "scope": tc["scope"],
        "no_training": True, "selection": "training validation CE only; no checkpoint/threshold tuning on these 512 questions"}
    c.update(contract=contract, weights=weights, layouts={x["sample_id"]: x for x in layouts},
             plans=plans, encoders=encoders, feature_hash=fh, feature_root=feature_root, pending=pending)
    atomic_json(a.output_root / "INPUT_PREFLIGHT.json", {"n": len(c["rows"]), "gpu_executed": False,
        "pending_training": pending, "selected": selected, "datasets": dict(Counter(r["dataset"] for r in c["rows"])),
        "mapping": c["statistics"], "feature_texts": {k: len(v) for k, v in plans.items()}})
    progress.complete(f"same 512 prompts, model, parser and baselines; pending training={pending}")
    return c


@torch.no_grad()
def generate_batch(model, controller, piece, c):
    prompts = [r["text_prompt_ids"] for r in piece]
    width = max(map(len, prompts))
    controller.set_layouts([c["layouts"][r["sample_id"]] for r in piece], [width-len(p) for p in prompts])
    tok = c["tokenizer"]
    return generate_tokens(model, prompts, pad_id=tok.eos_token_id,
        eos_ids=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")], max_new_tokens=c["generation"]["max_new_tokens"])


def generate_retry(model, controller, piece, c, progress):
    micro = len(piece)
    while True:
        try:
            result = []
            for start in range(0, len(piece), micro): result.extend(generate_batch(model, controller, piece[start:start+micro], c))
            controller.clear()
            return result, micro
        except torch.cuda.OutOfMemoryError:
            controller.clear(); gc.collect(); torch.cuda.empty_cache()
            if micro == 1: raise
            micro = max(1, micro//2)
            progress.log(f"[OOM] generation micro={micro}; next durable shard restores batch 16")


@training_attention_backend()
def evaluate(a, c, run_hash, progress):
    rows = c["rows"]
    chunks = list(range(0, len(rows), 16))
    caches = {v: {start: read_shard(a.output_root / v / "predictions" / f"{start//16:04d}.json",
                                   run_hash, rows[start:start+16]) for start in chunks} for v in a.variants}
    all_cached = all(items is not None for cache in caches.values() for items in cache.values())
    if not all_cached:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
            raise RuntimeError("Use physical GPU 1 via the wrapper; CUDA unavailable/wrong selection")
        bank = encode_bank(c["feature_root"], c["plans"], c["encoders"], c["feature_hash"], progress,
                           {"query": 3, "article": 4}, 32, torch_save, load_torch)
        progress.stage(5, 1, unit="model")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        set_seed(c["generation"]["seed"])
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to("cuda:0")
        model = PeftModel.from_pretrained(model, c["lora"], is_trainable=False, local_files_only=True).get_base_model()
        model.requires_grad_(False); model.eval(); model.gradient_checkpointing_disable()
        progress.complete("same frozen Llama + medical LoRA; inference only")
    else:
        for i in (3, 4, 5):
            progress.stage(i, 1, initial=1); progress.complete("all predictions cached; skip GPU work")
    predictions = dict(c["baselines"])
    for index, variant in enumerate(a.variants):
        parity_path = a.output_root / variant / "ZERO_PARITY.json"
        controller = None
        if not all_cached:
            controller = DynamicBias(**c["contract"]["architecture"], variant=variant).to("cuda:0")
            controller.load_state_dict(c["weights"][variant]); controller.bind_bank(bank); controller.eval(); controller.attach(model)
        progress.stage(6+2*index, 16, initial=16 if parity_path.exists() else 0, unit="q")
        if parity_path.exists():
            p = json.loads(parity_path.read_text())
            if p["contract_sha256"] != run_hash or not p["passed"]: raise RuntimeError("Zero parity contract mismatch")
        elif all_cached:
            raise RuntimeError("Cached predictions lack zero-bias parity evidence")
        else:
            weight, bias = controller.output.weight.detach().clone(), controller.output.bias.detach().clone()
            try:
                with torch.no_grad(): controller.output.weight.zero_(); controller.output.bias.zero_()
                outputs, micro = generate_retry(model, controller, rows[:16], c, progress)
            finally:
                with torch.no_grad(): controller.output.weight.copy_(weight); controller.output.bias.copy_(bias)
            failures = [r["sample_id"] for r, out, old in zip(rows[:16], outputs, c["baselines"]["text"][:16], strict=True)
                        if out["generated_ids"] != old["generated_ids"]]
            atomic_json(parity_path, {"contract_sha256": run_hash, "n": 16, "passed": not failures,
                "mismatch_ids": failures, "actual_micro_batch": micro, "purpose": "backend check only, no tuning"})
            if failures: raise RuntimeError("Zero-bias tokens differ from cached Text; do not claim a controlled comparison")
            progress.update(16)
        progress.complete(f"{variant}: zero-bias TEXT parity verified")
        progress.stage(7+2*index, len(rows), initial=sum(len(x) for x in caches[variant].values() if x is not None),
                       unit="q", kind="generation")
        items = []
        for start in chunks:
            cached = caches[variant][start]
            if cached is not None: items.extend(cached); continue
            batch = rows[start:start+16]
            outputs, micro = generate_retry(model, controller, batch, c, progress)
            part = []
            for r, out in zip(batch, outputs, strict=True):
                text = c["tokenizer"].decode(out["generated_ids"], skip_special_tokens=True)
                part.append({"sample_id": r["sample_id"], "dataset": r["dataset"], **out, "output_text": text,
                    "prediction": parse_answer(text, r["options"]), "reused": False,
                    "prompt_tokens": len(r["text_prompt_ids"]), "actual_micro_batch": micro})
            atomic_json(a.output_root / variant / "predictions" / f"{start//16:04d}.json",
                        {"contract_sha256": run_hash, "rows": part, "rows_sha256": digest(part)})
            items.extend(part); progress.update(len(part))
            # Same operation/model/batch in both arms, never use training throughput.
            for future in range(progress.index+1, len(progress.names)):
                if progress.names[future].startswith("generate "):
                    progress.weights[future] = progress.weights[progress.index]
        predictions[variant] = items
        progress.complete(f"{variant}: 512 predictions saved; no refitting")
        if controller is not None: controller.clear()
    progress.stage(len(progress.names)-1, 1, unit="report")
    summary = paired_summary(rows, predictions, base.summarize)
    summary.update(contract_sha256=run_hash, selected=c["contract"]["selected"], mapping_statistics=c["statistics"])
    atomic_json(a.output_root / "summary.json", summary)
    atomic_jsonl(a.output_root / "paired_predictions.jsonl", ({"sample_id": r["sample_id"], "dataset": r["dataset"],
        "gold": r["answer"], "conditions": {name: {**values[i], "correct": values[i]["prediction"] == r["answer"]}
            for name, values in predictions.items()}} for i, r in enumerate(rows)))
    table = [{"condition": name, **{k: stats[k] for k in ("n", "correct", "micro_pct", "macro3_pct", "parse_failures")}}
             for name, stats in summary["conditions"].items()]
    atomic_json(a.output_root / "summary_table.json", table)
    atomic_json(a.output_root / "COMPLETE.json", {"contract_sha256": run_hash, "n": len(rows),
        "summary_sha256": sha_file(a.output_root / "summary.json"),
        "paired_sha256": sha_file(a.output_root / "paired_predictions.jsonl")})
    progress.update(); progress.complete(f"paired metrics/C->W/W->C/95% CIs: {a.output_root / 'summary.json'}")
    progress.log(f"[result] {json.dumps(table)}")


def main(argv=None):
    a = arguments(argv)
    names = ["CPU model/checkpoint contracts", "static token alignment", "dynamic source alignment"]
    if not (a.input_only or a.preflight_only):
        names += ["MedCPT query windows", "MedCPT article windows", "load frozen Llama"]
        for v in a.variants: names += [f"zero parity {v}", f"generate {v}"]
        names += ["paired summary"]
    progress = Progress(names)
    if names[-1] == "paired summary": progress.weights[-1] = 2.
    try:
        from transformers.utils import logging
        logging.set_verbosity_error(); logging.disable_progress_bar(); torch.set_num_threads(4)
        progress.stage(0, 1, unit="contract")
        progress.log("[plan] same 512 development questions; reuse No-RAG/TEXT/static; only new selected local/graph inference; GPU 1")
        progress.log("[ETA] new feature encoding/dynamic generation rates unmeasured; calibrate separately on fresh work")
        with lock_file(a.output_root / ".run.lock"):
            c = prepare(a, progress)
            if a.input_only:
                progress.log(f"[result] CPU input preflight passed; pending training={c['pending']}; no inference")
                return
            run_hash = ensure_contract(a.output_root / "manifest.json", c["contract"])
            if not (a.output_root / "provenance.json").exists():
                atomic_json(a.output_root / "provenance.json", {"contract_sha256": run_hash,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
            if a.preflight_only:
                progress.log("[result] CPU inputs/checkpoints verified; no inference")
                return
            evaluate(a, c, run_hash, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; "
            f"remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; "
            f"durable state={a.output_root}; identical command resumes, never deletes old results")
        raise
    finally: progress.close()


if __name__ == "__main__": main()

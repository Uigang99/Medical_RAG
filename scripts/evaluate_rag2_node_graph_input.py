#!/usr/bin/env python3
"""Evaluate the completed node-wise graph adapter on the fixed 512 MCQs.

Only the NEW node-graph condition is generated. Same-LLM No-RAG, TEXT and old
pooled-GRAPH predictions and all evidence/features are reused read-only. Never
replace the trained 4-layer/full-Top8 scope with question-selected 1-hop inputs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from medrag.node_graph_evaluation import BASELINES, NODE_CONDITION, generate_tokens, summarize
from medrag.text_graph_evaluation import GROUPS, parse_answer
from medrag.training.graph_input_data import (atomic_json, atomic_jsonl, digest, ensure_contract,
    graph_from_edges, identity, jsonl, sha_file)
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.training.node_graph_input import NodeGraphPrompt, node_prompt_batch
from train_rag2_node_graph_input import checked_manifest
from train_rag2_text_kg_input_pilot import bundle_identity, load_torch, lock_file, set_seed, torch_save


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, default=ROOT / "results/rag2_node_graph_input_v1/shared_both")
    p.add_argument("--reference-root", type=Path, default=ROOT / "results/rag2_text_graph_six_conditions_v1")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/pilots/rag2_text_kg_input_test_resplit_v1")
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_node_graph_input_evaluation_v1")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args(argv)


def verify_rows(rows, assignments, max_context, max_new_tokens):
    ids = [r["sample_id"] for r in rows]
    expected = {r["sample_id"] for r in assignments if r["split"] == "pilot_evaluation"}
    if len(ids) != len(set(ids)) or set(ids) != expected or len(ids) != 512:
        raise RuntimeError("Evaluation must use the SAME 512 pilot IDs; no train/validation/reserve examples")
    sizes, lengths = [], []
    for r in rows:
        if graph_from_edges(r["edges"]) != r["graph"]:
            raise RuntimeError(f"Frozen edge/node mapping changed: {r['sample_id']}")
        if len(r["document_ids"]) != 8 or len(set(r["document_ids"])) != 8:
            raise RuntimeError("Expected original Top-8 document list")
        if any(e["document_id"] not in r["document_ids"] for e in r["edges"]):
            raise RuntimeError("Graph evidence outside frozen Top-8")
        n = len(r["graph"]["node_ids"])
        size = len(r["graph_prompt_ids"]) + n + max_new_tokens
        if size > max_context:
            raise RuntimeError(f"Node context overflow at {r['sample_id']}: {size}>{max_context}; no silent truncation")
        sizes.append(n); lengths.append(size)
    return {"n": len(rows), "node_tokens_mean": sum(sizes)/len(sizes), "node_tokens_max": max(sizes),
            "max_generation_context": max(lengths), "empty_graphs": sizes.count(0),
            "groups": dict(Counter("mmlu_pooled" if r["dataset"].startswith("mmlu_") else r["dataset"] for r in rows))}


def preflight(args):
    import torch
    if args.batch_size < 1 or args.cpu_threads < 1: raise ValueError("Positive batch/CPU thread counts required")
    for root in (args.training_root, args.reference_root, args.data_root, args.model):
        a, b = args.output_root.resolve(), root.resolve()
        if a == b or a in b.parents or b in a.parents: raise ValueError("Output overlaps an immutable input")
    path = args.training_root / "COMPLETE.json"
    if not path.is_file(): raise RuntimeError(f"Node training not complete: {path}; smoke checkpoints are not evaluable")
    done = json.loads(path.read_text()); nm = checked_manifest(args.training_root / "manifest.json"); nc = nm["contract"]
    if (done["contract_sha256"] != nm["contract_sha256"] or nc["purpose"] != "node_graph_training_only" or
            done["epochs"] != nc["training"]["epochs"] or done["pilot_evaluation_questions_used"] != 0):
        raise RuntimeError("Node training incomplete or wrong data-use contract")
    best_path = args.training_root / "best.pt"
    if best_path.resolve() != Path(done["best_checkpoint"]).resolve() or sha_file(best_path) != done["best_checkpoint_sha256"]:
        raise RuntimeError("Selected node checkpoint changed")
    best = load_torch(best_path)
    if (best["contract_sha256"] != nm["contract_sha256"] or best["architecture"] != nc["architecture"] or
            best["validation_loss"] != done["best_validation_answer_ce"]):
        raise RuntimeError("Node checkpoint selection/architecture mismatch")
    if not all(torch.isfinite(v).all() for v in best["adapter"].values()):
        raise RuntimeError("Non-finite node encoder weights")
    history = json.loads((args.training_root / "history.json").read_text())
    if len(history) != done["epochs"] or best["epoch"] != min(history, key=lambda h:h["validation_answer_ce"])["epoch"]:
        raise RuntimeError("Best node checkpoint was not selected by validation CE")
    selected = next(h for h in history if h["epoch"] == best["epoch"])
    if (selected["validation_answer_ce"] != best["validation_loss"] or
            any(h["train_n"] != nc["counts"]["counts"]["train"] or
                h["validation_n"] != nc["counts"]["counts"]["validation"] for h in history)):
        raise RuntimeError("Training history counts/selected loss mismatch")
    if nc["architecture"]["class"] != "NodeGraphPrompt" or nc["architecture"]["global_pooling"]:
        raise RuntimeError("Expected the node-wise, non-pooled graph architecture")
    for rel, expected in nc["code"].items():
        if sha_file(ROOT / rel) != expected: raise RuntimeError(f"Node training implementation changed: {rel}")
    rm = checked_manifest(args.reference_root / "manifest.json"); rc = rm["contract"]
    rd = json.loads((args.reference_root / "COMPLETE.json").read_text())
    if (rd["contract_sha256"] != rm["contract_sha256"] or rd["n"] != 512 or rd["conditions"] != 6 or
            sha_file(args.reference_root / "paired_predictions.jsonl") != rd["paired_predictions_sha256"] or
            sha_file(args.reference_root / "summary.json") != rd["summary_sha256"]):
        raise RuntimeError("Completed baseline outputs corrupted or incompatible")
    dm = checked_manifest(args.data_root / "manifest.json")
    dc = json.loads((args.data_root / "COMPLETE.json").read_text())
    if nc["data_contract"] != dm["contract_sha256"] or rc["paired_data_contract"] != dm["contract_sha256"] or dc["contract_sha256"] != dm["contract_sha256"]:
        raise RuntimeError("Node training and evaluation use different data partitions")
    for name, expected in nc["data_files"].items():
        if dc["file_sha256"][name] != expected or sha_file(args.data_root / name) != expected:
            raise RuntimeError(f"Frozen training split changed: {name}")
    assignments = list(jsonl(args.data_root / "split_assignments.jsonl"))
    if len(assignments) != 6545 or len({r["sample_id"] for r in assignments}) != 6545:
        raise RuntimeError("Broken full split assignment")
    groups = {}
    for r in assignments: groups.setdefault(r["duplicate_group"], set()).add(r["split"])
    if any(len(v) != 1 for v in groups.values()): raise RuntimeError("Duplicate question group crosses splits")
    if bundle_identity(args.model) != nc["model"] or nc["model"] != rc["base_llm"]:
        raise RuntimeError("Use the EXACT frozen target Llama checkpoint")
    lora = None
    if nc["target_llm"] == "shared":
        if nc["frozen_lora"] != rc["lora"] or nc["shared_lora_contract"] != rc["training_contract"]:
            raise RuntimeError("Node encoder and baseline Llama use different LoRA")
        lora = Path(done["frozen_lora"])
        for entry in nc["frozen_lora"]:
            if identity(entry["path"], full_hash=True) != entry: raise RuntimeError("Frozen LoRA files changed")
        sources = {"no_rag": "6_trained_no_rag", "text": "3_trained_text", "pooled_graph": "5_trained_graph"}
    elif nc["target_llm"] == "base":
        if nc["frozen_lora"] is not None: raise RuntimeError("Base target unexpectedly has LoRA")
        sources = {"no_rag": "1_base_no_rag", "text": "2_base_text", "pooled_graph": "4_base_graph"}
    else: raise RuntimeError("Unknown target Llama")
    if nc["features"] != rc["feature_contract"] or nc["prompt"] != rc["system_prompt"]:
        raise RuntimeError("Encoder attributes or question prompt differ from cached baselines")
    for library in ("torch", "transformers", "peft"):
        version = importlib.metadata.version(library)
        if version != nc["environment"][library] or version != rc["environment"][library]:
            raise RuntimeError(f"Incompatible model runtime: {library}")
    marker = json.loads((args.reference_root / "PAIRS_COMPLETE.json").read_text())
    pairs_path = args.reference_root / "evaluation_pairs.jsonl"
    if sha_file(pairs_path) != marker["sha256"]: raise RuntimeError("Cached evaluation evidence changed")
    rows = list(jsonl(pairs_path))
    if digest([r["sample_id"] for r in rows]) != marker["ids_sha256"]: raise RuntimeError("Evaluation order changed")
    gen = rc["generation"]
    if gen["do_sample"] or gen["num_beams"] != 1: raise RuntimeError("Expected frozen greedy generation")
    if args.batch_size != gen["batch_size"]:
        raise ValueError(f"Use original batch size {gen['batch_size']}; automatic OOM splitting is supported")
    stats = verify_rows(rows, assignments, nc["max_context"], gen["max_new_tokens"])
    old = list(jsonl(args.reference_root / "paired_predictions.jsonl"))
    if [r["sample_id"] for r in old] != [r["sample_id"] for r in rows]:
        raise RuntimeError("Baseline answer rows are not aligned")
    baselines = {name: [] for name in BASELINES}
    for r, saved in zip(rows, old, strict=True):
        if r["answer"] != saved["gold"] or r["dataset"] != saved["dataset"]:
            raise RuntimeError("Gold/dataset mismatch in immutable baseline")
        for name, source in sources.items():
            item = saved["conditions"][source]
            if item["prediction"] != parse_answer(item["output_text"], r["options"]):
                raise RuntimeError("Baseline parser differs; no silent re-scoring")
            baselines[name].append({**item, "condition": name, "source_condition": source, "reused": True})
    feature_root = args.reference_root / "features"
    texts = json.loads((feature_root / "texts.json").read_text())
    expected = sorted({t for r in rows for t in r["graph"]["node_texts"]+r["graph"]["edge_texts"]})
    if texts != expected: raise RuntimeError("Evaluation feature vocabulary differs from graph")
    feature_files = [identity(feature_root / f"{i//512:05d}.pt", full_hash=True) for i in range(0, len(texts), 512)]
    contract = {"version": "node_graph_mcq_evaluation_v1", "hypothesis": "Node-wise graph input can replace pooled graph input without discarding node-level representations",
        "training_contract": nm["contract_sha256"], "node_checkpoint": identity(best_path, full_hash=True),
        "architecture": nc["architecture"], "selected_epoch": best["epoch"],
        "checkpoint_selection": "already selected minimum train-run validation CE; never select by pilot accuracy",
        "target_llm": nc["target_llm"], "model": nc["model"], "lora": nc["frozen_lora"],
        "reference_contract": rm["contract_sha256"], "reference_outputs": identity(args.reference_root / "paired_predictions.jsonl", full_hash=True),
        "paired_inputs": identity(pairs_path, full_hash=True), "data_contract": dm["contract_sha256"],
        "cohort": "same 512 pilot_evaluation questions, in original order", "statistics": stats,
        "baselines": sources, "new_condition": NODE_CONDITION, "features": feature_files,
        "input": "all node tokens after BOS, in frozen node-ID order; no text appended, no node selection",
        "scope": "same Top-8 edge union as training/baselines; no new neighbors, no 1-hop replacement",
        "message_passing": f"{nc['architecture']['layers']} trained layers; original edge roles preserved; no inverse KG facts",
        "system_prompt": rc["system_prompt"], "empty_graph": "zero node tokens, question retained as trained",
        "generation": gen, "max_context": nc["max_context"], "truncation": "none; overflow fails in CPU preflight",
        "parser": "same frozen strict parse_answer as cached six-condition evaluation; invalid counts as wrong, raw outputs saved",
        "metrics": "Macro-3 primary; micro, subgroups, paired C/W transitions, 2000 stratified paired bootstrap draws seed 42",
        "scale_up_gate": "same exploratory gate: node minus TEXT Macro-3 >=2pp and 95% CI lower >0",
        "claim_limit": "development TEST repartition; topology/readout/target alignment are confounded; no held-out tuning",
        "no_training": True,
        "code": {str(p.relative_to(ROOT)): sha_file(p) for p in (Path(__file__), ROOT / "medrag/node_graph_evaluation.py",
            ROOT / "medrag/training/node_graph_input.py", ROOT / "medrag/training/graph_input_modules.py",
            ROOT / "medrag/text_graph_evaluation.py", ROOT / "medrag/training/graph_input_data.py")},
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft")}}
    return {"contract": contract, "rows": rows, "baselines": baselines, "weights": best["adapter"],
            "model": args.model, "lora": lora, "feature_root": feature_root, "texts": texts}


def load_features(context, progress):
    import torch
    texts = context["texts"]; dim = context["contract"]["architecture"]["input_dim"]
    progress.stage(1, len(texts), unit="text")
    features = torch.empty((len(texts), dim), dtype=torch.float32)
    for i in range(0, len(texts), 512):
        item = load_torch(context["feature_root"] / f"{i//512:05d}.pt")
        n = min(512, len(texts)-i); values = item["values"]
        if (item["text_sha256"] != digest(texts[i:i+n]) or values.shape != (n, dim) or
                values.dtype != torch.float32 or not torch.isfinite(values).all()):
            raise RuntimeError(f"Invalid cached evaluation features at {i}")
        features[i:i+n].copy_(values); progress.update(n)
    progress.complete(f"{len(texts):,} frozen features validated; no embedding generation; {context['feature_root']}")
    return features, {t: i for i, t in enumerate(texts)}


def validate_node_cache(items, rows, llm_dim):
    import torch
    if [x["sample_id"] for x in items] != [r["sample_id"] for r in rows]:
        raise RuntimeError("Node cache question order mismatch")
    for item, row in zip(items, rows, strict=True):
        g = row["graph"]; values = item["values"]
        if (item["node_ids"] != g["node_ids"] or item["graph_sha256"] != digest(g) or
                values.shape != (len(g["node_ids"]), llm_dim) or values.dtype != torch.float32 or
                not torch.isfinite(values).all()):
            raise RuntimeError(f"Node vector alignment/scope mismatch: {row['sample_id']}")


def encode_nodes(args, context, features, lookup, contract_hash, progress):
    import torch
    rows = context["rows"]; arch = context["contract"]["architecture"]
    chunks = [(i, args.output_root / "node_tokens" / f"{i//32:05d}.pt") for i in range(0, len(rows), 32)]
    cached = sum(min(32, len(rows)-i) for i, p in chunks if p.exists())
    progress.stage(2, 1, initial=int(cached == len(rows)), unit="model")
    encoder = None
    if cached != len(rows):
        encoder = NodeGraphPrompt(arch["input_dim"], arch["llm_dim"], arch["hidden_dim"],
                        arch["layers"], arch["heads"], arch["bidirectional"]).to("cuda:0")
        encoder.load_state_dict(context["weights"]); encoder.eval(); encoder.requires_grad_(False)
    progress.complete("node encoder loaded" if encoder is not None else "all node vectors cached; encoder load skipped")
    progress.stage(3, len(rows), initial=cached, unit="q", kind="node_encoding")
    result = {}
    with torch.inference_mode():
        for i, path in chunks:
            batch = rows[i:i+32]
            if path.exists():
                saved = load_torch(path)
                if saved["contract_sha256"] != contract_hash: raise RuntimeError("Node shard contract mismatch")
                items = saved["rows"]
            else:
                items = []
                for row in batch:
                    z = node_prompt_batch(encoder, [row], features, lookup, "cuda:0")[0].cpu()
                    items.append({"sample_id": row["sample_id"], "node_ids": row["graph"]["node_ids"],
                                  "graph_sha256": digest(row["graph"]), "values": z})
                validate_node_cache(items, batch, arch["llm_dim"])
                torch_save(path, {"contract_sha256": contract_hash, "rows": items})
                progress.update(len(batch))
            validate_node_cache(items, batch, arch["llm_dim"])
            result.update((item["sample_id"], item["values"]) for item in items)
    progress.complete(f"node vectors for {len(rows)} questions; IDs/scope retained; {args.output_root / 'node_tokens'}")
    del encoder
    context.pop("weights", None)
    gc.collect(); torch.cuda.empty_cache()
    return result


def read_prediction_shard(path, ids, contract_hash):
    value = json.loads(path.read_text())
    if value["contract_sha256"] != contract_hash or [r["sample_id"] for r in value["rows"]] != ids:
        raise RuntimeError("Prediction shard contract/ID mismatch")
    return value["rows"]


@training_attention_backend()
def generate(args, context, vectors, contract_hash, progress):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error(); logging.disable_progress_bar()
    rows = context["rows"]; settings = context["contract"]["generation"]
    chunks = [(i, args.output_root / "predictions" / f"{i//args.batch_size:05d}.json")
              for i in range(0, len(rows), args.batch_size)]
    cached = {i: read_prediction_shard(p, [r["sample_id"] for r in rows[i:i+args.batch_size]], contract_hash)
              for i, p in chunks if p.exists()}
    all_cached = sum(map(len, cached.values())) == len(rows)
    progress.stage(4, 1, initial=int(all_cached), unit="model")
    llm = None
    if not all_cached:
        tokenizer = AutoTokenizer.from_pretrained(context["model"], local_files_only=True)
        llm = AutoModelForCausalLM.from_pretrained(context["model"], local_files_only=True,
                     dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        if context["lora"]:
            from peft import PeftModel
            wrapper = PeftModel.from_pretrained(llm, context["lora"], is_trainable=False, local_files_only=True)
            llm = wrapper.get_base_model()
        llm.requires_grad_(False); llm.eval(); llm.gradient_checkpointing_disable()
        eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        if None in eos: raise RuntimeError("Missing frozen Llama stop token")
    progress.complete("same frozen target Llama loaded" if not all_cached else "all predictions cached; Llama load skipped")
    progress.weights[6] = 1.  # Explicit small CPU summary cost estimate, replaced by measured duration.
    progress.stage(5, len(rows), initial=sum(map(len, cached.values())), unit="q", kind="generation")
    result = []
    for start, path in chunks:
        batch = rows[start:start+args.batch_size]
        if start in cached:
            result.extend(cached[start]); continue
        micro = len(batch)
        while True:
            try:
                outputs = []
                for i in range(0, len(batch), micro):
                    piece = batch[i:i+micro]
                    outputs.extend(generate_tokens(llm, [r["graph_prompt_ids"] for r in piece],
                        [vectors[r["sample_id"]] for r in piece], pad_id=tokenizer.eos_token_id,
                        eos_ids=eos, max_new_tokens=settings["max_new_tokens"]))
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if micro == 1: raise
                micro = max(1, micro//2)
                progress.log(f"[OOM] batch={start}; retry micro={micro}; restore {args.batch_size} next batch")
        items = []
        for row, out in zip(batch, outputs, strict=True):
            text = tokenizer.decode(out["generated_ids"], skip_special_tokens=True)
            n = len(row["graph"]["node_ids"])
            items.append({"sample_id": row["sample_id"], "dataset": row["dataset"], "condition": NODE_CONDITION,
                **out, "output_text": text, "prediction": parse_answer(text, row["options"]),
                "node_tokens": n, "prompt_tokens": len(row["graph_prompt_ids"])+n,
                "actual_micro_batch": micro, "reused": False})
        atomic_json(path, {"contract_sha256": contract_hash, "rows": items})
        result.extend(items); progress.update(len(batch))
    atomic_jsonl(args.output_root / "node_predictions.jsonl", result)
    progress.complete(f"new generations={len(rows)-sum(map(len, cached.values()))}; total={len(result)}; {args.output_root / 'node_predictions.jsonl'}")
    return result


def report(args, context, new_predictions, contract_hash, progress):
    progress.stage(6, 1, unit="report")
    rows = context["rows"]; predictions = {**context["baselines"], NODE_CONDITION: new_predictions}
    summary = summarize(rows, predictions)
    summary.update(contract_sha256=contract_hash, target_llm=context["contract"]["target_llm"],
                   baseline_sources=context["contract"]["baselines"], evidence_statistics=context["contract"]["statistics"])
    atomic_json(args.output_root / "summary.json", summary)
    by_id = {name: {p["sample_id"]: p for p in values} for name, values in predictions.items()}
    atomic_jsonl(args.output_root / "paired_predictions.jsonl", ({"sample_id": r["sample_id"], "dataset": r["dataset"],
        "gold": r["answer"], "conditions": {n: {**v[r["sample_id"]], "correct": v[r["sample_id"]]["prediction"] == r["answer"]}
                                             for n, v in by_id.items()}} for r in rows))
    path = args.output_root / "summary.csv"; temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n", "correct", "parse_failures", "micro_pct", "macro3_pct", *GROUPS])
        for name in BASELINES + (NODE_CONDITION,):
            s = summary["conditions"][name]
            writer.writerow([name, s["n"], s["correct"], s["parse_failures"], s["micro_pct"], s["macro3_pct"],
                             *[s["groups"][g]["accuracy_pct"] for g in GROUPS]])
    os.replace(temp, path)
    atomic_json(args.output_root / "COMPLETE.json", {"contract_sha256": contract_hash, "n": len(rows),
        "new_conditions": 1, "cached_baseline_conditions": 3, "summary_sha256": sha_file(args.output_root / "summary.json"),
        "paired_predictions_sha256": sha_file(args.output_root / "paired_predictions.jsonl")})
    progress.update(); progress.complete(f"paired accuracies, transitions and CIs saved; {path}")
    s = summary["conditions"][NODE_CONDITION]
    progress.log(f"[result] node GRAPH n={s['n']}; Micro={s['micro_pct']:.2f}%; Macro-3={s['macro3_pct']:.2f}%; parse failures={s['parse_failures']}")


def main(argv=None):
    args = arguments(argv)
    names = ["verify training + pilot", "read cached attributes", "load node encoder", "encode node tokens",
             "load frozen Llama", "MCQ node GRAPH", "paired summary"]
    if args.preflight_only: names = names[:2]
    progress = Progress(names)
    try:
        import torch
        torch.set_num_threads(max(1, args.cpu_threads))
        progress.stage(0, 1, unit="contract")
        progress.log("[plan] completed node encoder -> SAME 512 pilot questions/features -> all node tokens -> "
                     "one new greedy MCQ condition -> compare 3 cached baselines; NO training/retrieval")
        progress.log("[ETA] new node-vector/generation rates calibrate on fresh batches; cached baselines generate zero answers; GPU 1 only")
        context = preflight(args)
        progress.complete(f"best epoch={context['contract']['selected_epoch']}; {json.dumps(context['contract']['statistics'])}")
        with lock_file(args.output_root / ".run.lock"):
            contract_hash = ensure_contract(args.output_root / "manifest.json", context["contract"])
            if (args.output_root / "COMPLETE.json").exists():
                done = json.loads((args.output_root / "COMPLETE.json").read_text())
                if (done["contract_sha256"] != contract_hash or done["summary_sha256"] != sha_file(args.output_root / "summary.json") or
                        done["paired_predictions_sha256"] != sha_file(args.output_root / "paired_predictions.jsonl")):
                    raise RuntimeError("Evaluation completion mismatch")
                progress.log("[already complete] no regeneration"); return
            features, lookup = load_features(context, progress)
            if args.preflight_only:
                a = context["contract"]["architecture"]
                encoder = NodeGraphPrompt(a["input_dim"], a["llm_dim"], a["hidden_dim"],
                                          a["layers"], a["heads"], a["bidirectional"]).eval()
                encoder.load_state_dict(context["weights"])
                with torch.inference_mode():
                    first = context["rows"][0]
                    values = node_prompt_batch(encoder, [first], features, lookup, "cpu")[0]
                    if not torch.isfinite(values).all(): raise RuntimeError("Non-finite trained node encoder output")
                atomic_json(args.output_root / "PREFLIGHT.json", {"contract_sha256": contract_hash,
                    "statistics": context["contract"]["statistics"], "gpu_generation_executed": False,
                    "cpu_encoder_check": {"sample_id": first["sample_id"], "shape": list(values.shape), "finite": True},
                    "cached_baseline_counts": {n: len(v) for n, v in context["baselines"].items()}})
                progress.log("[preflight passed] node checkpoint/LoRA/split/source/features/context/baselines and one actual CPU node encoding validated; no LLM/GPU generation")
                return
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
                raise RuntimeError("Use the wrapper on the GPU server with physical GPU 1")
            with lock_file(args.training_root / ".training.lock"):
                if not (args.output_root / "provenance.json").exists():
                    atomic_json(args.output_root / "provenance.json", {"git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "gpu": torch.cuda.get_device_name(0)})
                set_seed(context["contract"]["generation"]["seed"])
                vectors = encode_nodes(args, context, features, lookup, contract_hash, progress)
                del features, lookup
                results = generate(args, context, vectors, contract_hash, progress)
                report(args, context, results, contract_hash, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)}, item={progress.done}/{progress.total}, "
                     f"remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}. "
                     f"Durable node_tokens/predictions shards: {args.output_root}; identical command resumes the same contract.")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

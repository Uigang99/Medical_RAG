#!/usr/bin/env python3
"""Matched PCED CoT versus PCED CoT with temporary joint-document experts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_loft_pced_paper as paper
from scripts import evaluate_loft_pced_pass as io
from scripts.evaluate_loft_pced_pass_cot import load_reference, PilotProgress
from medrag.generation import pced_joint as method

VERSION = "loft_temporary_joint_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, default=ROOT/"results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--limit-per-dataset", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="+", choices=("pced", "joint"), default=["pced", "joint"])
    p.add_argument("--reasoning-tokens", type=int, default=64)
    p.add_argument("--answer-tokens", type=int, default=32)
    p.add_argument("--joint-warmup", type=int, default=8)
    p.add_argument("--joint-check-every", type=int, default=8)
    p.add_argument("--joint-lifetime", type=int, default=32)
    p.add_argument("--joint-max-creations", type=int, default=2)
    p.add_argument("--joint-relevance-ratio", type=float, default=.25)
    p.add_argument("--joint-min-effect-jsd", type=float, default=.001)
    p.add_argument("--joint-min-pair-jsd", type=float, default=.05)
    p.add_argument("--joint-prior", choices=("geometric", "arithmetic", "minimum"), default="geometric")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.limit_per_dataset <= 100 or not 1 <= a.reasoning_tokens <= 512 or not 1 <= a.answer_tokens <= 128:
        p.error("Invalid cohort size or generation budget")
    a.joint_config = method.JointConfig(warmup=a.joint_warmup, check_every=a.joint_check_every,
        lifetime=a.joint_lifetime, max_creations=a.joint_max_creations,
        relevance_ratio=a.joint_relevance_ratio, min_effect_jsd=a.joint_min_effect_jsd,
        min_pair_jsd=a.joint_min_pair_jsd, prior_method=a.joint_prior)
    a.reference_root = a.reference_root.resolve()
    a.output_root = (a.output_root or ROOT/"results"/f"loft_pced_joint_expert_v1_n{5*a.limit_per_dataset}").resolve()
    if (a.output_root == a.reference_root or a.output_root in a.reference_root.parents or
            a.reference_root in a.output_root.parents):
        p.error("Output must be separate from the immutable reference")
    a.arms = list(dict.fromkeys(a.arms))
    return a


def prompts(tokenizer, question, texts):
    prefixes, suffix = paper.prompt_inputs(tokenizer, question, texts)
    suffix = suffix.replace(paper.QUESTION.format(question=question), method.QUESTION.format(question=question), 1)
    if not suffix.endswith(paper.ANSWER_PREFIX):
        raise RuntimeError("Unexpected paper prompt boundary")
    return prefixes, suffix[:-len(paper.ANSWER_PREFIX)] + method.PREFIX


def preflight(a, progress):
    from transformers import AutoTokenizer
    rows, references, model_path, revision = load_reference(a)  # immutable input selection only, no PASS prompts
    index = model_path/"model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    for name in [*shards, "config.json", "tokenizer.json", "tokenizer_config.json"]:
        if not (model_path/name).is_file():
            raise FileNotFoundError(model_path/name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    capacity = json.loads((model_path/"config.json").read_text())["max_position_embeddings"]
    reserve = a.reasoning_tokens + a.answer_tokens + len(tokenizer.encode(method.ANSWER_MARKER, add_special_tokens=False))
    lengths = []
    for row in rows:
        prefixes, suffix = prompts(tokenizer, row["question"], row["texts"])
        row["context_ids"] = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        row["suffix_ids"] = tokenizer.encode(suffix, add_special_tokens=False)
        # A conservative bound; actual selected joint prompts are checked again.
        length = 2 * max(map(len, row["context_ids"])) + len(row["suffix_ids"]) + reserve + 16
        if length > capacity:
            raise RuntimeError(f"Potential joint context overflow (no truncation): {row['qid']}, {length}>{capacity}")
        lengths.append(length)
        progress.update()
    code = [Path(__file__), Path(method.__file__), Path(paper.__file__), Path(io.__file__),
            ROOT/"scripts/evaluate_loft_pced_pass_cot.py", ROOT/"medrag/training/graph_input_progress.py",
            ROOT/"cache/pced_official/src/methods/pced.py"]
    contract = {
        "version": VERSION, "n": len(rows), "arms": a.arms, "seed": a.seed,
        "hypothesis": "A temporary joint-document expert can resolve document-induced prediction differences during shared CoT",
        "split": "previously inspected LOFT TEST diagnostic; not held-out selection or generalization evidence",
        "cohort": [{"dataset": r["dataset"], "qid": r["qid"], "source_sha256": r["source_sha256"]} for r in rows],
        "references": references, "reference_root": str(a.reference_root), "official_commit": revision,
        "model": str(model_path), "model_config_sha256": paper.sha(model_path/"config.json"),
        "model_shards": {s: {"bytes": (model_path/s).stat().st_size, "mtime_ns": (model_path/s).stat().st_mtime_ns} for s in shards},
        "tokenizer_sha256": paper.sha(model_path/"tokenizer.json"),
        "tokenizer_config_sha256": paper.sha(model_path/"tokenizer_config.json"),
        "prompt": {"system": paper.SYSTEM, "question": method.QUESTION, "prefix": method.PREFIX,
                   "forced_answer_marker": method.ANSWER_MARKER, "joint_separator": "\n\n"},
        "retrieval": "same cached CE Top90 order and raw documents; no retrieval, filtering, or truncation changes",
        "priors": "paper BGE/CE harmonic prior for originals; joint combines TWO FINAL priors, never raw scores",
        "joint": asdict(a.joint_config),
        "selection": "all eligible unused pairs; FP32 scalar JSD(softmax(z_i-z0),softmax(z_j-z0)); block16 full vocabulary",
        "effect_filter": "FP32 scalar JSD(softmax(z_i),softmax(z0)) in nats",
        "scoring": "pinned PCED elementwise JSD beta, frozen at first position; gamma2.5; expert-first tie order",
        "joint_beta": "pinned elementwise beta versus current amateur at activation; frozen for that lifetime",
        "joint_lifecycle": "one active; no repeated pair; creations only in reasoning; lifetime counts sampled reasoning/answer positions",
        "history": "same accepted token IDs and forced delimiter in ALL streams; EOS not reinserted; no rollback/branch/residual scoring",
        "generation": {"greedy": True, "backend": "sdpa", "dtype": "bfloat16", "physical_gpu": 1,
                       "reasoning_tokens": a.reasoning_tokens, "answer_tokens": a.answer_tokens},
        "parser": "decode answer segment only; empty=wrong; no reasoning-field EM credit",
        "phase_protocol": "reasoning EOS/budget then forced newline Answer:; answer EOS/budget; same both arms",
        "metric": "paper Subspan EM; Macro-5 primary, Micro, per-dataset and paired correct/wrong transitions",
        "pilot_gate": "diagnostic only; expand only after held-out pilot shows >=2pp Macro-5, gains>harms, <=1.5x latency and no format regression; not tuned on these TEST rows",
        "code_sha256": {str(p.relative_to(ROOT)): paper.sha(p) for p in code},
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
    }
    manifest = {"contract": contract, "contract_hash": io.digest(contract),
                "max_context_bound": max(lengths),
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    path = a.output_root/"manifest.json"
    if path.exists() and json.loads(path.read_text()).get("contract_hash") != manifest["contract_hash"]:
        raise RuntimeError("Existing output has a different contract; choose a new --output-root")
    if not path.exists() and any(a.output_root.glob("*/predictions/*/*.json")):
        raise RuntimeError("Predictions without manifest: refuse unsafe resume")
    return rows, tokenizer, model_path, manifest


def summarize(a, rows, h):
    records, conditions = {}, {}
    csv = ["dataset,arm,n,correct,subspan_em,empty_answer,joint_activated,mean_seconds"]
    for arm in a.arms:
        values = [io.cached_record(a, arm, row, h) for row in rows]
        if any(v is None for v in values):
            raise RuntimeError(f"Incomplete {arm}; no partial score presented as complete")
        records[arm] = values
        groups = {}
        for dataset in paper.DATASETS:
            subset = [r for r in values if r["dataset"] == dataset]
            n = len(subset)
            g = {"n": n, "correct": sum(r["subspan_em"] for r in subset),
                 "empty_answer": sum(r["empty_answer"] for r in subset),
                 "joint_activated": sum(bool(r["joint_events"]) for r in subset),
                 "mean_seconds": sum(r["seconds"] for r in subset)/n}
            g["subspan_em"] = g["correct"]/n
            groups[dataset] = g
            csv.append(f"{dataset},{arm},{n},{g['correct']},{g['subspan_em']},{g['empty_answer']},{g['joint_activated']},{g['mean_seconds']}")
        traces = [t for r in values for t in r["trace"]]
        conditions[arm] = {"n": len(values), "missing": 0, "datasets": groups,
            "correct": sum(r["subspan_em"] for r in values),
            "micro": sum(r["subspan_em"] for r in values)/len(values),
            "macro5": sum(g["subspan_em"] for g in groups.values())/len(groups),
            "empty_answer": sum(r["empty_answer"] for r in values),
            "empty_reasoning": sum(not r["reasoning_text"] for r in values),
            "reasoning_budget_hits": sum(r["reasoning_status"] == "token_budget" for r in values),
            "answer_budget_hits": sum(r["answer_status"] == "token_budget" for r in values),
            "joint_activated_questions": sum(bool(r["joint_events"]) for r in values),
            "joint_creations": sum(len(r["joint_events"]) for r in values),
            "sampled_positions_including_eos": len(traces),
            "joint_wins": sum(t["winner_kind"] == "joint" for t in traces),
            "changed_tokens_same_prefix": sum(t["changed_token_same_prefix"] for t in traces),
            "mean_seconds": sum(r["seconds"] for r in values)/len(values)}
    paired = None
    if "pced" in records and "joint" in records:
        pairs = [(bool(x["subspan_em"]), bool(y["subspan_em"])) for x, y in zip(records["pced"], records["joint"])]
        cw, wc = sum(x and not y for x, y in pairs), sum(not x and y for x, y in pairs)
        n = len(pairs)
        # Stratified paired bootstrap: preserve each dataset's Macro-5 weight.
        rng = random.Random(17)
        strata = [[int(y["subspan_em"])-int(x["subspan_em"]) for x, y in zip(records["pced"], records["joint"])
                   if x["dataset"] == ds] for ds in paper.DATASETS]
        draws = sorted(sum(sum(rng.choices(s, k=len(s)))/len(s) for s in strata)/len(strata) * 100 for _ in range(2000))
        paired = {"n": n, "baseline": "pced", "condition": "joint", "c_to_w": cw, "w_to_c": wc,
                  "c_to_c": sum(x and y for x, y in pairs), "w_to_w": sum(not x and not y for x, y in pairs),
                  "c_to_w_fraction_all": cw/n, "w_to_c_fraction_all": wc/n,
                  "harm_rate_baseline_correct": cw/max(1, sum(x for x, _ in pairs)),
                  "recovery_rate_baseline_wrong": wc/max(1, sum(not x for x, _ in pairs)),
                  "macro5_delta_pp": 100*(conditions["joint"]["macro5"]-conditions["pced"]["macro5"]),
                  "macro5_delta_bootstrap95_pp": [draws[50], draws[1949]],
                  "latency_ratio": conditions["joint"]["mean_seconds"]/conditions["pced"]["mean_seconds"]}
    summary = {"contract_hash": h, "conditions": conditions, "paired": paired,
               "interpretation": "Inspected TEST diagnostic only. Same-prefix token changes are not causal correctness attribution."}
    paper.atomic_json(a.output_root/"summary.json", summary)
    path = a.output_root/"summary.csv"
    temp = path.with_suffix(".csv.tmp")
    temp.write_text("\n".join(csv)+"\n", encoding="utf-8")
    temp.replace(path)
    return summary


def main(argv=None):
    a = arguments(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("Physical GPU 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_warning()
    hf_logging.disable_progress_bar()
    n = 5*a.limit_per_dataset
    names = ["contract/tokenize", "load Llama"] + [f"{arm} LOFT TEST" for arm in a.arms] + ["paired summary"]
    progress = PilotProgress(names)
    progress.weights[-1] = 1.
    active = "preflight"
    try:
        progress.stage(0, n, unit="q")
        rows, tokenizer, model_path, manifest = preflight(a, progress)
        h = manifest["contract_hash"]
        counts = {}
        for stage, arm in enumerate(a.arms, 2):
            cached = [io.cached_record(a, arm, row, h) for row in rows]
            counts[arm] = sum(r is not None for r in cached)
            progress.cached_fractions[stage] = counts[arm]/n
            times = [r["seconds"] for r in cached if r is not None]
            if times:
                progress.weights[stage] = sum(times)/len(times)*n
        progress.complete(f"N={n}; unchanged Top90; cached={counts}; {a.output_root}")
        progress.log(f"[plan] contract -> frozen Llama -> {' -> '.join(a.arms)} -> summary; remaining={sum(n-v for v in counts.values())}; GPU1")
        progress.log("[scope] shared original CoT; no PASS/residual/rollback/training; inspected TEST diagnostic, not tuning")
        progress.log("[ETA] New matched CoT/joint rate unavailable; first 5 fresh questions per arm calibrate separately. Overall ETA stays unknown until both arm costs are known.")
        if a.dry_run:
            progress.log("[dry-run OK] data/model/tokenizer/context/cache contracts; no model weights, GPU allocation, or outputs")
            return
        if not (a.output_root/"manifest.json").exists():
            paper.atomic_json(a.output_root/"manifest.json", manifest)
        progress.stage(1, 1)
        model = None
        if any(count < n for count in counts.values()):
            import torch
            from transformers import AutoModelForCausalLM
            paper.gpu_gate(40)
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(a.seed)
            torch.manual_seed(a.seed)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        progress.complete("one frozen Llama, 91 batched original streams + at most one joint KV cache")
        sys.path.insert(0, str(ROOT/"cache/pced_official"))
        from src.methods.pced import prefill, get_jsd
        for stage, arm in enumerate(a.arms, 2):
            progress.stage(stage, n, initial=counts[arm], unit="q")
            times = []
            for row in rows:
                active = f"{arm}/{row['dataset']}/{row['qid']}"
                if io.cached_record(a, arm, row, h) is not None:
                    continue
                started = time.monotonic()
                context, suffix = io.batch_inputs(row, tokenizer, model.device)
                prior = paper.prior_tensor(row["scores"], "paper", model.device, model.dtype)
                cache = prefill(model, context)
                builder = (method.make_joint_builder(model, tokenizer, row, prompts, context["input_ids"].shape[1])
                           if arm == "joint" else None)
                stream = method.JointStream(model, cache, context, suffix, prior, get_jsd,
                                            config=a.joint_config, joint_builder=builder)
                result = method.generate(stream, tokenizer, reasoning_tokens=a.reasoning_tokens, answer_tokens=a.answer_tokens)
                # Labels are read ONLY here, after generation has finished.
                result.update(contract_hash=h, arm=arm, dataset=row["dataset"], qid=row["qid"],
                    question=row["question"], answers=row["answers"],
                    subspan_em=paper.subspan_em(result["prediction"], row["answers"]),
                    top90_pids=row["scores"]["pids"], retrieval_prior=prior.float().cpu().tolist(),
                    seconds=time.monotonic()-started)
                paper.atomic_json(io.output_path(a, arm, row), result)
                times.append(result["seconds"])
                del stream, cache, context, suffix, prior, result, builder
                progress.update()
                if len(times) >= 5:
                    progress.weights[stage] = sum(times)/len(times)*n
            progress.complete(f"{arm}={n}/{n}; {a.output_root/arm}")
        progress.stage(len(names)-1, 1)
        summary = summarize(a, rows, h)
        progress.complete(str(a.output_root/"summary.json"))
        progress.log("[result] "+"; ".join(f"{arm}: {v['correct']}/{n}, Macro-5={v['macro5']:.4f}" for arm, v in summary["conditions"].items()))
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={active}; completed={progress.done}/{progress.total}; remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; durable={a.output_root}; identical command resumes committed questions")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

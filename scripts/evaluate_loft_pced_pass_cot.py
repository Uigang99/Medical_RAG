#!/usr/bin/env python3
"""Four controlled short-CoT/PASS scoring arms on immutable LOFT retrieval caches."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_loft_pced_pass as legacy
from scripts import run_loft_pced_paper as paper
from medrag.generation import pced_pass_cot as method
from medrag.generation.pced_pass import pass_token_ids
from medrag.training.graph_input_progress import Progress, duration

VERSION = "loft_pass_ratio_cot_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, default=ROOT/"results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--limit-per-dataset", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="+", choices=method.ARMS, default=list(method.ARMS))
    p.add_argument("--pass-floor", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--step-tokens", type=int, default=32)
    p.add_argument("--answer-tokens", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.limit_per_dataset <= 100 or not 1 <= a.max_steps <= 8:
        p.error("limit-per-dataset must be 1..100; max-steps must be 1..8")
    if not 8 <= a.step_tokens <= 128 or not 8 <= a.answer_tokens <= 128:
        p.error("Token budgets must be 8..128")
    if not math.isfinite(a.pass_floor) or not 0 < a.pass_floor < 1:
        p.error("pass-floor must be finite and in (0,1)")
    a.reference_root = a.reference_root.resolve()
    a.output_root = (a.output_root or ROOT/"results"/f"loft_llama_pass_ratio_cot_v1_n{5*a.limit_per_dataset}").resolve()
    if (a.output_root == a.reference_root or a.output_root in a.reference_root.parents or
            a.reference_root in a.output_root.parents):
        p.error("Output must be separate from immutable references")
    a.arms = list(dict.fromkeys(a.arms))
    return a


def prompts(tokenizer, question, texts):
    prefixes, suffix = [], None
    sentinel = "<PCED_QUESTION_BOUNDARY_42>"
    for index, text in enumerate(["", *texts]):
        if sentinel in text or sentinel in question:
            raise ValueError("Prompt boundary collision")
        rendered = tokenizer.apply_chat_template([
            {"role": "system", "content": method.SYSTEM + ("\n"+method.PASS_RULES if index else "")},
            {"role": "user", "content": "Passage:\n"+text+"\n\n"+sentinel}],
            tokenize=False, add_generation_prompt=True)
        prefix, trailing = rendered.split(sentinel)
        current = method.QUESTION.format(question=question) + trailing + method.PREFIX
        if suffix is not None and suffix != current:
            raise RuntimeError("Inconsistent expert/amateur suffix")
        prefixes.append(prefix)
        suffix = current
    return prefixes, suffix


def load_reference(a):
    """Same cohort/source checks as direct PASS, without modifying its frozen code."""
    revision = subprocess.check_output(["git", "-C", str(ROOT/"cache/pced_official"), "rev-parse", "HEAD"], text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError("Official PCED revision changed")
    rows, references, model_path = [], {}, None
    for dataset in paper.DATASETS:
        folder = a.reference_root/dataset
        ref = json.loads((folder/"manifest.json").read_text())
        summary = json.loads((folder/"summary.json").read_text())
        expected = {"top_k": 90, "max_new_tokens": 32, "candidate_mode": "all",
                    "score_policy": "paper", "attention_impl": "sdpa", "questions": 100}
        if any(ref.get(k) != v for k, v in expected.items()) or summary.get("n") != 100 or summary.get("missing") != 0:
            raise RuntimeError(f"Incompatible source experiment: {dataset}")
        current_model = Path(ref["llama"])
        if current_model.name != "Llama-3.1-8B-Instruct" or model_path not in (None, current_model):
            raise RuntimeError("Expected same frozen reference Llama-3.1-8B")
        model_path = current_model
        source = paper.arguments(["--phase", "preflight", "--dataset", dataset])
        cp, qp, corpus, queries = paper.load_inputs(source)
        if paper.sha(cp) != ref["corpus_sha256"] or paper.sha(qp) != ref["queries_sha256"]:
            raise RuntimeError(f"Source data changed: {dataset}")
        by_pid = {str(doc["pid"]): doc for doc in corpus}
        selected = sorted(queries, key=lambda q: legacy.digest([a.seed, dataset, str(q["qid"])]))[:a.limit_per_dataset]
        references[dataset] = {"manifest_sha256": paper.sha(folder/"manifest.json"),
                               "question_sha256": paper.sha(qp), "corpus_sha256": paper.sha(cp)}
        for q in selected:
            qid = str(q["qid"])
            paths = [paper.result_file(folder/x, qid) for x in ("rankings", "expert_scores", "predictions")]
            ranking, scores, original = [json.loads(path.read_text()) for path in paths]
            pids = [str(d["pid"]) for d in ranking["top90"]]
            if (len(pids) != 90 or len(set(pids)) != 90 or scores["pids"] != pids or
                    original["qid"] != qid or original["dataset"] != dataset or
                    original["answers"] != q["answers"] or
                    original["subspan_em"] != paper.subspan_em(original["prediction"], q["answers"])):
                raise RuntimeError(f"Source alignment failed: {dataset}/{qid}")
            for key in ("bge_dense_pair_raw", "ce_normalized"):
                if len(scores[key]) != 90 or not all(math.isfinite(float(v)) for v in scores[key]):
                    raise RuntimeError(f"Invalid scores: {dataset}/{qid}/{key}")
            rows.append({"dataset": dataset, "qid": qid, "question": q["query_text"],
                         "answers": q["answers"], "texts": [paper.format_doc(by_pid[x]) for x in pids],
                         "scores": scores, "original": original,
                         "source_sha256": [paper.sha(path) for path in paths]})
    return rows, references, model_path, revision


def preflight(a, progress):
    from transformers import AutoTokenizer
    rows, references, model_path, revision = load_reference(a)
    index = model_path/"model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    for name in (*shards, "config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model_path/name).is_file():
            raise FileNotFoundError(model_path/name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    ids = pass_token_ids(tokenizer)
    capacity = json.loads((model_path/"config.json").read_text())["max_position_embeddings"]
    controls = [f"\nStep {i}:" for i in range(2, a.max_steps+1)] + [method.ANSWER_MARKER]
    reserve = a.max_steps*a.step_tokens + a.answer_tokens + sum(
        len(tokenizer.encode(x, add_special_tokens=False)) for x in controls)
    lengths = []
    for row in rows:
        prefixes, suffix = prompts(tokenizer, row["question"], row["texts"])
        row["context_ids"] = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        row["suffix_ids"] = tokenizer.encode(suffix, add_special_tokens=False)
        length = max(map(len, row["context_ids"])) + len(row["suffix_ids"]) + reserve
        if length > capacity:
            raise RuntimeError(f"Context overflow; no truncation: {row['qid']} {length}>{capacity}")
        lengths.append(length)
        progress.update()
    code = [Path(__file__), Path(method.__file__), Path(legacy.__file__), Path(paper.__file__),
            ROOT/"medrag/generation/pced_pass.py", ROOT/"cache/pced_official/src/methods/pced.py"]
    contract = {
        "version": VERSION, "n": len(rows), "seed": a.seed, "arms": a.arms,
        "hypothesis": "PASS-relative scoring with updated shared reasoning improves grounded expert participation",
        "split": "previously inspected LOFT TEST diagnostic; NOT held-out selection or final generalization evidence",
        "cohort": [{"dataset": r["dataset"], "qid": r["qid"], "sources": r["source_sha256"]} for r in rows],
        "references": references, "reference_root": str(a.reference_root),
        "model": str(model_path), "model_config_sha256": paper.sha(model_path/"config.json"),
        "model_shards": {s: {"bytes": (model_path/s).stat().st_size,
                             "mtime_ns": (model_path/s).stat().st_mtime_ns} for s in sorted(shards)},
        "tokenizer_sha256": paper.sha(model_path/"tokenizer.json"),
        "tokenizer_config_sha256": paper.sha(model_path/"tokenizer_config.json"),
        "system": method.SYSTEM, "expert_rules": method.PASS_RULES,
        "question_template": method.QUESTION, "assistant_prefix": method.PREFIX,
        "pass_token_ids": list(ids), "pass_floor": a.pass_floor,
        "scores": {"pced": "original first-token JSD + gamma2.5 prior; PASS masked; no gate",
                   "probability": "log P(v); PASS masked; no gate",
                   "pass_static": "log P_t(v)-log max(P_0(PASS),floor); raw argmax exclusion frozen at t0",
                   "pass_dynamic": "log P_t(v)-log max(P_t(PASS),floor); raw argmax exclusion each token"},
        "all_pass_policy": "restore experts for that position; retain ratio score; log fallback",
        "generation": {"greedy": True, "backend": "sdpa", "dtype": "bfloat16", "gpu": 1,
                       "max_steps": a.max_steps, "step_tokens": a.step_tokens, "answer_tokens": a.answer_tokens},
        "phase_protocol": "newline advances Step; EOS requests Answer; budget/header/empty ends reasoning; forced headers only",
        "parser": "Answer field only, first nonempty output line, optional Answer prefix stripped; empty=wrong",
        "metric": "unchanged paper Subspan EM; no reasoning-field credit; micro/macro5 and paired transitions",
        "retrieval": "unchanged cached CE Top90 IDs/order, no new retrieval; prior used only in pced arm",
        "amateur": "same CoT instructions, no document or PASS rules; contrastive reference only in pced arm",
        "budgets": "same 91 batched streams/headers/budgets in all arms; no extra decision forward",
        "primary_comparison": "pass_dynamic vs probability; dynamic vs static tests temporal updating",
        "pilot_gate": "diagnostic only: inspect later-step PASS grounding and paired gains/harms before any expansion; no automatic tuning",
        "official_commit": revision,
        "code_sha256": {str(p.relative_to(ROOT)): paper.sha(p) for p in code},
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
    }
    manifest = {"contract": contract, "contract_hash": legacy.digest(contract),
                "max_prompt_plus_output_tokens": max(lengths),
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    path = a.output_root/"manifest.json"
    if path.exists() and json.loads(path.read_text()).get("contract_hash") != manifest["contract_hash"]:
        raise RuntimeError(f"Incompatible existing experiment: use a NEW --output-root, not {a.output_root}")
    if not path.exists() and any(a.output_root.glob("*/predictions/*/*.json")):
        raise RuntimeError("Predictions without manifest; refuse unsafe resume")
    return rows, tokenizer, model_path, ids, manifest


def summarize(a, rows, h):
    conditions, objects = {}, {}
    table = ["dataset,arm,n,correct,subspan_em,empty_answer,answer_budget,reasoning_budget"]
    for arm in a.arms:
        records = [legacy.cached_record(a, arm, r, h) for r in rows]
        if any(r is None for r in records):
            raise RuntimeError(f"Incomplete arm: {arm}")
        objects[arm] = records
        groups = {}
        for dataset in paper.DATASETS:
            subset = [r for r in records if r["dataset"] == dataset]
            n = len(subset)
            g = {"n": n, "correct": sum(r["subspan_em"] for r in subset),
                 "empty_answer": sum(r["empty_answer"] for r in subset),
                 "answer_budget": sum(r["answer_status"] == "token_budget" for r in subset),
                 "reasoning_budget": sum(r["reasoning_stop"] == "token_budget" for r in subset)}
            g["subspan_em"] = g["correct"]/n
            groups[dataset] = g
            table.append(f"{dataset},{arm},{n},{g['correct']},{g['subspan_em']:.6f},{g['empty_answer']},{g['answer_budget']},{g['reasoning_budget']}")
        phase_stats = {}
        for phase in ("reasoning_first", "reasoning_later_start", "reasoning_later_token", "answer"):
            traces = [t for r in records for t in r["trace"] if (
                "answer" if t["phase"] == "answer" else
                "reasoning_first" if t["step"] == 0 else
                "reasoning_later_start" if t["local_step"] == 0 else "reasoning_later_token") == phase]
            expert_n = sum(len(t["pass_probability"]) for t in traces)
            phase_stats[phase] = {"positions": len(traces), "expert_positions": expert_n,
                "mean_pass_probability": sum(sum(t["pass_probability"]) for t in traces)/expert_n if expert_n else None,
                "pass_top1_expert_positions": sum(len(t["abstaining_experts"]) for t in traces),
                "below_floor_expert_positions": sum(t["below_floor_experts"] for t in traces),
                "all_pass_fallback_positions": sum(t["all_pass_fallback"] for t in traces),
                "token_changed_vs_probability_same_prefix": sum(t["token_id"] != t["counterfactual_same_prefix"]["probability"]["token_id"] for t in traces)}
        n = len(records)
        conditions[arm] = {"n": n, "missing": 0, "correct": sum(r["subspan_em"] for r in records),
                           "micro": sum(r["subspan_em"] for r in records)/n,
                           "macro5": sum(g["subspan_em"] for g in groups.values())/5,
                           "datasets": groups, "phase_diagnostics": phase_stats,
                           "seconds": sum(r["seconds"] for r in records)}
    pairs = {}
    for left in a.arms:
        for right in a.arms[a.arms.index(left)+1:]:
            delta = [y["subspan_em"]-x["subspan_em"] for x, y in zip(objects[left], objects[right])]
            rng = random.Random(42)
            draws = sorted(sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(2000))
            pairs[f"{right}_vs_{left}"] = {"n": len(delta), "delta_pp": 100*sum(delta)/len(delta),
                "c_to_w": delta.count(-1), "w_to_c": delta.count(1),
                "prediction_changed": sum(x["prediction"] != y["prediction"] for x, y in zip(objects[left], objects[right])),
                "descriptive_paired_bootstrap_95ci_pp": [100*draws[49], 100*draws[1949]]}
    summary = {"contract_hash": h, "conditions": conditions, "paired": pairs,
               "note": "Previously inspected test diagnostic, no threshold selection. Historical direct PCED is not a controlled CoT baseline."}
    paper.atomic_json(a.output_root/"summary.json", summary)
    path = a.output_root/"summary.csv"
    temp = path.with_suffix(".csv.tmp")
    temp.write_text("\n".join(table)+"\n", encoding="utf-8")
    temp.replace(path)
    return summary


class PilotProgress(Progress):
    """Include cached future arms in wall-time-weighted progress and pending ETA."""
    def __init__(self, names):
        self.cached_fractions = [0.] * len(names)
        super().__init__(names)

    def lines(self):
        first, second = super().lines()
        if not all(w is not None for w in self.weights):
            return first, second
        weights = [max(.001, w) for w in self.weights]
        complete = sum(weights[:self.index]) + weights[self.index]*self.done/self.total
        complete += sum(weights[j]*self.cached_fractions[j] for j in range(self.index+1, len(weights)))
        fraction = complete/sum(weights)
        fresh = self.done-self.initial
        elapsed = time.monotonic()-self.stage_started
        current = ((self.total-self.done)*elapsed/fresh if fresh else
                   weights[self.index]*(1-self.done/self.total))
        eta = current + sum(weights[j]*(1-self.cached_fractions[j]) for j in range(self.index+1, len(weights)))
        bars = min(10, int(fraction*10))
        first = (f"Overall: {fraction*100:.0f}%|{'█'*bars}{'-'*(10-bars)}| "
                 f"[stage {self.index+1}/{len(weights)}, elapsed {duration(time.monotonic()-self.started)}, ETA {duration(eta)}]")
        return first, second


def main(argv=None):
    a = arguments(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("Physical GPU 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    n = 5*a.limit_per_dataset
    names = ["contract/tokenize", "load Llama"] + [f"{arm} LOFT" for arm in a.arms] + ["paired summary"]
    progress = PilotProgress(names)
    progress.weights[-1] = 1.
    active = "preflight"
    try:
        progress.stage(0, n, unit="q")
        rows, tokenizer, model_path, pass_ids, manifest = preflight(a, progress)
        h = manifest["contract_hash"]
        counts = {arm: sum(legacy.cached_record(a, arm, r, h) is not None for r in rows) for arm in a.arms}
        for stage, arm in enumerate(a.arms, 2):
            progress.cached_fractions[stage] = counts[arm]/n
            cached = [legacy.cached_record(a, arm, r, h) for r in rows]
            times = [r["seconds"] for r in cached if r is not None]
            if times:
                progress.weights[stage] = sum(times)/len(times)*n
        progress.complete(f"N={n}; same Top90/model; cached={counts}; {a.output_root}")
        progress.log(f"[plan] contract -> model -> {','.join(a.arms)} -> summary; remaining={sum(n-v for v in counts.values())}; GPU1 only")
        progress.log(f"[scope] TEST diagnostic, no tuning/training/retrieval; common CoT <= {a.max_steps}x{a.step_tokens}+{a.answer_tokens} tokens; PASS floor={a.pass_floor}")
        progress.log("[ETA] 50q x4 arms: provisional 45-120min if 60-128 generated positions/q at prior ~0.23s/position; not a measured CoT rate. First 5 fresh questions calibrate; later arms update separately.")
        if a.dry_run:
            progress.log("[dry-run OK] source hashes, model files, tokenizer, context bounds, cache contracts; no weights/GPU/outputs")
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
        progress.complete("one frozen Llama; 91 batched streams, same one-forward/token in all arms")
        sys.path.insert(0, str(ROOT/"cache/pced_official"))
        from src.methods.pced import prefill, get_jsd
        for stage, arm in enumerate(a.arms, 2):
            progress.stage(stage, n, initial=counts[arm], unit="q")
            fresh_times = []
            for row in rows:
                active = f"{arm}/{row['dataset']}/{row['qid']}"
                if legacy.cached_record(a, arm, row, h) is not None:
                    continue
                started = time.monotonic()
                context, suffix = legacy.batch_inputs(row, tokenizer, model.device)
                prior = paper.prior_tensor(row["scores"], "paper", model.device, model.dtype)
                cache = prefill(model, context)
                stream = method.Stream(model, cache, context, suffix, prior, pass_ids, get_jsd, arm, a.pass_floor)
                result = method.generate(stream, tokenizer, max_steps=a.max_steps,
                                         step_tokens=a.step_tokens, answer_tokens=a.answer_tokens)
                result.update(contract_hash=h, arm=arm, dataset=row["dataset"], qid=row["qid"],
                    question=row["question"], answers=row["answers"],
                    subspan_em=paper.subspan_em(result["prediction"], row["answers"]),
                    top90_pids=row["scores"]["pids"], retrieval_prior=prior.float().cpu().tolist(),
                    seconds=time.monotonic()-started)
                paper.atomic_json(legacy.output_path(a, arm, row), result)
                fresh_times.append(result["seconds"])
                del stream, cache, context, suffix, prior, result
                progress.update()
                if len(fresh_times) >= 5:
                    for later in range(stage+1, len(names)-1):
                        if progress.weights[later] is None:
                            progress.weights[later] = sum(fresh_times)/len(fresh_times)*n
            progress.complete(f"{arm}={n}/{n}; {a.output_root/arm}")
        progress.stage(len(names)-1, 1)
        summary = summarize(a, rows, h)
        progress.complete(str(a.output_root/"summary.json"))
        progress.log("[result] "+"; ".join(f"{arm}={v['correct']}/{n}" for arm, v in summary["conditions"].items()))
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={active}; completed={progress.done}/{progress.total}; remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; durable={a.output_root}; identical command resumes committed questions")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

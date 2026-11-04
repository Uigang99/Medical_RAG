#!/usr/bin/env python3
"""LOFT DEV50: separate sentence-level participation decisions with frozen Llama."""
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
from scripts import evaluate_loft_pced_pass_bounded as previous
from scripts import evaluate_loft_pced_pass as common
from scripts import run_loft_pced_paper as paper
from scripts.evaluate_loft_pced_pass_cot import PilotProgress
from medrag.generation import pced_sentence_pass as method
from medrag.generation.pced_pass import pass_token_ids

VERSION = "loft_sentence_pass_dev_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retrieval-root", type=Path, default=ROOT/"results/gemma12_pced_comparison_v1/dev")
    p.add_argument("--model", type=Path, default=ROOT.parent/"models/Llama-3.1-8B-Instruct")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--arms", nargs="+", choices=method.ARMS, default=list(method.ARMS))
    p.add_argument("--limit-per-dataset", type=int, default=10)
    p.add_argument("--judge-batch-size", type=int, default=16)
    p.add_argument("--contribution-weight", type=float, default=4.)
    p.add_argument("--reasoning-style", choices=method.REASONING_STYLES, default="legacy")
    p.add_argument("--reasoning-tokens", type=int)
    p.add_argument("--answer-tokens", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.reasoning_tokens = a.reasoning_tokens if a.reasoning_tokens is not None else (192 if a.reasoning_style == "evidence_v2" else 128)
    if not 1 <= a.limit_per_dataset <= 10 or not 1 <= a.judge_batch_size <= 90:
        p.error("limit-per-dataset=1..10, judge-batch-size=1..90")
    if not math.isfinite(a.contribution_weight) or not 0 <= a.contribution_weight <= 20:
        p.error("contribution-weight must be finite, 0..20")
    if not 8 <= a.reasoning_tokens <= 256 or not 8 <= a.answer_tokens <= 128:
        p.error("reasoning-tokens=8..256, answer-tokens=8..128")
    a.retrieval_root, a.model = a.retrieval_root.resolve(), a.model.resolve()
    name = "loft_llama_sentence_pass_reasoning_v2" if a.reasoning_style == "evidence_v2" else "loft_llama_sentence_pass_dev_v1"
    a.output_root = (a.output_root or ROOT/"results"/f"{name}_n{5*a.limit_per_dataset}").resolve()
    for source in (a.retrieval_root, a.model, ROOT/"datasets"):
        if a.output_root == source or source in a.output_root.parents or a.output_root in source.parents:
            p.error("Output must be separate from immutable sources")
    a.arms, a.seed = list(dict.fromkeys(a.arms)), 42
    return a


def preflight(a, progress):
    from transformers import AutoTokenizer, DynamicCache
    rows, sources = previous.load_rows(a)
    official = ROOT/"cache/pced_official"
    revision = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError("Pinned official PCED revision changed")
    cfg = json.loads((a.model/"config.json").read_text())
    if cfg.get("model_type") != "llama" or not callable(getattr(DynamicCache, "crop", None)):
        raise RuntimeError("Requires local Llama and DynamicCache.crop")
    index = a.model/"model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in (*shards, "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if not (a.model/name).is_file():
            raise FileNotFoundError(a.model/name)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    labels, pass_ids = method.decision_ids(tokenizer), pass_token_ids(tokenizer)
    max_gen, max_judge, peak_kv = 0, 0, 0
    per_token = 2*cfg["num_hidden_layers"]*cfg["num_key_value_heads"]*(cfg["hidden_size"]//cfg["num_attention_heads"])*2
    for row in rows:
        # Exactly the no-PASS generation prompt in every arm. Decisions never enter it.
        if a.reasoning_style == "evidence_v2":
            prefixes, suffix = method.generation_prompts(tokenizer, row["question"], row["texts"])
        else:
            prefixes, suffix = previous.prompts(tokenizer, row["question"], row["texts"], pass_prompt=False)
        row["layout"] = {"context_ids": tokenizer(prefixes, add_special_tokens=False)["input_ids"],
                         "suffix_ids": tokenizer.encode(suffix, add_special_tokens=False)}
        jp, tail = method.judge_prompts(tokenizer, row["question"], row["texts"])
        row["judge_ids"] = tokenizer(jp, add_special_tokens=False)["input_ids"]
        row["judge_tail"] = tail
        reserve = a.reasoning_tokens+a.answer_tokens+len(tokenizer.encode(method.base.ANSWER_MARKER, add_special_tokens=False))
        gen = max(map(len, row["layout"]["context_ids"]))+len(row["layout"]["suffix_ids"])+reserve
        # Retokenized shared text is checked again at runtime, never truncated.
        judge_suffix_bound = 2*a.reasoning_tokens+max(len(method.decision_suffix(tokenizer, "", phase, tail)) for phase in method.JUDGE_PHASE)
        lengths = sorted(map(len, row["judge_ids"]))
        judge = max(lengths)+judge_suffix_bound
        if max(gen, judge) > cfg["max_position_embeddings"]:
            raise RuntimeError(f"Context overflow before weights: {row['qid']}")
        judge_cache_tokens = sum(len(lengths[k:k+a.judge_batch_size])*max(lengths[k:k+a.judge_batch_size])
                                 for k in range(0, len(lengths), a.judge_batch_size))
        # Simultaneous main 91-stream KV + all decision prefixes + one decision suffix batch.
        live_tokens = 91*gen+judge_cache_tokens+a.judge_batch_size*judge_suffix_bound
        peak_kv = max(peak_kv, per_token*live_tokens/2**30)
        max_gen, max_judge = max(max_gen, gen), max(max_judge, judge)
        progress.update()
    paths = [Path(__file__), Path(method.__file__), Path(previous.__file__), Path(method.base.__file__),
             Path(common.__file__), Path(paper.__file__), ROOT/"medrag/generation/pced_pass.py",
             ROOT/"scripts/evaluate_loft_pced_pass_cot.py", ROOT/"medrag/training/graph_input_progress.py",
             official/"src/methods/pced.py", ROOT/"scripts/run_loft_pced_sentence_pass.sh"]
    if a.reasoning_style == "evidence_v2":
        paths.append(ROOT/"scripts/run_loft_pced_sentence_pass_reasoning_v2.sh")
    natural = a.reasoning_style == "evidence_v2"
    contract = {
        "version": "loft_sentence_pass_reasoning_v2" if natural else VERSION,
        "reasoning_style": a.reasoning_style,
        "hypothesis": "Sentence-boundary participation improves PCED under a shared reasoning protocol; no Joint or separate final-answer model",
        "split": "official LOFT DEV50, previously inspected; not untouched test", "n": len(rows), "seed": a.seed,
        "cohort": [{"dataset": r["dataset"], "qid": r["qid"], "sources": r["source_sha256"],
                    "question_sha256": common.digest(r["question"]), "labels_sha256": common.digest([r["answers"], r["qrels"]])} for r in rows],
        "retrieval_sources": sources, "retrieval_root": str(a.retrieval_root),
        "retrieval": "unchanged full-corpus CE Top90, cached BGE-M3/CE harmonic prior; no gold routing",
        "model": str(a.model), "model_files": {name: paper.sha(a.model/name) for name in
            ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")},
        "weight_shards": {s: {"bytes": (a.model/s).stat().st_size, "mtime_ns": (a.model/s).stat().st_mtime_ns} for s in sorted(shards)},
        "prompts": {"generation_system": method.EVIDENCE_SYSTEM if natural else method.base.SYSTEM, "amateur": method.base.AMATEUR_RULES,
                    "assistant_prefix": method.base.PREFIX, "judge_system": method.JUDGE_SYSTEM,
                    "judge_phase": method.JUDGE_PHASE, "judge_template": "Question:\\n{q}\\n\\nPassage:\\n{d}\\n\\nShared reasoning so far:\\n{cot}\\n\\n{phase}\\n<chat assistant>Decision:"},
        "arms": a.arms, "decision_label_ids_C_P": labels, "reserved_generation_pass_ids": pass_ids,
        "decision": "softmax of C/P logits only; c=P(C|C,P), p=1-c; c>0.5 active; probabilities are not calibrated",
        "scores": "base=(1+beta)*expert-beta*amateur+2.5*log(prior); active-only max over expert/token; weighted adds lambda*log(max(c,1e-6))",
        "contribution_weight": a.contribution_weight, "gamma": 2.5,
        "beta": "official first-generation-position vocabulary-wise JSD, fixed",
        "sentence_boundary": "token-aligned .!? endings incl quotes; ignore common abbreviations, initials and numeric periods; no newline-only split, no token rollback",
        "gate_protocol": "initial reasoning and detected sentence starts; fixed within sentence; all experts rejoin decision; Answer gets one fresh gate fixed to its end",
        "all_pass": "all document experts excluded for whole segment; amateur ordinary greedy token per step; explicitly logged; retry only next boundary",
        "generation": {"reasoning_tokens": a.reasoning_tokens, "answer_tokens": a.answer_tokens,
                       "dtype": "bfloat16", "attention": "sdpa", "greedy": True, "gpu": 1, "streams": 91,
                       "judge_batch_size": a.judge_batch_size, "cache": "separate retained judge prefixes, append shared CoT/phase then crop; generation KV never modified by judge"},
        "format": "same no-PASS generation prompt for all arms; other specials/PASS aliases masked; EOS after alphanumeric Answer only; "+
                  ("natural Answer marker required; budget stop -> incomplete, empty prediction, no forced header or repair" if natural else "force Answer at reasoning cap"),
        "parser_metric": "unchanged stripped Answer-only Subspan EM; no reasoning credit or answer repair",
        "answer_decoder": "same PCED expert/amateur streams and document KV; fresh answer gate; NOT Joint, NOT a separate Llama answer call",
        "comparison_scope": "Within this run only gate policy varies. Versus v1, prompt, reasoning budget and cap handling change together; not an isolated prompt or PASS effect. Incomplete cases remain in the full-cohort denominator.",
        "primary": "sentence_weighted vs pced; sentence_exclude isolates hard exclusion; weighted vs exclude isolates log-c score",
        "pilot_gate": "at N=50: weighted >= pced+3 correct, <=1 empty answer; qualitative evidence of grounded reentry/selection needed; no automatic expansion",
        "official_commit": revision, "code_sha256": {str(p.relative_to(ROOT)): paper.sha(p) for p in paths},
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
    }
    manifest = {"contract": contract, "contract_hash": common.digest(contract),
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "max_generation_tokens": max_gen, "max_judge_tokens_estimate": max_judge,
                "estimated_peak_kv_gib": peak_kv, "min_free_gib_before_model": max(40., 16+peak_kv*1.25+10)}
    path = a.output_root/"manifest.json"
    if path.exists() and json.loads(path.read_text()).get("contract_hash") != manifest["contract_hash"]:
        raise RuntimeError(f"Incompatible resume contract; choose a new output root: {a.output_root}")
    if not path.exists() and any(a.output_root.glob("*/predictions/*/*.json")):
        raise RuntimeError("Predictions without manifest")
    return rows, tokenizer, labels, pass_ids, manifest


def score_prediction(result, answers):
    # An unfinished response is always wrong, even for a degenerate gold alias
    # that normalizes to an empty string under the inherited Subspan scorer.
    if result["answer_status"] == "skipped_incomplete_reasoning":
        return 0
    return paper.subspan_em(result["prediction"], answers)


def summarize(a, rows, h):
    records, conditions, paired = {}, {}, {}
    csv = ["dataset,arm,n,correct,subspan_em,delta_vs_pced_pp,c_to_w_vs_pced,w_to_c_vs_pced"]
    for arm in a.arms:
        records[arm] = [common.cached_record(a, arm, row, h) for row in rows]
        if any(r is None for r in records[arm]):
            raise RuntimeError(f"Incomplete {arm}; no partial full-cohort score")
    for arm, rs in records.items():
        groups = {}
        for ds in paper.DATASETS:
            subset = [r for r in rs if r["dataset"] == ds]
            ref = [r for r in records.get("pced", []) if r["dataset"] == ds]
            n, correct = len(subset), sum(r["subspan_em"] for r in subset)
            cw = sum(x["subspan_em"] and not y["subspan_em"] for x, y in zip(ref, subset)) if ref else None
            wc = sum(not x["subspan_em"] and y["subspan_em"] for x, y in zip(ref, subset)) if ref else None
            delta = 100*(wc-cw)/n if ref else None
            groups[ds] = {"n": n, "correct": correct, "subspan_em": correct/n,
                          "delta_vs_pced_pp": delta, "c_to_w": cw, "w_to_c": wc}
            csv.append(",".join(str(x) if x is not None else "" for x in (ds, arm, n, correct, correct/n, delta, cw, wc)))
        traces = [t for r in rs for t in r["trace"]]
        events = [e for r in rs for e in r["decisions"] if e["decision_evaluated"]]
        reentries, exits = 0, 0
        for r in rs:
            for left, right in zip(r["decisions"], r["decisions"][1:]):
                reentries += len(set(right["active_experts"])-set(left["active_experts"]))
                exits += len(set(left["active_experts"])-set(right["active_experts"]))
        phases = {}
        for phase in ("reasoning", "answer"):
            es = [e for e in events if e["phase"] == phase]
            phases[phase] = {"decision_events": len(es), "expert_decisions": len(es)*90,
                "abstentions": sum(90-len(e["active_experts"]) for e in es),
                "all_pass_events": sum(not e["active_experts"] for e in es),
                "mean_unconditional_label_mass": sum(sum(e["unconditional_label_mass"]) for e in es)/(90*len(es)) if es else None}
        switches = {}
        for left, right in (("pced", "sentence_exclude"), ("sentence_exclude", "sentence_weighted")):
            switches[right+"_vs_"+left] = {k: sum(t["counterfactual_same_prefix"][left][k] != t["counterfactual_same_prefix"][right][k] for t in traces)
                                         for k in ("winner", "token_id")}
        conditions[arm] = {"n": len(rs), "missing": 0, "correct": sum(r["subspan_em"] for r in rs),
            "micro": sum(r["subspan_em"] for r in rs)/len(rs), "macro5": sum(g["subspan_em"] for g in groups.values())/5,
            "datasets": groups, "empty_answer": sum(r["empty_answer"] for r in rs),
            "reasoning_budget": sum(r["reasoning_stop"] == "token_budget" for r in rs),
            "natural_answer_transitions": sum(r["reasoning_stop"] == "answer_marker" for r in rs),
            "empty_reasoning": sum(r["empty_reasoning"] for r in rs),
            "mean_reasoning_tokens": sum(len(r["reasoning_token_ids"]) for r in rs)/len(rs),
            "answer_skipped_incomplete_reasoning": sum(r["answer_status"] == "skipped_incomplete_reasoning" for r in rs),
            "answer_budget": sum(r["answer_status"] == "token_budget" for r in rs),
            "seconds": sum(r["seconds"] for r in rs), "judge_seconds": sum(r["judge_seconds"] for r in rs),
            "phase_diagnostics": phases, "expert_reentries": reentries, "expert_exits": exits,
            "generation_positions": len(traces), "same_prefix_changes": switches,
            "amateur_fallback_positions": sum(t["amateur_fallback"] for t in traces)}
    for i, left in enumerate(a.arms):
        for right in a.arms[i+1:]:
            deltas = [b["subspan_em"]-a_["subspan_em"] for a_, b in zip(records[left], records[right])]
            rng = random.Random(42)
            draws = sorted(100*sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(2000))
            paired[right+"_vs_"+left] = {"n": len(deltas), "delta_pp": 100*sum(deltas)/len(deltas),
                "c_to_w": deltas.count(-1), "w_to_c": deltas.count(1),
                "prediction_changed": sum(x["prediction"] != y["prediction"] for x, y in zip(records[left], records[right])),
                "descriptive_paired_bootstrap_95ci_pp": [draws[49], draws[1949]]}
    eligible = len(rows) == 50 and {"pced", "sentence_weighted"} <= conditions.keys()
    gate = (conditions["sentence_weighted"]["correct"] >= conditions["pced"]["correct"]+3 and
            conditions["sentence_weighted"]["empty_answer"] <= 1) if eligible else None
    result = {"contract_hash": h, "conditions": conditions, "paired": paired, "numeric_pilot_gate_passed": gate,
              "note": "Previously inspected development set; no generalization claim. Qrels are not complete per-sentence contribution labels. Qualitative review required before expansion."}
    paper.atomic_json(a.output_root/"summary.json", result)
    target = a.output_root/"summary.csv"
    temp = target.with_suffix(".csv.tmp")
    temp.write_text("\n".join(csv)+"\n", encoding="utf-8")
    temp.replace(target)
    return result


def main(argv=None):
    a = arguments(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("Physical GPU 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    stages = [(arm, ds) for arm in a.arms for ds in paper.DATASETS]
    names = ["DEV contract/tokenize", "load Llama"]+[f"{arm.replace('sentence_', '')}/{ds}" for arm, ds in stages]+["paired summary"]
    progress = PilotProgress(names)
    progress.weights[-1] = 1.
    active = "preflight"
    try:
        progress.stage(0, 5*a.limit_per_dataset, unit="q")
        rows, tokenizer, labels, pass_ids, manifest = preflight(a, progress)
        h, counts = manifest["contract_hash"], {}
        for stage, (arm, ds) in enumerate(stages, 2):
            subset = [r for r in rows if r["dataset"] == ds]
            cached = [common.cached_record(a, arm, r, h) for r in subset]
            counts[(arm, ds)] = sum(r is not None for r in cached)
            progress.cached_fractions[stage] = counts[(arm, ds)]/len(subset)
            times = [r["seconds"] for r in cached if r is not None]
            if times:
                progress.weights[stage] = sum(times)/len(times)*len(subset)
        progress.complete(f"DEV={len(rows)}; cached={sum(counts.values())}/{len(rows)*len(a.arms)}; max generation={manifest['max_generation_tokens']}, judge={manifest['max_judge_tokens_estimate']}; {a.output_root}")
        progress.log(f"[plan] contract -> one frozen Llama -> {','.join(a.arms)} -> summary; remaining={len(rows)*len(a.arms)-sum(counts.values())}; GPU1 only; no retrieval/training/TEST")
        progress.log(f"[scope] same generation prompt/Top90; C/P decision at sentence boundaries; c>0.5 active; score+= {a.contribution_weight}*log(c); fresh Answer decision; all-PASS -> amateur for segment")
        progress.log(f"[reasoning] style={a.reasoning_style}; cap={a.reasoning_tokens}; same PCED final decoder; no Joint/separate answer model; cap without Answer="+
                     ("incomplete/empty prediction counted in full-cohort accuracy" if a.reasoning_style == "evidence_v2" else "forced Answer header"))
        progress.log("[ETA] Prior sentence DEV50 x3 arms: 25m49s at reasoning cap128. At cap192 allow roughly 30-60min on idle H200 GPU1; prompt-dependent, not matched throughput. First 5 fresh questions per policy recalibrate; resumed time excludes cached work.")
        if a.dry_run:
            progress.log(f"[dry-run OK] sources, DEV/TEST disjoint IDs/text, local tokenizer C/P={labels}, contexts and resume checked; estimated combined KV={manifest['estimated_peak_kv_gib']:.1f}GiB; required free before load={manifest['min_free_gib_before_model']:.1f}GiB; no weights/GPU/result writes")
            return
        if not (a.output_root/"manifest.json").exists():
            paper.atomic_json(a.output_root/"manifest.json", manifest)
        progress.stage(1, 1)
        model = None
        if sum(counts.values()) < len(rows)*len(a.arms):
            import torch
            from transformers import AutoModelForCausalLM
            paper.gpu_gate(manifest["min_free_gib_before_model"])
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(a.seed)
            torch.manual_seed(a.seed)
            model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        progress.complete("one Llama; separate generation/judge KV; generation=91 streams, judge prefixes length-bucketed")
        sys.path.insert(0, str(ROOT/"cache/pced_official"))
        from src.methods.pced import prefill, get_jsd
        fresh_times = {arm: [] for arm in a.arms}
        for stage, (arm, ds) in enumerate(stages, 2):
            subset = [r for r in rows if r["dataset"] == ds]
            progress.stage(stage, len(subset), initial=counts[(arm, ds)], unit="q")
            for row in subset:
                active = f"{arm}/{ds}/{row['qid']}"
                if common.cached_record(a, arm, row, h) is not None:
                    continue
                started = time.monotonic()
                context, suffix = common.batch_inputs(row["layout"], tokenizer, model.device)
                prior = paper.prior_tensor(row["scores"], "paper", model.device, model.dtype)
                cache = prefill(model, context)
                judge = (None if arm == "pced" else method.CachedJudge(model, tokenizer, row["judge_ids"],
                         row["judge_tail"], labels, a.judge_batch_size))
                stream = method.Stream(model, cache, context, suffix, prior, pass_ids, get_jsd, arm,
                                       judge, weight=a.contribution_weight, special_ids=tokenizer.all_special_ids)
                result = method.generate(stream, tokenizer, reasoning_tokens=a.reasoning_tokens, answer_tokens=a.answer_tokens,
                                         force_answer_at_budget=a.reasoning_style == "legacy")
                result.update(contract_hash=h, arm=arm, dataset=ds, qid=row["qid"], question=row["question"],
                    answers=row["answers"], qrels_for_analysis_only=row["qrels"],
                    subspan_em=score_prediction(result, row["answers"]),
                    top90_pids=row["scores"]["pids"], retrieval_prior=prior.float().cpu().tolist(),
                    judge_seconds=(judge.prefill_seconds if judge else 0)+sum(e["seconds"] for e in result["decisions"]),
                    seconds=time.monotonic()-started)
                paper.atomic_json(common.output_path(a, arm, row), result)
                fresh_times[arm].append(result["seconds"])
                del stream, judge, cache, context, suffix, prior, result
                progress.update()
                if len(fresh_times[arm]) >= 5:
                    for later, (later_arm, _) in enumerate(stages, 2):
                        if later > stage and later_arm == arm and progress.weights[later] is None:
                            progress.weights[later] = sum(fresh_times[arm][-5:])/5*a.limit_per_dataset
            progress.complete(f"{arm}/{ds}={len(subset)}/{len(subset)}; {a.output_root/arm/'predictions'/ds}")
        progress.stage(len(names)-1, 1)
        result = summarize(a, rows, h)
        progress.complete(str(a.output_root/"summary.json"))
        progress.log("[result] "+"; ".join(f"{arm}={v['correct']}/{v['n']}" for arm, v in result["conditions"].items()))
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={active}; stage_done={progress.done}/{progress.total}; stage_remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; durable={a.output_root}; identical command resumes committed questions, unfinished question restarts")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

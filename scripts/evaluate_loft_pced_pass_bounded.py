#!/usr/bin/env python3
"""Four controlled Llama/PASS arms on the 50 official LOFT development queries."""
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
from scripts import evaluate_loft_pced_pass as common
from scripts import run_loft_pced_paper as paper
from scripts.evaluate_loft_pced_pass_cot import PilotProgress
from medrag.generation import pced_pass_bounded as method
from medrag.generation.pced_pass import pass_token_ids

VERSION = "loft_pass_bounded_dev_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retrieval-root", type=Path, default=ROOT/"results/gemma12_pced_comparison_v1/dev")
    p.add_argument("--model", type=Path, default=ROOT.parent/"models/Llama-3.1-8B-Instruct")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--limit-per-dataset", type=int, default=10)
    p.add_argument("--arms", nargs="+", choices=method.ARMS, default=list(method.ARMS))
    p.add_argument("--pass-penalty", type=float, default=4.)
    p.add_argument("--reasoning-tokens", type=int, default=128)
    p.add_argument("--answer-tokens", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.limit_per_dataset <= 10:
        p.error("Development cohort has 10 queries per dataset")
    if not math.isfinite(a.pass_penalty) or not 0 <= a.pass_penalty <= 20:
        p.error("pass-penalty must be finite and in [0,20]")
    if not 8 <= a.reasoning_tokens <= 256 or not 8 <= a.answer_tokens <= 128:
        p.error("reasoning-tokens=8..256, answer-tokens=8..128")
    a.retrieval_root, a.model = a.retrieval_root.resolve(), a.model.resolve()
    a.output_root = (a.output_root or ROOT/"results"/
                     f"loft_llama_pass_bounded_dev_v1_n{5*a.limit_per_dataset}").resolve()
    for source in (a.retrieval_root, a.model, ROOT/"datasets"):
        if a.output_root == source or source in a.output_root.parents or a.output_root in source.parents:
            p.error("Output root must be separate from source data/model/retrieval")
    a.arms = list(dict.fromkeys(a.arms))
    a.seed = 42
    return a


def prompts(tokenizer, question, texts, *, pass_prompt):
    boundary = "<PCED_BOUNDARY_42>"
    prefixes, suffix = [], None
    for index, text in enumerate(["", *texts]):
        if boundary in question or boundary in text:
            raise ValueError("Prompt boundary collision")
        rules = method.AMATEUR_RULES if index == 0 else method.PASS_RULES if pass_prompt else ""
        body = "Your passage:\n"+text+"\n\n"+boundary
        rendered = tokenizer.apply_chat_template([
            {"role": "system", "content": method.SYSTEM+"\n"+rules},
            {"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
        prefix, trailing = rendered.split(boundary)
        current = "Question:\n"+question+"\n"+trailing+method.PREFIX
        if suffix is not None and suffix != current:
            raise RuntimeError("Expert and amateur must share the question/response suffix")
        prefixes.append(prefix)
        suffix = current
    return prefixes, suffix


def load_rows(a):
    rows, sources = [], {}
    for dataset in paper.DATASETS:
        folder = ROOT/"datasets/benchmark/loft/rag"/dataset/"128k"
        cp, qp, tp = (folder/name for name in ("corpus.jsonl", "dev_queries.jsonl", "test_queries.jsonl"))
        corpus, queries, test = (list(paper.jsonl(path)) for path in (cp, qp, tp))
        by_pid = {str(d["pid"]): d for d in corpus}
        qids = [str(q["qid"]) for q in queries]
        if len(corpus) != paper.COUNTS[dataset] or len(by_pid) != len(corpus) or len(queries) != 10 or len(set(qids)) != 10:
            raise RuntimeError(f"Unexpected development/corpus size: {dataset}")
        if set(qids) & {str(q["qid"]) for q in test}:
            raise RuntimeError(f"Development/test ID overlap: {dataset}")
        # Check literal question overlap as well; this does not establish no paraphrase leakage.
        if {q["query_text"].strip() for q in queries} & {q["query_text"].strip() for q in test}:
            raise RuntimeError(f"Development/test exact question overlap: {dataset}")
        cache = a.retrieval_root/dataset
        mp = cache/"prepare_manifest.json"
        manifest = json.loads(mp.read_text())
        expected = {"phase": "prepare-dev", "dataset": dataset, "n": 10, "qids": qids,
                    "corpus_sha256": paper.sha(cp), "dev_sha256": paper.sha(qp),
                    "selection": "full-corpus CE; torch.topk(float16(CE),90)",
                    "prior": "BGE-M3 dense pair raw + CE normalized fp16"}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Incompatible development retrieval manifest: {mp}")
        sources[dataset] = {"manifest_sha256": paper.sha(mp), "corpus_sha256": paper.sha(cp),
                            "dev_sha256": paper.sha(qp), "test_disjoint_check_sha256": paper.sha(tp),
                            "retrieval_contract": manifest}
        selected = sorted(queries, key=lambda q: common.digest([a.seed, dataset, str(q["qid"])]))[:a.limit_per_dataset]
        for q in selected:
            qid = str(q["qid"])
            paths = [paper.result_file(cache/name, qid) for name in ("rankings", "expert_scores")]
            rank, scores = [json.loads(path.read_text()) for path in paths]
            pids = [str(d["pid"]) for d in rank["top90"]]
            if (rank["qid"] != qid or scores["qid"] != qid or len(pids) != 90 or
                    len(set(pids)) != 90 or scores["pids"] != pids or any(x not in by_pid for x in pids)):
                raise RuntimeError(f"Retrieval ID alignment failed: {dataset}/{qid}")
            for key in ("bge_dense_pair_raw", "ce_normalized"):
                if len(scores[key]) != 90 or not all(math.isfinite(float(x)) for x in scores[key]):
                    raise RuntimeError(f"Invalid {key}: {dataset}/{qid}")
            if (scores["ce_normalized"] != [d["rerank_score_fp16"] for d in rank["top90"]] or
                    not all(0 <= x <= 1 for x in scores["ce_normalized"])):
                raise RuntimeError(f"CE score alignment/range failed: {dataset}/{qid}")
            if not q.get("query_text") or not q.get("answers"):
                raise RuntimeError(f"Missing question/answers: {dataset}/{qid}")
            qrels = q.get("metadata", {}).get("qrels", [])
            if any(str(pid) not in by_pid for pid, rel in qrels if int(rel) > 0):
                raise RuntimeError(f"Missing gold document: {dataset}/{qid}")
            rows.append({"dataset": dataset, "qid": qid, "question": q["query_text"],
                         "answers": q["answers"], "qrels": qrels, "scores": scores,
                         "texts": [paper.format_doc(by_pid[pid]) for pid in pids],
                         "source_sha256": [paper.sha(path) for path in paths]})
    return rows, sources


def preflight(a, progress):
    from transformers import AutoTokenizer
    rows, sources = load_rows(a)
    revision = subprocess.check_output(["git", "-C", str(ROOT/"cache/pced_official"), "rev-parse", "HEAD"], text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError("Pinned official PCED revision changed")
    cfg = json.loads((a.model/"config.json").read_text())
    if cfg.get("model_type") != "llama":
        raise RuntimeError("This pilot is validated for the existing Llama only")
    index = a.model/"model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in (*shards, "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if not (a.model/name).is_file():
            raise FileNotFoundError(a.model/name)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    pass_ids = pass_token_ids(tokenizer)
    max_length = 0
    reserve = a.reasoning_tokens+a.answer_tokens+len(tokenizer.encode(method.ANSWER_MARKER, add_special_tokens=False))
    for row in rows:
        row["layouts"] = {}
        for layout in ("base", "pass"):
            prefixes, suffix = prompts(tokenizer, row["question"], row["texts"], pass_prompt=layout == "pass")
            context_ids = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            length = max(map(len, context_ids))+len(suffix_ids)+reserve
            if length > cfg["max_position_embeddings"]:
                raise RuntimeError(f"Context overflow, no silent truncation: {row['qid']}")
            row["layouts"][layout] = {"context_ids": context_ids, "suffix_ids": suffix_ids}
            max_length = max(max_length, length)
        progress.update()
    dependencies = [Path(__file__), Path(method.__file__), Path(common.__file__), Path(paper.__file__),
                    ROOT/"scripts/evaluate_loft_pced_pass_cot.py", ROOT/"medrag/generation/pced_pass.py",
                    ROOT/"medrag/training/graph_input_progress.py", ROOT/"cache/pced_official/src/methods/pced.py",
                    ROOT/"scripts/run_loft_pced_pass_bounded.sh"]
    kv_bytes = 2*cfg["num_hidden_layers"]*cfg["num_key_value_heads"]*(cfg["hidden_size"]//cfg["num_attention_heads"])*2*91*max_length
    contract = {
        "version": VERSION, "split": "official LOFT dev; previously used in other development experiments; not final TEST",
        "hypothesis": "Same-forward dynamic PASS exclusion and bounded token-specific penalty improve PCED grounding",
        "n": len(rows), "seed": a.seed, "arms": a.arms,
        "cohort": [{"dataset": r["dataset"], "qid": r["qid"], "sources": r["source_sha256"],
                    "question_sha256": common.digest(r["question"]), "labels_sha256": common.digest([r["answers"], r["qrels"]])} for r in rows],
        "retrieval_sources": sources, "retrieval_root": str(a.retrieval_root),
        "retrieval": "cached full-corpus CE Top90; BGE-M3 dense pair/CE harmonic prior; no retrieval or gold routing",
        "model": str(a.model), "model_files": {name: paper.sha(a.model/name) for name in
            ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")},
        "weight_shards": {s: {"bytes": (a.model/s).stat().st_size, "mtime_ns": (a.model/s).stat().st_mtime_ns} for s in sorted(shards)},
        "prompts": {"system": method.SYSTEM, "pass_rules": method.PASS_RULES, "amateur": method.AMATEUR_RULES,
                    "assistant_prefix": method.PREFIX, "question_template": "Question:\\n{question}\\n"},
        "pass_ids": list(pass_ids), "penalty": a.pass_penalty, "gamma": 2.5,
        "beta": "official vocabulary-wise JSD at first generated position, fixed thereafter",
        "scores": "original PCED - lambda*sigmoid(logsumexp(PASS aliases)-raw_logit(v)); no probability floor",
        "arms_contract": {"pced": "no PASS prompt; no exclusion/penalty", "prompt_only": "PASS prompt only",
                          "exclude": "PASS prompt + per-token raw top1 exclusion",
                          "bounded": "PASS prompt + exclusion + bounded candidate-token penalty"},
        "format": "other special IDs blocked; EOS blocked until Answer has an alphanumeric character; format masks before hard PASS top1",
        "phase_protocol": "natural reasoning; line-start Answer: enters answer; budget forces Answer:; whitespace never ends reasoning",
        "fallback": "all passage experts remain excluded; use existing amateur's highest ordinary format-allowed token",
        "parser": "entire stripped Answer field only; no CoT credit, semantic repair or answer substitution",
        "generation": {"reasoning_tokens": a.reasoning_tokens, "answer_tokens": a.answer_tokens,
                       "greedy": True, "dtype": "bfloat16", "attention": "sdpa", "physical_gpu": 1,
                       "streams": 91, "extra_judge_forwards": 0},
        "metrics": "unchanged Subspan EM; per-dataset, Micro, Macro-5; paired c->w/w->c and descriptive bootstrap CI",
        "primary_comparison": "bounded vs prompt_only; exclude vs prompt_only isolates hard gate; bounded vs exclude isolates penalty",
        "pilot_gate": "at N=50: >=3 additional correct vs prompt_only, no net loss vs pced, <=1 empty answer; qualitative later-token grounding still required before expansion",
        "official_commit": revision, "code_sha256": {str(p.relative_to(ROOT)): paper.sha(p) for p in dependencies},
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
    }
    manifest = {"contract": contract, "contract_hash": common.digest(contract),
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "max_prompt_plus_output_tokens": max_length, "estimated_peak_kv_gib": kv_bytes/2**30,
                "min_free_gib_before_model": max(40., 16+kv_bytes/2**30*1.25+8)}
    path = a.output_root/"manifest.json"
    if path.exists() and json.loads(path.read_text()).get("contract_hash") != manifest["contract_hash"]:
        raise RuntimeError(f"Different contract: choose a new --output-root, not {a.output_root}")
    if not path.exists() and any(a.output_root.glob("*/predictions/*/*.json")):
        raise RuntimeError("Predictions without manifest; unsafe resume")
    return rows, tokenizer, pass_ids, manifest


def summarize(a, rows, h):
    records, conditions, paired = {}, {}, {}
    lines = ["dataset,arm,n,correct,subspan_em,delta_vs_pced_pp,c_to_w_vs_pced,w_to_c_vs_pced,empty_answer,reasoning_budget,answer_budget"]
    for arm in a.arms:
        records[arm] = [common.cached_record(a, arm, row, h) for row in rows]
        if any(r is None for r in records[arm]):
            raise RuntimeError(f"Incomplete arm {arm}")
    for arm in a.arms:
        groups = {}
        for dataset in paper.DATASETS:
            subset = [r for r in records[arm] if r["dataset"] == dataset]
            ref = [r for r in records.get("pced", []) if r["dataset"] == dataset]
            n, correct = len(subset), sum(r["subspan_em"] for r in subset)
            cw = sum(x["subspan_em"] and not y["subspan_em"] for x, y in zip(ref, subset)) if ref else None
            wc = sum(not x["subspan_em"] and y["subspan_em"] for x, y in zip(ref, subset)) if ref else None
            delta = 100*(wc-cw)/n if ref else None
            g = {"n": n, "correct": correct, "subspan_em": correct/n, "delta_vs_pced_pp": delta,
                 "c_to_w_vs_pced": cw, "w_to_c_vs_pced": wc,
                 "empty_answer": sum(r["empty_answer"] for r in subset),
                 "reasoning_budget": sum(r["reasoning_stop"] == "token_budget" for r in subset),
                 "answer_budget": sum(r["answer_status"] == "token_budget" for r in subset)}
            groups[dataset] = g
            lines.append(",".join(str(x) if x is not None else "" for x in
                (dataset, arm, n, correct, correct/n, delta, cw, wc, g["empty_answer"], g["reasoning_budget"], g["answer_budget"])))
        phases = {}
        for phase in ("reasoning_first", "reasoning_later", "answer"):
            traces = [t for r in records[arm] for t in r["trace"] if
                      ("answer" if t["phase"] == "answer" else "reasoning_first" if t["step"] == 0 else "reasoning_later") == phase]
            expert_n = sum(len(t["pass_probability"]) for t in traces)
            phases[phase] = {"positions": len(traces), "expert_positions": expert_n,
                "mean_pass_probability": sum(sum(t["pass_probability"]) for t in traces)/expert_n if expert_n else None,
                "pass_top1_expert_positions": sum(len(t["abstaining_experts"]) for t in traces),
                "amateur_fallback_positions": sum(t["amateur_fallback"] for t in traces),
                "bounded_vs_exclude_token_changes_same_prefix": sum(t["counterfactual_same_prefix"]["bounded"]["token_id"] !=
                    t["counterfactual_same_prefix"]["exclude"]["token_id"] for t in traces)}
        n = len(rows)
        conditions[arm] = {"n": n, "missing": 0, "correct": sum(r["subspan_em"] for r in records[arm]),
            "micro": sum(r["subspan_em"] for r in records[arm])/n,
            "macro5": sum(g["subspan_em"] for g in groups.values())/5,
            "empty_answer": sum(r["empty_answer"] for r in records[arm]), "datasets": groups,
            "phase_diagnostics": phases, "seconds": sum(r["seconds"] for r in records[arm])}
    for i, left in enumerate(a.arms):
        for right in a.arms[i+1:]:
            delta = [y["subspan_em"]-x["subspan_em"] for x, y in zip(records[left], records[right])]
            rng = random.Random(42)
            draws = sorted(sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(2000))
            paired[f"{right}_vs_{left}"] = {"n": len(delta), "delta_pp": 100*sum(delta)/len(delta),
                "c_to_w": delta.count(-1), "w_to_c": delta.count(1),
                "prediction_changed": sum(x["prediction"] != y["prediction"] for x, y in zip(records[left], records[right])),
                "descriptive_paired_bootstrap_95ci_pp": [100*draws[49], 100*draws[1949]]}
    eligible = len(rows) == 50 and all(x in conditions for x in ("bounded", "prompt_only", "pced"))
    gate = (conditions["bounded"]["correct"]-conditions["prompt_only"]["correct"] >= 3 and
            conditions["bounded"]["correct"] >= conditions["pced"]["correct"] and
            conditions["bounded"]["empty_answer"] <= 1) if eligible else None
    summary = {"contract_hash": h, "conditions": conditions, "paired": paired,
               "numeric_pilot_gate_passed": gate,
               "note": "Development pilot only. Numeric gate is insufficient without qualitative grounding review. Historical TEST scores are not matched baselines."}
    paper.atomic_json(a.output_root/"summary.json", summary)
    path = a.output_root/"summary.csv"
    temp = path.with_suffix(".csv.tmp")
    temp.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temp.replace(path)
    return summary


def main(argv=None):
    a = arguments(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("Physical GPU 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    stages = [(arm, dataset) for arm in a.arms for dataset in paper.DATASETS]
    names = ["dev contract/tokenize", "load Llama"]+[f"{arm}/{ds} dev" for arm, ds in stages]+["paired summary"]
    progress = PilotProgress(names)
    progress.weights[-1] = 1.
    active = "preflight"
    try:
        progress.stage(0, 5*a.limit_per_dataset, unit="q")
        rows, tokenizer, pass_ids, manifest = preflight(a, progress)
        h = manifest["contract_hash"]
        counts = {}
        for stage, (arm, ds) in enumerate(stages, 2):
            subset = [r for r in rows if r["dataset"] == ds]
            cached = [common.cached_record(a, arm, r, h) for r in subset]
            count = sum(r is not None for r in cached)
            counts[(arm, ds)] = count
            progress.cached_fractions[stage] = count/len(subset)
            timings = [r["seconds"] for r in cached if r is not None]
            if timings:
                progress.weights[stage] = sum(timings)/len(timings)*len(subset)
        progress.complete(f"DEV N={len(rows)}; Top90; cached={sum(counts.values())}/{len(rows)*len(a.arms)}; {a.output_root}")
        progress.log(f"[plan] dev contract -> load one Llama -> {','.join(a.arms)} -> paired summary; GPU1; no retrieval/training")
        progress.log(f"[scope] 91 batched streams; lambda={a.pass_penalty}; gamma=2.5; reasoning<={a.reasoning_tokens}, answer<={a.answer_tokens}; one forward/token; all-PASS -> amateur")
        progress.log("[ETA] 50 dev x4: provisional 30-120min at 50-160 positions/q and prior ~0.18-0.23s/position; no matched bounded-gate measurement. First 5 fresh questions calibrate; model/cache stages measured separately.")
        if a.dry_run:
            progress.log(f"[dry-run OK] dev/test IDs+literal questions disjoint; 50-query source cache contracts, tokenizer, context, resume checked; max_tokens={manifest['max_prompt_plus_output_tokens']}; estimated KV={manifest['estimated_peak_kv_gib']:.1f}GiB; no weights/GPU/output writes")
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
        progress.complete("frozen Llama; same model and 91 streams for all four arms")
        sys.path.insert(0, str(ROOT/"cache/pced_official"))
        from src.methods.pced import prefill, get_jsd
        fresh_times = []
        for stage, (arm, ds) in enumerate(stages, 2):
            subset = [r for r in rows if r["dataset"] == ds]
            progress.stage(stage, len(subset), initial=counts[(arm, ds)], unit="q")
            for row in subset:
                active = f"{arm}/{ds}/{row['qid']}"
                if common.cached_record(a, arm, row, h) is not None:
                    continue
                started = time.monotonic()
                layout = row["layouts"]["base" if arm == "pced" else "pass"]
                context, suffix = common.batch_inputs(layout, tokenizer, model.device)
                prior = paper.prior_tensor(row["scores"], "paper", model.device, model.dtype)
                cache = prefill(model, context)
                stream = method.Stream(model, cache, context, suffix, prior, pass_ids, get_jsd, arm,
                                       penalty=a.pass_penalty, special_ids=tokenizer.all_special_ids)
                result = method.generate(stream, tokenizer, reasoning_tokens=a.reasoning_tokens, answer_tokens=a.answer_tokens)
                result.update(contract_hash=h, arm=arm, dataset=ds, qid=row["qid"], question=row["question"],
                    answers=row["answers"], qrels_for_analysis_only=row["qrels"],
                    subspan_em=paper.subspan_em(result["prediction"], row["answers"]),
                    top90_pids=row["scores"]["pids"], retrieval_prior=prior.float().cpu().tolist(),
                    seconds=time.monotonic()-started)
                paper.atomic_json(common.output_path(a, arm, row), result)
                fresh_times.append(result["seconds"])
                del stream, cache, context, suffix, prior, result
                progress.update()
                if len(fresh_times) >= 5:
                    for later in range(stage+1, len(names)-1):
                        if progress.weights[later] is None:
                            progress.weights[later] = sum(fresh_times[-5:])/5*a.limit_per_dataset
            progress.complete(f"{arm}/{ds}={len(subset)}/{len(subset)}; {a.output_root/arm/'predictions'/ds}")
        progress.stage(len(names)-1, 1)
        summary = summarize(a, rows, h)
        progress.complete(str(a.output_root/"summary.json"))
        progress.log("[result] "+"; ".join(f"{arm}={v['correct']}/{v['n']}" for arm, v in summary["conditions"].items()))
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={active}; completed={progress.done}/{progress.total}; remaining={progress.total-progress.done}; {type(exc).__name__}: {exc}; durable={a.output_root}; identical command resumes committed questions")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

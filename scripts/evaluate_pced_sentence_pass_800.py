#!/usr/bin/env python3
"""Frozen reasoning-v2 / sentence-PASS on LOFT500 + the existing medical300."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_loft_pced_sentence_pass as pilot
from scripts import run_medical_pced_100 as medical
from medrag.generation import pced_sentence_reuse as reuse

paper, common, method = pilot.paper, pilot.common, pilot.method
DATASETS = tuple(paper.DATASETS)+tuple(medical.DATASETS)
VERSION = "sentence_pass_reasoning_v2_800_reuse_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loft-reference", type=Path, default=ROOT/"results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--medical-reference", type=Path, default=ROOT/"results/medical_pced_seeded100_v1")
    p.add_argument("--model", type=Path, default=ROOT.parent/"models/Llama-3.1-8B-Instruct")
    p.add_argument("--output-root", type=Path, default=ROOT/"results/pced_sentence_pass_reasoning_v2_800_v1")
    p.add_argument("--arms", nargs="+", choices=method.ARMS, default=list(method.ARMS))
    p.add_argument("--judge-batch-size", type=int, default=16)
    p.add_argument("--independent-arms", action="store_true", help="Diagnostic: disable cross-arm reuse; use a separate output root")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.judge_batch_size <= 90:
        p.error("judge-batch-size=1..90")
    for key in ("loft_reference", "medical_reference", "model", "output_root"):
        setattr(a, key, getattr(a, key).resolve())
    for source in (a.loft_reference, a.medical_reference, a.model, ROOT/"datasets", medical.DATA_ROOT):
        if a.output_root == source or source in a.output_root.parents or a.output_root in source.parents:
            p.error("Output must not overlap a source/model/data directory")
    a.arms = list(dict.fromkeys(a.arms))
    a.retrieval_root, a.seed = a.loft_reference, 42
    a.reasoning_style, a.reasoning_tokens, a.answer_tokens = "evidence_v2", 192, 32
    a.contribution_weight, a.reuse_arm_prefixes = 4., not a.independent_arms
    return a


def load_rows(a):
    rows, sources = [], {}
    for dataset in DATASETS:
        folder = (a.loft_reference if dataset in paper.DATASETS else a.medical_reference)/dataset
        mp = folder/"manifest.json"
        manifest = json.loads(mp.read_text())
        if (manifest.get("questions") != 100 or manifest.get("top_k") != 90 or
                manifest.get("score_policy") != "paper" or manifest.get("official_pced_commit") != paper.OFFICIAL_SHA):
            raise RuntimeError(f"Incompatible frozen retrieval: {mp}")
        label_path = None
        chunk_labels = None
        if dataset in paper.DATASETS:
            base = ROOT/"datasets/benchmark/loft/rag"/dataset/"128k"
            cp, qp = base/"corpus.jsonl", base/"test_queries.jsonl"
            corpus, queries = list(paper.jsonl(cp)), list(paper.jsonl(qp))
            if manifest.get("candidate_mode") != "all" or manifest.get("split") != "test":
                raise RuntimeError("Expected original LOFT TEST retrieval")
        else:
            if manifest.get("version") != medical.VERSION or manifest.get("candidate_mode") != "faiss":
                raise RuntimeError("Expected medical seeded100 FAISS/CE cache")
            selector = argparse.Namespace(dataset=dataset, fetch_k=manifest["fetch_k"], sample_seed=manifest["sample_seed"])
            cp, qp, corpus, queries = medical.selected_inputs(selector)
            if [str(q["qid"]) for q in queries] != manifest["sample_qids"]:
                raise RuntimeError("Medical cohort differs from existing 300 questions")
            label_path = medical.DATASETS[dataset]/"chunk_qrels.jsonl"
            if label_path.exists():
                chunk_labels = {str(r["qid"]): r for r in paper.jsonl(label_path)}
        hashes = [paper.sha(cp), paper.sha(qp)]
        if hashes != [manifest["corpus_sha256"], manifest["queries_sha256"]]:
            raise RuntimeError(f"Corpus/questions changed: {dataset}")
        by_pid = {str(d["pid"]): d for d in corpus}
        if len(by_pid) != len(corpus) or len(queries) != 100 or len({str(q['qid']) for q in queries}) != 100:
            raise RuntimeError(f"Unexpected IDs/count: {dataset}")
        sources[dataset] = {"manifest_sha256": paper.sha(mp), "corpus_sha256": hashes[0],
                            "queries_sha256": hashes[1], "retrieval_contract": manifest,
                            "chunk_qrels_sha256": paper.sha(label_path) if chunk_labels is not None else None}
        for q in queries:
            qid = str(q["qid"])
            paths = [paper.result_file(folder/name, qid) for name in ("rankings", "expert_scores")]
            rank, scores = [json.loads(p.read_text()) for p in paths]
            pids = [str(r["pid"]) for r in rank["top90"]]
            if (rank["qid"] != qid or scores["qid"] != qid or len(pids) != 90 or len(set(pids)) != 90
                    or scores["pids"] != pids or any(pid not in by_pid for pid in pids)):
                raise RuntimeError(f"Top90 alignment error: {dataset}/{qid}")
            for field in ("bge_dense_pair_raw", "ce_normalized"):
                if len(scores[field]) != 90 or not all(math.isfinite(float(v)) for v in scores[field]):
                    raise RuntimeError("Nonfinite/missing retrieval scores")
            if scores["ce_normalized"] != [r["rerank_score_fp16"] for r in rank["top90"]]:
                raise RuntimeError("Ranking/prior CE score mismatch")
            if not q.get("query_text") or not q.get("answers"):
                raise RuntimeError("Missing question/answer labels")
            qrels = q.get("metadata", {}).get("qrels", [])
            if chunk_labels is not None:
                qrels = [[pid, 1] for pid in chunk_labels[qid]["gold_chunk_ids"]]
            rows.append({"dataset": dataset, "qid": qid, "question": q["query_text"], "answers": q["answers"],
                         "qrels": qrels, "scores": scores, "texts": [paper.format_doc(by_pid[pid]) for pid in pids],
                         "source_sha256": [paper.sha(path) for path in paths]})
    if len(rows) != 800:
        raise RuntimeError("Expected exactly LOFT500 + medical300")
    return rows, sources


def cached(a, arm, row, h):
    p = common.output_path(a, arm, row)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if any(r.get(k) != v for k, v in {"contract_hash": h, "arm": arm, "dataset": row["dataset"], "qid": row["qid"]}.items()):
        raise RuntimeError(f"Incompatible result: {p}")
    if r["answers"] != row["answers"] or r["subspan_em"] != pilot.score_prediction(r, row["answers"]):
        raise RuntimeError(f"Invalid labels/score: {p}")
    return r


def summarize(a, rows, h):
    # Stream large token/expert traces one question at a time, not 2400 in RAM.
    groups, metrics, compact = {}, {}, {}
    for arm in a.arms:
        counts, stats, small = {ds: Counter() for ds in DATASETS}, Counter(), []
        for row in rows:
            r = cached(a, arm, row, h)
            if r is None:
                raise RuntimeError(f"Cannot summarize incomplete cohort: {arm}/{row['qid']}")
            counts[row["dataset"]].update(n=1, correct=r["subspan_em"])
            stats.update(empty_answer=int(r["empty_answer"]), empty_reasoning=int(r["empty_reasoning"]),
                reasoning_budget=int(r["reasoning_stop"] == "token_budget"),
                natural_answer_transitions=int(r["reasoning_stop"] == "answer_marker"),
                answer_budget=int(r["answer_status"] == "token_budget"),
                answer_skipped_incomplete_reasoning=int(r["answer_status"] == "skipped_incomplete_reasoning"),
                reasoning_tokens=len(r["reasoning_token_ids"]), generation_positions=len(r["trace"]),
                reused_generation_positions=r["reused_generation_positions"], seconds=r["seconds"],
                judge_seconds=r["judge_seconds"], amateur_fallback_positions=sum(t["amateur_fallback"] for t in r["trace"]))
            for event in r["decisions"]:
                if event["decision_evaluated"]:
                    stats.update(decision_events=1, reused_decision_events=int(event.get("exact_prefix_cache_hit", False)),
                                 expert_abstentions=90-len(event["active_experts"]))
            for left, right in zip(r["decisions"], r["decisions"][1:]):
                stats.update(expert_reentries=len(set(right["active_experts"])-set(left["active_experts"])),
                             expert_exits=len(set(left["active_experts"])-set(right["active_experts"])))
            small.append({"dataset": row["dataset"], "correct": r["subspan_em"], "prediction": r["prediction"]})
        groups[arm] = {ds: dict(c, subspan_em=c["correct"]/c["n"]) for ds, c in counts.items()}
        metrics[arm] = dict(stats, n=len(rows), missing=0, correct=sum(s["correct"] for s in small),
            micro=sum(s["correct"] for s in small)/len(rows), macro8=sum(g["subspan_em"] for g in groups[arm].values())/8,
            loft_macro5=sum(groups[arm][ds]["subspan_em"] for ds in paper.DATASETS)/5,
            medical_macro3=sum(groups[arm][ds]["subspan_em"] for ds in medical.DATASETS)/3,
            mean_reasoning_tokens=stats["reasoning_tokens"]/len(rows), datasets=groups[arm])
        compact[arm] = small
    paired = {}
    for i, left in enumerate(a.arms):
        for right in a.arms[i+1:]:
            ds_results = {}
            for ds in (*DATASETS, "ALL"):
                pairs = [(x, y) for x, y in zip(compact[left], compact[right]) if ds == "ALL" or x["dataset"] == ds]
                delta = [y["correct"]-x["correct"] for x, y in pairs]
                rng = random.Random(42)
                draws = sorted(100*sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(2000))
                ds_results[ds] = {"n": len(delta), "c_to_w": delta.count(-1), "w_to_c": delta.count(1),
                    "delta_pp": 100*sum(delta)/len(delta), "paired_bootstrap_95ci_pp": [draws[49], draws[1949]],
                    "prediction_changed": sum(x["prediction"] != y["prediction"] for x, y in pairs)}
            paired[right+"_vs_"+left] = ds_results
    result = {"contract_hash": h, "conditions": metrics, "paired": paired,
              "note": "LOFT test previously inspected + medical train-origin pilots. Frozen exploratory evaluation, not untouched final test. Timings allocate shared compute to the first arm; do not compare per-arm latency as independent runs."}
    paper.atomic_json(a.output_root/"summary.json", result)
    lines = ["dataset,arm,n,correct,subspan_em"]
    for arm in a.arms:
        for ds, g in groups[arm].items():
            lines.append(f"{ds},{arm},{g['n']},{g['correct']},{g['subspan_em']}")
    temp = a.output_root/"summary.csv.tmp"
    temp.write_text("\n".join(lines)+"\n")
    temp.replace(a.output_root/"summary.csv")
    return result


def main(argv=None):
    a = arguments(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("Physical GPU1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    names = ["800 contract/tokenize", "load Llama / KV safety"]+[f"{ds} all arms" for ds in DATASETS]+["paired summary"]
    progress = pilot.PilotProgress(names)
    progress.weights[-1] = 10.
    active = "preflight"
    done = {}
    try:
        progress.stage(0, 800, unit="q")
        prepared = load_rows(a)
        rows, tokenizer, labels, pass_ids, manifest = pilot.preflight(a, progress, prepared=prepared,
            contract_extra={"version": VERSION, "split": "LOFT TEST500 previously inspected + frozen medical train-origin300",
                "dataset_counts": {ds: 100 for ds in DATASETS}, "medical_reference": str(a.medical_reference),
                "retrieval": "LOFT full-corpus CE Top90; medical frozen FAISS Top256 -> CE Top90; no new search",
                "runtime": "exact same-prefix arm reuse with independent causal KV fork at divergence" if a.reuse_arm_prefixes else "independent arms",
                "execution_order": "weighted, exclude, pced; no token/prior/prompt/model change; one model and 91 batched streams",
                "runtime_parity": "before evaluation compare live-model independent vs forked-cache next token and expert scores; abort on mismatch",
                "comparison_scope": "All 800 use Llama/Top90/gamma2.5, NOT old medical Qwen/Top10/gamma4. Across arms only gating differs. Prior medical results are not controlled baselines.",
                "primary": "sentence_weighted minus pced Macro8; dataset results, LOFT Macro5, medical Macro3 and paired C/W",
                "pilot_gate": "Frozen evaluation at user request; no automatic tuning/selection/expansion. Practical target +2 percentage points Macro8 with uncertainty and no increased incomplete-answer rate; not a preregistered confirmatory test."},
            extra_paths=(Path(__file__), Path(reuse.__file__), Path(medical.__file__), ROOT/"scripts/run_pced_sentence_pass_800.sh"))
        h = manifest["contract_hash"]
        done, saved_times = {}, {ds: [] for ds in DATASETS}
        for row in rows:
            found = {arm: cached(a, arm, row, h) for arm in a.arms}
            done[(row["dataset"], row["qid"])] = {arm for arm, r in found.items() if r is not None}
            if all(r is not None for r in found.values()):
                saved_times[row["dataset"]].append(sum(r["seconds"] for r in found.values()))
            del found
        completed = sum(len(v) == len(a.arms) for v in done.values())
        for j, ds in enumerate(DATASETS, 2):
            progress.cached_fractions[j] = sum(len(done[(r["dataset"], r["qid"])]) == len(a.arms) for r in rows if r["dataset"] == ds)/100
            if saved_times[ds]:
                progress.weights[j] = 100*sum(saved_times[ds])/len(saved_times[ds])
        progress.complete(f"800 questions, {len(a.arms)*800} question-arms, complete questions={completed}; max tokens={manifest['max_generation_tokens']}; peak KV estimate={manifest['estimated_peak_kv_gib']:.1f}GiB; {a.output_root}")
        progress.log("[plan] fixed corpus/cohort checks -> one frozen Llama + cache-fork parity -> 8 datasets -> paired summary; GPU1 only; no retrieval/training/Joint")
        progress.log("[runtime] 91 expert streams batched; judge batch16 by default; retain judge prefixes; reuse exact CoT decisions and generated prefixes across arms; independent KV after first divergent token; question-arm atomic resume")
        progress.log("[ETA] DEV50 old three-arm run was 25m49s (~6.9h/800 before longer reasoning). Budget 3-7h provisionally; cap-heavy outputs may take longer. First 5 fresh bundles calibrate a context-length cost estimate for pending datasets; each dataset updates its own measured rate.")
        if a.dry_run:
            progress.log(f"[dry-run OK] all 800 fixed IDs/ranks, source hashes, model/tokenizer, labels={labels}, contexts/resume; required free VRAM={manifest['min_free_gib_before_model']:.1f}GiB; no weights/GPU/result writes")
            return
        if not (a.output_root/"manifest.json").exists():
            paper.atomic_json(a.output_root/"manifest.json", manifest)
        progress.stage(1, 1)
        model = None
        if completed < 800:
            import torch
            from transformers import AutoModelForCausalLM
            paper.gpu_gate(manifest["min_free_gib_before_model"])
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(a.seed)
            torch.manual_seed(a.seed)
            model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        sys.path.insert(0, str(ROOT/"cache/pced_official"))
        from src.methods.pced import prefill, get_jsd
        if model is not None:
            first = next(r for r in rows if len(done[(r["dataset"], r["qid"])]) < len(a.arms))
            ctx, suffix = common.batch_inputs(first["layout"], tokenizer, model.device)
            prior = paper.prior_tensor(first["scores"], "paper", model.device, model.dtype)
            safety = reuse.check_live_fork(model, ctx, suffix, prior, pass_ids, get_jsd, prefill, tokenizer.all_special_ids)
            paper.atomic_json(a.output_root/"runtime_safety.json", dict(safety, contract_hash=h))
            del ctx, suffix, prior
        progress.complete("same model/precision/batch; runtime KV parity passed or fully cached")
        calibration = []
        def cost_proxy(r):
            # Same model/91 streams/three policies. This is an explicit length
            # cost proxy for still-unmeasured datasets, not a matched runtime.
            return max(map(len, r["layout"]["context_ids"]))+a.reasoning_tokens
        for stage, ds in enumerate(DATASETS, 2):
            subset = [r for r in rows if r["dataset"] == ds]
            initial = sum(len(done[(ds, r["qid"])]) == len(a.arms) for r in subset)
            progress.stage(stage, 100, initial=initial, unit="q")
            for row in subset:
                todo = [arm for arm in ("sentence_weighted", "sentence_exclude", "pced") if arm in a.arms and arm not in done[(ds, row["qid"])]]
                if not todo:
                    continue
                active = f"{ds}/{row['qid']}"
                bundle_started = time.monotonic()
                anchor = judge = memo = None
                context, suffix = common.batch_inputs(row["layout"], tokenizer, model.device)
                prior = paper.prior_tensor(row["scores"], "paper", model.device, model.dtype)
                for arm in todo:
                    active = f"{ds}/{row['qid']}/{arm}"
                    started = time.monotonic()
                    judge_prefill = 0.
                    if arm != "pced" and (judge is None or not a.reuse_arm_prefixes):
                        if not a.reuse_arm_prefixes:
                            judge = memo = None
                        judge = method.CachedJudge(model, tokenizer, row["judge_ids"], row["judge_tail"], labels, a.judge_batch_size)
                        judge_prefill = judge.prefill_seconds
                        memo = reuse.MemoJudge(judge) if a.reuse_arm_prefixes else judge
                    judge_call = None if arm == "pced" else memo
                    if anchor is None or not a.reuse_arm_prefixes:
                        cache = prefill(model, context)
                        stream = method.Stream(model, cache, context, suffix, prior, pass_ids, get_jsd, arm,
                                               judge_call, weight=a.contribution_weight, special_ids=tokenizer.all_special_ids)
                        del cache
                    else:
                        stream = reuse.ReplayStream(model, None, context, suffix, prior, pass_ids, get_jsd, arm,
                            judge_call, anchor=anchor, weight=a.contribution_weight, special_ids=tokenizer.all_special_ids)
                    result = method.generate(stream, tokenizer, reasoning_tokens=192, answer_tokens=32, force_answer_at_budget=False)
                    result.update(contract_hash=h, arm=arm, dataset=ds, qid=row["qid"], question=row["question"],
                        answers=row["answers"], qrels_for_analysis_only=row["qrels"],
                        subspan_em=pilot.score_prediction(result, row["answers"]),
                        top90_pids=row["scores"]["pids"], retrieval_prior=prior.float().cpu().tolist(),
                        reused_generation_positions=getattr(stream, "reused_steps", 0),
                        judge_seconds=judge_prefill+sum(e["seconds"] for e in result["decisions"]),
                        seconds=time.monotonic()-started)
                    paper.atomic_json(common.output_path(a, arm, row), result)
                    done[(ds, row["qid"])].add(arm)
                    if anchor is None and a.reuse_arm_prefixes:
                        anchor = stream
                    del stream, result
                del anchor, judge, memo, context, suffix, prior
                if len(todo) == len(a.arms):
                    calibration.append((time.monotonic()-bundle_started)/cost_proxy(row))
                    if len(calibration) >= 5:
                        unit_cost = sum(calibration[-5:])/5
                        for j, later in enumerate(DATASETS, 2):
                            if j > stage and not saved_times[later]:
                                progress.weights[j] = unit_cost*sum(cost_proxy(r) for r in rows if r["dataset"] == later)
                progress.update()
            progress.complete(f"{ds}=100/100, all requested arms durable; {a.output_root}")
        progress.stage(len(names)-1, 1)
        result = summarize(a, rows, h)
        progress.update()
        progress.complete(str(a.output_root/"summary.json"))
        progress.log("[result] "+"; ".join(f"{arm}={v['correct']}/{v['n']}, Macro8={v['macro8']:.4f}" for arm, v in result["conditions"].items()))
    except BaseException as exc:
        complete_arms = sum(len(v) for v in done.values())
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={active}; stage_done={progress.done}/{progress.total}; remaining={progress.total-progress.done}; durable_arms={complete_arms}/{800*len(a.arms)} remaining_arms={800*len(a.arms)-complete_arms}; {type(exc).__name__}: {exc}; durable={a.output_root}; identical command resumes completed question-arms")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

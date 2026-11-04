#!/usr/bin/env python3
"""Fixed LOFT Top-90 / Llama direct-answer dynamic PASS pilot, GPU 1 only.

No training, retrieval, extra decision forward, CoT or test-set tuning.
Paper predictions are reused. Two new arms isolate prompting from gating.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts import run_loft_pced_paper as paper
from medrag.training.graph_input_progress import Progress

VERSION = "loft_dynamic_pass_v1"
ARMS = ("prompt_only", "pass")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path,
                   default=ROOT / "results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--limit-per-dataset", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pass-penalty", type=float, default=1.0)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.limit_per_dataset <= 100:
        p.error("--limit-per-dataset must be 1..100")
    if not math.isfinite(a.pass_penalty) or a.pass_penalty < 0:
        p.error("--pass-penalty must be finite and nonnegative")
    a.reference_root = a.reference_root.resolve()
    a.output_root = (a.output_root or ROOT / "results" /
                     f"loft_llama_dynamic_pass_v1_n{5*a.limit_per_dataset}").resolve()
    if (a.output_root == a.reference_root or a.output_root in a.reference_root.parents or
            a.reference_root in a.output_root.parents):
        p.error("Output and immutable reference roots must be separate")
    a.arms = list(dict.fromkeys(a.arms))
    return a


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prompts(tokenizer, question, texts, rules):
    # Preserve the original amateur and the original question/Answer: suffix.
    prefixes, suffix = paper.prompt_inputs(tokenizer, question, texts)
    for i, text in enumerate(texts, start=1):
        separator = "\n" + "#" * (len(text) + 1)
        rendered = tokenizer.apply_chat_template([
            {"role": "system", "content": paper.SYSTEM + "\n\n" + rules},
            {"role": "user", "content": text + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prefix, trailing = rendered.split(separator)
        if paper.QUESTION.format(question=question) + trailing + paper.ANSWER_PREFIX != suffix:
            raise RuntimeError("Unexpected expert/amateur chat suffix mismatch")
        prefixes[i] = prefix
    return prefixes, suffix


def preflight(a, progress):
    from transformers import AutoTokenizer
    from medrag.generation.pced_pass import PASS_RULES, pass_token_ids
    official = ROOT / "cache/pced_official"
    revision = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError("Official PCED revision changed")
    rows, references, source_model = [], {}, None
    for dataset in paper.DATASETS:
        base = a.reference_root / dataset
        ref = json.loads((base / "manifest.json").read_text())
        summary = json.loads((base / "summary.json").read_text())
        if (ref.get("top_k") != 90 or ref.get("max_new_tokens") != 32 or
                ref.get("candidate_mode") != "all" or ref.get("score_policy") != "paper" or
                ref.get("attention_impl") != "sdpa" or ref.get("questions") != 100 or
                summary.get("n") != 100 or summary.get("missing") != 0):
            raise RuntimeError(f"Unmatched direct-answer reference: {dataset}")
        if source_model is None:
            source_model = Path(ref["llama"])
        if str(source_model) != ref["llama"] or source_model.name != "Llama-3.1-8B-Instruct":
            raise RuntimeError("This pilot must use the reference Llama-3.1-8B checkpoint")
        source = paper.arguments(["--phase", "preflight", "--dataset", dataset])
        cp, qp, corpus, queries = paper.load_inputs(source)
        if paper.sha(cp) != ref["corpus_sha256"] or paper.sha(qp) != ref["queries_sha256"]:
            raise RuntimeError(f"Reference input changed: {dataset}")
        by_pid = {str(doc["pid"]): doc for doc in corpus}
        chosen = sorted(queries, key=lambda q: digest([a.seed, dataset, str(q["qid"])]))[:a.limit_per_dataset]
        references[dataset] = {"manifest_sha256": paper.sha(base / "manifest.json"),
                               "question_sha256": paper.sha(qp), "corpus_sha256": paper.sha(cp)}
        for q in chosen:
            qid = str(q["qid"])
            rank_path = paper.result_file(base / "rankings", qid)
            score_path = paper.result_file(base / "expert_scores", qid)
            pred_path = paper.result_file(base / "predictions", qid)
            ranking = json.loads(rank_path.read_text())
            scores = json.loads(score_path.read_text())
            original = json.loads(pred_path.read_text())
            pids = [str(item["pid"]) for item in ranking["top90"]]
            if (len(pids) != 90 or len(set(pids)) != 90 or scores["pids"] != pids or
                    original["qid"] != qid or original["dataset"] != dataset or
                    original.get("answers") != q["answers"] or
                    original["subspan_em"] != paper.subspan_em(original["prediction"], q["answers"])):
                raise RuntimeError(f"Reference alignment failed: {dataset}/{qid}")
            for key in ("bge_dense_pair_raw", "ce_normalized"):
                if len(scores[key]) != 90 or not all(math.isfinite(float(v)) for v in scores[key]):
                    raise RuntimeError(f"Invalid {key}: {dataset}/{qid}")
            rows.append({"dataset": dataset, "qid": qid, "question": q["query_text"],
                         "answers": q["answers"], "texts": [paper.format_doc(by_pid[x]) for x in pids],
                         "scores": scores, "original": original,
                         "source_sha256": [paper.sha(p) for p in (rank_path, score_path, pred_path)]})
    # No model weights/GPU load in preflight. Check checkpoint shards first.
    index = source_model / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    for name in (*shards, "config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (source_model / name).is_file():
            raise FileNotFoundError(source_model / name)
    tokenizer = AutoTokenizer.from_pretrained(str(source_model), local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    pass_ids = pass_token_ids(tokenizer)
    config = json.loads((source_model / "config.json").read_text())
    max_positions = config["max_position_embeddings"]
    lengths = []
    for row in rows:
        prefixes, suffix = prompts(tokenizer, row["question"], row["texts"], PASS_RULES)
        contexts = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        length = max(map(len, contexts)) + len(suffix_ids) + 32
        if length > max_positions:
            raise RuntimeError(f"Context overflow (no truncation allowed): {row['qid']} {length}")
        row["context_ids"], row["suffix_ids"] = contexts, suffix_ids
        lengths.append(length)
        progress.update()
    decoder = ROOT / "medrag/generation/pced_pass.py"
    contract = {
        "version": VERSION, "hypothesis": "Dynamic same-forward abstention improves PCED expert selection",
        "split": "existing LOFT test diagnostic, not an untouched validation set; no tuning",
        "n": len(rows), "seed": a.seed, "arms": a.arms,
        "cohort": [{"dataset": r["dataset"], "qid": r["qid"], "sources": r["source_sha256"]} for r in rows],
        "reference_root": str(a.reference_root), "references": references,
        "model": str(source_model), "model_config_sha256": paper.sha(source_model / "config.json"),
        "model_shards": {s: {"bytes": (source_model/s).stat().st_size,
                             "mtime_ns": (source_model/s).stat().st_mtime_ns} for s in sorted(shards)},
        "tokenizer_sha256": paper.sha(source_model / "tokenizer.json"),
        "tokenizer_config_sha256": paper.sha(source_model / "tokenizer_config.json"),
        "system": paper.SYSTEM, "question_template": paper.QUESTION, "answer_prefix": paper.ANSWER_PREFIX,
        "pass_rules": PASS_RULES, "pass_token_ids": list(pass_ids),
        "amateur": "unchanged original paper prompt, contrastive reference only",
        "pass_policy": "raw argmax in PASS IDs => exclude this step; summed raw PASS mass => soft penalty",
        "all_pass_policy": "restore experts for this step; keep penalty; select non-PASS token (logged)",
        "prompt_only_policy": "same PASS prompt; mask PASS output; no exclusion or penalty",
        "pass_penalty": a.pass_penalty, "epsilon": 1e-6, "gamma": 2.5,
        "beta": "official vocabulary-wise JSD on first generation step; held fixed",
        "top_k": 90, "retrieval": "unchanged reference Top90 IDs/order and CE+BGE prior",
        "generation": {"greedy": True, "max_new_tokens": 32, "backend": "sdpa", "dtype": "bfloat16"},
        "metric": "unchanged paper Subspan EM", "gpu": 1,
        "code_sha256": {str(p.relative_to(ROOT)): paper.sha(p) for p in
                        (Path(__file__), decoder, Path(paper.__file__), official / "src/methods/pced.py")},
        "official_commit": revision,
        "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
    }
    contract_hash = digest(contract)
    manifest = {"contract": contract, "contract_hash": contract_hash,
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "max_prompt_plus_output_tokens": max(lengths)}
    path = a.output_root / "manifest.json"
    if path.exists() and json.loads(path.read_text()).get("contract_hash") != contract_hash:
        raise RuntimeError(f"Different experiment in {a.output_root}; select a NEW --output-root")
    if not path.exists() and any(a.output_root.glob("*/predictions/*/*.json")):
        raise RuntimeError("Predictions without a manifest: refuse unsafe resume")
    return rows, tokenizer, source_model, pass_ids, manifest


def output_path(a, arm, row):
    return paper.result_file(a.output_root / arm / "predictions" / row["dataset"], row["qid"])


def cached_record(a, arm, row, contract_hash):
    path = output_path(a, arm, row)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if (record.get("contract_hash") != contract_hash or record.get("qid") != row["qid"] or
            record.get("dataset") != row["dataset"] or record.get("arm") != arm or
            record.get("subspan_em") != paper.subspan_em(record["prediction"], row["answers"])):
        raise RuntimeError(f"Invalid durable result: {path}")
    return record


def batch_inputs(row, tokenizer, device):
    import torch
    length = max(map(len, row["context_ids"]))
    ids = [[tokenizer.pad_token_id] * (length-len(x)) + x for x in row["context_ids"]]
    mask = [[0] * (length-len(x)) + [1] * len(x) for x in row["context_ids"]]
    context = {"input_ids": torch.tensor(ids, device=device),
               "attention_mask": torch.tensor(mask, device=device)}
    suffix = torch.tensor([row["suffix_ids"]] * len(ids), device=device)
    return context, {"input_ids": suffix, "attention_mask": torch.ones_like(suffix)}


def summarize(a, rows, contract_hash):
    conditions = {"paper": [r["original"] for r in rows]}
    for arm in a.arms:
        conditions[arm] = [cached_record(a, arm, r, contract_hash) for r in rows]
        if any(x is None for x in conditions[arm]):
            raise RuntimeError(f"Incomplete {arm}; refusing a misleading full summary")
    results, lines = {}, ["dataset,condition,n,correct,subspan_em,delta_vs_paper_pp,c_to_w,w_to_c"]
    for arm, records in conditions.items():
        groups = {}
        for dataset in paper.DATASETS:
            indices = [i for i, r in enumerate(rows) if r["dataset"] == dataset]
            correct = sum(records[i]["subspan_em"] for i in indices)
            base = sum(rows[i]["original"]["subspan_em"] for i in indices)
            cw = sum(rows[i]["original"]["subspan_em"] and not records[i]["subspan_em"] for i in indices)
            wc = sum(not rows[i]["original"]["subspan_em"] and records[i]["subspan_em"] for i in indices)
            n = len(indices)
            groups[dataset] = {"n": n, "correct": correct, "subspan_em": correct/n,
                               "delta_vs_paper_pp": 100*(correct-base)/n, "c_to_w": cw, "w_to_c": wc}
            lines.append(f"{dataset},{arm},{n},{correct},{correct/n:.6f},{100*(correct-base)/n:.3f},{cw},{wc}")
        correct = sum(r["subspan_em"] for r in records)
        steps = [t for r in records for t in r.get("trace", [])]
        results[arm] = {"n": len(rows), "missing": 0, "correct": correct,
                        "micro_subspan_em": correct/len(rows),
                        "macro5_subspan_em": sum(g["subspan_em"] for g in groups.values())/len(groups),
                        "datasets": groups,
                        "diagnostics": {"generated_steps": len(steps),
                            "steps_with_pass_top1": sum(bool(t["abstaining_experts"]) for t in steps),
                            "all_pass_steps": sum(t["all_pass"] for t in steps),
                            "all_pass_fallback_steps": sum(t["all_pass_fallback"] for t in steps),
                            "winner_changed_same_prefix": sum(t["winner"] != t["ungated_winner_same_prefix"] for t in steps)}}
    pairs = {}
    for arm in a.arms:
        reference_arm = "prompt_only" if arm == "pass" and "prompt_only" in conditions else "paper"
        left, right = conditions[reference_arm], conditions[arm]
        if arm == reference_arm:
            continue
        delta = [(b["subspan_em"] - x["subspan_em"]) for x, b in zip(left, right)]
        # Paired bootstrap, descriptive only; not a test-set selection criterion.
        rng = random.Random(42)
        draws = sorted(sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(2000))
        pairs[f"{arm}_vs_{reference_arm}"] = {
            "n": len(delta), "delta_pp": 100*sum(delta)/len(delta),
            "c_to_w": sum(d == -1 for d in delta), "w_to_c": sum(d == 1 for d in delta),
            "paired_bootstrap_95ci_pp": [100*draws[49], 100*draws[1949]],
            "prediction_changed": sum(x["prediction"] != b["prediction"] for x, b in zip(left, right))}
    result = {"contract_hash": contract_hash, "conditions": results, "paired": pairs,
              "interpretation": "Existing test diagnostic; no hyperparameter selection; no untouched-test claim"}
    paper.atomic_json(a.output_root / "summary.json", result)
    # Small recomputable human-readable table, not a new evaluator.
    table = a.output_root / "summary.csv"
    temporary = table.with_suffix(".csv.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(table)
    return result


def main(argv=None):
    a = arguments(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("This experiment is authorized for physical GPU 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    n = 5 * a.limit_per_dataset
    stages = ["contract / tokenize", "load Llama GPU1"] + [f"{x} direct LOFT" for x in a.arms] + ["paired summary"]
    progress = Progress(stages)
    progress.weights[:] = [None] * len(stages)
    progress.weights[-1] = 1.0
    active, done = None, 0
    try:
        progress.stage(0, n, unit="q")
        rows, tokenizer, model_path, pass_ids, manifest = preflight(a, progress)
        h = manifest["contract_hash"]
        counts = {arm: sum(cached_record(a, arm, r, h) is not None for r in rows) for arm in a.arms}
        progress.complete(f"same fixed Top90; N={n}; PASS={pass_ids}; cached={counts}; {a.output_root}")
        progress.log(f"[plan] contract/tokenize -> load Llama -> {','.join(a.arms)} -> paired summary; "
                     f"cached={sum(counts.values())}, remaining={n*len(a.arms)-sum(counts.values())}; "
                     "no retrieval/training/CoT; one forward per token; GPU1 only")
        progress.log("[policy] raw PASS top1 excludes an expert for this step; log(1-pPASS) penalty; "
                     "all-PASS restores experts for this step (logged); PASS never enters the shared answer")
        progress.log("[ETA] new dynamic-PASS throughput unknown; each arm calibrates on fresh questions; "
                     "past direct PCED was roughly 3-11 seconds/q but excludes added prompt/trace overhead")
        if a.dry_run:
            progress.log("[dry-run OK] paths, source hashes, tokenizer, prompt lengths and resume contracts; no weights/GPU/output loaded or written")
            return
        if not (a.output_root / "manifest.json").exists():
            paper.atomic_json(a.output_root / "manifest.json", manifest)
        progress.stage(1, 1)
        model = None
        if any(v < n for v in counts.values()):
            import torch
            from transformers import AutoModelForCausalLM
            paper.gpu_gate(40)
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(a.seed)
            torch.manual_seed(a.seed)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        progress.complete("one frozen Llama; 91 streams batched, no separate decision model")
        sys.path.insert(0, str(ROOT / "cache/pced_official"))
        from src.methods.pced import prefill, get_jsd
        from medrag.generation.pced_pass import decode
        for stage, arm in enumerate(a.arms, start=2):
            done = counts[arm]
            progress.stage(stage, n, initial=done, unit="q")
            for row in rows:
                active = f"{arm}/{row['dataset']}/{row['qid']}"
                if cached_record(a, arm, row, h) is not None:
                    continue
                started = time.monotonic()
                context, suffix = batch_inputs(row, tokenizer, model.device)
                prior = paper.prior_tensor(row["scores"], "paper", model.device, model.dtype)
                cache = prefill(model, context)
                ids, traces = decode(model, tokenizer, cache, context, suffix, prior, pass_ids, get_jsd,
                                     enabled=arm == "pass", penalty=a.pass_penalty)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                result = {"contract_hash": h, "arm": arm, "dataset": row["dataset"], "qid": row["qid"],
                          "prediction": text, "answers": row["answers"],
                          "subspan_em": paper.subspan_em(text, row["answers"]),
                          "generated_token_ids": ids, "expert_winners": [t["winner"] for t in traces],
                          "trace": traces, "top90_pids": row["scores"]["pids"],
                          "retrieval_prior": prior.float().cpu().tolist(), "seconds": time.monotonic()-started}
                paper.atomic_json(output_path(a, arm, row), result)
                del cache, context, suffix, prior
                done += 1
                progress.update()
            progress.complete(f"{arm}={n}/{n}; {a.output_root/arm}")
        progress.stage(len(stages)-1, 1)
        result = summarize(a, rows, h)
        progress.complete(str(a.output_root / "summary.json"))
        progress.log("[result] " + "; ".join(f"{k}={v['correct']}/{n}" for k, v in result["conditions"].items()))
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(stages)} item={active}; "
                     f"stage_completed={progress.done}/{progress.total}; stage_remaining={progress.total-progress.done}; "
                     f"{type(exc).__name__}: {exc}; durable={a.output_root}; identical command resumes committed question results")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

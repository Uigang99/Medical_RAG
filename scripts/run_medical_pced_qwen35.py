#!/usr/bin/env python3
"""Run frozen medical PCED-100 evidence with local Qwen3.5-9B.

The official PCED decoder, fixed CE Top-90, BGE/CE retrieval prior, prompts,
gamma=2.5, dynamic beta, 32-token greedy output, and Subspan EM are unchanged.
Only the target model and its required cache/logit API adaptation differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.training.graph_input_progress import Progress
from scripts import run_loft_pced_paper as paper
from scripts import run_medical_pced_100 as medical
from scripts.evaluate_medical_pced_100_baselines import make_prompt

VERSION = "medical_pced_qwen35_9b_seeded100_v1"
ARMS = ("pced", "all-ranked", "no-rag", "single", "all")
MODEL = ROOT.parent / "models/Qwen3.5-9B"


class QwenPCEDAdapter:
    """Map the official decoder's Llama logits kwarg to Qwen's equivalent."""

    def __init__(self, base):
        self.base = base
        self.model = base.model
        self.device = base.device
        self.generation_config = base.generation_config

    def __call__(self, *args, num_logits_to_keep=1, **kwargs):
        return self.base(*args, logits_to_keep=num_logits_to_keep, **kwargs)


def qwen_prefill(base, tokenized_context):
    """Official prefill operation with Qwen3.5's config-aware hybrid cache."""
    import torch
    from transformers import DynamicCache
    cache = DynamicCache(config=base.config)
    with torch.inference_mode():
        _ = base.model(**tokenized_context, past_key_values=cache, use_cache=True)
    return cache


def source_data(a, dataset):
    source = a.reference_root / dataset
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Medical retrieval cache missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("version") != medical.VERSION or manifest.get("questions") != 100
            or manifest.get("top_k") != 90 or manifest.get("score_policy") != "paper"
            or manifest.get("official_pced_commit") != paper.OFFICIAL_SHA):
        raise RuntimeError(f"Unexpected source cache contract: {manifest_path}")
    selector = argparse.Namespace(dataset=dataset, fetch_k=manifest["fetch_k"],
                                  sample_seed=manifest["sample_seed"])
    _, _, corpus, queries = medical.selected_inputs(selector)
    qids = [str(q["qid"]) for q in queries]
    if qids != manifest["sample_qids"]:
        raise RuntimeError(f"Source cohort changed: {dataset}")
    by_pid = {str(doc["pid"]): doc for doc in corpus}
    rankings, scores = {}, {}
    fingerprint = hashlib.sha256()
    for q in queries:
        qid = str(q["qid"])
        rank_path = paper.result_file(source / "rankings", qid)
        prior_path = paper.result_file(source / "expert_scores", qid)
        if not rank_path.is_file() or not prior_path.is_file():
            raise FileNotFoundError(f"Fixed Top-90 or expert prior missing: {qid}")
        rank = json.loads(rank_path.read_text())
        score = json.loads(prior_path.read_text())
        pids = [str(item["pid"]) for item in rank["top90"]]
        if (len(pids) != 90 or len(set(pids)) != 90 or score["pids"] != pids
                or len(score["bge_dense_pair_raw"]) != 90
                or len(score["ce_normalized"]) != 90
                or any(pid not in by_pid for pid in pids)):
            raise RuntimeError(f"Invalid fixed Top-90/prior alignment: {dataset}/{qid}")
        rankings[qid], scores[qid] = rank, score
        for path in (rank_path, prior_path):
            fingerprint.update(qid.encode())
            fingerprint.update(paper.sha(path).encode())
    return queries, by_pid, rankings, scores, manifest_path, fingerprint.hexdigest()


def contract(a, dataset, queries, source_manifest, source_digest):
    import importlib.metadata
    return {
        "version": VERSION, "dataset": dataset, "n": 100,
        "sample_qids": [str(q["qid"]) for q in queries],
        "source_manifest_sha256": paper.sha(source_manifest),
        "source_rankings_and_priors_sha256": source_digest,
        "model": str(a.model), "model_config_sha256": paper.sha(a.model / "config.json"),
        "tokenizer_sha256": paper.sha(a.model / "tokenizer.json"),
        "weights_index_sha256": paper.sha(a.model / "model.safetensors.index.json"),
        "model_family": "Qwen3.5-9B, local instruction-tuned checkpoint; not paper Qwen3-8B",
        "arms": list(a.arms), "same_question_prompt": True,
        "retrieval": "frozen BGE-M3 FAISS Top-256 -> CE Top-90",
        "document_order": "CE-ranked except official shuffled All",
        "prior": "same paper-policy BGE/CE harmonic score; no normalization",
        "pced_decoder": "pinned official src.methods.pced.decode_from_cache",
        "pced_decoder_sha256": paper.sha(ROOT / "cache/pced_official/src/methods/pced.py"),
        "official_pced_commit": paper.OFFICIAL_SHA,
        "beta": "first-token JSD, fixed afterwards", "gamma": 2.5,
        "max_new_tokens": 32, "do_sample": False, "seed": 42,
        "chat_template": "model-native; same system/user text, enable_thinking=False",
        "model_api_adaptation": "config-aware DynamicCache; num_logits_to_keep -> logits_to_keep; chat EOS",
        "metric": "paper Subspan EM; gold only for scoring",
        "gpu": "GPU 1 only", "precision": "bfloat16", "backend": "sdpa",
        "environment": {"python": sys.version.split()[0],
                        "torch": importlib.metadata.version("torch"),
                        "transformers": importlib.metadata.version("transformers")},
        "script_sha256": paper.sha(Path(__file__)),
        "code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "output_root": str(a.output_root),
    }


def preflight(a, dataset, *, write=False):
    for filename in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not (a.model / filename).is_file():
            raise FileNotFoundError(a.model / filename)
    index = json.loads((a.model / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (a.model / shard).is_file():
            raise FileNotFoundError(a.model / shard)
    if json.loads((a.model / "config.json").read_text()).get("model_type") != "qwen3_5":
        raise RuntimeError(f"Expected Qwen3.5 checkpoint: {a.model}")
    official = ROOT / "cache/pced_official"
    revision = subprocess.check_output(
        ["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError(f"Official decoder checkout changed: {revision}")
    queries, by_pid, rankings, scores, source_manifest, source_digest = source_data(a, dataset)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Qwen chat EOS is undefined")
    first = queries[0]
    example_pid = rankings[str(first["qid"])]["top90"][0]["pid"]
    paper.prompt_inputs(tokenizer, first["query_text"],
                        [paper.format_doc(by_pid[str(example_pid)])])
    import numpy as np
    for arm in a.arms:
        if arm != "pced":
            make_prompt(tokenizer, argparse.Namespace(arm=arm), first,
                        rankings[str(first["qid"])], by_pid,
                        np.random.default_rng(125))
    expected = contract(a, dataset, queries, source_manifest, source_digest)
    path = a.output_root / dataset / "manifest.json"
    if path.is_file():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(f"Incompatible Qwen run; choose a fresh output root: {path}")
    elif write:
        paper.atomic_json(path, expected)
    for arm in a.arms:
        folder = a.output_root / dataset / arm / "predictions"
        for q in queries:
            path = paper.result_file(folder, str(q["qid"]))
            if path.is_file():
                row = json.loads(path.read_text())
                if row.get("qid") != str(q["qid"]) or row.get("arm") != arm:
                    raise RuntimeError(f"Incompatible Qwen prediction shard: {path}")
    return queries, by_pid, rankings, scores


def summarize(a, dataset, queries):
    import numpy as np
    from scipy.stats import binomtest
    summary = {"version": VERSION, "dataset": dataset, "n": len(queries),
               "metric": "Subspan EM", "arms": {}}
    for arm in a.arms:
        directory = a.output_root / dataset / arm / "predictions"
        rows = [json.loads(paper.result_file(directory, str(q["qid"])).read_text())
                for q in queries]
        correct = sum(row["subspan_em"] for row in rows)
        summary["arms"][arm] = {"correct": correct, "n": len(rows),
                                 "subspan_em": correct / len(rows), "missing": 0,
                                 "mean_seconds_per_question":
                                     sum(row["seconds"] for row in rows) / len(rows)}
    if "pced" in summary["arms"] and "all-ranked" in summary["arms"]:
        pced_dir = a.output_root / dataset / "pced/predictions"
        all_dir = a.output_root / dataset / "all-ranked/predictions"
        paired = [(json.loads(paper.result_file(pced_dir, str(q["qid"])).read_text())["subspan_em"],
                   json.loads(paper.result_file(all_dir, str(q["qid"])).read_text())["subspan_em"])
                  for q in queries]
        diff = np.asarray([x - y for x, y in paired], dtype=int)
        gain, harm = int((diff == 1).sum()), int((diff == -1).sum())
        rng = np.random.default_rng(20261103)
        bootstrap = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(axis=1) * 100
        summary["pced_vs_all_ranked"] = {
            "delta_pp": float(diff.mean() * 100),
            "all_wrong_pced_correct": gain, "all_correct_pced_wrong": harm,
            "paired_bootstrap_ci95_pp": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "mcnemar_exact_p": float(binomtest(min(gain, harm), gain + harm, 0.5).pvalue)
                               if gain + harm else 1.0,
        }
    paper.atomic_json(a.output_root / dataset / "summary.json", summary)
    return summary


def evaluate(a, dataset):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    stages = ["contract/cache", "load frozen Qwen", *[f"{dataset} {arm}" for arm in a.arms],
              "Subspan EM summary"]
    progress = Progress(stages)
    current_qid = None
    try:
        progress.stage(0, 1)
        queries, by_pid, rankings, scores = preflight(a, dataset, write=True)
        tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True,
                                                  padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        # Keep the original model-independent prompt words and output budget.
        # Qwen's own chat template is necessary to encode those words correctly.
        qid0 = str(queries[0]["qid"])
        sample_text = paper.format_doc(by_pid[rankings[qid0]["top90"][0]["pid"]])
        paper.prompt_inputs(tokenizer, queries[0]["query_text"], [sample_text])
        pending = {arm: sum(not paper.result_file(
            a.output_root / dataset / arm / "predictions", str(q["qid"])).is_file()
            for q in queries) for arm in a.arms}
        progress.complete(f"100 fixed questions; pending={pending}; {a.output_root/dataset/'manifest.json'}")
        progress.log(f"[plan] contract/cache -> Qwen load -> "
                     + " -> ".join(f"{arm} {pending[arm]}/100" for arm in a.arms)
                     + " -> summary; GPU 1; ETA calibrating on first fresh Qwen questions")
        progress.log("[scope] Fixed Top90/prior/prompt words/γ2.5/β/32 greedy tokens; "
                     "model-native chat template and cache API only")
        progress.stage(1, 1)
        if any(pending.values()):
            paper.gpu_gate(a.min_free_gib)
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(42)
            torch.manual_seed(42)
            base = AutoModelForCausalLM.from_pretrained(
                str(a.model), local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda:0").eval()
            base.requires_grad_(False)
            stop_ids = {int(tokenizer.eos_token_id)}
            old_eos = base.generation_config.eos_token_id
            if old_eos is not None:
                stop_ids.update(int(x) for x in (old_eos if isinstance(old_eos, list) else [old_eos]))
            base.generation_config.eos_token_id = sorted(stop_ids)
            sys.path.insert(0, str(ROOT / "cache/pced_official"))
            from src.methods.pced import decode_from_cache
            pced_model = QwenPCEDAdapter(base)
        progress.complete("one frozen Qwen model; no LoRA; loaded only if shards remain")

        for stage_no, arm in enumerate(a.arms, start=2):
            folder = a.output_root / dataset / arm / "predictions"
            done = 100 - pending[arm]
            progress.stage(stage_no, 100, initial=done, unit="q")
            rng = np.random.default_rng(125)
            for q in queries:
                current_qid = str(q["qid"])
                path = paper.result_file(folder, current_qid)
                rank = rankings[current_qid]
                if arm != "pced":
                    baseline_args = argparse.Namespace(arm=arm)
                    prompt, pids = make_prompt(tokenizer, baseline_args, q, rank, by_pid, rng)
                if path.is_file():
                    continue
                started = time.monotonic()
                if arm == "pced":
                    selected = rank["top90"]
                    pids = [str(row["pid"]) for row in selected]
                    docs = [paper.format_doc(by_pid[pid]) for pid in pids]
                    prefixes, question = paper.prompt_inputs(tokenizer, q["query_text"], docs)
                    prefix_ids = tokenizer(prefixes, return_tensors="pt", add_special_tokens=False,
                                           padding=True, truncation=True,
                                           max_length=base.config.max_position_embeddings-32).to(base.device)
                    question_ids = tokenizer([question] * len(prefixes), return_tensors="pt",
                                             add_special_tokens=False).to(base.device)
                    prior = paper.prior_tensor(scores[current_qid], "paper", base.device, base.dtype)
                    with torch.inference_mode():
                        cache = qwen_prefill(base, prefix_ids)
                        ids, winners = decode_from_cache(
                            pced_model, tokenizer, cache, prefix_ids, question_ids,
                            max_new_tokens=32, beta=None, retrieval_scores=prior,
                            gamma=2.5, normalize=False, return_winner_trace=True)
                    output_ids = ids.tolist()
                    del cache, prefix_ids, question_ids, prior, ids
                else:
                    token_ids = tokenizer(prompt, return_tensors="pt",
                                          add_special_tokens=False).to(base.device)
                    if token_ids["input_ids"].shape[1] + 32 > base.config.max_position_embeddings:
                        raise RuntimeError(f"Qwen context exceeded: {current_qid}")
                    with torch.inference_mode():
                        generated = base.generate(**token_ids, do_sample=False,
                            max_new_tokens=32, eos_token_id=base.generation_config.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id, use_cache=True)
                    output_ids = generated[0, token_ids["input_ids"].shape[1]:].tolist()
                    winners = None
                    del token_ids, generated
                prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
                paper.atomic_json(path, {
                    "qid": current_qid, "dataset": dataset, "arm": arm,
                    "prediction": prediction, "answers": q["answers"],
                    "subspan_em": paper.subspan_em(prediction, q["answers"]),
                    "generated_token_ids": output_ids, "document_pids": pids,
                    "expert_winners": winners, "seconds": time.monotonic() - started,
                })
                progress.update()
            progress.complete(f"100 predictions ({done} cached); {folder}")
        progress.stage(len(stages)-1, 1)
        summary = summarize(a, dataset, queries)
        progress.complete(f"{a.output_root/dataset/'summary.json'}")
        progress.log(f"[result] {dataset}: " + ", ".join(
            f"{arm}={summary['arms'][arm]['correct']}/100" for arm in a.arms))
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(stages)} qid={current_qid} "
                     f"completed={progress.done}/{progress.total}; durable={a.output_root/dataset}; "
                     f"same command resumes; {type(exc).__name__}: {exc}")
        raise
    finally:
        progress.close()


def summarize_all(a):
    import numpy as np
    reports = {}
    for dataset in medical.DATASETS:
        queries, _, _, _ = preflight(a, dataset, write=False)
        reports[dataset] = summarize(a, dataset, queries)
    macro = {arm: sum(reports[name]["arms"][arm]["subspan_em"]
                      for name in medical.DATASETS) / len(medical.DATASETS)
             for arm in a.arms}
    out = {"version": VERSION, "n": 300, "datasets": reports,
           "macro3_subspan_em": macro, "micro_subspan_em": macro,
           "note": "same 100 train-origin questions per medical dataset; Qwen3.5-9B, not paper Qwen3-8B"}
    if "pced" in a.arms and "all-ranked" in a.arms:
        paired = []
        for dataset in medical.DATASETS:
            queries, _, _, _ = preflight(a, dataset, write=False)
            pced_dir = a.output_root / dataset / "pced/predictions"
            all_dir = a.output_root / dataset / "all-ranked/predictions"
            paired.append(np.asarray([
                json.loads(paper.result_file(pced_dir, str(q["qid"])).read_text())["subspan_em"]
                - json.loads(paper.result_file(all_dir, str(q["qid"])).read_text())["subspan_em"]
                for q in queries], dtype=int))
        differences = np.stack(paired)
        rng = np.random.default_rng(20261103)
        bootstrap = np.stack([
            differences[j][rng.integers(0, 100, (20000, 100))].mean(axis=1)
            for j in range(3)], axis=1).mean(axis=1) * 100
        out["pced_vs_all_ranked_macro3"] = {
            "delta_pp": float(differences.mean() * 100),
            "paired_bootstrap_ci95_pp": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        }
    paper.atomic_json(a.output_root / "summary.json", out)
    print(json.dumps({"output": str(a.output_root / "summary.json"),
                      "macro3_subspan_em": macro}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("preflight", "evaluate", "summary"), required=True)
    p.add_argument("--dataset", choices=(*medical.DATASETS, "all"), required=True)
    p.add_argument("--arms", default="pced,all-ranked,no-rag,single,all")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--reference-root", type=Path,
                   default=ROOT / "results/medical_pced_seeded100_v1")
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "results/medical_pced_qwen35_9b_seeded100_v1")
    p.add_argument("--min-free-gib", type=float, default=40.0)
    a = p.parse_args()
    a.arms = tuple(part.strip() for part in a.arms.split(","))
    if not a.arms or len(set(a.arms)) != len(a.arms) or any(arm not in ARMS for arm in a.arms):
        p.error("--arms must be unique names from pced,all-ranked,no-rag,single,all")
    for name in ("model", "reference_root", "output_root"):
        setattr(a, name, getattr(a, name).resolve())
    if a.output_root == a.reference_root:
        p.error("Qwen output root must not overwrite the Llama experiment")
    if a.phase == "summary":
        if a.dataset != "all":
            p.error("--phase summary requires --dataset all")
        summarize_all(a)
    elif a.dataset == "all":
        p.error("--dataset all is only for the summary phase")
    elif a.phase == "preflight":
        queries, _, _, _ = preflight(a, a.dataset, write=False)
        print(f"[preflight OK] {a.dataset}: {len(queries)} fixed questions, "
              f"arms={','.join(a.arms)}; model={a.model}; no GPU allocated")
    else:
        evaluate(a, a.dataset)


if __name__ == "__main__":
    main()

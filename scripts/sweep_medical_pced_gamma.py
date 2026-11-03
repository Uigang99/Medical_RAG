#!/usr/bin/env python3
"""Decode the existing medical PCED-100 cohort with alternative gamma values.

Retrieval, reranking, expert priors, prompts, beta, and scoring remain frozen.
This is a sensitivity analysis on the previously evaluated 100 questions per
dataset, NOT an unbiased estimate of a gamma selected on those same questions.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_loft_pced_paper as loft
from scripts import run_medical_pced_100 as medical
from medrag.training.graph_input_progress import Progress

REFERENCE_GAMMA = Decimal("2.5")
DEFAULT_GAMMAS = "1,2.5,4"
VERSION = "medical_pced_gamma_sensitivity_v1"


def parse_gammas(raw: str) -> tuple[Decimal, ...]:
    try:
        values = tuple(Decimal(part.strip()) for part in raw.split(","))
    except InvalidOperation as exc:
        raise ValueError("--gammas must be comma-separated nonnegative numbers") from exc
    if not values or any(not value.is_finite() or value < 0 for value in values):
        raise ValueError("--gammas must contain finite, nonnegative numbers")
    if len(set(values)) != len(values):
        raise ValueError("--gammas contains duplicate values")
    return values


def gamma_key(value: Decimal) -> str:
    return "gamma_" + format(value.normalize(), "f").replace(".", "p")


def prediction_path(root: Path, dataset: str, gamma: Decimal, qid: str) -> Path:
    if gamma == REFERENCE_GAMMA:
        folder = root / dataset / "predictions"
    else:
        raise ValueError("Reference-path helper accepts only gamma=2.5")
    return loft.result_file(folder, qid)


def digest_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.name).encode())
        digest.update(loft.sha(path).encode())
    return digest.hexdigest()


def source_contract(dataset: str, reference_root: Path, baseline_root: Path):
    import subprocess
    source = reference_root / dataset
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare/evaluate medical PCED-100 first: {manifest_path}")
    reference = json.loads(manifest_path.read_text())
    if reference.get("version") != medical.VERSION or reference.get("score_policy") != "paper":
        raise RuntimeError(f"Unexpected source experiment: {manifest_path}")
    if reference.get("official_pced_commit") != loft.OFFICIAL_SHA:
        raise RuntimeError(f"Official decoder contract changed: {manifest_path}")
    official_path = ROOT / "cache/pced_official"
    actual_commit = subprocess.check_output(
        ["git", "-C", str(official_path), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != loft.OFFICIAL_SHA:
        raise RuntimeError(f"Official decoder checkout changed: {actual_commit}")
    for name in ("config.json", "tokenizer.json"):
        path = Path(reference["llama"]) / name
        if not path.is_file():
            raise FileNotFoundError(path)
    selector = argparse.Namespace(dataset=dataset, fetch_k=reference["fetch_k"],
                                  sample_seed=reference["sample_seed"])
    _, _, corpus, queries = medical.selected_inputs(selector)
    qids = [str(row["qid"]) for row in queries]
    if qids != reference["sample_qids"] or len(qids) != 100:
        raise RuntimeError(f"Source cohort changed: {dataset}")
    source_paths = [manifest_path]
    for qid in qids:
        ranking_path = loft.result_file(source / "rankings", qid)
        score_path = loft.result_file(source / "expert_scores", qid)
        pced_path = prediction_path(reference_root, dataset, REFERENCE_GAMMA, qid)
        ranked_path = loft.result_file(baseline_root / dataset / "all-ranked" / "predictions", qid)
        for path in (ranking_path, score_path, pced_path, ranked_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        ranking = json.loads(ranking_path.read_text())
        scores = json.loads(score_path.read_text())
        pced = json.loads(pced_path.read_text())
        ranked = json.loads(ranked_path.read_text())
        pids = [item["pid"] for item in ranking["top90"]]
        if (len(pids) != 90 or scores["pids"] != pids or pced["top90_pids"] != pids
                or ranked["document_pids"] != pids or pced["qid"] != qid
                or ranked["qid"] != qid):
            raise RuntimeError(f"Source evidence/prediction mismatch: {dataset}/{qid}")
        source_paths.extend((ranking_path, score_path, pced_path, ranked_path))
    return corpus, queries, reference, digest_files(source_paths)


def output_contract(a, dataset: str, reference: dict, source_digest: str,
                    queries: list[dict]) -> dict:
    import subprocess
    return {
        "version": VERSION, "purpose": "exploratory gamma sensitivity, not final test selection",
        "dataset": dataset, "n": len(queries), "qids": [str(q["qid"]) for q in queries],
        "gammas": [str(value) for value in a.gammas],
        "source_manifest_sha256": loft.sha(a.reference_root / dataset / "manifest.json"),
        "source_cache_and_baseline_sha256": source_digest,
        "reference_gamma": "2.5", "reference_predictions_reused": True,
        "llama": reference["llama"], "score_policy": reference["score_policy"],
        "top_k": 90, "max_new_tokens": 32, "beta": "dynamic JSD, first token",
        "normalize_retrieval_prior": False, "prompt": "unchanged LOFT PCED",
        "sample_seed": reference["sample_seed"], "official_pced_commit": loft.OFFICIAL_SHA,
        "official_decoder_sha256": loft.sha(reference_path() / "src/methods/pced.py"),
        "script_sha256": loft.sha(Path(__file__)),
        "code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": "GPU 1", "attention_impl": "sdpa", "decoding": "greedy",
        "output_root": str(a.output_root),
    }


def reference_path():
    return ROOT / "cache/pced_official"


def ensure_manifest(path: Path, expected: dict, *, write: bool):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(f"Incompatible existing gamma sweep; use a fresh output root: {path}")
    elif write:
        loft.atomic_json(path, expected)


def gamma_output(a, dataset: str, gamma: Decimal, qid: str) -> Path:
    return loft.result_file(a.output_root / dataset / gamma_key(gamma) / "predictions", qid)


def preflight(a, dataset: str, *, write: bool):
    corpus, queries, reference, source_digest = source_contract(
        dataset, a.reference_root, a.baseline_root)
    expected = output_contract(a, dataset, reference, source_digest, queries)
    manifest = a.output_root / dataset / "manifest.json"
    ensure_manifest(manifest, expected, write=write)
    for gamma in a.gammas:
        if gamma == REFERENCE_GAMMA:
            continue
        for query in queries:
            path = gamma_output(a, dataset, gamma, str(query["qid"]))
            if path.is_file():
                result = json.loads(path.read_text())
                if result.get("qid") != str(query["qid"]) or result.get("gamma") != str(gamma):
                    raise RuntimeError(f"Incompatible prediction shard: {path}")
    return corpus, queries, reference, manifest


def summarize_dataset(a, dataset: str, queries: list[dict]):
    import numpy as np
    baseline_dir = a.baseline_root / dataset / "all-ranked" / "predictions"
    baseline = np.asarray([
        json.loads(loft.result_file(baseline_dir, str(q["qid"])).read_text())["subspan_em"]
        for q in queries], dtype=int)
    reference = np.asarray([
        json.loads(prediction_path(a.reference_root, dataset, REFERENCE_GAMMA,
                                   str(q["qid"])).read_text())["subspan_em"]
        for q in queries], dtype=int)
    summary = {"dataset": dataset, "n": len(queries), "metric": "Subspan EM",
               "selection_warning": "same 100 questions were previously evaluated; exploratory only",
               "all_ranked_correct": int(baseline.sum()),
               "reference_gamma_correct": int(reference.sum()), "gamma": {}}
    for gamma in a.gammas:
        rows = []
        for q in queries:
            qid = str(q["qid"])
            path = (prediction_path(a.reference_root, dataset, gamma, qid)
                    if gamma == REFERENCE_GAMMA else gamma_output(a, dataset, gamma, qid))
            row = json.loads(path.read_text())
            if row.get("qid") != qid or row.get("subspan_em") not in (0, 1):
                raise RuntimeError(f"Invalid gamma prediction: {path}")
            rows.append(row)
        scores = np.asarray([row["subspan_em"] for row in rows], dtype=int)
        diff = scores - baseline
        change = scores - reference
        summary["gamma"][str(gamma)] = {
            "correct": int(scores.sum()), "accuracy": float(scores.mean()),
            "delta_vs_all_ranked_pp": float(diff.mean() * 100),
            "all_wrong_pced_correct": int((diff == 1).sum()),
            "all_correct_pced_wrong": int((diff == -1).sum()),
            "delta_vs_gamma_2p5_pp": float(change.mean() * 100),
            "gamma_2p5_wrong_new_correct": int((change == 1).sum()),
            "gamma_2p5_correct_new_wrong": int((change == -1).sum()),
            "reference_reused": gamma == REFERENCE_GAMMA,
        }
    loft.atomic_json(a.output_root / dataset / "summary.json", summary)
    return summary


def evaluate(a, dataset: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    progress = Progress(["source/contract", "load Llama", f"decode {dataset}", "summary"])
    qid, current_gamma = None, None
    try:
        progress.stage(0, 1)
        corpus, queries, reference, manifest = preflight(a, dataset, write=True)
        by_pid = {str(doc["pid"]): doc for doc in corpus}
        work = [(q, [gamma for gamma in a.gammas
                      if gamma != REFERENCE_GAMMA and not gamma_output(
                          a, dataset, gamma, str(q["qid"])).is_file()]) for q in queries]
        total = len(queries) * sum(g != REFERENCE_GAMMA for g in a.gammas)
        remaining = sum(len(pending) for _, pending in work)
        progress.complete(f"cohort=100, pending={remaining}/{total}; {manifest}")
        progress.log(f"[plan] {dataset}: frozen Top90+prior; gamma={','.join(map(str,a.gammas))}; "
                     f"reuse gamma2.5; decode pending={remaining}; one Llama load, one prefill/question")
        progress.stage(1, 1)
        if remaining:
            loft.gpu_gate(a.min_free_gib)
            torch.backends.cuda.enable_cudnn_sdp(False)
            sys.path.insert(0, str(reference_path()))
            from src.methods.pced import prefill, decode_from_cache
            tokenizer = AutoTokenizer.from_pretrained(reference["llama"], local_files_only=True,
                                                        padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                reference["llama"], local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
            random.seed(42)
            torch.manual_seed(42)
        progress.complete("frozen Llama-3.1-8B-Instruct; no model load when fully cached")
        progress.stage(2, max(1, total), initial=total-remaining, unit="qγ")
        parity_checked = False
        for q, pending in work:
            if not pending:
                continue
            qid = str(q["qid"])
            source = a.reference_root / dataset
            ranking = json.loads(loft.result_file(source / "rankings", qid).read_text())
            scores = json.loads(loft.result_file(source / "expert_scores", qid).read_text())
            selected = ranking["top90"]
            texts = [loft.format_doc(by_pid[item["pid"]]) for item in selected]
            prompts, question = loft.prompt_inputs(tokenizer, q["query_text"], texts)
            tokenized_prompt = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                padding=True, truncation=True,
                max_length=model.config.max_position_embeddings-32).to(model.device)
            tokenized_question = tokenizer([question] * len(prompts), return_tensors="pt",
                                           add_special_tokens=False).to(model.device)
            retrieval_scores = loft.prior_tensor(scores, "paper", model.device, model.dtype)
            with torch.inference_mode():
                base_cache = prefill(model, tokenized_prompt)
            if not parity_checked:
                parity_cache = copy.deepcopy(base_cache)
                with torch.inference_mode():
                    parity_ids, _ = decode_from_cache(
                        model, tokenizer, parity_cache, tokenized_prompt, tokenized_question,
                        max_new_tokens=32, beta=None, retrieval_scores=retrieval_scores,
                        gamma=2.5, normalize=False, return_winner_trace=True)
                original = json.loads(prediction_path(
                    a.reference_root, dataset, REFERENCE_GAMMA, qid).read_text())
                if parity_ids.tolist() != original["generated_token_ids"]:
                    raise RuntimeError(f"Gamma 2.5 parity failed; refusing unmatched sweep: {qid}")
                del parity_cache, parity_ids
                parity_checked = True
                progress.log(f"[safety] gamma2.5 exact-token parity passed: {qid}")
            for i, gamma in enumerate(pending):
                current_gamma = gamma
                started = time.monotonic()
                # Decoder mutates its KV cache. Copy the shared prefill except
                # for the final gamma, which may consume the original directly.
                cache = base_cache if i == len(pending)-1 else copy.deepcopy(base_cache)
                with torch.inference_mode():
                    ids, winners = decode_from_cache(
                        model, tokenizer, cache, tokenized_prompt, tokenized_question,
                        max_new_tokens=32, beta=None, retrieval_scores=retrieval_scores,
                        gamma=float(gamma), normalize=False, return_winner_trace=True)
                prediction = tokenizer.decode(ids, skip_special_tokens=True)
                loft.atomic_json(gamma_output(a, dataset, gamma, qid), {
                    "qid": qid, "dataset": dataset, "gamma": str(gamma),
                    "prediction": prediction, "subspan_em": loft.subspan_em(
                        prediction, q["answers"]), "answers": q["answers"],
                    "top90_pids": scores["pids"], "expert_winners": winners,
                    "generated_token_ids": ids.tolist(),
                    "decode_seconds": time.monotonic()-started,
                    "score_policy": "paper", "backend": "sdpa",
                })
                del cache, ids
                progress.update()
            del base_cache, tokenized_prompt, tokenized_question, retrieval_scores
        progress.complete(f"decoded {total} qγ shards ({total-remaining} resumed); {a.output_root/dataset}")
        progress.stage(3, 1)
        summary = summarize_dataset(a, dataset, queries)
        progress.complete(f"{a.output_root/dataset/'summary.json'}")
        progress.log(f"[result] {dataset} all-ranked={summary['all_ranked_correct']}/100; "
                     + ", ".join(f"γ={g}:{v['correct']}/100" for g,v in summary["gamma"].items()))
    except Exception as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/4 qid={qid} gamma={current_gamma} "
                     f"durable={a.output_root/dataset}; rerun same command to resume; {exc}")
        raise
    finally:
        progress.close()


def summarize_all(a):
    reports = {}
    for dataset in medical.DATASETS:
        _, queries, _, _ = preflight(a, dataset, write=False)
        reports[dataset] = summarize_dataset(a, dataset, queries)
    macro = {str(gamma): sum(reports[name]["gamma"][str(gamma)]["accuracy"]
                             for name in medical.DATASETS) / len(medical.DATASETS)
             for gamma in a.gammas}
    report = {"version": VERSION, "n": 300, "datasets": reports,
              "macro3_subspan_em": macro,
              "interpretation": "Exploratory sensitivity on previously evaluated cohort; do not select and report on same questions"}
    loft.atomic_json(a.output_root / "summary.json", report)
    print(json.dumps({"summary": str(a.output_root / "summary.json"),
                      "macro3_subspan_em": macro}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "evaluate", "summary"), required=True)
    parser.add_argument("--dataset", choices=(*medical.DATASETS, "all"), required=True)
    parser.add_argument("--gammas", default=DEFAULT_GAMMAS)
    parser.add_argument("--reference-root", type=Path,
                        default=ROOT / "results/medical_pced_seeded100_v1")
    parser.add_argument("--baseline-root", type=Path,
                        default=ROOT / "results/medical_pced_seeded100_v1_baselines")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/medical_pced_gamma_sensitivity_v1")
    parser.add_argument("--min-free-gib", type=float, default=40)
    a = parser.parse_args()
    try:
        a.gammas = parse_gammas(a.gammas)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("reference_root", "baseline_root", "output_root"):
        setattr(a, name, getattr(a, name).resolve())
    if a.output_root in (a.reference_root, a.baseline_root):
        parser.error("Gamma output root must not overwrite the original results")
    if a.phase == "summary":
        if a.dataset != "all":
            parser.error("--phase summary requires --dataset all")
        summarize_all(a)
    elif a.dataset == "all":
        parser.error("--dataset all is only for the summary phase")
    elif a.phase == "preflight":
        _, queries, _, manifest = preflight(a, a.dataset, write=False)
        print(f"[preflight OK] {a.dataset}: {len(queries)} matched questions; "
              f"gamma={','.join(map(str,a.gammas))}; {manifest}; no GPU allocated")
    else:
        evaluate(a, a.dataset)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce LOFT Corpus-in-Ctx All with the official vanilla raw-CE ranking.

This run is separate from the PCED-cache All baseline. The official vanilla
script scores every corpus passage with the raw reranker, selects the top 90
with NumPy argsort, then applies its seeded permutation. All predictions are
saved under a new root; the completed PCED and previous All runs are read-only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medrag.training.graph_input_progress import Progress
from scripts import evaluate_loft_corpus_in_context as cic
from scripts import run_loft_pced_paper as paper


VERSION = "loft_cic_all_official_raw_v2"
DATASETS = paper.DATASETS


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path,
                        default=ROOT / "results/loft_pced_paper_v1_full500_parallel2")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/loft_cic_all_official_raw_v2_500")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Keep the prior All generation batch size for an isolated ranking comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check CPU-only input/model contracts without writing or loading GPU models")
    args = parser.parse_args(argv)
    args.reference_root = args.reference_root.resolve()
    args.output_root = args.output_root.resolve()
    args.arm = "all"
    if args.batch_size < 1 or args.output_root == args.reference_root:
        parser.error("Batch size must be positive and output root must differ from the reference")
    return args


def preflight(args):
    official = ROOT / "cache/pced_official"
    revision = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError(f"Official PCED code revision changed: {revision}")
    llama = ROOT.parent / "models/Llama-3.1-8B-Instruct"
    reranker = ROOT.parent / "models/bge-reranker-v2-m3-pced"
    if not (llama / "config.json").is_file() or not (reranker / "config.json").is_file():
        raise FileNotFoundError("Local Llama or BGE reranker model is missing")
    dataset_data = {}
    source_hashes = {}
    for dataset in DATASETS:
        source = paper.arguments(["--phase", "preflight", "--dataset", dataset,
                                  "--output-root", str(args.reference_root)])
        corpus_path, query_path, corpus, queries = paper.load_inputs(source)
        old = json.loads((args.reference_root / dataset / "manifest.json").read_text())
        old_summary = json.loads((args.reference_root / dataset / "summary.json").read_text())
        corpus_sha, queries_sha = paper.sha(corpus_path), paper.sha(query_path)
        if (old.get("corpus_sha256") != corpus_sha or
                old.get("queries_sha256") != queries_sha or
                old.get("top_k") != 90 or old.get("candidate_mode") != "all" or
                old_summary.get("n") != 100 or old_summary.get("missing") != 0):
            raise RuntimeError(f"Reference cohort/corpus contract mismatch: {dataset}")
        dataset_data[dataset] = (queries, {str(row["pid"]): row for row in corpus},
                                 corpus, corpus_sha)
        source_hashes[dataset] = {"corpus_sha256": corpus_sha,
                                  "queries_sha256": queries_sha,
                                  "documents": len(corpus), "questions": len(queries)}
    manifest = {
        "version": VERSION, "arm": "all", "n": 500, "datasets": list(DATASETS),
        "reference_root": str(args.reference_root), "output_root": str(args.output_root),
        "sources": source_hashes, "model": str(llama), "reranker": str(reranker),
        "reranker_revision": paper.MODEL_RERANK,
        "ranking": "official run_vanilla.py: raw FlagReranker.compute_score on full passages; "
                   "numpy argsort(scores)[-90:][::-1]; numpy default_rng(125) permutation",
        "prompt": "official LOFT system/context/question/Answer: template",
        "generation": {"backend": "transformers-sdpa", "greedy": True,
                       "max_new_tokens": cic.MAX_NEW_TOKENS, "seed": 42,
                       "batch_size": args.batch_size},
        "metric": "paper Subspan EM", "gpu": 1,
        "official_pced_commit": revision,
        "official_vanilla_sha256": paper.sha(official / "src/run_vanilla.py"),
        "script_sha256": paper.sha(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT),
                                                "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": sys.version.split()[0],
                        "torch": importlib.metadata.version("torch"),
                        "transformers": importlib.metadata.version("transformers")},
    }
    existing = args.output_root / "manifest.json"
    if existing.is_file() and json.loads(existing.read_text()) != manifest:
        raise RuntimeError(f"Incompatible existing output manifest: {existing}")
    return dataset_data, llama, reranker, manifest


def rank_every_passage(args, dataset_data, reranker_path, progress):
    from FlagEmbedding import FlagReranker
    total = sum(len(item[0]) for item in dataset_data.values())
    cached = sum(paper.result_file(args.output_root / dataset / "selection", str(q["qid"])).is_file()
                 for dataset, (queries, _, _, _) in dataset_data.items() for q in queries)
    progress.stage(1, total, initial=cached, unit="q")
    if cached == total:
        progress.complete(f"raw-CE selections cached={cached}/{total}")
        return
    paper.gpu_gate(40)
    reranker = FlagReranker(str(reranker_path), use_fp16=True, devices=["cuda:0"])
    for dataset in DATASETS:
        queries, _, corpus, corpus_sha = dataset_data[dataset]
        texts = [paper.format_doc(row) for row in corpus]
        for q in queries:
            qid = str(q["qid"])
            path = paper.result_file(args.output_root / dataset / "selection", qid)
            if path.is_file():
                continue
            args.current_qid = f"{dataset}/{qid}"
            pairs = [[q["query_text"], text] for text in texts]
            raw = paper.math_sdpa_call(reranker.compute_score, pairs, normalize=False)
            scores = np.asarray(raw, dtype=np.float64).reshape(-1)
            if len(scores) != len(corpus) or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid raw CE scores: {dataset}/{qid}")
            top_indices = np.argsort(scores)[-90:][::-1].tolist()
            paper.atomic_json(path, {"dataset": dataset, "qid": qid,
                                     "corpus_sha256": corpus_sha,
                                     "raw_ce_scores_corpus_order": scores.tolist(),
                                     "top90_indices": top_indices,
                                     "top90_pids": [str(corpus[i]["pid"]) for i in top_indices]})
            progress.update()
    del reranker
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    progress.complete(f"raw-CE selections={total}/{total}; {args.output_root}")


def make_layouts(args, dataset_data, llama_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(llama_path), local_files_only=True)
    model_context = int(json.loads((llama_path / "config.json").read_text())["max_position_embeddings"])
    layouts = []
    for dataset in DATASETS:
        queries, by_pid, corpus, corpus_sha = dataset_data[dataset]
        rng = np.random.default_rng(125)  # Official vanilla starts a fresh process for each dataset.
        for q in queries:
            qid = str(q["qid"])
            selection = json.loads(paper.result_file(args.output_root / dataset / "selection", qid).read_text())
            ranked = selection["top90_pids"]
            if (selection.get("corpus_sha256") != corpus_sha or
                    selection.get("qid") != qid or len(ranked) != 90 or
                    ranked != [str(corpus[i]["pid"]) for i in selection["top90_indices"]]):
                raise RuntimeError(f"Incompatible raw-CE selection: {dataset}/{qid}")
            perm = rng.permutation(90)
            prompt_pids = [ranked[int(i)] for i in perm]
            prompt = cic.vanilla_prompt(tokenizer, q["query_text"],
                                        [paper.format_doc(by_pid[pid]) for pid in prompt_pids])
            length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if length + cic.MAX_NEW_TOKENS > model_context:
                raise RuntimeError(f"All context exceeds Llama limit: {dataset}/{qid}, "
                                   f"{length}+{cic.MAX_NEW_TOKENS}>{model_context}")
            layouts.append({"dataset": dataset, "qid": qid, "ranked_pids": ranked,
                            "prompt_pids": prompt_pids, "all_prompt_tokens": length})
    path = args.output_root / "layouts.json"
    if path.is_file() and json.loads(path.read_text()) != layouts:
        raise RuntimeError(f"Incompatible saved layout: {path}")
    if not path.is_file():
        paper.atomic_json(path, layouts)
    return tokenizer, layouts


def main(argv=None):
    args = arguments(argv)
    progress = Progress(["CPU contract", "raw-CE rerank and exact layouts", "load Llama",
                         "All 500q", "Subspan EM summary"])
    progress.weights[:] = [1., None, 3., None, 1.]
    try:
        progress.stage(0, 1)
        data, llama, reranker, manifest = preflight(args)
        progress.complete(f"500 questions; {sum(len(x[2]) * len(x[0]) for x in data.values())} "
                          "raw-CE pairs; model/corpus contracts checked")
        cached = sum(paper.result_file(args.output_root / dataset / "selection", str(q["qid"])).is_file()
                     for dataset, (queries, _, _, _) in data.items() for q in queries)
        progress.log(f"[plan] official raw-CE all-corpus ranking -> seeded Top-90 order -> "
                     f"same SDPA Llama and batch={args.batch_size} as prior All -> Subspan EM; "
                     f"GPU1 only; cached rankings={cached}/500; {args.output_root}")
        if args.dry_run:
            progress.log("[dry-run OK] CPU contracts only; no files written or GPU models loaded")
            return
        path = args.output_root / "manifest.json"
        if not path.is_file():
            paper.atomic_json(path, manifest)
        rank_every_passage(args, data, reranker, progress)
        tokenizer, layouts = make_layouts(args, data, llama)
        cic.generate(args, data, layouts, tokenizer, llama, progress)
        progress.stage(4, 1)
        summary = cic.summarize(args, layouts, path)
        progress.complete(f"{summary['correct']}/500; {args.output_root/'summary.json'}")
        progress.log(f"[result] official-ranking All {summary['correct']}/500")
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/5, item={getattr(args, 'current_qid', 'setup')}, "
                     f"completed={progress.done}/{progress.total}; {exc}; "
                     f"durable={args.output_root}; identical command resumes compatible shards")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

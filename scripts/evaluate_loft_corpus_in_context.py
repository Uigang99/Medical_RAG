#!/usr/bin/env python3
"""LOFT Corpus-in-Context Single/All with the fixed PCED Top-90 candidates.

The official vanilla baseline selects Single by BGE-M3 sparse score within
the reranked Top-90; All concatenates that same Top-90 after the official
seed-125 shuffle. Neither arm uses gold passages to select its input.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.training.graph_input_progress import Progress
from scripts import run_loft_pced_paper as paper

VERSION = "loft_cic_v1"
MAX_NEW_TOKENS = 32  # Match the completed PCED comparison, not run_vanilla.py's CLI default of 100.
SPARSE_MAX_LENGTH = 512  # Official BGEM3FlagModel.compute_score defaults.


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("single", "all"))
    p.add_argument("--reference-root", type=Path,
                   default=ROOT / "results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--batch-size", type=int, help="Default: 16 for Single, 2 for All")
    p.add_argument("--sparse-batch-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true", help="CPU-only contract and exact token budget")
    a = p.parse_args(argv)
    a.reference_root = a.reference_root.resolve()
    a.output_root = (a.output_root or ROOT / "results" / f"loft_cic_{a.arm}_v1_500").resolve()
    a.batch_size = (16 if a.arm == "single" else 2) if a.batch_size is None else a.batch_size
    if a.batch_size < 1 or a.sparse_batch_size < 1:
        p.error("Batch sizes must be positive")
    if a.output_root == a.reference_root:
        p.error("CiC results cannot overwrite the completed PCED reference")
    return a


def vanilla_prompt(tokenizer, question: str, documents: list[str]) -> str:
    """Official run_vanilla.py user-message layout, without its silent truncation."""
    context = "\n\n".join(documents)
    user = context + paper.QUESTION.format(question=question)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": paper.SYSTEM}, {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    ) + paper.ANSWER_PREFIX


def shuffled_top90(ranking: dict, rng) -> list[str]:
    import numpy as np
    selected = ranking["top90"]
    if len(selected) != 90 or len({str(x["pid"]) for x in selected}) != 90:
        raise RuntimeError(f"Invalid fixed Top-90 for qid={ranking.get('qid')}")
    # Same order operation as official run_vanilla.py after reranking.
    permutation = rng.permutation(90)
    return [str(selected[int(i)]["pid"]) for i in permutation]


def preflight(a, progress):
    import numpy as np
    from transformers import AutoTokenizer

    official = ROOT / "cache/pced_official"
    revision = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError(f"Official PCED code revision changed: {revision}")
    model_path = Path(json.loads((a.reference_root / "hotpotqa/manifest.json").read_text())["llama"])
    config = json.loads((model_path / "config.json").read_text())
    max_positions = int(config["max_position_embeddings"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    data, layouts, tokens, ranking_hashes = {}, [], [], {}
    for dataset in paper.DATASETS:
        source = paper.arguments(["--phase", "preflight", "--dataset", dataset,
                                  "--output-root", str(a.reference_root)])
        corpus_path, query_path, corpus, queries = paper.load_inputs(source)
        manifest_path = a.reference_root / dataset / "manifest.json"
        summary_path = a.reference_root / dataset / "summary.json"
        reference = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        if (reference.get("queries_sha256") != paper.sha(query_path) or
                reference.get("corpus_sha256") != paper.sha(corpus_path) or
                reference.get("top_k") != 90 or reference.get("candidate_mode") != "all" or
                reference.get("llama") != str(model_path) or summary.get("n") != 100 or
                summary.get("missing") != 0):
            raise RuntimeError(f"Completed PCED reference changed: {dataset}")
        by_pid = {str(doc["pid"]): doc for doc in corpus}
        data[dataset] = (queries, by_pid, paper.sha(query_path), paper.sha(manifest_path))
        rng = np.random.default_rng(125)  # Official vanilla script restarts per dataset.
        ranking_hash = hashlib.sha256()
        for q in queries:
            qid = str(q["qid"])
            ranking_path = paper.result_file(a.reference_root / dataset / "rankings", qid)
            ranking = json.loads(ranking_path.read_text())
            ranking_hash.update(bytes.fromhex(paper.sha(ranking_path)))
            if str(ranking.get("qid")) != qid:
                raise RuntimeError(f"Ranking qid mismatch: {dataset}/{qid}")
            ordered_pids = shuffled_top90(ranking, rng)
            if any(pid not in by_pid for pid in ordered_pids):
                raise RuntimeError(f"Ranking references missing corpus pid: {dataset}/{qid}")
            prompt = vanilla_prompt(tokenizer, q["query_text"],
                                    [paper.format_doc(by_pid[pid]) for pid in ordered_pids])
            length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if length + MAX_NEW_TOKENS > max_positions:
                raise RuntimeError(f"All exceeds Llama context: {dataset}/{qid}, "
                                   f"{length}+{MAX_NEW_TOKENS}>{max_positions}; no truncation allowed")
            tokens.append(length)
            layouts.append({"dataset": dataset, "qid": qid,
                            "ranked_pids": [str(item["pid"]) for item in ranking["top90"]],
                            "prompt_pids": ordered_pids, "all_prompt_tokens": length})
        ranking_hashes[dataset] = ranking_hash.hexdigest()
    if len(layouts) != 500:
        raise RuntimeError(f"Expected 500 LOFT questions; found {len(layouts)}")
    token_stats = {"min": min(tokens), "mean": round(sum(tokens) / len(tokens), 1),
                   "total": sum(tokens), "median": int(np.median(tokens)),
                   "p95": int(np.percentile(tokens, 95)), "max": max(tokens),
                   "model_context": max_positions, "reserved_generation": MAX_NEW_TOKENS}
    expected = {"version": VERSION, "arm": a.arm, "n": 500,
                "datasets": list(paper.DATASETS), "reference_root": str(a.reference_root),
                "output_root": str(a.output_root), "references": {
                    dataset: {"queries_sha256": values[2], "manifest_sha256": values[3],
                              "rankings_sha256": ranking_hashes[dataset]}
                    for dataset, values in data.items()},
                "model": str(model_path), "gpu": 1, "backend": "transformers-sdpa",
                "generation": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS,
                               "seed": 42, "batch_size": a.batch_size},
                "selection": ("BGE-M3 sparse top-1 within fixed CE Top-90, max_query_length=512, "
                              "max_passage_length=512" if a.arm == "single" else "all fixed CE Top-90"),
                "bge_model": str(ROOT.parent / "models/bge-m3-pced") if a.arm == "single" else None,
                "bge_revision": paper.MODEL_BGE if a.arm == "single" else None,
                "document_order": "numpy default_rng(125), independently reset for each dataset",
                "prompt_layout": "official run_vanilla.py: system + one user message of "
                                 "joined context immediately followed by question_template; Answer: prefix",
                "system_prompt": paper.SYSTEM, "question_template": paper.QUESTION,
                "answer_prefix": paper.ANSWER_PREFIX, "metric": "paper Subspan EM",
                "token_budget": token_stats, "sparse_batch_size": a.sparse_batch_size,
                "official_pced_commit": revision,
                "official_vanilla_sha256": paper.sha(official / "src/run_vanilla.py"),
                "script_sha256": paper.sha(Path(__file__)),
                "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                      text=True).strip(),
                "environment": {"python": sys.version.split()[0],
                                "torch": importlib.metadata.version("torch"),
                                "transformers": importlib.metadata.version("transformers")}}
    existing = a.output_root / "manifest.json"
    if existing.is_file() and json.loads(existing.read_text()) != expected:
        raise RuntimeError(f"Incompatible CiC resume contract: {existing}")
    progress.complete(f"same 5x100 questions / fixed Top-90; All prompt tokens "
                      f"min/median/p95/max={tokens_stats_text(token_stats)}; "
                      f"Llama limit={max_positions}; no truncation")
    return data, layouts, model_path, expected, tokenizer


def tokens_stats_text(stats):
    return "/".join(str(stats[key]) for key in ("min", "median", "p95", "max"))


def encode_sparse_checked(embedder, texts, batch_size):
    """Same BGE-M3 token weights as compute_score, encoded once per unique text.

    Avoid FlagEmbedding's empty-batch OOM probe that failed in the earlier LOFT
    run. The official compute_score's dense/ColBERT branches do not affect its
    returned sparse score or the selected top-1 passage.
    """
    import torch
    special = set()
    for key in ("cls_token", "eos_token", "pad_token", "unk_token"):
        if key in embedder.tokenizer.special_tokens_map:
            special.add(embedder.tokenizer.convert_tokens_to_ids(
                embedder.tokenizer.special_tokens_map[key]))
    device = next(embedder.model.parameters()).device
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = embedder.tokenizer(texts[start:start + batch_size], padding=True,
                                   truncation=True, max_length=SPARSE_MAX_LENGTH,
                                   return_token_type_ids=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = paper.math_sdpa_call(embedder.model, batch, return_dense=False,
                                          return_sparse=True, return_colbert_vecs=False)
        weights = output["sparse_vecs"].squeeze(-1).float().cpu().tolist()
        ids = batch["input_ids"].cpu().tolist()
        for row_weights, row_ids in zip(weights, ids):
            values = {}
            for weight, token_id in zip(row_weights, row_ids):
                if token_id not in special and weight > 0:
                    key = str(token_id)
                    values[key] = max(values.get(key, 0.0), weight)
            vectors.append(values)
        del batch, output
    return vectors


def sparse_match(query: dict, document: dict) -> float:
    return sum(weight * document.get(token, 0.0) for token, weight in query.items())


def select_single(a, data, layouts, progress):
    import numpy as np
    import torch
    from FlagEmbedding import BGEM3FlagModel

    cached = sum(paper.result_file(a.output_root / x["dataset"] / "selection", x["qid"]).is_file()
                 for x in layouts)
    progress.stage(1, 500, initial=cached, unit="q")
    if cached == 500:
        progress.complete(f"cached sparse selections=500/500; {a.output_root}")
        return
    paper.gpu_gate(40)
    model_path = ROOT.parent / "models/bge-m3-pced"
    embedder = BGEM3FlagModel(str(model_path), use_fp16=True, devices=["cuda:0"])
    embedder.model.half().to("cuda:0").eval()
    for dataset in paper.DATASETS:
        questions, by_pid, _, _ = data[dataset]
        subset = [x for x in layouts if x["dataset"] == dataset]
        pending = [x for x in subset if not paper.result_file(
            a.output_root / dataset / "selection", x["qid"]).is_file()]
        if not pending:
            continue
        pids = list(by_pid)
        doc_sparse = encode_sparse_checked(embedder,
            [paper.format_doc(by_pid[pid]) for pid in pids], a.sparse_batch_size)
        query_sparse = encode_sparse_checked(embedder,
            [q["query_text"] for q in questions], a.sparse_batch_size)
        by_doc = dict(zip(pids, doc_sparse))
        by_query = {str(q["qid"]): vec for q, vec in zip(questions, query_sparse)}
        for x in pending:
            a.current_qid = f"{dataset}/{x['qid']}"
            scores = [sparse_match(by_query[x["qid"]], by_doc[pid]) for pid in x["prompt_pids"]]
            if not all(np.isfinite(scores)):
                raise RuntimeError(f"Non-finite sparse scores: {dataset}/{x['qid']}")
            winner = int(np.argsort(scores)[-1])  # Official run_vanilla.py tie rule.
            paper.atomic_json(paper.result_file(a.output_root / dataset / "selection", x["qid"]), {
                "dataset": dataset, "qid": x["qid"], "prompt_pids": x["prompt_pids"],
                "sparse_scores": scores, "selected_index": winner,
                "selected_pid": x["prompt_pids"][winner],
                "bge_model": str(model_path), "bge_revision": paper.MODEL_BGE,
                "max_query_length": SPARSE_MAX_LENGTH,
                "max_passage_length": SPARSE_MAX_LENGTH})
            progress.update()
        del doc_sparse, query_sparse, by_doc, by_query
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    progress.complete(f"BGE-M3 sparse top-1 saved for all 500; {a.output_root}")


def build_prompt(a, x, data, tokenizer, max_positions):
    dataset, qid = x["dataset"], x["qid"]
    questions, by_pid, _, _ = data[dataset]
    q = next(row for row in questions if str(row["qid"]) == qid)
    if a.arm == "single":
        selection = json.loads(paper.result_file(a.output_root / dataset / "selection", qid).read_text())
        if selection["prompt_pids"] != x["prompt_pids"]:
            raise RuntimeError(f"Sparse selection order mismatch: {dataset}/{qid}")
        pids = [selection["selected_pid"]]
    else:
        pids = x["prompt_pids"]
    prompt = vanilla_prompt(tokenizer, q["query_text"],
                            [paper.format_doc(by_pid[pid]) for pid in pids])
    length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if length + MAX_NEW_TOKENS > max_positions:
        raise RuntimeError(f"Prompt exceeds context: {dataset}/{qid}, "
                           f"{length}+{MAX_NEW_TOKENS}>{max_positions}")
    return prompt, q, pids, length


def generate(a, data, layouts, tokenizer, model_path, progress):
    import torch
    from transformers import AutoModelForCausalLM

    cached = sum(paper.result_file(a.output_root / x["dataset"] / "predictions", x["qid"]).is_file()
                 for x in layouts)
    progress.stage(2, 1)
    if cached < 500:
        paper.gpu_gate(40)
        torch.backends.cuda.enable_cudnn_sdp(False)
        random.seed(42)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        model.requires_grad_(False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = None
    progress.complete(f"frozen Llama-3.1-8B; cached predictions={cached}/500")
    progress.stage(3, 500, initial=cached, unit="q")
    pending = [x for x in layouts if not paper.result_file(
        a.output_root / x["dataset"] / "predictions", x["qid"]).is_file()]
    index, current_batch = 0, a.batch_size
    max_positions = int(json.loads((model_path / "config.json").read_text())["max_position_embeddings"])
    while index < len(pending):
        chosen = pending[index:index + current_batch]
        a.current_qid = f"{chosen[0]['dataset']}/{chosen[0]['qid']}"
        records = [build_prompt(a, x, data, tokenizer, max_positions) for x in chosen]
        try:
            inputs = tokenizer([row[0] for row in records], return_tensors="pt",
                               add_special_tokens=False, padding=True).to(model.device)
            with torch.inference_mode():
                raw = model.generate(**inputs, do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id,
                    use_cache=True)
        except torch.cuda.OutOfMemoryError:
            del records
            if "inputs" in locals():
                del inputs
            torch.cuda.empty_cache()
            if current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            progress.log(f"[memory] retry same question with batch_size={current_batch}")
            continue
        for row_index, (x, (_, q, pids, prompt_tokens)) in enumerate(zip(chosen, records)):
            ids = raw[row_index, inputs["input_ids"].shape[1]:].tolist()
            prediction = tokenizer.decode(ids, skip_special_tokens=True)
            paper.atomic_json(paper.result_file(
                a.output_root / x["dataset"] / "predictions", x["qid"]), {
                "dataset": x["dataset"], "qid": x["qid"], "arm": a.arm,
                "prediction": prediction, "answers": q["answers"],
                "subspan_em": paper.subspan_em(prediction, q["answers"]),
                "generated_token_ids": ids, "prompt_tokens": prompt_tokens,
                "document_pids": pids, "effective_batch_size": len(chosen)})
            progress.update()
        index += len(chosen)
        del inputs, raw, records
    progress.complete(f"predictions=500/500; {a.output_root}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize(a, layouts, manifest):
    scores = {}
    for dataset in paper.DATASETS:
        records = [json.loads(paper.result_file(a.output_root / dataset / "predictions", x["qid"]).read_text())
                   for x in layouts if x["dataset"] == dataset]
        if len(records) != 100 or any(row.get("arm") != a.arm for row in records):
            raise RuntimeError(f"Incomplete or mixed predictions: {dataset}")
        correct = sum(row["subspan_em"] for row in records)
        scores[dataset] = {"n": 100, "correct": correct,
                           "subspan_em": correct / 100, "missing": 0}
    correct = sum(item["correct"] for item in scores.values())
    summary = {"arm": a.arm, "n": 500, "correct": correct,
               "micro_subspan_em": correct / 500,
               "macro5_subspan_em": sum(item["subspan_em"] for item in scores.values()) / 5,
               "datasets": scores, "manifest": str(manifest)}
    paper.atomic_json(a.output_root / "summary.json", summary)
    return summary


def main(argv=None):
    a = arguments(argv)
    stage_one = "BGE-M3 sparse Single" if a.arm == "single" else "fixed Top-90 All"
    progress = Progress(["CPU contract / exact context", stage_one, "load Llama",
                         f"CiC {a.arm} 500q", "Subspan EM summary"])
    progress.weights[:] = [None, None, 3., None, 1.]
    try:
        progress.stage(0, 1)
        data, layouts, model_path, expected, tokenizer = preflight(a, progress)
        selected = sum(paper.result_file(a.output_root / x["dataset"] / "selection", x["qid"]).is_file()
                       for x in layouts) if a.arm == "single" else 500
        predicted = sum(paper.result_file(a.output_root / x["dataset"] / "predictions", x["qid"]).is_file()
                        for x in layouts)
        progress.log(f"[plan] CPU contract -> {stage_one} -> frozen Llama -> CiC {a.arm} -> summary; "
                     f"selection cached={selected}/500, predictions cached={predicted}/500; "
                     f"GPU1 only; fixed Top-90; greedy 32; no gold; {a.output_root}")
        if a.dry_run:
            progress.log("[dry-run OK] CPU-only; no output written or GPU model loaded")
            return
        manifest = a.output_root / "manifest.json"
        if not manifest.exists():
            paper.atomic_json(manifest, expected)
        layout_path = a.output_root / "layouts.json"
        if layout_path.exists() and json.loads(layout_path.read_text()) != layouts:
            raise RuntimeError(f"Incompatible saved CiC layout: {layout_path}")
        if not layout_path.exists():
            paper.atomic_json(layout_path, layouts)
        if a.arm == "single":
            select_single(a, data, layouts, progress)
        else:
            progress.stage(1, 1)
            progress.complete("all 90 fixed candidates per question; no sparse re-selection")
        generate(a, data, layouts, tokenizer, model_path, progress)
        progress.stage(4, 1)
        summary = summarize(a, layouts, manifest)
        progress.complete(f"{a.output_root/'summary.json'}")
        progress.log(f"[result] CiC {a.arm} {summary['correct']}/500")
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/5, item={getattr(a, 'current_qid', 'setup')}, "
                     f"completed={progress.done}/{progress.total}; "
                     f"{exc}; durable={a.output_root}; identical command resumes compatible shards")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

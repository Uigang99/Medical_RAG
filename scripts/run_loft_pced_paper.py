#!/usr/bin/env python3
"""LOFT-RAG / PCED paper replication with durable embeddings and rankings.

The paper arm reranks EVERY passage in each LOFT corpus. Its FAISS index is
archived for later work, not used to select paper-arm candidates. The optional
``faiss`` arm is a distinct retrieval experiment, never a paper reproduction.
The generation algorithm is imported from the pinned official PCED checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import string
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medrag.training.graph_input_progress import Progress

DATASETS = ("hotpotqa", "nq", "musique", "qampari", "quest")
COUNTS = {"hotpotqa": 1222, "nq": 883, "musique": 824, "qampari": 755, "quest": 328}
OFFICIAL_SHA = "b3a00c7172b46effeec466c16a689648e290fa33"
VERSION = "loft_pced_paper_v1"
SYSTEM = ("You will be given a list of documents. You need to read carefully and "
          "understand all of them. Then you will be given a query, and your goal is "
          "to answer the query based on the documents you have read.")
QUESTION = ("Based on the documents above, can you answer the following query? "
            "Write a concise answer.\nquery: {question}")
ANSWER_PREFIX = "Answer:"
MODEL_BGE = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_RERANK = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("preflight", "prepare", "evaluate"), required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--corpus-root", type=Path, default=ROOT / "datasets/benchmark/loft/rag")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/loft_pced_paper_v1")
    p.add_argument("--official-code", type=Path, default=ROOT / "cache/pced_official")
    p.add_argument("--llama", type=Path, default=ROOT.parent / "models/Llama-3.1-8B-Instruct")
    p.add_argument("--bge", type=Path, default=ROOT.parent / "models/bge-m3-pced")
    p.add_argument("--reranker", type=Path, default=ROOT.parent / "models/bge-reranker-v2-m3-pced")
    p.add_argument("--candidate-mode", choices=("all", "faiss"), default="all")
    p.add_argument("--fetch-k", type=int, default=200, help="Only for the non-paper FAISS arm")
    p.add_argument("--top-k", type=int, default=90)
    p.add_argument("--score-policy", choices=("paper", "released-script"), default="paper")
    p.add_argument("--limit", type=int, default=0, help="0=all 100 test queries; >0=smoke only")
    p.add_argument("--embedding-batch", type=int, default=16)
    p.add_argument("--attention-impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    p.add_argument("--min-free-gib", type=float, default=40.0)
    a = p.parse_args(argv)
    if a.limit < 0 or a.limit > 100 or a.top_k != 90 or a.fetch_k < a.top_k or a.embedding_batch < 1:
        p.error("Use paper Top-90; limit=0..100, fetch-k>=90, embedding-batch>=1")
    for key in ("corpus_root", "output_root", "official_code", "llama", "bge", "reranker"):
        setattr(a, key, getattr(a, key).resolve())
    if a.limit and a.output_root == (ROOT / "results/loft_pced_paper_v1").resolve():
        a.output_root = a.output_root.with_name(f"loft_pced_paper_v1_smoke{a.limit}")
    return a


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        try:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
            name = f.name
        except BaseException:
            Path(f.name).unlink(missing_ok=True)
            raise
    os.replace(name, path)


def atomic_array(path, array):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        try:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
            name = f.name
        except BaseException:
            Path(f.name).unlink(missing_ok=True)
            raise
    os.replace(name, path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def result_file(directory, qid):
    return directory / (hashlib.sha256(qid.encode()).hexdigest() + ".json")


def format_doc(row):
    # Byte-for-byte formatting in official create_hf_dataset.py.
    return f"({row.get('title_text') or ''}) CONTENT: {row.get('passage_text') or ''}"


def load_inputs(a):
    folder = a.corpus_root / a.dataset / "128k"
    cp, qp = folder / "corpus.jsonl", folder / "test_queries.jsonl"
    if not cp.is_file() or not qp.is_file():
        raise FileNotFoundError(f"Missing LOFT 128k input: {cp} or {qp}")
    corpus = list(jsonl(cp))
    queries = list(jsonl(qp))
    pids = [str(row["pid"]) for row in corpus]
    qids = [str(row["qid"]) for row in queries]
    if len(corpus) != COUNTS[a.dataset] or len(queries) != 100:
        raise RuntimeError(f"Unexpected LOFT size: {a.dataset} docs={len(corpus)}, test={len(queries)}")
    if len(set(pids)) != len(pids) or len(set(qids)) != len(qids):
        raise RuntimeError("Duplicate LOFT pid/qid")
    if any(not row.get("query_text") or not row.get("answers") for row in queries):
        raise RuntimeError("Missing LOFT question or gold answer")
    if any(str(pid) not in set(pids) for q in queries
           for pid, rel in q.get("metadata", {}).get("qrels", []) if int(rel) > 0):
        raise RuntimeError("Positive LOFT qrel not present in corpus")
    return cp, qp, corpus, queries[:a.limit or None]


def contract(a, cp, qp, n):
    return {"version": VERSION, "dataset": a.dataset, "size": "128k", "split": "test",
            "questions": n, "corpus_sha256": sha(cp), "queries_sha256": sha(qp),
            "candidate_mode": a.candidate_mode, "fetch_k": a.fetch_k if a.candidate_mode == "faiss" else None,
            "top_k": 90, "score_policy": a.score_policy, "max_new_tokens": 32,
            "llama": str(a.llama), "bge": str(a.bge), "reranker": str(a.reranker),
            "bge_revision": MODEL_BGE, "reranker_revision": MODEL_RERANK,
            "official_pced_commit": OFFICIAL_SHA, "attention_impl": a.attention_impl,
            "seed": 42, "script_sha256": sha(Path(__file__)),
            "paper_difference": ("FAISS first-stage shortlist; not the published LOFT selection"
                                 if a.candidate_mode == "faiss" else
                                 "No first-stage shortlist; all LOFT documents reranked"),
            "backend_difference": ("SDPA instead of official FlashAttention-2"
                                   if a.attention_impl == "sdpa" else "Official FlashAttention-2")}


def preflight(a, *, write_manifest):
    cp, qp, corpus, queries = load_inputs(a)
    required = [(a.llama, "config.json"), (a.llama, "tokenizer.json"),
                (a.bge, "pytorch_model.bin"), (a.bge, "tokenizer.json"),
                (a.reranker, "model.safetensors"), (a.reranker, "tokenizer.json")]
    for directory, filename in required:
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Missing local model file: {directory / filename}")
    source = a.official_code / "src/methods/pced.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing official PCED checkout: {source}")
    actual = subprocess.check_output(["git", "-C", str(a.official_code), "rev-parse", "HEAD"], text=True).strip()
    if actual != OFFICIAL_SHA:
        raise RuntimeError(f"Official PCED commit changed: {actual} != {OFFICIAL_SHA}")
    expected = contract(a, cp, qp, len(queries))
    manifest = a.output_root / a.dataset / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise RuntimeError(f"Incompatible completed/partial experiment: {manifest}")
    elif write_manifest:
        atomic_json(manifest, expected)
    return corpus, queries, manifest


def gpu_gate(min_free_gib):
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("This experiment is GPU-1 only: set CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Set CUDA_VISIBLE_DEVICES=1 on the GPU server.")
    free, _ = torch.cuda.mem_get_info()
    if free < min_free_gib * 2**30:
        raise RuntimeError(f"GPU has only {free / 2**30:.1f} GiB free; require {min_free_gib:g} GiB before loading models")


def _windows(text):
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + 1024]) for i in range(0, len(words), 1024)]


def length_safe_scores(question, texts, score_fn):
    # Exact LOFT scoring policy in official src/run_pced.py: 1024-word
    # non-overlapping windows, batch 64, max-over-windows per document.
    pairs, owners = [], []
    for i, text in enumerate(texts):
        for window in _windows(text):
            pairs.append([question, window])
            owners.append(i)
    result = [float("-inf")] * len(texts)
    for start in range(0, len(pairs), 64):
        scores = score_fn(pairs[start:start + 64])
        if isinstance(scores, (float, int)):
            scores = [scores]
        if len(scores) != len(pairs[start:start + 64]):
            raise RuntimeError("FlagEmbedding returned wrong score count")
        for owner, score in zip(owners[start:start + 64], scores):
            result[owner] = max(result[owner], float(score))
    return result


def _vectors(a, corpus, queries, folder, progress):
    import faiss
    import numpy as np
    docs_path, qs_path, index_path = folder / "corpus_dense.npy", folder / "query_dense.npy", folder / "corpus.faiss"
    if all(x.exists() for x in (docs_path, qs_path, index_path)):
        docs, qs = np.load(docs_path), np.load(qs_path)
        if docs.shape[0] != len(corpus) or qs.shape[0] != len(queries) or docs.shape[1] != qs.shape[1]:
            raise RuntimeError("Cached embedding shape mismatch")
        return docs, qs
    gpu_gate(a.min_free_gib)
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(str(a.bge), use_fp16=True, devices=["cuda:0"])
    texts = [format_doc(doc) for doc in corpus]
    qtexts = [q["query_text"] for q in queries]
    docs = np.asarray(model.encode(texts, batch_size=a.embedding_batch, max_length=8192,
                                   return_dense=True, return_sparse=False,
                                   return_colbert_vecs=False)["dense_vecs"], dtype=np.float32)
    qs = np.asarray(model.encode(qtexts, batch_size=a.embedding_batch, max_length=8192,
                                 return_dense=True, return_sparse=False,
                                 return_colbert_vecs=False)["dense_vecs"], dtype=np.float32)
    if not np.isfinite(docs).all() or not np.isfinite(qs).all():
        raise RuntimeError("Non-finite BGE-M3 embedding")
    index = faiss.IndexFlatIP(docs.shape[1])
    index.add(docs)
    atomic_array(docs_path, docs)
    atomic_array(qs_path, qs)
    tmp = index_path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, index_path)
    atomic_json(folder / "embedding_ids.json", {
        "document_pids": [str(d["pid"]) for d in corpus],
        "question_qids": [str(q["qid"]) for q in queries],
        "shape": list(docs.shape), "dtype": "float32",
        "selection_usage": "archived only in paper all-candidates arm"})
    del model
    import torch
    torch.cuda.empty_cache()
    return docs, qs


def prepare(a):
    import faiss
    import numpy as np
    import torch
    from FlagEmbedding import FlagReranker, BGEM3FlagModel
    progress = Progress(["contract", "BGE-M3 vectors/index", "all-document rerank", "BGE-M3 expert scores", "cache summary"])
    progress.weights[0] = progress.weights[-1] = 1.0
    try:
        progress.stage(0, 1)
        corpus, queries, manifest = preflight(a, write_manifest=True)
        folder = manifest.parent
        progress.complete(f"{a.dataset} docs={len(corpus)} questions={len(queries)}; {manifest}")
        progress.log("[plan] paper arm: archive dense vectors/FAISS; score ALL corpus passages with CE; top90; score BGE-M3 dense expert priors. No generation or gold selection.")
        progress.stage(1, 1)
        docs, qs = _vectors(a, corpus, queries, folder, progress)
        progress.complete(f"vectors={docs.shape}/{qs.shape}; index={folder/'corpus.faiss'}")

        rankdir = folder / "rankings"
        pending = [q for q in queries if not result_file(rankdir, str(q["qid"])).exists()]
        progress.stage(2, len(queries), initial=len(queries)-len(pending), unit="q")
        if pending:
            gpu_gate(a.min_free_gib)
            reranker = FlagReranker(str(a.reranker), use_fp16=True, devices=["cuda:0"])
            index = faiss.read_index(str(folder / "corpus.faiss")) if a.candidate_mode == "faiss" else None
            qlookup = {str(q["qid"]): i for i, q in enumerate(queries)}
            for q in pending:
                qid, qi = str(q["qid"]), qlookup[str(q["qid"])]
                if index is None:
                    candidates = list(range(len(corpus)))
                else:
                    _, found = index.search(qs[qi:qi+1], min(a.fetch_k, len(corpus)))
                    candidates = [int(x) for x in found[0] if x >= 0]
                texts = [format_doc(corpus[i]) for i in candidates]
                ce = length_safe_scores(q["query_text"], texts,
                                        lambda batch: reranker.compute_score(batch, normalize=True))
                if not all(np.isfinite(ce)):
                    raise RuntimeError(f"Non-finite reranker scores: {qid}")
                ce_half = torch.as_tensor(ce, dtype=torch.float16)
                chosen = torch.topk(ce_half, k=90).indices.tolist()  # official Top-K selection
                all_order = torch.argsort(ce_half, descending=True).tolist()
                dense_sim = (docs[candidates] @ qs[qi]).tolist()
                rows = [{"pid": str(corpus[candidates[j]]["pid"]), "corpus_index": candidates[j],
                         "rerank_score": float(ce[j]), "rerank_score_fp16": float(ce_half[j]),
                         "dense_vector_similarity": float(dense_sim[j])} for j in range(len(candidates))]
                atomic_json(result_file(rankdir, qid), {
                    "qid": qid, "candidate_mode": a.candidate_mode,
                    "candidate_count": len(candidates), "all_ranked": [rows[j] for j in all_order],
                    "top90": [rows[j] for j in chosen],
                    "selection": "torch.topk(float16(CE), 90) as in official code"})
                progress.update()
            del reranker
            torch.cuda.empty_cache()
        progress.complete(f"ranked={len(queries)}; durable={rankdir}")

        priordir = folder / "expert_scores"
        pending = [q for q in queries if not result_file(priordir, str(q["qid"])).exists()]
        progress.stage(3, len(queries), initial=len(queries)-len(pending), unit="q")
        if pending:
            gpu_gate(a.min_free_gib)
            model = BGEM3FlagModel(str(a.bge), use_fp16=True, devices=["cuda:0"])
            for q in pending:
                qid = str(q["qid"])
                ranking = json.loads(result_file(rankdir, qid).read_text())
                selected = ranking["top90"]
                texts = [format_doc(corpus[item["corpus_index"]]) for item in selected]
                dense = length_safe_scores(q["query_text"], texts,
                    lambda batch: model.compute_score(batch, weights_for_different_modes=[0.4, 0.2, 0.4])["dense"])
                if len(dense) != 90 or not all(np.isfinite(dense)):
                    raise RuntimeError(f"Bad BGE expert scores: {qid}")
                atomic_json(result_file(priordir, qid), {"qid": qid, "pids": [x["pid"] for x in selected],
                            "bge_dense_pair_raw": dense,
                            "bge_dense_vector_similarity": [x["dense_vector_similarity"] for x in selected],
                            "ce_normalized": [x["rerank_score_fp16"] for x in selected]})
                progress.update()
            del model
            torch.cuda.empty_cache()
        progress.complete(f"expert score records={len(queries)}; durable={priordir}")
        progress.stage(4, 1)
        atomic_json(folder / "cache_summary.json", {"dataset": a.dataset, "questions": len(queries),
                    "corpus_documents": len(corpus), "all_rankings": len(list(rankdir.glob('*.json'))),
                    "expert_scores": len(list(priordir.glob('*.json'))), "paper_arm": a.candidate_mode == "all"})
        progress.complete(f"{folder/'cache_summary.json'}")
    finally:
        progress.close()


def normalize_answer(value):
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def subspan_em(prediction, answers):
    # Official PCED scorer: ANY normalized gold answer appearing as a substring.
    pred = normalize_answer(prediction)
    return int(any(normalize_answer(answer) in pred for answer in answers))


def prompt_inputs(tokenizer, question, texts):
    # Deliberately preserve the official run_pced.py chat-template/suffix layout.
    prompts = []
    suffix = None
    for text in [None, *texts]:
        body = "" if text is None else text
        separator = "\n" + "#" * (len(body) + 1)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": body + separator}]
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=False, enable_thinking=False)
        prefix, trailing = rendered.split(separator)
        prompts.append(prefix)
        if suffix is None:
            suffix = trailing
    question_text = QUESTION.format(question=question) + suffix + ANSWER_PREFIX
    return prompts, question_text


def prior_tensor(scores, policy, device, dtype):
    import torch
    raw = torch.tensor(scores["bge_dense_pair_raw"], device=device, dtype=dtype)
    ce = torch.tensor(scores["ce_normalized"], device=device, dtype=torch.float16)
    if policy == "paper":
        raw = ((raw.float() + 1.0) / 2.0).clamp(0, 1 - 1e-8).to(dtype)
    # released-script omits --normalize even though its filename says normalize.
    return 2 * ce * raw / (ce + raw + 1e-8)


def evaluate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    progress = Progress(["contract/cache", "load frozen Llama", "PCED test generation", "Subspan EM summary"])
    progress.weights[0] = progress.weights[-1] = 1.0
    try:
        progress.stage(0, 1)
        corpus, queries, manifest = preflight(a, write_manifest=False)
        if not manifest.exists():
            raise RuntimeError("Prepare retrieval/ranking cache before evaluation")
        folder = manifest.parent
        by_pid = {str(doc["pid"]): doc for doc in corpus}
        for q in queries:
            qid = str(q["qid"])
            rank_path = result_file(folder / "rankings", qid)
            prior_path = result_file(folder / "expert_scores", qid)
            if not rank_path.exists() or not prior_path.exists():
                raise FileNotFoundError(f"Missing complete cache for {qid}: {rank_path} / {prior_path}")
        progress.complete(f"dataset={a.dataset} test={len(queries)} docs={len(corpus)}; {manifest}")
        progress.log("[plan] frozen Llama-3.1-8B-Instruct + official PCED decoder; paper LOFT prompts; top90 document experts plus amateur; greedy 32 tokens; resumable per question")
        outdir = folder / "predictions"
        pending = [q for q in queries if not result_file(outdir, str(q["qid"])).exists()]
        progress.stage(1, 1)
        if pending:
            gpu_gate(a.min_free_gib)
            sys.path.insert(0, str(a.official_code))
            from src.methods.pced import prefill, decode_from_cache
            tokenizer = AutoTokenizer.from_pretrained(str(a.llama), local_files_only=True, padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(str(a.llama), local_files_only=True,
                        torch_dtype=torch.bfloat16, attn_implementation=a.attention_impl).to("cuda:0").eval()
            model.requires_grad_(False)
            random.seed(42)
            torch.manual_seed(42)
        progress.complete(f"one shared frozen model; backend={a.attention_impl}; cached={len(queries)-len(pending)}")
        progress.stage(2, len(queries), initial=len(queries)-len(pending), unit="q")
        for q in pending:
            qid = str(q["qid"])
            ranking = json.loads(result_file(folder / "rankings", qid).read_text())
            scores = json.loads(result_file(folder / "expert_scores", qid).read_text())
            selected = ranking["top90"]
            if len(selected) != 90 or scores["pids"] != [x["pid"] for x in selected]:
                raise RuntimeError(f"Top90 / expert-score ordering mismatch: {qid}")
            texts = [format_doc(by_pid[item["pid"]]) for item in selected]
            prompts, question = prompt_inputs(tokenizer, q["query_text"], texts)
            tokenized_prompt = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                padding=True, truncation=True, max_length=model.config.max_position_embeddings-32).to(model.device)
            tokenized_question = tokenizer([question] * len(prompts), return_tensors="pt",
                                           add_special_tokens=False).to(model.device)
            retrieval_scores = prior_tensor(scores, a.score_policy, model.device, model.dtype)
            started = time.monotonic()
            with torch.inference_mode():
                cache = prefill(model, tokenized_prompt)
                ids, winners = decode_from_cache(model, tokenizer, cache, tokenized_prompt,
                    tokenized_question, max_new_tokens=32, beta=None,
                    retrieval_scores=retrieval_scores, gamma=2.5, normalize=False,
                    return_winner_trace=True)
            prediction = tokenizer.decode(ids, skip_special_tokens=True)
            atomic_json(result_file(outdir, qid), {
                "qid": qid, "dataset": a.dataset, "prediction": prediction,
                "subspan_em": subspan_em(prediction, q["answers"]), "answers": q["answers"],
                "top90_pids": scores["pids"], "expert_winners": winners,
                "generated_token_ids": ids.tolist(), "seconds": time.monotonic() - started,
                "score_policy": a.score_policy, "backend": a.attention_impl})
            del cache, tokenized_prompt, tokenized_question, retrieval_scores, ids
            progress.update()
        progress.complete(f"predictions={len(queries)}; {outdir}")
        progress.stage(3, 1)
        rows = [json.loads(result_file(outdir, str(q["qid"])).read_text()) for q in queries]
        summary = {"dataset": a.dataset, "n": len(rows), "missing": 0,
                   "correct": sum(x["subspan_em"] for x in rows),
                   "subspan_em": sum(x["subspan_em"] for x in rows) / len(rows),
                   "paper_arm": a.candidate_mode == "all", "score_policy": a.score_policy,
                   "backend": a.attention_impl, "top_k": 90,
                   "mean_seconds_per_question": sum(x["seconds"] for x in rows) / len(rows)}
        atomic_json(folder / "summary.json", summary)
        progress.complete(f"{folder/'summary.json'}")
        progress.log(f"[result] {a.dataset} Subspan EM={summary['subspan_em']:.3f} ({summary['correct']}/{summary['n']})")
    finally:
        progress.close()


def main():
    a = arguments()
    if a.phase == "preflight":
        corpus, queries, manifest = preflight(a, write_manifest=False)
        print(f"[preflight OK] {a.dataset}: corpus={len(corpus)}, test={len(queries)}, manifest={manifest}; no GPU allocated")
    elif a.phase == "prepare":
        prepare(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()

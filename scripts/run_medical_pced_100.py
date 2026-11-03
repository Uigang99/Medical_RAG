#!/usr/bin/env python3
"""Run the unchanged LOFT PCED decoder on seeded 100-question medical pilots.

Only the input adapter and first-stage candidate selection differ: existing
BGE-M3/FAISS vectors shortlist passages before the same CE Top-90 policy.
Gold document IDs and answers are never consulted by retrieval or generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "datasets/medical_qa_gold_union_v1"
sys.path.insert(0, str(ROOT))
from scripts import run_loft_pced_paper as loft

DATASETS = {
    "bioasq11_factoid": DATA_ROOT / "pced_style_short_answer_v1/bioasq11_factoid",
    "papersearchqa": DATA_ROOT / "pced_style_short_answer_v1/papersearchqa",
    "covid_qa": DATA_ROOT / "pced_style_short_answer_covid_300tok_v1/covid_qa",
}
VERSION = "medical_pced_seeded100_v1"


def selected_inputs(a):
    folder = DATASETS[a.dataset]
    cp, qp = folder / "index_passages.jsonl", folder / "queries.jsonl"
    corpus, all_queries = list(loft.jsonl(cp)), list(loft.jsonl(qp))
    if len(corpus) < a.fetch_k or len(all_queries) < 100:
        raise RuntimeError("Medical corpus or query set is unexpectedly small")
    ids = json.loads((folder / "vector_db/ids.json").read_text())
    pids = [str(doc["pid"]) for doc in corpus]
    qids = [str(q["qid"]) for q in all_queries]
    if (pids != ids["document_pids"] or qids != ids["question_qids"] or
            len(set(pids)) != len(pids) or len(set(qids)) != len(qids)):
        raise RuntimeError("Input order/IDs do not match the frozen vector index")
    eligible = [i for i, q in enumerate(all_queries)
                if q.get("metadata", {}).get("split") == "train" and
                q.get("query_text") and q.get("answers")]
    if len(eligible) < 100:
        raise RuntimeError(f"Only {len(eligible)} eligible train questions")
    indices = sorted(random.Random(a.sample_seed).sample(eligible, 100))
    return cp, qp, corpus, [all_queries[i] for i in indices]


def medical_contract(a, cp, qp, n):
    folder = DATASETS[a.dataset]
    _, _, _, queries = selected_inputs(a)
    return {
        "version": VERSION, "hypothesis": "PCED improves use of the same fixed Top-90 evidence",
        "dataset": a.dataset, "split": "train-origin seeded pilot", "questions": n,
        "sample_seed": a.sample_seed, "sample_qids": [str(q["qid"]) for q in queries],
        "corpus_sha256": loft.sha(cp), "queries_sha256": loft.sha(qp),
        "vector_ids_sha256": loft.sha(folder / "vector_db/ids.json"),
        "vector_manifest_sha256": loft.sha(folder / "vector_db/manifest.json"),
        "candidate_mode": "faiss", "fetch_k": a.fetch_k, "top_k": 90,
        "score_policy": a.score_policy, "max_new_tokens": 32,
        "llama": str(a.llama), "bge": str(a.bge), "reranker": str(a.reranker),
        "bge_revision": loft.MODEL_BGE, "reranker_revision": loft.MODEL_RERANK,
        "retriever_sdpa_backend": "math", "llama_cudnn_sdpa_enabled": False,
        "official_pced_commit": loft.OFFICIAL_SHA, "attention_impl": a.attention_impl,
        "seed": 42, "script_sha256": loft.sha(Path(__file__)),
        "loft_decoder_script_sha256": loft.sha(Path(loft.__file__)),
        "paper_difference": "Existing dataset BGE-M3 FAISS Top-256, then CE Top-90; not full-corpus CE",
        "backend_difference": ("SDPA instead of official FlashAttention-2"
                               if a.attention_impl == "sdpa" else "Official FlashAttention-2"),
        "gold_use": "answers for scoring only; qrels never used by retrieval or generation",
    }


def medical_vectors(a, corpus, queries, output_folder, progress):
    import faiss
    import numpy as np
    source = DATASETS[a.dataset] / "vector_db"
    docs = np.load(source / "corpus_dense.npy", mmap_mode="r")
    all_qs = np.load(source / "query_dense.npy", mmap_mode="r")
    ids = json.loads((source / "ids.json").read_text())
    if (docs.shape != (len(corpus), 1024) or
            all_qs.shape != (len(ids["question_qids"]), 1024)):
        raise RuntimeError("Existing dense vector shapes do not match medical inputs")
    positions = {qid: i for i, qid in enumerate(ids["question_qids"])}
    qs = np.asarray(all_qs[[positions[str(q["qid"])] for q in queries]])
    index_path = output_folder / "corpus.faiss"
    if not index_path.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "corpus.faiss", index_path)
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(corpus) or index.d != 1024:
        raise RuntimeError("FAISS index is incompatible with medical corpus")
    loft.atomic_json(output_folder / "embedding_ids.json", {
        "document_pids": ids["document_pids"],
        "question_qids": [str(q["qid"]) for q in queries],
        "source_vector_db": str(source), "selection_usage": "FAISS Top-256"})
    return docs, qs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "prepare", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/medical_pced_seeded100_v1")
    parser.add_argument("--sample-seed", type=int, default=20261103)
    parser.add_argument("--fetch-k", type=int, default=256)
    parser.add_argument("--min-free-gib", type=float, default=40)
    a = parser.parse_args()
    if a.fetch_k < 90:
        parser.error("--fetch-k must be >=90")
    a.output_root = a.output_root.resolve()
    a.llama = (ROOT.parent / "models/Llama-3.1-8B-Instruct").resolve()
    a.bge = (ROOT.parent / "models/bge-m3-pced").resolve()
    a.reranker = (ROOT.parent / "models/bge-reranker-v2-m3-pced").resolve()
    a.official_code = (ROOT / "cache/pced_official").resolve()
    a.candidate_mode, a.score_policy, a.attention_impl = "faiss", "paper", "sdpa"
    a.limit, a.embedding_batch = 0, 16
    a.corpus_root = DATA_ROOT
    loft.load_inputs = selected_inputs
    loft.contract = medical_contract
    loft._vectors = medical_vectors
    if a.phase == "preflight":
        corpus, queries, manifest = loft.preflight(a, write_manifest=False)
        print(f"[preflight OK] {a.dataset}: sampled={len(queries)}, docs={len(corpus)}, "
              f"first_qid={queries[0]['qid']}, output={manifest}; no GPU allocated")
    elif a.phase == "prepare":
        print("[medical plan] Existing BGE-M3/FAISS Top-256 -> CE Top-90 -> "
              "BGE-M3 expert priors; train-origin seeded 100; no gold in ranking", flush=True)
        loft.prepare(a)
    else:
        print("[medical plan] Unchanged official PCED decoder; fixed CE Top-90; "
              "train-origin seeded 100; no gold in generation", flush=True)
        loft.evaluate(a)


if __name__ == "__main__":
    main()

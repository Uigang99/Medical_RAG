#!/usr/bin/env python3
"""Prepare reusable A/B/C proposition-pair candidates. No KEV, Llama or QA evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import proposition_pair_candidates as c
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, sha_file
from medrag.training.graph_input_progress import Progress

STAGES = ("source inventory", "KG mapping", "load MedCPT", "embed propositions", "select common pairs", "export candidates")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true", help="Read-only CPU metadata/first-row check; no full source scan or model loading")
    p.add_argument("--source-cohort", type=Path, default=ROOT / "results/rag2_route_feedback_kev_test6545_v2/cohort.jsonl")
    p.add_argument("--graph", type=Path, default=ROOT / "datasets/kg/rag2_test6545_union88294_stage7_surface_kg_v1/graph.sqlite3")
    p.add_argument("--encoder", type=Path, default=ROOT.parent / "models/MedCPT-Article-Encoder")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_common_pair_candidates_test6545_v1")
    p.add_argument("--shared-entity-k", type=int, default=1)
    p.add_argument("--direct-kg-k", type=int, default=1)
    p.add_argument("--semantic-k", type=int, default=2)
    p.add_argument("--min-cosine", type=float, default=.70, help="Untuned pilot floor, not a calibrated probability")
    p.add_argument("--max-common-fraction", type=float, default=.01)
    p.add_argument("--audit-per-stratum", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument("--embedding-chunk", type=int, default=2048)
    p.add_argument("--cpu-threads", type=int, default=8)
    a = p.parse_args(argv)
    if not (0 <= a.shared_entity_k <= 8 and 0 <= a.direct_kg_k <= 8 and 1 <= a.semantic_k <= 8
            and 0 <= a.min_cosine <= 1 and 0 < a.max_common_fraction <= 1
            and 0 <= a.audit_per_stratum <= 16 and 1 <= a.batch_size <= 1024
            and 512 <= a.batch_tokens <= 131072 and 1 <= a.cpu_threads <= 64
            and 64 <= a.embedding_chunk <= 8192):
        p.error("Invalid candidate/resource bounds")
    return a


def settings(a):
    return {"shared_entity_k": a.shared_entity_k, "direct_kg_k": a.direct_kg_k,
            "semantic_k": a.semantic_k, "document_adjacent_k": 2,
            "min_cosine": a.min_cosine, "max_common_fraction": a.max_common_fraction,
            "reserve_cross_document": True, "audit_per_stratum": a.audit_per_stratum, "seed": a.seed}


def contract(a):
    # This check happens BEFORE output writes, full cohort scans, tokenization or CUDA.
    files = {"cohort": a.source_cohort, "KG": a.graph,
             "encoder_config": a.encoder / "config.json", "tokenizer_config": a.encoder / "tokenizer_config.json",
             "vocab": a.encoder / "vocab.txt",
             "candidate_code": ROOT / "medrag/proposition_pair_candidates.py", "runner": Path(__file__),
             "window_code": ROOT / "medrag/training/dynamic_kg_features.py"}
    weights = sorted(a.encoder.glob("*.safetensors")) or sorted(a.encoder.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No local encoder weights: {a.encoder}")
    files.update({f"encoder_weights_{i}": path for i, path in enumerate(weights)})
    tokenizer_json = a.encoder / "tokenizer.json"
    if tokenizer_json.exists():
        files["tokenizer_json"] = tokenizer_json
    inputs = {k: identity(path, full_hash=k in ("cohort", "candidate_code", "runner", "window_code")) for k, path in files.items()}
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Use the launcher (CUDA_VISIBLE_DEVICES=1); physical GPU1 only")
    config = json.loads((a.encoder / "config.json").read_text())
    if config.get("model_type") != "bert" or config.get("max_position_embeddings", 0) < 512:
        raise ValueError("Expected local MedCPT BERT with 512-token windows")
    with sqlite3.connect(f"file:{a.graph.resolve()}?mode=ro", uri=True) as db:
        for table, columns in {"proposition": "proposition_id,document_id,claim",
                               "mention": "proposition_id,entity_node_id,assertion_node_id",
                               "entity": "entity_node_id,preferred_surface,canonical_concept_id,normalization_level",
                               "binary_relation": "edge_id,assertion_node_id,proposition_id,subject_entity_id,relation_surface,object_entity_id"}.items():
            db.execute(f"SELECT {columns} FROM {table} LIMIT 0")
    with a.source_cohort.open() as f:
        first = json.loads(next(f))
    docs = first["documents"]
    if len(docs) != 8 or [d["rank"] for d in docs] != list(range(1, 9)):
        raise ValueError("Expected original fixed Top-8 cohort")
    # No questions/options/gold are copied into downstream artifacts.
    value = dict(version=c.VERSION, hypothesis="Cheap multi-route proposals retain useful A/B/C relations without all-pairs KEV",
                 inputs=inputs, settings=settings(a), expected_questions=6545, expected_propositions=283462,
                 scope="Original TEST6545 Top-8 only; no new retrieval; no gold or question-conditioned annotation",
                 encoder=dict(pooling="content-length-weighted mean of window CLS, then L2 normalization",
                              max_window_tokens=512, truncation=False, dtype="float32 weights with bfloat16 autocast",
                              dimension=config["hidden_size"], batch_size=a.batch_size, batch_tokens=a.batch_tokens,
                              chunk=a.embedding_chunk, sdpa_backends=["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]),
                 runtime=dict(gpu=1, cpu_threads=a.cpu_threads),
                 packages={x: importlib.metadata.version(x) for x in ("torch", "transformers", "numpy", "threadpoolctl")},
                 graph_labels="None: common candidates only, independent A/B/C evaluation is a separate stage",
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    old = a.output_root / "manifest.json"
    if old.exists() and json.loads(old.read_text()) != {"contract": value, "contract_sha256": digest(value)}:
        raise RuntimeError(f"Incompatible existing output: {old}; use a NEW output-root")
    parent = a.output_root
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < 8 * 1024**3:
        raise RuntimeError(f"Less than 8 GiB free for database/embeddings/atomic exports: {free/1024**3:.2f} GiB")
    return value


def encode(a, db, progress, dimension):
    import numpy as np
    path = a.output_root / "embeddings.npy"
    n = c.get_meta(db, "inventory_stats")["unique_propositions"]
    done = c.get_meta(db, "embedding_next", 0)
    if path.exists():
        vectors = np.lib.format.open_memmap(path, mode="r+")
        if vectors.shape != (n, dimension) or vectors.dtype != np.float32:
            raise ValueError("Embedding cache shape/dtype mismatch")
    else:
        if done:
            raise FileNotFoundError("Committed embedding cache is missing")
        vectors = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, dimension))
    progress.stage(2, 1, initial=int(done == n), unit="model")
    if done == n:
        if sha_file(path) != c.get_meta(db, "embedding_sha256"):
            raise ValueError("Completed embedding cache checksum mismatch")
        progress.complete("MedCPT loading skipped; complete embedding cache verified")
        progress.stage(3, n, initial=n, unit="prop")
        progress.complete(f"reused {n} embeddings; {path}")
        return vectors
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging
    from medrag.training.dynamic_kg_features import token_windows
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU1 is not available; no silent CPU fallback")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(a.encoder, local_files_only=True)
    model = AutoModel.from_pretrained(a.encoder, local_files_only=True, dtype=torch.float32,
                                    attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    progress.complete("frozen MedCPT Article Encoder; length buckets; CUDA device 0 = physical GPU1")
    progress.stage(3, n, initial=done, unit="prop")
    long_count = c.get_meta(db, "long_propositions", 0)
    while done < n:
        rows = list(db.execute("SELECT idx,claim FROM proposition WHERE idx>=? ORDER BY idx LIMIT ?", (done, a.embedding_chunk)))
        windows = []
        for owner, r in enumerate(rows):
            pieces = token_windows(tokenizer, r["claim"], 512)
            long_count += len(pieces) > 1
            for ids in pieces:
                windows.append((owner, ids, max(1, len(ids)-2)))
        windows.sort(key=lambda x: (len(x[1]), x[0]))
        sums = np.zeros((len(rows), dimension), dtype=np.float32)
        mass = np.zeros(len(rows), dtype=np.float32)
        cursor, ceiling = 0, a.batch_size
        while cursor < len(windows):
            count = min(ceiling, len(windows)-cursor)
            while count > 1 and len(windows[cursor+count-1][1])*count > a.batch_tokens:
                count = max(1, count//2)
            try:
                part = windows[cursor:cursor+count]
                batch = tokenizer.pad({"input_ids": [x[1] for x in part]}, padding=True, return_tensors="pt").to("cuda:0")
                with torch.inference_mode(), sdpa_kernel(backends), torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = model(**batch).last_hidden_state[:, 0].float().cpu().numpy()
                if not np.isfinite(encoded).all():
                    raise FloatingPointError("Nonfinite encoder output")
                for (owner, _, weight), v in zip(part, encoded, strict=True):
                    sums[owner] += v*weight
                    mass[owner] += weight
                cursor += count
                del batch, encoded
            except torch.cuda.OutOfMemoryError:
                if count == 1:
                    raise
                if "batch" in locals():
                    del batch
                if "encoded" in locals():
                    del encoded
                torch.cuda.empty_cache()
                ceiling = max(1, count//2)
                progress.log(f"[OOM] uncommitted encoder batch retry at {ceiling} windows")
        sums /= mass[:, None]
        norm = np.linalg.norm(sums, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or (norm <= 0).any():
            raise ValueError("Invalid embedding norm")
        vectors[done:done+len(rows)] = sums / norm
        vectors.flush()
        with path.open("rb") as f:
            os.fsync(f.fileno())
        with db:
            done += len(rows)
            c.set_meta(db, "embedding_next", done)
            c.set_meta(db, "long_propositions", long_count)
            if done == n:
                c.set_meta(db, "embedding_sha256", sha_file(path))
        progress.update(len(rows))
    del model
    torch.cuda.empty_cache()
    progress.complete(f"{n} embeddings; no token truncation; multi-window propositions={long_count}; {path}")
    return vectors


def select_all(a, db, vectors, progress):
    import numpy as np
    from threadpoolctl import threadpool_limits
    stats = c.get_meta(db, "inventory_stats")
    done = db.execute("SELECT COUNT(*) FROM question_result").fetchone()[0]
    progress.stage(4, stats["questions"], initial=done, unit="question")
    todo = [r[0] for r in db.execute("""SELECT q.qid FROM question q LEFT JOIN question_result r ON q.qid=r.qid
        WHERE r.qid IS NULL ORDER BY q.ordinal""")]
    # Small 50x50 products should not launch dozens of BLAS threads per question.
    with threadpool_limits(limits=1, user_api="blas"):
        for qid in todo:
            ps, entities, triples = c.question_inputs(db, qid)
            v = np.asarray(vectors[[p["idx"] for p in ps]], dtype=np.float32)
            cosine = np.clip(v @ v.T, -1, 1)
            result = c.select_pairs(ps, entities, triples, cosine, settings(a), stats["unique_documents"], qid)
            c.store_question(db, result)
            progress.update()
    progress.complete(f"all {stats['questions']} question records committed; global pair IDs deduplicated")


def export(a, db, progress):
    stats = c.get_meta(db, "inventory_stats")
    progress.stage(5, 4, unit="file")
    atomic_jsonl(a.output_root / "propositions.jsonl", (
        dict(proposition_id=p["pid"], document_id=p["did"], position=p["pos"], claim=p["claim"], kg_present=bool(p["kg_present"]))
        for p in db.execute("SELECT * FROM proposition ORDER BY idx")))
    progress.update()
    atomic_jsonl(a.output_root / "pairs.jsonl", (
        dict(pair_id=digest(sorted((p["a_pid"], p["b_pid"]))),
             source=min(p["a_pid"], p["b_pid"]), target=max(p["a_pid"], p["b_pid"]),
             graphs_to_evaluate=["A", "B", "C"], status="unassessed")
        for p in db.execute("""SELECT a.pid a_pid,b.pid b_pid FROM pair x
            JOIN proposition a ON a.idx=x.a JOIN proposition b ON b.idx=x.b ORDER BY x.a,x.b""")))
    progress.update()
    totals, channels = Counter(), Counter()
    audit_populations, audit_samples = Counter(), Counter()

    def question_records():
        for row in db.execute("SELECT r.body FROM question_result r JOIN question q ON q.qid=r.qid ORDER BY q.ordinal"):
            r = json.loads(row[0])
            for k, v in r["stats"].items():
                if k != "channels":
                    totals[k] += v
            channels.update(r["stats"]["channels"])
            for k, s in r["audit_strata"].items():
                audit_populations[k] += s["population"]
                audit_samples[k] += s["sample_size"]
            for pair in r["pairs"]:
                pair.pop("indices", None)
                for t in pair["bridge_triples"]:
                    t["proposition_id"] = db.execute("SELECT pid FROM proposition WHERE idx=?", (t.pop("idx"),)).fetchone()[0]
            for pair in r["rejected_audit"]:
                for t in pair["bridge_triples"]:
                    t["proposition_id"] = db.execute("SELECT pid FROM proposition WHERE idx=?", (t.pop("idx"),)).fetchone()[0]
            yield r
    atomic_jsonl(a.output_root / "question_candidates.jsonl", question_records())
    progress.update()
    unique = db.execute("SELECT COUNT(*) FROM pair").fetchone()[0]
    report = {**stats, "selected_pair_occurrences": totals["selected_pairs"], "unique_selected_pairs": unique,
              "pair_count_reduction": 1-unique/stats["unique_cooccurring_pairs"],
              "candidate_isolated_proposition_occurrences": totals["isolated_propositions"],
              "selected_cross_document_pair_occurrences": totals["cross_document_pairs"],
              "channel_pair_occurrences_overlapping": dict(channels),
              "canonical_entity_proposition_occurrences": totals["canonical_entity_propositions"],
              "missing_KG_unique_propositions": db.execute("SELECT COUNT(*) FROM proposition WHERE kg_present=0").fetchone()[0],
              "long_propositions_windowed_not_truncated": c.get_meta(db, "long_propositions", 0),
              "rejected_audit_populations": dict(audit_populations), "rejected_audit_sample_counts": dict(audit_samples),
              "KEV_calls": 0, "relation_recall": None, "relation_precision": None,
              "warning": "Candidate coverage is NOT semantic relation recall. No A/B/C labels or QA scores were computed.",
              "settings": settings(a)}
    atomic_json(a.output_root / "summary.json", report)
    progress.update()
    progress.complete(f"{unique:,} unique common candidates; {a.output_root}")
    return report


def run(a):
    value = contract(a)
    if a.preflight:
        print(json.dumps({"preflight": "passed", "source_scan": "first row only; full counts checked at execution",
                          "expected_questions": 6545, "expected_unique_propositions": 283462,
                          "settings": settings(a), "GPU_loaded": False, "writes": False,
                          "output_root": str(a.output_root)}, ensure_ascii=False, indent=2))
        return
    a.output_root.mkdir(parents=True, exist_ok=True)
    with (a.output_root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another candidate job owns this output root") from None
        h = ensure_contract(a.output_root / "manifest.json", value)
        db = c.connect(a.output_root / "candidates.sqlite3")
        c.initialize(db, h)
        progress = Progress(STAGES)
        runtime_file = a.output_root / "RUNTIME.json"
        history = {}
        if runtime_file.exists():
            history = json.loads(runtime_file.read_text())
            if history["contract_hash"] != h:
                raise ValueError("Timing contract mismatch")
            progress.weights = history["weights"]
        try:
            progress.log("[plan] original TEST6545 inventory -> indexed KG -> MedCPT GPU1 -> common candidates -> global export; no KEV/QA")
            progress.log(f"[resume] inventory={bool(c.get_meta(db, 'inventory_done'))}; KG={c.get_meta(db, 'kg_next', 0)}/283462; "
                         f"embeddings={c.get_meta(db, 'embedding_next', 0)}/283462; "
                         f"questions={db.execute('SELECT COUNT(*) FROM question_result').fetchone()[0]}/6545")
            progress.log("[ETA] New operation rates unknown; each stage calibrates separately; overall stays unknown until all pending costs are available")
            progress.stage(0, 6545, initial=6545 if c.get_meta(db, "inventory_done") else 0, unit="question")
            stats = c.ingest(db, a.source_cohort, tick=progress.update)
            progress.complete(f"{stats['unique_propositions']} unique propositions; source text/order preserved; {a.output_root / 'candidates.sqlite3'}")
            progress.stage(1, stats["unique_propositions"], initial=c.get_meta(db, "kg_next", 0), unit="prop")
            c.import_kg(db, a.graph, tick=progress.update)
            progress.complete("canonical identities and source triples mapped; missing KG propositions retained")
            vectors = encode(a, db, progress, value["encoder"]["dimension"])
            select_all(a, db, vectors, progress)
            result = export(a, db, progress)
        except Exception as exc:
            progress.log(f"[FAILED] stage={progress.index+1}/{len(STAGES)} item={progress.done}/{progress.total}; {type(exc).__name__}: {exc}; "
                         f"durable={a.output_root}; rerun identical command to resume committed KG/embedding/question blocks")
            raise
        finally:
            atomic_json(runtime_file, {"contract_hash": h, "weights": progress.weights})
            progress.close()
            db.close()
        print(f"[complete] questions=6545; unique_candidates={result['unique_selected_pairs']}; KEV_calls=0; output={a.output_root}")


if __name__ == "__main__":
    run(arguments())

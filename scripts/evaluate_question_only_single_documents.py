#!/usr/bin/env python3
"""Evaluate each of the first eight retrieved chunks independently.

Three frozen retrieval databases, four rank lists per database, and eight
positions give 96 slots per question. Identical rendered prompts are inferred
once per batch. Gold labels are used only for the saved diagnostic summary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_question_only_ce_mcq_baselines as base
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, jsonl, sha_file
from medrag.training.graph_input_progress import Progress

VERSION = "question_only_single_document_test6545_v1"
DBS = {
    "full": ROOT / "databases/run_cache/question_only_ip_fourway_test6545_v1",
    "scoped": ROOT / "databases/run_cache/question_only_ip_fourway_test6545_pmc_scope_v1",
    "no_pmc": ROOT / "databases/run_cache/question_only_ip_fourway_test6545_pmc_scope_no_pmc_v1",
}
POOLS = ("global_dense", "global_ce", "balanced_dense", "balanced_ce")
ARMS = tuple(f"{db}_{pool}" for db in DBS for pool in POOLS)
REFERENCE = ROOT / "results/question_only_dense_mcq_baselines_test6545_v1/llama-3-8b-instruct_72422241"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=base.MODEL)
    p.add_argument("--benchmark-root", type=Path, default=ROOT / "datasets/benchmark/mcq/unified")
    p.add_argument("--full-root", type=Path, default=DBS["full"])
    p.add_argument("--scoped-root", type=Path, default=DBS["scoped"])
    p.add_argument("--no-pmc-root", type=Path, default=DBS["no_pmc"])
    p.add_argument("--reference-root", type=Path, default=REFERENCE)
    p.add_argument("--output-root", type=Path, default=ROOT / "results/question_only_single_document_test6545_v1")
    p.add_argument("--shard-size", type=int, default=16, help="Questions per durable vLLM batch")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=.88)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args(argv)
    if a.shard_size < 1 or a.max_model_len < 256 or not 0 < a.gpu_memory_utilization < 1:
        p.error("Invalid shard size, context limit, or GPU fraction")
    for key in ("model", "benchmark_root", "full_root", "scoped_root", "no_pmc_root",
                "reference_root", "output_root"):
        setattr(a, key, getattr(a, key).resolve())
    inputs = (a.model, a.benchmark_root, a.full_root, a.scoped_root, a.no_pmc_root,
              a.reference_root)
    if any(a.output_root == x or a.output_root in x.parents or x in a.output_root.parents
           for x in inputs):
        p.error("Output directory overlaps an input")
    return a


def retrieval_paths(a):
    roots = {"full": a.full_root, "scoped": a.scoped_root, "no_pmc": a.no_pmc_root}
    paths = {f"{db}_{pool}": root / f"{pool}.jsonl"
             for db, root in roots.items() for pool in POOLS}
    required = [a.model / "config.json", a.model / "tokenizer.json",
                a.reference_root / "manifest.json"]
    required += [root / "manifest.json" for root in roots.values()]
    required += list(paths.values())
    required += [a.benchmark_root / d / "test.jsonl" for d in base.DATASETS]
    missing = [str(x) for x in required if not x.is_file() or x.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Required input missing/empty: " + ", ".join(missing))
    manifests = {}
    for db, root in roots.items():
        value = json.loads((root / "manifest.json").read_text())
        if (value.get("experiment") != "question_only_ip_fourway_test6545_v1" or
                value.get("ce", {}).get("question_only") is not True or
                value.get("split") not in (None, "test")):
            raise RuntimeError(f"Unexpected question-only TEST retrieval: {db}")
        manifests[db] = value
    return paths, manifests


def reference_rows(a, samples):
    manifest = json.loads((a.reference_root / "manifest.json").read_text())
    old = manifest["contract"]
    current_weights = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                       for p in sorted(a.model.glob("*.safetensors"))}
    if (old["model"] != str(a.model) or old["weights"] != current_weights or
            old["model_config_sha256"] != sha_file(a.model / "config.json") or
            old["tokenizer_sha256"] != sha_file(a.model / "tokenizer.json") or
            old["prompt_system"] != base.SYSTEM or
            old["prompt_order"] != "question-options-evidence-answer" or
            old["max_model_len"] != a.max_model_len or old["seed"] != 42 or
            old["base_code_sha256"] != sha_file(Path(base.__file__)) or
            old["answer"] != "Argmax of raw next-token logprobs for the four space-prefixed option labels; no allowed-token mask; ties A/B/C/D; no rationale"):
        raise RuntimeError("No-RAG reference does not match the frozen model/prompt")
    if manifest["contract_sha256"] != digest(old):
        raise RuntimeError("Invalid No-RAG reference contract")
    shard_files = sorted((a.reference_root / "shards").glob("shard_*.jsonl"))
    rows = [row for path in shard_files for row in jsonl(path)]
    if len(rows) != len(samples):
        raise RuntimeError("Incomplete No-RAG reference")
    for row, sample in zip(rows, samples, strict=True):
        if (row["sample_id"] != sample.id or row["dataset"] != sample.dataset or
                row["gold"] != sample.answer or row["predictions"]["no_rag"] not in "ABCD" or
                row["contract_sha256"] != manifest["contract_sha256"]):
            raise RuntimeError(f"Misaligned No-RAG reference: {sample.id}")
    payload = "\n".join(f"{r['sample_id']}\t{r['gold']}\t{r['predictions']['no_rag']}" for r in rows)
    return rows, {"path": str(a.reference_root), "manifest_sha256": sha_file(a.reference_root / "manifest.json"),
                  "predictions_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def records(samples, paths):
    with ExitStack() as stack:
        handles = {arm: stack.enter_context(path.open(encoding="utf-8")) for arm, path in paths.items()}
        for sample in samples:
            row_docs = {}
            seen = {}
            for arm, handle in handles.items():
                line = handle.readline()
                if not line:
                    raise RuntimeError(f"Short retrieval file: {arm}/{sample.id}")
                row = json.loads(line)
                docs = row.get("documents", [])
                if (row.get("sample_id") != sample.id or row.get("dataset") != sample.dataset or
                        row.get("row_idx") != sample.row_idx or row.get("query_text") != sample.question or
                        len(docs) != 32):
                    raise RuntimeError(f"Misaligned retrieval: {arm}/{sample.id}")
                selected = docs[:8]
                if len({d.get("corpus_id") for d in selected}) != 8:
                    raise RuntimeError(f"Duplicate retrieved document: {arm}/{sample.id}")
                for rank, doc in enumerate(selected, 1):
                    rank_field = "rerank_rank" if arm.endswith("_ce") else "dense_rank"
                    if (doc.get(rank_field) != rank or not doc.get("corpus_id") or
                            not doc.get("text") or not doc.get("source")):
                        raise RuntimeError(f"Invalid document/rank: {arm}/{sample.id}/{rank}")
                    content = (doc.get("title") or "", doc["text"])
                    prior = seen.setdefault(doc["corpus_id"], content)
                    if prior != content:
                        raise RuntimeError(f"Same document ID has different text: {sample.id}/{doc['corpus_id']}")
                row_docs[arm] = selected
            yield sample, row_docs
        for arm, handle in handles.items():
            if handle.readline():
                raise RuntimeError(f"Long retrieval file: {arm}")


def plan_prompts(tokenizer, block):
    prompts, indexes, mappings = [], {}, []
    for sample, row_docs in block:
        mapping = {}
        for arm in ARMS:
            mapping[arm] = []
            for doc in row_docs[arm]:
                rendered = base.prompt(tokenizer, sample, [doc])
                if rendered not in indexes:
                    indexes[rendered] = len(prompts)
                    prompts.append(rendered)
                mapping[arm].append(indexes[rendered])
        mappings.append(mapping)
    return prompts, mappings


def shard_path(a, start):
    return a.output_root / "shards" / f"shard_{start // a.shard_size:05d}.jsonl"


def validate_shard(path, block, references, h):
    if not path.exists():
        return False
    rows = list(jsonl(path))
    if len(rows) != len(block):
        raise RuntimeError(f"Invalid cached shard length: {path}")
    for row, (sample, docs), ref in zip(rows, block, references, strict=True):
        if (row.get("sample_id") != sample.id or row.get("dataset") != sample.dataset or
                row.get("gold") != sample.answer or row.get("contract_sha256") != h or
                row.get("no_rag") != ref["predictions"]["no_rag"] or
                set(row.get("predictions", {})) != set(ARMS) or
                set(row.get("document_ids", {})) != set(ARMS)):
            raise RuntimeError(f"Incompatible cached shard: {path}/{sample.id}")
        for arm in ARMS:
            if (len(row["predictions"][arm]) != 8 or
                    any(v not in "ABCD" for v in row["predictions"][arm]) or
                    row["document_ids"][arm] != [d["corpus_id"] for d in docs[arm]]):
                raise RuntimeError(f"Invalid cached predictions/documents: {path}/{sample.id}/{arm}")
    return True


def score_block(tokenizer, llm, params, ids, block, references, h):
    prompts, mappings = plan_prompts(tokenizer, block)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise RuntimeError("Model output count differs from unique prompt count")
    answers = [base.select_answer(output, ids) for output in outputs]
    rows = []
    for (sample, docs), ref, mapping in zip(block, references, mappings, strict=True):
        rows.append({"sample_id": sample.id, "dataset": sample.dataset, "gold": sample.answer,
                     "contract_sha256": h, "no_rag": ref["predictions"]["no_rag"],
                     "predictions": {arm: [answers[i] for i in mapping[arm]] for arm in ARMS},
                     "document_ids": {arm: [d["corpus_id"] for d in docs[arm]] for arm in ARMS},
                     "document_sources": {arm: [d["source"] for d in docs[arm]] for arm in ARMS}})
    return rows, len(prompts)


def paired_stats(correct, baseline):
    n = len(correct)
    wc = sum(not b and c for b, c in zip(baseline, correct, strict=True))
    cw = sum(b and not c for b, c in zip(baseline, correct, strict=True))
    delta = (wc - cw) / n
    margin = (1.96 * math.sqrt(max(0., (wc + cw - n * delta * delta) / (n - 1)) / n)
              if n > 1 else 0.)
    return {"correct": sum(correct), "total": n, "accuracy": sum(correct) / n,
            "delta_vs_no_rag": delta, "delta_95pct_wald_ci": [delta - margin, delta + margin],
            "wrong_to_correct": wc, "correct_to_wrong": cw}


def summarize(a, samples, h):
    rows = [row for start in range(0, len(samples), a.shard_size) for row in jsonl(shard_path(a, start))]
    if [r["sample_id"] for r in rows] != [s.id for s in samples]:
        raise RuntimeError("Incomplete or misordered result shards")
    baseline = [r["no_rag"] == r["gold"] for r in rows]
    summary = {"version": VERSION, "questions": len(rows), "missing": 0,
               "no_rag": paired_stats(baseline, baseline), "arms": {}}
    csv_rows = []
    for arm in ARMS:
        rank_stats = {}
        per_question = []
        for rank in range(8):
            flags = [r["predictions"][arm][rank] == r["gold"] for r in rows]
            s = paired_stats(flags, baseline)
            s["answer_changed_vs_no_rag"] = sum(r["predictions"][arm][rank] != r["no_rag"] for r in rows)
            by_group, by_source = {}, {}
            for group in ("medmcqa", "medqa", "mmlu"):
                subset = [i for i, r in enumerate(rows) if ("mmlu" if r["dataset"].startswith("mmlu_")
                                                           else r["dataset"]) == group]
                by_group[group] = paired_stats([flags[i] for i in subset], [baseline[i] for i in subset])
            for source in sorted({r["document_sources"][arm][rank] for r in rows}):
                subset = [i for i, r in enumerate(rows) if r["document_sources"][arm][rank] == source]
                by_source[source] = paired_stats([flags[i] for i in subset], [baseline[i] for i in subset])
            s["by_group"] = by_group
            s["by_source"] = by_source
            s["macro3_accuracy"] = sum(v["accuracy"] for v in by_group.values()) / 3
            rank_stats[str(rank + 1)] = s
            csv_rows.append([arm, rank + 1, s["correct"], s["total"], s["accuracy"],
                             s["macro3_accuracy"], s["delta_vs_no_rag"],
                             s["wrong_to_correct"], s["correct_to_wrong"],
                             s["answer_changed_vs_no_rag"]])
        for r in rows:
            per_question.append(sum(v == r["gold"] for v in r["predictions"][arm]) / 8)
        mean = sum(per_question) / len(rows)
        baseline_rate = summary["no_rag"]["accuracy"]
        differences = [x - int(b) for x, b in zip(per_question, baseline, strict=True)]
        delta = sum(differences) / len(rows)
        se = math.sqrt(sum((x - delta) ** 2 for x in differences) / (len(rows) * (len(rows) - 1)))
        summary["arms"][arm] = {
            "ranks": rank_stats, "mean_single_doc_accuracy": mean,
            "mean_delta_vs_no_rag": mean - baseline_rate,
            "mean_delta_95pct_question_paired_ci": [delta - 1.96 * se, delta + 1.96 * se],
            "any_single_doc_correct_gold_oracle": sum(any(v == r["gold"] for v in r["predictions"][arm]) for r in rows),
            "no_rag_or_any_single_doc_correct_gold_oracle": sum(
                r["no_rag"] == r["gold"] or any(v == r["gold"] for v in r["predictions"][arm]) for r in rows),
        }
    atomic_json(a.output_root / "summary.json", summary)
    csv_path = a.output_root / "rank_results.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "retrieval_rank", "correct", "total", "micro_accuracy", "macro3_accuracy",
                         "delta_vs_no_rag", "wrong_to_correct", "correct_to_wrong", "answer_changed_vs_no_rag"])
        writer.writerows(csv_rows)
    os.replace(tmp, csv_path)
    return summary


def main(argv=None):
    a = arguments(argv)
    progress = Progress(["validate inputs/baseline", "check 96 slots/question", "load frozen Llama",
                         "single-document TEST-6545", "summarize ranks/methods"])
    completed = fresh = unique_inferred = 0
    current = "preflight"
    try:
        progress.stage(0, 1, unit="check")
        paths, manifests = retrieval_paths(a)
        samples = base.cohort(a)
        references, ref_contract = reference_rows(a, samples)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        ids = base.label_ids(tokenizer, base.prompt(tokenizer, samples[0], []))
        progress.complete(f"6545 questions; 3 DB x 4 ranks; same No-RAG model/prompt; {a.output_root}")

        progress.stage(1, len(samples), unit="q")
        slots = unique = tokens = maximum = 0
        for sample, docs in records(samples, paths):
            prompts, _ = plan_prompts(tokenizer, [(sample, docs)])
            lengths = [len(v) for v in tokenizer(prompts, add_special_tokens=False).input_ids]
            if max(lengths) + 1 > a.max_model_len:
                raise RuntimeError(f"Single-document context overflow: {sample.id}")
            slots += 96
            unique += len(prompts)
            tokens += sum(lengths)
            maximum = max(maximum, max(lengths))
            progress.update()
        stats = {"questions": len(samples), "slots": slots, "unique_question_prompts": unique,
                 "duplicates_eliminated": slots - unique, "input_tokens": tokens,
                 "max_prompt_tokens": maximum, "label_ids": ids}
        progress.complete(f"{slots:,} slots -> {unique:,} unique question/document prompts; "
                          f"max={maximum} tokens; duplicates={slots-unique:,}")
        if a.preflight_only:
            progress.log("[preflight complete] No GPU/model load or TEST prediction; " + json.dumps(stats))
            return

        source_ids = {str(p): sha_file(p) for p in paths.values()}
        source_manifests = {db: {"path": str(root / "manifest.json"),
                                 "sha256": sha_file(root / "manifest.json"),
                                 "sources": manifests[db].get("sources")}
                            for db, root in (("full", a.full_root), ("scoped", a.scoped_root),
                                             ("no_pmc", a.no_pmc_root))}
        c = {"version": VERSION, "hypothesis": "One retrieved chunk can help or harm answer selection independently",
             "dataset": "Eight unified TEST datasets; 6545 paired MCQs", "arms": ARMS, "positions": list(range(1, 9)),
             "primary_metric": "Mean rank-wise accuracy across eight independent single-document inputs per DB/method",
             "secondary": "Per-rank accuracy, paired corrections/harms, source mix, gold-conditioned any-correct diagnostic",
             "retrieval_files_sha256": source_ids, "retrieval_manifests": source_manifests,
             "benchmark_sha256": {d: sha_file(a.benchmark_root / d / "test.jsonl") for d in base.DATASETS},
             "reference_no_rag": ref_contract, "model": str(a.model),
             "model_weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                               for p in sorted(a.model.glob("*.safetensors"))},
             "model_config_sha256": sha_file(a.model / "config.json"),
             "tokenizer_sha256": sha_file(a.model / "tokenizer.json"),
             "prompt_system": base.SYSTEM, "prompt_order": "question-options-evidence-answer",
             "document_format": "Each original rank supplied alone, displayed as [1] with exact title/text; no scores or original rank shown",
             "answer": "Argmax of raw next-token logprobs for four space-prefixed option labels; no rationale",
             "deduplication": "Exact rendered prompt within each inference block; original source/rank mapped back",
             "gold_policy": "Gold omitted from prompts and model calls; used only after prediction for metrics",
             "oracle_policy": "Any-correct Top-8 is a post-hoc diagnostic, not a deployable selector",
             "max_model_len": a.max_model_len, "shard_size": a.shard_size,
             "gpu_memory_utilization": a.gpu_memory_utilization, "seed": 42,
             "preflight": stats, "code_sha256": sha_file(Path(__file__)),
             "base_code_sha256": sha_file(Path(base.__file__)),
             "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        h = ensure_contract(a.output_root / "manifest.json", c)
        for start in range(0, len(samples), a.shard_size):
            if shard_path(a, start).exists():
                completed += min(a.shard_size, len(samples) - start)
        progress.log(f"[plan] 3 DB x 4 methods x 8 standalone docs x 6545; "
                     f"{unique:,} unique prompts; cached={completed}; GPU1 only; first fresh block calibrates ETA")

        progress.stage(2, 1, unit="model")
        llm = params = None
        if completed < len(samples):
            from vllm import LLM, SamplingParams
            llm = LLM(model=str(a.model), dtype="bfloat16", max_model_len=a.max_model_len,
                      gpu_memory_utilization=a.gpu_memory_utilization, enforce_eager=False,
                      enable_prefix_caching=True, trust_remote_code=False, seed=42)
            params = SamplingParams(temperature=0, max_tokens=1, logprob_token_ids=list(ids.values()))
        progress.complete(f"frozen {a.model.name}; model {'loaded' if llm is not None else 'cached'}")

        progress.stage(3, len(samples), initial=completed, unit="q", kind="validation")
        iterator = records(samples, paths)
        started = time.monotonic()
        for start in range(0, len(samples), a.shard_size):
            size = min(a.shard_size, len(samples) - start)
            block = [next(iterator) for _ in range(size)]
            current = block[0][0].id
            path = shard_path(a, start)
            refs = references[start:start + size]
            if validate_shard(path, block, refs, h):
                continue
            result, count = score_block(tokenizer, llm, params, ids, block, refs, h)
            atomic_jsonl(path, result)
            unique_inferred += count
            fresh += size
            progress.update(size)
            if fresh >= 128 and fresh % 512 < size:
                elapsed = max(time.monotonic() - started, 1e-6)
                progress.log(f"[throughput] fresh={fresh}; prompts={unique_inferred}; "
                             f"{unique_inferred/elapsed:.1f} prompts/s; "
                             f"{fresh/elapsed:.2f} questions/s")
        progress.complete(f"fresh={fresh}; cached={completed}; unique model calls={unique_inferred}; "
                          f"{a.output_root / 'shards'}")
        progress.stage(4, 1, unit="summary")
        summary = summarize(a, samples, h)
        progress.complete(f"12 methods x 8 independent ranks; {a.output_root / 'summary.json'}")
        progress.log(f"[result] No-RAG={summary['no_rag']['accuracy']:.4f}; "
                     f"unique prompts preflight={unique:,}; no gold used in inference")
    except Exception as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/5 item={current} "
                     f"cached_questions={completed} fresh_questions={fresh} "
                     f"durable={a.output_root / 'shards'}; identical command resumes; "
                     f"{type(exc).__name__}: {exc}")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

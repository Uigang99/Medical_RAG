#!/usr/bin/env python3
"""64 TEST questions x (8 fixed original documents + amateur), PLANS ONLY.

--preflight-only selects the cohort, checks all prompts and guided JSON grammar,
and writes the immutable contract WITHOUT loading model weights or using a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from medrag import evidence_planning as task
from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path
from extract_rag2_top8_standalone_propositions import (
    TwoLineProgress, atomic_json, atomic_jsonl, atomic_text, iter_jsonl,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--benchmark-root", type=Path, default=ROOT / "datasets/benchmark")
    p.add_argument("--candidate-cache", type=Path, default=ROOT / "databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2/top8/candidates.jsonl")
    p.add_argument("--reference-root", type=Path, default=ROOT / "results/rag2_pced_direct_reasoning_prompt_top8_test_v4_final")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / task.VERSION)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-output-tokens", type=int, default=1536)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--preflight-only", action="store_true")
    return p


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(args):
    reference_path = args.reference_root / "experiment_manifest.json"
    manifest = json.loads(reference_path.read_text())
    if manifest["questions"] != 6545 or manifest["split"] != "test" or manifest["top_k"] != 8:
        raise RuntimeError("Not the frozen Top-8 TEST6545 reference")
    candidate_sha = sha(args.candidate_cache)
    if candidate_sha != manifest["candidate_cache"]["sha256"]:
        raise RuntimeError("Top-8 cache differs from the No-RAG reference contract")
    if sha(args.model / "config.json") != manifest["llama"]["config_sha256"]:
        raise RuntimeError("Planner model differs from the stratifying Llama baseline")
    if sha(args.model / "tokenizer_config.json") != manifest["llama"]["tokenizer_config_sha256"]:
        raise RuntimeError("Planner tokenizer differs from the baseline")
    for weight in manifest["llama"]["weights"]:
        if (args.model / weight["name"]).stat().st_size != weight["size"]:
            raise RuntimeError("Missing/incomplete checkpoint weight")
    references = list(iter_jsonl(args.reference_root / "predictions.jsonl"))
    refs = {r["sample_id"]: r for r in references}
    samples = []
    benchmark_hashes = {}
    for ds in task.DATASETS:
        path = resolve_benchmark_path(args.benchmark_root, "mcq", "unified", ds, "test")
        benchmark_hashes[ds] = sha(path)
        samples.extend(load_benchmark_samples(path, "mcq", "unified", ds, "test"))
    if len(samples) != 6545 or len(refs) != 6545 or len(references) != 6545 or set(refs) != {s.id for s in samples}:
        raise RuntimeError("Benchmark/reference missing or duplicate IDs")
    population = []
    for sample in samples:
        ref = refs[sample.id]
        if ref["dataset"] != sample.dataset or ref["row_idx"] != sample.row_idx or ref["gold_answer"] != sample.answer:
            raise RuntimeError(f"Benchmark/reference mismatch: {sample.id}")
        pred = ref["predictions"]["no_rag"]
        if pred not in "ABCD" or len(pred) != 1:
            raise RuntimeError(f"Invalid No-RAG answer: {sample.id}")
        population.append({"sample_id": sample.id, "dataset": sample.dataset, "row_idx": sample.row_idx,
                           "question": sample.question, "options": sample.options, "gold_answer": sample.answer,
                           "no_rag_answer": pred, "no_rag_correct": pred == sample.answer})
    selected = task.select_cohort(population, args.questions, args.seed)
    by_id = {r["sample_id"]: r for r in selected}
    seen = set()
    for candidate in iter_jsonl(args.candidate_cache):
        sid = candidate["sample_id"]
        if sid in seen or sid not in refs:
            raise RuntimeError("Invalid candidate IDs")
        seen.add(sid)
        if sid not in by_id:
            continue
        row = by_id[sid]
        if candidate["dataset"] != row["dataset"] or candidate["row_idx"] != row["row_idx"]:
            raise RuntimeError("Selected candidate alignment failed")
        if candidate["key"] != f"{row['dataset']}::test::{sid}::{row['row_idx']}":
            raise RuntimeError("Candidate split/key mismatch")
        docs = candidate["reranked_documents"]
        if [d["rerank_rank"] for d in docs] != list(range(1, 9)):
            raise RuntimeError("Expected exactly eight ordered documents")
        row["documents"] = []
        for doc in docs:
            if not doc["text"].strip():
                raise RuntimeError("Empty document")
            row["documents"].append({"rank": doc["rerank_rank"], "source": doc["source"],
                                      "document_id": doc.get("corpus_id") or doc["db_id"],
                                      "title": doc.get("title", ""), "text": doc["text"]})
    if seen != set(refs) or any("documents" not in r for r in selected):
        raise RuntimeError("Candidate coverage failed")
    identity = {"candidate_cache": {"path": str(args.candidate_cache), "sha256": candidate_sha},
                "reference_manifest_sha256": sha(reference_path),
                "reference_predictions_sha256": sha(args.reference_root / "predictions.jsonl"),
                "benchmark_sha256": benchmark_hashes, "model": manifest["llama"]}
    return selected, task.sampling_report(population, selected), identity


def render_and_check(args, requests):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
    params = SamplingParams(temperature=0.0, seed=args.seed, max_tokens=args.max_output_tokens,
                            structured_outputs=StructuredOutputsParams(json=task.schema()))
    validate_xgrammar_grammar(params)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = {}
    lengths = {}
    for req in requests:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": task.SYSTEM}, {"role": "user", "content": req["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if length + args.max_output_tokens > args.max_model_len:
            raise RuntimeError(f"No truncation allowed: {req['request_id']} input={length} output={args.max_output_tokens}")
        prompts[req["request_id"]] = prompt
        lengths[req["request_id"]] = length
    return prompts, lengths, params


def check_manifest(args, cohort, requests, identity):
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
    contract = {"version": task.VERSION, "hypothesis": "Evidence changes grounded plans and information needs; no answer generation or automatic semantic grading",
                "scope": "Exploratory sample from repeatedly inspected TEST; do not tune on this sample and call it confirmatory",
                "questions": len(cohort), "requests": len(requests), "seed": args.seed,
                "sampling": "Hierarchical largest remainder: main dataset -> C/W -> MMLU subject/C/W; deterministic within-cell hash selection",
                "cohort_sha256": task.digest(cohort), "requests_sha256": task.digest(requests), "output_dir": str(args.output_dir),
                "identity": identity, "question_input": "stem_only; choices absent; under-specification retained",
                "planning_policy": "open-ended; no answer-choice enumeration/comparison/elimination or invented candidate-answer list",
                "evidence": "fixed reranked Top-8 original chunk text; no propositions, KG, reranking, or fallback",
                "labels": "Used only for selection and offline audit, never planner inputs",
                "system_prompt": task.SYSTEM, "schema": task.schema(),
                "generation": {"temperature": 0.0, "seed": args.seed, "max_output_tokens": args.max_output_tokens,
                               "max_model_len": args.max_model_len, "dtype": "bfloat16", "batch_size": args.batch_size,
                               "gpu_memory_utilization": args.gpu_memory_utilization, "enforce_eager": True,
                               "structured_output_backend": "xgrammar", "max_num_seqs": args.batch_size,
                               "max_num_batched_tokens": 16384, "prefix_caching": True},
                "code": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), Path(task.__file__), ROOT / "scripts/extract_rag2_top8_standalone_propositions.py", ROOT / "medrag/benchmark.py")},
                "environment": {"python": platform.python_version(), **{p: version(p) for p in ("torch", "vllm", "transformers", "xgrammar", "jsonschema")}},
                "success_definition": "Human review must assess usefulness/faithfulness/missing information/answer leakage; schema success is not semantic success"}
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["contract"] != contract:
            raise RuntimeError("Planning pilot resume contract mismatch; use a new --output-dir")
    else:
        if (args.output_dir / "generations").exists():
            raise RuntimeError("Generation cache exists without a manifest")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        atomic_json(path, {"contract": contract, "creation_git_commit": commit,
                           "working_tree_dirty_at_creation": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())})
    return task.digest(contract)


def load_cached(directory, requests, contract_sha):
    expected = {r["request_id"]: r for r in requests}
    found = {}
    for path in sorted(directory.glob("batch_*.jsonl")):
        for row in iter_jsonl(path):
            rid = row["request_id"]
            if rid not in expected or rid in found or row["contract_sha256"] != contract_sha or row["request_sha256"] != expected[rid]["request_sha256"]:
                raise RuntimeError(f"Invalid/duplicate durable cache: {path}")
            found[rid] = row
    return found


def save_generation_batch(args, llm, batch, prompts, params, contract_sha, batch_index):
    outputs = llm.generate([prompts[r["request_id"]] for r in batch], params, use_tqdm=False)
    if len(outputs) != len(batch):
        raise RuntimeError("LLM output count mismatch")
    rows = []
    for req, output in zip(batch, outputs, strict=True):
        choice = output.outputs[0]
        plan, flags = task.parse_plan(choice.text, req, choice.finish_reason)
        rows.append({"request_id": req["request_id"], "request_sha256": req["request_sha256"],
                     "contract_sha256": contract_sha, "sample_id": req["sample_id"], "expert_index": req["expert_index"],
                     "dataset": req["dataset"], "document_id": (req["document"] or {}).get("document_id"),
                     "raw_output": choice.text, "plan": plan, "diagnostic_flags": flags,
                     "finish_reason": choice.finish_reason, "output_tokens": len(choice.token_ids)})
    path = args.output_dir / "generations" / f"batch_{batch_index:05d}.jsonl"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite durable generation: {path}")
    atomic_jsonl(path, rows)
    return rows


def write_reports(args, cohort, requests, outputs, report):
    ordered = [outputs[r["request_id"]] for r in requests]
    atomic_jsonl(args.output_dir / "plans.jsonl", ordered)
    flags = Counter(f for row in ordered for f in row["diagnostic_flags"])
    atomic_json(args.output_dir / "summary.json", {"questions": len(cohort), "expected_plans": len(requests),
                "saved_plans": len(ordered), "missing_plans": len(requests) - len(ordered),
                "parsed_schema_valid": sum(r["plan"] is not None for r in ordered),
                "diagnostic_flag_counts": dict(flags), "output_tokens": sum(r["output_tokens"] for r in ordered),
                "sampling": report, "semantic_quality": "NOT_EVALUATED; manual comparison required",
                "qa_accuracy": "NOT_EVALUATED; no answers generated"})
    grouped = []
    reviews = []
    for index, row in enumerate(cohort):
        plans = [outputs[f"{row['sample_id']}::expert{i}"] for i in range(9)]
        grouped.append({**row, "plans": plans})
        for plan in plans:
            reviews.append({"request_id": plan["request_id"], "goal_correct": None, "plan_actionable": None,
                            "evidence_faithful": None, "missing_info_appropriate": None, "answer_leakage": None,
                            "difference_vs_amateur": None, "notes": ""})
    atomic_jsonl(args.output_dir / "by_question.jsonl", grouped)
    review_path = args.output_dir / "manual_review_template.jsonl"
    if not review_path.exists():
        atomic_jsonl(review_path, reviews)
    for offset in range(0, len(grouped), 8):
        lines = ["# Planning comparison — human semantic review required", "",
                 "No answers were requested. Baseline correctness is in cohort.jsonl, not shown here to reduce reviewer bias.", ""]
        for row in grouped[offset:offset + 8]:
            lines += [f"## {row['sample_id']} ({row['dataset']})", "", row["question"], ""]
            for plan in row["plans"]:
                i = plan["expert_index"]
                lines += [f"### {'Amateur' if i == 0 else f'Expert {i}'}", ""]
                if i:
                    doc = row["documents"][i - 1]
                    lines += [f"Source: {doc['document_id']}", "", doc["text"], ""]
                lines += ["```json", json.dumps(plan["plan"], ensure_ascii=False, indent=2) if plan["plan"] is not None else plan["raw_output"], "```", "",
                          f"Structural flags (not quality verdict): {plan['diagnostic_flags']}", ""]
        atomic_text(args.output_dir / "review" / f"questions_{offset+1:02d}_{min(offset+8,len(grouped)):02d}.md", "\n".join(lines))


def main():
    args = args_parser().parse_args()
    if args.batch_size < 1 or args.questions < 1 or args.max_output_tokens < 1 or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("Invalid generation configuration")
    names = ("select TEST cohort", "schema/tokenizer + load Llama", "generate plans", "comparison reports")
    progress = TwoLineProgress(names)
    progress.estimates = [None, None, None, None]
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(5):
            if progress.index:
                progress.render()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    llm = None
    try:
        progress.plan(("6545 TEST -> proportional 64; reuse No-RAG C/W", "CPU grammar/context preflight; GPU1 load only when not preflight",
                       "576 short plans; cached counts below; ETA after fresh batch32", "8 question-grouped review files; no semantic auto-grading"))
        progress.start(1, 1, unit="cohort")
        cohort, report, identity = load_inputs(args)
        requests = task.make_requests(cohort)
        contract_sha = check_manifest(args, cohort, requests, identity)
        atomic_jsonl(args.output_dir / "cohort.jsonl", cohort)
        atomic_jsonl(args.output_dir / "requests.jsonl", requests)
        atomic_json(args.output_dir / "sampling_summary.json", report)
        atomic_text(args.output_dir / "system_prompt.txt", task.SYSTEM)
        cached = load_cached(args.output_dir / "generations", requests, contract_sha)
        remaining = [r for r in requests if r["request_id"] not in cached]
        progress.complete(f"N={len(cohort)} requests={len(requests)} cached={len(cached)} remaining={len(remaining)}; {args.output_dir}")
        progress.start(2, 1, unit="model")
        prompts, lengths, params = render_and_check(args, requests)
        atomic_json(args.output_dir / "preflight.json", {"requests": len(requests), "max_input_tokens": max(lengths.values()),
                    "max_output_tokens": args.max_output_tokens, "context": args.max_model_len,
                    "truncated_inputs": 0, "structured_schema_compiles": True,
                    "question_input": "stem_only", "gpu_generation_executed": False})
        if args.preflight_only:
            progress.complete(f"CPU preflight passed; max input={max(lengths.values())}; NO GPU/model-weight load")
            return
        if remaining:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
                raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1; the wrapper does this")
            from vllm import LLM
            llm = LLM(model=str(args.model), tensor_parallel_size=1, dtype="bfloat16", seed=args.seed,
                      max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                      max_num_batched_tokens=16384, gpu_memory_utilization=args.gpu_memory_utilization,
                      enable_prefix_caching=True, enforce_eager=True, disable_log_stats=True,
                      structured_outputs_config={"backend": "xgrammar"})
        progress.complete("Llama loaded" if remaining else "All generations cached; model load skipped")
        progress.estimates[3] = 3.0  # Lightweight local reports, replaced by actual duration.
        progress.start(3, len(requests), initial=len(cached), unit="plan")
        batch_index = max((int(p.stem.split("_")[1]) for p in (args.output_dir / "generations").glob("batch_*.jsonl")), default=-1) + 1
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start + args.batch_size]
            rows = save_generation_batch(args, llm, batch, prompts, params, contract_sha, batch_index)
            cached.update({r["request_id"]: r for r in rows})
            batch_index += 1
            progress.update(len(rows))
        progress.complete(f"saved={len(cached)}/{len(requests)}; {args.output_dir / 'generations'}")
        progress.start(4, 1, unit="report", estimate=3.0)
        write_reports(args, cohort, requests, cached, report)
        progress.update(1)
        progress.complete(f"{args.output_dir / 'summary.json'}; qualitative review pending")
    except Exception as exc:
        progress.log(f"[FAILED] stage={progress.index}/{len(names)} item={progress.done}/{progress.total}; {type(exc).__name__}: {exc}. Durable batches: {args.output_dir / 'generations'}. Rerun identical command after resolving error; no completed batch is overwritten.")
        raise
    finally:
        stop.set()
        thread.join(timeout=1)
        progress.close()
    print(f"[complete] Plans only; review files: {args.output_dir / 'review'}", flush=True)


if __name__ == "__main__":
    main()

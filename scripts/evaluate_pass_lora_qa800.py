#!/usr/bin/env python3
"""Matched chain_v5 QA with learned PASS, frozen amateur and frozen final reader.

GPU1 only; cached LOFT500 + medical300 retrieval; no training/new retrieval.
The primary arm uses the validation-selected 0.7 gate; 0.95 is a prespecified
participation-priority sensitivity arm. All reported accuracy uses final answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.generation import pass_chain_prompts as prompts
from medrag.training.graph_input_data import atomic_json, ensure_contract, sha_file
from medrag.training.graph_input_progress import Progress, duration

VERSION = "pass_pced_chain_v5_800_v1"
ARMS = ("no_rag", "all", "pced", "pced_lora", "pass_pced", "pass_pced_safe")
DATASETS = ("hotpotqa", "nq", "musique", "qampari", "quest",
            "bioasq11_factoid", "papersearchqa", "covid_qa")
FINAL_MARKER = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:")
REFUSAL = re.compile(r"\b(?:cannot (?:answer|determine|be answered)|insufficient information|"
                     r"not enough information|does not (?:mention|provide|specify)|"
                     r"no (?:relevant )?information|unable to (?:answer|determine))\b", re.I)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("preflight", "smoke", "run", "summarize"), default="run")
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter-root", type=Path, default=ROOT / "results/pass_lora_v1")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--per-dataset", type=int, default=100)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--gamma", type=float, default=2.5)
    p.add_argument("--pass-weight", type=float, default=1.0)
    p.add_argument("--fallback", choices=("pced", "amateur"), default="pced")
    p.add_argument("--reasoning-tokens", type=int, default=256)
    p.add_argument("--answer-tokens", type=int, default=48)
    p.add_argument("--batch-questions", type=int, default=2)
    p.add_argument("--context-limit", type=int, default=32768)
    p.add_argument("--trace-top-n", type=int, default=8)
    p.add_argument("--min-free-gib", type=float, default=50.)
    p.add_argument("--disk-reserve-gib", type=float, default=30.)
    p.add_argument("--min-host-available-gib", type=float, default=24.)
    p.add_argument("--max-host-rss-gib", type=float, default=48.)
    a = p.parse_args(argv)
    if not 1 <= a.per_dataset <= 100 or not 2 <= a.top_k <= 90:
        p.error("per-dataset=1..100; top-k=2..90")
    if not 1 <= a.batch_questions <= 8 or not 1 <= a.trace_top_n <= 32:
        p.error("batch-questions=1..8; trace-top-n=1..32")
    if not 8 <= a.reasoning_tokens <= 1024 or not 8 <= a.answer_tokens <= 256:
        p.error("Invalid generation cap")
    for name in ("gamma", "pass_weight", "min_free_gib", "disk_reserve_gib",
                 "min_host_available_gib", "max_host_rss_gib"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    a.model, a.adapter_root = a.model.resolve(), a.adapter_root.resolve()
    a.arms = [arm for arm in ARMS if arm in a.arms]
    suffix = "" if a.per_dataset == 100 and a.fallback == "pced" else f"_n{8*a.per_dataset}_{a.fallback}"
    if a.phase == "smoke":
        suffix += "_smoke"
    a.output_root = (a.output_root or ROOT / "results" / (VERSION + suffix)).resolve()
    protected = [a.model, a.adapter_root, ROOT / "datasets", ROOT.parent / "datasets", ROOT / "databases"]
    if a.output_root == ROOT / "results" or any(
            a.output_root == v or v in a.output_root.parents or a.output_root in v.parents for v in protected):
        p.error("Output must not overlap models, training outputs or input datasets")
    return a


def memory_status():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.strip().split()[0]) * 1024
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return info["MemAvailable"] / 2**30, rss / 2**30


def resource_check(a, *, gpu=False, required_gpu=0.):
    available, rss = memory_status()
    if available < a.min_host_available_gib or rss > a.max_host_rss_gib:
        raise RuntimeError(f"Host RAM guard: available={available:.1f} GiB, process RSS={rss:.1f} GiB")
    disk_root = a.output_root
    while not disk_root.exists():
        disk_root = disk_root.parent
    free_disk = shutil.disk_usage(disk_root).free / 2**30
    if free_disk < a.disk_reserve_gib:
        raise RuntimeError(f"Disk guard: {free_disk:.1f} GiB free < reserve {a.disk_reserve_gib}")
    if gpu:
        import torch
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
            raise RuntimeError("GPU1 unavailable; run on the GPU host with CUDA_VISIBLE_DEVICES=1")
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 < required_gpu:
            raise RuntimeError(f"GPU1 guard: {free/2**30:.1f}/{total/2**30:.1f} GiB free; require {required_gpu:.1f}")
    return {"available_host_gib": available, "rss_gib": rss, "free_disk_gib": free_disk}


class StatusProgress(Progress):
    def __init__(self, names, path):
        self.status_path, self.status_last = path, 0.
        super().__init__(names)

    def render(self, force=False):
        super().render(force)
        now = time.monotonic()
        if self.status_path and (force or now - self.status_last >= 5):
            self.status_last = now
            one, two = self.lines()
            atomic_json(self.status_path, {"updated": datetime.datetime.now().isoformat(timespec="seconds"),
                "stage": self.names[self.index], "completed": self.done, "total": self.total,
                "overall": one, "current": two})


def key(row):
    return hashlib.sha256(row["qid"].encode()).hexdigest()


def path_for(a, folder, arm, row, ext=".json"):
    return a.output_root / folder / arm / row["dataset"] / (key(row) + ext)


def check_trace(path, expected_steps, size):
    import numpy as np
    if not path.is_file() or path.stat().st_size != size:
        raise RuntimeError(f"Missing/incomplete trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        if len(data["token_ids"]) != expected_steps:
            raise RuntimeError(f"Trace count mismatch: {path}")


def cached(a, row, arm, digest, *, reasoning=False):
    path = path_for(a, "reasoning" if reasoning else "predictions", arm, row)
    if not path.exists():
        return None
    item = json.loads(path.read_text())
    if any(item.get(k) != v for k, v in {"contract_sha256": digest, "dataset": row["dataset"],
                                       "qid": row["qid"], "arm": arm}.items()):
        raise RuntimeError(f"Incompatible record: {path}")
    check_trace(a.output_root / item["trace"], item["decision_steps"], item["trace_bytes"])
    if not reasoning:
        check_trace(a.output_root / item["answer_trace"], item["answer_decision_steps"], item["answer_trace_bytes"])
    return item


def prepare(a, progress):
    from transformers import AutoTokenizer
    import torch
    from scripts import evaluate_pced_sentence_pass_800 as source
    from scripts import run_loft_pced_paper as paper
    from medrag.generation.pced_trained_pass import PASS_TOKEN

    resource = resource_check(a)
    required = [a.model / x for x in ("config.json", "tokenizer.json", "model.safetensors.index.json")]
    required += [a.adapter_root / x for x in ("manifest.json", "split.json", "judgment_test.json",
        "best/adapter_config.json", "best/adapter_model.safetensors", "best/pass_head.pt")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    adapter = json.loads((a.adapter_root / "best/adapter_config.json").read_text())
    if Path(adapter["base_model_name_or_path"]).resolve() != a.model:
        raise RuntimeError("Adapter/base checkpoint mismatch")
    if set(adapter["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj"} or adapter["modules_to_save"]:
        raise RuntimeError("Expected the frozen ordinary LM head and q/k/v/o PASS adapter")
    trained = json.loads((a.adapter_root / "manifest.json").read_text())["contract"]
    if trained["prompt_version"] != prompts.PROMPT_VERSION:
        raise RuntimeError("Training/inference prompt version mismatch")
    judgment = json.loads((a.adapter_root / "judgment_test.json").read_text())
    thresholds = judgment["thresholds_chosen_on_val"]
    if thresholds["best_balanced"] != .7 or thresholds["safe_step_in"] != .95:
        raise RuntimeError("Expected already selected val thresholds .7/.95; no QA-set threshold tuning")
    head = torch.load(a.adapter_root / "best/pass_head.pt", map_location="cpu", weights_only=True)
    config = json.loads((a.model / "config.json").read_text())
    if head["w"].numel() != config["hidden_size"] or not torch.isfinite(head["w"]).all() or not torch.isfinite(head["b"]):
        raise RuntimeError("Invalid PASS head")
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, padding_side="left")
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if tok.encode(PASS_TOKEN, add_special_tokens=False) != [int(head["pass_id"])]:
        raise RuntimeError("PASS reserved ID/tokenizer mismatch")

    bridge = argparse.Namespace(loft_reference=ROOT / "results/loft_pced_paper_v1_full500_parallel2",
                                medical_reference=ROOT / "results/medical_pced_seeded100_v1")
    all_rows, sources = source.load_rows(bridge)
    rows = []
    for ds in DATASETS:
        rows.extend([r for r in all_rows if r["dataset"] == ds][:a.per_dataset])
    lengths = Counter()
    for row in rows:
        row["texts"] = row["texts"][:a.top_k]
        row["pids"] = row["scores"]["pids"][:a.top_k]
        row["scores"] = {k: v[:a.top_k] if isinstance(v, list) else v for k, v in row["scores"].items()}
        row["log_prior"] = paper.prior_tensor(row["scores"], "paper", "cpu", torch.float32).clamp_min(1e-8).log().tolist()
        encode = lambda p: tok.encode(p, add_special_tokens=False)
        row["tokens"] = {"no_rag": encode(prompts.render_chain(tok, row["question"], [])),
                         "all": encode(prompts.render_chain(tok, row["question"], row["texts"]))}
        row["tokens"]["experts"] = [row["tokens"]["no_rag"]] + [
            encode(prompts.render_chain(tok, row["question"], [text])) for text in row["texts"]]
        for name in ("no_rag", "all"):
            lengths[name] = max(lengths[name], len(row["tokens"][name]))
        lengths["expert"] = max(lengths["expert"], max(map(len, row["tokens"]["experts"])))
    if len(rows) != 8 * a.per_dataset or len({(r['dataset'], r['qid']) for r in rows}) != len(rows):
        raise RuntimeError("Cohort mismatch")
    limit = min(a.context_limit, config["max_position_embeddings"])
    if max(lengths.values()) + a.reasoning_tokens > limit:
        raise RuntimeError(f"Untruncated prompt exceeds context limit {limit}: {dict(lengths)}")
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    kv_per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * 2
    kv_gib = max((a.top_k + 1) * (lengths["expert"] + a.reasoning_tokens),
                 lengths["all"] + a.reasoning_tokens) * a.batch_questions * kv_per_token / 2**30
    # Conservative, uncompressed upper bound; actual NPZs and typical shorter chains use less.
    def token_bytes(streams):
        candidate_bound = a.trace_top_n * (2 * streams - 1) + 4
        return streams * config["hidden_size"] * 2 + candidate_bound * (4 * streams + 4) + streams * a.trace_top_n * 20 + 1024
    disk_bound = len(rows) * sum(a.reasoning_tokens * token_bytes(1 if arm in ("no_rag", "all") else a.top_k + 1)
                               + a.answer_tokens * token_bytes(1) for arm in a.arms) / 2**30
    completed = sum(path_for(a, "predictions", arm, row).is_file() for arm in a.arms for row in rows)
    remaining_fraction = 1 - completed / (len(rows) * len(a.arms))
    if a.phase == "run" and resource["free_disk_gib"] < a.disk_reserve_gib + disk_bound * remaining_fraction:
        raise RuntimeError(f"Disk preflight: require reserve+uncompressed trace bound "
                           f"{a.disk_reserve_gib+disk_bound*remaining_fraction:.1f} GiB; free {resource['free_disk_gib']:.1f}")
    code = [Path(__file__), Path(prompts.__file__), Path(source.__file__), Path(paper.__file__),
            ROOT / "medrag/generation/pced_trained_pass.py", ROOT / "scripts/run_pass_lora_qa800.sh"]
    # Record imported local helpers too; some historical source loaders are transitive imports.
    code += [Path(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)
             and str(m.__file__).startswith(str(ROOT)) and str(m.__file__).endswith(".py")
             and ".venv" not in str(m.__file__)]
    base_weights = sorted(a.model.glob("model-*.safetensors"))
    contract = {"version": VERSION, "n": len(rows), "dataset_order": DATASETS,
        "question_ids": [[r["dataset"], r["qid"]] for r in rows], "retrieval_sources": sources,
        "top_k": a.top_k, "document_order": "cached CE rank; no retrieval/reranking changes",
        "model": str(a.model), "model_config_sha256": sha_file(a.model / "config.json"),
        "tokenizer_sha256": sha_file(a.model / "tokenizer.json"),
        "base_weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in base_weights},
        "adapter_root": str(a.adapter_root), "adapter_sha256": sha_file(a.adapter_root / "best/adapter_model.safetensors"),
        "head_sha256": sha_file(a.adapter_root / "best/pass_head.pt"), "checkpoint_step": int(head["step"]),
        "training_manifest_sha256": sha_file(a.adapter_root / "manifest.json"),
        "threshold_provenance_sha256": sha_file(a.adapter_root / "judgment_test.json"),
        "thresholds": {"pass_pced": .7, "pass_pced_safe": .95}, "arms": a.arms,
        "prompts": {"version": prompts.PROMPT_VERSION, "reason_system": prompts.CHAIN_SYSTEM,
                    "fewshot": prompts.FEWSHOT, "reason_prefix": prompts.CHAIN_PREFIX,
                    "reader_system": prompts.CHAIN_FINAL_SYSTEM, "reader_prefix": prompts.FINAL_PREFIX},
        "decoder": {"gamma": a.gamma, "pass_weight": a.pass_weight,
            "beta": "scalar mean expert/amateur JSD at first token, frozen; existing MCQ-style CoT PCED",
            "normal_score": "(1+beta)*expert_logit-beta*amateur_logit+gamma*log(paper_prior)",
            "pass_probability": "same V+1 separate-head denominator as train/eval_pass_lora; original reserved row in denominator only",
            "gate": "p(pass)>=threshold excludes an expert for this token; reconsider at next token",
            "one_active": "raw greedy token from sole active document expert",
            "multiple_active": "max expert-token PCED score + pass_weight*log(1-p_pass)",
            "all_pass": a.fallback, "amateur": "frozen base checkpoint, never LoRA; not a candidate except amateur fallback",
            "all_pass_pced": "restore all adapted experts without PASS penalty at current prefix",
            "reader": "frozen base, question+reasoning only, no documents, all arms identical",
            "ordinary_pass_mask": int(head["pass_id"]), "tie_break": "token ID then expert ID"},
        "generation": {"dtype": "bfloat16", "attention": "sdpa, no cuDNN SDP", "greedy": True,
            "reason_cap": a.reasoning_tokens, "answer_cap": a.answer_tokens, "seed": 42,
            "stop": "EOS or token cap; no forced completion; any Answer marker preserved in raw trace and excluded from reader reasoning",
            "truncated_reasoning": "reader still runs, incomplete flag always retained", "batch_questions": a.batch_questions,
            "context_limit": limit, "gpu": 1, "compile": False},
        "traces": {"version": 1, "top_n": a.trace_top_n, "hidden": "all streams, every step, BF16 uint16 bit pattern",
            "candidate_logits": "FP32 exact forward values on union of raw/PCED top-N and chosen/counterfactual tokens",
            "replay": "hidden@frozen lm_head reconstructs full vocabulary; GEMM device/dtype can cause small numeric differences",
            "counterfactual": "same-prefix score ablations only, not alternative free-running answers"},
        "metric": "official Subspan EM of first final answer line; empty answer wrong; all 800 remain in denominator",
        "primary": "pass_pced vs pced_lora (PASS effect); pass_pced vs all (research objective)",
        "success_criteria": {"completion": "800/800 in each arm, no omitted or silently repaired outputs",
            "worthwhile_qa_gain": "at least +2 percentage points vs pced_lora and vs all; report paired interval even if it includes zero",
            "interpretation": "PASS selection improvements alone do not establish QA improvement; no tuning on this cohort",
            "stop": "do not scale/tune from this cohort if PASS harms QA or its benefit is indistinguishable from the LoRA-only control"},
        "evaluation_status": "exploratory previously inspected LOFT TEST500 + medical train-origin300; no QA threshold tuning",
        "code_sha256": {str(f.relative_to(ROOT)): sha_file(f) for f in sorted(set(code))},
        "environment": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "safetensors", "numpy")},
    }
    manifest = a.output_root / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        # Use the shared contract helper's canonical serialization when checking/writing below.
        if old["contract"] != json.loads(json.dumps(contract)):
            changed = sorted(k for k in set(old["contract"]) | set(contract) if old["contract"].get(k) != json.loads(json.dumps(contract)).get(k))
            raise RuntimeError(f"Contract differs at {changed}; use a new output directory")
    extra = {"prompt_lengths": dict(lengths), "estimated_kv_gib": kv_gib,
             "trace_upper_bound_gib": disk_bound, "resource": resource,
             "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    return rows, tok, head, contract, extra


def clean_reasoning(raw):
    return FINAL_MARKER.split(raw, maxsplit=1)[0].strip()


def clean_answer(raw):
    first = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    return FINAL_MARKER.sub("", first, count=1).strip()


def reasoning_stats(gen):
    counts = Counter()
    winners, previous_active = [], None
    for frame in gen.frames:
        if "active" not in frame:
            continue
        n = int(frame["active"].sum())
        if previous_active is not None:
            counts.update(expert_reentries=int((~previous_active & frame["active"]).sum()),
                          expert_exits=int((previous_active & ~frame["active"]).sum()))
        previous_active = frame["active"]
        counts.update(competition_steps=1, active_total=n, all_pass=int(n == 0), single_active=int(n == 1),
                      multiple_active=int(n > 1), token_changed=int(frame["token_ids"] != frame["counter_token"]),
                      winner_changed=int(frame["winner"] != frame["counter_winner"]),
                      amateur_selected=int(frame["winner"] == 0))
        winners.append(int(frame["winner"]))
    counts["unique_document_winners"] = len(set(winners) - {0})
    counts["winner_switches"] = sum(x != y for x, y in zip(winners, winners[1:]))
    return dict(counts)


def batch_rows(rows, size):
    for ds in DATASETS:
        ordered = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: len(r["tokens"]["all"]))
        for start in range(0, len(ordered), size):
            yield ordered[start:start + size]


def process_batch(a, rows, arm, engine, tok, digest, progress, gpu_working):
    from medrag.generation.pced_trained_pass import save_trace
    from scripts import run_loft_pced_paper as paper
    resource_check(a, gpu=True, required_gpu=gpu_working)
    began = time.monotonic()
    reasons = [cached(a, r, arm, digest, reasoning=True) for r in rows]
    missing = [r for r, old in zip(rows, reasons) if old is None]
    if missing:
        beat = lambda step, done: progress.render()
        if arm in ("no_rag", "all"):
            generated = engine.plain([r["tokens"][arm] for r in missing], a.reasoning_tokens,
                                     top_n=a.trace_top_n, heartbeat=beat)
        else:
            generated = engine.competitive([r["tokens"]["experts"] for r in missing],
                [r["log_prior"] for r in missing], arm, a.reasoning_tokens,
                threshold=.95 if arm == "pass_pced_safe" else .7, gamma=a.gamma, penalty=a.pass_weight,
                fallback=a.fallback, top_n=a.trace_top_n, heartbeat=beat)
        for row, gen in zip(missing, generated):
            trace = path_for(a, "traces", arm, row, ".npz")
            save_trace(trace, gen.frames, beta=gen.beta, log_prior=row["log_prior"] if arm not in ("no_rag", "all") else [])
            raw = tok.decode(gen.ids, skip_special_tokens=True)
            rec = {"contract_sha256": digest, "dataset": row["dataset"], "qid": row["qid"], "arm": arm,
                "reasoning_raw": raw, "reasoning": clean_reasoning(raw), "reasoning_token_ids": gen.ids,
                "reasoning_finish": gen.finish, "reasoning_incomplete": gen.finish == "token_budget",
                "reasoning_contains_answer_marker": bool(FINAL_MARKER.search(raw)),
                "reasoning_refusal_pattern": bool(REFUSAL.search(raw)), "decision_steps": len(gen.frames),
                "beta": gen.beta, "stats": reasoning_stats(gen), "trace": str(trace.relative_to(a.output_root)),
                "trace_bytes": trace.stat().st_size}
            atomic_json(path_for(a, "reasoning", arm, row), rec)
        del generated
        reasons = [cached(a, r, arm, digest, reasoning=True) for r in rows]
    final_inputs = [tok.encode(prompts.render_chain_final(tok, r["question"], rec["reasoning"]),
                               add_special_tokens=False) for r, rec in zip(rows, reasons)]
    if max(map(len, final_inputs)) + a.answer_tokens > a.context_limit:
        raise RuntimeError("Reader context limit exceeded; reasoning checkpoints preserved")
    answers = engine.plain(final_inputs, a.answer_tokens, top_n=a.trace_top_n)
    for row, rec, gen in zip(rows, reasons, answers):
        trace = path_for(a, "answer_traces", arm, row, ".npz")
        save_trace(trace, gen.frames, beta=0., log_prior=[])
        raw = tok.decode(gen.ids, skip_special_tokens=True)
        answer = clean_answer(raw)
        result = dict(rec, final_raw=raw, final_answer=answer, answer_token_ids=gen.ids, answer_finish=gen.finish,
            answer_decision_steps=len(gen.frames), answer_trace=str(trace.relative_to(a.output_root)),
            answer_trace_bytes=trace.stat().st_size, correct=int(bool(answer) and paper.subspan_em(answer, row["answers"])),
            empty_answer=not bool(answer), answers=row["answers"], document_pids=[] if arm == "no_rag" else row["pids"],
            batch_allocated_seconds=(time.monotonic() - began)/len(rows), batch_questions=len(rows))
        atomic_json(path_for(a, "predictions", arm, row), result)
    return time.monotonic() - began


def summarize(a, rows, digest):
    import numpy as np
    from scripts import run_loft_pced_paper as paper
    results, compact, summaries = {}, [], {}
    for arm in a.arms:
        items = [cached(a, row, arm, digest) for row in rows]
        if any(x is None for x in items):
            raise RuntimeError(f"Cannot summarize incomplete arm {arm}")
        for item, row in zip(items, rows):
            if item["correct"] != int(bool(item["final_answer"]) and paper.subspan_em(item["final_answer"], row["answers"])):
                raise RuntimeError("Stored metric differs from frozen evaluator")
        results[arm] = items
        by = {}
        for group in (*DATASETS, "LOFT", "MEDICAL", "ALL"):
            subset = [x for x in items if group == "ALL" or x["dataset"] == group or
                      (group == "LOFT" and x["dataset"] in DATASETS[:5]) or
                      (group == "MEDICAL" and x["dataset"] in DATASETS[5:])]
            count, good = len(subset), sum(x["correct"] for x in subset)
            by[group] = {"n": count, "correct": good, "accuracy": good/count, "missing": 0}
        totals = Counter()
        for item in items:
            totals.update(item["stats"])
            compact.append({k: item[k] for k in ("dataset", "qid", "arm", "correct", "final_answer",
                                                 "reasoning_finish", "reasoning_refusal_pattern")})
        summaries[arm] = {"groups": by, "stats": dict(totals),
            "mean_reasoning_tokens": sum(len(x["reasoning_token_ids"]) for x in items)/len(items),
            "reasoning_incomplete": sum(x["reasoning_incomplete"] for x in items),
            "reasoning_refusal_patterns": sum(x["reasoning_refusal_pattern"] for x in items),
            "empty_answers": sum(x["empty_answer"] for x in items)}
    pairs = {}
    for left, right in (("pced_lora", "pass_pced"), ("all", "pass_pced"), ("pced", "pced_lora"),
                        ("pced", "pass_pced"), ("pass_pced", "pass_pced_safe"), ("no_rag", "all")):
        if left not in results or right not in results:
            continue
        per = {}
        for group in (*DATASETS, "LOFT", "MEDICAL", "ALL"):
            delta = [b["correct"]-c["correct"] for c, b in zip(results[left], results[right])
                     if group == "ALL" or c["dataset"] == group or
                     (group == "LOFT" and c["dataset"] in DATASETS[:5]) or
                     (group == "MEDICAL" and c["dataset"] in DATASETS[5:])]
            n = len(delta)
            losses, gains = delta.count(-1), delta.count(1)
            draws = np.random.default_rng(42).multinomial(n, [losses/n, gains/n, 1-(losses+gains)/n], size=5000)
            ci = np.quantile(100*(draws[:, 1]-draws[:, 0])/n, [.025, .975]).tolist()
            per[group] = {"n": n, "correct_to_wrong": losses, "wrong_to_correct": gains,
                          "delta_pp": 100*sum(delta)/n, "paired_bootstrap95_pp": ci}
        pairs[f"{right}_minus_{left}"] = per
    report = {"contract_sha256": digest, "n": len(rows), "conditions": summaries, "paired": pairs,
        "notes": ["All reported accuracy uses final answer only; incomplete reasoning remains in denominator.",
                  "Prior/contrast/gate counterfactual traces share the observed prefix; not free-running interventions.",
                  "Previously inspected QA cohorts; exploratory, not a new untouched test."]}
    atomic_json(a.output_root / "summary.json", report)
    with (a.output_root / "per_question.jsonl.tmp").open("w") as fh:
        for item in compact:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(a.output_root / "per_question.jsonl.tmp", a.output_root / "per_question.jsonl")
    lines = ["dataset,arm,n,correct,accuracy"]
    lines += [f"{ds},{arm},{m['n']},{m['correct']},{m['accuracy']:.6f}" for arm, s in summaries.items()
              for ds, m in s["groups"].items()]
    (a.output_root / "summary.csv.tmp").write_text("\n".join(lines) + "\n")
    os.replace(a.output_root / "summary.csv.tmp", a.output_root / "summary.csv")
    return report


def load_engine(a, tok, head, extra):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from medrag.generation.pced_trained_pass import Engine
    resource_check(a, gpu=True, required_gpu=max(a.min_free_gib, 20 + extra["estimated_kv_gib"] + 16))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    base = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, a.adapter_root / "best", is_trainable=False).eval()
    model.requires_grad_(False)
    return Engine(model, tok, head)


def smoke(a, rows, tok, engine, digest, progress, extra):
    """Bounded real-checkpoint validation; does not start the 800-question job."""
    import torch
    import numpy as np
    from medrag.generation.pced_trained_pass import save_trace, pass_probabilities
    # Real prompts from distinct datasets, all K expert streams; four decode steps.
    sample = [rows[0], next(r for r in rows if r["dataset"] in DATASETS[5:])]
    groups, prior = [r["tokens"]["experts"] for r in sample], [r["log_prior"] for r in sample]
    began = time.monotonic()
    adapted = engine.competitive(groups, prior, "pass_pced", 4, gamma=a.gamma,
        threshold=.7, penalty=a.pass_weight, fallback=a.fallback, top_n=a.trace_top_n)
    timings = {"pass_pced": time.monotonic() - began}
    began = time.monotonic()
    control = engine.competitive(groups, prior, "pced_lora", 4, gamma=a.gamma, top_n=a.trace_top_n)
    timings["pced_lora"] = time.monotonic() - began
    began = time.monotonic()
    plain = engine.plain([r["tokens"]["all"] for r in sample], 4, top_n=a.trace_top_n)
    timings["all"] = time.monotonic() - began
    began = time.monotonic()
    base = engine.competitive(groups, prior, "pced", 4, gamma=a.gamma, top_n=a.trace_top_n)
    timings["pced"] = time.monotonic() - began
    solo = engine.competitive(groups[:1], prior[:1], "pass_pced", 4, gamma=a.gamma,
        threshold=.7, penalty=a.pass_weight, fallback=a.fallback, top_n=a.trace_top_n)
    if solo[0].ids != adapted[0].ids:
        raise RuntimeError("Batch/single greedy mismatch in real-checkpoint smoke; use batch-questions=1 in a new output")
    rec = {"contract_sha256": digest, "questions": [[r['dataset'], r['qid']] for r in sample],
           "batch_single_token_parity": True, "steps_per_arm": 4, "arms_checked": ["all", "pced", "pced_lora", "pass_pced"],
           "outputs": {}, "two_question_four_step_seconds": timings,
           "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30}
    for name, gens in (("all", plain), ("pced", base), ("pced_lora", control), ("pass_pced", adapted)):
        rec["outputs"][name] = [g.ids for g in gens]
        for i, gen in enumerate(gens):
            p = a.output_root / "smoke_traces" / f"{name}_{i}.npz"
            save_trace(p, gen.frames, beta=gen.beta, log_prior=prior[i] if name != "all" else [])
            with np.load(p) as saved:
                assert len(saved["token_ids"]) == len(gen.frames)
    # A direct no-cache prefix recomputation verifies teacher-forcing / cache alignment.
    from medrag.generation.pced_trained_pass import StreamBatch
    with torch.inference_mode():
        prefix = adapted[0].ids[:2]
        direct = StreamBatch(engine, [p + prefix for p in groups[0][1:]], True)
        zp, lp, ln = pass_probabilities(direct.hidden, direct.logits, engine.weight, engine.bias)
        frame = adapted[0].frames[2]
        error = float(np.max(np.abs(lp.exp().cpu().numpy() - np.exp(frame["pass_log_probability"]))))
        if error > .03:
            raise RuntimeError(f"Cache/teacher-force PASS mismatch {error:.4f}")
        rec["pass_cache_vs_full_max_probability_error"] = error
    # Frozen reader path too.
    reader = [tok.encode(prompts.render_chain_final(tok, r["question"], tok.decode(g.ids)), add_special_tokens=False)
              for r, g in zip(sample, adapted)]
    rec["reader_tokens"] = [g.ids for g in engine.plain(reader, 4)]
    atomic_json(a.output_root / "smoke_report.json", rec)
    progress.log(f"[smoke OK] real K={a.top_k} base/adapted/reader paths, PASS cache alignment, batch/single tokens, NPZ traces")
    return rec


def run(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
        raise RuntimeError("Physical GPU1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    phases = ["CPU contract", "load Llama+LoRA", "calibrate arms"] + [f"{x} QA" for x in a.arms] + ["paired summary"]
    progress = StatusProgress(phases, None if a.phase == "preflight" else a.output_root / "status.json")
    active, completed = "preflight", 0
    lock = None
    try:
        progress.stage(0, 1)
        rows, tok, head, contract, extra = prepare(a, progress)
        if a.phase == "preflight":
            progress.complete(f"N={len(rows)}, arms={a.arms}, checkpoint step={int(head['step'])}, prompt lengths={extra['prompt_lengths']}")
            progress.log(f"[preflight OK] KV estimate={extra['estimated_kv_gib']:.1f} GiB; uncompressed full-cap trace upper bound="
                         f"{extra['trace_upper_bound_gib']:.1f} GiB; free disk={extra['resource']['free_disk_gib']:.1f} GiB; no GPU load")
            return
        a.output_root.mkdir(parents=True, exist_ok=True)
        lock = (a.output_root / ".run.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this output directory")
        digest = ensure_contract(a.output_root / "manifest.json", contract)
        atomic_json(a.output_root / "runtime_preflight.json", extra)
        for row in rows:
            target = a.output_root / "inputs" / row["dataset"] / (key(row) + ".json")
            if not target.exists():
                atomic_json(target, {k: v for k, v in row.items() if k != "tokens"})
        records = {arm: [cached(a, row, arm, digest) for row in rows] for arm in a.arms}
        completed = sum(x is not None for v in records.values() for x in v)
        total = len(rows) * len(a.arms)
        progress.complete(f"N={len(rows)}, K={a.top_k}, cached={completed}/{total}; {a.output_root}")
        progress.log(f"[plan] contract -> one base+LoRA load -> first {a.batch_questions} fresh QA/arm calibration -> "
                     f"{', '.join(a.arms)} -> paired report; GPU1 only; remaining={total-completed}; ETA calibrating")
        progress.log("[trace] exact FP32 candidate logits from every expert + BF16 hidden states for full-vocabulary replay; "
                     "all streams consume shared tokens, including abstainers")
        if a.phase == "summarize":
            summarize(a, rows, digest)
            progress.log(f"[complete] summary: {a.output_root / 'summary.json'}")
            return
        progress.stage(1, 1)
        engine = load_engine(a, tok, head, extra) if completed < total or a.phase == "smoke" else None
        progress.complete("Llama-3.1-8B, frozen base/amateur/reader; LoRA only document experts; no compile")
        if a.phase == "smoke":
            progress.stage(2, 1)
            smoke(a, rows, tok, engine, digest, progress, extra)
            progress.complete("bounded real-model smoke complete")
            return
        gpu_working = max(12., extra["estimated_kv_gib"] + 8.)
        progress.stage(2, len(a.arms), unit="arm")
        for ai, arm in enumerate(a.arms):
            pending = [r for r, saved in zip(rows, records[arm]) if saved is None]
            if pending:
                ordered = sorted(pending, key=lambda r: len(r["tokens"]["all"]))
                middle = max(0, len(ordered)//2 - a.batch_questions//2)
                batch = ordered[middle:middle+a.batch_questions]
                active = f"calibration/{arm}/{batch[0]['dataset']}:{batch[0]['qid']}"
                elapsed = process_batch(a, batch, arm, engine, tok, digest, progress, gpu_working)
                progress.weights[3+ai] = elapsed / len(batch) * len(rows)
                completed += len(batch)
                progress.log(f"[calibration] {arm}: {elapsed/len(batch):.2f}s/QA including traces; "
                             f"remaining estimate {duration(elapsed/len(batch)*(len(pending)-len(batch)))}")
            else:
                progress.weights[3+ai] = .001
            progress.update()
        progress.weights[-1] = 5.
        progress.complete("one observed rate per arm; estimates update during execution")
        for ai, arm in enumerate(a.arms):
            pending = [r for r in rows if cached(a, r, arm, digest) is None]
            progress.stage(3+ai, len(rows), initial=len(rows)-len(pending), unit="q", kind="generation")
            for batch in batch_rows(pending, a.batch_questions):
                active = f"{arm}/{batch[0]['dataset']}:{batch[0]['qid']}"
                progress.names[3+ai] = f"{arm}/{batch[0]['dataset']}"
                process_batch(a, batch, arm, engine, tok, digest, progress, gpu_working)
                completed += len(batch)
                progress.update(len(batch))
            progress.complete(f"{arm}: {len(rows)} completed; {a.output_root / 'predictions' / arm}")
        progress.stage(len(phases)-1, 1)
        report = summarize(a, rows, digest)
        progress.complete(f"saved {a.output_root / 'summary.json'}")
        progress.log("[complete] " + "; ".join(f"{arm} {report['conditions'][arm]['groups']['ALL']['correct']}/{len(rows)}"
                                              for arm in a.arms))
    except BaseException as error:
        progress.log(f"[FAILED] item={active}; completed={completed}; durable={a.output_root}; "
                     f"{type(error).__name__}: {error}. Same command resumes compatible completed question records.")
        raise
    finally:
        progress.close()
        if lock:
            lock.close()


if __name__ == "__main__":
    run(arguments())

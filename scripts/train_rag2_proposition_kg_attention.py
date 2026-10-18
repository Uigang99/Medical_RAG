#!/usr/bin/env python3
"""Official Train9k / Val1k proposition-pair attention, pristine frozen Llama.

Training only: no test examples, old LoRA, rationale supervision or retrieval.
Use --preflight-only for metadata checks, --prepare-only for CPU token packing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from torch.nn import functional as F

from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, jsonl, sha_file
from medrag.training.graph_input_progress import Progress
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.proposition_kg_attention_data import inspect_inputs, build_rows, shuffle_pairs
from medrag.generation.proposition_kg_attention import PropositionPairController, PropositionAttentionModel

VERSION = "official_train_proposition_attention_v1"
ARMS = ("graph", "importance", "shuffled")
CODE = ("scripts/train_rag2_proposition_kg_attention.py",
        "medrag/training/proposition_kg_attention_data.py",
        "medrag/generation/proposition_kg_attention.py",
        "medrag/training/graph_input_progress.py",
        "medrag/training/graph_input_data.py",
        "medrag/training/graph_input_modules.py")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/kg/rag2_official_train10000_top8_h200_throughput_v2")
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_official_train_proposition_attention_v1")
    p.add_argument("--reuse-prepared-from", type=Path,
                   help="Read a compatible prior output root's prepared cache without modifying it")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--bias-penalty", type=float, default=1e-3)
    p.add_argument("--effective-batch-size", type=int, default=16)
    p.add_argument("--max-micro-batch-size", type=int, default=16)
    p.add_argument("--initial-micro-batch-size", type=int, default=2)
    p.add_argument("--validation-batch-size", type=int, default=32)
    p.add_argument("--memory-target", type=float, default=.88)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-prompt-tokens", type=int, default=8064)
    p.add_argument("--checkpoint-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args(argv)
    if min(a.epochs, a.effective_batch_size, a.max_micro_batch_size,
           a.initial_micro_batch_size, a.validation_batch_size, a.hidden,
           a.query_chunk, a.checkpoint_steps, a.cpu_threads) < 1:
        p.error("Batch, epoch, dimension and checkpoint settings must be positive")
    if not 0 < a.memory_target < .98 or a.max_prompt_tokens > 8064 or a.max_prompt_tokens < 64:
        p.error("Invalid memory target or prompt limit (maximum8064)")
    if a.learning_rate <= 0 or min(a.weight_decay, a.bias_penalty) < 0:
        p.error("Invalid optimization settings")
    if len(a.arms) != len(set(a.arms)):
        p.error("Duplicate arms")
    if sum((a.preflight_only, a.prepare_only)) > 1:
        p.error("Choose only one preparation mode")
    return a


def code_identity():
    return {name: sha_file(ROOT / name) for name in CODE}


def contract_for(a, inputs):
    import transformers
    # Runtime disk availability is not data identity: writing a checkpoint must
    # never invalidate its own resume manifest.
    stable_inputs = {k: v for k, v in inputs.items() if k != "source_disk_free_bytes"}
    return {"version": VERSION, "inputs": stable_inputs, "code": code_identity(),
            "environment": {"torch": torch.__version__, "transformers": transformers.__version__},
            "model": str(a.model.resolve()), "lora": None,
            "scientific_scope": "Official Train only; no TEST model fitting or selection. No final evaluation in this command.",
            "hypothesis": "Question-conditioned proposition-pair coupling improves held-out MCQ accuracy over identical TEXT and source-only importance.",
            "registered_arms": list(ARMS),
            "features": "Frozen Llama layer12 input representations: proposition/stem mean+last only; options/labels never pooled; no learned global IDs.",
            "positions": "All token pairs in KG-adjacent proposition blocks; original causal mask; layers13-16 only; shared across heads; no direct question/answer query rows.",
            "known_input_conditioning": "Full prompt prefill; known later stem may condition earlier evidence; never future answers.",
            "evidence": "All usable Top8 propositions in source order; trailing whole-proposition budget truncation shared across arms; graphless questions retained.",
            "controller": {"hidden": a.hidden, "rounds": 2, "cut_layer": 12, "layers_zero_based": [12, 13, 14, 15], "max_bias": 2., "initial_output": "zero"},
            "loss": "Per-question CE over valid answer-label logits at forced Answer: prefix + lambda*per-question mean(bias^2); prompt contains no gold token.",
            "selection": "Maximum validation Macro2, then minimum answer CE, then earlier epoch; includes zero-bias checkpoint.",
            "training": {key: getattr(a, key) for key in ("epochs", "learning_rate", "weight_decay", "bias_penalty", "effective_batch_size", "max_micro_batch_size", "initial_micro_batch_size", "validation_batch_size", "memory_target", "query_chunk", "max_prompt_tokens", "seed")},
            "optimizer": "AdamW,5%warmup/cosine,clip1,BF16 frozen Llama/FP32 controller; no dropout",
            "user_override": "Skip tiny-set overfit and small held-out performance pilots; retain runtime safety checks; run full Train9000/Val1000 directly.",
            "shuffle": "Deterministic per-question causal degree-preserving edge swaps where possible; retain counts/roles, report unchanged graph fraction.",
            "runtime": "Length-bucket adaptive microbatch, effective batch invariant; OOM rolls back complete optimizer window; two-line wall-time-weighted progress."}


def ensure_manifest(path, contract, *, create=False):
    value = {"contract": contract, "contract_sha256": digest(contract)}
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get("contract") != contract or prior.get("contract_sha256") != value["contract_sha256"]:
            old = prior.get("contract", {})
            changed = sorted(k for k in old.keys() | contract.keys() if old.get(k) != contract.get(k))
            raise RuntimeError(f"Resume contract mismatch at {changed}; keep prior outputs, use a new --output-root")
    elif create:
        def git_value(*args):
            try:
                return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.SubprocessError):
                return None
        # Commit/command are provenance, not resume identity: unrelated commits
        # must not invalidate byte-identical training code and data.
        value["provenance"] = {"git_commit": git_value("rev-parse", "HEAD"),
                               "git_branch": git_value("branch", "--show-current"),
                               "argv": sys.argv, "output_root": str(path.parent.resolve()),
                               "requested_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        atomic_json(path, value)
    return value["contract_sha256"]


def prepared_cache_source(a, contract):
    """Validate cached rows before tokenizer/model loading; never rewrite donors."""
    root, expected_hash = a.output_root, digest(contract)
    complete = root / "prepared/COMPLETE.json"
    if not complete.exists():
        if a.reuse_prepared_from is None:
            return None
        root = a.reuse_prepared_from
        prior = json.loads((root / "manifest.json").read_text())
        old = prior.get("contract")
        if not isinstance(old, dict) or prior.get("contract_sha256") != digest(old):
            raise RuntimeError("Prepared donor manifest digest mismatch; preserve old outputs")
        changed = sorted(k for k in old.keys() | contract.keys()
                         if k != "code" and (k not in old or k not in contract or old[k] != contract[k]))
        allowed = {"scripts/train_rag2_proposition_kg_attention.py",
                   "medrag/generation/proposition_kg_attention.py"}
        old_code, new_code = old.get("code", {}), contract["code"]
        if not isinstance(old_code, dict) or old_code.keys() != new_code.keys():
            changed.append("code.files")
        else:
            changed.extend("code." + name for name in old_code
                           if old_code[name] != new_code[name] and name not in allowed)
        if changed:
            raise RuntimeError(f"Prepared donor contract mismatch at {changed}; no cache reused")
        expected_hash = prior["contract_sha256"]
        complete = root / "prepared/COMPLETE.json"
    meta = json.loads(complete.read_text())
    rows_path = root / "prepared/rows.jsonl"
    if meta.get("contract_sha256") != expected_hash or sha_file(rows_path) != meta.get("rows_sha256"):
        raise RuntimeError("Prepared data cache mismatch; preserve old outputs")
    return rows_path


def save_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def controller_state(ctrl):
    return {k: v.detach().cpu().clone() for k, v in ctrl.state_dict().items()}


def state_identity(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); torch.set_rng_state(state["torch"])
    if state["cuda"]: torch.cuda.set_rng_state_all(state["cuda"])


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(error).lower()


def release_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()


class AdaptiveBatch:
    """Microbatch caps by padded length, with periodic recovery after OOM."""
    def __init__(self, maximum, initial, target=.88, state=None):
        self.maximum, self.initial, self.target = maximum, min(initial, maximum), target
        self.buckets = state or {}

    def key(self, rows):
        n = max(len(r["input_ids"]) for r in rows)
        return str(next((x for x in (1024, 2048, 3072, 4096, 6144, 8064) if n <= x), n))

    def choose(self, rows):
        key = self.key(rows)
        b = self.buckets.setdefault(key, {"cap": self.initial, "seen": 0, "oom": 0})
        cap = b["cap"]
        if b["seen"] and b["seen"] % 24 == 0 and b.get("cooldown", 0) == 0:
            cap = min(self.maximum, cap * 2)
        return min(cap, len(rows), self.maximum)

    def failed(self, rows, micro):
        b = self.buckets.setdefault(self.key(rows), {"cap": self.initial, "seen": 0, "oom": 0})
        b["cap"] = max(1, micro // 2); b["oom"] += 1; b["cooldown"] = 12

    def succeeded(self, rows, micro, peak, capacity):
        b = self.buckets.setdefault(self.key(rows), {"cap": self.initial, "seen": 0, "oom": 0})
        b["seen"] += 1; b["cap"] = micro
        cooling = b.get("cooldown", 0) > 0
        b["cooldown"] = max(0, b.get("cooldown", 0)-1)
        if capacity and peak < capacity * self.target * .65 and not cooling:
            b["cap"] = min(self.maximum, micro * 2)
        elif capacity and peak > capacity * self.target:
            b["cap"] = max(1, micro // 2)
        b["peak_bytes"] = int(peak)


def gpu_capacity():
    if not torch.cuda.is_available(): return 0
    free, total = torch.cuda.mem_get_info()
    outside = max(0, total - free - torch.cuda.memory_reserved())
    return max(1, total - outside)


def epoch_order(rows, epoch, seed, batch_size):
    indices = list(range(len(rows)))
    rng = random.Random(f"{seed}:{epoch}"); rng.shuffle(indices)
    windows = []
    for start in range(0, len(indices), batch_size * 32):
        pool = sorted(indices[start:start + batch_size * 32], key=lambda i: len(rows[i]["input_ids"]))
        windows.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
    full = [w for w in windows if len(w) == batch_size]
    tail = [w for w in windows if len(w) != batch_size]
    rng.shuffle(full)
    return [i for window in full + tail for i in window]


def choice_losses(wrapper, rows, pad_id):
    output = wrapper(rows, pad_id=pad_id)
    vocabulary = sorted({t for r in rows for t in r["choice_token_ids"]})
    index = {token: i for i, token in enumerate(vocabulary)}
    ids = torch.tensor(vocabulary, dtype=torch.long, device=output.last_hidden.device)
    logits = F.linear(output.last_hidden, wrapper.llm.lm_head.weight.index_select(0, ids)).float()
    width = max(len(r["choice_token_ids"]) for r in rows)
    indices = [[index[t] for t in r["choice_token_ids"]] + [0] * (width - len(r["choice_token_ids"])) for r in rows]
    selected = logits.gather(1, torch.tensor(indices, device=logits.device))
    valid = torch.arange(width, device=logits.device)[None] < torch.tensor([len(r["choice_token_ids"]) for r in rows], device=logits.device)[:, None]
    selected = selected.masked_fill(~valid, -torch.inf)
    target = torch.tensor([r["gold_index"] for r in rows], device=logits.device)
    ce = F.cross_entropy(selected, target, reduction="none")
    return ce, output.bias_penalty, selected, output


def backward_micro(wrapper, rows, pad_id, penalty, denominator):
    """On OOM, all activation references unwind before the caller retries."""
    ce, reg, logits, output = choice_losses(wrapper, rows, pad_id)
    loss = (ce + penalty*reg).sum() / denominator
    if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss; no optimizer update performed")
    loss.backward()
    return float(ce.detach().sum()), float(reg.detach().sum())


def accumulate_window(wrapper, rows, pad_id, micro, penalty):
    total_ce = total_penalty = 0.
    with training_attention_backend():
        for start in range(0, len(rows), micro):
            answer, regularizer = backward_micro(wrapper, rows[start:start+micro], pad_id, penalty, len(rows))
            total_ce += answer; total_penalty += regularizer
    norm = torch.nn.utils.clip_grad_norm_(wrapper.controller.parameters(), 1., error_if_nonfinite=True)
    synchronize()
    return total_ce, total_penalty, float(norm)


def train_window(wrapper, optimizer, rows, pad_id, adaptive, penalty, progress=None):
    """Accumulate exactly these examples; failed attempts cannot leave gradients."""
    initial_rng = rng_state()
    micro = adaptive.choose(rows)
    started = time.monotonic()
    while True:
        optimizer.zero_grad(set_to_none=True); restore_rng(initial_rng)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        capacity = gpu_capacity()
        try:
            total_ce, total_penalty, norm = accumulate_window(wrapper, rows, pad_id, micro, penalty)
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            adaptive.succeeded(rows, micro, peak, capacity)
            break
        except RuntimeError as error:
            optimizer.zero_grad(set_to_none=True)
            if not is_oom(error) or micro == 1: raise
            adaptive.failed(rows, micro)
            micro = max(1, micro // 2)
            if progress: progress.log(f"[OOM] retry identical {len(rows)}-question optimizer window at micro={micro}; other length buckets retain their caps")
        release_memory()
    # An optimizer failure may already have changed parameters. Never retry it
    # in-memory: abort and resume the last complete durable optimizer checkpoint.
    optimizer.step()
    synchronize()
    return {"answer_ce_sum": total_ce, "bias_penalty_sum": total_penalty,
            "examples": len(rows), "micro_batch": micro, "peak_gpu_bytes": int(peak),
            "gradient_norm": norm, "seconds": time.monotonic() - started}


def evaluation_batch(wrapper, rows, pad_id):
    with torch.no_grad(), training_attention_backend():
        ce, _, logits, output = choice_losses(wrapper, rows, pad_id)
    result = []
    for row, loss, score in zip(rows, ce.tolist(), logits.tolist()):
        score = score[:len(row["option_labels"])]
        pred = max(range(len(score)), key=score.__getitem__)
        result.append({"sample_id": row["sample_id"], "dataset": row["dataset"],
                       "gold": row["option_labels"][row["gold_index"]],
                       "prediction": row["option_labels"][pred], "correct": bool(pred == row["gold_index"]),
                       "choice_logits": score, "answer_ce": loss})
    return result


def summarize_predictions(predictions):
    counts = Counter(r["dataset"] for r in predictions)
    correct = Counter(r["dataset"] for r in predictions if r["correct"])
    accuracy = {name: correct[name] / n for name, n in counts.items()}
    if set(accuracy) != {"medmcqa", "medqa"}: raise RuntimeError("Validation must contain both official train datasets")
    return {"questions": len(predictions), "counts": dict(counts), "correct_counts": dict(correct),
            "accuracy": accuracy, "macro2": sum(accuracy.values()) / 2,
            "micro": sum(correct.values()) / len(predictions),
            "answer_ce": sum(r["answer_ce"] for r in predictions) / len(predictions),
            "missing": 0, "parsing_failures": 0}


def evaluation(wrapper, rows, pad_id, batch_size, progress, stage, path, run_hash, model_hash):
    meta_path = path.with_suffix(".meta.json")
    identity = {"contract_sha256": run_hash, "controller_sha256": model_hash,
                "sample_ids_sha256": digest([r["sample_id"] for r in rows])}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity or sha_file(path) != meta["predictions_sha256"]:
            raise RuntimeError(f"Validation cache mismatch: {path}")
        progress.stage(stage, len(rows), initial=len(rows), unit="q", kind="validation")
        progress.complete(f"cached {path}")
        return meta["summary"]
    progress.stage(stage, len(rows), unit="q", kind="validation")
    wrapper.controller.eval()
    ordered = sorted(rows, key=lambda r: len(r["input_ids"]))
    predictions, position = [], 0
    adaptive = AdaptiveBatch(batch_size, min(4, batch_size))
    started = time.monotonic()
    while position < len(ordered):
        prospective = ordered[position:position + batch_size]
        micro = adaptive.choose(prospective)
        while True:
            batch = ordered[position:position + micro]
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            capacity = gpu_capacity()
            try:
                predictions.extend(evaluation_batch(wrapper, batch, pad_id))
                peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                adaptive.succeeded(prospective, micro, peak, capacity)
                break
            except RuntimeError as error:
                if not is_oom(error) or micro == 1: raise
                adaptive.failed(prospective, micro); micro = max(1, micro // 2)
            release_memory()
        position += len(batch); progress.update(len(batch))
    predictions.sort(key=lambda r: r["sample_id"])
    summary = summarize_predictions(predictions)
    summary["elapsed_seconds"] = time.monotonic() - started
    atomic_jsonl(path, predictions)
    atomic_json(meta_path, {"identity": identity, "summary": summary, "predictions_sha256": sha_file(path)})
    progress.complete(f"Macro2={summary['macro2']*100:.2f}% CE={summary['answer_ce']:.5f}; {path}")
    return summary


def is_better(summary, best):
    return (summary["macro2"], -summary["answer_ce"]) > (best["macro2"], -best["answer_ce"])


def make_wrapper(a, llm, arm):
    set_seed(a.seed)
    controller = PropositionPairController(llm.config.hidden_size, hidden=a.hidden,
                                         rounds=2, variant=arm, layers=4, max_bias=2.)
    controller.to(device=llm.device, dtype=torch.float32)
    return PropositionAttentionModel(llm, controller, cut_layer=12,
                                    attention_layers=(12, 13, 14, 15),
                                    gradient_checkpointing=True, query_chunk=a.query_chunk)


def propagate_estimates(progress, train_seconds_per_q=None, val_seconds_per_q=None, train_n=9000, val_n=1000):
    # Same frozen model/layers/answer head across arms; explicit target-compute
    # cost model until each arm obtains its own measured rate. Never mix val/train.
    for i in range(progress.index + 1, len(progress.names)):
        name = progress.names[i]
        if " train epoch" in name and train_seconds_per_q is not None:
            progress.weights[i] = train_n * train_seconds_per_q
        elif (" val epoch" in name or name == "TEXT validation") and val_seconds_per_q is not None:
            progress.weights[i] = val_n * val_seconds_per_q


def train_arm(a, llm, rows, pad_id, arm, baseline, run_hash, progress, stage):
    root = a.output_root / arm
    done_path = root / "COMPLETE.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done["contract_sha256"] != run_hash or sha_file(root / "best.pt") != done["best_sha256"]:
            raise RuntimeError(f"Completed checkpoint changed: {root}")
        for i in range(2*a.epochs):
            progress.stage(stage+i, 1, initial=1); progress.complete(f"cached {arm}")
        return done
    data = [shuffle_pairs(r, a.seed) if arm == "shuffled" else r for r in rows]
    if arm == "shuffled":
        stats = [r["shuffle_statistics"] for r in data]
        total_pairs = sum(x["pairs"] for x in stats)
        atomic_json(root / "SHUFFLE.json", {"questions": len(data), "pairs": total_pairs,
            "changed_pairs": sum(x["changed_pairs"] for x in stats),
            "changed_pair_fraction": sum(x["changed_pairs"] for x in stats)/max(1,total_pairs),
            "unchanged_questions": sum(x["unshufflable_or_unchanged"] for x in stats),
            "per_question": [{"sample_id": r["sample_id"], **r["shuffle_statistics"]} for r in data]})
    train = [r for r in data if r["split"] == "train"]
    validation = [r for r in data if r["split"] == "validation"]
    wrapper = make_wrapper(a, llm, arm)
    optimizer = torch.optim.AdamW(wrapper.controller.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    total_steps = math.ceil(len(train) / a.effective_batch_size) * a.epochs
    warmup = max(1, math.ceil(total_steps * .05))
    def factor(step):
        if step < warmup: return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step-warmup) / max(1, total_steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    adaptive = AdaptiveBatch(min(a.max_micro_batch_size, a.effective_batch_size), a.initial_micro_batch_size, a.memory_target)
    epoch, cursor, updates = 0, 0, 0
    best, best_epoch = dict(baseline), 0
    running = {"answer_ce_sum": 0., "bias_penalty_sum": 0., "examples": 0, "seconds": 0., "peak_gpu_bytes": 0}
    history = []
    if (root / "last.pt").exists():
        saved = load_torch(root / "last.pt")
        if saved["contract_sha256"] != run_hash or saved["arm"] != arm:
            raise RuntimeError(f"Checkpoint contract mismatch: {root}")
        wrapper.controller.load_state_dict(saved["controller"])
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        epoch, cursor, updates = saved["epoch"], saved["cursor"], saved["updates"]
        best, best_epoch, running, history = saved["best"], saved["best_epoch"], saved["running"], saved["history"]
        adaptive.buckets = saved["adaptive_buckets"]; restore_rng(saved["rng"])
    else:
        save_torch(root / "best.pt", {"contract_sha256": run_hash, "arm": arm, "epoch": 0,
                    "controller": controller_state(wrapper.controller), "validation": best})
    def checkpoint():
        save_torch(root / "last.pt", {"contract_sha256": run_hash, "arm": arm,
            "controller": controller_state(wrapper.controller), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "cursor": cursor, "updates": updates,
            "best": best, "best_epoch": best_epoch, "running": running, "history": history,
            "adaptive_buckets": adaptive.buckets, "rng": rng_state()})
    if not (root / "last.pt").exists(): checkpoint()
    for completed in range(epoch):
        for i in (0, 1):
            progress.stage(stage + 2*completed+i, 1, initial=1); progress.complete(f"cached {arm} epoch{completed+1}")
    while epoch < a.epochs:
        order = epoch_order(train, epoch, a.seed, a.effective_batch_size)
        if not 0 <= cursor <= len(train): raise RuntimeError("Invalid resume cursor")
        progress.stage(stage + 2*epoch, len(train), initial=cursor, unit="q", kind="train")
        wrapper.controller.train(); llm.eval()
        fresh_seconds = 0.; fresh_count = 0
        while cursor < len(train):
            window = [train[i] for i in order[cursor:cursor+a.effective_batch_size]]
            statistics = train_window(wrapper, optimizer, window, pad_id, adaptive, a.bias_penalty, progress)
            scheduler.step(); updates += 1; cursor += len(window)
            for k in ("answer_ce_sum", "bias_penalty_sum", "examples", "seconds"): running[k] += statistics[k]
            running["peak_gpu_bytes"] = max(running["peak_gpu_bytes"], statistics["peak_gpu_bytes"])
            fresh_seconds += statistics["seconds"]; fresh_count += len(window)
            progress.update(len(window))
            if fresh_count >= 128:
                propagate_estimates(progress, train_seconds_per_q=(time.monotonic()-progress.stage_started)/fresh_count,
                                    val_seconds_per_q=baseline.get("elapsed_seconds", 0)/len(validation) or None,
                                    train_n=len(train), val_n=len(validation))
            if updates % a.checkpoint_steps == 0 or cursor == len(train): checkpoint()
        progress.complete(f"answer CE={running['answer_ce_sum']/max(1,running['examples']):.5f}; updates={updates}; durable={root/'last.pt'}")
        selected_state = controller_state(wrapper.controller)
        metric = evaluation(wrapper, validation, pad_id, a.validation_batch_size, progress, stage+2*epoch+1,
                            root / f"validation_epoch_{epoch+1:02}.jsonl", run_hash, state_identity(selected_state))
        propagate_estimates(progress, val_seconds_per_q=metric["elapsed_seconds"]/len(validation),
                            train_n=len(train), val_n=len(validation))
        history = [x for x in history if x["epoch"] != epoch+1]
        history.append({"epoch": epoch+1, "train": dict(running), "validation": metric})
        if is_better(metric, best):
            best, best_epoch = metric, epoch+1
            save_torch(root / "best.pt", {"contract_sha256": run_hash, "arm": arm,
                       "epoch": best_epoch, "controller": selected_state, "validation": best})
        atomic_json(root / "history.json", history)
        epoch += 1; cursor = 0
        running = {"answer_ce_sum": 0., "bias_penalty_sum": 0., "examples": 0, "seconds": 0., "peak_gpu_bytes": 0}
        checkpoint()
    done = {"contract_sha256": run_hash, "arm": arm, "train_questions": len(train),
            "validation_questions": len(validation), "test_questions_used": 0, "epochs": a.epochs,
            "updates": updates, "best_epoch": best_epoch, "best_validation": best,
            "text_validation": baseline, "macro2_delta_pp": 100*(best["macro2"]-baseline["macro2"]),
            "best_sha256": sha_file(root / "best.pt"), "trainable_parameters": sum(p.numel() for p in wrapper.controller.parameters()),
            "target_llm_weights_updated": False, "lora_loaded": False, "adaptive_buckets": adaptive.buckets}
    atomic_json(done_path, done)
    del wrapper, optimizer, scheduler
    release_memory()
    return done


def runtime_checks(a, llm, rows, pad_id):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    wrapper = make_wrapper(a, llm, "graph")
    eligible = [r for r in rows if r["split"] == "train" and r["pairs"]]
    if not eligible: raise RuntimeError("No controllable train proposition pairs; do not launch vacuous training")
    sample = sorted(eligible, key=lambda r: len(r["input_ids"]))[:2]
    report = {"questions": len(sample), "target_frozen": all(not p.requires_grad for p in llm.parameters()), "test_used": False}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(llm.device)
        report["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not report["target_frozen"]: raise RuntimeError("Target Llama is not frozen")
    with sdpa_kernel(SDPBackend.MATH):
        with torch.no_grad():
            max_difference = 0.
            native_max_difference = 0.
            native_max_probability_difference = 0.
            for row in sample:
                ids = torch.tensor([row["input_ids"]], device=llm.device)
                native = llm.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
                raw = wrapper.zero_bias_reference(row)
                result = wrapper([row], pad_id=pad_id).last_hidden
                diff = float((raw.float()-result.float()).abs().max())
                max_difference = max(max_difference, diff)
                native_max_difference = max(native_max_difference,
                                             float((native.float()-result.float()).abs().max()))
                torch.testing.assert_close(result, raw, atol=.03, rtol=.02)
                w = llm.lm_head.weight[torch.tensor(row["choice_token_ids"], device=llm.device)]
                reference_logits = F.linear(raw, w).float()
                native_logits = F.linear(native, w).float()
                logits = F.linear(result, w).float()
                native_max_probability_difference = max(native_max_probability_difference,
                    float((native_logits.softmax(-1)-logits.softmax(-1)).abs().max()))
                if reference_logits.argmax().item() != logits.argmax().item():
                    raise RuntimeError("Zero-bias answer parity failed")
                if native_logits.argmax().item() != logits.argmax().item():
                    raise RuntimeError("Native-vs-chunked BF16 answer parity failed; "
                                       f"sample={row['sample_id']}, native_logits={native_logits.tolist()}, "
                                       f"chunked_logits={logits.tolist()}; no training started")
        ce, reg, _, out = choice_losses(wrapper, [sample[0]], pad_id)
        (ce + a.bias_penalty*reg).mean().backward()
        norm = sum(float(p.grad.abs().sum()) for p in wrapper.controller.parameters() if p.grad is not None)
        if not math.isfinite(norm) or norm <= 0: raise RuntimeError("Controller has no finite nonzero answer gradient")
        if any(p.grad is not None for p in llm.parameters()): raise RuntimeError("Frozen target accumulated gradients")
        del ce, reg, out
    # Test leakage with nonzero output parameters; zero initialization alone is vacuous.
    # This check is outside the MATH-only parity/gradient scope above. Keep the
    # same non-cuDNN backend policy as training and validation here as well.
    with torch.no_grad(), training_attention_backend():
        torch.manual_seed(a.seed+991)
        wrapper.controller.output.weight.normal_(0, .01)
        wrapper.controller.output.bias.fill_(.01)
        row = sample[0]
        changed = dict(row, gold_index=(row["gold_index"]+1) % len(row["option_labels"]))
        inputs = list(row["input_ids"])
        end = row["question_span"][1]
        for i in range(end+1, len(inputs)-1): inputs[i] = pad_id
        changed["input_ids"] = inputs
        first = wrapper([row], pad_id=pad_id).biases[0]
        second = wrapper([changed], pad_id=pad_id).biases[0]
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    report.update(zero_bias_max_hidden_difference=max_difference,
                  parity_reference="native Llama decoder loop; independently built zero masks; matched chunk/backend schedule",
                  native_bf16_max_hidden_difference=native_max_difference,
                  native_bf16_max_choice_probability_difference=native_max_probability_difference,
                  native_answer_parity=True,
                  controller_gradient_l1=norm, answer_suffix_and_gold_do_not_change_bias=True)
    del wrapper
    release_memory()
    return report


def prepare(a, progress, run_hash, cached_rows_path=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    rows_path = a.output_root / "prepared/rows.jsonl"
    complete = a.output_root / "prepared/COMPLETE.json"
    if cached_rows_path is not None:
        progress.stage(1, 10000, initial=10000, unit="q")
        rows = list(jsonl(cached_rows_path)); progress.complete(f"cached {cached_rows_path}")
    else:
        progress.stage(1, 10000, unit="q", kind="tokenization")
        phase = [None]
        def callback(done, total, name):
            if name != phase[0]:
                phase[0] = name
                progress.names[1] = f"CPU prepare: {name}"
                progress.stage(1, total, unit="item", kind="tokenization")
            progress.update(max(0, done-progress.done))
        rows, summary = build_rows(a.data_root, tokenizer, max_prompt_tokens=a.max_prompt_tokens,
                                   progress_callback=callback)
        atomic_jsonl(rows_path, rows)
        meta = {"contract_sha256": run_hash, "rows_sha256": sha_file(rows_path), "summary": summary}
        atomic_json(complete, meta)
        progress.complete(f"10,000 official train questions; {rows_path}")
    if Counter(r["split"] for r in rows) != {"train": 9000, "validation": 1000}:
        raise RuntimeError("Expected exact official train9000/val1000 counts")
    return rows, tokenizer


def main(argv=None):
    a = arguments(argv)
    inputs = inspect_inputs(a.data_root, a.model)
    contract = contract_for(a, inputs)
    run_hash = ensure_manifest(a.output_root / "manifest.json", contract)
    cached_rows_path = prepared_cache_source(a, contract)
    if a.preflight_only:
        print(json.dumps({"status": "metadata_preflight_ok", "data": inputs,
                          "arms_requested": a.arms, "contract_sha256": run_hash,
                          "GPU_loaded": False, "tokenization_started": False,
                          "outputs_written": False, "full_training_launched": False}, indent=2, ensure_ascii=False))
        return
    if not a.prepare_only:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("Use the supplied wrapper (CUDA_VISIBLE_DEVICES=1); no model loaded")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU 1 is not accessible in this process. Check nvidia-smi/driver outside the app; no tokenization or model load started.")
    a.output_root.mkdir(parents=True, exist_ok=True)
    lock = (a.output_root / ".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ensure_manifest(a.output_root / "manifest.json", contract, create=True)
    torch.set_num_threads(a.cpu_threads)
    names = ["metadata contract", "CPU prepare"]
    if not a.prepare_only:
        names += ["load frozen Llama", "gradient/parity safety", "TEXT validation"]
        for arm in a.arms:
            for epoch in range(1, a.epochs+1):
                names += [f"{arm} train epoch{epoch}", f"{arm} val epoch{epoch}"]
    names += ["save completion"]
    progress = Progress(names)
    progress.log("[plan] official Train9,000/Val1,000 -> exact source text+KG -> pristine frozen Llama8B -> safety checks -> TEXT validation -> "
                 + ", ".join(a.arms) + f" ({a.epochs} epochs each); no test evaluation. GPU1 only; ETA calibrating on new full-step throughput.")
    progress.log("[batch] effective=" + str(a.effective_batch_size) + f"; adaptive micro={a.initial_micro_batch_size}..{a.max_micro_batch_size}, separate length buckets, memory target={a.memory_target:.0%}; OOM rolls back whole optimizer window.")
    # Metadata/final reporting weights are explicit small cost-model placeholders;
    # actual load/check/train/val durations must be measured before total ETA exists.
    progress.weights[0] = 1.; progress.weights[-1] = 1.
    try:
        progress.stage(0, 1); progress.update(); progress.complete(f"contract={run_hash}; no old LoRA/test split")
        rows, tokenizer = prepare(a, progress, run_hash, cached_rows_path)
        if a.prepare_only:
            prepared_root = cached_rows_path.parent if cached_rows_path is not None else a.output_root / "prepared"
            progress.stage(len(names)-1, 1); progress.update(); progress.complete(f"CPU cache ready: {prepared_root}")
            progress.log("[complete] preparation only; no GPU load/training/evaluation")
            return
        progress.stage(2, 1, unit="model")
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                    dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
        llm.requires_grad_(False); llm.eval()
        if llm.config.hidden_size != 4096 or len(llm.model.layers) != 32:
            raise RuntimeError("This frozen contract requires Llama3-8B 32 layers/4096 hidden")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        progress.update(); progress.complete("pristine Llama loaded; all weights frozen; no adapter")
        progress.stage(3, 1, unit="check")
        safety = runtime_checks(a, llm, rows, pad_id)
        atomic_json(a.output_root / "SAFETY.json", safety)
        progress.update(); progress.complete(f"zero parity, no gold/option leakage, nonzero gradient; {a.output_root/'SAFETY.json'}")
        baseline_wrapper = make_wrapper(a, llm, "graph")
        # Zero-initialized biases leave attention scores unchanged. Keep the
        # same block-attention backend for numerical parity and comparable
        # validation ETA; this timing is NOT ordinary TEXT-RAG latency.
        val = [r for r in rows if r["split"] == "validation"]
        baseline = evaluation(baseline_wrapper, val, pad_id, a.validation_batch_size, progress, 4,
                              a.output_root / "TEXT_validation.jsonl", run_hash,
                              state_identity(controller_state(baseline_wrapper.controller)))
        del baseline_wrapper
        release_memory()
        propagate_estimates(progress, val_seconds_per_q=baseline["elapsed_seconds"]/len(val))
        outcomes = {}
        for i, arm in enumerate(a.arms):
            outcomes[arm] = train_arm(a, llm, rows, pad_id, arm, baseline, run_hash, progress, 5+2*a.epochs*i)
        progress.stage(len(names)-1, 1, unit="summary")
        completion = {"contract_sha256": run_hash, "arms_completed_this_invocation": a.arms,
                      "results": outcomes, "test_questions_used": 0,
                      "evaluation_contract": "Future test: pristine same Llama, forced choice logits, same source propositions/order/budget; do not reuse old LoRA/rationale results."}
        atomic_json(a.output_root / "TRAINING_SUMMARY.json", completion)
        progress.update(); progress.complete(str(a.output_root / "TRAINING_SUMMARY.json"))
        progress.log("[complete] training and validation only; selected controllers saved; no 6,545-question test evaluation executed")
    except BaseException as error:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total} remaining={progress.total-progress.done}; {type(error).__name__}: {error}; durable root={a.output_root}; identical command resumes last optimizer checkpoint, preserving old results")
        raise
    finally:
        progress.close(); lock.close()


if __name__ == "__main__":
    main()

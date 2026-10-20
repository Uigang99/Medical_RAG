#!/usr/bin/env python3
"""Bounded, conditional oracle for added token-level MedCPT cross-attention."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from medrag.generation.pair_memory_attention import (
    PairMemoryBridge, FrozenLlamaMemory, apply_messages, answer_loss, score_record)
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, identity, ensure_contract
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.training.pair_memory_oracle_data import prepare, stable, variants

CODE = ["scripts/pilot_rag2_pair_memory_oracle.py", "scripts/run_rag2_pair_memory_oracle.sh",
        "medrag/generation/pair_memory_attention.py", "medrag/training/pair_memory_oracle_data.py",
        "medrag/training/graph_input_progress.py", "medrag/training/graph_input_modules.py",
        "medrag/training/proposition_kg_attention_data.py", "medrag/training/graph_input_data.py"]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, default=ROOT / "results/rag2_official_train_proposition_attention_v1")
    p.add_argument("--source-root", type=Path, default=ROOT / "results/rag2_attention_identifiability_v1")
    p.add_argument("--encoder", type=Path, default=Path("/home/user/Uiheon/models/MedCPT-Cross-Encoder"))
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_pair_memory_oracle_v1")
    p.add_argument("--train-questions", type=int, default=256)
    p.add_argument("--dev-questions", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--tiny-steps", type=int, default=64)
    p.add_argument("--oracle-steps", type=int, default=8, help="Per start, two fixed starts 0 and 1")
    p.add_argument("--seed", type=int, default=314159)
    p.add_argument("--encoder-batch", type=int, default=32)
    p.add_argument("--memory-fraction", type=float, default=.88)
    p.add_argument("--checkpoint-upper", action="store_true", help="Slower, lower VRAM; new output contract")
    p.add_argument("--preflight-only", action="store_true", help="CPU exact text/token checks; no models/GPU")
    p.add_argument("--safety-only", action="store_true", help="One real-row parity/gradient/timing check; no training/oracle")
    a = p.parse_args(argv)
    if min(a.train_questions, a.dev_questions) < 16 or a.epochs < 1 or a.tiny_steps < 8 or a.oracle_steps < 1:
        p.error("Require >=16 train/dev, >=1 epoch/oracle step, >=8 tiny steps")
    if not .5 <= a.memory_fraction <= .92 or a.encoder_batch < 1:
        p.error("memory-fraction [0.5,.92] and encoder-batch >=1 required")
    for source in (a.prepared_root, a.source_root, a.encoder):
        if source.resolve() == a.output_root.resolve() or source.resolve() in a.output_root.resolve().parents:
            p.error("Output must be separate from source artifacts/models")
    return a


def save_pt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Store:
    def __init__(self, root, contract_hash):
        self.root, self.h = root, contract_hash

    def path(self, category, row, suffix="pt"):
        return self.root / category / (stable(0, row["sample_id"]) + "." + suffix)

    def load(self, path):
        value = (json.loads(path.read_text()) if path.suffix == ".json"
                 else torch.load(path, map_location="cpu", weights_only=False))
        if value["contract_sha256"] != self.h:
            raise ValueError(f"Incompatible cache {path}")
        return value["data"]

    def save(self, path, data):
        value = {"contract_sha256": self.h, "data": data}
        (atomic_json if path.suffix == ".json" else save_pt)(path, value)


def stats(records, baseline=None):
    n = len(records)
    if not n:
        return {"n": 0}
    out = {"n": n, "correct": sum(r["correct"] for r in records),
           "accuracy": sum(r["correct"] for r in records) / n,
           "ce": sum(r["ce"] for r in records) / n,
           "p_gold": sum(r["p_gold"] for r in records) / n}
    if baseline is not None:
        if len(baseline) != n:
            raise ValueError("Paired denominator mismatch")
        base = stats(baseline)
        out.update(baseline_accuracy=base["accuracy"],
                   delta_accuracy_pp=100 * (out["accuracy"] - base["accuracy"]),
                   delta_ce=out["ce"] - base["ce"], delta_p_gold=out["p_gold"] - base["p_gold"],
                   wrong_to_correct=sum(not b["correct"] and r["correct"] for b, r in zip(baseline, records)),
                   correct_to_wrong=sum(b["correct"] and not r["correct"] for b, r in zip(baseline, records)))
    return out


def encode_memory(a, store, specs, rows, progress):
    pending = [r for r in rows if not store.path("memory", r).exists()]
    progress.total, progress.done, progress.initial = max(1, len(rows)), len(rows) - len(pending), len(rows) - len(pending)
    if not pending:
        return
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.encoder, local_files_only=True, use_fast=True)
    # AutoModel loads BERT backbone; the scalar reranking classifier is discarded.
    encoder = AutoModel.from_pretrained(a.encoder, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
    for row in pending:
        memory = {}
        for mode in ("kg", "rewired"):
            records = specs[row["sample_id"]][mode]
            items, offset, batch = [], 0, a.encoder_batch
            while offset < len(records):
                chunk = records[offset:offset + batch]
                try:
                    padded = tok.pad([r["encoding"] for r in chunk], padding=True, return_tensors="pt")
                    with torch.no_grad(), training_attention_backend():
                        hidden = encoder(**{k: v.cuda() for k, v in padded.items()}).last_hidden_state.cpu()
                    for spec, states in zip(chunk, hidden):
                        left, right = spec["pair"]
                        items.extend([
                            {"receiver": left, "sender": right, "states": states[spec["b_tokens"]].half().contiguous()},
                            {"receiver": right, "sender": left, "states": states[spec["a_tokens"]].half().contiguous()}])
                    offset += len(chunk)
                    del hidden, padded
                except torch.cuda.OutOfMemoryError:
                    if batch == 1:
                        raise
                    batch //= 2
                    torch.cuda.empty_cache()
            memory[mode] = items
        store.save(store.path("memory", row), memory)
        progress.update()
    del encoder
    torch.cuda.empty_cache()


def prefix(store, wrapper, row):
    path = store.path("prefix", row)
    if not path.exists():
        with training_attention_backend():
            value = wrapper.prefix(row).cpu()
        store.save(path, value)
    return store.load(path).to(wrapper.device)


def row_messages(store, bridge, row, hidden, mode):
    return bridge.messages(hidden, row["proposition_spans"], store.load(store.path("memory", row))[mode])


def predict(wrapper, row, hidden, messages=None, actions=None):
    if messages:
        hidden = apply_messages(hidden, row["proposition_spans"], messages, actions)
    with training_attention_backend():
        return wrapper.logits(row, hidden)


@torch.no_grad()
def validation(store, wrapper, bridge, rows, progress=None, random_control=False):
    output = []
    for row in rows:
        hidden = prefix(store, wrapper, row)
        base = score_record(predict(wrapper, row, hidden), row["gold_index"])
        record = {"sample_id": row["sample_id"], "dataset": row["dataset"], "base": base}
        for mode in ("kg", "rewired"):
            messages = row_messages(store, bridge, row, hidden, mode)
            actions = torch.ones(len(messages), device=wrapper.device)
            record[mode] = score_record(predict(wrapper, row, hidden, messages, actions), row["gold_index"])
            if random_control and mode == "kg":
                generator = torch.Generator().manual_seed(int(stable(314159, row["sample_id"])[:12], 16))
                random_actions = (2 * torch.rand(len(messages), generator=generator)).to(wrapper.device)
                record["random_kg"] = score_record(
                    predict(wrapper, row, hidden, messages, random_actions), row["gold_index"])
        output.append(record)
        if progress:
            progress.update()
    return output


def initial_bridge(a):
    torch.manual_seed(a.seed)
    return PairMemoryBridge().cuda()


def fit(a, store, wrapper, bridge, rows, *, tiny, epoch, progress):
    """Checkpoint only complete effective-batch windows; no partial Adam resume."""
    name = "tiny" if tiny else f"epoch{epoch}"
    path = store.root / "bridge" / f"{name}.pt"
    total = a.tiny_steps if tiny else len(rows)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3 if tiny else 2e-4, weight_decay=.01)
    done, history = 0, []
    if path.exists():
        state = store.load(path)
        bridge.load_state_dict(state["bridge"])
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer state was loaded on CPU with the durable file.
        for state_values in optimizer.state.values():
            for key, val in state_values.items():
                if torch.is_tensor(val) and key != "step":
                    state_values[key] = val.cuda()
        done, history = state["done"], state["history"]
    elif not tiny and epoch > 1:
        previous = store.load(store.root / "bridge" / f"epoch{epoch-1}.pt")
        optimizer.load_state_dict(previous["optimizer"])
        for state_values in optimizer.state.values():
            for key, val in state_values.items():
                if torch.is_tensor(val) and key != "step":
                    state_values[key] = val.cuda()
    progress.total, progress.done, progress.initial = total, done, done
    effective = 1 if tiny else 8
    bridge.train()
    while done < total:
        count = min(effective, total - done)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for offset in range(count):
            idx = done + offset
            row = rows[idx % len(rows)]
            # Each training question sees real AND rewired memories across 2 epochs.
            # This makes the rewired oracle a trained-channel control, not just OOD.
            mode = "kg" if tiny or (idx + epoch) % 2 == 0 else "rewired"
            hidden = prefix(store, wrapper, row)
            messages = row_messages(store, bridge, row, hidden, mode)
            actions = torch.ones(len(messages), device=wrapper.device)
            with training_attention_backend():
                logits = predict(wrapper, row, hidden, messages, actions)
                loss = answer_loss(logits, row["gold_index"])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite bridge loss")
                (loss / count).backward()
            loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        done += count
        history.append({"done": done, "ce": loss_sum / count, "gradient_norm": float(norm)})
        store.save(path, {"done": done, "history": history, "bridge": bridge.state_dict(),
                          "optimizer": optimizer.state_dict()})
        progress.update(count)
    bridge.eval()
    return history


def safety(a, store, wrapper, bridge, row):
    """One real row: exact native/zero parity, module gradients, gate gradients."""
    hidden = prefix(store, wrapper, row)
    ids = torch.tensor([row["input_ids"]], device=wrapper.device)
    with torch.no_grad(), training_attention_backend():
        reference = wrapper.llm.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        choices = torch.tensor(row["choice_token_ids"], device=wrapper.device)
        native = torch.nn.functional.linear(reference.float(), wrapper.llm.lm_head.weight[choices].float())[0]
        zero = predict(wrapper, row, hidden)
    torch.testing.assert_close(zero, native, atol=.03, rtol=.02)
    messages = row_messages(store, bridge, row, hidden, "kg")
    actions = torch.ones(len(messages), device=wrapper.device, requires_grad=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    with training_attention_backend():
        output = predict(wrapper, row, hidden, messages, actions)
        torch.testing.assert_close(output, zero, atol=0, rtol=0)
        answer_loss(output, row["gold_index"]).backward()
    torch.cuda.synchronize()
    bridge_seconds = time.monotonic() - started
    if bridge.o.weight.grad is None or not torch.isfinite(bridge.o.weight.grad).all() or bridge.o.weight.grad.norm() == 0:
        raise RuntimeError("No usable gradient through MedCPT->Llama bridge")
    if any(p.grad is not None for p in wrapper.llm.parameters()):
        raise RuntimeError("Frozen Llama unexpectedly has gradients")
    output_grad = float(bridge.o.weight.grad.norm())
    bridge.zero_grad(set_to_none=True)
    # Test the gate channel with a small temporary nonzero projection, then RESET.
    with torch.no_grad():
        bridge.o.weight.normal_(std=1e-3)
        messages = [(r, i, x.detach()) for r, i, x in row_messages(store, bridge, row, hidden, "kg")]
    gates = torch.full((len(messages),), .5, device=wrapper.device, requires_grad=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    with training_attention_backend():
        gate_logits = predict(wrapper, row, hidden, messages, gates)
        answer_loss(gate_logits, row["gold_index"]).backward()
    torch.cuda.synchronize()
    gate_seconds = time.monotonic() - started
    if gates.grad is None or not torch.isfinite(gates.grad).all() or gates.grad.norm() == 0:
        raise RuntimeError("No gate gradient")
    with torch.no_grad():
        null = predict(wrapper, row, hidden, messages, torch.zeros_like(gates))
        torch.testing.assert_close(null, zero, atol=0, rtol=0)
        bridge.o.weight.zero_()
    bridge.zero_grad(set_to_none=True)
    return {"sample_id": row["sample_id"], "tokens": len(row["input_ids"]),
            "directed_pairs": len(messages), "native_zero_max_logit_error": float((native-zero).abs().max()),
            "bridge_output_gradient_norm": output_grad, "gate_gradient_norm": float(gates.grad.norm()),
            "bridge_forward_backward_seconds": bridge_seconds, "gate_forward_backward_seconds": gate_seconds,
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "scope": "One-row calibration only; no retained parameter update; not full feasibility/quality validation"}


def better(candidate, current):
    return (candidate["correct"], -candidate["ce"]) > (current["correct"], -current["ce"])


def oracle(a, store, wrapper, bridge, row, baseline, mode, progress):
    path = store.path("oracle_" + mode, row, "json")
    if path.exists():
        value = store.load(path)
        return value
    # Accuracy-ceiling diagnosis: true correct examples keep the gold no-op.
    # This artificially guarantees no harm and is NOT an inference policy.
    if baseline["correct"] or not row["pairs"]:
        value = {"best": baseline, "selected": "gold_noop", "search_updates": 0, "graphless": not row["pairs"]}
        store.save(path, value)
        progress.update()
        return value
    hidden = prefix(store, wrapper, row)
    with torch.no_grad():
        messages = [(r, i, x.detach()) for r, i, x in row_messages(store, bridge, row, hidden, mode)]
    n = len(messages)
    checkpoint_path = store.path("oracle_state_" + mode, row)
    state = store.load(checkpoint_path) if checkpoint_path.exists() else {
        "start": 0, "step": 0, "best": baseline, "selected": "no_op", "best_gates": [0.] * n,
        "search_updates": 0, "history": []}
    for start in range(state["start"], 2):
        gates = torch.full((n,), float(start), device=wrapper.device, requires_grad=True)
        opt = torch.optim.Adam([gates], lr=.25)
        begin = state["step"] if start == state["start"] else 0
        if "gates" in state and start == state["start"]:
            gates.data.copy_(state["gates"].cuda())
            opt.load_state_dict(state["optimizer"])
            for vals in opt.state.values():
                for k, v in vals.items():
                    if torch.is_tensor(v) and k != "step":
                        vals[k] = v.cuda()
        for step in range(begin, a.oracle_steps + 1):
            opt.zero_grad(set_to_none=True)
            with training_attention_backend():
                logits = predict(wrapper, row, hidden, messages, gates)
                record = score_record(logits, row["gold_index"])
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite oracle logits")
                state["history"].append({"start": start, "step": step, **record})
                if better(record, state["best"]):
                    state.update(best=record, selected=f"start{start}/step{step}",
                                 best_gates=gates.detach().cpu().tolist())
                if state["best"]["correct"] or step == a.oracle_steps:
                    break
                answer_loss(logits, row["gold_index"]).backward()
            if gates.grad is None or not torch.isfinite(gates.grad).all():
                raise ValueError("Non-finite/missing oracle action gradient")
            opt.step()
            with torch.no_grad():
                gates.clamp_(0., 2.)
            state["search_updates"] += 1
            state.update(start=start, step=step + 1, gates=gates.detach().cpu(), optimizer=opt.state_dict())
            store.save(checkpoint_path, state)
        if state["best"]["correct"]:
            break
        state.update(start=start + 1, step=0)
        state.pop("gates", None)
        state.pop("optimizer", None)
        store.save(checkpoint_path, state)
    # Replay the chosen joint action; no mixing separate single-pair measurements.
    with torch.no_grad():
        chosen = torch.tensor(state["best_gates"], device=wrapper.device)
        replay = score_record(predict(wrapper, row, hidden, messages, chosen), row["gold_index"])
    if replay["prediction"] != state["best"]["prediction"] or abs(replay["ce"] - state["best"]["ce"]) > .03:
        raise RuntimeError("Oracle best joint action did not replay")
    value = {k: state[k] for k in ("best", "selected", "best_gates", "search_updates", "history")}
    value["replay"] = replay
    value["action_order"] = [
        {"receiver": row["proposition_ids"][item["receiver"]],
         "sender": row["proposition_ids"][item["sender"]]}
        for item in store.load(store.path("memory", row))[mode]]
    value["changed_fraction"] = variants(row, a.seed)["rewired"]["shuffle_statistics"]["changed_fraction"]
    store.save(path, value)
    progress.update()
    return value


def report(rows, diagnosis, oracles, selected_epoch, selection, runtime, seed=314159):
    base = [r["base"] for r in diagnosis]
    conditions = {"text_only": stats(base)}
    conditions["random_kg_gates"] = stats([r["random_kg"] for r in diagnosis], base)
    for mode in ("kg", "rewired"):
        conditions["fixed_bridge_" + mode] = stats([r[mode] for r in diagnosis], base)
        conditions["oracle_" + mode] = stats([r["best"] for r in oracles[mode]], base)
    recovered = conditions["oracle_kg"]["wrong_to_correct"]
    control = conditions["oracle_rewired"]["wrong_to_correct"]
    # Paired nonparametric bootstrap, diagnostic uncertainty only (not test inference).
    g = torch.Generator().manual_seed(314159)
    delta = torch.tensor([float(o["best"]["correct"]) - float(b["correct"])
                          for o, b in zip(oracles["kg"], base)])
    samples = delta[torch.randint(len(rows), (5000, len(rows)), generator=g)].mean(1) * 100
    criteria = {"minimum_recoveries": 8, "minimum_advantage_over_rewired": 4,
                "recovery_pass": recovered >= 8, "structure_control_pass": recovered - control >= 4}
    result = {"status": "conditional_oracle_complete", "n": len(rows), "missing": 0,
              "conditions": conditions, "criteria": criteria,
              "graphless": sum(not row["pairs"] for row in rows),
              "rewiring": [dict(sample_id=row["sample_id"],
                               **variants(row, seed)["rewired"]["shuffle_statistics"]) for row in rows],
              "kg_delta_pp_bootstrap_95": [float(x) for x in torch.quantile(samples, torch.tensor([.025, .975]))],
              "selected_epoch": selected_epoch, "bridge_selection": selection, "runtime": runtime,
              "interpretation": [
                  "Gold no-op eliminates harms by construction; deployable accuracy is NOT measured.",
                  "This is finite-search achievable headroom conditional on a separately trained/frozen bridge, NOT a global upper bound.",
                  "Failure cannot distinguish insufficient bridge alignment from absence of architectural headroom.",
                  "Previously studied TRAIN128 is exploratory; no benchmark TEST and no new generalization claim.",
                  "Pair residual multipliers are [0,2], jointly optimized; native self-attention remains untouched.",
                  "Rewired control has same text and ordered degrees; unchanged rewires weaken structural attribution.",
                  "Forced MCQ next-label scoring, not rationale generation or open-ended QA."]}
    result["by_dataset"] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["dataset"] == dataset]
        result["by_dataset"][dataset] = stats([oracles["kg"][i]["best"] for i in ids], [base[i] for i in ids])
    result["per_question"] = [{**d, "kg_oracle": oracles["kg"][i], "rewired_oracle": oracles["rewired"][i]}
                              for i, d in enumerate(diagnosis)]
    return result


def main():
    a = arguments()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    names = ["CPU exact cohorts/pairs", "encode MedCPT memories", "load frozen Llama",
             "zero/gradient safety", "tiny8 alignment fit", "tiny8 fit check"]
    for epoch in range(1, a.epochs + 1):
        # Progress propagates compatible rates by the prefix before " epoch".
        # Train and validation must therefore have different prefixes.
        names += [f"bridge train epoch{epoch}", f"bridge val epoch{epoch}"]
    names += ["fixed bridge TRAIN128", "KG oracle TRAIN128", "rewired oracle TRAIN128", "paired report"]
    p = Progress(names)
    a.output_root.mkdir(parents=True, exist_ok=True)
    lock = (a.output_root / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = None
    try:
        p.log(f"[plan] official-train bridge{a.train_questions}/dev{a.dev_questions} -> frozen token memory bridge -> TRAIN128 conditional oracle; no TEST.")
        p.log(f"[scope] KG and degree-rewired control; directed joint gates [0,2], starts 0/1, {a.oracle_steps} updates/start. Gold no-op is allowed.")
        p.log("[ETA] No compatible throughput yet. --safety-only calibrates one real row before full training; CPU pair preflight runs first.")
        p.stage(0, 128 + a.train_questions + a.dev_questions)
        for rel in ("config.json", "tokenizer_config.json"):
            if not (a.encoder / rel).exists():
                raise FileNotFoundError(a.encoder / rel)
        if shutil.disk_usage(a.output_root).free < 20 * 2**30:
            raise RuntimeError("Need 20 GiB free for versioned token memory/prefix caches")
        groups, specs, metadata, prior = prepare(a, p)
        settings = {k: v for k, v in vars(a).items() if k not in
                    ("preflight_only", "safety_only", "output_root", "prepared_root", "source_root", "encoder")}
        model_files = [identity(f) for f in sorted(a.encoder.iterdir())
                       if f.is_file() and f.suffix in (".json", ".txt", ".bin", ".safetensors")]
        contract = {"version": "pair_memory_oracle_v1", "input": metadata, "encoder": model_files,
            "settings": settings, "code": {f: identity(ROOT / f, full_hash=True)["sha256"] for f in CODE},
            "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "tokenizers")},
            "architecture": {"cut_after_layers": 12, "cross_attention_width": 256, "heads": 4,
                             "frozen": ["pristine_Llama", "MedCPT_BERT"],
                             "trainable": "Q/K/V/O bridge only; no graph embedding, no answer vectors",
                             "oracle": "per-directed-pair value residual multipliers [0,2], shared across receiver tokens"},
            "training": {"loss": "four-choice CE", "effective_batch": 8, "lr": 2e-4,
                         "tiny": "8 training questions, 64 updates default; discard weights; require >=20% CE reduction",
                         "selection": "lowest mean real+rewired DEV CE among nonzero trained epochs; no diagnosis labels",
                         "rewired_exposure": "alternate by question/epoch; both modes across default two epochs"},
            "success": {"recoveries_over_same_text": 8, "recoveries_advantage_over_rewired": 4},
            "hypothesis": "Additional question/pair-conditioned token memory offers larger controllable headroom than native attention-logit bias"}
        h = ensure_contract(a.output_root / "manifest.json", contract)
        store = Store(a.output_root, h)
        if not (a.output_root / "PROVENANCE.json").exists():
            atomic_json(a.output_root / "PROVENANCE.json", {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)})
        atomic_json(a.output_root / "COHORTS.json", metadata)
        p.complete(f"train={len(groups['train'])}, dev={len(groups['dev'])}, diagnosis=128; no source mutation")
        if a.preflight_only:
            p.log("[preflight passed] Exact source/512-token checks passed; no GPU or model loaded.")
            return
        completed_summary = a.output_root / "SUMMARY.json"
        if completed_summary.exists():
            previous_result = json.loads(completed_summary.read_text())
            if previous_result.get("contract_sha256") != h:
                raise ValueError("Completed summary has incompatible contract")
            p.log(f"[already completed] {previous_result['status']}; {completed_summary}; no GPU loaded.")
            return
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
            raise RuntimeError("Run wrapper on GPU1; GPU not available, no training launched")
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
        torch.set_num_threads(8)
        torch.manual_seed(a.seed)
        torch.use_deterministic_algorithms(True)
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()
        chosen = max(groups["diagnosis"], key=lambda r: len(r["input_ids"]) if r["pairs"] else 0)
        all_rows = [chosen] if a.safety_only else groups["train"] + groups["dev"] + groups["diagnosis"]
        p.stage(1, len(all_rows), unit="question")
        encode_memory(a, store, specs, all_rows, p)
        p.complete("MedCPT backbone token memories cached; no pooled reranking scores")
        p.stage(2, 1)
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(metadata["model"], local_files_only=True,
              dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
        wrapper = FrozenLlamaMemory(llm, checkpoint_upper=a.checkpoint_upper)
        bridge = initial_bridge(a)
        p.complete("Pristine Llama and encoder frozen; only new bridge learns")
        p.stage(3, 1)
        runtime = safety(a, store, wrapper, bridge, chosen)
        atomic_json(a.output_root / "SAFETY_RUNTIME.json", {"contract_sha256": h, **runtime})
        # Conservative count model, not a claim of matched full-cohort throughput.
        updates = a.tiny_steps + a.train_questions * a.epochs + 2 * 32 * 2 * a.oracle_steps
        runtime["provisional_compute_seconds_excluding_encoding_validation_io"] = (
            (a.tiny_steps + a.train_questions * a.epochs) * runtime["bridge_forward_backward_seconds"]
            + 2 * 32 * 2 * a.oracle_steps * runtime["gate_forward_backward_seconds"])
        runtime["maximum_planned_forward_backward_calls"] = updates
        p.complete(f"One-row parity/gradients passed; bridge step={runtime['bridge_forward_backward_seconds']:.3f}s, "
                   f"gate step={runtime['gate_forward_backward_seconds']:.3f}s, peak={runtime['max_allocated_gib']:.1f} GiB")
        atomic_json(a.output_root / "SAFETY_RUNTIME.json", {"contract_sha256": h, **runtime})
        if a.safety_only:
            p.log("[safety complete] No bridge training or cohort oracle run. One-row timings are provisional.")
            return
        bridge = initial_bridge(a)  # No safety probe parameters/RNG leakage.
        p.weights[4] = a.tiny_steps * runtime["bridge_forward_backward_seconds"]
        p.weights[5] = 8 * runtime["bridge_forward_backward_seconds"]
        for epoch in range(a.epochs):
            p.weights[6+2*epoch] = a.train_questions * runtime["bridge_forward_backward_seconds"]
            p.weights[7+2*epoch] = 3*a.dev_questions * runtime["bridge_forward_backward_seconds"] / 2
        end = 6 + 2*a.epochs
        p.weights[end] = 4*128*runtime["bridge_forward_backward_seconds"] / 2
        p.weights[end+1] = p.weights[end+2] = 32*2*a.oracle_steps*runtime["gate_forward_backward_seconds"]
        p.weights[end+3] = 2.
        tiny = groups["train"][:8]
        p.stage(4, a.tiny_steps, unit="update", kind="train")
        fit(a, store, wrapper, bridge, tiny, tiny=True, epoch=0, progress=p)
        p.complete("Tiny-set optimization finished; these parameters will be discarded")
        p.stage(5, len(tiny), unit="question", kind="validation")
        tiny_result = validation(store, wrapper, bridge, tiny, p)
        before, after = stats([x["base"] for x in tiny_result]), stats([x["kg"] for x in tiny_result])
        tiny_pass = after["ce"] <= .8 * before["ce"]
        atomic_json(a.output_root / "TINY_FIT.json", {"contract_sha256": h, "before": before,
                    "after": after, "pass": tiny_pass, "criterion": "mean CE reduction >=20%"})
        p.complete(f"tiny8 CE {before['ce']:.4f}->{after['ce']:.4f}; pass={tiny_pass}")
        if not tiny_pass:
            atomic_json(a.output_root / "SUMMARY.json", {"status": "alignment_tiny_fit_failed",
                        "interpretation": "Stop: bridge optimization not established; no architectural headroom conclusion.",
                        "contract_sha256": h, "tiny_before": before, "tiny_after": after})
            return
        bridge = initial_bridge(a)
        selection = []
        for epoch in range(1, a.epochs + 1):
            p.stage(6+2*(epoch-1), a.train_questions, unit="question", kind="train")
            fit(a, store, wrapper, bridge, groups["train"], tiny=False, epoch=epoch, progress=p)
            p.complete(f"bridge epoch{epoch} durable; Llama unchanged")
            p.stage(7+2*(epoch-1), a.dev_questions, unit="question", kind="validation")
            vp = store.root / "bridge" / f"dev{epoch}.json"
            if vp.exists():
                dev = store.load(vp)
                p.initial = p.done = p.total
            else:
                dev = validation(store, wrapper, bridge, groups["dev"], p)
                store.save(vp, dev)
            mean_ce = sum(x[m]["ce"] for x in dev for m in ("kg", "rewired")) / (2 * len(dev))
            selection.append({"epoch": epoch, "mean_kg_rewired_ce": mean_ce,
                              "text": stats([x["base"] for x in dev]),
                              "kg": stats([x["kg"] for x in dev], [x["base"] for x in dev]),
                              "rewired": stats([x["rewired"] for x in dev], [x["base"] for x in dev])})
            p.complete(f"DEV mean kg+rewired CE={mean_ce:.4f}; no diagnostic selection")
        best_epoch = min(selection, key=lambda r: r["mean_kg_rewired_ce"])["epoch"]
        bridge.load_state_dict(store.load(store.root / "bridge" / f"epoch{best_epoch}.pt")["bridge"])
        bridge.eval().requires_grad_(False)
        p.stage(end, 128, unit="question", kind="validation")
        dp = store.root / "diagnosis_fixed.json"
        if dp.exists():
            diagnosis = store.load(dp)
            p.initial = p.done = p.total
        else:
            diagnosis = validation(store, wrapper, bridge, groups["diagnosis"], p, random_control=True)
            store.save(dp, diagnosis)
        if any(r["base"]["prediction"] != prior[r["sample_id"]]["base"]["prediction"] for r in diagnosis):
            raise RuntimeError("Recomputed text-only baseline predictions differ from fixed original TRAIN128")
        p.complete(f"Text baseline={stats([r['base'] for r in diagnosis])}; fixed bridge effects recorded BEFORE oracle")
        oracles = {}
        for j, mode in enumerate(("kg", "rewired")):
            cached = sum(store.path("oracle_" + mode, row, "json").exists() for row in groups["diagnosis"])
            p.stage(end+1+j, 128, initial=cached, unit="question", kind="validation")
            oracles[mode] = [oracle(a, store, wrapper, bridge, row, record["base"], mode, p)
                             for row, record in zip(groups["diagnosis"], diagnosis)]
            p.complete(f"{mode}: bounded JOINT directed gates complete; gold fallback retains original correct cases")
        p.stage(end+3, 1)
        result = report(groups["diagnosis"], diagnosis, oracles, best_epoch, selection, runtime, a.seed)
        result["contract_sha256"] = h
        atomic_jsonl(a.output_root / "per_question.jsonl", result.pop("per_question"))
        atomic_json(a.output_root / "SUMMARY.json", result)
        p.complete(str(a.output_root / "SUMMARY.json"))
        p.log(f"[complete] {result['conditions']}; this is conditional oracle headroom, not deployable accuracy.")
    except BaseException as exc:
        p.log(f"[FAILED] stage={p.index+1}/{len(names)} item={p.done}/{p.total}; {type(exc).__name__}: {exc}; "
              f"durable={a.output_root}; identical command resumes saved questions/full optimizer windows; no source deletion.")
        raise
    finally:
        p.close()
        lock.close()


if __name__ == "__main__":
    main()

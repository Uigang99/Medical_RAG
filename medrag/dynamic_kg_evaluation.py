"""Frozen-checkpoint selection and paired reporting for dynamic KG attention."""
from __future__ import annotations

import json
from pathlib import Path

from medrag.training.graph_input_data import digest, sha_file


def selected_controllers(root, manifest, variants, load_torch, *, allow_pending=False):
    root = Path(root)
    contract, run_hash = manifest["contract"], manifest["contract_sha256"]
    selected, weights, pending = {}, {}, []
    for variant in variants:
        if variant not in contract["variants"]:
            raise RuntimeError(f"Variant {variant} was not in the training contract")
        directory = root / variant / "full"
        required = [directory / n for n in ("COMPLETE.json", "best.pt", "history.json")]
        required.append(root / variant / "tiny32/GATE.json")
        if any(not p.is_file() for p in required):
            pending.append(variant)
            continue
        gate = json.loads(required[-1].read_text())
        done = json.loads(required[0].read_text())
        history = json.loads(required[2].read_text())
        if (gate["contract_sha256"] != run_hash or not gate["passed"] or
            done["contract_sha256"] != run_hash or done["variant"] != variant or
            done["evaluation_questions_used"] != 0 or done["tiny_weights_reused"] or
            done["train_questions"] != 2048 or done["validation_questions"] != 256 or
            done["epochs"] != contract["training"]["epochs"] or
            sha_file(required[1]) != done["best_sha256"]):
            raise RuntimeError(f"Invalid completed training/checkpoint contract: {variant}")
        epochs = contract["training"]["epochs"]
        if (len(history) != epochs+1 or [h["epoch"] for h in history] != list(range(epochs+1)) or
            any(h["validation_n"] != 256 for h in history) or
            any(h["train_n"] != 2048 for h in history[1:]) or
            min(h["validation_ce"] for h in history) != done["best_validation_ce"]):
            raise RuntimeError(f"Invalid validation selection history: {variant}")
        best = load_torch(required[1])
        if (best["contract_sha256"] != run_hash or best["variant"] != variant or
            best["epoch"] != done["selected_epoch"] or best["validation_ce"] != done["best_validation_ce"] or
            history[best["epoch"]]["validation_ce"] != best["validation_ce"]):
            raise RuntimeError(f"Selected checkpoint does not match validation result: {variant}")
        selected[variant] = {"path": str(required[1].resolve()), "sha256": done["best_sha256"],
                             "epoch": best["epoch"], "validation_ce": best["validation_ce"]}
        weights[variant] = best["controller"]
    if pending and not allow_pending:
        raise RuntimeError(f"Training is not complete for {pending}; wait for <variant>/full/COMPLETE.json. "
                           "Refusing tiny32/last/intermediate checkpoints; --prepare-only checks inputs without evaluation.")
    return selected, weights, pending


def read_shard(path, run_hash, rows):
    path = Path(path)
    if not path.exists(): return None
    item = json.loads(path.read_text())
    if (item["contract_sha256"] != run_hash or item["rows_sha256"] != digest(item["rows"]) or
        [x["sample_id"] for x in item["rows"]] != [x["sample_id"] for x in rows]):
        raise RuntimeError(f"Cached prediction contract/IDs/checksum mismatch: {path}")
    return item["rows"]


def paired_summary(rows, predictions, static_summarize):
    """Reuse the frozen metric definitions/paired bootstrap; do not tune on 512."""
    target = "graph" if "graph" in predictions else "local"
    result = static_summarize(rows, {**predictions, "graph_attention": predictions[target]})
    stats = {k: v for k, v in result["conditions"].items() if k != "graph_attention"}
    comparisons = {}
    pairs = [(v, b) for v in ("local", "graph") if v in predictions
             for b in ("no_rag", "text", "static_attention")]
    if "local" in predictions and "graph" in predictions: pairs.append(("graph", "local"))
    for condition, baseline in pairs:
        part = static_summarize(rows, {"no_rag": predictions["no_rag"], "text": predictions[baseline],
                                       "graph_attention": predictions[condition]})["primary"]
        comparisons[f"{condition}_vs_{baseline}"] = {**part, "baseline": baseline, "condition": condition}
    primary = comparisons[f"{target}_vs_text"]
    return {"n": len(rows), "conditions": stats, "paired_comparisons": comparisons,
            "primary": primary, "exploratory_gate": {"minimum_delta_macro3_pp": 2.,
                "require_ci95_lower_above_zero": True,
                "passed": bool(primary["delta_macro3_pp"] >= 2. and primary["macro3_ci95_pp"][0] > 0)},
            "scope": "fixed, previously used TEST-repartition 512-question development evaluation; not untouched final TEST",
            "mechanism_comparison": "graph_vs_local isolates neighboring node states; local still uses KG-aligned positions",
            "model": "all conditions use the same frozen Llama plus selected shared medical LoRA"}

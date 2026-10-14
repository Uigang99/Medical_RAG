"""Variable-length node-token generation and paired MCQ reporting.

Keep the original six-condition evaluator immutable. Its generation helper
requires exactly one graph token and must not be used for this representation.
"""
from __future__ import annotations

from collections import Counter

from medrag.text_graph_evaluation import GROUPS, group, wilson

NODE_CONDITION = "node_graph"
BASELINES = ("no_rag", "text", "pooled_graph")


def generation_inputs(model, prompts, soft_prompts, pad_id):
    """BOS, all node tokens in frozen node-ID order, remaining question prompt.

    No gold labels or target IDs are accepted. Empty graphs add zero tokens.
    Placeholder token IDs only identify the returned generate() prefix.
    """
    import torch
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    if not prompts or len(prompts) != len(soft_prompts) or any(not p for p in prompts):
        raise ValueError("Invalid prompt/node vector batch")
    items = []
    for prompt, soft in zip(prompts, soft_prompts, strict=True):
        ids = torch.tensor(prompt, device=device, dtype=torch.long)
        emb = embedding(ids)
        if soft.ndim != 2 or soft.shape[1] != emb.shape[-1] or not torch.isfinite(soft).all():
            raise ValueError("Expected finite [node_count, LLM hidden_size] vectors")
        soft = soft.to(device=device, dtype=emb.dtype)
        ids = torch.cat((ids[:1], ids.new_full((len(soft),), pad_id), ids[1:]))
        emb = torch.cat((emb[:1], soft, emb[1:]))
        items.append((ids, emb))
    width = max(len(ids) for ids, _ in items)
    pad = embedding(torch.tensor(pad_id, device=device))
    ids_out, embeds_out, masks = [], [], []
    for ids, emb in items:
        extra = width - len(ids)
        ids_out.append(torch.cat((ids.new_full((extra,), pad_id), ids)))
        embeds_out.append(torch.cat((pad.expand(extra, -1), emb)))
        masks.append([0]*extra + [1]*len(ids))
    return {"input_ids": torch.stack(ids_out), "inputs_embeds": torch.stack(embeds_out),
            "attention_mask": torch.tensor(masks, device=device, dtype=torch.long)}


def generate_tokens(model, prompts, soft_prompts, *, pad_id, eos_ids, max_new_tokens):
    import torch
    from transformers import GenerationConfig
    with torch.inference_mode():
        packed = generation_inputs(model, prompts, soft_prompts, pad_id)
        width = packed["input_ids"].shape[1]
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id, eos_token_id=eos_ids, bos_token_id=model.config.bos_token_id, use_cache=True)
        output = model.generate(**packed, generation_config=config)
        if not torch.equal(output[:, :width], packed["input_ids"]):
            raise RuntimeError("Unexpected generated prefix; refuse incorrect output slicing")
        result = []
        for tokens in output[:, width:].tolist():
            stop = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
            result.append({"generated_ids": tokens if stop is None else tokens[:stop+1],
                           "finish_reason": "length" if stop is None else "eos"})
        return result


def summarize(rows, predictions, bootstrap_replicates=2000, seed=42):
    """Same Macro-3/micro definitions and paired stratified bootstrap as before."""
    import numpy as np
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError("Duplicate question IDs")
    expected = set(BASELINES + (NODE_CONDITION,))
    if set(predictions) != expected: raise ValueError("Missing comparison condition")
    indices = {g: [i for i, r in enumerate(rows) if group(r["dataset"]) == g] for g in GROUPS}
    if any(not v for v in indices.values()): raise ValueError("Macro-3 needs all benchmark groups")
    rng = np.random.default_rng(seed)
    draws = {g: rng.integers(0, len(v), (bootstrap_replicates, len(v))) for g, v in indices.items()}
    def bootstrap(values):
        means = [values[v][draws[g]].mean(axis=1) for g, v in indices.items()]
        macro = np.mean(means, axis=0)
        micro = sum(s*len(indices[g]) for s, g in zip(means, GROUPS)) / len(rows)
        return {"micro_ci95_pp": (100*np.quantile(micro, [.025, .975])).tolist(),
                "macro3_ci95_pp": (100*np.quantile(macro, [.025, .975])).tolist()}
    stats, correct = {}, {}
    for name in BASELINES + (NODE_CONDITION,):
        items = {p["sample_id"]: p for p in predictions[name]}
        if len(items) != len(predictions[name]) or set(items) != set(ids):
            raise ValueError(f"Missing/extra/duplicate predictions: {name}")
        values = np.array([items[r["sample_id"]]["prediction"] == r["answer"] for r in rows], dtype=float)
        correct[name] = values
        def score(selected):
            n = len(selected); c = int(values[selected].sum())
            return {"n": n, "correct": c, "accuracy_pct": 100*c/n,
                "parse_failures": sum(items[ids[i]]["prediction"] is None for i in selected),
                "length_limited": sum(items[ids[i]]["finish_reason"] == "length" for i in selected),
                "wilson_ci95_pct": wilson(c, n)}
        groups = {g: score(v) for g, v in indices.items()}
        stats[name] = {**score(list(range(len(rows)))), "missing": 0, "micro_pct": 100*values.mean(),
            "macro3_pct": sum(s["accuracy_pct"] for s in groups.values())/3, "groups": groups,
            "datasets": {d: score([i for i, r in enumerate(rows) if r["dataset"] == d])
                         for d in sorted({r["dataset"] for r in rows})}, "bootstrap": bootstrap(values)}
    pairs = []
    for name in BASELINES:
        a, b = correct[name], correct[NODE_CONDITION]
        counts = Counter(("C" if x else "W")+"_"+("C" if y else "W") for x, y in zip(a, b))
        pairs.append({"baseline": name, "condition": NODE_CONDITION, "n": len(rows),
            "baseline_micro_pct": stats[name]["micro_pct"], "condition_micro_pct": stats[NODE_CONDITION]["micro_pct"],
            "baseline_macro3_pct": stats[name]["macro3_pct"], "condition_macro3_pct": stats[NODE_CONDITION]["macro3_pct"],
            "delta_micro_pp": 100*(b-a).mean(),
            "delta_macro3_pp": stats[NODE_CONDITION]["macro3_pct"]-stats[name]["macro3_pct"],
            "transitions": {k: counts[k] for k in ("C_C", "C_W", "W_C", "W_W")}, **bootstrap(b-a)})
    primary = next(p for p in pairs if p["baseline"] == "text")
    return {"n": len(rows), "conditions": stats, "paired_comparisons": pairs,
        "primary_metric": "node GRAPH minus same-LLM TEXT, paired Macro-3 accuracy",
        "primary_comparison": primary,
        "exploratory_scale_up_gate": {"minimum_delta_macro3_pp": 2., "require_ci95_lower_above_zero": True,
            "passed": primary["delta_macro3_pp"] >= 2 and primary["macro3_ci95_pp"][0] > 0},
        "scope": "fixed TEST-repartition development pilot, already used for previous model evaluation; not a new untouched test",
        "claim_limit": "node readout, two-way messages and target alignment changed; not a pure topology causal estimate"}

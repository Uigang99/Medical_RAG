"""Paired, held-out six-condition TEXT/KG evaluation; no model training."""
from __future__ import annotations

import math
import re
from collections import Counter

from medrag.training.graph_input_data import graph_from_edges, graph_stats, prompt_ids, source_propositions

# Preserve the condition numbering agreed with the researcher.
CONDITIONS = (
    ("1_base_no_rag", False, "none"),
    ("2_base_text", False, "text"),
    ("3_trained_text", True, "text"),
    ("4_base_graph", False, "graph"),
    ("5_trained_graph", True, "graph"),
    ("6_trained_no_rag", True, "none"),
)
GROUPS = ("medmcqa", "medqa", "mmlu_pooled")


def group(dataset):
    return "mmlu_pooled" if dataset.startswith("mmlu_") else dataset


def pack_evaluation_pair(tokenizer, row, edges, max_context=8192, max_new_tokens=16):
    """Evidence selection never reads an answer, including for context budgeting.

    Preserve the training source/order/whole-proposition policy. Reserve the
    fixed generation budget instead of the gold target length used in training.
    Remove the same source propositions AND their edges from both conditions.
    """
    props = source_propositions(edges)
    before = len(props)
    query = prompt_ids(tokenizer, row, [], "graph")
    if len(query) + 1 + max_new_tokens > max_context:
        raise ValueError(f"Question alone exceeds context: {row['sample_id']}")
    while True:
        text = prompt_ids(tokenizer, row, props, "text")
        if max(len(text), len(query) + 1) + max_new_tokens <= max_context:
            break
        props.pop()
    pids = {p["proposition_id"] for p in props}
    kept = [e for e in edges if e["proposition_id"] in pids]
    graph = graph_from_edges(kept)
    return {**row, "propositions": props, "edges": kept, "graph": graph,
            "text_prompt_ids": text, "graph_prompt_ids": query, "none_prompt_ids": query,
            "packing": {"before_propositions": before, "kept_propositions": len(props),
                        "removed_propositions": before - len(props)}, "graph_stats": graph_stats(graph)}


def generation_inputs(model, prompts, pad_id, soft_prompts=None):
    """Left padding + optional graph token AFTER BOS, exactly as during training.

    Dummy IDs at graph-token positions are output-prefix placeholders only;
    inputs_embeds, not these IDs, are consumed on the first generation forward.
    This function cannot accept gold labels or target token IDs.
    """
    import torch
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    items, ids = [], []
    if not prompts or any(not p for p in prompts):
        raise ValueError("Empty generation prompts")
    if soft_prompts is not None and len(soft_prompts) != len(prompts):
        raise ValueError("Graph prompt count mismatch")
    for i, prompt in enumerate(prompts):
        tok = torch.tensor(prompt, dtype=torch.long, device=device)
        emb = embedding(tok)
        if soft_prompts is not None:
            soft = soft_prompts[i].to(device=device, dtype=emb.dtype)
            if soft.ndim != 2 or soft.shape != (1, emb.shape[-1]) or not torch.isfinite(soft).all():
                raise ValueError("Expected one finite graph soft token")
            emb = torch.cat((emb[:1], soft, emb[1:]))
            tok = torch.cat((tok[:1], tok.new_full((len(soft),), pad_id), tok[1:]))
        items.append(emb)
        ids.append(tok)
    width = max(len(x) for x in ids)
    masks, packed, token_ids = [], [], []
    pad = embedding(torch.tensor(pad_id, device=device))
    for tok, emb in zip(ids, items, strict=True):
        extra = width - len(tok)
        packed.append(torch.cat((pad.expand(extra, -1), emb)))
        token_ids.append(torch.cat((tok.new_full((extra,), pad_id), tok)))
        masks.append([0] * extra + [1] * len(tok))
    return {"input_ids": torch.stack(token_ids), "inputs_embeds": torch.stack(packed),
            "attention_mask": torch.tensor(masks, device=device, dtype=torch.long)}


def generate_tokens(model, prompts, *, pad_id, eos_ids, max_new_tokens, soft_prompts=None):
    import torch
    from transformers import GenerationConfig
    with torch.inference_mode():
        packed = generation_inputs(model, prompts, pad_id, soft_prompts)
        width = packed["input_ids"].shape[1]
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                  pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                                  bos_token_id=model.config.bos_token_id)
        generated = model.generate(**packed, generation_config=config)
        # HF returns the input ID prefix even when inputs_embeds supplies its values.
        if not torch.equal(generated[:, :width], packed["input_ids"]):
            raise RuntimeError("Unexpected generate() prefix; refuse to score incorrectly sliced outputs")
        result = []
        for raw in generated[:, width:].tolist():
            stop = next((i for i, token in enumerate(raw) if token in eos_ids), None)
            result.append({"generated_ids": raw if stop is None else raw[:stop + 1],
                           "finish_reason": "length" if stop is None else "eos"})
        return result


def parse_answer(text, options):
    """One fixed strict parser; never infer a letter from option text or gold."""
    match = re.fullmatch(r"\s*(?:Answer\s*:\s*)?([A-Za-z])\s*[.]?\s*", text, flags=re.I)
    if match and match[1].upper() in options:
        return match[1].upper()
    return None


def wilson(correct, n):
    if not n:
        return None
    p, z = correct / n, 1.959963984540054
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [100*(centre-half), 100*(centre+half)]


def summarize(rows, predictions, bootstrap_replicates=2000, seed=42):
    """All denominators include parse failures. Missing/duplicate rows are fatal.

    Macro-3 = unweighted mean of MedMCQA, MedQA, pooled six MMLU subjects.
    Paired bootstrap resamples inside these three fixed benchmark groups.
    """
    import numpy as np
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation IDs")
    by_id = {r["sample_id"]: r for r in rows}
    groups = {g: [i for i, r in enumerate(rows) if group(r["dataset"]) == g] for g in GROUPS}
    if any(not indices for indices in groups.values()):
        raise ValueError("Macro-3 requires all three benchmark groups")
    rng = np.random.default_rng(seed)
    draws = {g: rng.integers(0, len(v), (bootstrap_replicates, len(v))) for g, v in groups.items()}
    def bootstrap(values):
        samples = [values[v][draws[g]].mean(axis=1) for g, v in groups.items()]
        macro = np.mean(samples, axis=0)
        micro = sum(s*len(groups[g]) for s, g in zip(samples, GROUPS)) / len(rows)
        return {"micro_ci95_pp": (100*np.quantile(micro, [.025,.975])).tolist(),
                "macro3_ci95_pp": (100*np.quantile(macro, [.025,.975])).tolist()}
    stats, correctness, paired = {}, {}, []
    for name, _, _ in CONDITIONS:
        items = predictions[name]
        found = {r["sample_id"]: r for r in items}
        if len(found) != len(items) or set(found) != set(ids):
            raise ValueError(f"Missing, extra or duplicate predictions: {name}")
        values = np.array([found[s]["prediction"] == by_id[s]["answer"] for s in ids], dtype=float)
        correctness[name] = values
        def score(indices):
            n, correct = len(indices), int(values[indices].sum())
            return {"n": n, "correct": correct, "accuracy_pct": 100*correct/n,
                    "wilson_ci95_pct": wilson(correct, n),
                    "parse_failures": sum(found[ids[i]]["prediction"] is None for i in indices),
                    "length_limited": sum(found[ids[i]]["finish_reason"] == "length" for i in indices)}
        subgroups = {g: score(v) for g, v in groups.items()}
        datasets = {ds: score([i for i,r in enumerate(rows) if r["dataset"] == ds])
                    for ds in sorted({r["dataset"] for r in rows})}
        stats[name] = {**score(list(range(len(rows)))), "micro_pct": 100*values.mean(),
                       "macro3_pct": sum(v["accuracy_pct"] for v in subgroups.values())/3,
                       "groups": subgroups, "datasets": datasets, "missing": 0,
                       "bootstrap": bootstrap(values)}
    pairs = [("1_base_no_rag", "6_trained_no_rag"), ("2_base_text", "3_trained_text"),
             ("4_base_graph", "5_trained_graph"), ("1_base_no_rag", "2_base_text"),
             ("1_base_no_rag", "4_base_graph"), ("2_base_text", "4_base_graph"),
             ("6_trained_no_rag", "3_trained_text"), ("6_trained_no_rag", "5_trained_graph"),
             ("3_trained_text", "5_trained_graph")]
    for before, after in pairs:
        a, b = correctness[before], correctness[after]
        transitions = Counter(("C" if x else "W") + "_" + ("C" if y else "W") for x,y in zip(a,b))
        paired.append({"baseline": before, "condition": after, "n": len(rows),
                       "transitions": {k: transitions[k] for k in ("C_C", "C_W", "W_C", "W_W")},
                       "baseline_micro_pct": stats[before]["micro_pct"],
                       "condition_micro_pct": stats[after]["micro_pct"],
                       "delta_micro_pp": 100*(b-a).mean(),
                       "baseline_macro3_pct": stats[before]["macro3_pct"],
                       "condition_macro3_pct": stats[after]["macro3_pct"],
                       "delta_macro3_pp": stats[after]["macro3_pct"]-stats[before]["macro3_pct"],
                       **bootstrap(b-a)})
    primary = paired[-1]
    return {"n": len(rows), "conditions": stats, "paired_comparisons": paired,
            "primary_comparison": primary,
            "primary_metric": "paired Macro-3 accuracy difference: trained GRAPH minus trained TEXT",
            "exploratory_scale_up_gate": {"minimum_delta_macro3_pp": 2.,
                "require_ci95_lower_above_zero": True,
                "passed": primary["delta_macro3_pp"] >= 2 and primary["macro3_ci95_pp"][0] > 0},
            "scope": "TEST-repartition pilot, not untouched benchmark TEST; no tuning on these results",
            "claim_limit": "matched sources and shared LoRA; not a pure topology or pure domain-knowledge causal effect"}

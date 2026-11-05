"""Exact-prefix reuse across sentence-PASS ablations, not a new decoder.

Run one anchor policy normally. Another policy may borrow its per-token decision
only while its full token prefix, phase and EOS mask are identical. At the first
different token proposal, fork the causal KV at that prefix and decode normally.
No vocabulary distribution, expert or shared token is averaged across policies.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from transformers import DynamicCache

from medrag.generation import pced_sentence_pass as method


def fork_prefix(cache, length):
    if length < 0 or length > cache.get_seq_length():
        raise ValueError("Invalid causal cache fork length")
    pairs = []
    for layer in cache.layers:
        if layer.keys.ndim != 4 or layer.values.shape != layer.keys.shape:
            raise ValueError("Only full-attention Llama KV caches are supported")
        pairs.append((layer.keys[..., :length, :].clone(), layer.values[..., :length, :].clone()))
    return DynamicCache(pairs)


class MemoJudge:
    """Reuse decisions only for the exact same question-local shared text/phase."""
    def __init__(self, judge):
        self.judge, self.memo = judge, {}
        self.hits, self.misses = 0, 0

    def __call__(self, text, phase):
        key = (text, phase)
        if key in self.memo:
            self.hits += 1
            return dict(deepcopy(self.memo[key]), seconds=0., exact_prefix_cache_hit=True)
        self.misses += 1
        result = self.judge(text, phase)
        self.memo[key] = deepcopy(result)
        return dict(result, exact_prefix_cache_hit=False)


class ReplayStream(method.Stream):
    def __init__(self, *args, anchor, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cache is not None:
            raise ValueError("Replay stream starts without a private cache")
        if anchor.weight != self.weight or anchor.gamma != self.gamma:
            raise ValueError("Only the gate policy may differ between reused streams")
        self.anchor, self.replaying = anchor, True
        self.reused_steps = 0

    def step(self, phase, local_step, *, allow_eos=False):
        index = len(self.traces)
        if self.replaying and index < len(self.anchor.traces):
            source = self.anchor.traces[index]
            if (source["phase"], source["local_step"], source["eos_allowed"]) != (phase, local_step, allow_eos):
                raise RuntimeError("Identical-prefix reuse reached a different generation phase")
            if self.arm != "pced":
                original_c = self.anchor.events[source["event_index"]]["contribute_probability"]
                if self.contribution != original_c:
                    raise RuntimeError("Identical-prefix reuse reached a different gate decision")
            selected = source["counterfactual_same_prefix"][self.arm]
            if selected["token_id"] == source["token_id"]:
                trace = deepcopy(source)
                trace.update(selected, event_index=self.event_index, model_forward_reused=True)
                if self.arm == "pced":
                    # The no-gate baseline has c=1 for every expert. Its own
                    # counterfactual diagnostics must not inherit the anchor gate.
                    trace.update(active_experts=list(range(len(self.prior))), excluded_experts=[], all_pass=False,
                                 weighted_best_scores=trace["pced_best_scores"].copy(),
                                 counterfactual_same_prefix={name: dict(selected) for name in method.ARMS})
                else:
                    trace["excluded_experts"] = [i for i, c in enumerate(self.contribution) if c <= .5]
                self.traces.append(trace)
                self.reused_steps += 1
                self.beta = self.anchor.beta
                self.pending = None
                token = selected["token_id"]
                if token not in self.eos:
                    self.force([token])
                return token, token in self.eos
        if self.replaying:
            # Every earlier output token is identical. Future tokens in the
            # anchor cache cannot affect this causal prefix. Never mutate anchor.
            width = self.mask.shape[-1]-self.pending.shape[-1]
            self.cache = fork_prefix(self.anchor.cache, width)
            self.replaying = False
        token, eos = super().step(phase, local_step, allow_eos=allow_eos)
        self.traces[-1]["model_forward_reused"] = False
        return token, eos


@torch.inference_mode()
def check_live_fork(model, context, suffix, prior, pass_ids, jsd, prefill, special_ids=()):
    """Three real decode positions; compare a cloned KV continuation before long work."""
    cache = prefill(model, context)
    left = method.Stream(model, cache, context, suffix, prior, pass_ids, jsd, "pced", special_ids=special_ids)
    left.begin_segment("", "reasoning", "start")
    for i in range(2):
        left.step("reasoning", i, allow_eos=False)
    copied = fork_prefix(left.cache, left.cache.get_seq_length())
    right = method.Stream(model, copied, context, suffix, prior, pass_ids, jsd, "pced", special_ids=special_ids)
    right.pending, right.mask, right.beta = left.pending.clone(), left.mask.clone(), left.beta.clone()
    right.begin_segment("", "reasoning", "start")
    a = left.step("reasoning", 2, allow_eos=False)
    b = right.step("reasoning", 2, allow_eos=False)
    if a != b:
        raise RuntimeError("Cache fork changed selected token; refusing optimized run")
    l, r = left.traces[-1], right.traces[-1]
    for name in ("pced_best_token_ids", "pced_best_scores", "selected_token_raw_logits", "selected_token_beta"):
        torch.testing.assert_close(torch.tensor(l[name]), torch.tensor(r[name]), atol=0., rtol=0.)
    return {"passed": True, "next_token_id": a[0], "expert_scores_exact": True,
            "note": "Cache fork runtime parity; generation quality is not tested by this check"}

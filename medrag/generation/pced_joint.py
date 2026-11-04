"""Temporary two-document experts; independent PCED scores remain unchanged.

Selection JSD is a scalar divergence in nats, computed in float32. PCED beta
deliberately retains the pinned decoder's *elementwise*, first-step JSD.
Those two calculations must not be conflated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn.functional as F

QUESTION = ("Based on the documents above, answer the following query. "
            "Write brief reasoning before a concise final answer.\nquery: {question}")
PREFIX = "Reasoning:"
ANSWER_MARKER = "\nAnswer:"


@dataclass(frozen=True)
class JointConfig:
    gamma: float = 2.5
    warmup: int = 8
    check_every: int = 8
    lifetime: int = 32
    max_creations: int = 2
    relevance_ratio: float = 0.25
    min_effect_jsd: float = 0.001
    min_pair_jsd: float = 0.05
    prior_method: str = "geometric"

    def __post_init__(self):
        if self.warmup < 0 or self.check_every < 1 or self.lifetime < 1 or self.max_creations < 0:
            raise ValueError("Invalid joint scheduling budget")
        if self.prior_method not in ("geometric", "arithmetic", "minimum"):
            raise ValueError("Unknown joint prior method")
        for name in ("gamma", "relevance_ratio", "min_effect_jsd", "min_pair_jsd"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.relevance_ratio > 1 or max(self.min_effect_jsd, self.min_pair_jsd) > math.log(2):
            raise ValueError("Ratios must be <=1; JSD thresholds must be <=log(2)")


def joint_prior(a: float, b: float, method="geometric") -> float:
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (a, b)):
        raise ValueError("Joint prior requires normalized finite priors in [0,1]")
    a, b = max(a, 1e-12), max(b, 1e-12)
    if method == "geometric":
        return math.exp((math.log(a) + math.log(b)) / 2)
    if method == "arithmetic":
        return (a + b) / 2
    if method == "minimum":
        return min(a, b)
    raise ValueError(method)


def probability_jsd(p, q):
    """Normalized probability vectors, last dimension vocabulary; nats."""
    p, q = p.float(), q.float()
    log_m = ((p + q) * 0.5).clamp_min(1e-30).log()
    return (0.5 * (p * (p.clamp_min(1e-30).log() - log_m) +
                   q * (q.clamp_min(1e-30).log() - log_m))).sum(-1).clamp_min(0)


@torch.no_grad()
def select_pair(amateur, experts, priors, config, used=()):
    """Exact maximum over ALL eligible pairs, not a top-1-token proxy.

    Only 16 vocabulary-sized pair comparisons are materialized together.
    Nothing here consumes a gold answer or interprets disagreement as truth.
    """
    effect = probability_jsd(F.softmax(experts.float(), -1), F.softmax(amateur.float(), -1))
    prior = priors.float()
    eligible = ((prior > 0) & (prior >= config.relevance_ratio * prior.max()) &
                (effect >= config.min_effect_jsd)).nonzero().flatten().tolist()
    used = set(used)
    pairs = [(a, b) for k, a in enumerate(eligible) for b in eligible[k+1:] if (a, b) not in used]
    report = {"eligible_experts": [i+1 for i in eligible], "eligible_pairs": len(pairs)}
    if not pairs:
        return None, report
    # A logit difference is not itself a probability distribution.
    delta_distribution = F.softmax(experts.float() - amateur.float(), dim=-1)
    indices = torch.tensor(pairs, dtype=torch.long, device=experts.device)
    values = []
    for block in indices.split(16):
        values.append(probability_jsd(delta_distribution[block[:, 0]], delta_distribution[block[:, 1]]))
    divergences = torch.cat(values)
    winner = int(divergences.argmax().item())
    divergence = float(divergences[winner].item())
    a, b = pairs[winner]
    report.update(pair_ranks=[a+1, b+1], pair_jsd=divergence,
                  document_effect_jsd=[float(effect[a]), float(effect[b])],
                  parent_priors=[float(prior[a]), float(prior[b])])
    if divergence < config.min_pair_jsd:
        return None, report
    return (a, b), report


def pced_scores(amateur, expert, beta, priors, gamma):
    # Same arithmetic/precision and expert-first tie order as official PCED.
    return (1 + beta) * expert - beta * amateur + gamma * priors.clamp_min(1e-12).log().unsqueeze(-1)


class CachedStream:
    def __init__(self, model, cache, pending, mask):
        self.model, self.cache, self.pending, self.mask = model, cache, pending, mask

    @torch.no_grad()
    def logits(self):
        if self.pending is None:
            raise RuntimeError("Missing shared prefix token")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.pending.is_cuda):
            result = self.model(input_ids=self.pending, attention_mask=self.mask,
                                past_key_values=self.cache, use_cache=True, num_logits_to_keep=1)
        self.cache = result.past_key_values
        self.pending = None
        logits = result.logits[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Non-finite next-token logits; refusing to cache an invalid prediction")
        return logits

    def queue(self, ids):
        if not ids:
            raise ValueError("Cannot append an empty prefix")
        block = torch.tensor(ids, dtype=torch.long, device=self.mask.device).unsqueeze(0).expand(self.mask.shape[0], -1)
        self.pending = block if self.pending is None else torch.cat((self.pending, block), -1)
        self.mask = torch.cat((self.mask, self.mask.new_ones(block.shape)), -1)


class JointStream:
    def __init__(self, model, cache, context, suffix, priors, get_jsd, *, config=None, joint_builder=None):
        self.config = config or JointConfig()
        self.base = CachedStream(model, cache, suffix["input_ids"],
                                 torch.cat((context["attention_mask"], suffix["attention_mask"]), -1))
        self.priors, self.get_jsd = priors, get_jsd
        self.builder = joint_builder  # None means exact independent-only baseline.
        self.beta = None
        self.joint = None
        self.joint_beta = None
        self.history, self.trace, self.events, self.checks = [], [], [], []
        self.used = set()
        self.position = self.creations = self.joint_used = 0
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def force(self, ids):
        self.history.extend(int(i) for i in ids)
        self.base.queue(ids)
        if self.joint is not None:
            self.joint.queue(ids)

    def retire(self, reason):
        if self.joint is not None:
            self.events[-1].update(end_position=self.position, used_positions=self.joint_used, end_reason=reason)
        self.joint = self.joint_beta = None

    @torch.no_grad()
    def step(self, phase):
        logits = self.base.logits()
        amateur, experts = logits[:1], logits[1:]
        if self.beta is None:
            self.beta = self.get_jsd(amateur, experts).clamp(0, 1)
        scores = pced_scores(amateur, experts, self.beta, self.priors, self.config.gamma)
        best_scores, best_tokens = scores.max(-1)
        baseline_winner = int(best_scores.argmax())
        baseline_token = int(best_tokens[baseline_winner])
        cfg = self.config
        if (self.builder is not None and self.joint is None and phase == "reasoning" and
                self.creations < cfg.max_creations and self.position >= cfg.warmup and
                (self.position-cfg.warmup) % cfg.check_every == 0):
            pair, check = select_pair(amateur, experts, self.priors, cfg, self.used)
            check.update(position=self.position)
            self.checks.append(check)
            if pair is not None:
                self.joint, metadata = self.builder(pair, list(self.history))
                self.joint_beta, self.joint_used = None, 0
                self.joint_prior = joint_prior(*check["parent_priors"], cfg.prior_method)
                self.used.add(pair)
                self.creations += 1
                self.events.append({**check, **metadata, "joint_id": self.creations,
                                    "prior": self.joint_prior, "prior_method": cfg.prior_method,
                                    "start_position": self.position, "history_tokens": len(self.history)})
        winner, token = baseline_winner, baseline_token
        joint_best = None
        if self.joint is not None:
            joint_logits = self.joint.logits()
            if self.joint_beta is None:
                self.joint_beta = self.get_jsd(amateur, joint_logits).clamp(0, 1)
            # Preserve parent score dtype, so adding a candidate cannot perturb
            # any original score through a promotion or global normalization.
            prior = self.priors.new_tensor([self.joint_prior])
            joint_scores = pced_scores(amateur, joint_logits, self.joint_beta, prior, cfg.gamma)
            value, index = joint_scores[0].max(-1)
            joint_best = {"token": int(index), "score": float(value)}
            if bool(value > best_scores[baseline_winner]):
                winner, token = len(self.priors), int(index)
        eos = token in self.eos
        self.trace.append({"position": self.position, "phase": phase, "token": token,
                           "winner_kind": "joint" if winner == len(self.priors) else "independent",
                           "winner_rank": winner+1 if winner < len(self.priors) else None,
                           "active_joint_id": self.creations if self.joint is not None else None,
                           "baseline_token_same_prefix": baseline_token,
                           "baseline_expert_rank": baseline_winner+1,
                           "changed_token_same_prefix": token != baseline_token,
                           "baseline_best_score": float(best_scores[baseline_winner]),
                           "joint_best": joint_best, "eos": eos})
        if not eos:
            self.force([token])
        self.position += 1
        if self.joint is not None:
            self.joint_used += 1
            if self.joint_used >= cfg.lifetime:
                self.retire("lifetime")
        return token, eos


def make_joint_builder(model, tokenizer, row, prefix_function, base_pad_length):
    """Build an actual two-document context, never merge independent KV caches."""
    def build(pair, history):
        a, b = pair  # ascending original retrieval rank, not score-dependent order
        prefix = prefix_function(tokenizer, row["question"], [row["texts"][a] + "\n\n" + row["texts"][b]])[0][1]
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        pad = max(0, base_pad_length-len(ids))
        full = [tokenizer.pad_token_id]*pad + ids + row["suffix_ids"] + history
        if len(full) > model.config.max_position_embeddings:
            raise RuntimeError("Joint context overflow: no truncation/no independent-only fallback")
        input_ids = torch.tensor([full], device=model.device, dtype=torch.long)
        mask = torch.ones_like(input_ids)
        mask[:, :pad] = 0
        return CachedStream(model, None, input_ids, mask), {
            "pair_pids": [row["scores"]["pids"][a], row["scores"]["pids"][b]],
            "context_tokens": len(full), "left_padding": pad}
    return build


@torch.no_grad()
def generate(stream, tokenizer, *, reasoning_tokens=64, answer_tokens=32):
    result = {}
    for phase, budget in (("reasoning", reasoning_tokens), ("answer", answer_tokens)):
        if phase == "answer":
            stream.force(tokenizer.encode(ANSWER_MARKER, add_special_tokens=False))
        ids, status = [], "token_budget"
        for _ in range(budget):
            token, eos = stream.step(phase)
            if eos:
                status = "eos"
                break
            ids.append(token)
        result[phase+"_ids"] = ids
        result[phase+"_text"] = tokenizer.decode(ids, skip_special_tokens=True).strip()
        result[phase+"_status"] = status
    stream.retire("generation_complete")
    result.update(prediction=result["answer_text"], empty_answer=not bool(result["answer_text"]),
                  trace=stream.trace, joint_events=stream.events, joint_checks=stream.checks,
                  shared_history_ids=stream.history, config=asdict(stream.config))
    return result

"""Output-head PASS gating on a shared PCED token history.

No gold labels are accepted here. The separate PASS head and its denominator
match train_pass_lora.pass_logprob / eval_pass_lora exactly. The original
reserved-token row remains in that denominator for checkpoint compatibility,
but neither it nor the learned PASS action is broadcast as generated text.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ARMS = ("no_rag", "all", "pced", "pced_lora", "pass_pced", "pass_pced_safe")
ADAPTED = frozenset(("pced_lora", "pass_pced", "pass_pced_safe"))
GATED = frozenset(("pass_pced", "pass_pced_safe"))
PASS_TOKEN = "<|reserved_special_token_10|>"
MODES = {0: "plain", 1: "pced", 2: "all_pass_pced", 3: "single_expert",
         4: "active_pced", 5: "all_pass_amateur"}


def initial_beta(logits):
    """The existing MCQ-style CoT decoder's first-step mean JSD, one per question."""
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    mix = (.5 * (p[:, :1] + p[:, 1:])).clamp_min(1e-12).log()
    return (.5 * ((p[:, 1:] * (logp[:, 1:] - mix)).sum(-1)
                  + (p[:, :1] * (logp[:, :1] - mix)).sum(-1))).mean(-1)


def pass_probabilities(hidden, logits, weight, bias):
    """Identical to the learned checkpoint's V+1 normalisation, in FP32."""
    zpass = hidden.float() @ weight.float() + bias.float()
    lse = logits.float().logsumexp(-1)
    denominator = torch.logaddexp(lse, zpass)
    return zpass, zpass - denominator, lse - denominator


def token_first_max(scores):
    values, experts = scores.max(dim=0)
    token = values.argmax()
    return token, experts[token] + 1


def choose(logits, beta, log_prior, log_p, log_not, *, gated, threshold,
           gamma=2.5, penalty=1.0, fallback="pced", forbidden=()):
    """One question, streams [base amateur, document experts].

    One active expert -> its ordinary greedy token. Two or more -> PCED
    competition plus penalty*log(1-p_pass). None -> ungated adapted PCED by
    default, or the frozen amateur in an explicitly separate experiment.
    """
    z = logits.float()
    base = (1 + beta) * z[1:] - beta * z[:1] + gamma * log_prior[:, None]
    base[:, list(forbidden)] = -torch.inf
    counter_token, counter_winner = token_first_max(base)
    active = torch.ones(z.shape[0] - 1, dtype=torch.bool, device=z.device)
    token, winner, mode = counter_token, counter_winner, 1
    effective = base
    if gated:
        active = log_p.exp() < threshold
        count = int(active.sum())
        if count == 0:
            mode = 2
            if fallback == "amateur":
                ordinary = z[0].clone()
                ordinary[list(forbidden)] = -torch.inf
                token, winner, mode = ordinary.argmax(), token.new_zeros(()), 5
        elif count == 1:
            winner = active.nonzero()[0, 0] + 1
            ordinary = z[winner].clone()
            ordinary[list(forbidden)] = -torch.inf
            token, mode = ordinary.argmax(), 3
        else:
            effective = base + penalty * log_not[:, None]
            effective = effective.masked_fill(~active[:, None], -torch.inf)
            token, winner = token_first_max(effective)
            mode = 4
    no_prior = base - gamma * log_prior[:, None]
    no_contrast = z[1:].clone() + gamma * log_prior[:, None]
    no_contrast[:, list(forbidden)] = -torch.inf
    np_token, np_winner = token_first_max(no_prior)
    nc_token, nc_winner = token_first_max(no_contrast)
    return {"token": int(token), "winner": int(winner), "mode": mode,
            "active": active, "base_scores": base, "effective_scores": effective,
            "counter_token": int(counter_token), "counter_winner": int(counter_winner),
            "no_prior_token": int(np_token), "no_prior_winner": int(np_winner),
            "no_contrast_token": int(nc_token), "no_contrast_winner": int(nc_winner)}


def _cpu(tensor, dtype=None):
    arr = tensor.detach().cpu().numpy()
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def trace_frame(hidden, logits, decision, *, log_p=None, log_not=None,
                pass_logit=None, top_n=8, forbidden=()):
    """Exact candidate logits plus BF16 hidden states for full-vocabulary replay.

    Candidate union includes every stream's raw top-N, every expert's PCED
    top-N, selected token, and same-prefix counterfactual winners. No vocabulary
    distribution is accumulated across steps in GPU or host RAM.
    """
    z = logits.float()
    allowed = z.clone()
    allowed[:, list(forbidden)] = -torch.inf
    n = min(top_n, z.shape[-1] - len(forbidden))
    raw_values, raw_ids = allowed.topk(n, dim=-1)
    candidates = [raw_ids.flatten(), torch.tensor([decision["token"]], device=z.device)]
    if "base_scores" in decision:
        score_values, score_ids = decision["base_scores"].topk(n, dim=-1)
        candidates += [score_ids.flatten(), torch.tensor(
            [decision[k] for k in ("counter_token", "no_prior_token", "no_contrast_token")], device=z.device)]
    else:
        score_values, score_ids = raw_values[:0], raw_ids[:0]
    ids = torch.unique(torch.cat(candidates), sorted=True)
    selected = z[:, decision["token"]]
    frames = {
        "token_ids": np.int32(decision["token"]), "winner": np.int16(decision["winner"]),
        "mode": np.int8(decision["mode"]),
        "hidden_bf16": _cpu(hidden.to(torch.bfloat16).view(torch.uint16)),
        "candidate_ids": _cpu(ids, np.int32), "candidate_logits": _cpu(z[:, ids].T),
        "raw_top_ids": _cpu(raw_ids, np.int32), "raw_top_logits": _cpu(raw_values),
        "pced_top_ids": _cpu(score_ids, np.int32), "pced_top_scores": _cpu(score_values),
        "logsumexp": _cpu(z.logsumexp(-1)),
        "selected_token_logits": _cpu(selected),
        "selected_token_rank": _cpu((allowed > selected[:, None]).sum(-1) + 1, np.int32),
    }
    if log_p is not None:
        frames.update(pass_logit=_cpu(pass_logit), pass_log_probability=_cpu(log_p),
                      log_not_pass=_cpu(log_not), active=_cpu(decision["active"]))
        for name in ("counter_token", "counter_winner", "no_prior_token", "no_prior_winner",
                     "no_contrast_token", "no_contrast_winner"):
            frames[name] = np.int32(decision[name])
    return frames


def save_trace(path, frames, *, beta, log_prior):
    if not frames:
        raise ValueError("A completed generation must have at least one decision")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: np.stack([f[key] for f in frames]) for key in frames[0]
            if key not in ("candidate_ids", "candidate_logits")}
    sizes = [len(f["candidate_ids"]) for f in frames]
    data["candidate_offsets"] = np.cumsum([0] + sizes, dtype=np.int64)
    data["candidate_ids"] = np.concatenate([f["candidate_ids"] for f in frames])
    data["candidate_logits"] = np.concatenate([f["candidate_logits"] for f in frames])
    data["beta"] = np.asarray(beta, dtype=np.float32)
    data["log_prior"] = np.asarray(log_prior, dtype=np.float32)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@dataclass
class Generation:
    ids: list
    finish: str
    frames: list
    beta: float = 0.0


class StreamBatch:
    """An independent causal cache; PASS experts always consume the shared token."""
    def __init__(self, engine, prompts, adapted):
        self.engine, self.adapted = engine, adapted
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), engine.pad, device=engine.device, dtype=torch.long)
        self.mask = torch.zeros_like(ids)
        for j, prompt in enumerate(prompts):
            ids[j, -len(prompt):] = torch.tensor(prompt, device=engine.device)
            self.mask[j, -len(prompt):] = 1
        self.cache = None
        self._forward(ids, (self.mask.cumsum(-1) - 1).clamp_min(0))

    def _forward(self, ids, positions):
        with (nullcontext() if self.adapted else self.engine.model.disable_adapter()):
            out = self.engine.inner(input_ids=ids, attention_mask=self.mask, position_ids=positions,
                                    past_key_values=self.cache, use_cache=True, return_dict=True)
            self.hidden = out.last_hidden_state[:, -1, :].contiguous()
            self.logits = self.engine.head(self.hidden).float()
            self.cache = out.past_key_values
        if not bool(torch.isfinite(self.logits).all()):
            raise RuntimeError("Nonfinite ordinary logits")

    def advance(self, tokens):
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.engine.device)[:, None]
        self.mask = torch.cat((self.mask, self.mask.new_ones((len(tokens), 1))), dim=1)
        self._forward(ids, self.mask.sum(-1, keepdim=True) - 1)


class Engine:
    def __init__(self, model, tokenizer, pass_head):
        self.model = model.eval()
        self.inner = model.base_model.model.model
        self.head = model.base_model.model.lm_head
        self.device = next(model.parameters()).device
        self.pad = tokenizer.pad_token_id
        self.pass_id = int(pass_head["pass_id"])
        self.weight = pass_head["w"].to(self.device, torch.float32)
        self.bias = pass_head["b"].to(self.device, torch.float32)
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if tokenizer.eos_token_id is not None:
            self.eos.add(tokenizer.eos_token_id)
        if self.pass_id in self.eos:
            raise ValueError("PASS cannot be EOS")

    @torch.inference_mode()
    def plain(self, prompts, cap, *, top_n=8, heartbeat=None):
        streams = StreamBatch(self, prompts, False)
        out = [Generation([], "token_budget", []) for _ in prompts]
        done = [False] * len(prompts)
        for step in range(cap):
            allowed = streams.logits.clone()
            allowed[:, self.pass_id] = -torch.inf
            tokens = allowed.argmax(-1).tolist()
            for j, token in enumerate(tokens):
                if done[j]:
                    continue
                decision = {"token": token, "winner": 0, "mode": 0}
                out[j].frames.append(trace_frame(streams.hidden[j:j+1], streams.logits[j:j+1], decision,
                                                 top_n=top_n, forbidden=(self.pass_id,)))
                if token in self.eos:
                    out[j].finish, done[j] = "eos", True
                else:
                    out[j].ids.append(token)
            if heartbeat:
                heartbeat(step + 1, sum(done))
            if all(done) or step + 1 == cap:
                break
            streams.advance(tokens)
        return out

    @torch.inference_mode()
    def competitive(self, prompt_groups, log_priors, arm, cap, *, threshold=.7, gamma=2.5,
                    penalty=1., fallback="pced", top_n=8, heartbeat=None):
        if arm not in ARMS[2:]:
            raise ValueError(arm)
        b, streams = len(prompt_groups), len(prompt_groups[0])
        if streams < 2 or any(len(p) != streams for p in prompt_groups):
            raise ValueError("Expected one amateur and fixed K experts per question")
        adapted = arm in ADAPTED
        if adapted:
            amateur = StreamBatch(self, [p[0] for p in prompt_groups], False)
            experts = StreamBatch(self, [v for p in prompt_groups for v in p[1:]], True)
        else:
            combined = StreamBatch(self, [v for p in prompt_groups for v in p], False)
        prior = torch.as_tensor(log_priors, dtype=torch.float32, device=self.device)
        results, done, beta = [Generation([], "token_budget", []) for _ in range(b)], [False] * b, None
        for step in range(cap):
            if adapted:
                hidden = torch.cat((amateur.hidden[:, None], experts.hidden.view(b, streams-1, -1)), dim=1)
                logits = torch.cat((amateur.logits[:, None], experts.logits.view(b, streams-1, -1)), dim=1)
            else:
                hidden = combined.hidden.view(b, streams, -1)
                logits = combined.logits.view(b, streams, -1)
            if beta is None:
                beta = initial_beta(logits)
                for j in range(b):
                    results[j].beta = float(beta[j])
            zp, lp, ln = pass_probabilities(hidden[:, 1:], logits[:, 1:], self.weight, self.bias)
            tokens = []
            for j in range(b):
                if done[j]:
                    tokens.append(min(self.eos))
                    continue
                decision = choose(logits[j], beta[j], prior[j], lp[j], ln[j], gated=arm in GATED,
                                  threshold=threshold, gamma=gamma, penalty=penalty, fallback=fallback,
                                  forbidden=(self.pass_id,))
                token = decision["token"]
                results[j].frames.append(trace_frame(hidden[j], logits[j], decision, log_p=lp[j],
                    log_not=ln[j], pass_logit=zp[j], top_n=top_n, forbidden=(self.pass_id,)))
                tokens.append(token)
                if token in self.eos:
                    results[j].finish, done[j] = "eos", True
                else:
                    results[j].ids.append(token)
            if heartbeat:
                heartbeat(step + 1, sum(done))
            if all(done) or step + 1 == cap:
                break
            if adapted:
                amateur.advance(tokens)
                experts.advance([t for t in tokens for _ in range(streams - 1)])
            else:
                combined.advance([t for t in tokens for _ in range(streams)])
        return results

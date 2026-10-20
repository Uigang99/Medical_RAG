"""Token-level, pair-conditioned MedCPT memory for a frozen causal Llama.

The native self-attention is never replaced. An additional residual is written
only at evidence proposition tokens, after a fixed Llama layer. Oracle actions
scale per-directed-pair VALUE contributions, not native attention logits.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def pair_encoding(tokenizer, question, left, right, limit=512):
    """No choices/answer, no truncation; retain every endpoint token separately."""
    evidence = left + "\n\n" + right
    encoded = tokenizer(question, evidence, return_offsets_mapping=True,
                        truncation=False, add_special_tokens=True)
    if len(encoded["input_ids"]) > limit:
        raise ValueError(f"MedCPT pair has {len(encoded['input_ids'])} tokens > {limit}")
    seq = encoded.sequence_ids()
    offsets = encoded.pop("offset_mapping")
    a = [i for i, ((s, e), sid) in enumerate(zip(offsets, seq))
         if sid == 1 and e > s and s < len(left) and e > 0]
    b = [i for i, ((s, e), sid) in enumerate(zip(offsets, seq))
         if sid == 1 and e > s and e > len(left) + 2]
    if not a or not b or set(a) & set(b):
        raise ValueError("Missing/overlapping MedCPT endpoint token spans")
    return dict(encoded), a, b


class PairMemoryBridge(nn.Module):
    """Shared trainable projections, with token-wise (not pooled) cross-attention."""

    def __init__(self, hidden_size=4096, memory_size=768, width=256, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.width = heads, width
        self.q = nn.Linear(hidden_size, width, bias=False)
        self.k = nn.Linear(memory_size, width, bias=False)
        self.v = nn.Linear(memory_size, width, bias=False)
        self.o = nn.Linear(width, hidden_size, bias=False)
        # Exact initial no-op, no answer-bearing vector or trainable Llama weight.
        nn.init.zeros_(self.o.weight)

    def messages(self, hidden, spans, memory):
        """Return (receiver, action index, tokenwise residual) for each direction.

        All neighbors compete in the SAME softmax for a receiver, plus a zero
        null value. Splitting the value sum preserves ordinary cross-attention
        exactly when all action multipliers are one. The oracle may attenuate
        or amplify these fixed contributions without renormalizing the softmax.
        """
        grouped = defaultdict(list)
        for i, item in enumerate(memory):
            grouped[int(item["receiver"])].append((i, item["states"]))
        result = []
        d = self.width // self.heads
        for receiver, entries in grouped.items():
            start, end = spans[receiver]
            query = F.layer_norm(hidden[0, start:end].float(), (hidden.shape[-1],))
            q = self.q(query).view(-1, self.heads, d).transpose(0, 1)
            lengths = [len(states) for _, states in entries]
            states = torch.cat([states.to(hidden.device, dtype=torch.float32)
                                for _, states in entries])
            states = F.layer_norm(states, (states.shape[-1],))
            k = self.k(states).view(-1, self.heads, d).transpose(0, 1)
            v = self.v(states).view(-1, self.heads, d).transpose(0, 1)
            # Null memory: using no added information is always possible.
            k = torch.cat([k, k.new_zeros(self.heads, 1, d)], dim=1)
            # Chunk queries to bound the extra attention matrix on long claims.
            chunks = [[] for _ in entries]
            for offset in range(0, q.shape[1], 32):
                weights = (q[:, offset:offset + 32] @ k.transpose(-1, -2)
                           / math.sqrt(d)).softmax(-1)
                pos = 0
                for j, length in enumerate(lengths):
                    value = weights[..., pos:pos + length] @ v[:, pos:pos + length]
                    value = value.transpose(0, 1).reshape(-1, self.width)
                    chunks[j].append(self.o(value))
                    pos += length
            for (index, _), pieces in zip(entries, chunks):
                result.append((receiver, index, torch.cat(pieces, dim=0)))
        return result


def apply_messages(hidden, spans, messages, actions):
    """Differentiable residual; non-proposition tokens remain bitwise unchanged."""
    grouped = defaultdict(list)
    for receiver, index, value in messages:
        grouped[receiver].append(actions[index] * value)
    pieces, pos = [], 0
    for receiver, (start, end) in enumerate(spans):
        pieces.append(hidden[:, pos:start])
        original = hidden[:, start:end]
        if receiver in grouped:
            update = torch.stack(grouped[receiver]).sum(0).unsqueeze(0)
            original = original + update.to(original.dtype)
        pieces.append(original)
        pos = end
    pieces.append(hidden[:, pos:])
    return torch.cat(pieces, dim=1)


class FrozenLlamaMemory:
    """Cache the unchanged lower layers; backpropagate through frozen upper ones."""

    def __init__(self, llm, cut=12, checkpoint_upper=False):
        self.llm, self.cut, self.checkpoint_upper = llm, cut, checkpoint_upper
        if not 0 < cut < len(llm.model.layers):
            raise ValueError("Injection must be strictly inside Llama")
        llm.eval().requires_grad_(False)

    @property
    def device(self):
        return self.llm.model.embed_tokens.weight.device

    def positions(self, hidden):
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
        rotary = self.llm.model.rotary_emb(hidden, position_ids=positions)
        return positions, rotary

    @torch.no_grad()
    def prefix(self, row):
        ids = torch.tensor([row["input_ids"]], device=self.device)
        hidden = self.llm.model.embed_tokens(ids)
        positions, rotary = self.positions(hidden)
        for layer in self.llm.model.layers[:self.cut]:
            hidden = layer(hidden, position_ids=positions,
                           position_embeddings=rotary, use_cache=False)
        return hidden.detach()

    def logits(self, row, hidden):
        positions, rotary = self.positions(hidden)
        for layer in self.llm.model.layers[self.cut:]:
            def forward(x, block=layer):
                return block(x, position_ids=positions,
                             position_embeddings=rotary, use_cache=False)
            if self.checkpoint_upper and torch.is_grad_enabled() and hidden.requires_grad:
                hidden = checkpoint(forward, hidden, use_reentrant=False)
            else:
                hidden = forward(hidden)
        last = self.llm.model.norm(hidden[:, -1])
        choices = torch.tensor(row["choice_token_ids"], device=last.device)
        return F.linear(last.float(), self.llm.lm_head.weight[choices].float())[0]


def answer_loss(logits, gold):
    return F.cross_entropy(logits[None], torch.tensor([gold], device=logits.device))


def score_record(logits, gold):
    x = logits.detach().float().cpu()
    other = x.clone()
    other[gold] = -torch.inf
    pred = int(x.argmax())
    return {"logits": x.tolist(), "ce": float(answer_loss(x, gold)),
            "p_gold": float(x.softmax(-1)[gold]), "margin": float(x[gold] - other.max()),
            "prediction": pred, "gold": int(gold), "correct": bool(pred == gold)}

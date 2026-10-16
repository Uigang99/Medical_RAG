"""Input-conditioned, source-linked attention biases on unchanged evidence tokens.

The frozen encoders describe text. This small network, not a second generative
LLM, learns how to use those descriptions. No learned entity-ID embeddings.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

import torch
from torch import nn
from torch.nn import functional as F
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv

from medrag.generation.kg_structure_attention import KINDS, token_layout
from medrag.training.graph_input_data import SYSTEM

NAME = "rag2_dynamic_kg_attention_v1"


def source_layout(tokenizer, row):
    """Retain the static experiment's exact positions, adding link provenance.

Options are used ONLY to verify the unchanged target prompt. Controller feature
inputs below contain the question stem, source text and KG, never options/gold.
"""
    original = token_layout(tokenizer, row)
    question = row["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(row["options"].items()))
    user = "Evidence:\n" + "\n".join(p["claim"] for p in row["propositions"]) + "\n\nQuestion:\n" + question
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    cursor = rendered.index(user) + len("Evidence:\n")
    props, starts = {}, {}
    for i, p in enumerate(row["propositions"]):
        props[p["proposition_id"]] = i
        starts[p["proposition_id"]] = cursor
        cursor += len(p["claim"]) + 1
    node_ids, node_texts, nodes = [], [], {}
    edges, seen_edges, mentions, bound = [], set(), {}, []

    def endpoint(e, role):
        entity, surface = e[role + "_entity_id"], e[role + "_surface"]
        if entity not in nodes:
            nodes[entity] = len(node_ids); node_ids.append(entity); node_texts.append(surface)
        p = props[e["proposition_id"]]
        claim = row["propositions"][p]["claim"]
        matches = list(re.finditer(re.escape(surface), claim)) if surface else []
        matches = [m for m in matches if
            (not surface[0].isalnum() or m.start() == 0 or not claim[m.start()-1].isalnum()) and
            (not surface[-1].isalnum() or m.end() == len(claim) or not claim[m.end()].isalnum())]
        if len(matches) != 1: return nodes[entity], None
        a, b = (starts[e["proposition_id"]] + matches[0].start(),
                starts[e["proposition_id"]] + matches[0].end())
        ts = tuple(i for i, (x, y) in enumerate(offsets) if y > a and x < b and y > x)
        if not ts: return nodes[entity], None
        key = (p, a, b, nodes[entity])
        mentions[key] = ts
        return nodes[entity], key

    for e in row["edges"]:
        signature = tuple(e[k] for k in ("proposition_id", "subject_entity_id", "object_entity_id",
                                         "subject_surface", "object_surface", "relation_surface"))
        if signature in seen_edges: continue
        seen_edges.add(signature)
        s, sm = endpoint(e, "subject"); o, om = endpoint(e, "object")
        edges.append([s, o, props[e["proposition_id"]], e["relation_surface"]])
        bound.append((sm, om, len(edges)-1))
    owners = defaultdict(set)
    for m in mentions: owners[m[:-1]].add(m[-1])
    by_entity = defaultdict(list)
    for m in mentions:
        if len(owners[m[:-1]]) == 1: by_entity[m[-1]].append(m)
    links, link_ids, pair_links = [], {}, defaultdict(set)

    def add(kind, read, send, edge):
        rt, st = mentions[read], mentions[send]
        valid = [(q, k) for q in rt for k in st if k <= q]
        if not valid: return
        link = (kind, read[-1], send[-1], read[0], send[0], edge)
        if link not in link_ids:
            link_ids[link] = len(links); links.append(link)
        for q, k in valid: pair_links[(kind, q, k)].add(link_ids[link])

    for occurrences in by_entity.values():
        for read in occurrences:
            for send in occurrences:
                if mentions[read] != mentions[send]: add(0, read, send, -1)
    for sm, om, edge in bound:
        if sm is not None and om is not None:
            add(1, om, sm, edge); add(2, sm, om, edge)
    pairs = sorted(pair_links)
    if pairs != [tuple(p) for p in original["pairs"]]:
        raise RuntimeError(f"Dynamic/static attention positions differ: {row['sample_id']}")
    return {**original, "question_text": row["question"], "node_texts": node_texts,
            "proposition_texts": [p["claim"] for p in row["propositions"]],
            "edges": edges, "links": links,
            "pair_links": [(i, j) for i, pair in enumerate(pairs) for j in sorted(pair_links[pair])]}


class DynamicBias(nn.Module):
    def __init__(self, layers, heads, hidden=128, rounds=2, variant="graph",
                 input_dim=768, max_bias=2., query_chunk=128):
        super().__init__()
        if variant not in ("graph", "local"): raise ValueError(variant)
        self.layers, self.heads, self.hidden, self.rounds = layers, heads, hidden, rounds
        self.variant, self.max_bias, self.query_chunk = variant, float(max_bias), query_chunk
        self.query_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden))
        self.article_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden))
        self.query_pool = nn.Linear(hidden, 1, bias=False)
        self.role = nn.Embedding(2, hidden)
        self.kind = nn.Embedding(len(KINDS), hidden)
        self.messages = nn.ModuleList(nn.Sequential(nn.Linear(5*hidden, hidden), nn.GELU(),
                                                   nn.Linear(hidden, hidden+1)) for _ in range(rounds))
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(3*hidden, hidden), nn.GELU(),
                                                  nn.LayerNorm(hidden)) for _ in range(rounds))
        self.readout = nn.Sequential(nn.Linear(7*hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.output = nn.Linear(hidden, layers*heads)
        # Only the final projection is zero; upstream representations remain live.
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.enabled = True
        self.layouts, self.biases = [], []
        self.bank = None

    def bind_bank(self, bank):
        self.bank = bank

    def link_biases(self, spec):
        device = self.output.weight.device
        qraw, araw, groups = self.bank.for_layout(spec, device)
        qh = self.query_projection(qraw)
        q = (qh * self.query_pool(qh).softmax(0)).sum(0)
        ah = self.article_projection(araw)
        # Question-conditioned pooling retains every encoder window in the cache.
        score = (ah * q).sum(-1) / math.sqrt(self.hidden)
        count = len(spec["node_texts"]) + len(spec["proposition_texts"]) + len(spec["edges"])
        maximum = score.new_full((count,), -torch.inf).scatter_reduce(0, groups, score, reduce="amax")
        exp = (score - maximum[groups]).exp()
        denom = exp.new_zeros(count).index_add(0, groups, exp)
        pooled = ah.new_zeros(count, self.hidden).index_add(0, groups, ah * (exp / denom[groups])[:, None])
        n, p = len(spec["node_texts"]), len(spec["proposition_texts"])
        h, propositions, relations = pooled[:n], pooled[n:n+p], pooled[n+p:]
        edges = torch.tensor([e[:3] for e in spec["edges"]], dtype=torch.long, device=device).reshape(-1, 3)
        if len(edges):
            s, o, prop = edges.unbind(1)
            send, receive = torch.cat((s, o)), torch.cat((o, s))
            context = (relations + propositions[prop]).repeat(2, 1)
            roles = self.role(torch.cat((torch.zeros_like(s), torch.ones_like(s))))
            degree = torch.bincount(receive, minlength=n).clamp_min(1)[:, None]
        for message, update in zip(self.messages, self.updates):
            aggregate = h.new_zeros(h.shape)
            if len(edges):
                # Local control keeps incident semantics/compute but never passes
                # a neighbor's state. It is NOT a graph-free/retrieval control.
                source = h[send] if self.variant == "graph" else h[receive]
                msg = message(torch.cat((source, h[receive], context, roles, q.expand(len(send), -1)), -1))
                aggregate = aggregate.index_add(0, receive, msg[:, :-1] * msg[:, -1:].sigmoid()) / degree
            h = h + update(torch.cat((h, aggregate, q.expand(n, -1)), -1))
        links = torch.tensor(spec["links"], dtype=torch.long, device=device).reshape(-1, 6)
        if not len(links): return h.new_zeros((0, self.layers, self.heads))
        kind, read, send, rp, sp, edge = links.unbind(1)
        relation = relations[edge.clamp_min(0)] * (edge >= 0)[:, None]
        z = self.readout(torch.cat((q.expand(len(links), -1), h[read], h[send],
                                    propositions[rp], propositions[sp], relation, self.kind(kind)), -1))
        return self.max_bias * self.output(z).reshape(-1, self.layers, self.heads).tanh()

    def set_layouts(self, layouts, shifts=None):
        if self.bank is None: raise RuntimeError("Frozen semantic feature bank not bound")
        shifts = [0]*len(layouts) if shifts is None else shifts
        self.layouts, self.biases = [], []
        device = self.output.weight.device
        for spec, shift in zip(layouts, shifts, strict=True):
            pairs = torch.tensor(spec["pairs"], dtype=torch.long, device=device).reshape(-1, 3)
            pairs[:, 1:] += shift
            if self.enabled and len(pairs):
                values = self.link_biases(spec)
                membership = torch.tensor(spec["pair_links"], dtype=torch.long, device=device).reshape(-1, 2)
                pi, li = membership.unbind(1)
                # Duplicate links within the SAME kind/position are averaged,
                # never summed. Distinct kinds retain the static additive rule.
                means = values.new_zeros(len(pairs), self.layers, self.heads).index_add(0, pi, values[li])
                means = means / torch.bincount(pi, minlength=len(pairs))[:, None, None]
            else:
                means = self.output.weight.new_zeros(len(pairs), self.layers, self.heads)
            self.layouts.append(pairs); self.biases.append(means)

    def clear(self):
        self.layouts, self.biases = [], []

    def attach(self, model):
        ALL_ATTENTION_FUNCTIONS.register(NAME, dynamic_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        model.config._attn_implementation = NAME
        for layer in model.model.layers:
            object.__setattr__(layer.self_attn, "_kg_dynamic", self)


def dynamic_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
    controller = module._kg_dynamic
    key, value = repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups)
    batch, heads, nq, _ = query.shape
    nk = key.shape[-2]
    if len(controller.layouts) != batch or dropout:
        raise RuntimeError("Layout batch mismatch or nonzero frozen attention dropout")
    chunks = []
    for start in range(0, nq, controller.query_chunk):
        stop = min(nq, start+controller.query_chunk)
        origin = nk-nq+start
        qabs = torch.arange(origin, nk-nq+stop, device=query.device)
        causal = torch.arange(nk, device=query.device)[None, :] <= qabs[:, None]
        mask = query.new_zeros(batch, 1, stop-start, nk).masked_fill(~causal[None, None], -torch.inf)
        if attention_mask is not None:
            source = attention_mask[..., start:stop, :nk]
            mask = mask.masked_fill(~source, -torch.inf) if source.dtype == torch.bool else mask + source
        if controller.enabled:
            overlay = query.new_zeros(batch, heads, stop-start, nk)
            for b, pairs in enumerate(controller.layouts):
                active = (pairs[:, 1] >= origin) & (pairs[:, 1] < nk-nq+stop)
                if active.any():
                    kind, q, k = pairs[active].unbind(1)
                    values = controller.biases[b][active, module.layer_idx].T.to(query.dtype)
                    for t in range(len(KINDS)):
                        selected = kind == t
                        if selected.any(): overlay[b, :, q[selected]-origin, k[selected]] += values[:, selected]
            mask = mask + overlay
        if torch.is_grad_enabled() and mask.requires_grad:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                out = F.scaled_dot_product_attention(query[:, :, start:stop], key, value,
                    attn_mask=mask, dropout_p=0., is_causal=False, scale=scaling)
        else:
            out = F.scaled_dot_product_attention(query[:, :, start:stop], key, value,
                attn_mask=mask, dropout_p=0., is_causal=False, scale=scaling)
        chunks.append(out)
    return torch.cat(chunks, 2).transpose(1, 2).contiguous(), None

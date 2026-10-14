"""Learned KG biases on EXISTING evidence tokens; no new tokens or facts.

The decoder remains causal. Graph direction names describe query/key roles,
not inverse medical predicates. Unmapped text is never removed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

import torch
from torch import nn
from torch.nn import functional as F
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv

from medrag.training.graph_input_data import SYSTEM

NAME = "rag2_kg_structure_v1"
KINDS = ("same_entity", "object_reads_subject", "subject_reads_object")


def token_layout(tokenizer, row):
    """Align exact endpoint surfaces ONLY inside their own source proposition.

    Answer labels are not read. Ambiguous surface occurrences are omitted.
    Nested spans are valid: a compound endpoint can contain another endpoint.
    Conflicting IDs at the SAME span disable only its same-entity links, not
    its independently source-grounded subject/object relation.
    """
    question = row["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(row["options"].items()))
    evidence = "\n".join(p["claim"] for p in row["propositions"])
    user = (f"Evidence:\n{evidence}\n\n" if evidence else "") + "Question:\n" + question
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != row["text_prompt_ids"]:
        raise RuntimeError(f"Text prompt identity mismatch: {row['sample_id']}")
    offsets = encoded["offset_mapping"]
    if rendered.count(user) != 1:
        raise RuntimeError("Chat template does not contain exactly one unchanged user message")
    cursor = rendered.index(user) + len("Evidence:\n")
    claims = {}
    for p in row["propositions"]:
        if p["proposition_id"] in claims:
            raise RuntimeError("Duplicate source proposition")
        claims[p["proposition_id"]] = (p["claim"], cursor)
        cursor += len(p["claim"]) + 1
    stats = Counter(edges=len(row["edges"]), endpoint_occurrences=2 * len(row["edges"]),
                    mapped_edges=0, mapped_endpoint_occurrences=0, missing_surface=0,
                    ambiguous_surface=0, missing_tokens=0)
    mentions, bound = {}, []

    def mention(edge, role):
        claim, start = claims[edge["proposition_id"]]
        surface = edge[role + "_surface"]
        entity = edge[role + "_entity_id"]
        matches = list(re.finditer(re.escape(surface), claim)) if surface else []
        # Avoid mapping e.g. "age" inside "stage" without inventing a synonym.
        matches = [m for m in matches if
                   (not surface[0].isalnum() or m.start() == 0 or not claim[m.start()-1].isalnum()) and
                   (not surface[-1].isalnum() or m.end() == len(claim) or not claim[m.end()].isalnum())]
        if len(matches) != 1:
            stats["missing_surface" if not matches else "ambiguous_surface"] += 1
            return None
        a, b = start + matches[0].start(), start + matches[0].end()
        tokens = tuple(i for i, (x, y) in enumerate(offsets) if y > a and x < b and y > x)
        if not tokens:
            stats["missing_tokens"] += 1
            return None
        key = (edge["proposition_id"], a, b, entity)
        mentions[key] = tokens
        return key

    for edge in row["edges"]:
        if edge["proposition_id"] not in claims:
            raise RuntimeError("Edge without source proposition")
        bound.append((mention(edge, "subject"), mention(edge, "object")))
    owners = defaultdict(set)
    for key in mentions:
        owners[key[:-1]].add(key[-1])
    invalid = {key for key in mentions if len(owners[key[:-1]]) > 1}
    stats["identity_ambiguous_mentions"] = len(invalid)
    pairs = [set() for _ in KINDS]
    by_entity = defaultdict(list)
    for key, ts in mentions.items():
        if key not in invalid:
            by_entity[key[-1]].append(ts)
    for occurrences in by_entity.values():
        for a in occurrences:
            for b in occurrences:
                if a != b:
                    pairs[0].update((q, k) for q in a for k in b if k <= q)
    for s, o in bound:
        good_s, good_o = s is not None, o is not None
        stats["mapped_endpoint_occurrences"] += int(good_s) + int(good_o)
        if good_s and good_o:
            stats["mapped_edges"] += 1
            pairs[1].update((q, k) for q in mentions[o] for k in mentions[s] if k <= q)
            pairs[2].update((q, k) for q in mentions[s] for k in mentions[o] if k <= q)
    # Multiple duplicate KG edges do not multiply the bias.
    packed = sorted((kind, q, k) for kind, items in enumerate(pairs) for q, k in items)
    stats["causal_token_pairs"] = len(packed)
    stats["has_structure"] = int(bool(packed))
    return {"sample_id": row["sample_id"], "pairs": packed, "statistics": dict(stats)}


class StructureBias(nn.Module):
    def __init__(self, layers, heads, max_bias=2., query_chunk=128):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(layers, heads, len(KINDS)))
        self.max_bias, self.query_chunk = float(max_bias), int(query_chunk)
        self.layouts = []
        self.enabled = True

    def set_layouts(self, layouts, shifts=None):
        shifts = shifts if shifts is not None else [0] * len(layouts)
        self.layouts = []
        for layout, shift in zip(layouts, shifts, strict=True):
            pairs = torch.tensor(layout["pairs"], dtype=torch.long, device=self.raw.device).reshape(-1, 3)
            if len(pairs):
                pairs[:, 1:] += shift
            self.layouts.append(pairs)

    def attach(self, model):
        register()
        model.config._attn_implementation = NAME
        for layer in model.model.layers:
            # Do not register one trainable controller as a child of every layer.
            object.__setattr__(layer.self_attn, "_kg_structure", self)


def structure_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
    controller = module._kg_structure
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    batch, heads, nq, _ = query.shape
    nk = key.shape[-2]
    if len(controller.layouts) != batch:
        raise RuntimeError("Attention layout batch mismatch")
    if dropout:
        raise RuntimeError("Expected frozen Llama attention dropout=0")
    weights = controller.max_bias * controller.raw[module.layer_idx].tanh()
    chunks = []
    for start in range(0, nq, controller.query_chunk):
        stop = min(nq, start + controller.query_chunk)
        qabs = torch.arange(nk - nq + start, nk - nq + stop, device=query.device)
        causal = torch.arange(nk, device=query.device)[None, :] <= qabs[:, None]
        mask = query.new_zeros(batch, 1, stop-start, nk).masked_fill(~causal[None, None], float("-inf"))
        if attention_mask is not None:
            source = attention_mask[..., start:stop, :nk]
            if source.dtype == torch.bool:
                mask = mask.masked_fill(~source, float("-inf"))
            else:
                mask = mask + source
        if controller.enabled:
            overlay = query.new_zeros(batch, heads, stop-start, nk)
            for b, entries in enumerate(controller.layouts):
                active = entries[(entries[:, 1] >= nk-nq+start) & (entries[:, 1] < nk-nq+stop)]
                if len(active):
                    kind, q, k = active.unbind(1)
                    q = q - (nk-nq+start)
                    # Add independent features; repeated edge pairs were deduplicated.
                    for t in range(len(KINDS)):
                        selected = kind == t
                        if selected.any():
                            overlay[b, :, q[selected], k[selected]] += weights[:, t, None].to(query.dtype)
            mask = mask + overlay
        # SDPA supports additive-mask gradients via the math backend. Explicit
        # math for training avoids a fused kernel silently dropping that gradient.
        if torch.is_grad_enabled() and mask.requires_grad:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                out = F.scaled_dot_product_attention(query[:, :, start:stop], key, value,
                    attn_mask=mask, dropout_p=0., is_causal=False, scale=scaling)
        else:
            out = F.scaled_dot_product_attention(query[:, :, start:stop], key, value,
                attn_mask=mask, dropout_p=0., is_causal=False, scale=scaling)
        chunks.append(out)
    return torch.cat(chunks, dim=2).transpose(1, 2).contiguous(), None


def register():
    ALL_ATTENTION_FUNCTIONS.register(NAME, structure_attention)
    # Always materialize the original causal/padding mask, including cached decode.
    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    return NAME


def answer_losses(model, controller, rows, layouts, pad_id):
    controller.set_layouts(layouts)
    device = model.get_input_embeddings().weight.device
    sequences = [r["text_prompt_ids"] + r["target_ids"] for r in rows]
    width = max(map(len, sequences))
    ids = torch.tensor([s + [pad_id]*(width-len(s)) for s in sequences], device=device)
    mask = torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in sequences], device=device)
    result = model.model(input_ids=ids, attention_mask=mask,
        position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False, return_dict=True)
    values = []
    for i, r in enumerate(rows):
        a = len(r["text_prompt_ids"])-1
        n = len(r["target_ids"])
        logits = model.lm_head(result.last_hidden_state[i, a:a+n]).float()
        values.append(F.cross_entropy(logits, ids[i, a+1:a+1+n]))
    return torch.stack(values)

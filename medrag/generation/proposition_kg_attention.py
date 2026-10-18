"""Frozen Llama with question-conditioned, whole-proposition attention biases.

This is a full-known-prompt, answer-selection wrapper, NOT a cached generation
backend. Only input proposition spans and the question STEM supply controller
features. The later known question may condition earlier evidence attention;
the underlying attention mask nevertheless remains causal. Choice/answer text
must follow the question span. No second Llama pass or learned node IDs are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


NAME = "rag2_proposition_kg_attention_v1"


def validate_layout(row: dict[str, Any]) -> None:
    """Reject ambiguous positions before any model computation."""
    n = len(row["input_ids"])
    if not n:
        raise ValueError("Empty prompt")
    qa, qb = map(int, row["question_span"])
    if not 0 <= qa < qb <= n:
        raise ValueError("Question stem must have a nonempty in-prompt span")
    previous = 0
    for a, b in row["proposition_spans"]:
        if not previous <= a < b <= qa:
            raise ValueError("Proposition spans must be ordered, disjoint, and before question stem")
        previous = b
    pairs, roles = row["pairs"], row["pair_roles"]
    if len(pairs) != len(roles):
        raise ValueError("Pair/role count mismatch")
    if len({tuple(p) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate proposition pairs must be merged before model input")
    p = len(row["proposition_spans"])
    for (receiver, sender), role in zip(pairs, roles, strict=True):
        if not 0 <= sender < receiver < p:
            raise ValueError("Only causal ordered proposition pairs receiver > sender are permitted")
        if len(role) != 4 or any(x not in (0, 1, False, True) for x in role):
            raise ValueError("Pair role must be four binary flags SS, SO, OS, OO")


class PropositionPairController(nn.Module):
    """Only this small module is trained; input features are frozen Llama states.

    The importance control has the same message passing and parameter shapes,
    but its pair readout cannot see the receiver or the receiver-specific roles.
    Consequently every pair sharing a sender receives the same bias. It is not
    a graph-free control. Shuffled topology is supplied by the immutable data.
    """

    def __init__(self, input_dim: int, hidden: int = 128, rounds: int = 2,
                 variant: str = "graph", layers: int = 4, max_bias: float = 2.):
        super().__init__()
        if variant not in ("graph", "importance", "shuffled"):
            raise ValueError(variant)
        if min(input_dim, hidden, layers) <= 0 or rounds < 0 or max_bias <= 0:
            raise ValueError("Invalid controller dimensions")
        self.input_dim, self.hidden, self.rounds = input_dim, hidden, rounds
        self.variant, self.layers, self.max_bias = variant, layers, float(max_bias)
        self.enabled = True
        self.proposition_projection = nn.Sequential(nn.Linear(2 * input_dim, hidden), nn.LayerNorm(hidden))
        self.question_projection = nn.Sequential(nn.Linear(2 * input_dim, hidden), nn.LayerNorm(hidden))
        self.role_projection = nn.Linear(4, hidden, bias=False)
        self.messages = nn.ModuleList(nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden + 1)) for _ in range(rounds))
        self.updates = nn.ModuleList(nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)) for _ in range(rounds))
        self.readout = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.output = nn.Linear(hidden, layers)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, question: torch.Tensor, propositions: torch.Tensor,
                pairs: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        # Explicit FP32 controller; do not accidentally inherit Llama BF16.
        device = self.output.weight.device
        q = self.question_projection(question.to(device=device, dtype=torch.float32))
        p = self.proposition_projection(propositions.to(device=device, dtype=torch.float32))
        pairs = pairs.to(device=device, dtype=torch.long).reshape(-1, 2)
        roles = roles.to(device=device, dtype=torch.float32).reshape(-1, 4)
        if not len(pairs):
            # Permit empty-graph batches to have a differentiable zero penalty.
            return self.output.weight.sum() * 0 + p.new_zeros((0, self.layers))
        receiver, sender = pairs.unbind(1)
        receive = torch.cat((receiver, sender))
        send = torch.cat((sender, receiver))
        # Reverse endpoint roles when sending the graph message the other way.
        edge_roles = self.role_projection(torch.cat((roles, roles[:, [0, 2, 1, 3]])))
        degree = torch.bincount(receive, minlength=len(p)).clamp_min(1)[:, None]
        h = p
        for message, update in zip(self.messages, self.updates, strict=True):
            msg = message(torch.cat((h[send], h[receive], edge_roles, q.expand(len(send), -1)), -1))
            aggregate = h.new_zeros(h.shape).index_add(
                0, receive, msg[:, :-1] * msg[:, -1:].sigmoid()) / degree
            h = h + update(torch.cat((h, aggregate, q.expand(len(h), -1)), -1))
        if self.variant == "importance":
            pr, hr, r = (p.new_zeros((len(pairs), self.hidden)) for _ in range(3))
        else:
            pr, hr, r = p[receiver], h[receiver], self.role_projection(roles)
        z = self.readout(torch.cat((q.expand(len(pairs), -1), pr, p[sender], hr, h[sender], r), -1))
        return self.max_bias * self.output(z).tanh()


@dataclass(frozen=True)
class _AttentionContext:
    lengths: torch.Tensor
    token_propositions: tuple[torch.Tensor, ...]
    matrices: tuple[torch.Tensor, ...] | None
    query_chunk: int
    reader_spans: tuple[tuple[tuple[int, int], ...], ...] = ()


@dataclass
class PropositionAttentionOutput:
    last_hidden: torch.Tensor
    bias_penalty: torch.Tensor
    biases: tuple[torch.Tensor, ...]
    diagnostics: dict[str, Any]


def block_bias(context: _AttentionContext, start: int, stop: int, dtype: torch.dtype) -> torch.Tensor:
    """[batch, 1, query-chunk, key-length]; never allocates a head-sized overlay."""
    if context.matrices is None:
        raise ValueError("No proposition matrices in this attention context")
    chunks = [matrix[token_ids[start:stop, None], token_ids[None, :]].to(dtype)
              for token_ids, matrix in zip(context.token_propositions, context.matrices, strict=True)]
    return torch.stack(chunks)[:, None]


def proposition_attention(module, query, key, value, attention_mask, scaling,
                          dropout=0., proposition_context=None, zero_reference_contexts=None, **kwargs):
    """Standard SDPA outside controlled layers; exact block-bias attention inside.

    PyTorch's fused SDPA backends do not currently provide additive-mask
    gradients in all configurations. The training path uses the MATH backend
    explicitly when the mask requires gradients, with bounded query chunks.
    """
    if dropout:
        raise RuntimeError("Frozen target attention dropout must be zero")
    context = (zero_reference_contexts[module.layer_idx]
               if zero_reference_contexts is not None else proposition_context)
    gqa = query.shape[1] != key.shape[1]
    if context is None:
        # Also supports raw zero-bias baseline calls on the attached Llama.
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.,
            is_causal=attention_mask is None and query.shape[-2] > 1,
            scale=scaling, enable_gqa=gqa)
        return out.transpose(1, 2).contiguous(), None
    nq, nk = query.shape[-2], key.shape[-2]
    if nq != nk:
        raise RuntimeError("Only full-prompt prefill is supported; incremental KV decoding is not")
    valid_keys = torch.arange(nk, device=query.device)[None, :] < context.lengths[:, None]
    if context.matrices is None:
        # All rows are RIGHT padded. A real token cannot causally read a later
        # padding token, so a padding mask is unnecessary on the full-prefill
        # fast path. Padded query outputs are never pooled/scored. Leaving the
        # mask None permits Flash SDPA for the ordinary frozen Llama layers.
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None,
            dropout_p=0., is_causal=True, scale=scaling, enable_gqa=gqa)
        return out.transpose(1, 2).contiguous(), None
    from torch.nn.attention import SDPBackend, sdpa_kernel
    result = []
    keys = torch.arange(nk, device=query.device)
    for start in range(0, nq, context.query_chunk):
        stop = min(nq, start + context.query_chunk)
        causal = keys[None, :] <= torch.arange(start, stop, device=query.device)[:, None]
        allowed = causal[None, None] & valid_keys[:, None, None, :]
        controlled = not context.reader_spans or any(
            a < stop and b > start for spans in context.reader_spans for a, b in spans)
        # Question/output and other unmodifiable query chunks need no mask
        # gradient and can use a fused backend even during controller training.
        mask = (block_bias(context, start, stop, query.dtype).masked_fill(~allowed, -torch.inf)
                if controlled else allowed)
        call = dict(attn_mask=mask, dropout_p=0., is_causal=False, scale=scaling, enable_gqa=gqa)
        if torch.is_grad_enabled() and mask.requires_grad:
            with sdpa_kernel(SDPBackend.MATH):
                out = F.scaled_dot_product_attention(query[:, :, start:stop], key, value, **call)
        else:
            out = F.scaled_dot_product_attention(query[:, :, start:stop], key, value, **call)
        result.append(out)
    return torch.cat(result, 2).transpose(1, 2).contiguous(), None


class PropositionAttentionModel(nn.Module):
    """One-pass, frozen-Llama answer-selection model with immutable batch state.

    ``forward(rows, pad_id)`` returns the last prompt-token hidden state, not
    vocabulary logits. The trainer applies the frozen lm_head rows for each
    allowed answer-label token, preventing an unnecessary [B,L,vocab] tensor.
    Layer indices in ``attention_layers`` are zero-based; ``cut_layer`` is the
    number of ordinary lower layers. There is no global cache to clear before
    checkpoint recomputation or after a forward/OOM.
    """

    def __init__(self, llm, controller: PropositionPairController, cut_layer: int = 12,
                 attention_layers=(12, 13, 14, 15), gradient_checkpointing: bool = True,
                 query_chunk: int = 128):
        super().__init__()
        self.llm, self.controller = llm, controller
        self.cut_layer, self.attention_layers = int(cut_layer), tuple(attention_layers)
        self.gradient_checkpointing, self.query_chunk = bool(gradient_checkpointing), int(query_chunk)
        n = len(llm.model.layers)
        if not 0 < cut_layer < n or not self.attention_layers or query_chunk < 1:
            raise ValueError("Invalid layer cut/attention layer/query chunk configuration")
        if len(set(self.attention_layers)) != len(self.attention_layers) or any(
                x < cut_layer or x >= n for x in self.attention_layers):
            raise ValueError("Controlled layers must be distinct and at/after the frozen feature cut")
        if controller.layers != len(self.attention_layers) or controller.input_dim != llm.config.hidden_size:
            raise ValueError("Controller and target dimensions disagree")
        if getattr(llm.config, "attention_dropout", 0.) != 0.:
            raise ValueError("Use a zero-dropout frozen Llama")
        llm.requires_grad_(False)
        llm.eval()
        # We own the layer checkpointing, avoiding double/nested model settings.
        if hasattr(llm, "gradient_checkpointing_disable"):
            llm.gradient_checkpointing_disable()
        ALL_ATTENTION_FUNCTIONS.register(NAME, proposition_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        llm.config._attn_implementation = NAME
        for layer in llm.model.layers:
            layer.self_attn.config._attn_implementation = NAME

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self

    @staticmethod
    def _pool(states: torch.Tensor, span) -> torch.Tensor:
        a, b = map(int, span)
        part = states[a:b].float()
        return torch.cat((part.mean(0), part[-1]), -1)

    @torch.no_grad()
    def zero_bias_reference(self, row: dict[str, Any]) -> torch.Tensor:
        """Native Llama decoder loop with independently built zero-bias masks.

        Match the production chunk/mask schedule, not the controller output.
        BF16 full-query and chunk-query kernels need not be bitwise identical.
        The caller separately checks native unchunked answer parity. This path
        never invokes the controller or the wrapper's manual decoder loop.
        """
        validate_layout(row)
        device = self.llm.model.embed_tokens.weight.device
        ids = torch.tensor([row["input_ids"]], device=device)
        n = len(row["input_ids"])
        spans = row["proposition_spans"]
        token_ids = torch.full((n,), len(spans), device=device, dtype=torch.long)
        for i, (start, stop) in enumerate(spans):
            token_ids[start:stop] = i
        lengths = torch.tensor([n], device=device)
        plain = _AttentionContext(lengths, (token_ids,), None, self.query_chunk)
        readers = (tuple(tuple(spans[i]) for i in sorted({p[0] for p in row["pairs"]})),)
        zero = _AttentionContext(lengths, (token_ids,),
                                 (torch.zeros((len(spans)+1, len(spans)+1), device=device),),
                                 self.query_chunk, readers)
        contexts = {i: zero if i in self.attention_layers else plain
                    for i in range(len(self.llm.model.layers))}
        return self.llm.model(input_ids=ids, use_cache=False,
                              zero_reference_contexts=contexts).last_hidden_state[:, -1]

    def forward(self, rows: list[dict[str, Any]], pad_id: int) -> PropositionAttentionOutput:
        if not rows:
            raise ValueError("Empty row batch")
        for row in rows:
            validate_layout(row)
        device = self.llm.model.embed_tokens.weight.device
        if self.controller.output.weight.device != device:
            raise ValueError("Move the FP32 controller to the target Llama device before forward")
        lengths = torch.tensor([len(row["input_ids"]) for row in rows], device=device, dtype=torch.long)
        longest = max(len(row["input_ids"]) for row in rows)
        if longest > self.llm.config.max_position_embeddings:
            raise ValueError("Input exceeds target Llama context; truncate whole propositions in data preparation")
        ids = torch.full((len(rows), longest), int(pad_id), dtype=torch.long, device=device)
        token_props = []
        for batch, row in enumerate(rows):
            ids[batch, :len(row["input_ids"])] = torch.tensor(row["input_ids"], device=device)
            positions = torch.full((longest,), len(row["proposition_spans"]), device=device, dtype=torch.long)
            for i, (a, b) in enumerate(row["proposition_spans"]):
                positions[a:b] = i
            token_props.append(positions)
        ordinary = _AttentionContext(lengths, tuple(token_props), None, self.query_chunk)
        position_ids = torch.arange(longest, device=device)[None, :]
        with torch.no_grad():
            hidden = self.llm.model.embed_tokens(ids)
            position_embeddings = self.llm.model.rotary_emb(hidden, position_ids=position_ids)
            for layer in self.llm.model.layers[:self.cut_layer]:
                hidden = layer(hidden, position_ids=position_ids, position_embeddings=position_embeddings,
                               use_cache=False, proposition_context=ordinary)
        biases, penalties = [], []
        for i, row in enumerate(rows):
            question = self._pool(hidden[i], row["question_span"])
            spans = row["proposition_spans"]
            propositions = (torch.stack([self._pool(hidden[i], span) for span in spans]) if spans
                            else hidden.new_empty((0, 2 * hidden.shape[-1]), dtype=torch.float32))
            # Guard against any ambient mixed-precision context changing the controller.
            with torch.autocast(device_type=device.type, enabled=False):
                bias = self.controller(question, propositions,
                                       torch.tensor(row["pairs"], device=device, dtype=torch.long).reshape(-1, 2),
                                       torch.tensor(row["pair_roles"], device=device, dtype=torch.float32).reshape(-1, 4))
                if not self.controller.enabled:
                    bias = bias * 0.
            biases.append(bias)
            penalties.append(bias.square().mean() if bias.numel() else bias.sum() * 0.)
        contexts = {}
        for output_index, layer_index in enumerate(self.attention_layers if self.controller.enabled else ()):
            matrices = []
            for row, bias in zip(rows, biases, strict=True):
                size = len(row["proposition_spans"]) + 1  # final index is non-proposition sentinel
                matrix = bias.new_zeros((size, size))
                if row["pairs"]:
                    pair = torch.tensor(row["pairs"], device=device, dtype=torch.long)
                    matrix = matrix.index_put((pair[:, 0], pair[:, 1]), bias[:, output_index])
                matrices.append(matrix)
            reader_spans = tuple(tuple(tuple(row["proposition_spans"][i])
                                       for i in sorted({p[0] for p in row["pairs"]})) for row in rows)
            contexts[layer_index] = _AttentionContext(
                lengths, tuple(token_props), tuple(matrices), self.query_chunk, reader_spans)
        for index in range(self.cut_layer, len(self.llm.model.layers)):
            layer = self.llm.model.layers[index]
            context = contexts.get(index, ordinary)

            def execute(value, layer=layer, context=context):
                return layer(value, position_ids=position_ids, position_embeddings=position_embeddings,
                             use_cache=False, proposition_context=context)

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(execute, hidden, use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = execute(hidden)
        # RMSNorm is token-local, so normalizing only the selected token is exact.
        last = hidden[torch.arange(len(rows), device=device), lengths - 1]
        last = self.llm.model.norm(last)
        return PropositionAttentionOutput(last, torch.stack(penalties), tuple(biases), {
            "questions": len(rows), "prompt_tokens": sum(len(row["input_ids"]) for row in rows),
            "padded_tokens": len(rows) * longest,
            "candidate_pairs": sum(len(row["pairs"]) for row in rows),
            "controlled_layers_zero_based": list(self.attention_layers),
            "feature_cut_layers": self.cut_layer,
        })

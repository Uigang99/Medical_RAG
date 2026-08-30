from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv


ATTENTION_NAME = "rag2_semantic_eager"


def suppression_bias(
    helpful_margin: float,
    strength: float,
    max_suppression_factor: float,
) -> float:
    """Map a calibrated Helpful-vs-Not-Helpful logit margin to a safe bias.

    Positive semantic evidence is deliberately left unchanged in the MVP.
    Negative evidence receives an additive attention-logit bias, capped so the
    prior attention odds cannot be reduced by more than
    ``max_suppression_factor``.
    """

    if strength < 0:
        raise ValueError("Semantic-attention strength must be non-negative")
    if max_suppression_factor < 1:
        raise ValueError("max_suppression_factor must be at least 1")
    raw = min(0.0, float(strength) * float(helpful_margin))
    return max(-math.log(float(max_suppression_factor)), raw)


def semantic_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Llama eager attention with a document-key prior on selected queries.

    ``semantic_token_bias`` stores one additive bias per absolute key token.
    ``semantic_query_mask`` marks assistant rationale/answer query positions.
    Both tensors may include reserved future positions for cached generation.
    The bias is only active from ``semantic_layer_start`` onward.
    """

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    token_bias = kwargs.get("semantic_token_bias")
    query_mask = kwargs.get("semantic_query_mask")
    layer_start = int(kwargs.get("semantic_layer_start", 0))
    layer_index = int(getattr(module, "layer_idx", 0))
    if token_bias is not None and query_mask is not None and layer_index >= layer_start:
        query_length = int(query.shape[-2])
        key_length = int(key_states.shape[-2])
        if token_bias.ndim != 2 or query_mask.ndim != 2:
            raise ValueError("Semantic attention tensors must have shape [batch, sequence]")
        if token_bias.shape[0] != query.shape[0] or query_mask.shape[0] != query.shape[0]:
            raise ValueError("Semantic attention batch size does not match the attention batch")
        if token_bias.shape[1] < key_length or query_mask.shape[1] < key_length:
            raise ValueError(
                "Semantic attention tensors are shorter than the current KV cache: "
                f"bias={token_bias.shape[1]} query_mask={query_mask.shape[1]} key={key_length}"
            )
        query_start = key_length - query_length
        active_queries = query_mask[:, query_start:key_length].to(
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        active_keys = token_bias[:, :key_length].to(
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        semantic_bias = active_queries[:, None, :, None] * active_keys[:, None, None, :]
        attn_weights = attn_weights + semantic_bias

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def register_semantic_attention() -> str:
    """Register the custom attention and matching causal-mask implementation."""

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, semantic_eager_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        ATTENTION_NAME,
        ALL_MASK_ATTENTION_FUNCTIONS["eager"],
    )
    return ATTENTION_NAME

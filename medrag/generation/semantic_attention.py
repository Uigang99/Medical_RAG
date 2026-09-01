from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv


ATTENTION_NAME = "rag2_semantic_eager"


@dataclass
class DocumentAttentionCollector:
    """Accumulate document-span attention mass without retaining attention maps.

    The collector is intentionally diagnostic.  It sums attention from marked
    assistant query tokens to each mapped document span for selected batch
    rows.  Per-layer tensors are only ``[selected_batch, documents]``; the
    original ``[batch, heads, queries, keys]`` attention maps are never kept.
    """

    document_count: int
    selected_batch_indices: tuple[int, ...] = (0,)
    collect_value_norm: bool = False
    layer_document_mass: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_document_value_norm: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_query_head_mass: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.document_count <= 0:
            raise ValueError("document_count must be positive")
        if not self.selected_batch_indices or any(index < 0 for index in self.selected_batch_indices):
            raise ValueError("selected_batch_indices must contain non-negative indices")

    def update(
        self,
        *,
        layer_index: int,
        attention_weights: torch.Tensor,
        value_states: torch.Tensor | None = None,
        token_document_ids: torch.Tensor,
        active_query_mask: torch.Tensor,
    ) -> None:
        """Accumulate one layer's post-softmax attention for selected rows."""

        if attention_weights.ndim != 4:
            raise ValueError("attention_weights must have shape [batch, heads, queries, keys]")
        batch, heads, queries, keys = attention_weights.shape
        if token_document_ids.shape != (batch, keys):
            raise ValueError("token_document_ids must align with attention keys")
        if active_query_mask.shape != (batch, queries):
            raise ValueError("active_query_mask must align with attention queries")
        if max(self.selected_batch_indices) >= batch:
            raise ValueError("selected batch index is out of range")

        rows = torch.tensor(
            self.selected_batch_indices,
            dtype=torch.long,
            device=attention_weights.device,
        )
        weights = attention_weights.index_select(0, rows).float()
        document_ids = token_document_ids.to(attention_weights.device).index_select(0, rows).long()
        query_mask = active_query_mask.to(attention_weights.device).index_select(0, rows).float()
        valid_document = (document_ids >= 0) & (document_ids < self.document_count)
        safe_document_ids = document_ids.clamp(min=0, max=self.document_count - 1)
        one_hot = torch.nn.functional.one_hot(
            safe_document_ids,
            num_classes=self.document_count,
        ).float()
        one_hot = one_hot * valid_document.unsqueeze(-1)
        # Sum over heads, active assistant queries, and mapped document keys.
        document_mass = torch.einsum("bhqk,bq,bkd->bd", weights, query_mask, one_hot)
        if value_states is not None and self.collect_value_norm:
            if value_states.shape[:3] != (batch, heads, keys):
                raise ValueError(
                    "value_states must have shape [batch, heads, keys, head_dim]"
                )
            selected_values = value_states.index_select(0, rows).float()
            # This is a diagnostic pre-output-projection proxy.  For each
            # document, sum attention-weighted value vectors over active query
            # positions and document keys, retain the head axes, then take the
            # vector norm.  Norms are accumulated across layers in
            # ``summarize``; vectors from different residual spaces are never
            # added directly.
            document_values_by_row: list[torch.Tensor] = []
            for row_index in range(weights.shape[0]):
                active_queries = torch.nonzero(
                    query_mask[row_index] > 0,
                    as_tuple=False,
                ).flatten()
                if active_queries.numel() == 0:
                    document_values_by_row.append(
                        torch.zeros(
                            (
                                self.document_count,
                                heads,
                                int(selected_values.shape[-1]),
                            ),
                            device=weights.device,
                        )
                    )
                    continue
                active_weights = weights[row_index].index_select(1, active_queries)
                active_weights = active_weights * query_mask[row_index].index_select(
                    0, active_queries
                )[None, :, None]
                document_values_by_row.append(
                    torch.einsum(
                        "hqk,kd,hkv->dhv",
                        active_weights,
                        one_hot[row_index],
                        selected_values[row_index],
                    )
                )
            document_values = torch.stack(document_values_by_row)
            self.layer_document_value_norm[int(layer_index)] = torch.linalg.vector_norm(
                document_values,
                dim=(-2, -1),
            ).detach()
        query_head_mass = query_mask.sum(dim=1) * float(heads)
        self.layer_document_mass[int(layer_index)] = document_mass.detach()
        self.layer_query_head_mass[int(layer_index)] = query_head_mass.detach()

    def summarize(self, *, layer_start: int | None = None) -> dict[str, torch.Tensor]:
        """Return relative document shares and absolute document-attention mass."""

        selected_layers = [
            layer
            for layer in sorted(self.layer_document_mass)
            if layer_start is None or layer >= layer_start
        ]
        if not selected_layers:
            batch = len(self.selected_batch_indices)
            return {
                "document_mass": torch.zeros((batch, self.document_count)),
                "document_share": torch.zeros((batch, self.document_count)),
                "document_value_norm": torch.zeros((batch, self.document_count)),
                "document_value_share": torch.zeros((batch, self.document_count)),
                "document_attention_fraction": torch.zeros(batch),
                "layers": torch.tensor([], dtype=torch.long),
            }
        mass = torch.stack([self.layer_document_mass[layer] for layer in selected_layers]).sum(dim=0)
        possible = torch.stack(
            [self.layer_query_head_mass[layer] for layer in selected_layers]
        ).sum(dim=0)
        total_document_mass = mass.sum(dim=1)
        share = mass / total_document_mass.unsqueeze(1).clamp_min(1e-30)
        fraction = total_document_mass / possible.clamp_min(1e-30)
        value_layers = [
            layer
            for layer in selected_layers
            if layer in self.layer_document_value_norm
        ]
        if value_layers:
            value_norm = torch.stack(
                [self.layer_document_value_norm[layer] for layer in value_layers]
            ).sum(dim=0)
        else:
            value_norm = torch.zeros_like(mass)
        value_share = value_norm / value_norm.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return {
            "document_mass": mass.cpu(),
            "document_share": share.cpu(),
            "document_value_norm": value_norm.cpu(),
            "document_value_share": value_share.cpu(),
            "document_attention_fraction": fraction.cpu(),
            "layers": torch.tensor(selected_layers, dtype=torch.long),
        }


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

    ``semantic_blocked_document_ids`` is a separate, exact intervention used
    by mechanism audits.  For every batch row it identifies one document whose
    mapped key/value positions are masked for *all* queries from
    ``semantic_document_block_layer_start`` onward.  Setting that start to zero
    prevents the document from exporting information through attention at any
    layer; unlike the soft semantic bias, it is not restricted to assistant
    queries and uses a true ``-inf`` mask before softmax.
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

    blocked_document_ids = kwargs.get("semantic_blocked_document_ids")
    document_ids = kwargs.get("semantic_token_document_ids")
    block_layer_start = int(kwargs.get("semantic_document_block_layer_start", 0))
    if blocked_document_ids is not None and layer_index >= block_layer_start:
        if document_ids is None:
            raise ValueError(
                "semantic_blocked_document_ids requires semantic_token_document_ids"
            )
        if blocked_document_ids.ndim != 1 or blocked_document_ids.shape[0] != query.shape[0]:
            raise ValueError("semantic_blocked_document_ids must have shape [batch]")
        key_length = int(key_states.shape[-2])
        if document_ids.ndim != 2 or document_ids.shape[0] != query.shape[0]:
            raise ValueError("semantic_token_document_ids must have shape [batch, sequence]")
        if document_ids.shape[1] < key_length:
            raise ValueError("semantic_token_document_ids is shorter than the current KV cache")
        blocked_keys = document_ids[:, :key_length].to(attn_weights.device).eq(
            blocked_document_ids.to(attn_weights.device).unsqueeze(1)
        )
        if bool(blocked_keys.all(dim=1).any()):
            raise ValueError("Exact document mask cannot block every attention key")
        attn_weights = attn_weights.masked_fill(
            blocked_keys[:, None, None, :],
            float("-inf"),
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    collector = kwargs.get("semantic_attention_collector")
    if collector is not None:
        if document_ids is None or query_mask is None:
            raise ValueError(
                "semantic attention collection requires token document IDs and query mask"
            )
        query_length = int(query.shape[-2])
        key_length = int(key_states.shape[-2])
        query_start = key_length - query_length
        active_queries = query_mask[:, query_start:key_length]
        collector.update(
            layer_index=layer_index,
            attention_weights=attn_weights,
            value_states=value_states,
            token_document_ids=document_ids[:, :key_length],
            active_query_mask=active_queries,
        )
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

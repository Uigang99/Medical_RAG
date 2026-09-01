"""Set-conditioned semantic document gates for frozen-Llama control.

The gate is an actuator, not an attribution estimator.  It converts frozen
semantic question-document features into one suppression-only attention bias
per document while allowing every document's bias to depend on the other
documents in the same set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SemanticContextGateOutput:
    residual: torch.Tensor
    prior_bias: torch.Tensor
    document_bias: torch.Tensor

    @property
    def keep_gate(self) -> torch.Tensor:
        return self.document_bias.exp()


class SemanticContextGate(nn.Module):
    """Predict permutation-equivariant document access gates.

    Inputs are available at inference: frozen semantic-encoder features and
    its Helpful-minus-Not-Helpful margin.  Gold semantic class IDs are never
    inputs; they are used only by a weak ordering loss during training.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        prior_strength: float = 0.25,
        max_suppression_factor: float = 20.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if layers <= 0 or max_suppression_factor < 1.0:
            raise ValueError("layers must be positive and suppression factor >= 1")
        self.input_dim = int(input_dim)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.margin_projection = nn.Linear(1, hidden_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("prior_strength", torch.tensor(float(prior_strength)))
        self.register_buffer(
            "max_suppression_bias",
            torch.tensor(math.log(float(max_suppression_factor))),
        )

    def forward(
        self,
        document_features: torch.Tensor,
        semantic_margin: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> SemanticContextGateOutput:
        if document_features.ndim != 3 or document_features.shape[-1] != self.input_dim:
            raise ValueError("document_features must be [batch, documents, input_dim]")
        if semantic_margin.shape != document_features.shape[:2]:
            raise ValueError("semantic_margin must be [batch, documents]")
        if document_mask.shape != semantic_margin.shape:
            raise ValueError("document_mask must match semantic_margin")
        mask = document_mask.to(device=document_features.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every question must contain at least one active document")
        margin = semantic_margin.to(document_features.dtype)
        hidden = self.feature_projection(document_features)
        hidden = hidden + self.margin_projection(margin.unsqueeze(-1))
        hidden = self.set_encoder(hidden, src_key_padding_mask=~mask)
        residual = self.residual_head(hidden).squeeze(-1)
        max_bias = self.max_suppression_bias.to(residual.dtype)
        strength = self.prior_strength.to(residual.dtype)
        prior_bias = torch.clamp(strength * margin, min=-max_bias, max=0.0)
        # The controller can restore or suppress relative to the semantic prior,
        # but the final actuator remains suppression-only.
        document_bias = torch.clamp(prior_bias + residual, min=-max_bias, max=0.0)
        residual = residual.masked_fill(~mask, 0.0)
        prior_bias = prior_bias.masked_fill(~mask, 0.0)
        document_bias = document_bias.masked_fill(~mask, 0.0)
        return SemanticContextGateOutput(
            residual=residual,
            prior_bias=prior_bias,
            document_bias=document_bias,
        )

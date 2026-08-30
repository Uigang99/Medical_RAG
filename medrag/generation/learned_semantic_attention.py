"""Trainable semantic residual gates for document-key attention.

The controller starts from a frozen semantic classifier margin and learns a
small residual from its encoder representation.  It predicts a non-positive
additive attention-logit bias per document; it does *not* treat a semantic
label as the ground-truth amount of attention.

This module deliberately contains no target-LLM wrapper.  A training loop can
freeze the target LLM, build a differentiable token bias with the helpers
below, and backpropagate a downstream answer loss through the frozen LLM into
the controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SemanticResidualAttentionOutput:
    """Outputs of :class:`SemanticResidualAttentionController`.

    ``combined_score`` is positive for evidence that should be preserved and
    negative for evidence that should be suppressed.  ``document_bias`` is an
    additive attention-logit bias in ``[-max_suppression_bias, 0]``.
    """

    residual: torch.Tensor
    prior_bias: torch.Tensor
    combined_score: torch.Tensor
    document_bias: torch.Tensor


class SemanticResidualAttentionController(nn.Module):
    """Map independent semantic document features to attention biases.

    The fixed semantic policy starts from a smooth relaxation of the
    suppression-only MVP::

        prior_bias = clamp(prior_strength * r / temperature, -B, 0)

    It converts that bias to an attention keep gate in ``[exp(-B), 1]`` and
    adds the learned residual in gate-logit space.  A zero residual therefore
    starts close to the already validated suppression-only policy,
    while a positive/negative residual can respectively restore/suppress a
    document.  The boundary epsilon preserves a useful residual gradient at
    zero and maximum suppression so answer loss can repair a confident
    semantic-classifier error.

    Thus a large positive Helpful margin produces a bias close to zero, while
    a large negative margin approaches the maximum negative suppression.  The
    last residual layer is zero-initialized.  With the default epsilon, the
    initial keep gate differs from the hard MVP boundary by at most 5% of its
    available range.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
        max_suppression_bias: float = math.log(4.0),
        prior_strength: float = 0.25,
        boundary_epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_suppression_bias <= 0:
            raise ValueError("max_suppression_bias must be positive")
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        if not 0 < boundary_epsilon < 0.5:
            raise ValueError("boundary_epsilon must be in (0, 0.5)")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        # Begin from the differentiable relaxation of the frozen semantic prior.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        self.register_buffer("temperature", torch.tensor(float(temperature)), persistent=True)
        self.register_buffer(
            "max_suppression_bias",
            torch.tensor(float(max_suppression_bias)),
            persistent=True,
        )
        self.register_buffer("prior_strength", torch.tensor(float(prior_strength)), persistent=True)
        self.register_buffer(
            "boundary_epsilon",
            torch.tensor(float(boundary_epsilon)),
            persistent=True,
        )

    def forward(
        self,
        document_features: torch.Tensor,
        semantic_margin: torch.Tensor,
        document_mask: torch.Tensor | None = None,
    ) -> SemanticResidualAttentionOutput:
        """Return one differentiable attention bias per document.

        Args:
            document_features: ``[..., documents, input_dim]`` independent
                question-document representations from the frozen semantic
                encoder.
            semantic_margin: ``[..., documents]`` Helpful-minus-Not-Helpful
                logit margins aligned with ``document_features``.
            document_mask: Optional boolean tensor shaped like
                ``semantic_margin``.  Padded documents receive zero residual
                and zero attention bias (no intervention).
        """

        if document_features.ndim < 2 or document_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"document_features must end in input_dim={self.input_dim}; "
                f"got {tuple(document_features.shape)}"
            )
        expected_margin_shape = document_features.shape[:-1]
        if semantic_margin.shape != expected_margin_shape:
            raise ValueError(
                "semantic_margin must match document feature leading dimensions: "
                f"expected={tuple(expected_margin_shape)} got={tuple(semantic_margin.shape)}"
            )
        if document_mask is not None and document_mask.shape != semantic_margin.shape:
            raise ValueError("document_mask must have the same shape as semantic_margin")

        residual = self.residual_mlp(document_features).squeeze(-1)
        margin = semantic_margin.to(device=residual.device, dtype=residual.dtype)
        temperature = self.temperature.to(device=residual.device, dtype=residual.dtype)
        max_bias = self.max_suppression_bias.to(device=residual.device, dtype=residual.dtype)
        strength = self.prior_strength.to(device=residual.device, dtype=residual.dtype)
        epsilon = self.boundary_epsilon.to(device=residual.device, dtype=residual.dtype)
        prior_bias = torch.clamp(strength * margin / temperature, min=-max_bias, max=0.0)

        minimum_gate = torch.exp(-max_bias)
        prior_gate = torch.exp(prior_bias)
        relative_keep = (prior_gate - minimum_gate) / (1.0 - minimum_gate)
        relative_keep = relative_keep.clamp(min=epsilon, max=1.0 - epsilon)
        prior_log_odds = torch.logit(relative_keep)
        combined_score = prior_log_odds + residual
        adjusted_relative_keep = torch.sigmoid(combined_score)
        adjusted_gate = minimum_gate + (1.0 - minimum_gate) * adjusted_relative_keep
        document_bias = torch.log(adjusted_gate)

        if document_mask is not None:
            valid = document_mask.to(device=residual.device, dtype=torch.bool)
            residual = residual.masked_fill(~valid, 0.0)
            prior_bias = prior_bias.masked_fill(~valid, 0.0)
            combined_score = combined_score.masked_fill(~valid, 0.0)
            document_bias = document_bias.masked_fill(~valid, 0.0)

        return SemanticResidualAttentionOutput(
            residual=residual,
            prior_bias=prior_bias,
            combined_score=combined_score,
            document_bias=document_bias,
        )


def document_bias_to_token_bias(
    document_bias: torch.Tensor,
    token_document_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather document biases onto key tokens without detaching gradients.

    Args:
        document_bias: Tensor ``[batch, documents]``.
        token_document_ids: Long tensor ``[batch, key_length]`` containing a
            document index for document tokens and ``-1`` for every other
            token.

    Returns:
        Tensor ``[batch, key_length]`` suitable for the existing semantic
        attention convention.  No Python float conversion or tensor copy is
        used, so downstream answer gradients reach ``document_bias``.
    """

    if document_bias.ndim != 2:
        raise ValueError("document_bias must have shape [batch, documents]")
    if token_document_ids.ndim != 2:
        raise ValueError("token_document_ids must have shape [batch, key_length]")
    if document_bias.shape[0] != token_document_ids.shape[0]:
        raise ValueError("document and token bias batch sizes differ")
    if token_document_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("token_document_ids must be an integer tensor")
    if document_bias.shape[1] == 0:
        raise ValueError("document_bias must contain at least one document")

    ids = token_document_ids.to(device=document_bias.device, dtype=torch.long)
    valid = ids >= 0
    if bool((ids[valid] >= document_bias.shape[1]).any()):
        raise ValueError("token_document_ids contains an out-of-range document index")
    safe_ids = ids.clamp_min(0)
    gathered = torch.gather(document_bias, dim=1, index=safe_ids)
    return gathered * valid.to(dtype=document_bias.dtype)


def single_query_document_attention_bias(
    document_bias: torch.Tensor,
    token_document_ids: torch.Tensor,
) -> torch.Tensor:
    """Return a broadcastable ``[batch, 1, 1, key_length]`` q_len=1 bias."""

    token_bias = document_bias_to_token_bias(document_bias, token_document_ids)
    return token_bias[:, None, None, :]


def semantic_ordering_hinge_loss(
    document_bias: torch.Tensor,
    semantic_target: torch.Tensor,
    document_mask: torch.Tensor | None = None,
    margin: float = 0.1,
) -> torch.Tensor:
    """Question-balanced hinge loss for semantic support ordering.

    ``semantic_target`` uses one for Direct/Supporting evidence and zero for
    No/Misleading evidence.  The loss only requires a support document to have
    a *higher* (less negative) bias than a non-support document; it does not
    prescribe either document's absolute amount of attention.
    """

    if document_bias.ndim != 2 or semantic_target.shape != document_bias.shape:
        raise ValueError("document_bias and semantic_target must share [batch, documents]")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if document_mask is None:
        valid = torch.ones_like(document_bias, dtype=torch.bool)
    else:
        if document_mask.shape != document_bias.shape:
            raise ValueError("document_mask must match document_bias")
        valid = document_mask.to(device=document_bias.device, dtype=torch.bool)
    target = semantic_target.to(device=document_bias.device)
    target_valid = target[valid]
    if target_valid.numel() and not bool(((target_valid == 0) | (target_valid == 1)).all()):
        raise ValueError("valid semantic targets must be binary 0/1")

    positive = valid & (target == 1)
    negative = valid & (target == 0)
    pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)
    # Axis 1 is the support document; axis 2 is the non-support document.
    pair_difference = document_bias.unsqueeze(2) - document_bias.unsqueeze(1)
    pair_loss = F.relu(float(margin) - pair_difference)
    pair_count = pair_mask.sum(dim=(1, 2))
    valid_questions = pair_count > 0
    if not bool(valid_questions.any()):
        return document_bias.sum() * 0.0
    per_question = (pair_loss * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    return per_question[valid_questions].mean()


def residual_anchor_loss(
    residual: torch.Tensor,
    document_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared residual that anchors the controller to its semantic prior."""

    if document_mask is None:
        return residual.square().mean() if residual.numel() else residual.sum() * 0.0
    if document_mask.shape != residual.shape:
        raise ValueError("document_mask must match residual")
    valid = document_mask.to(device=residual.device, dtype=torch.bool)
    if not bool(valid.any()):
        return residual.sum() * 0.0
    return residual[valid].square().mean()


def group_robust_answer_loss(
    option_logits: torch.Tensor,
    gold_option: torch.Tensor,
    no_rag_correct: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    balance_strength: float = 0.5,
) -> torch.Tensor:
    """Blend natural and group-equalized No-RAG-correct/wrong answer CE.

    All available examples are retained.  If a minibatch contains only one
    group, that group's mean is returned instead of fabricating an absent-group
    term.
    """

    if not 0.0 <= balance_strength <= 1.0:
        raise ValueError("balance_strength must be in [0, 1]")
    if option_logits.ndim != 2:
        raise ValueError("option_logits must have shape [batch, choices]")
    batch_size = option_logits.shape[0]
    if gold_option.shape != (batch_size,) or no_rag_correct.shape != (batch_size,):
        raise ValueError("gold_option and no_rag_correct must have shape [batch]")
    if sample_mask is None:
        valid = torch.ones(batch_size, dtype=torch.bool, device=option_logits.device)
    else:
        if sample_mask.shape != (batch_size,):
            raise ValueError("sample_mask must have shape [batch]")
        valid = sample_mask.to(device=option_logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return option_logits.sum() * 0.0

    targets = gold_option.to(device=option_logits.device, dtype=torch.long)
    losses = F.cross_entropy(option_logits, targets, reduction="none")
    groups = no_rag_correct.to(device=option_logits.device, dtype=torch.bool)
    group_means: list[torch.Tensor] = []
    for group_value in (False, True):
        members = valid & (groups == group_value)
        if bool(members.any()):
            group_means.append(losses[members].mean())
    natural = losses[valid].mean()
    balanced = torch.stack(group_means).mean()
    return (1.0 - float(balance_strength)) * natural + float(balance_strength) * balanced


def freeze_module_for_controller_training(module: nn.Module) -> nn.Module:
    """Freeze target parameters without disabling gradients through its inputs.

    Callers must still run the biased q_len=1 forward *outside* ``no_grad`` or
    ``inference_mode``.  Frozen parameters then receive no gradients, while an
    answer loss can backpropagate through the target computations into a
    controller-produced attention bias.
    """

    module.requires_grad_(False)
    module.eval()
    return module

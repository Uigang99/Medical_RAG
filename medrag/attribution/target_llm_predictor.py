"""Set-conditional target-LLM removal-attribution predictor.

The target LLM is frozen.  This module consumes document-span and global
prefix representations extracted from that target model and predicts three
separate, deliberately non-conserved quantities:

* the total conditional leave-one-document-out (LOO) signal;
* the relative distribution of that signal over the documents in the set;
* the output shift between the complete document set and an empty-document
  counterfactual.

The last quantity is not an "internal-knowledge percentage", and the LOO
signals are not assumed to sum to the set shift.  Both restrictions are
important because Jensen-Shannon divergence and document interactions are not
additive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AttributionPrediction:
    """Outputs of :class:`TargetLLMAttributionPredictor`."""

    document_logits: torch.Tensor
    log_total_loo: torch.Tensor
    log_set_shift: torch.Tensor


class TargetLLMAttributionPredictor(nn.Module):
    """Predict variable-size document-set removal sensitivity.

    ``document_features`` and ``global_features`` contain frozen target-LLM
    representations from the same selected layers.  The learned layer mixture
    is shared by the document and global branches.  A rank/length/K feature
    makes prompt order explicit while a masked sequence Transformer models
    interactions between the currently present documents.
    """

    def __init__(
        self,
        *,
        target_hidden_size: int,
        selected_layer_count: int,
        model_dim: int = 256,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if target_hidden_size <= 0 or selected_layer_count <= 0 or model_dim <= 0:
            raise ValueError("hidden sizes and selected_layer_count must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.target_hidden_size = int(target_hidden_size)
        self.selected_layer_count = int(selected_layer_count)
        self.model_dim = int(model_dim)

        self.layer_logits = nn.Parameter(torch.zeros(self.selected_layer_count))
        combined_dim = 4 * self.target_hidden_size
        self.content_projection = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, self.model_dim),
            nn.GELU(),
        )
        # relative rank, normalized token length, inverse set size
        self.structure_projection = nn.Sequential(
            nn.Linear(3, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.set_projection = nn.Sequential(
            nn.LayerNorm(self.target_hidden_size),
            nn.Linear(self.target_hidden_size, self.model_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.document_head = nn.Linear(self.model_dim, 1)
        self.total_head = nn.Linear(self.model_dim, 1)
        self.set_shift_head = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        document_features: torch.Tensor,
        global_features: torch.Tensor,
        document_mask: torch.Tensor,
        relative_rank: torch.Tensor,
        normalized_length: torch.Tensor,
    ) -> AttributionPrediction:
        """Return document and set-level predictions.

        Args:
            document_features: ``[batch, documents, layers, hidden]``.
            global_features: ``[batch, layers, hidden]``.
            document_mask: boolean ``[batch, documents]``.
            relative_rank: ``[batch, documents]`` in ``[0, 1]``.
            normalized_length: ``[batch, documents]`` non-negative feature.
        """

        if document_features.ndim != 4:
            raise ValueError("document_features must be [batch, documents, layers, hidden]")
        batch, documents, layers, hidden = document_features.shape
        if (layers, hidden) != (self.selected_layer_count, self.target_hidden_size):
            raise ValueError(
                "document feature contract mismatch: "
                f"expected layers/hidden={(self.selected_layer_count, self.target_hidden_size)} "
                f"got={(layers, hidden)}"
            )
        if global_features.shape != (batch, layers, hidden):
            raise ValueError("global_features shape mismatch")
        expected = (batch, documents)
        if document_mask.shape != expected or relative_rank.shape != expected:
            raise ValueError("document mask/rank shape mismatch")
        if normalized_length.shape != expected:
            raise ValueError("normalized_length shape mismatch")
        valid = document_mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every set must contain at least one document")

        weights = torch.softmax(self.layer_logits, dim=0).to(document_features.dtype)
        mixed_documents = torch.einsum("bkld,l->bkd", document_features, weights)
        mixed_global = torch.einsum("bld,l->bd", global_features, weights)
        expanded_global = mixed_global.unsqueeze(1).expand(-1, documents, -1)
        interaction = torch.cat(
            (
                mixed_documents,
                expanded_global,
                mixed_documents - expanded_global,
                mixed_documents * expanded_global,
            ),
            dim=-1,
        )
        content = self.content_projection(interaction)
        set_sizes = valid.sum(dim=1, keepdim=True).to(content.dtype)
        inverse_size = set_sizes.reciprocal().expand(-1, documents)
        structure = torch.stack(
            (
                relative_rank.to(content.dtype),
                normalized_length.to(content.dtype),
                inverse_size,
            ),
            dim=-1,
        )
        documents_encoded = content + self.structure_projection(structure)
        set_token = self.set_projection(mixed_global).unsqueeze(1)
        sequence = torch.cat((set_token, documents_encoded), dim=1)
        padding = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.bool, device=valid.device),
                ~valid,
            ),
            dim=1,
        )
        encoded = self.sequence_encoder(sequence, src_key_padding_mask=padding)
        encoded = self.output_norm(encoded)
        set_state = encoded[:, 0]
        document_states = encoded[:, 1:]
        document_logits = self.document_head(document_states).squeeze(-1)
        document_logits = document_logits.masked_fill(~valid, torch.finfo(document_logits.dtype).min)
        return AttributionPrediction(
            document_logits=document_logits,
            log_total_loo=self.total_head(set_state).squeeze(-1),
            log_set_shift=self.set_shift_head(set_state).squeeze(-1),
        )


def masked_document_distribution(
    logits: torch.Tensor,
    document_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked softmax over the present documents."""

    if logits.shape != document_mask.shape:
        raise ValueError("logits and document_mask must have identical shape")
    valid = document_mask.bool()
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=-1)
    return probabilities.masked_fill(~valid, 0.0)


def question_balanced_rank_loss(
    predicted_logits: torch.Tensor,
    teacher_influence: torch.Tensor,
    document_mask: torch.Tensor,
    *,
    minimum_log_ratio: float = 0.25,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """RankNet loss for teacher document pairs separated beyond noise."""

    if predicted_logits.shape != teacher_influence.shape or predicted_logits.shape != document_mask.shape:
        raise ValueError("rank-loss tensors must share [batch, documents]")
    valid = document_mask.bool()
    log_teacher = torch.log(teacher_influence.clamp_min(epsilon))
    teacher_difference = log_teacher.unsqueeze(2) - log_teacher.unsqueeze(1)
    predicted_difference = predicted_logits.unsqueeze(2) - predicted_logits.unsqueeze(1)
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    pair_mask &= torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
    pair_mask &= teacher_difference.abs() >= float(minimum_log_ratio)
    target = (teacher_difference > 0).to(predicted_difference.dtype)
    pair_loss = F.binary_cross_entropy_with_logits(
        predicted_difference,
        target,
        reduction="none",
    )
    pair_count = pair_mask.sum(dim=(1, 2))
    usable = pair_count > 0
    if not bool(usable.any()):
        return predicted_logits.sum() * 0.0
    per_question = (pair_loss * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    return per_question[usable].mean()


def attribution_loss(
    prediction: AttributionPrediction,
    *,
    teacher_influence: torch.Tensor,
    teacher_total_loo: torch.Tensor,
    teacher_set_shift: torch.Tensor,
    document_mask: torch.Tensor,
    minimum_total_for_share: float = 1e-6,
    epsilon: float = 1e-12,
    total_weight: float = 1.0,
    share_weight: float = 0.5,
    set_shift_weight: float = 0.5,
    rank_weight: float = 0.1,
    minimum_rank_log_ratio: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Compute non-additive conditional-removal attribution objectives."""

    valid = document_mask.bool()
    teacher_influence = teacher_influence.masked_fill(~valid, 0.0)
    target_log_total = torch.log(teacher_total_loo.clamp_min(epsilon))
    target_log_set_shift = torch.log(teacher_set_shift.clamp_min(epsilon))
    total_loss = F.smooth_l1_loss(prediction.log_total_loo, target_log_total)
    set_shift_loss = F.smooth_l1_loss(prediction.log_set_shift, target_log_set_shift)

    measurable = teacher_total_loo >= float(minimum_total_for_share)
    if bool(measurable.any()):
        target_share = teacher_influence[measurable] / teacher_total_loo[measurable, None].clamp_min(epsilon)
        predicted_log_share = F.log_softmax(
            prediction.document_logits[measurable],
            dim=-1,
        )
        share_loss = -(target_share * predicted_log_share).sum(dim=-1).mean()
        rank_loss = question_balanced_rank_loss(
            prediction.document_logits[measurable],
            teacher_influence[measurable],
            valid[measurable],
            minimum_log_ratio=minimum_rank_log_ratio,
            epsilon=epsilon,
        )
    else:
        share_loss = prediction.document_logits.sum() * 0.0
        rank_loss = prediction.document_logits.sum() * 0.0
    loss = (
        float(total_weight) * total_loss
        + float(share_weight) * share_loss
        + float(set_shift_weight) * set_shift_loss
        + float(rank_weight) * rank_loss
    )
    return {
        "loss": loss,
        "total": total_loss,
        "share": share_loss,
        "set_shift": set_shift_loss,
        "rank": rank_loss,
        "measurable_questions": measurable.sum(),
    }

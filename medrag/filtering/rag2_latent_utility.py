from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

from medrag.filtering.rag2_official import OFFICIAL_INSTRUCTION, convert_legacy_filter_input


MODEL_VERSION = "rag2_latent_utility_scorer_v1"


def split_official_filter_input(value: str) -> tuple[str, str]:
    """Split the released RAG2 filter prompt into question/options and evidence."""

    normalized = convert_legacy_filter_input(value)
    prefix = f"{OFFICIAL_INSTRUCTION}\n\nEvidence: "
    marker = "\n\nQuestion: "
    if not normalized.startswith(prefix):
        raise ValueError("official_filter_input_prefix_mismatch")
    boundary = normalized.find(marker, len(prefix))
    if boundary < 0:
        raise ValueError("official_filter_input_question_marker_missing")
    evidence = normalized[len(prefix) : boundary].strip()
    question = normalized[boundary + len(marker) :].strip()
    if not evidence or not question:
        raise ValueError("official_filter_input_empty_field")
    return question, evidence


def centered_by_question(delta_h: torch.Tensor, document_to_question: torch.Tensor) -> torch.Tensor:
    """Center document interventions within each question in a batch."""

    if delta_h.ndim != 2 or document_to_question.ndim != 1:
        raise ValueError("Expected delta_h=[documents, hidden] and one group ID per document")
    question_count = int(document_to_question.max().item()) + 1
    totals = delta_h.new_zeros((question_count, delta_h.shape[-1]))
    totals.index_add_(0, document_to_question, delta_h)
    counts = torch.bincount(document_to_question, minlength=question_count).to(delta_h.dtype)
    means = totals / counts.clamp_min(1).unsqueeze(-1)
    return delta_h - means[document_to_question]


@dataclass(frozen=True)
class LatentUtilityConfig:
    base_model_name_or_path: str
    hidden_size: int = 4096
    latent_size: int = 256
    dropout: float = 0.1
    trainable_text_encoder_layers: int = 4
    decision_threshold: float = 0.4
    decision_temperature: float = 0.15
    model_version: str = MODEL_VERSION


class LatentUtilityScorer(nn.Module):
    """Predict document utility without receiving gold answer direction ``c``.

    The question branch creates a latent axis from question/options text and
    ``h0``.  The document branch represents evidence text and the intervention
    ``hD-h0`` (both absolute and question-centered).  Only their interaction and
    a question-document semantic interaction can produce the scalar score;
    there is deliberately no direct ``h0 -> score`` shortcut.
    """

    def __init__(self, config: LatentUtilityConfig) -> None:
        super().__init__()
        self.utility_config = config
        self.text_encoder = T5EncoderModel.from_pretrained(config.base_model_name_or_path)
        text_size = int(self.text_encoder.config.d_model)
        latent = int(config.latent_size)
        hidden = int(config.hidden_size)
        dropout = float(config.dropout)

        self.question_text_projection = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.document_text_projection = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.h0_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.centered_delta_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.question_axis = nn.Sequential(
            nn.Linear(latent * 2, latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent, latent),
        )
        self.document_state = nn.Sequential(
            nn.Linear(latent * 3 + 1, latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent, latent),
        )
        self.semantic_residual = nn.Sequential(
            nn.Linear(latent * 3, latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent, 1),
        )
        self.log_alignment_scale = nn.Parameter(torch.tensor(math.log(2.0), dtype=torch.float32))
        self._set_text_encoder_trainable_layers(config.trainable_text_encoder_layers)

    def _set_text_encoder_trainable_layers(self, count: int) -> None:
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        blocks = self.text_encoder.encoder.block
        if count < 0 or count > len(blocks):
            raise ValueError(f"trainable_text_encoder_layers must be in [0,{len(blocks)}]")
        if count:
            for block in blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.text_encoder.encoder.final_layer_norm.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(output.last_hidden_state, attention_mask)

    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        document_input_ids: torch.Tensor,
        document_attention_mask: torch.Tensor,
        h0: torch.Tensor,
        delta_h: torch.Tensor,
        document_to_question: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        question_text = self.question_text_projection(
            self._encode(question_input_ids, question_attention_mask)
        )
        document_text = self.document_text_projection(
            self._encode(document_input_ids, document_attention_mask)
        )
        h0_state = self.h0_projection(h0.float())
        axis = F.normalize(self.question_axis(torch.cat((question_text, h0_state), dim=-1)), dim=-1)

        delta_float = delta_h.float()
        centered = centered_by_question(delta_float, document_to_question)
        magnitude = torch.log1p(torch.linalg.vector_norm(delta_float, dim=-1, keepdim=True))
        document_state = F.normalize(
            self.document_state(
                torch.cat(
                    (
                        document_text,
                        self.delta_projection(delta_float),
                        self.centered_delta_projection(centered),
                        magnitude,
                    ),
                    dim=-1,
                )
            ),
            dim=-1,
        )
        question_for_document = question_text[document_to_question]
        axis_for_document = axis[document_to_question]
        alignment = (axis_for_document * document_state).sum(dim=-1)
        alignment = alignment * self.log_alignment_scale.exp().clamp(max=20.0)
        residual = self.semantic_residual(
            torch.cat(
                (
                    question_for_document,
                    document_text,
                    question_for_document * document_text,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        score = alignment + residual
        probability = torch.sigmoid(
            (score - float(self.utility_config.decision_threshold))
            / float(self.utility_config.decision_temperature)
        )
        return {
            "utility_score": score,
            "helpful_probability": probability,
            "alignment_score": alignment,
            "semantic_residual": residual,
        }

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in names
        }

    def save_trainable(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "latent_utility_config.json").write_text(
            json.dumps(asdict(self.utility_config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.trainable_state_dict(), output_dir / "trainable_model.bin")

    @classmethod
    def from_trainable(cls, path: Path, map_location: str | torch.device = "cpu") -> "LatentUtilityScorer":
        config = LatentUtilityConfig(
            **json.loads((path / "latent_utility_config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        state = torch.load(path / "trainable_model.bin", map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(trainable.intersection(missing))
        if missing_trainable or unexpected:
            raise RuntimeError(
                f"Invalid trainable checkpoint: missing={missing_trainable} unexpected={unexpected}"
            )
        return model

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel


MODEL_VERSION = "rag2_extreme_utility_scorer_v1"
INPUT_MODES = ("text_only", "text_delta", "text_h0_delta")


@dataclass(frozen=True)
class ExtremeUtilityConfig:
    base_model_name_or_path: str
    hidden_size: int = 4096
    latent_size: int = 256
    dropout: float = 0.1
    trainable_text_encoder_layers: int = 4
    input_mode: Literal["text_only", "text_delta", "text_h0_delta"] = "text_delta"
    source_layer: int = 28
    source_anchor: str = "pre_choice"
    label_threshold: float = 0.4
    model_version: str = MODEL_VERSION


class ExtremeUtilityScorer(nn.Module):
    """Predict a scalar document utility from RAG2 text and state transition.

    The main ``text_delta`` contract deliberately excludes the absolute no-RAG
    state.  This prevents the high-capacity text model from using ``h0`` as an
    easy proxy for no-RAG correctness.  Direction and magnitude are supplied
    separately so L2 normalization does not erase intervention strength.
    """

    def __init__(self, config: ExtremeUtilityConfig) -> None:
        super().__init__()
        if config.input_mode not in INPUT_MODES:
            raise ValueError(f"Unsupported input_mode={config.input_mode!r}")
        self.utility_config = config
        self.text_encoder = T5EncoderModel.from_pretrained(config.base_model_name_or_path)
        text_size = int(self.text_encoder.config.d_model)
        latent = int(config.latent_size)
        hidden = int(config.hidden_size)
        dropout = float(config.dropout)

        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        uses_delta = config.input_mode in {"text_delta", "text_h0_delta"}
        uses_h0 = config.input_mode == "text_h0_delta"
        if uses_delta:
            self.delta_projection = nn.Sequential(
                nn.Linear(hidden, latent, bias=False),
                nn.LayerNorm(latent),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.magnitude_projection = nn.Sequential(
                nn.Linear(1, latent),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.delta_projection = None
            self.magnitude_projection = None

        if uses_h0:
            self.h0_projection = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, latent),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.h0_projection = None

        branch_count = 1
        if uses_delta:
            branch_count += 3  # delta, text*delta, magnitude
        if uses_h0:
            branch_count += 2  # h0, text*h0
        self.score_head = nn.Sequential(
            nn.Linear(branch_count * latent, latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent, latent // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent // 2, 1),
        )
        self._set_text_encoder_trainable_layers(config.trainable_text_encoder_layers)

    @property
    def input_mode(self) -> str:
        return self.utility_config.input_mode

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        delta_h: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text = self.text_projection(self._mean_pool(encoded.last_hidden_state, attention_mask))
        branches = [text]

        if self.delta_projection is not None:
            if delta_h is None:
                raise ValueError(f"delta_h is required for input_mode={self.input_mode}")
            delta_float = delta_h.float()
            magnitude = torch.linalg.vector_norm(delta_float, dim=-1, keepdim=True)
            direction = F.normalize(delta_float, dim=-1, eps=1e-12)
            delta = self.delta_projection(direction)
            magnitude_state = self.magnitude_projection(torch.log1p(magnitude))
            branches.extend((delta, text * delta, magnitude_state))

        if self.h0_projection is not None:
            if h0 is None:
                raise ValueError("h0 is required for input_mode=text_h0_delta")
            h0_state = self.h0_projection(h0.float())
            branches.extend((h0_state, text * h0_state))

        score = self.score_head(torch.cat(branches, dim=-1)).squeeze(-1)
        return {"utility_score": score}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in names
        }

    def save_trainable(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "extreme_utility_config.json").write_text(
            json.dumps(asdict(self.utility_config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.trainable_state_dict(), output_dir / "trainable_model.bin")

    @classmethod
    def from_trainable(
        cls,
        path: Path,
        map_location: str | torch.device = "cpu",
    ) -> "ExtremeUtilityScorer":
        config = ExtremeUtilityConfig(
            **json.loads((path / "extreme_utility_config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        state = torch.load(path / "trainable_model.bin", map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(trainable.intersection(missing))
        if missing_trainable or unexpected:
            raise RuntimeError(
                f"Invalid extreme utility checkpoint: missing={missing_trainable} "
                f"unexpected={unexpected}"
            )
        return model


def balanced_extreme_pairwise_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    no_rag_state: torch.Tensor,
    document_to_question: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Rank Helpful above Harmful documents within the same question.

    Correct and wrong no-RAG questions are averaged separately and then given
    equal mass.  This prevents the more common no-RAG-correct group from
    dominating the ranking signal without resampling the pointwise examples.
    """

    grouped: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    counts = {"no_rag_correct": 0, "no_rag_wrong": 0}
    for question in torch.unique(document_to_question):
        mask = document_to_question.eq(question)
        q_score = score[mask]
        q_target = target[mask]
        helpful = q_score[q_target.eq(1)]
        harmful = q_score[q_target.eq(0)]
        if not helpful.numel() or not harmful.numel():
            continue
        states = torch.unique(no_rag_state[mask])
        if states.numel() != 1:
            raise RuntimeError("no-RAG state must be constant inside a question")
        state = int(states.item())
        differences = helpful[:, None] - harmful[None, :]
        grouped[state].append(F.softplus(-differences / float(temperature)).mean())
        counts["no_rag_correct" if state == 0 else "no_rag_wrong"] += int(differences.numel())
    values = [torch.stack(grouped[state]).mean() for state in (0, 1) if grouped[state]]
    if not values:
        return score.sum() * 0.0, counts
    return torch.stack(values).mean(), counts


def extreme_curriculum_loss(
    score: torch.Tensor,
    teacher_score: torch.Tensor,
    band: torch.Tensor,
    no_rag_state: torch.Tensor,
    document_to_question: torch.Tensor,
    *,
    stage: Literal["extreme", "neutral"],
    threshold: float,
    neutral_loss_weight: float = 0.1,
    pairwise_loss_weight: float = 0.5,
    pairwise_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Two-stage objective for pure extremes followed by neutral calibration.

    ``band`` uses -2/+2 for Harmful/Helpful and -1/0/+1 for the three
    neutral sub-bands.  The
    extreme BCE is always retained.  During neutral calibration, a low-weight
    Huber loss teaches the continuous score while balancing total regression
    mass between extreme and neutral examples.  Neutral positive/zero/negative
    sub-bands are balanced by the collator's ``band``-agnostic teacher sign.
    """

    if stage not in {"extreme", "neutral"}:
        raise ValueError(f"Unsupported curriculum stage={stage!r}")
    extreme = band.abs().eq(2)
    if stage == "extreme" and not bool(extreme.any()):
        raise RuntimeError("A batch must contain at least one extreme example")
    if bool(extreme.any()):
        target = band[extreme].gt(0).to(score.dtype)
        pointwise = F.binary_cross_entropy_with_logits(score[extreme], target)
        pairwise, pair_counts = balanced_extreme_pairwise_loss(
            score[extreme],
            target,
            no_rag_state[extreme],
            document_to_question[extreme],
            temperature=pairwise_temperature,
        )
    else:
        pointwise = score.sum() * 0.0
        pairwise = score.sum() * 0.0
        pair_counts = {"no_rag_correct": 0, "no_rag_wrong": 0}
    regression = score.sum() * 0.0
    if stage == "neutral":
        normalized_target = torch.clamp(teacher_score / float(threshold), -1.0, 1.0)
        per_row = F.smooth_l1_loss(torch.tanh(score), normalized_target, reduction="none")
        parts: list[torch.Tensor] = []
        if bool(extreme.any()):
            parts.append(per_row[extreme].mean())
        neutral = ~extreme
        if bool(neutral.any()):
            neutral_parts = []
            for selector in (band.eq(1), band.eq(0), band.eq(-1)):
                if bool(selector.any()):
                    neutral_parts.append(per_row[selector].mean())
            parts.append(torch.stack(neutral_parts).mean())
        regression = torch.stack(parts).mean()
    total = pointwise + float(pairwise_loss_weight) * pairwise
    if stage == "neutral":
        total = total + float(neutral_loss_weight) * regression
    return total, {
        "loss": float(total.detach().cpu()),
        "pointwise_loss": float(pointwise.detach().cpu()),
        "pairwise_loss": float(pairwise.detach().cpu()),
        "regression_loss": float(regression.detach().cpu()),
        "pair_count_no_rag_correct": float(pair_counts["no_rag_correct"]),
        "pair_count_no_rag_wrong": float(pair_counts["no_rag_wrong"]),
    }

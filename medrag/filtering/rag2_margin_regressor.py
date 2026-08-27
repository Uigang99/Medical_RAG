from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import T5EncoderModel


MODEL_VERSION = "rag2_text_margin_regressor_v1"


@dataclass(frozen=True)
class MarginRegressorConfig:
    base_model_name_or_path: str
    hidden_size: int = 256
    dropout: float = 0.1
    trainable_encoder_layers: int = 4
    model_version: str = MODEL_VERSION


class TextMarginRegressor(nn.Module):
    """Predict one bounded document-utility score from question/document text.

    Gold answers, teacher logits, margins, answer transitions, and hidden states
    are deliberately absent from ``forward``.  They are training metadata only.
    """

    def __init__(self, config: MarginRegressorConfig) -> None:
        super().__init__()
        self.margin_config = config
        self.encoder = T5EncoderModel.from_pretrained(config.base_model_name_or_path)
        encoder_size = int(self.encoder.config.d_model)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(encoder_size),
            nn.Linear(encoder_size, int(config.hidden_size)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_size), 1),
        )
        self._set_trainable_encoder_layers(int(config.trainable_encoder_layers))

    def _set_trainable_encoder_layers(self, count: int) -> None:
        blocks = self.encoder.encoder.block
        if count < 0 or count > len(blocks):
            raise ValueError(f"trainable_encoder_layers must be in [0,{len(blocks)}]")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if count:
            for block in blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.encoder.encoder.final_layer_norm.parameters():
                parameter.requires_grad = True

    @staticmethod
    def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(encoded.last_hidden_state, attention_mask)
        raw_score = self.regression_head(pooled).squeeze(-1)
        # The teacher utility sigmoid(m_D)-sigmoid(m_0) is exactly in [-1,1].
        utility_score = torch.tanh(raw_score)
        return {"utility_score": utility_score, "raw_score": raw_score}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: value.detach().cpu().contiguous()
            for name, value in self.state_dict().items()
            if name in trainable
        }

    def save_trainable(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "margin_regressor_config.json").write_text(
            json.dumps(asdict(self.margin_config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        save_file(self.trainable_state_dict(), str(output_dir / "trainable_model.safetensors"))

    @classmethod
    def from_trainable(
        cls,
        path: Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "TextMarginRegressor":
        config = MarginRegressorConfig(
            **json.loads((path / "margin_regressor_config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        state = load_file(str(path / "trainable_model.safetensors"), device=str(map_location))
        missing, unexpected = model.load_state_dict(state, strict=False)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(trainable.intersection(missing))
        if missing_trainable or unexpected:
            raise RuntimeError(
                f"Invalid margin-regressor checkpoint: missing={missing_trainable} unexpected={unexpected}"
            )
        return model

    def trainable_parameter_groups(
        self,
        *,
        encoder_learning_rate: float,
        head_learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        encoder_parameters = [
            parameter for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("encoder.")
        ]
        head_parameters = [
            parameter for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("encoder.")
        ]
        groups: list[dict[str, Any]] = []
        if encoder_parameters:
            groups.append(
                {
                    "params": encoder_parameters,
                    "lr": float(encoder_learning_rate),
                    "weight_decay": float(weight_decay),
                }
            )
        groups.append(
            {
                "params": head_parameters,
                "lr": float(head_learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
        return groups

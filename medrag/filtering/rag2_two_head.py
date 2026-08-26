from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedModel, T5Config, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput


TWO_HEAD_CONFIG = "rag2_two_head_config.json"


class Rag2TwoHeadFilterModel(PreTrainedModel):
    """Flan-T5 encoder with separate decisiveness and utility heads.

    The decisive head is trained on every RAG2 label (H/NH -> decisive,
    Discard -> abstain).  The utility head is trained only on decisive H/NH
    examples, so Discard never receives an artificial positive/negative
    direction target.
    """

    config_class = T5Config
    base_model_prefix = "encoder"
    _tied_weights_keys = {"encoder.encoder.embed_tokens.weight": "encoder.shared.weight"}
    _keys_to_ignore_on_load_missing = [r"encoder\.encoder\.embed_tokens\.weight"]

    def __init__(self, config: T5Config) -> None:
        super().__init__(config)
        self.encoder = T5EncoderModel(config)
        hidden_size = int(self.config.d_model)
        dropout = float(getattr(config, "rag2_two_head_dropout", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.decisive_head = nn.Linear(hidden_size, 2)
        self.utility_head = nn.Linear(hidden_size, 2)
        self.decisive_loss_weight = float(getattr(config, "rag2_decisive_loss_weight", 1.0))
        self.utility_loss_weight = float(getattr(config, "rag2_utility_loss_weight", 1.0))
        self.post_init()

    @classmethod
    def from_base_model(
        cls,
        model_name_or_path: str | Path,
        *,
        dropout: float = 0.1,
        decisive_loss_weight: float = 1.0,
        utility_loss_weight: float = 1.0,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> "Rag2TwoHeadFilterModel":
        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if dtype is not None:
            kwargs["dtype"] = dtype
        encoder = T5EncoderModel.from_pretrained(model_name_or_path, **kwargs)
        encoder.config.rag2_two_head_dropout = float(dropout)
        encoder.config.rag2_decisive_loss_weight = float(decisive_loss_weight)
        encoder.config.rag2_utility_loss_weight = float(utility_loss_weight)
        encoder.config.rag2_filter_architecture = "flan_t5_encoder_masked_mean_two_head_v1"
        model = cls(encoder.config)
        model.encoder = encoder
        return model

    @classmethod
    def from_two_head_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> "Rag2TwoHeadFilterModel":
        checkpoint_dir = Path(checkpoint_dir)
        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return cls.from_pretrained(checkpoint_dir, **kwargs)

    @staticmethod
    def _masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (last_hidden_state * mask).sum(dim=1) / denominator

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        del sample_weight  # Used by the balanced sampler, not by the loss.
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.dropout(self._masked_mean(encoded.last_hidden_state, attention_mask))
        decisive_logits = self.decisive_head(pooled)
        utility_logits = self.utility_head(pooled)
        logits = torch.cat((decisive_logits, utility_logits), dim=-1)

        loss = None
        if labels is not None:
            decisive_targets = labels[:, 0].long()
            utility_targets = labels[:, 1].long()
            decisive_loss = F.cross_entropy(decisive_logits, decisive_targets)
            utility_mask = utility_targets >= 0
            if torch.any(utility_mask):
                utility_loss = F.cross_entropy(utility_logits[utility_mask], utility_targets[utility_mask])
                loss = (
                    self.decisive_loss_weight * decisive_loss
                    + self.utility_loss_weight * utility_loss
                )
            else:
                loss = self.decisive_loss_weight * decisive_loss

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)

    def save_two_head_pretrained(
        self,
        output_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.rag2_two_head_dropout = float(self.dropout.p)
        self.config.rag2_decisive_loss_weight = self.decisive_loss_weight
        self.config.rag2_utility_loss_weight = self.utility_loss_weight
        self.config.rag2_filter_architecture = "flan_t5_encoder_masked_mean_two_head_v1"
        self.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        config = {
            "model_type": "rag2_two_head_filter",
            "architecture": "flan_t5_encoder_masked_mean_two_head_v1",
            "dropout": float(self.dropout.p),
            "decisive_loss_weight": self.decisive_loss_weight,
            "utility_loss_weight": self.utility_loss_weight,
            "head_1": {"classes": ["discard", "decisive"], "target_scope": "all rows"},
            "head_2": {
                "classes": ["not helpful", "helpful"],
                "target_scope": "Helpful and Not Helpful rows only",
            },
            **(metadata or {}),
        }
        (output_dir / TWO_HEAD_CONFIG).write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_dir


def two_head_probabilities(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[-1] != 4:
        raise ValueError(f"Expected [batch, 4] two-head logits, got {tuple(logits.shape)}")
    return torch.softmax(logits[:, :2], dim=-1)[:, 1], torch.softmax(logits[:, 2:], dim=-1)[:, 1]


class Rag2TwoHeadFilterPredictor:
    """Inference helper using the validation-frozen selective thresholds."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 64,
        max_length: int = 768,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.metadata = json.loads((checkpoint_dir / TWO_HEAD_CONFIG).read_text(encoding="utf-8"))
        self.theta_decisive = float(self.metadata["theta_decisive"])
        self.theta_helpful = float(self.metadata["theta_helpful"])
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        self.model = Rag2TwoHeadFilterModel.from_two_head_checkpoint(
            checkpoint_dir,
            dtype=dtype if self.device.type == "cuda" else None,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_inputs(self, inputs: list[str], *, show_progress: bool = True) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        progress = tqdm(
            total=len(inputs),
            desc="TwoHeadFilter",
            unit="pair",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for start in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_inputs,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**encoded)
            p_decisive, p_helpful = two_head_probabilities(output.logits.float())
            for decisive, helpful in zip(p_decisive.cpu().tolist(), p_helpful.cpu().tolist()):
                if decisive < self.theta_decisive:
                    prediction = "discard"
                elif helpful >= self.theta_helpful:
                    prediction = "helpful"
                else:
                    prediction = "not helpful"
                results.append(
                    {
                        "prediction": prediction,
                        "p_decisive": float(decisive),
                        "p_helpful_given_decisive": float(helpful),
                        "passes_filter": prediction == "helpful",
                    }
                )
            progress.update(len(batch_inputs))
        progress.close()
        return results

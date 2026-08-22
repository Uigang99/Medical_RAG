from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import medrag.filtering.rag2_latent_utility as latent


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=8)
        self.embedding = nn.Embedding(32, 8)
        self.encoder = SimpleNamespace(
            block=nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)]),
            final_layer_norm=nn.LayerNorm(8),
        )

    def parameters(self, recurse: bool = True):
        yield from self.embedding.parameters(recurse)
        yield from self.encoder.block.parameters(recurse)
        yield from self.encoder.final_layer_norm.parameters(recurse)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.embedding(input_ids)
        for block in self.encoder.block:
            hidden = torch.tanh(block(hidden))
        hidden = self.encoder.final_layer_norm(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_split_official_filter_input() -> None:
    value = (
        "Given the following evidence, determine whether it helps answer the provided question.\n\n"
        "Evidence: useful medical fact\n\nQuestion: clinical question\nA) one\nB) two"
    )
    question, evidence = latent.split_official_filter_input(value)
    assert question == "clinical question\nA) one\nB) two"
    assert evidence == "useful medical fact"


def test_centered_by_question_has_zero_group_means() -> None:
    delta = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    group = torch.tensor([0, 0, 1])
    centered = latent.centered_by_question(delta, group)
    assert torch.allclose(centered[:2].mean(dim=0), torch.zeros(2))
    assert torch.allclose(centered[2], torch.zeros(2))


def test_latent_utility_outputs_one_score_per_document(monkeypatch) -> None:
    monkeypatch.setattr(latent.T5EncoderModel, "from_pretrained", lambda *_args, **_kwargs: TinyEncoder())
    config = latent.LatentUtilityConfig(
        base_model_name_or_path="unused",
        hidden_size=6,
        latent_size=4,
        dropout=0.0,
        trainable_text_encoder_layers=1,
        decision_threshold=0.4,
        decision_temperature=0.2,
    )
    model = latent.LatentUtilityScorer(config)
    output = model(
        question_input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
        question_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        document_input_ids=torch.tensor([[6, 7], [8, 9], [10, 11]]),
        document_attention_mask=torch.ones(3, 2, dtype=torch.long),
        h0=torch.randn(2, 6),
        delta_h=torch.randn(3, 6),
        document_to_question=torch.tensor([0, 0, 1]),
    )
    assert output["utility_score"].shape == (3,)
    assert output["helpful_probability"].shape == (3,)
    assert bool(torch.isfinite(output["utility_score"]).all())
    assert bool(((output["helpful_probability"] >= 0) & (output["helpful_probability"] <= 1)).all())


def test_only_requested_text_layers_are_trainable(monkeypatch) -> None:
    monkeypatch.setattr(latent.T5EncoderModel, "from_pretrained", lambda *_args, **_kwargs: TinyEncoder())
    model = latent.LatentUtilityScorer(
        latent.LatentUtilityConfig(
            base_model_name_or_path="unused",
            hidden_size=6,
            latent_size=4,
            trainable_text_encoder_layers=1,
        )
    )
    assert not any(parameter.requires_grad for parameter in model.text_encoder.embedding.parameters())
    assert not any(parameter.requires_grad for parameter in model.text_encoder.encoder.block[0].parameters())
    assert all(parameter.requires_grad for parameter in model.text_encoder.encoder.block[1].parameters())
    assert all(parameter.requires_grad for parameter in model.text_encoder.encoder.final_layer_norm.parameters())

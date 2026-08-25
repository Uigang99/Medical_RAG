from __future__ import annotations

import torch

from medrag.filtering.rag2_extreme_utility import (
    ExtremeUtilityConfig,
    balanced_extreme_pairwise_loss,
    extreme_curriculum_loss,
)


def test_primary_config_excludes_absolute_h0() -> None:
    config = ExtremeUtilityConfig(base_model_name_or_path="unused")
    assert config.input_mode == "text_delta"
    assert config.source_layer == 28
    assert config.source_anchor == "pre_choice"
    assert config.label_threshold == 0.4


def test_pairwise_loss_balances_no_rag_groups() -> None:
    score = torch.tensor([2.0, -2.0, 1.0, -1.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    state = torch.tensor([0, 0, 1, 1])
    question = torch.tensor([0, 0, 1, 1])
    loss, counts = balanced_extreme_pairwise_loss(score, target, state, question)
    assert 0.0 < float(loss.detach()) < 0.2
    assert counts == {"no_rag_correct": 1, "no_rag_wrong": 1}
    loss.backward()
    assert score.grad is not None
    assert score.grad[0] < 0 < score.grad[1]


def test_extreme_stage_uses_only_extreme_codes() -> None:
    score = torch.tensor([1.5, -1.5, 0.2], requires_grad=True)
    teacher = torch.tensor([0.8, -0.8, 0.0])
    band = torch.tensor([2, -2, 0])
    state = torch.tensor([0, 0, 0])
    question = torch.tensor([0, 0, 0])
    loss, details = extreme_curriculum_loss(
        score,
        teacher,
        band,
        state,
        question,
        stage="extreme",
        threshold=0.4,
    )
    assert torch.isfinite(loss)
    assert details["regression_loss"] == 0.0
    loss.backward()
    assert score.grad is not None
    assert float(score.grad[2]) == 0.0


def test_neutral_stage_balances_three_neutral_subbands() -> None:
    score = torch.tensor([1.0, -1.0, 0.2, 0.0, -0.2], requires_grad=True)
    teacher = torch.tensor([0.8, -0.8, 0.2, 0.0, -0.2])
    band = torch.tensor([2, -2, 1, 0, -1])
    state = torch.tensor([0, 0, 0, 0, 0])
    question = torch.zeros(5, dtype=torch.long)
    loss, details = extreme_curriculum_loss(
        score,
        teacher,
        band,
        state,
        question,
        stage="neutral",
        threshold=0.4,
        neutral_loss_weight=0.1,
    )
    assert torch.isfinite(loss)
    assert details["regression_loss"] > 0
    loss.backward()
    assert score.grad is not None
    assert bool(torch.isfinite(score.grad).all())


def test_neutral_stage_accepts_question_with_no_extreme_document() -> None:
    score = torch.tensor([0.1, 0.0, -0.1], requires_grad=True)
    teacher = torch.tensor([0.2, 0.0, -0.2])
    band = torch.tensor([1, 0, -1])
    state = torch.tensor([1, 1, 1])
    question = torch.zeros(3, dtype=torch.long)
    loss, details = extreme_curriculum_loss(
        score,
        teacher,
        band,
        state,
        question,
        stage="neutral",
        threshold=0.4,
    )
    assert torch.isfinite(loss)
    assert details["pointwise_loss"] == 0.0
    assert details["regression_loss"] > 0.0
    loss.backward()
    assert bool(torch.isfinite(score.grad).all())

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

import torch
import torch.nn.functional as F


SEMANTIC_POSITIVE = "direct_support"
SEMANTIC_NEGATIVE = ("no_evidence", "misleading_evidence")


def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    """Finite Jensen-Shannon divergence for two categorical distributions."""

    p = torch.as_tensor(left, dtype=torch.float64).clamp_min(1e-12)
    q = torch.as_tensor(right, dtype=torch.float64).clamp_min(1e-12)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    value = 0.5 * (
        torch.sum(p * (p.log() - midpoint.log()))
        + torch.sum(q * (q.log() - midpoint.log()))
    )
    return float(value.item())


def choose_semantic_behavior_pair(
    documents: Sequence[dict[str, Any]],
    *,
    violation_threshold: float = 0.0,
) -> dict[str, Any] | None:
    """Choose one under-used Direct document and one sensitive negative.

    The positive is the Direct Support document with the smallest behavioral
    margin change.  The negative is selected primarily by how much it changes
    the frozen target model's answer distribution and secondarily by its gold
    margin change.  Both documents always belong to the same question.
    """

    positives = [
        row for row in documents if row.get("semantic_label") == SEMANTIC_POSITIVE
    ]
    negatives = [
        row for row in documents if row.get("semantic_label") in SEMANTIC_NEGATIVE
    ]
    if not positives or not negatives:
        return None

    positive = min(
        positives,
        key=lambda row: (
            float(row["gold_margin_delta"]),
            int(row.get("doc_rank", 10**9)),
            str(row["pair_id"]),
        ),
    )
    negative = max(
        negatives,
        key=lambda row: (
            float(row["answer_js_divergence"]),
            float(row["gold_margin_delta"]),
            -int(row.get("doc_rank", 10**9)),
            str(row["pair_id"]),
        ),
    )
    violation = float(negative["with_document_gold_margin"]) - float(
        positive["with_document_gold_margin"]
    )
    return {
        "positive": positive,
        "negative": negative,
        "semantic_preference_violation": violation,
        "pair_group": "hard" if violation >= float(violation_threshold) else "aligned",
    }


def stratified_pair_limit(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int | None,
    hard_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically limit pairs while retaining hard/aligned examples.

    Misleading and No-Evidence negatives are separately shuffled inside each
    hard/aligned pool so the much larger No-Evidence class cannot erase the
    rarer misleading-document signal.
    """

    values = list(rows)
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0,1]")
    if not values:
        return []

    requested = len(values) if limit is None or limit <= 0 else min(limit, len(values))
    hard_available = sum(str(row["pair_group"]) == "hard" for row in values)
    aligned_available = len(values) - hard_available
    if hard_fraction <= 0.0:
        feasible = min(requested, aligned_available)
    elif hard_fraction >= 1.0:
        feasible = min(requested, hard_available)
    else:
        feasible = min(
            requested,
            int(hard_available / hard_fraction),
            int(aligned_available / (1.0 - hard_fraction)),
        )
    if feasible <= 0:
        raise RuntimeError(
            "The requested hard/aligned mixture is impossible because one group is empty"
        )

    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        grouped[(str(row["pair_group"]), str(row["negative_semantic_label"]))].append(row)
    for group in grouped.values():
        rng.shuffle(group)

    hard_target = min(hard_available, int(round(feasible * hard_fraction)))
    aligned_target = min(aligned_available, feasible - hard_target)
    # Resolve any one-example rounding gap without violating availability.
    if hard_target + aligned_target < feasible:
        if hard_target < hard_available:
            hard_target += 1
        elif aligned_target < aligned_available:
            aligned_target += 1

    def take_group(group_name: str, target: int) -> list[dict[str, Any]]:
        no_evidence = grouped.get((group_name, "no_evidence"), [])
        misleading = grouped.get((group_name, "misleading_evidence"), [])
        # Reserve up to one quarter for misleading evidence, then backfill from
        # whichever semantic-negative pool still has examples.
        misleading_target = min(len(misleading), math.ceil(target * 0.25))
        selected = misleading[:misleading_target]
        remaining = target - len(selected)
        selected.extend(no_evidence[:remaining])
        remaining = target - len(selected)
        if remaining:
            selected.extend(misleading[misleading_target : misleading_target + remaining])
        return selected

    selected = take_group("hard", hard_target) + take_group("aligned", aligned_target)
    if len(selected) != feasible:
        raise RuntimeError(
            f"Stratified selection produced {len(selected)} rows, expected {feasible}"
        )
    rng.shuffle(selected)
    return selected


def gold_margins(choice_logits: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
    """Gold logit minus the strongest non-gold logit for four-choice MCQ."""

    if choice_logits.ndim != 2 or choice_logits.shape[1] != 4:
        raise ValueError(f"choice_logits must be [batch,4], got {tuple(choice_logits.shape)}")
    gold = choice_logits.gather(1, gold_indices[:, None]).squeeze(1)
    wrong = choice_logits.masked_fill(
        F.one_hot(gold_indices, num_classes=4).to(dtype=torch.bool),
        torch.finfo(choice_logits.dtype).min,
    ).amax(dim=1)
    return gold - wrong


def semantic_behavior_losses(
    *,
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    no_rag_logits: torch.Tensor,
    frozen_no_rag_probabilities: torch.Tensor,
    gold_indices: torch.Tensor,
    preference_margin: float,
    positive_weight: float,
    preference_weight: float,
    negative_invariance_weight: float,
    no_rag_preservation_weight: float,
) -> dict[str, torch.Tensor]:
    """Single-document semantic-behavior LoRA objective.

    The frozen No-RAG distribution is a stop-gradient cached teacher.  Only
    the three student branches receive gradients.
    """

    teacher = frozen_no_rag_probabilities.detach().to(dtype=torch.float32)
    teacher = teacher.clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True)
    positive_ce = F.cross_entropy(positive_logits.float(), gold_indices)
    positive_margin = gold_margins(positive_logits.float(), gold_indices)
    negative_margin = gold_margins(negative_logits.float(), gold_indices)
    preference = F.relu(float(preference_margin) - (positive_margin - negative_margin)).mean()
    negative_invariance = F.kl_div(
        F.log_softmax(negative_logits.float(), dim=-1), teacher, reduction="batchmean"
    )
    no_rag_preservation = F.kl_div(
        F.log_softmax(no_rag_logits.float(), dim=-1), teacher, reduction="batchmean"
    )
    total = (
        float(positive_weight) * positive_ce
        + float(preference_weight) * preference
        + float(negative_invariance_weight) * negative_invariance
        + float(no_rag_preservation_weight) * no_rag_preservation
    )
    return {
        "loss": total,
        "positive_ce": positive_ce,
        "preference": preference,
        "negative_invariance": negative_invariance,
        "no_rag_preservation": no_rag_preservation,
        "positive_margin": positive_margin.mean(),
        "negative_margin": negative_margin.mean(),
    }


def transition_name(before_correct: bool, after_correct: bool) -> str:
    return f"{'C' if before_correct else 'W'}->{'C' if after_correct else 'W'}"

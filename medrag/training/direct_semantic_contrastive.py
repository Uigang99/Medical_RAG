from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from medrag.training.direct_semantic_mismatch import gold_margins, rowwise_kl


TRAIN_GROUPS = (
    "direct_support_correction",
    "direct_support_preservation",
    "no_evidence_invariance",
    "same_question_contrast",
)


def teacher_margin(row: Mapping[str, Any], field: str = "frozen_document_probabilities") -> float:
    probabilities = [max(float(value), 1e-12) for value in row[field]]
    gold = "ABCD".index(str(row["gold_answer"]))
    return math.log(probabilities[gold]) - max(
        math.log(value) for index, value in enumerate(probabilities) if index != gold
    )


def build_training_groups(
    values: Sequence[dict[str, Any]], min_pair_teacher_gap: float
) -> dict[str, list[Any]]:
    direct_wrong: list[dict[str, Any]] = []
    direct_correct: list[dict[str, Any]] = []
    no_evidence_safe: list[dict[str, Any]] = []
    by_question: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"direct_support": [], "no_evidence": []}
    )

    for value in values:
        row = value["row"]
        semantic = str(row["semantic_label"])
        if semantic == "direct_support":
            target = direct_correct if bool(row["frozen_document_correct"]) else direct_wrong
            target.append(value)
        elif semantic == "no_evidence":
            if bool(row["frozen_no_rag_correct"]):
                no_evidence_safe.append(value)
        else:
            continue
        by_question[str(row["sample_id"])][semantic].append(value)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample_id in sorted(by_question):
        groups = by_question[sample_id]
        if not groups["direct_support"] or not groups["no_evidence"]:
            continue
        direct = max(groups["direct_support"], key=lambda value: teacher_margin(value["row"]))
        no_evidence = min(groups["no_evidence"], key=lambda value: teacher_margin(value["row"]))
        gap = teacher_margin(direct["row"]) - teacher_margin(no_evidence["row"])
        if gap >= float(min_pair_teacher_gap):
            pairs.append((direct, no_evidence))

    result = {
        "direct_support_correction": direct_wrong,
        "direct_support_preservation": direct_correct,
        "no_evidence_invariance": no_evidence_safe,
        "same_question_contrast": pairs,
    }
    missing = [name for name, rows in result.items() if not rows]
    if missing:
        raise RuntimeError(f"Empty semantic-contrastive training groups: {missing}")
    return result


def balanced_epoch_samples(
    groups: Mapping[str, Sequence[Any]], *, epoch: int, seed: int
) -> tuple[int, dict[str, list[Any]]]:
    target = max(
        len(groups["direct_support_correction"]),
        len(groups["direct_support_preservation"]),
    )
    if target <= 0:
        raise RuntimeError("Balanced epoch target is empty")
    selected: dict[str, list[Any]] = {}
    for group_index, name in enumerate(TRAIN_GROUPS):
        source = list(groups[name])
        rng = random.Random(int(seed) + 1009 * int(epoch) + 97 * group_index)
        values: list[Any] = []
        while len(values) < target:
            cycle = list(source)
            rng.shuffle(cycle)
            values.extend(cycle)
        selected[name] = values[:target]
    return target, selected


def semantic_contrastive_losses(
    *,
    correction_logits: torch.Tensor,
    correction_gold: torch.Tensor,
    preservation_logits: torch.Tensor,
    preservation_teacher: torch.Tensor,
    invariance_logits: torch.Tensor,
    invariance_teacher: torch.Tensor,
    pair_direct_logits: torch.Tensor,
    pair_no_evidence_logits: torch.Tensor,
    pair_gold: torch.Tensor,
    boundary_margin: float,
    pair_margin: float,
) -> dict[str, torch.Tensor]:
    correction = F.relu(
        float(boundary_margin) - gold_margins(correction_logits.float(), correction_gold)
    ).mean()
    preservation = rowwise_kl(preservation_logits, preservation_teacher).mean()
    invariance = rowwise_kl(invariance_logits, invariance_teacher).mean()
    direct_margin = gold_margins(pair_direct_logits.float(), pair_gold)
    no_evidence_margin = gold_margins(pair_no_evidence_logits.float(), pair_gold)
    contrast = F.relu(float(pair_margin) - (direct_margin - no_evidence_margin)).mean()
    total = (correction + preservation + invariance + contrast) / len(TRAIN_GROUPS)
    return {
        "loss": total,
        "direct_support_correction": correction,
        "direct_support_preservation": preservation,
        "no_evidence_invariance": invariance,
        "same_question_contrast": contrast,
    }

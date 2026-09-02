from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


TRAIN_CASES = (
    "direct_support_w2w",
    "direct_support_c2w",
    "direct_support_preserve",
    "no_evidence_c2w",
    "no_evidence_preserve",
)


def gold_margins(logits: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
    """Gold-option logit minus the strongest wrong-option logit."""

    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError(f"Expected [batch,4] choice logits, got {tuple(logits.shape)}")
    gold = logits.gather(1, gold_indices[:, None]).squeeze(1)
    wrong = logits.masked_fill(
        F.one_hot(gold_indices, num_classes=4).bool(),
        torch.finfo(logits.dtype).min,
    ).amax(dim=1)
    return gold - wrong


def rowwise_kl(student_logits: torch.Tensor, teacher_probabilities: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student), independently for every row."""

    teacher = teacher_probabilities.detach().float().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    return F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher,
        reduction="none",
    ).sum(dim=-1)


def semantic_mismatch_losses(
    *,
    document_logits: torch.Tensor,
    no_rag_logits: torch.Tensor,
    frozen_document_probabilities: torch.Tensor,
    frozen_no_rag_probabilities: torch.Tensor,
    gold_indices: torch.Tensor,
    case_indices: torch.Tensor,
    boundary_margin: float,
    gain_margin: float,
    case_weights: Mapping[str, float],
    no_rag_preservation_weight: float,
    no_rag_row_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Loss for the bounded direct-choice semantic/behavior mismatch MVP.

    Semantic labels and frozen-model outcomes select the case outside this
    function.  They are never model inputs.  Per-case means prevent the large
    No-Evidence group from dominating merely because it contains more rows.
    """

    if document_logits.shape != no_rag_logits.shape:
        raise ValueError("Document and no-RAG logits must have identical shapes")
    document_margin = gold_margins(document_logits.float(), gold_indices)
    no_rag_margin = gold_margins(no_rag_logits.float(), gold_indices)
    restore_to_no_rag = rowwise_kl(document_logits, frozen_no_rag_probabilities)
    preserve_document = rowwise_kl(document_logits, frozen_document_probabilities)
    preserve_no_rag = rowwise_kl(no_rag_logits, frozen_no_rag_probabilities)

    per_case: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    for index, case in enumerate(TRAIN_CASES):
        mask = case_indices == index
        if not bool(mask.any()):
            continue
        if case == "direct_support_w2w":
            boundary = F.relu(float(boundary_margin) - document_margin[mask])
            gain = F.relu(
                float(gain_margin)
                - (document_margin[mask] - no_rag_margin[mask])
            )
            value = (boundary + gain).mean()
        elif case in {"direct_support_c2w", "no_evidence_c2w", "no_evidence_preserve"}:
            value = restore_to_no_rag[mask].mean()
        elif case == "direct_support_preserve":
            value = preserve_document[mask].mean()
        else:  # pragma: no cover - TRAIN_CASES is exhaustive
            raise AssertionError(case)
        weight = float(case_weights.get(case, 0.0))
        per_case[case] = value
        if weight > 0.0:
            weighted.append(weight * value)

    if not weighted:
        raise RuntimeError("The batch has no positively weighted training case")
    # A fixed denominator keeps ``normal_case_weight=0.1`` meaningful even
    # when a shuffled mini-batch happens to contain only preservation rows.
    case_loss = torch.stack(weighted).sum() / len(TRAIN_CASES)
    if no_rag_row_weights is None:
        no_rag_loss = preserve_no_rag.mean()
    else:
        row_weights = no_rag_row_weights.to(
            device=preserve_no_rag.device, dtype=preserve_no_rag.dtype
        ).clamp_min(0.0)
        no_rag_loss = (preserve_no_rag * row_weights).sum() / row_weights.sum().clamp_min(1e-8)
    total = case_loss + float(no_rag_preservation_weight) * no_rag_loss
    return {
        "loss": total,
        "case_loss": case_loss,
        "no_rag_preservation": no_rag_loss,
        "mean_document_margin": document_margin.mean(),
        "mean_no_rag_margin": no_rag_margin.mean(),
        **{f"case/{name}": value for name, value in per_case.items()},
    }

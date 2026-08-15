from __future__ import annotations

"""Shared contracts for held-out RAG2/hidden-state oracle evaluation."""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any


RAG2_HELPFUL = "Helpful"


def deterministic_question_sample(
    sample_ids: Iterable[str],
    *,
    dataset: str,
    limit: int,
    seed: int,
) -> list[str]:
    """Select a stable, input-order-independent question sample."""

    unique = sorted(set(str(value) for value in sample_ids if str(value)))
    if limit <= 0 or limit >= len(unique):
        return unique

    def key(sample_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}\0{dataset}\0{sample_id}".encode("utf-8")).digest()
        return digest, sample_id

    return sorted(unique, key=key)[:limit]


def canonicalize_rag2_labels(
    rows: Iterable[dict[str, Any]],
    *,
    selected_sample_ids: set[str],
    max_rank: int,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Deduplicate RAG2 traces, preferring the sole quality-passing trace.

    MedQA contains a failed first generation followed by one valid replacement
    for some canonical pairs.  A pair with more than one quality-passing trace
    is ambiguous and is rejected instead of being silently order-dependent.
    """

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    source_rows = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in selected_sample_ids:
            continue
        rank = int(row.get("doc_rank") or 0)
        if rank < 1 or rank > max_rank:
            continue
        stable_id = str(row.get("doc_stable_id") or "")
        if not stable_id:
            raise ValueError(f"RAG2 label row has no doc_stable_id: {sample_id} rank={rank}")
        grouped[(sample_id, rank, stable_id)].append(row)
        source_rows += 1

    labels: dict[str, dict[str, str]] = defaultdict(dict)
    duplicate_keys = 0
    extra_rows = 0
    valid_replacements = 0
    all_invalid_duplicates = 0
    label_counts: Counter[str] = Counter()
    for (sample_id, rank, stable_id), values in grouped.items():
        if len(values) > 1:
            duplicate_keys += 1
            extra_rows += len(values) - 1
        valid = [row for row in values if bool(row.get("quality_pass"))]
        if len(valid) > 1:
            descriptions = [str(row.get("pseudo_label") or "") for row in valid]
            raise ValueError(
                "Multiple quality-passing RAG2 labels for one canonical pair: "
                f"{sample_id} rank={rank} doc={stable_id} labels={descriptions}"
            )
        if valid:
            chosen = valid[0]
            if len(values) > 1:
                valid_replacements += 1
        else:
            # Every unusable technical trace is one excluded oracle decision.
            chosen = values[-1]
            if len(values) > 1:
                all_invalid_duplicates += 1
        label = str(chosen.get("pseudo_label") or "Excluded")
        if not bool(chosen.get("quality_pass")):
            label = "Excluded"
        labels[sample_id][stable_id] = label
        label_counts[label] += 1

    audit = {
        "source_rows": source_rows,
        "canonical_pairs": len(grouped),
        "duplicate_keys": duplicate_keys,
        "extra_rows": extra_rows,
        "valid_replacements": valid_replacements,
        "all_invalid_duplicates": all_invalid_duplicates,
        **{f"label_{key}": value for key, value in sorted(label_counts.items())},
    }
    return {key: dict(value) for key, value in labels.items()}, audit


def load_hidden_projection_labels(
    rows: Iterable[dict[str, Any]],
    *,
    selected_sample_ids: set[str],
    max_rank: int,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Load one layer-28 projection score for each selected canonical pair."""

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    rows_read = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in selected_sample_ids:
            continue
        rank = int(row.get("doc_rank") or 0)
        if rank < 1 or rank > max_rank:
            continue
        stable_id = str(row.get("doc_stable_id") or "")
        if not stable_id:
            raise ValueError(f"Hidden label row has no doc_stable_id: {sample_id} rank={rank}")
        if stable_id in scores[sample_id]:
            raise ValueError(f"Duplicate hidden projection row: {sample_id} rank={rank} doc={stable_id}")
        scores[sample_id][stable_id] = float(row["projection_score"])
        rows_read += 1
    return {key: dict(value) for key, value in scores.items()}, {"rows": rows_read}


def oracle_document_is_helpful(
    *,
    policy: str,
    rag2_label: str | None,
    hidden_projection: float | None,
    hidden_threshold: float | None,
) -> bool:
    if policy == "rag2":
        return str(rag2_label or "") == RAG2_HELPFUL
    if policy.startswith("hidden_tau"):
        if hidden_projection is None or hidden_threshold is None:
            raise ValueError(f"Missing hidden projection/threshold for policy {policy}")
        return float(hidden_projection) > float(hidden_threshold)
    raise ValueError(f"Unknown oracle policy: {policy}")


def hidden_policy_name(threshold: float) -> str:
    rendered = format(float(threshold), ".8g").replace("-", "m").replace(".", "p")
    return f"hidden_tau_{rendered}"

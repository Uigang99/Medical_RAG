#!/usr/bin/env python3
"""Select semantic-candidate document subsets without consulting the gold answer.

The input is the completed exact-subset score cache produced by
``materialize_rag2_semantic_behavioral_subset_oracle.py``.  Every selection
function in this file receives only the four next-choice logits/probabilities
and subset identities.  Gold answers are read only after selection to audit
the resulting direct-choice accuracy and regret against the gold Oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_semantic_gold_free_subset_selection_v1"
CHOICES = ("A", "B", "C", "D")
POLICIES = (
    "gold_free_max_confidence",
    "gold_free_min_entropy",
    "gold_free_consensus_confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=6545)
    parser.add_argument("--expected-top-k", type=int, default=8)
    parser.add_argument(
        "--expected-candidate-semantic-labels",
        nargs="+",
        default=["direct_support", "supporting_evidence"],
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def finite_vector(value: Any, *, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def prediction_index(subset: dict[str, Any]) -> int:
    logits = finite_vector(subset.get("choice_logits"), length=4, field="choice_logits")
    return max(range(4), key=lambda index: (logits[index], -index))


def confidence_gap(subset: dict[str, Any]) -> float:
    logits = sorted(
        finite_vector(subset.get("choice_logits"), length=4, field="choice_logits"),
        reverse=True,
    )
    return logits[0] - logits[1]


def entropy(subset: dict[str, Any]) -> float:
    probabilities = finite_vector(
        subset.get("choice_probabilities"), length=4, field="choice_probabilities"
    )
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("choice_probabilities have non-positive mass")
    probabilities = [max(0.0, value / total) for value in probabilities]
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def subset_tie_key(subset: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    ranks = tuple(int(value) for value in subset.get("selected_document_ranks") or [])
    return len(ranks), ranks


def choose_max_confidence(subsets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose with logits only; gold labels are deliberately unavailable here."""

    return min(subsets, key=lambda subset: (-confidence_gap(subset), *subset_tie_key(subset)))


def choose_min_entropy(subsets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose with probabilities only; gold labels are deliberately unavailable here."""

    return min(subsets, key=lambda subset: (entropy(subset), *subset_tie_key(subset)))


def consensus_choice(subsets: Sequence[dict[str, Any]]) -> int:
    probability_sums = [0.0] * 4
    for subset in subsets:
        probabilities = finite_vector(
            subset.get("choice_probabilities"), length=4, field="choice_probabilities"
        )
        total = sum(probabilities)
        if total <= 0.0:
            raise ValueError("choice_probabilities have non-positive mass")
        for index, value in enumerate(probabilities):
            probability_sums[index] += value / total
    return max(range(4), key=lambda index: (probability_sums[index], -index))


def choose_consensus_confidence(subsets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose the most confident subset agreeing with probability consensus."""

    consensus = consensus_choice(subsets)
    agreeing = [subset for subset in subsets if prediction_index(subset) == consensus]
    if not agreeing:
        # This is mathematically possible when probability averaging produces
        # a class that is never the per-subset argmax.  Prefer the subset that
        # assigns that consensus class the most probability.
        return min(
            subsets,
            key=lambda subset: (
                -finite_vector(
                    subset.get("choice_probabilities"),
                    length=4,
                    field="choice_probabilities",
                )[consensus],
                -confidence_gap(subset),
                *subset_tie_key(subset),
            ),
        )
    return min(agreeing, key=lambda subset: (-confidence_gap(subset), *subset_tie_key(subset)))


SELECTORS: dict[str, Callable[[Sequence[dict[str, Any]]], dict[str, Any]]] = {
    "gold_free_max_confidence": choose_max_confidence,
    "gold_free_min_entropy": choose_min_entropy,
    "gold_free_consensus_confidence": choose_consensus_confidence,
}


def validate_question_row(
    row: dict[str, Any], *, expected_top_k: int, expected_labels: list[str]
) -> None:
    if int(row.get("top_k", -1)) != expected_top_k:
        raise RuntimeError(f"Top-k mismatch for {row.get('sample_key')}")
    if list(row.get("candidate_semantic_labels") or []) != expected_labels:
        raise RuntimeError(f"Semantic-candidate contract mismatch for {row.get('sample_key')}")
    documents = list(row.get("top_k_documents") or [])
    if len(documents) != expected_top_k:
        raise RuntimeError(f"Expected {expected_top_k} documents for {row.get('sample_key')}")
    candidates = list(row.get("semantic_candidates") or [])
    subsets = list(row.get("subsets") or [])
    expected_subsets = 1 << len(candidates)
    if len(subsets) != expected_subsets:
        raise RuntimeError(
            f"Subset-count mismatch for {row.get('sample_key')}: "
            f"{len(subsets)} != {expected_subsets}"
        )
    masks = {int(subset.get("mask", -1)) for subset in subsets}
    if masks != set(range(expected_subsets)):
        raise RuntimeError(f"Subset mask coverage mismatch for {row.get('sample_key')}")
    document_ids = [str(document.get("doc_stable_id") or "") for document in documents]
    if not all(document_ids) or len(document_ids) != len(set(document_ids)):
        raise RuntimeError(f"Invalid Top-k document identities for {row.get('sample_key')}")
    for subset in subsets:
        finite_vector(subset.get("choice_logits"), length=4, field="choice_logits")
        probabilities = finite_vector(
            subset.get("choice_probabilities"), length=4, field="choice_probabilities"
        )
        if sum(probabilities) <= 0.0:
            raise RuntimeError(f"Invalid probability mass for {row.get('sample_key')}")


def selection_record(policy: str, subset: dict[str, Any], consensus: int) -> dict[str, Any]:
    return {
        "selected_document_ids": list(subset.get("selected_document_ids") or []),
        "selected_document_ranks": list(subset.get("selected_document_ranks") or []),
        "subset_size": len(subset.get("selected_document_ids") or []),
        "prediction": CHOICES[prediction_index(subset)],
        "confidence_gap": confidence_gap(subset),
        "entropy": entropy(subset),
        "consensus_choice": CHOICES[consensus],
        "policy": policy,
        "gold_used_for_selection": False,
    }


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Gold-free semantic-candidate subset selector audit",
        "",
        f"- Questions: {summary['questions']:,}",
        f"- Exact scored subsets reused: {summary['subset_scores']:,}",
        "- Selection inputs: four choice logits/probabilities only; gold is audit-only",
        "",
        "| Policy | Direct-choice acc. | Avg docs | Empty | Exact Oracle subset | "
        "Mean gold-margin regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        values = summary["policies"][policy]
        lines.append(
            f"| {policy} | {values['direct_choice_accuracy'] * 100:.2f} | "
            f"{values['mean_selected_documents']:.3f} | "
            f"{values['empty_subset_rate'] * 100:.2f}% | "
            f"{values['exact_oracle_subset_rate'] * 100:.2f}% | "
            f"{values['mean_gold_margin_regret']:.4f} |"
        )
    audit = summary["audit_references"]
    lines.extend(
        [
            "",
            "| Direct-choice reference | Accuracy |",
            "|---|---:|",
            f"| Empty subset (No-RAG) | {audit['empty_subset_accuracy'] * 100:.2f} |",
            f"| All Direct+Supporting candidates | {audit['all_candidates_accuracy'] * 100:.2f} |",
            f"| Gold-margin best subset | {audit['gold_oracle_accuracy'] * 100:.2f} |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_summary_path = args.subset_score_root / "summary.json"
    if not source_summary_path.is_file():
        raise FileNotFoundError(source_summary_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    expected_labels = list(args.expected_candidate_semantic_labels)
    if source_summary.get("status") != "complete":
        raise RuntimeError("Exact-subset source cache is incomplete")
    if int(source_summary.get("questions", -1)) != args.expected_questions:
        raise RuntimeError("Exact-subset source cohort size mismatch")
    if int(source_summary.get("top_k", -1)) != args.expected_top_k:
        raise RuntimeError("Exact-subset source Top-k mismatch")
    if list(source_summary.get("candidate_semantic_labels") or []) != expected_labels:
        raise RuntimeError("Exact-subset semantic-candidate contract mismatch")
    shard_paths = sorted((args.subset_score_root / "score_shards").glob("*/*/shard_*/questions.jsonl"))
    if not shard_paths:
        raise FileNotFoundError("No exact-subset question shards found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    selections_partial = args.output_root / "question_selections.jsonl.partial"
    selections: dict[str, dict[str, Any]] = {}
    policy_counts = {policy: Counter() for policy in POLICIES}
    reference_counts = Counter()
    seen: set[str] = set()
    progress = PipelineProgress(
        overall_total=args.expected_questions * 2,
        desc="GoldFreeSubsetSelection",
    )
    try:
        progress.set_stage("1/2 validate cached logits and select gold-free subsets", total=args.expected_questions)
        with selections_partial.open("w", encoding="utf-8") as output:
            for shard_path in shard_paths:
                complete_path = shard_path.with_name("COMPLETE.json")
                if not complete_path.is_file():
                    raise RuntimeError(f"Subset shard is incomplete: {shard_path.parent}")
                for row in iter_jsonl(shard_path):
                    validate_question_row(
                        row,
                        expected_top_k=args.expected_top_k,
                        expected_labels=expected_labels,
                    )
                    sample_key = str(row.get("sample_key") or "")
                    if not sample_key or sample_key in seen:
                        raise RuntimeError(f"Duplicate or empty sample key: {sample_key!r}")
                    seen.add(sample_key)
                    subsets = list(row["subsets"])
                    consensus = consensus_choice(subsets)
                    selected = {
                        policy: selection_record(policy, selector(subsets), consensus)
                        for policy, selector in SELECTORS.items()
                    }
                    # The fields below are used only after each selector has
                    # returned.  They must never enter a selector function.
                    gold = str(row.get("gold_answer") or "")
                    if gold not in CHOICES:
                        raise RuntimeError(f"Invalid audit gold answer for {sample_key}")
                    oracle = dict((row.get("optima") or {}).get("behavioral_best_semantic_candidates") or {})
                    if not oracle:
                        raise RuntimeError(f"Missing gold Oracle optimum for {sample_key}")
                    empty = next(subset for subset in subsets if int(subset["mask"]) == 0)
                    all_candidates = next(
                        subset for subset in subsets if int(subset["mask"]) == len(subsets) - 1
                    )
                    reference_counts["questions"] += 1
                    reference_counts["subset_scores"] += len(subsets)
                    reference_counts["empty_correct"] += int(empty["prediction"] == gold)
                    reference_counts["all_correct"] += int(all_candidates["prediction"] == gold)
                    reference_counts["oracle_correct"] += int(oracle["prediction"] == gold)
                    for policy, value in selected.items():
                        counter = policy_counts[policy]
                        counter["questions"] += 1
                        counter["selected_documents"] += value["subset_size"]
                        counter["empty"] += int(value["subset_size"] == 0)
                        counter["correct"] += int(value["prediction"] == gold)
                        counter["exact_oracle"] += int(
                            value["selected_document_ids"] == oracle["selected_document_ids"]
                        )
                        chosen_subset = next(
                            subset
                            for subset in subsets
                            if list(subset.get("selected_document_ids") or [])
                            == value["selected_document_ids"]
                        )
                        counter["gold_margin_regret_sum"] += max(
                            0.0,
                            float(oracle["gold_margin"]) - float(chosen_subset["gold_margin"]),
                        )
                    record = {
                        "schema_version": RUN_VERSION,
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "row_idx": row["row_idx"],
                        "sample_id": row["sample_id"],
                        "sample_key": sample_key,
                        "top_k": row["top_k"],
                        "candidate_semantic_labels": row["candidate_semantic_labels"],
                        "selections": selected,
                        "audit": {
                            "gold_answer": gold,
                            "empty_prediction": empty["prediction"],
                            "all_candidates_prediction": all_candidates["prediction"],
                            "gold_oracle": oracle,
                        },
                    }
                    selections[sample_key] = record
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.update(1)
        if len(seen) != args.expected_questions:
            raise RuntimeError(f"Expected {args.expected_questions} questions, found {len(seen)}")
        os.replace(selections_partial, args.output_root / "question_selections.jsonl")

        progress.set_stage("2/2 materialize per-document decisions for all fixed policies", total=args.expected_questions)
        partial_paths = {
            policy: args.output_root / f"{policy}_labels.jsonl.partial" for policy in POLICIES
        }
        handles = {policy: path.open("w", encoding="utf-8") for policy, path in partial_paths.items()}
        try:
            for shard_path in shard_paths:
                for row in iter_jsonl(shard_path):
                    sample_key = str(row["sample_key"])
                    record = selections[sample_key]
                    for policy in POLICIES:
                        selection = record["selections"][policy]
                        selected_ids = set(selection["selected_document_ids"])
                        for document in row["top_k_documents"]:
                            is_selected = document["doc_stable_id"] in selected_ids
                            label = {
                                "schema_version": RUN_VERSION,
                                "sample_key": sample_key,
                                "dataset": row["dataset"],
                                "sample_id": row["sample_id"],
                                "row_idx": row["row_idx"],
                                "doc_rank": document["doc_rank"],
                                "doc_stable_id": document["doc_stable_id"],
                                "source": document["source"],
                                "semantic_label": document["semantic_label"],
                                "candidate_semantic_labels": row["candidate_semantic_labels"],
                                "selection_policy": policy,
                                "selected": is_selected,
                                "pseudo_label": "Helpful" if is_selected else "Not Helpful",
                                "selected_subset_document_ids": selection["selected_document_ids"],
                                "selected_subset_size": selection["subset_size"],
                                "selected_subset_prediction": selection["prediction"],
                                "selected_subset_confidence_gap": selection["confidence_gap"],
                                "selected_subset_entropy": selection["entropy"],
                                "subset_consensus_choice": selection["consensus_choice"],
                                "gold_used_for_selection": False,
                            }
                            handles[policy].write(json.dumps(label, ensure_ascii=False) + "\n")
                    progress.update(1)
        finally:
            for handle in handles.values():
                handle.close()
        for policy, partial in partial_paths.items():
            os.replace(partial, args.output_root / f"{policy}_labels.jsonl")
    finally:
        progress.close()

    question_count = reference_counts["questions"]
    summary: dict[str, Any] = {
        "run_version": RUN_VERSION,
        "status": "complete",
        "created_at": utc_now(),
        "source_subset_score_root": str(args.subset_score_root.resolve()),
        "questions": question_count,
        "top_k": args.expected_top_k,
        "candidate_semantic_labels": expected_labels,
        "subset_scores": reference_counts["subset_scores"],
        "selection_uses_gold": False,
        "policies": {},
        "audit_references": {
            "empty_subset_accuracy": reference_counts["empty_correct"] / question_count,
            "all_candidates_accuracy": reference_counts["all_correct"] / question_count,
            "gold_oracle_accuracy": reference_counts["oracle_correct"] / question_count,
        },
        "label_paths": {
            policy: str((args.output_root / f"{policy}_labels.jsonl").resolve())
            for policy in POLICIES
        },
    }
    for policy in POLICIES:
        counter = policy_counts[policy]
        summary["policies"][policy] = {
            "questions": counter["questions"],
            "label_rows": counter["questions"] * args.expected_top_k,
            "direct_choice_accuracy": counter["correct"] / question_count,
            "mean_selected_documents": counter["selected_documents"] / question_count,
            "empty_subset_rate": counter["empty"] / question_count,
            "exact_oracle_subset_rate": counter["exact_oracle"] / question_count,
            "mean_gold_margin_regret": counter["gold_margin_regret_sum"] / question_count,
        }
    atomic_json(args.output_root / "summary.json", summary)
    table = render_summary(summary)
    atomic_text(args.output_root / "summary_table_pretty.txt", table + "\n")
    print(table)
    print(f"Gold-free subset selections: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()

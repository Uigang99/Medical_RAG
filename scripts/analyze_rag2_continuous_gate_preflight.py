#!/usr/bin/env python3
"""Analyze semantic calibration, set headroom, and answer-mode agreement.

This script consumes only an internal question-level validation cohort.  Gold
answers are used for diagnostics and Oracle upper bounds, never as controller
inputs.  It deliberately compares no-rationale direct-choice scoring with the
cached rationale + fixed-terminal-answer traces before either answer protocol
is selected for subsequent training.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import logging
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.core import BenchmarkSample  # noqa: E402
from medrag.filtering.rag2_filter import Rag2FlanT5Filter  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_continuous_gate_preflight_analysis_v1"
SUPPORT_LABELS = {"direct_support", "supporting_evidence"}
NONSUPPORT_LABELS = {"no_evidence", "misleading_evidence"}
DEFAULT_BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--semantic-model-medmcqa", type=Path, required=True)
    parser.add_argument("--semantic-model-medqa", type=Path, required=True)
    parser.add_argument("--no-rag-trace-root", type=Path, default=DEFAULT_BASE / "train_no_rag_anchored_features_v1/trace_shards")
    parser.add_argument("--document-trace-root", type=Path, default=DEFAULT_BASE / "document_traces_source_balanced32_rerank8_v1/trace_shards")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--semantic-max-input-length", type=int, default=1280)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--risk-kappas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("chunk_id") or document.get("db_id")
    if not value:
        raise ValueError("Document has no stable identity")
    return str(value)


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(rankdata(left), rankdata(right))


def mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def median_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def binary_calibration(labels: list[int], probabilities: list[float], bins: int) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    predictions = (p >= 0.5).astype(np.int64)
    ece = 0.0
    rows = []
    for index in range(bins):
        left = index / bins
        right = (index + 1) / bins
        mask = (p >= left) & ((p < right) if index + 1 < bins else (p <= right))
        count = int(mask.sum())
        if count == 0:
            rows.append({"left": left, "right": right, "count": 0})
            continue
        confidence = float(p[mask].mean())
        frequency = float(y[mask].mean())
        ece += count / len(y) * abs(confidence - frequency)
        rows.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "mean_probability": confidence,
                "support_frequency": frequency,
            }
        )
    return {
        "rows": len(labels),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": float((predictions == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "auroc": float(roc_auc_score(y, p)) if len(set(labels)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if len(set(labels)) == 2 else None,
        "reliability_bins": rows,
    }


def benchmark_sample(row: dict[str, Any]) -> BenchmarkSample:
    answer = str(row.get("answer") or "").upper()
    return BenchmarkSample(
        row_idx=int(row.get("row_idx", -1)),
        id=str(row["sample_id"]),
        task="mcq",
        collection="unified",
        dataset=str(row["dataset"]),
        split=str(row.get("split") or "val"),
        question=str(row["question"]),
        options={str(key): str(value) for key, value in (row.get("options") or {}).items()},
        answer=answer,
        answers=[answer],
        raw=row,
    )


def semantic_model_path(args: argparse.Namespace, dataset: str) -> Path:
    return args.semantic_model_medmcqa if dataset == "medmcqa" else args.semantic_model_medqa


def load_cohort(args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    path = args.cohort_root / "candidate_union" / dataset / args.split / "candidates_topk_union.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"Empty preflight cohort: {path}")
    return rows


def score_semantic_probabilities(
    args: argparse.Namespace,
    dataset: str,
    rows: list[dict[str, Any]],
    progress: PipelineProgress,
) -> dict[tuple[str, str], dict[str, Any]]:
    cache_path = args.output_root / "semantic_predictions" / f"{dataset}.jsonl"
    expected = sum(len(row["candidate_documents"]) for row in rows)
    if cache_path.is_file() and args.resume:
        cached = list(iter_jsonl(cache_path))
        if len(cached) == expected:
            logging.info("Reusing semantic prediction cache: %s", cache_path)
            progress.set_stage(f"1/3 semantic calibration {dataset} (cached)", total=expected, initial=expected)
            progress.update(expected)
            return {(str(row["sample_id"]), str(row["doc_stable_id"])): row for row in cached}

    samples: list[BenchmarkSample] = []
    evidences: list[str] = []
    identities: list[tuple[str, str, int]] = []
    gold_labels: dict[tuple[str, str], str] = {}
    semantic_path = args.cohort_root / "semantic_labels" / f"{dataset}.jsonl"
    for row in iter_jsonl(semantic_path):
        gold_labels[(str(row["sample_id"]), str(row["doc_stable_id"]))] = str(row["semantic_label"])
    for row in rows:
        sample = benchmark_sample(row)
        for rank, document in enumerate(row["candidate_documents"], start=1):
            document_id = stable_id(document)
            samples.append(sample)
            evidences.append(str(document.get("text") or ""))
            identities.append((sample.id, document_id, rank))
    progress.set_stage(f"1/3 semantic calibration {dataset}", total=expected)
    filterer = Rag2FlanT5Filter(
        model_path=semantic_model_path(args, dataset),
        batch_size=args.semantic_batch_size,
        max_input_length=args.semantic_max_input_length,
        max_doc_chars=0,
        device=args.device,
        bf16=args.dtype == "bfloat16",
        scoring_method="special_token",
        input_format="official",
    )
    try:
        scores = filterer.score_evidences(samples, evidences, progress_callback=progress.update)
    finally:
        filterer.close()
        del filterer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = []
    for (sample_id, document_id, rank), score in zip(identities, scores, strict=True):
        label = gold_labels.get((sample_id, document_id))
        if label is None:
            raise KeyError(f"Missing semantic gold label for {(sample_id, document_id)}")
        output.append(
            {
                "run_version": RUN_VERSION,
                "dataset": dataset,
                "sample_id": sample_id,
                "doc_stable_id": document_id,
                "doc_rank": rank,
                "semantic_label": label,
                "prob_support": float(score["prob_helpful"]),
                "semantic_margin": float(score["margin"]),
                "semantic_prediction": str(score["prediction"]),
            }
        )
    atomic_jsonl(cache_path, output)
    return {(row["sample_id"], row["doc_stable_id"]): row for row in output}


def load_subset_rows(args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    roots = sorted((args.subset_root / "score_shards" / dataset / args.split).glob("shard_*/questions.jsonl"))
    if not roots:
        raise FileNotFoundError(f"No subset score shards for {dataset}/{args.split}: {args.subset_root}")
    rows = []
    for path in roots:
        rows.extend(iter_jsonl(path))
    return rows


def subset_by_mask(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = {int(subset["mask"]): subset for subset in row["subsets"]}
    expected = 1 << len(row["semantic_candidates"])
    if set(values) != set(range(expected)):
        raise RuntimeError(f"Incomplete subset lattice for {row['sample_id']}")
    return values


def best_subset(subsets: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return max(
        subsets,
        key=lambda subset: (
            float(subset["gold_margin"]),
            -len(subset["selected_document_ids"]),
            -int(subset["mask"]),
        ),
    )


def policy_record(name: str, subset: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": name,
        "correct": str(subset["prediction"]) == str(subset.get("gold_answer", "")),
        "prediction": subset["prediction"],
        "gold_margin": float(subset["gold_margin"]),
        "documents": len(subset["selected_document_ids"]),
        "mask": int(subset["mask"]),
    }


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "mean_gold_margin": statistics.fmean(float(row["gold_margin"]) for row in rows),
        "mean_documents": statistics.fmean(int(row["documents"]) for row in rows),
        "empty_rate": sum(int(row["documents"]) == 0 for row in rows) / len(rows),
    }


def transition_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    transitions: Counter[str] = Counter()
    deltas = []
    for row in records:
        before = bool(row["before_correct"])
        after = bool(row["after_correct"])
        transitions[("C" if before else "W") + "->" + ("C" if after else "W")] += 1
        deltas.append(float(row["margin_delta"]))
    count = len(records)
    return {
        "count": count,
        "mean_margin_delta": statistics.fmean(deltas),
        "median_margin_delta": statistics.median(deltas),
        "positive_margin_rate": sum(delta > 0 for delta in deltas) / count,
        "accuracy_delta": (transitions["W->C"] - transitions["C->W"]) / count,
        "transitions": dict(transitions),
    }


def analyze_subsets(
    args: argparse.Namespace,
    dataset: str,
    rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, Any]],
    progress: PipelineProgress,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    policy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conditional: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    subset_index: dict[str, dict[int, dict[str, Any]]] = {}
    progress.set_stage(f"2/3 subset headroom and conditional effects {dataset}", total=len(rows))
    for row in rows:
        documents = list(row["semantic_candidates"])
        if len(documents) != args.top_k:
            raise RuntimeError(f"Expected Top-{args.top_k} exact subsets for {row['sample_id']}")
        subsets = subset_by_mask(row)
        subset_index[str(row["sample_id"])] = subsets
        gold = str(row["gold_answer"])
        for subset in subsets.values():
            subset["gold_answer"] = gold
        all_mask = (1 << len(documents)) - 1
        support_mask = sum(
            1 << index
            for index, document in enumerate(documents)
            if document["semantic_label"] in SUPPORT_LABELS
        )
        no_rag = subsets[0]
        all_documents = subsets[all_mask]
        hard_semantic = subsets[support_mask]
        oracle_all = best_subset(subsets.values())
        oracle_semantic = best_subset(
            subset for mask, subset in subsets.items() if mask & ~support_mask == 0
        )
        for name, subset in (
            ("no_rag", no_rag),
            ("all_documents", all_documents),
            ("hard_semantic", hard_semantic),
            ("oracle_all", oracle_all),
            ("oracle_semantic", oracle_semantic),
        ):
            policy_rows[name].append(policy_record(name, subset))

        support_probabilities = [
            float(predictions[(str(row["sample_id"]), str(document["doc_stable_id"]))]["prob_support"])
            for document in documents
        ]
        for kappa in args.risk_kappas:
            eligible = []
            for mask, subset in subsets.items():
                if mask == 0:
                    eligible.append(subset)
                    continue
                semantic_mass = sum(
                    support_probabilities[index]
                    for index in range(len(documents))
                    if mask & (1 << index)
                )
                nonsupport_mass = sum(
                    1.0 - support_probabilities[index]
                    for index in range(len(documents))
                    if mask & (1 << index)
                )
                if nonsupport_mass <= float(kappa) * semantic_mass + 1e-12:
                    eligible.append(subset)
            name = f"oracle_risk_kappa_{kappa:g}"
            policy_rows[name].append(policy_record(name, best_subset(eligible)))

        support_indices = [
            index for index, document in enumerate(documents) if document["semantic_label"] in SUPPORT_LABELS
        ]
        nonsupport_indices = [
            index for index, document in enumerate(documents) if document["semantic_label"] in NONSUPPORT_LABELS
        ]
        support_lattice = [0]
        for local_mask in range(1, 1 << len(support_indices)):
            actual_mask = sum(
                1 << support_indices[position]
                for position in range(len(support_indices))
                if local_mask & (1 << position)
            )
            support_lattice.append(actual_mask)
        for base_mask in support_lattice:
            before = subsets[base_mask]
            base_group = "support_present" if base_mask else "support_absent"
            for index in nonsupport_indices:
                after = subsets[base_mask | (1 << index)]
                label = str(documents[index]["semantic_label"])
                conditional[(label, base_group)].append(
                    {
                        "before_correct": str(before["prediction"]) == gold,
                        "after_correct": str(after["prediction"]) == gold,
                        "margin_delta": float(after["gold_margin"]) - float(before["gold_margin"]),
                    }
                )
        progress.update(1)

    policies = {name: summarize_policy(values) for name, values in sorted(policy_rows.items())}
    conditional_summary = {
        f"{label}|{base}": transition_stats(values)
        for (label, base), values in sorted(conditional.items())
    }
    hard_accuracy = policies["hard_semantic"]["accuracy"]
    all_oracle_accuracy = policies["oracle_all"]["accuracy"]
    best_risk_name = max(
        (name for name in policies if name.startswith("oracle_risk_kappa_")),
        key=lambda name: policies[name]["accuracy"],
    )
    best_risk_accuracy = policies[best_risk_name]["accuracy"]
    available_gain = max(1e-12, all_oracle_accuracy - hard_accuracy)
    retained_gain = (best_risk_accuracy - hard_accuracy) / available_gain
    no_absent = conditional_summary.get("no_evidence|support_absent", {})
    no_present = conditional_summary.get("no_evidence|support_present", {})
    conditional_difference = (
        float(no_present.get("accuracy_delta", 0.0))
        - float(no_absent.get("accuracy_delta", 0.0))
    )
    checks = {
        "risk_oracle_beats_hard_semantic_by_2pp": best_risk_accuracy >= hard_accuracy + 0.02,
        "risk_oracle_retains_half_available_gain": retained_gain >= 0.5,
        "no_evidence_is_at_least_2pp_safer_with_support": conditional_difference >= 0.02,
    }
    return (
        {
            "policies": policies,
            "conditional_addition": conditional_summary,
            "best_risk_policy": best_risk_name,
            "best_risk_gain_vs_hard_semantic": best_risk_accuracy - hard_accuracy,
            "best_risk_fraction_of_oracle_gain": retained_gain,
            "no_evidence_support_condition_accuracy_delta_difference": conditional_difference,
            "checks": checks,
        },
        subset_index,
    )


def gold_logprob(logprobs: dict[str, Any], gold: str) -> float | None:
    """Return the exact constrained gold-choice log probability when cached.

    Some legacy rationale traces contain a null value for one impossible
    choice, so a four-way margin cannot be reconstructed reliably.  The gold
    value itself is still exact and is directly comparable to log of the
    direct-choice scorer's saved gold probability.
    """

    value = logprobs.get(gold)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@functools.lru_cache(maxsize=32)
def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(iter_jsonl(path))


def analyze_answer_modes(
    args: argparse.Namespace,
    dataset: str,
    cohort: list[dict[str, Any]],
    subset_index: dict[str, dict[int, dict[str, Any]]],
    progress: PipelineProgress,
) -> dict[str, Any]:
    progress.set_stage(f"3/3 direct-choice vs rationale singleton utility {dataset}", total=len(cohort))
    global_direct: list[float] = []
    global_rationale: list[float] = []
    per_question_spearman: list[float | None] = []
    sign_agreements = 0
    sign_total = 0
    no_rag_direct_correct = 0
    no_rag_rationale_correct = 0
    oracle_direct_correct = 0
    oracle_rationale_correct = 0
    rationale_quality_failures = 0
    valid_questions = 0
    for row in cohort:
        sample_id = str(row["sample_id"])
        row_idx = int(row["row_idx"])
        no_rag_shard_index = row_idx // 256
        document_shard_index = row_idx // 128
        no_rag_path = (
            args.no_rag_trace_root
            / dataset
            / args.source_split
            / f"shard_{no_rag_shard_index:05d}"
            / "questions.jsonl"
        )
        document_path = (
            args.document_trace_root
            / dataset
            / args.source_split
            / f"shard_{document_shard_index:05d}"
            / "pairs.jsonl"
        )
        no_rag_matches = [trace for trace in load_trace_rows(no_rag_path) if str(trace["sample_id"]) == sample_id]
        document_matches = [trace for trace in load_trace_rows(document_path) if str(trace["sample_id"]) == sample_id]
        if len(no_rag_matches) != 1:
            raise RuntimeError(f"Expected one no-RAG trace for {sample_id}, got {len(no_rag_matches)}")
        no_trace = no_rag_matches[0]
        gold = str(row["answer"]).upper()
        no_rationale_logprob = gold_logprob(no_trace.get("choice_logprobs") or {}, gold)
        if no_rationale_logprob is None:
            rationale_quality_failures += 1
            progress.update(1)
            continue
        by_id = {
            stable_id(trace.get("document") or {}): trace
            for trace in document_matches
        }
        direct_subsets = subset_index[sample_id]
        direct_empty = direct_subsets[0]
        direct_utilities: list[float] = []
        rationale_utilities: list[float] = []
        rationale_predictions = [str(no_trace.get("answer") or "")]
        rationale_scores = [float(no_rationale_logprob)]
        valid = True
        for index, document in enumerate(row["candidate_documents"]):
            document_id = stable_id(document)
            trace = by_id.get(document_id)
            if trace is None:
                raise KeyError(f"Missing cached rationale trace for {(sample_id, document_id)}")
            document_gold_logprob = gold_logprob(trace.get("choice_logprobs") or {}, gold)
            if document_gold_logprob is None:
                valid = False
                break
            direct_singleton = direct_subsets[1 << index]
            direct_utility = math.log(max(float(direct_singleton["gold_probability"]), 1e-30)) - math.log(
                max(float(direct_empty["gold_probability"]), 1e-30)
            )
            rationale_utility = float(document_gold_logprob) - float(no_rationale_logprob)
            direct_utilities.append(direct_utility)
            rationale_utilities.append(rationale_utility)
            rationale_predictions.append(str(trace.get("answer") or ""))
            rationale_scores.append(float(document_gold_logprob))
            if trace.get("quality_flags"):
                rationale_quality_failures += 1
        if not valid:
            rationale_quality_failures += 1
            progress.update(1)
            continue
        valid_questions += 1
        global_direct.extend(direct_utilities)
        global_rationale.extend(rationale_utilities)
        per_question_spearman.append(spearman(direct_utilities, rationale_utilities))
        for direct, rationale in zip(direct_utilities, rationale_utilities, strict=True):
            if abs(direct) <= 1e-12 or abs(rationale) <= 1e-12:
                continue
            sign_total += 1
            sign_agreements += int((direct > 0) == (rationale > 0))
        no_rag_direct_correct += int(str(direct_empty["prediction"]) == gold)
        no_rag_rationale_correct += int(str(no_trace.get("answer")) == gold)
        direct_singletons = [direct_empty] + [direct_subsets[1 << index] for index in range(args.top_k)]
        oracle_direct_correct += int(str(best_subset(direct_singletons)["prediction"]) == gold)
        rationale_best = max(range(len(rationale_scores)), key=rationale_scores.__getitem__)
        oracle_rationale_correct += int(rationale_predictions[rationale_best] == gold)
        progress.update(1)
    if valid_questions == 0:
        raise RuntimeError(f"No valid answer-mode comparisons for {dataset}")
    global_rank = spearman(global_direct, global_rationale)
    median_rank = median_or_none(per_question_spearman)
    sign_rate = sign_agreements / sign_total if sign_total else None
    checks = {
        "median_question_spearman_at_least_0p5": median_rank is not None and median_rank >= 0.5,
        "global_sign_agreement_at_least_0p7": sign_rate is not None and sign_rate >= 0.7,
    }
    return {
        "questions": valid_questions,
        "document_pairs": len(global_direct),
        "global_utility_spearman": global_rank,
        "median_per_question_utility_spearman": median_rank,
        "utility_sign_agreement": sign_rate,
        "no_rag_accuracy": {
            "direct_choice": no_rag_direct_correct / valid_questions,
            "rationale_fixed_terminal": no_rag_rationale_correct / valid_questions,
        },
        "best_of_empty_plus_singletons_accuracy": {
            "direct_choice": oracle_direct_correct / valid_questions,
            "rationale_fixed_terminal": oracle_rationale_correct / valid_questions,
        },
        "rationale_quality_flag_count": rationale_quality_failures,
        "checks": checks,
        "interpretation": (
            "High agreement permits a direct-choice teacher with rationale evaluation. "
            "Low agreement means the protocols define different target-model behavior and must not be mixed."
        ),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "analysis_summary.json"
    if summary_path.is_file() and args.resume:
        logging.info("Complete preflight analysis already exists: %s", summary_path)
        return
    cohorts = {dataset: load_cohort(args, dataset) for dataset in args.datasets}
    total_units = sum(
        2 * len(rows) + sum(len(row["candidate_documents"]) for row in rows)
        for rows in cohorts.values()
    )
    progress = PipelineProgress(overall_total=total_units, desc="AnalyzeGatePreflight")
    dataset_summaries: dict[str, Any] = {}
    try:
        for dataset in args.datasets:
            cohort = cohorts[dataset]
            predictions = score_semantic_probabilities(args, dataset, cohort, progress)
            calibration_labels = []
            calibration_probs = []
            for prediction in predictions.values():
                label = str(prediction["semantic_label"])
                if label not in SUPPORT_LABELS | NONSUPPORT_LABELS:
                    continue
                calibration_labels.append(int(label in SUPPORT_LABELS))
                calibration_probs.append(float(prediction["prob_support"]))
            calibration = binary_calibration(calibration_labels, calibration_probs, args.ece_bins)
            calibration["checks"] = {
                "ece_at_most_0p10": calibration["ece"] <= 0.10,
                "brier_at_most_0p20": calibration["brier"] <= 0.20,
                "auroc_at_least_0p75": calibration["auroc"] is not None and calibration["auroc"] >= 0.75,
            }
            subset_rows = load_subset_rows(args, dataset)
            if len(subset_rows) != len(cohort):
                raise RuntimeError(
                    f"Subset/cohort question mismatch for {dataset}: {len(subset_rows)} != {len(cohort)}"
                )
            subset_summary, subset_index = analyze_subsets(
                args, dataset, subset_rows, predictions, progress
            )
            answer_modes = analyze_answer_modes(
                args, dataset, cohort, subset_index, progress
            )
            dataset_summaries[dataset] = {
                "questions": len(cohort),
                "semantic_calibration": calibration,
                "subset_analysis": subset_summary,
                "answer_mode_analysis": answer_modes,
            }
    finally:
        progress.close()
    all_calibrated = all(
        all(summary["semantic_calibration"]["checks"].values())
        for summary in dataset_summaries.values()
    )
    all_headroom = all(
        summary["subset_analysis"]["checks"]["risk_oracle_beats_hard_semantic_by_2pp"]
        and summary["subset_analysis"]["checks"]["risk_oracle_retains_half_available_gain"]
        for summary in dataset_summaries.values()
    )
    conditional_replication = all(
        summary["subset_analysis"]["checks"]["no_evidence_is_at_least_2pp_safer_with_support"]
        for summary in dataset_summaries.values()
    )
    answer_modes_agree = all(
        all(summary["answer_mode_analysis"]["checks"].values())
        for summary in dataset_summaries.values()
    )
    if answer_modes_agree:
        answer_recommendation = (
            "Direct-choice is acceptable as the cheaper teacher diagnostic, but final evaluation may retain "
            "rationale + fixed terminal answer because both protocols induce similar singleton utility orderings."
        )
    else:
        answer_recommendation = (
            "Direct-choice and rationale define materially different utility targets. Choose one protocol before "
            "training and use it consistently for teacher construction, baselines, and final evaluation; do not mix them."
        )
    summary = {
        "run_version": RUN_VERSION,
        "scope": "internal validation only; no final benchmark test data",
        "datasets": dataset_summaries,
        "cross_dataset_checks": {
            "semantic_probabilities_usable_without_calibration": all_calibrated,
            "risk_constrained_policy_has_oracle_headroom": all_headroom,
            "conditional_no_evidence_hypothesis_replicates": conditional_replication,
            "direct_and_rationale_utility_targets_agree": answer_modes_agree,
        },
        "answer_mode_recommendation": answer_recommendation,
        "provisional_decision_before_attention_contract": (
            "GO" if all_calibrated and all_headroom and conditional_replication else "STOP_OR_REVISE"
        ),
    }
    atomic_json(summary_path, summary)
    logging.info("Continuous-gate preflight analysis complete: %s", summary_path)
    logging.info("Provisional decision: %s", summary["provisional_decision_before_attention_contract"])


if __name__ == "__main__":
    main()

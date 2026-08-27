#!/usr/bin/env python3
"""Convert external-test gold-margin scores into three-class Oracle labels.

The input scores are exact cached A/B/C/D logit quantities.  For utility
``u = sigmoid(m_D / T) - sigmoid(m_0 / T)``, this script assigns:

* Helpful when ``u >= tau``;
* Harmful when ``u <= -tau``; and
* Neutral otherwise.

Only Helpful is passed by the downstream Oracle evaluator.  The dynamic
Top-k candidate identity and membership are retained so all k conditions use
the same independently labelled question-document pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_margin_utility_external_oracle_labels_v1"
SCORE_VERSION = "rag2_anchored_gold_margin_scores_v1"
DATASETS = (
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate-file", default="candidates_topk_union.jsonl")
    parser.add_argument("--utility-threshold", type=float, default=0.1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    if not value:
        value = f"{document.get('source')}:{document.get('local_id')}"
    return str(value)


def candidate_counts(root: Path, datasets: list[str], split: str) -> tuple[dict[str, int], dict[str, int]]:
    questions: dict[str, int] = {}
    pairs: dict[str, int] = {}
    for dataset in datasets:
        manifest = read_json(root / dataset / split / "candidate_manifest.json")
        if manifest.get("candidate_protocol") != "rag2_paper_balanced_dynamic_topk_union_v1":
            raise RuntimeError(f"Dynamic candidate contract mismatch for {dataset}")
        if list(manifest.get("dynamic_top_k_values") or []) != [1, 2, 4, 8, 16, 32]:
            raise RuntimeError(f"Dynamic Top-k values mismatch for {dataset}")
        questions[dataset] = int(manifest["selected_question_count"])
        pairs[dataset] = int(manifest["selected_pair_count"])
    return questions, pairs


def completed_output_matches(args: argparse.Namespace, expected_pairs: int) -> bool:
    manifest_path = args.output_root / "manifest.json"
    labels_path = args.output_root / "margin_utility_oracle_labels.jsonl"
    if not args.resume or not manifest_path.is_file() or not labels_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("run_version") == RUN_VERSION
        and int(manifest.get("pairs", -1)) == expected_pairs
        and manifest.get("datasets") == args.datasets
        and manifest.get("split") == args.split
        and math.isclose(
            float(manifest.get("utility_threshold", float("nan"))),
            args.utility_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def label_for(utility: float, threshold: float) -> str:
    if utility >= threshold:
        return "Helpful"
    if utility <= -threshold:
        return "Harmful"
    return "Neutral"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not 0.0 <= args.utility_threshold < 1.0 or not math.isfinite(args.utility_threshold):
        raise ValueError("--utility-threshold must be finite and in [0, 1)")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets are not allowed")

    score_manifest = read_json(args.score_root / "manifest.json")
    if score_manifest.get("run_version") != SCORE_VERSION:
        raise RuntimeError(f"Score run-version mismatch: {score_manifest.get('run_version')!r}")
    if score_manifest.get("candidate_contract") != "dynamic_topk_union":
        raise RuntimeError("Scores were not built from the dynamic external-test candidate union")
    if score_manifest.get("source_split") != args.split:
        raise RuntimeError("Score split mismatch")
    if list(score_manifest.get("datasets") or []) != args.datasets:
        raise RuntimeError("Score dataset ordering/coverage mismatch")

    question_counts, pair_counts = candidate_counts(args.candidate_root, args.datasets, args.split)
    expected_questions = sum(question_counts.values())
    expected_pairs = sum(pair_counts.values())
    if int(score_manifest.get("pairs", -1)) != expected_pairs:
        raise RuntimeError(f"Score pair count mismatch: {score_manifest.get('pairs')} != {expected_pairs}")
    if int(score_manifest.get("questions_in_no_rag_cache", -1)) != expected_questions:
        raise RuntimeError("Score question count mismatch")

    logging.info(
        "Margin-utility Oracle label plan: questions=%d pairs=%d tau=%.6g",
        expected_questions,
        expected_pairs,
        args.utility_threshold,
    )
    if completed_output_matches(args, expected_pairs):
        logging.info("Complete matching label cache exists; reusing: %s", args.output_root)
        return
    if args.dry_run:
        logging.info("Dry run complete; score and candidate contracts are valid.")
        return

    progress = PipelineProgress(
        overall_total=expected_questions + expected_pairs,
        desc="MarginUtilityOracleLabels",
    )
    candidate_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        progress.set_stage("1/2 index dynamic Top-k candidate identities", total=expected_questions)
        for dataset in args.datasets:
            path = args.candidate_root / dataset / args.split / args.candidate_file
            observed_pairs = 0
            for row in iter_jsonl(path):
                sample_id = str(row.get("sample_id") or "")
                sample_key = str(row.get("key") or "")
                if not sample_id or not sample_key:
                    raise ValueError(f"Candidate row lacks sample identity: {dataset}")
                selected_by_k = row.get("selected_document_ids_by_top_k") or {}
                for rank, document in enumerate(row.get("candidate_documents") or [], 1):
                    identifier = stable_id(document)
                    key = (dataset, sample_id, identifier)
                    if key in candidate_index:
                        raise ValueError(f"Duplicate candidate identity: {key}")
                    metadata = document.get("metadata") or {}
                    membership = list(metadata.get("oracle_dynamic_top_k_membership") or [])
                    ranks = dict(metadata.get("oracle_dynamic_rerank_rank_by_top_k") or {})
                    # Recheck membership against the question-level selection map.
                    expected_membership = [
                        int(top_k)
                        for top_k, identifiers in selected_by_k.items()
                        if identifier in identifiers
                    ]
                    if sorted(membership) != sorted(expected_membership):
                        raise RuntimeError(f"Dynamic membership mismatch: {key}")
                    candidate_index[key] = {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "sample_key": sample_key,
                        "row_idx": row.get("row_idx"),
                        "doc_rank": rank,
                        "doc_stable_id": identifier,
                        "source": document.get("source"),
                        "db_id": document.get("db_id"),
                        "local_id": document.get("local_id"),
                        "dynamic_top_k_membership": membership,
                        "dynamic_rerank_rank_by_top_k": ranks,
                    }
                    observed_pairs += 1
                progress.update()
                progress.set_detail(f"dataset={dataset}")
            if observed_pairs != pair_counts[dataset]:
                raise RuntimeError(f"[{dataset}] candidate pairs {observed_pairs} != {pair_counts[dataset]}")
        if len(candidate_index) != expected_pairs:
            raise RuntimeError(f"Candidate index {len(candidate_index)} != {expected_pairs}")

        args.output_root.mkdir(parents=True, exist_ok=True)
        combined_path = args.output_root / "margin_utility_oracle_labels.jsonl"
        combined_partial = combined_path.with_name(combined_path.name + ".partial")
        dataset_paths: dict[str, Path] = {}
        dataset_handles: dict[str, Any] = {}
        for dataset in args.datasets:
            path = args.output_root / dataset / args.split / "margin_utility_oracle_labels.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset_paths[dataset] = path
            dataset_handles[dataset] = path.with_name(path.name + ".partial").open(
                "w", encoding="utf-8", buffering=16 * 1024 * 1024
            )

        label_counts = {dataset: Counter() for dataset in args.datasets}
        transition_counts = {dataset: Counter() for dataset in args.datasets}
        seen: set[tuple[str, str, str]] = set()
        progress.set_stage("2/2 apply utility threshold and write Oracle labels", total=expected_pairs)
        with combined_partial.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as combined:
            for dataset in args.datasets:
                roots = sorted((args.score_root / "score_shards" / dataset / args.split).glob("shard_*"))
                if not roots:
                    raise FileNotFoundError(args.score_root / "score_shards" / dataset / args.split)
                for root in roots:
                    if not (root / "COMPLETE.json").is_file():
                        raise RuntimeError(f"Incomplete score shard: {root}")
                    for score in iter_jsonl(root / "rows.jsonl"):
                        key = (
                            dataset,
                            str(score.get("sample_id") or ""),
                            str(score.get("document_stable_id") or ""),
                        )
                        candidate = candidate_index.get(key)
                        if candidate is None or key in seen:
                            raise ValueError(f"Unknown/duplicate score identity: {key}")
                        seen.add(key)
                        utility = float(score["boundary_probability_delta"])
                        no_margin = float(score["no_document_gold_margin"])
                        doc_margin = float(score["with_document_gold_margin"])
                        finite = all(math.isfinite(value) for value in (utility, no_margin, doc_margin))
                        label = label_for(utility, args.utility_threshold) if finite else "Excluded"
                        output = {
                            "schema_version": 1,
                            "policy": "margin_utility_three_class_tau_external_dynamic_topk_v1",
                            **candidate,
                            "pair_id": score.get("pair_id"),
                            "pseudo_label": label,
                            "quality_pass": finite,
                            "quality_failures": [] if finite else ["non_finite_margin_score"],
                            "utility_score": utility if finite else None,
                            "utility_threshold": args.utility_threshold,
                            "utility_definition": "sigmoid(m_D/T)-sigmoid(m_0/T)",
                            "temperature": float(score_manifest["temperature"]),
                            "no_document_gold_margin": no_margin if finite else None,
                            "with_document_gold_margin": doc_margin if finite else None,
                            "gold_margin_delta": score.get("gold_margin_delta"),
                            "answer_transition": score.get("answer_transition"),
                            "no_document_correct": score.get("no_document_correct"),
                            "with_document_correct": score.get("with_document_correct"),
                            "gold_answer": score.get("gold_answer"),
                            "source_quality_flags": score.get("quality_flags") or [],
                        }
                        serialized = json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
                        combined.write(serialized)
                        dataset_handles[dataset].write(serialized)
                        label_counts[dataset][label] += 1
                        transition_counts[dataset][str(score.get("answer_transition"))] += 1
                        progress.update()
                        progress.set_detail(f"dataset={dataset} shard={root.name}")
            combined.flush()
            os.fsync(combined.fileno())

        for dataset, handle in dataset_handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            partial = dataset_paths[dataset].with_name(dataset_paths[dataset].name + ".partial")
            os.replace(partial, dataset_paths[dataset])
        os.replace(combined_partial, combined_path)
        if len(seen) != expected_pairs:
            raise RuntimeError(f"Score coverage {len(seen)} != {expected_pairs}")

        atomic_json(
            args.output_root / "manifest.json",
            {
                "run_version": RUN_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "policy": "margin_utility",
                "split": args.split,
                "datasets": args.datasets,
                "questions": expected_questions,
                "pairs": expected_pairs,
                "questions_by_dataset": question_counts,
                "pairs_by_dataset": pair_counts,
                "utility_threshold": args.utility_threshold,
                "temperature": float(score_manifest["temperature"]),
                "label_counts": {dataset: dict(label_counts[dataset]) for dataset in args.datasets},
                "answer_transitions": {
                    dataset: dict(transition_counts[dataset]) for dataset in args.datasets
                },
                "oracle_action": "pass Helpful only; block Neutral/Harmful/Excluded; no backfill",
                "candidate_root": str(args.candidate_root.resolve()),
                "score_root": str(args.score_root.resolve()),
                "combined_labels_path": str(combined_path.resolve()),
            },
        )
        logging.info("Margin-utility Oracle labels complete: %s", combined_path)
    finally:
        for handle in locals().get("dataset_handles", {}).values():
            if not handle.closed:
                handle.close()
        progress.close()


if __name__ == "__main__":
    main()

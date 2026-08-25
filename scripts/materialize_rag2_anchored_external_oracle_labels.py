#!/usr/bin/env python3
"""Apply frozen train-derived RAG2 thresholds to anchored MCQ test traces."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_rag2_anchored_paper_labels import (  # noqa: E402
    LABEL_EXCLUDED,
    answer_correct,
    assign_label,
    document_failures,
    no_rag_failures,
    rationale_ppl,
)
from generate_rag2_anchored_document_traces import document_pair_id  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402


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
    parser.add_argument("--document-trace-root", type=Path, required=True)
    parser.add_argument("--no-rag-root", type=Path, required=True)
    parser.add_argument("--training-label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate-file", default="candidates_topk_union.jsonl")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    if not value:
        value = f"{document.get('source')}:{document.get('local_id')}"
    return str(value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_tau(training_label_root: Path, route: str) -> tuple[float, dict[str, Any]]:
    path = training_label_root / route / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = manifest.get("label_protocol") or {}
    tau = float(protocol["tau"])
    if not math.isfinite(tau):
        raise ValueError(f"Invalid frozen tau in {path}: {tau}")
    expected = {
        "hidden_state_features_used": False,
        "teacher_forcing_used": False,
        "threshold_quantile": 0.75,
    }
    actual = {key: protocol.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Training RAG2 label protocol mismatch in {path}: {actual} != {expected}")
    return tau, manifest


def load_counts(root: Path) -> tuple[dict[str, int], dict[str, int]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    questions = {str(key): int(value) for key, value in (manifest.get("questions_by_dataset") or {}).items()}
    pairs = {str(key): int(value) for key, value in (manifest.get("pairs_by_dataset") or {}).items()}
    return questions, pairs


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets are not allowed")
    question_counts, pair_counts = load_counts(args.candidate_root)
    expected_questions = sum(question_counts[dataset] for dataset in args.datasets)
    expected_pairs = sum(pair_counts[dataset] for dataset in args.datasets)
    taus = {
        "medmcqa": load_tau(args.training_label_root, "medmcqa")[0],
        "medqa": load_tau(args.training_label_root, "medqa")[0],
    }

    progress = PipelineProgress(
        overall_total=expected_questions * 2 + expected_pairs,
        desc="RAG2AnchoredExternalLabels",
    )
    no_rag: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_pairs: dict[str, dict[str, Any]] = {}
    try:
        progress.set_stage("1/3 index anchored no-RAG traces", total=expected_questions)
        for dataset in args.datasets:
            path = args.no_rag_root / "no_rag" / dataset / args.split / "no_rag_generations.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            for row in iter_jsonl(path):
                sample_id = str(row.get("sample_id") or "")
                key = (dataset, sample_id)
                if not sample_id or key in no_rag:
                    raise ValueError(f"Invalid/duplicate no-RAG key: {key}")
                no_rag[key] = row
                progress.update()
        if len(no_rag) != expected_questions:
            raise RuntimeError(f"No-RAG count mismatch: {len(no_rag)} != {expected_questions}")

        progress.set_stage("2/3 index dynamic Top-k union identities", total=expected_questions)
        for dataset in args.datasets:
            path = args.candidate_root / dataset / args.split / args.candidate_file
            if not path.is_file():
                raise FileNotFoundError(path)
            observed_pairs = 0
            for row in iter_jsonl(path):
                sample_id = str(row.get("sample_id") or "")
                sample_key = str(row.get("key") or "")
                if (dataset, sample_id) not in no_rag or not sample_key:
                    raise ValueError(f"Candidate/no-RAG mismatch: {dataset}:{sample_id}")
                for rank, document in enumerate(row.get("candidate_documents") or [], 1):
                    identifier = document_pair_id(sample_id, document, rank)
                    if identifier in candidate_pairs:
                        raise ValueError(f"Duplicate candidate pair_id: {identifier}")
                    candidate_pairs[identifier] = {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "sample_key": sample_key,
                        "row_idx": row.get("row_idx"),
                        "doc_rank": rank,
                        "doc_stable_id": stable_id(document),
                        "source": document.get("source"),
                        "db_id": document.get("db_id"),
                        "local_id": document.get("local_id"),
                        "dynamic_top_k_membership": (
                            (document.get("metadata") or {}).get("oracle_dynamic_top_k_membership")
                        ),
                        "dynamic_rerank_rank_by_top_k": (
                            (document.get("metadata") or {}).get("oracle_dynamic_rerank_rank_by_top_k")
                        ),
                    }
                    observed_pairs += 1
                progress.update()
            if observed_pairs != pair_counts[dataset]:
                raise RuntimeError(f"[{dataset}] candidate pair mismatch: {observed_pairs} != {pair_counts[dataset]}")
        if len(candidate_pairs) != expected_pairs:
            raise RuntimeError(f"Candidate pair count mismatch: {len(candidate_pairs)} != {expected_pairs}")

        args.output_root.mkdir(parents=True, exist_ok=True)
        combined_path = args.output_root / "rag2_oracle_labels.jsonl"
        combined_partial = combined_path.with_suffix(".jsonl.partial")
        dataset_paths: dict[str, Path] = {}
        dataset_handles: dict[str, Any] = {}
        for dataset in args.datasets:
            directory = args.output_root / dataset / args.split
            directory.mkdir(parents=True, exist_ok=True)
            dataset_paths[dataset] = directory / "rag2_oracle_labels.jsonl"
            dataset_handles[dataset] = dataset_paths[dataset].with_suffix(".jsonl.partial").open(
                "w", encoding="utf-8", buffering=16 * 1024 * 1024
            )

        counts: dict[str, Counter[str]] = {dataset: Counter() for dataset in args.datasets}
        transitions: dict[str, Counter[str]] = {dataset: Counter() for dataset in args.datasets}
        failures_by_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in args.datasets}
        seen_pairs: set[str] = set()
        progress.set_stage("3/3 apply frozen train tau to independent test traces", total=expected_pairs)
        with combined_partial.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as combined_handle:
            for dataset in args.datasets:
                route = "medqa" if dataset == "medqa" else "medmcqa"
                tau = taus[route]
                paths = sorted(
                    (args.document_trace_root / "trace_shards" / dataset / args.split).glob("shard_*/pairs.jsonl")
                )
                if not paths:
                    raise FileNotFoundError(
                        args.document_trace_root / "trace_shards" / dataset / args.split / "shard_*/pairs.jsonl"
                    )
                for path in paths:
                    for trace in iter_jsonl(path):
                        pair_id = str(trace.get("pair_id") or "")
                        candidate = candidate_pairs.get(pair_id)
                        if candidate is None or pair_id in seen_pairs:
                            raise ValueError(f"Unknown/duplicate document trace pair_id: {pair_id}")
                        if candidate["dataset"] != dataset:
                            raise ValueError(f"Dataset mismatch for {pair_id}")
                        seen_pairs.add(pair_id)
                        baseline = no_rag[(dataset, candidate["sample_id"])]
                        failures = list(no_rag_failures(baseline))
                        failures.extend(document_failures(trace, baseline))
                        failures = sorted(set(failures))
                        no_ppl = rationale_ppl(baseline)
                        with_ppl = rationale_ppl(trace)
                        delta = (
                            float(no_ppl - with_ppl)
                            if no_ppl is not None and with_ppl is not None
                            else None
                        )
                        no_correct = answer_correct(baseline.get("answer"), baseline.get("gold_answer"))
                        with_correct = answer_correct(trace.get("answer"), trace.get("gold_answer"))
                        transition = f"{'C' if no_correct else 'W'}->{'C' if with_correct else 'W'}"
                        if failures or delta is None or not math.isfinite(delta):
                            label, retained = LABEL_EXCLUDED, False
                            if delta is None:
                                failures = sorted(set([*failures, "missing_delta_ppl"]))
                        else:
                            label, retained = assign_label(no_correct, with_correct, delta, tau)
                        output = {
                            "schema_version": 1,
                            "policy": "rag2_anchored_fixed_train_tau_external_dynamic_topk_v1",
                            **candidate,
                            "pair_id": pair_id,
                            "pseudo_label": label,
                            "quality_pass": not failures,
                            "quality_failures": failures,
                            "retained_for_binary_training": retained,
                            "no_doc_prediction": baseline.get("answer"),
                            "with_doc_prediction": trace.get("answer"),
                            "gold_answer": baseline.get("gold_answer"),
                            "no_doc_correct": no_correct,
                            "with_doc_correct": with_correct,
                            "answer_transition": transition,
                            "no_doc_rationale_ppl": no_ppl,
                            "with_doc_rationale_ppl": with_ppl,
                            "delta_ppl": delta,
                            "tau": tau,
                            "tau_route": route,
                            "trace_version": trace.get("trace_version"),
                            "prompt_version": trace.get("prompt_version"),
                            "ppl_scope_version": trace.get("ppl_scope_version"),
                            "generation_policy_version": trace.get("generation_policy_version"),
                        }
                        serialized = json.dumps(output, ensure_ascii=False) + "\n"
                        combined_handle.write(serialized)
                        dataset_handles[dataset].write(serialized)
                        counts[dataset][label] += 1
                        transitions[dataset][transition] += 1
                        failures_by_dataset[dataset].update(failures)
                        progress.update()
            combined_handle.flush()
            os.fsync(combined_handle.fileno())
        for dataset, handle in dataset_handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(dataset_paths[dataset].with_suffix(".jsonl.partial"), dataset_paths[dataset])
        os.replace(combined_partial, combined_path)

        if len(seen_pairs) != expected_pairs:
            missing = len(set(candidate_pairs) - seen_pairs)
            raise RuntimeError(f"Trace coverage mismatch: seen={len(seen_pairs)} expected={expected_pairs} missing={missing}")
        manifest = {
            "type": "rag2_anchored_external_dynamic_topk_oracle_labels",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "questions": expected_questions,
            "pairs": expected_pairs,
            "questions_by_dataset": {dataset: question_counts[dataset] for dataset in args.datasets},
            "pairs_by_dataset": {dataset: pair_counts[dataset] for dataset in args.datasets},
            "tau_routes": taus,
            "label_counts": {dataset: dict(counts[dataset]) for dataset in args.datasets},
            "answer_transitions": {dataset: dict(transitions[dataset]) for dataset in args.datasets},
            "quality_failures": {dataset: dict(failures_by_dataset[dataset]) for dataset in args.datasets},
            "candidate_root": str(args.candidate_root.resolve()),
            "document_trace_root": str(args.document_trace_root.resolve()),
            "no_rag_root": str(args.no_rag_root.resolve()),
            "training_label_root": str(args.training_label_root.resolve()),
            "combined_labels_path": str(combined_path.resolve()),
        }
        atomic_json(args.output_root / "manifest.json", manifest)
    finally:
        progress.close()

    logging.info("RAG2 external Oracle labels complete: %s", args.output_root)


if __name__ == "__main__":
    main()

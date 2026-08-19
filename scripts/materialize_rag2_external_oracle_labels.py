#!/usr/bin/env python3
from __future__ import annotations

"""Apply fixed train-derived RAG2 thresholds to frozen external-test document traces."""

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_rag2_filter_training_splits import (
    LABEL_EXCLUDED, assign_label, no_doc_correct, rationale_only_delta_ppl,
    row_ppl_protocol, trace_quality_failures, with_doc_correct,
)
from medrag.io_utils import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--candidate-file", default="candidates_top32_all.jsonl")
    p.add_argument("--trace-file", default="pseudo_label_traces.jsonl")
    p.add_argument("--medmcqa-tau", type=float, required=True)
    p.add_argument("--medqa-tau", type=float, required=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL: {path}:{number}") from error


def stable_id(doc: dict[str, Any]) -> str:
    return str(doc.get("stable_id") or doc.get("corpus_id") or doc.get("chunk_id") or doc.get("db_id") or f"{doc.get('source')}:{doc.get('local_id')}")


def pair_id(sample_id: str, rank: int, doc: dict[str, Any]) -> str:
    return f"{sample_id}::{rank}::{stable_id(doc)}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, Any]] = []
    report: dict[str, Any] = {"datasets": {}}
    for dataset in args.datasets:
        candidate_path = args.candidate_root / dataset / args.split / args.candidate_file
        trace_path = args.trace_root / dataset / args.split / args.trace_file
        trace_rows: dict[str, dict[str, Any]] = {}
        duplicate_traces = 0
        for trace in rows(trace_path):
            identifier = str(trace.get("pair_id") or "")
            if identifier in trace_rows:
                duplicate_traces += 1
            trace_rows[identifier] = trace  # latest resumable generation wins
        tau = args.medqa_tau if dataset == "medqa" else args.medmcqa_tau
        dataset_rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        expected_pairs = 0
        used_traces: set[str] = set()
        for candidate in rows(candidate_path):
            sample_id = str(candidate["sample_id"])
            for rank, doc in enumerate(candidate.get("candidate_documents") or [], 1):
                expected_pairs += 1
                identifier = pair_id(sample_id, rank, doc)
                trace = trace_rows.get(identifier)
                quality_failures: list[str]
                delta = None
                transition = None
                if trace is None:
                    label = LABEL_EXCLUDED
                    retained = False
                    quality_failures = ["missing_trace_or_invalid_no_rag_baseline"]
                    no_correct = with_correct = None
                    ppl_protocol = None
                else:
                    used_traces.add(identifier)
                    quality_failures = trace_quality_failures(trace)
                    delta = rationale_only_delta_ppl(trace)
                    no_correct = no_doc_correct(trace)
                    with_correct = with_doc_correct(trace)
                    transition = ("C" if no_correct else "W") + "->" + ("C" if with_correct else "W")
                    ppl_protocol = row_ppl_protocol(trace)
                    if quality_failures or delta is None or not math.isfinite(delta):
                        label, retained = LABEL_EXCLUDED, False
                        if delta is None:
                            quality_failures.append("missing_delta_ppl")
                    else:
                        label, retained = assign_label(no_correct, with_correct, delta, tau)
                output = {
                    "schema_version": 1,
                    "policy": "rag2_fixed_train_tau_external_test",
                    "dataset": dataset,
                    "sample_key": candidate.get("key"),
                    "sample_id": sample_id,
                    "row_idx": candidate.get("row_idx"),
                    "pair_id": identifier,
                    "doc_rank": rank,
                    "doc_stable_id": stable_id(doc),
                    "db_id": doc.get("db_id"),
                    "local_id": doc.get("local_id"),
                    "source": doc.get("source"),
                    "pseudo_label": label,
                    "quality_pass": not quality_failures,
                    "quality_failures": sorted(set(quality_failures)),
                    "retained_for_binary_training": retained,
                    "no_doc_correct": no_correct,
                    "with_doc_correct": with_correct,
                    "answer_transition": transition,
                    "delta_ppl": delta,
                    "tau": tau,
                    "ppl_protocol": ppl_protocol,
                }
                dataset_rows.append(output)
                counts[label] += 1
        unused = set(trace_rows) - used_traces
        if len(dataset_rows) != expected_pairs:
            raise RuntimeError(f"[{dataset}] output mismatch")
        destination = args.output_dir / dataset / args.split
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / "rag2_oracle_labels.jsonl", dataset_rows)
        combined.extend(dataset_rows)
        report["datasets"][dataset] = {
            "candidate_path": str(candidate_path), "trace_path": str(trace_path), "tau": tau,
            "pairs": expected_pairs, "trace_rows": len(trace_rows), "unused_trace_rows": len(unused),
            "duplicate_trace_rows": duplicate_traces, "labels": dict(counts),
        }
        logging.info("[%s] pairs=%s labels=%s missing/excluded=%s", dataset, expected_pairs, dict(counts), counts[LABEL_EXCLUDED])
    write_jsonl(args.output_dir / "rag2_oracle_labels.jsonl", combined)
    report["pairs"] = len(combined)
    write_json(args.output_dir / "manifest.json", report)


if __name__ == "__main__":
    main()

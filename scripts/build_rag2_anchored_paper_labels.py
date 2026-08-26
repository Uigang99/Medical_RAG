from __future__ import annotations

"""Materialize paper-style RAG2 pseudo-labels from anchored free generations.

This script intentionally uses no hidden-state feature and no teacher forcing.
For each question/document pair it compares independently generated no-RAG and
single-document answers, and the PPL of each independently generated rationale.
The RAG2 top-25% Delta-PPL threshold is estimated on train questions only and is
then frozen for validation and test.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import build_official_filter_input, format_options
from medrag.progress import PipelineProgress


LABEL_HELPFUL = "Helpful"
LABEL_NOT_HELPFUL = "Not Helpful"
LABEL_DISCARD = "Discard"
LABEL_EXCLUDED = "Excluded"
SPLITS = ("train", "val", "test")
EXPECTED_TRACE_VERSION = "rag2_paper_compatible_three_anchor_v1"
EXPECTED_PROMPT_VERSION = "rag2_paper_compatible_three_anchor_prompt_v1"
EXPECTED_PPL_SCOPE = "generated_rationale_v1"
EXPECTED_GENERATION_POLICY = "rag2_three_anchor_rationale_then_constrained_choice_v1"


try:
    import msgspec

    _DECODER = msgspec.json.Decoder()

    def decode_json(line: bytes) -> dict[str, Any]:
        return _DECODER.decode(line)

except ImportError:

    def decode_json(line: bytes) -> dict[str, Any]:
        return json.loads(line)


def parse_args() -> argparse.Namespace:
    base = (
        PROJECT_ROOT
        / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    )
    parser = argparse.ArgumentParser(
        description="Build RAG2 paper-style labels from anchored no-RAG and one-document traces."
    )
    parser.add_argument(
        "--no-rag-root",
        type=Path,
        default=base / "train_no_rag_anchored_features_v1",
    )
    parser.add_argument(
        "--document-trace-root",
        type=Path,
        default=base / "document_traces_source_balanced32_rerank8_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base / "filter_training_inputs_rag2_paper_reproduction_v1",
    )
    parser.add_argument(
        "--training-label-mode",
        choices=("binary", "three_class"),
        default="binary",
        help=(
            "Labels materialized into train/val/test. binary reproduces the paper and keeps Discard "
            "only in audit files; three_class additionally writes Discard as a trainable target."
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=("medmcqa", "medqa"), default=["medmcqa", "medqa"])
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.75,
        help="Train-only quantile; q=.75 keeps the largest 25%% Delta-PPL values.",
    )
    parser.add_argument(
        "--max-doc-rank",
        type=int,
        default=8,
        help="Use reranked documents up through this rank; zero keeps every generated pair.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=0,
        help="Optional evidence character cap. Zero preserves the complete retrieved chunk.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if min(ratios) < 0 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("--train-ratio, --val-ratio, and --test-ratio must be non-negative and sum to 1.")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be strictly between zero and one.")
    if args.max_doc_rank < 0 or args.max_doc_chars < 0:
        raise ValueError("Rank and character caps must be non-negative.")
    for root in (args.no_rag_root, args.document_trace_root):
        if not root.is_dir():
            raise FileNotFoundError(root)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output_root}. Pass --overwrite after checking it.")


def trace_paths(root: Path, datasets: Iterable[str], split: str, filename: str) -> list[Path]:
    paths: list[Path] = []
    for dataset in datasets:
        dataset_paths = sorted((root / "trace_shards" / dataset / split).glob(f"shard_*/{filename}"))
        if not dataset_paths:
            raise FileNotFoundError(f"No {filename} shards for {dataset}: {root}")
        paths.extend(dataset_paths)
    return paths


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield decode_json(line)
            except Exception as error:
                raise ValueError(f"Malformed JSONL row: {path}:{line_number}") from error


def manifest_count(root: Path, field: str, datasets: Iterable[str]) -> dict[str, int]:
    manifest_path = root / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = manifest.get(field)
    if not isinstance(values, dict):
        raise ValueError(f"Missing {field} mapping in {manifest_path}")
    result = {dataset: int(values.get(dataset, 0)) for dataset in datasets}
    if any(value <= 0 for value in result.values()):
        raise ValueError(f"Invalid {field} counts in {manifest_path}: {result}")
    return result


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def canonical_choice(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"A", "B", "C", "D"} else None


def rationale_ppl(row: dict[str, Any]) -> float | None:
    stats = row.get("rationale_stats") if isinstance(row.get("rationale_stats"), dict) else {}
    if int(stats.get("token_count") or 0) <= 0:
        return None
    return finite_positive(stats.get("ppl"))


def contract_failures(row: dict[str, Any]) -> list[str]:
    expected = {
        "trace_version": EXPECTED_TRACE_VERSION,
        "prompt_version": EXPECTED_PROMPT_VERSION,
        "ppl_scope_version": EXPECTED_PPL_SCOPE,
        "generation_policy_version": EXPECTED_GENERATION_POLICY,
    }
    return [f"{key}_mismatch" for key, value in expected.items() if row.get(key) != value]


def no_rag_failures(row: dict[str, Any]) -> list[str]:
    failures = contract_failures(row)
    if not bool(row.get("valid")):
        failures.append("no_rag_invalid")
    if row.get("truncated_by_max_tokens"):
        failures.append("no_rag_truncated")
    if canonical_choice(row.get("answer")) is None:
        failures.append("no_rag_invalid_answer")
    if canonical_choice(row.get("gold_answer")) is None:
        failures.append("invalid_gold_answer")
    if rationale_ppl(row) is None:
        failures.append("no_rag_invalid_rationale_ppl")
    return sorted(set(failures))


def document_failures(row: dict[str, Any], no_rag: dict[str, Any] | None) -> list[str]:
    failures = contract_failures(row)
    if no_rag is None:
        failures.append("missing_no_rag_row")
    if not bool(row.get("valid_for_layer_analysis")):
        failures.append("with_document_invalid")
    if canonical_choice(row.get("answer")) is None:
        failures.append("with_document_invalid_answer")
    if canonical_choice(row.get("gold_answer")) is None:
        failures.append("invalid_gold_answer")
    if rationale_ppl(row) is None:
        failures.append("with_document_invalid_rationale_ppl")
    if not clean_text(row.get("document_text_used")):
        failures.append("empty_document_text")
    if no_rag is not None:
        if canonical_choice(row.get("gold_answer")) != no_rag["gold_answer"]:
            failures.append("gold_answer_mismatch")
        for key in ("trace_version", "prompt_version", "ppl_scope_version", "generation_policy_version"):
            if row.get(key) != no_rag[key]:
                failures.append(f"no_doc_with_doc_{key}_mismatch")
    return sorted(set(failures))


def answer_correct(answer: Any, gold_answer: Any) -> bool:
    return canonical_choice(answer) == canonical_choice(gold_answer)


def assign_label(no_correct: bool, doc_correct: bool, delta_ppl: float, tau: float) -> tuple[str, bool]:
    """Exact RAG2 decision table from Figure 2 / Section 3.2."""

    in_top_quartile = delta_ppl >= tau
    if not no_correct and doc_correct:
        return LABEL_HELPFUL, True
    if no_correct and not doc_correct:
        return LABEL_NOT_HELPFUL, True
    if no_correct and doc_correct:
        return (LABEL_HELPFUL, True) if in_top_quartile else (LABEL_DISCARD, False)
    return (LABEL_NOT_HELPFUL, True) if in_top_quartile else (LABEL_DISCARD, False)


def make_assignments(
    sample_to_dataset: dict[str, str], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, dataset in sample_to_dataset.items():
        grouped[dataset].append(sample_id)
    assignments: dict[str, str] = {}
    for dataset, values in sorted(grouped.items()):
        ordered = sorted(values)
        random.Random(f"{seed}:{dataset}").shuffle(ordered)
        train_end = int(round(len(ordered) * train_ratio))
        val_end = min(len(ordered), train_end + int(round(len(ordered) * val_ratio)))
        for sample_id in ordered[:train_end]:
            assignments[sample_id] = "train"
        for sample_id in ordered[train_end:val_end]:
            assignments[sample_id] = "val"
        for sample_id in ordered[val_end:]:
            assignments[sample_id] = "test"
    return assignments


def scan_no_rag(
    paths: list[Path], expected_total: int, progress: PipelineProgress
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Counter[str]]]:
    progress.set_stage("1/3 index no-RAG traces", total=expected_total)
    records: dict[str, dict[str, Any]] = {}
    sample_to_dataset: dict[str, str] = {}
    failure_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            dataset = str(row.get("dataset") or "")
            if not sample_id or not dataset:
                raise ValueError(f"Missing no-RAG sample_id/dataset in {path}")
            if sample_id in records:
                raise ValueError(f"Duplicate no-RAG sample_id: {sample_id}")
            failures = no_rag_failures(row)
            failure_counts[dataset].update(failures)
            records[sample_id] = {
                "dataset": dataset,
                "answer": canonical_choice(row.get("answer")),
                "gold_answer": canonical_choice(row.get("gold_answer")),
                "correct": answer_correct(row.get("answer"), row.get("gold_answer")),
                "ppl": rationale_ppl(row),
                "failures": failures,
                "trace_version": row.get("trace_version"),
                "prompt_version": row.get("prompt_version"),
                "ppl_scope_version": row.get("ppl_scope_version"),
                "generation_policy_version": row.get("generation_policy_version"),
            }
            sample_to_dataset[sample_id] = dataset
            progress.update()
    if len(records) != expected_total:
        raise RuntimeError(f"No-RAG row count mismatch: {len(records)} != {expected_total}")
    return records, sample_to_dataset, failure_counts


def rank_is_selected(row: dict[str, Any], max_doc_rank: int) -> bool:
    if max_doc_rank <= 0:
        return True
    try:
        return int(row.get("doc_rank")) <= max_doc_rank
    except (TypeError, ValueError):
        return False


def compute_thresholds(
    paths: list[Path],
    expected_total: int,
    records: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    quantile: float,
    max_doc_rank: int,
    progress: PipelineProgress,
) -> tuple[dict[str, float], dict[str, Any]]:
    progress.set_stage("2/3 estimate train-only Delta-PPL tau", total=expected_total)
    deltas: dict[str, list[float]] = defaultdict(list)
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    observed = 0
    for path in paths:
        for row in iter_jsonl(path):
            observed += 1
            progress.update()
            if not rank_is_selected(row, max_doc_rank):
                counters[str(row.get("dataset"))]["rank_excluded"] += 1
                continue
            sample_id = str(row.get("sample_id") or "")
            no_rag = records.get(sample_id)
            failures = list(no_rag.get("failures") or []) if no_rag else []
            failures.extend(document_failures(row, no_rag))
            if failures:
                counters[str(row.get("dataset"))].update(set(failures))
                continue
            if assignments.get(sample_id) != "train":
                continue
            assert no_rag is not None and no_rag["ppl"] is not None
            doc_ppl = rationale_ppl(row)
            assert doc_ppl is not None
            deltas[str(row["dataset"])].append(float(no_rag["ppl"] - doc_ppl))
    if observed != expected_total:
        raise RuntimeError(f"Document pair row count mismatch: {observed} != {expected_total}")

    taus: dict[str, float] = {}
    summaries: dict[str, Any] = {}
    for dataset in sorted({str(value["dataset"]) for value in records.values()}):
        values = np.asarray(deltas.get(dataset, []), dtype=np.float64)
        if not values.size:
            raise RuntimeError(f"No valid train Delta-PPL values for {dataset}")
        tau = float(np.quantile(values, quantile))
        taus[dataset] = tau
        summaries[dataset] = {
            "valid_train_pairs": int(values.size),
            "threshold_quantile": quantile,
            "tau": tau,
            "delta_ppl": {
                "min": float(values.min()),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
                "max": float(values.max()),
            },
            "quality_failures_seen": dict(counters[dataset]),
        }
        logging.info("[%s] train-only q%.2f Delta-PPL tau=%.8f from %s pairs", dataset, quantile, tau, values.size)
    return taus, summaries


def evidence_text(row: dict[str, Any], max_doc_chars: int) -> str:
    value = clean_text(row.get("document_text_used"))
    if max_doc_chars > 0 and len(value) > max_doc_chars:
        value = value[: max_doc_chars - 3].rstrip() + "..."
    return value


def compact_record(
    row: dict[str, Any],
    no_rag: dict[str, Any] | None,
    split: str,
    label: str,
    use_for_training: bool,
    no_ppl: float | None,
    doc_ppl: float | None,
    delta_ppl: float | None,
    tau: float | None,
    failures: list[str],
) -> dict[str, Any]:
    document = row.get("document") if isinstance(row.get("document"), dict) else {}
    return {
        "id": row.get("pair_id"),
        "pair_id": row.get("pair_id"),
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "split": split,
        "row_idx": row.get("row_idx"),
        "doc_rank": row.get("doc_rank"),
        "source": document.get("source"),
        "doc_stable_id": document.get("stable_id"),
        "gold_answer": canonical_choice(row.get("gold_answer")),
        "no_doc_prediction": no_rag.get("answer") if no_rag else None,
        "with_doc_prediction": canonical_choice(row.get("answer")),
        "no_doc_correct": bool(no_rag.get("correct")) if no_rag else None,
        "with_doc_correct": answer_correct(row.get("answer"), row.get("gold_answer")),
        "no_doc_rationale_ppl": no_ppl,
        "with_doc_rationale_ppl": doc_ppl,
        "delta_ppl": delta_ppl,
        "rationale_only_delta_ppl": delta_ppl,
        "tau": tau,
        "pseudo_label": label,
        "use_for_training": use_for_training,
        "quality_pass": not failures,
        "quality_failures": failures,
        "trace_version": row.get("trace_version"),
        "prompt_version": row.get("prompt_version"),
        "ppl_scope_version": row.get("ppl_scope_version"),
        "generation_policy_version": row.get("generation_policy_version"),
        "label_protocol_version": "rag2_independent_answer_transition_generated_rationale_ppl_top25_v1",
    }


def atomic_output_handles(output_root: Path, datasets: Iterable[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Path]]]:
    handles: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, Path]] = {}
    for dataset in datasets:
        directory = output_root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        paths[dataset] = {split: directory / f"{split}.jsonl" for split in SPLITS}
        paths[dataset].update(
            {
                "labels_all": directory / "labels_all.jsonl",
                "discarded": directory / "discarded.jsonl",
                "excluded": directory / "excluded.jsonl",
            }
        )
        handles[dataset] = {
            name: path.with_suffix(path.suffix + ".partial").open(
                "w", encoding="utf-8", buffering=16 * 1024 * 1024
            )
            for name, path in paths[dataset].items()
        }
    return handles, paths


def finalize_outputs(handles: dict[str, dict[str, Any]], paths: dict[str, dict[str, Path]]) -> None:
    for dataset, values in handles.items():
        for name, handle in values.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            partial = paths[dataset][name].with_suffix(paths[dataset][name].suffix + ".partial")
            os.replace(partial, paths[dataset][name])


def abort_outputs(handles: dict[str, dict[str, Any]]) -> None:
    for values in handles.values():
        for handle in values.values():
            try:
                handle.close()
            except Exception:
                pass


def materialize(
    args: argparse.Namespace,
    paths_in: list[Path],
    expected_total: int,
    records: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    taus: dict[str, float],
    progress: PipelineProgress,
) -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    progress.set_stage("3/3 materialize labels and filter splits", total=expected_total)
    handles, output_paths = atomic_output_handles(args.output_root, args.datasets)
    stats: dict[str, dict[str, Any]] = {
        dataset: {
            "rows": 0,
            "sample_ids": {split: set() for split in SPLITS},
            "labels": {split: Counter() for split in SPLITS},
            "transitions": {split: Counter() for split in SPLITS},
            "training_sources": {split: Counter() for split in SPLITS},
            "quality_failures": Counter(),
        }
        for dataset in args.datasets
    }
    observed = 0
    try:
        for path in paths_in:
            for row in iter_jsonl(path):
                observed += 1
                progress.update()
                dataset = str(row.get("dataset") or "")
                sample_id = str(row.get("sample_id") or "")
                no_rag = records.get(sample_id)
                split = assignments.get(sample_id, "train")
                no_ppl = no_rag.get("ppl") if no_rag else None
                doc_ppl = rationale_ppl(row)
                failures = list(no_rag.get("failures") or []) if no_rag else []
                failures.extend(document_failures(row, no_rag))
                if not rank_is_selected(row, args.max_doc_rank):
                    failures.append("rank_excluded")
                failures = sorted(set(failures))
                delta = float(no_ppl - doc_ppl) if no_ppl is not None and doc_ppl is not None else None
                if failures or delta is None:
                    label, use_for_training, tau = LABEL_EXCLUDED, False, None
                else:
                    tau = taus[dataset]
                    assert no_rag is not None
                    label, use_for_training = assign_label(
                        bool(no_rag["correct"]),
                        answer_correct(row.get("answer"), row.get("gold_answer")),
                        delta,
                        tau,
                    )
                    if args.training_label_mode == "three_class" and label == LABEL_DISCARD:
                        use_for_training = True
                record = compact_record(
                    row,
                    no_rag,
                    split,
                    label,
                    use_for_training,
                    no_ppl,
                    doc_ppl,
                    delta,
                    tau,
                    failures,
                )
                handles[dataset]["labels_all"].write(json.dumps(record, ensure_ascii=False) + "\n")
                dataset_stats = stats[dataset]
                dataset_stats["rows"] += 1
                dataset_stats["sample_ids"][split].add(sample_id)
                dataset_stats["labels"][split][label] += 1
                transition = (
                    f"{'C' if record['no_doc_correct'] else 'W'}->"
                    f"{'C' if record['with_doc_correct'] else 'W'}"
                )
                dataset_stats["transitions"][split][transition] += 1
                dataset_stats["quality_failures"].update(failures)
                if label == LABEL_EXCLUDED:
                    handles[dataset]["excluded"].write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                if label == LABEL_DISCARD:
                    handles[dataset]["discarded"].write(json.dumps(record, ensure_ascii=False) + "\n")
                    if args.training_label_mode == "binary":
                        continue
                document = row.get("document") if isinstance(row.get("document"), dict) else {}
                input_text = build_official_filter_input(
                    question=clean_text(row.get("question")),
                    options=format_options(row.get("options") if isinstance(row.get("options"), dict) else {}),
                    evidence=evidence_text(row, args.max_doc_chars),
                )
                training_row = {
                    **record,
                    "input": input_text,
                    "target": {
                        LABEL_HELPFUL: "helpful",
                        LABEL_NOT_HELPFUL: "not helpful",
                        LABEL_DISCARD: "discard",
                    }[label],
                    "label": label,
                    "answer": {
                        LABEL_HELPFUL: "[HELPFUL]",
                        LABEL_NOT_HELPFUL: "[NOT_HELPFUL]",
                        LABEL_DISCARD: "[DISCARD]",
                    }[label],
                }
                handles[dataset][split].write(json.dumps(training_row, ensure_ascii=False) + "\n")
                dataset_stats["training_sources"][split][str(document.get("source") or "unknown")] += 1
        if observed != expected_total:
            raise RuntimeError(f"Document pair row count mismatch while labeling: {observed} != {expected_total}")
        finalize_outputs(handles, output_paths)
    except Exception:
        abort_outputs(handles)
        raise

    summaries: dict[str, Any] = {}
    for dataset, values in stats.items():
        split_summary: dict[str, Any] = {}
        split_sets = values["sample_ids"]
        leakage = {
            "train_val": len(split_sets["train"] & split_sets["val"]),
            "train_test": len(split_sets["train"] & split_sets["test"]),
            "val_test": len(split_sets["val"] & split_sets["test"]),
        }
        if any(leakage.values()):
            raise RuntimeError(f"Question leakage in {dataset}: {leakage}")
        for split in SPLITS:
            counts = values["labels"][split]
            split_summary[split] = {
                "questions": len(split_sets[split]),
                "pairs": int(sum(counts.values())),
                "label_counts": dict(counts),
                "training_pairs": int(
                    counts[LABEL_HELPFUL]
                    + counts[LABEL_NOT_HELPFUL]
                    + (counts[LABEL_DISCARD] if args.training_label_mode == "three_class" else 0)
                ),
                "answer_transitions": dict(values["transitions"][split]),
                "training_sources": dict(values["training_sources"][split]),
            }
        summaries[dataset] = {
            "rows": values["rows"],
            "splits": split_summary,
            "quality_failures": dict(values["quality_failures"]),
            "leakage_check": leakage,
        }
    return summaries, output_paths


def write_sample_ids(output_root: Path, assignments: dict[str, str], sample_to_dataset: dict[str, str]) -> None:
    values: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample_id, dataset in sample_to_dataset.items():
        values[dataset][assignments[sample_id]].append(sample_id)
    for dataset, split_values in values.items():
        directory = output_root / dataset / "sample_ids"
        directory.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            path = directory / f"{split}.txt"
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text("\n".join(sorted(split_values[split])) + "\n", encoding="utf-8")
            os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    no_counts = manifest_count(args.no_rag_root, "datasets", args.datasets)
    document_question_counts = manifest_count(args.document_trace_root, "datasets", args.datasets)
    pair_counts = manifest_count(args.document_trace_root, "pairs_by_dataset", args.datasets)
    no_total = sum(no_counts.values())
    pair_total = sum(pair_counts.values())
    no_paths = trace_paths(args.no_rag_root, args.datasets, args.source_split, "questions.jsonl")
    document_paths = trace_paths(args.document_trace_root, args.datasets, args.source_split, "pairs.jsonl")
    logging.info(
        "RAG2 labeling plan: datasets=%s questions=%s pairs=%s stages=3",
        args.datasets,
        no_counts,
        pair_counts,
    )
    if args.dry_run:
        return
    if args.overwrite:
        for dataset in args.datasets:
            directory = args.output_root / dataset
            if directory.is_dir():
                for path in directory.glob("*.jsonl*"):
                    path.unlink()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = PipelineProgress(
        overall_total=no_total + pair_total * 2,
        desc="RAG2PaperLabels",
        enabled=args.show_progress,
    )
    try:
        records, sample_to_dataset, no_failure_counts = scan_no_rag(no_paths, no_total, progress)
        assignments = make_assignments(sample_to_dataset, args.train_ratio, args.val_ratio, args.seed)
        taus, threshold_summaries = compute_thresholds(
            document_paths,
            pair_total,
            records,
            assignments,
            args.threshold_quantile,
            args.max_doc_rank,
            progress,
        )
        summaries, output_paths = materialize(
            args,
            document_paths,
            pair_total,
            records,
            assignments,
            taus,
            progress,
        )
        write_sample_ids(args.output_root, assignments, sample_to_dataset)
    finally:
        progress.close()

    for dataset in args.datasets:
        manifest = {
            "type": "rag2_paper_reproduction_filter_labels",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": dataset,
            "training_label_mode": args.training_label_mode,
            "training_target_labels": (
                ["helpful", "not helpful", "discard"]
                if args.training_label_mode == "three_class"
                else ["helpful", "not helpful"]
            ),
            "source_artifacts": {
                "no_rag_root": str(args.no_rag_root.resolve()),
                "document_trace_root": str(args.document_trace_root.resolve()),
                "no_rag_questions": no_counts[dataset],
                "document_trace_questions": document_question_counts[dataset],
                "document_pairs": pair_counts[dataset],
            },
            "split_protocol": {
                "unit": "sample_id/question",
                "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
                "seed": args.seed,
                "threshold_fit_split": "train only",
                "threshold_frozen_for": ["val", "test"],
            },
            "label_protocol": {
                "reference": "RAG2 Section 3.2 and Figure 2",
                "hidden_state_features_used": False,
                "teacher_forcing_used": False,
                "answer_transition": "independent no-RAG answer versus independent one-document answer",
                "ppl_scope": "PPL of each independently generated rationale span; final option is excluded",
                "delta_ppl": "PPL(no document rationale) - PPL(one document rationale)",
                "threshold": "per-dataset train-only 75th percentile (top 25% Delta-PPL)",
                "threshold_quantile": args.threshold_quantile,
                "tau": taus[dataset],
                "branches": {
                    "wrong_to_correct": "Helpful",
                    "correct_to_wrong": "Not Helpful",
                    "correct_to_correct": "Helpful if Delta-PPL >= tau, else Discard",
                    "wrong_to_wrong": "Not Helpful if Delta-PPL >= tau, else Discard",
                },
                "training_labels": (
                    ["Helpful", "Not Helpful", "Discard"]
                    if args.training_label_mode == "three_class"
                    else ["Helpful", "Not Helpful"]
                ),
                "training_label_mode": args.training_label_mode,
                "discard_is_not_trained": args.training_label_mode == "binary",
                "discard_semantics": (
                    "abstention/no-decision target; excluded from final answer-LLM context"
                    if args.training_label_mode == "three_class"
                    else "audit-only; excluded from classifier training"
                ),
            },
            "trace_contract": {
                "trace_version": EXPECTED_TRACE_VERSION,
                "prompt_version": EXPECTED_PROMPT_VERSION,
                "ppl_scope_version": EXPECTED_PPL_SCOPE,
                "generation_policy_version": EXPECTED_GENERATION_POLICY,
            },
            "retrieval_scope": {
                "max_doc_rank": args.max_doc_rank,
                "available_docs_per_question": pair_counts[dataset] // document_question_counts[dataset],
                "document_character_cap": args.max_doc_chars,
            },
            "filter_input": {
                "format": "released RAG2 evidence-then-question template",
                "features": "question + options + complete document text only",
                "targets": (
                    {
                        "Helpful": "[HELPFUL]",
                        "Not Helpful": "[NOT_HELPFUL]",
                        "Discard": "[DISCARD]",
                    }
                    if args.training_label_mode == "three_class"
                    else {"Helpful": "[HELPFUL]", "Not Helpful": "[NOT_HELPFUL]"}
                ),
            },
            "threshold_summary": threshold_summaries[dataset],
            "no_rag_quality_failures": dict(no_failure_counts[dataset]),
            "summary": summaries[dataset],
            "files": {name: str(path.resolve()) for name, path in output_paths[dataset].items()},
            "known_local_boundary": (
                "The paper does not publish its exact pseudo-label split IDs or generated traces. "
                "This run reproduces the published rule on the locally regenerated anchored traces."
            ),
            "command": " ".join(sys.argv),
        }
        path = args.output_root / dataset / "manifest.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        logging.info("[%s] completed: %s", dataset, summaries[dataset]["splits"])


if __name__ == "__main__":
    main()

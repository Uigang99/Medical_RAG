from __future__ import annotations

"""Create leakage-free RAG2 filter labels and question-level train/val/test splits.

The target paper trains a binary Flan-T5 filter with pseudo-labels from the
change in rationale perplexity. This builder keeps the experimental boundary
clean: it assigns each question to one split first, estimates the top-25%
Delta-PPL threshold only from the train questions, and applies that fixed
threshold to validation and test questions.
"""

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import build_official_filter_input, format_options


LABEL_HELPFUL = "Helpful"
LABEL_NOT_HELPFUL = "Not Helpful"
LABEL_DISCARD = "Discard"
LABEL_EXCLUDED = "Excluded"
SPLITS = ("train", "val", "test")


try:
    import msgspec

    _DECODER = msgspec.json.Decoder()

    def decode_json(line: bytes) -> dict[str, Any]:
        return _DECODER.decode(line)

except ImportError:

    def decode_json(line: bytes) -> dict[str, Any]:
        return json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG2 pseudo-label train/val/test splits without question leakage."
    )
    parser.add_argument(
        "--trace-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Completed single-document trace JSONL files, one or more datasets.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.75,
        help="Train-only quantile for the RAG2 rationale-only Delta-PPL threshold.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=0,
        help="Optional character cap before token-length filtering. Zero preserves the full corpus chunk.",
    )
    parser.add_argument(
        "--write-all-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write compact labels_all.jsonl, including discarded/excluded rows, for auditability.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory for a dataset in --trace-paths.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if min(args.train_ratio, args.val_ratio, args.test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("--train-ratio, --val-ratio, and --test-ratio must sum to 1.")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be strictly between zero and one.")
    if args.max_doc_chars < 0:
        raise ValueError("--max-doc-chars must be non-negative.")
    for path in args.trace_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing trace file: {path}")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield decode_json(line)
            except Exception as exc:
                raise ValueError(f"Malformed trace JSONL: {path}:{line_number}") from exc


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def is_fixed_target_trace(row: dict[str, Any]) -> bool:
    return isinstance(row.get("teacher_forced_ppl"), dict) and isinstance(row.get("free_with_document"), dict)


def canonical_answer_options(row: dict[str, Any]) -> set[str]:
    """Return the MCQ option letters stored on a trace row.

    ``no_doc_correct`` is a cached convenience field, not a source of truth:
    older paper-exact traces were written from no-RAG rows whose gold labels
    lived under ``gold_answers`` rather than ``answers``.  Recompute whenever
    the trace itself contains the canonical ``answers`` list so a stale cache
    cannot alter a pseudo-label.
    """

    values = row.get("answers")
    if not isinstance(values, list):
        values = [row.get("answer")]
    return {
        str(value).strip().upper()
        for value in values
        if str(value or "").strip()
    }


def prediction_is_correct(row: dict[str, Any], prediction: Any, stored_value: Any) -> bool:
    answers = canonical_answer_options(row)
    normalized_prediction = str(prediction or "").strip().upper()
    if answers and normalized_prediction:
        return normalized_prediction in answers
    return bool(stored_value)


def no_doc_correct(row: dict[str, Any]) -> bool:
    return prediction_is_correct(row, row.get("no_doc_prediction"), row.get("no_doc_correct"))


def with_doc_record(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("free_with_document") if is_fixed_target_trace(row) else row


def with_doc_prediction(row: dict[str, Any]) -> Any:
    return with_doc_record(row).get("prediction") if is_fixed_target_trace(row) else row.get("with_doc_prediction")


def with_doc_correct(row: dict[str, Any]) -> bool:
    record = with_doc_record(row)
    prediction = record.get("prediction") if is_fixed_target_trace(row) else row.get("with_doc_prediction")
    stored_value = record.get("correct") if is_fixed_target_trace(row) else row.get("with_doc_correct")
    return prediction_is_correct(row, prediction, stored_value)


def valid_with_doc_outcome(row: dict[str, Any]) -> bool:
    """Return whether the free generation is a labelable answer transition.

    A refusal/out-of-options response is deliberately not coerced into an MCQ
    letter. It is a stable harmful outcome when the no-document response was
    correct, so it can yield a Not Helpful label; malformed or technical
    generations remain excluded.
    """
    if not is_fixed_target_trace(row):
        return valid_prediction(row, "with_doc_prediction")
    free = with_doc_record(row)
    return free.get("answer_status") in {"valid_option", "out_of_options_or_refusal"}


def rationale_only_ppl(row: dict[str, Any], field: str) -> float | None:
    stats = row.get(field)
    if not isinstance(stats, dict):
        return None
    rationale_only = stats.get("rationale_only")
    if not isinstance(rationale_only, dict):
        return None
    return finite_positive(rationale_only.get("ppl"))


def valid_prediction(row: dict[str, Any], field: str) -> bool:
    prediction = str(row.get(field) or "").upper()
    options = row.get("options")
    return isinstance(options, dict) and prediction in {str(option).upper() for option in options}


def trace_quality_failures(row: dict[str, Any]) -> list[str]:
    """Reject only rows that cannot support a reproducible pseudo-label."""

    if is_fixed_target_trace(row):
        failures: list[str] = []
        tf = row.get("teacher_forced_ppl") or {}
        free = with_doc_record(row)
        failures.extend(f"teacher_forced_{issue}" for issue in (tf.get("quality_issues") or []))
        if not tf.get("quality_pass"):
            failures.append("teacher_forced_quality_failure")
        if finite_positive(((tf.get("no_doc_stats") or {}).get("rationale_only") or {}).get("ppl")) is None:
            failures.append("missing_fixed_no_doc_rationale_only_ppl")
        if finite_positive(((tf.get("with_doc_stats") or {}).get("rationale_only") or {}).get("ppl")) is None:
            failures.append("missing_fixed_with_doc_rationale_only_ppl")
        if finite_positive(tf.get("delta_ppl_rationale_only")) is None and tf.get("delta_ppl_rationale_only") is None:
            failures.append("missing_fixed_rationale_only_delta_ppl")
        if not valid_prediction(row, "no_doc_prediction"):
            failures.append("invalid_no_doc_prediction")
        if not valid_with_doc_outcome(row):
            failures.append(f"unusable_free_answer_status_{free.get('answer_status') or 'missing'}")
        return sorted(set(failures))

    failures: list[str] = []
    failures.extend(f"with_doc_{issue}" for issue in (row.get("with_doc_quality_issues") or []))
    failures.extend(f"no_doc_{issue}" for issue in (row.get("no_doc_parse_errors") or []))
    if row.get("with_doc_parse_errors"):
        failures.append("with_doc_parse_errors")
    if row.get("with_doc_finish_reason") == "length":
        failures.append("with_doc_max_tokens_exhausted")
    if not clean_text(row.get("no_doc_rationale_only")):
        failures.append("missing_no_doc_rationale")
    if not clean_text(row.get("with_doc_rationale_only")):
        failures.append("missing_with_doc_rationale")
    if not valid_prediction(row, "no_doc_prediction"):
        failures.append("invalid_no_doc_prediction")
    if not valid_prediction(row, "with_doc_prediction"):
        failures.append("invalid_with_doc_prediction")
    if rationale_only_ppl(row, "no_doc_generation_stats") is None:
        failures.append("missing_no_doc_rationale_only_ppl")
    if rationale_only_ppl(row, "with_doc_generation_stats") is None:
        failures.append("missing_with_doc_rationale_only_ppl")
    return sorted(set(failures))


def rationale_only_delta_ppl(row: dict[str, Any]) -> float | None:
    if is_fixed_target_trace(row):
        value = row.get("teacher_forced_ppl", {}).get("delta_ppl_rationale_only")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None
    no_doc = rationale_only_ppl(row, "no_doc_generation_stats")
    with_doc = rationale_only_ppl(row, "with_doc_generation_stats")
    if no_doc is None or with_doc is None:
        return None
    return no_doc - with_doc


def row_ppl_protocol(row: dict[str, Any]) -> str:
    """Return the explicitly recorded PPL unit for one trace row.

    ``paper_exact`` intentionally has no required rationale/answer delimiter.
    Its new traces consequently map the complete visible free response to the
    comparable PPL field.  Preserve that fact in the labeled dataset instead
    of describing it incorrectly as a parser-defined rationale-only span.
    """
    if is_fixed_target_trace(row):
        return "fixed_no_rag_target_teacher_forced_rationale_only_ppl"
    return str(row.get("ppl_comparison_version") or "independent_generation_ppl_legacy")


def assign_label(no_doc_correct: bool, with_doc_correct: bool, delta: float, tau: float) -> tuple[str, bool]:
    """Implement the RAG2 helpful/not-helpful/discard pseudo-label branches."""

    improved_confidence = delta >= tau
    if no_doc_correct and with_doc_correct:
        return (LABEL_HELPFUL, True) if improved_confidence else (LABEL_DISCARD, False)
    if no_doc_correct and not with_doc_correct:
        return LABEL_NOT_HELPFUL, True
    if not no_doc_correct and with_doc_correct:
        return LABEL_HELPFUL, True
    return (LABEL_NOT_HELPFUL, True) if improved_confidence else (LABEL_DISCARD, False)


def document_text(row: dict[str, Any], max_doc_chars: int) -> str:
    document = row.get("doc") if isinstance(row.get("doc"), dict) else {}
    text = clean_text(document.get("text")) or clean_text(document.get("title"))
    if max_doc_chars > 0 and len(text) > max_doc_chars:
        return text[: max_doc_chars - 3].rstrip() + "..."
    return text


def filter_input(row: dict[str, Any], max_doc_chars: int) -> str:
    return build_official_filter_input(
        question=clean_text(row.get("question")),
        options=format_options(row.get("options") if isinstance(row.get("options"), dict) else {}),
        evidence=document_text(row, max_doc_chars),
    )


def make_assignments(
    sample_to_dataset: dict[str, str],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, dataset in sample_to_dataset.items():
        grouped[dataset].append(sample_id)

    assignments: dict[str, str] = {}
    for dataset, sample_ids in sorted(grouped.items()):
        ordered = sorted(sample_ids)
        random.Random(f"{seed}:{dataset}").shuffle(ordered)
        n_train = int(round(len(ordered) * train_ratio))
        n_val = int(round(len(ordered) * val_ratio))
        if n_train + n_val > len(ordered):
            n_val = len(ordered) - n_train
        for sample_id in ordered[:n_train]:
            assignments[sample_id] = "train"
        for sample_id in ordered[n_train : n_train + n_val]:
            assignments[sample_id] = "val"
        for sample_id in ordered[n_train + n_val :]:
            assignments[sample_id] = "test"
    return assignments


def scan_questions(trace_paths: list[Path]) -> tuple[dict[str, str], Counter[str]]:
    sample_to_dataset: dict[str, str] = {}
    trace_rows: Counter[str] = Counter()
    for path in trace_paths:
        size = path.stat().st_size
        with tqdm(total=size, desc=f"Questions:{path.parent.parent.name}", unit="B", unit_scale=True) as progress:
            with path.open("rb", buffering=64 * 1024 * 1024) as handle:
                for line_number, line in enumerate(handle, start=1):
                    progress.update(len(line))
                    if not line.strip():
                        continue
                    row = decode_json(line)
                    # Support-only repairs are append-only copies of an
                    # existing pair. They change attribution only, never the
                    # fixed-target PPL or independent answer transition used
                    # for filter labels, and must not duplicate training rows.
                    if row.get("support_repair"):
                        continue
                    sample_id = str(row.get("sample_id") or "")
                    dataset = str(row.get("dataset") or "")
                    if not sample_id or not dataset:
                        raise ValueError(f"Missing sample_id or dataset at {path}:{line_number}")
                    previous = sample_to_dataset.setdefault(sample_id, dataset)
                    if previous != dataset:
                        raise ValueError(f"sample_id occurs in multiple datasets: {sample_id}")
                    trace_rows[dataset] += 1
    return sample_to_dataset, trace_rows


def train_thresholds(
    trace_paths: list[Path], assignments: dict[str, str], quantile: float
) -> tuple[dict[str, float], dict[str, Any]]:
    values: dict[str, list[float]] = defaultdict(list)
    quality_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in trace_paths:
        size = path.stat().st_size
        with tqdm(total=size, desc=f"Train tau:{path.parent.parent.name}", unit="B", unit_scale=True) as progress:
            with path.open("rb", buffering=64 * 1024 * 1024) as handle:
                for line in handle:
                    progress.update(len(line))
                    if not line.strip():
                        continue
                    row = decode_json(line)
                    if row.get("support_repair"):
                        continue
                    dataset = str(row["dataset"])
                    failures = trace_quality_failures(row)
                    if failures:
                        quality_counts[dataset].update(failures)
                        continue
                    delta = rationale_only_delta_ppl(row)
                    if delta is None:
                        quality_counts[dataset]["invalid_rationale_only_delta_ppl"] += 1
                        continue
                    if assignments[str(row["sample_id"])] == "train":
                        values[dataset].append(delta)

    taus: dict[str, float] = {}
    summaries: dict[str, Any] = {}
    for dataset, deltas in sorted(values.items()):
        if not deltas:
            raise ValueError(f"No valid train delta-PPL values for dataset={dataset}")
        array = np.asarray(deltas, dtype=np.float64)
        tau = float(np.quantile(array, quantile))
        taus[dataset] = tau
        summaries[dataset] = {
            "train_valid_delta_pairs": int(array.size),
            "tau": tau,
            "delta_ppl": {
                "min": float(np.min(array)),
                "p50": float(np.quantile(array, 0.50)),
                "p75": float(np.quantile(array, 0.75)),
                "p90": float(np.quantile(array, 0.90)),
                "max": float(np.max(array)),
            },
            "quality_exclusion_reasons_seen": dict(quality_counts[dataset]),
        }
        logging.info(
            "[%s] train-only rationale-only Delta-PPL tau(q=%.2f)=%.6f from %s pairs",
            dataset,
            quantile,
            tau,
            array.size,
        )
    return taus, summaries


def open_outputs(
    output_root: Path,
    datasets: set[str],
    *,
    write_all_labels: bool,
    overwrite: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Path]]]:
    handles: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, Path]] = {}
    for dataset in sorted(datasets):
        dataset_dir = output_root / dataset
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {dataset_dir}. Use --overwrite after checking it."
                )
            for path in dataset_dir.glob("*.jsonl"):
                path.unlink()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        paths[dataset] = {split: dataset_dir / f"{split}.jsonl" for split in SPLITS}
        paths[dataset]["discarded"] = dataset_dir / "discarded.jsonl"
        paths[dataset]["excluded"] = dataset_dir / "excluded.jsonl"
        if write_all_labels:
            paths[dataset]["all_labels"] = dataset_dir / "labels_all.jsonl"
        handles[dataset] = {
            key: path.open("w", encoding="utf-8", buffering=16 * 1024 * 1024)
            for key, path in paths[dataset].items()
        }
    return handles, paths


def close_outputs(handles: dict[str, dict[str, Any]]) -> None:
    for dataset_handles in handles.values():
        for handle in dataset_handles.values():
            handle.close()


def compact_label_record(
    row: dict[str, Any],
    *,
    split: str,
    label: str,
    use_for_training: bool,
    delta: float | None,
    tau: float | None,
    failures: list[str],
) -> dict[str, Any]:
    document = row.get("doc") if isinstance(row.get("doc"), dict) else {}
    free = with_doc_record(row)
    return {
        "id": row.get("pair_id"),
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "row_idx": row.get("row_idx"),
        "split": split,
        "source": document.get("source"),
        "doc_stable_id": document.get("stable_id"),
        "doc_rank": row.get("doc_rank"),
        "label_ppl_protocol": (
            "fixed_no_rag_target_teacher_forced" if is_fixed_target_trace(row) else "independent_generation_ppl"
        ),
        "ppl_comparison_version": row_ppl_protocol(row),
        "no_doc_correct": no_doc_correct(row),
        "with_doc_correct": with_doc_correct(row),
        "no_doc_prediction": row.get("no_doc_prediction"),
        "with_doc_prediction": with_doc_prediction(row),
        "with_doc_answer_status": free.get("answer_status") if is_fixed_target_trace(row) else "valid_option",
        "fixed_target_sha256": (row.get("teacher_forced_ppl") or {}).get("fixed_target_sha256"),
        "rationale_only_delta_ppl": delta,
        "tau": tau,
        "pseudo_label": label,
        "use_for_training": use_for_training,
        "quality_pass": not failures,
        "quality_failures": failures,
    }


def build_training_splits(
    args: argparse.Namespace,
    assignments: dict[str, str],
    taus: dict[str, float],
    trace_rows: Counter[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    datasets = set(trace_rows)
    handles, paths = open_outputs(
        args.output_root, datasets, write_all_labels=args.write_all_labels, overwrite=args.overwrite
    )
    stats: dict[str, dict[str, Any]] = {
        dataset: {
            "trace_rows": 0,
            "samples": {split: set() for split in SPLITS},
            "label_counts": {split: Counter() for split in SPLITS},
            "training_source": {split: Counter() for split in SPLITS},
            "training_doc_rank": {split: Counter() for split in SPLITS},
            "quality_failures": Counter(),
            "ppl_comparison_versions": Counter(),
        }
        for dataset in datasets
    }
    try:
        for path in args.trace_paths:
            size = path.stat().st_size
            with tqdm(total=size, desc=f"Labeling:{path.parent.parent.name}", unit="B", unit_scale=True) as progress:
                with path.open("rb", buffering=64 * 1024 * 1024) as handle:
                    for line in handle:
                        progress.update(len(line))
                        if not line.strip():
                            continue
                        row = decode_json(line)
                        if row.get("support_repair"):
                            continue
                        dataset = str(row["dataset"])
                        sample_id = str(row["sample_id"])
                        split = assignments[sample_id]
                        dataset_stats = stats[dataset]
                        dataset_stats["trace_rows"] += 1
                        dataset_stats["ppl_comparison_versions"][row_ppl_protocol(row)] += 1
                        dataset_stats["samples"][split].add(sample_id)
                        failures = trace_quality_failures(row)
                        delta = rationale_only_delta_ppl(row)
                        if failures or delta is None:
                            if delta is None and "invalid_rationale_only_delta_ppl" not in failures:
                                failures = [*failures, "invalid_rationale_only_delta_ppl"]
                            dataset_stats["quality_failures"].update(failures)
                            label, use_for_training, tau = LABEL_EXCLUDED, False, None
                        else:
                            tau = taus[dataset]
                            label, use_for_training = assign_label(
                                no_doc_correct(row), with_doc_correct(row), delta, tau
                            )
                        dataset_stats["label_counts"][split][label] += 1
                        record = compact_label_record(
                            row,
                            split=split,
                            label=label,
                            use_for_training=use_for_training,
                            delta=delta,
                            tau=tau,
                            failures=failures,
                        )
                        if args.write_all_labels:
                            handles[dataset]["all_labels"].write(json.dumps(record, ensure_ascii=False) + "\n")
                        if label == LABEL_EXCLUDED:
                            handles[dataset]["excluded"].write(json.dumps(record, ensure_ascii=False) + "\n")
                            continue
                        if label == LABEL_DISCARD:
                            handles[dataset]["discarded"].write(json.dumps(record, ensure_ascii=False) + "\n")
                            continue

                        document = row.get("doc") if isinstance(row.get("doc"), dict) else {}
                        training_row = {
                            **record,
                            "input": filter_input(row, args.max_doc_chars),
                            "target": "helpful" if label == LABEL_HELPFUL else "not helpful",
                            "label": label,
                            "answers": row.get("answers"),
                        }
                        handles[dataset][split].write(json.dumps(training_row, ensure_ascii=False) + "\n")
                        dataset_stats["training_source"][split][str(document.get("source") or "unknown")] += 1
                        dataset_stats["training_doc_rank"][split][str(row.get("doc_rank"))] += 1
    finally:
        close_outputs(handles)

    summaries: dict[str, Any] = {}
    for dataset, dataset_stats in sorted(stats.items()):
        splits = {}
        for split in SPLITS:
            counts = dataset_stats["label_counts"][split]
            splits[split] = {
                "sample_ids": len(dataset_stats["samples"][split]),
                "rows": int(sum(counts.values())),
                "label_counts": dict(counts),
                "training_rows": int(counts[LABEL_HELPFUL] + counts[LABEL_NOT_HELPFUL]),
                "training_source": dict(dataset_stats["training_source"][split]),
                "training_doc_rank": dict(dataset_stats["training_doc_rank"][split]),
            }
        split_sets = {split: dataset_stats["samples"][split] for split in SPLITS}
        leakage = {
            "train_val": len(split_sets["train"] & split_sets["val"]),
            "train_test": len(split_sets["train"] & split_sets["test"]),
            "val_test": len(split_sets["val"] & split_sets["test"]),
        }
        if any(leakage.values()):
            raise RuntimeError(f"Question leakage detected for {dataset}: {leakage}")
        summaries[dataset] = {
            "trace_rows": dataset_stats["trace_rows"],
            "splits": splits,
            "quality_failures": dict(dataset_stats["quality_failures"]),
            "ppl_comparison_versions": dict(dataset_stats["ppl_comparison_versions"]),
            "leakage_check": leakage,
        }
    return summaries, paths


def write_sample_id_files(output_root: Path, assignments: dict[str, str], sample_to_dataset: dict[str, str]) -> None:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample_id, dataset in sample_to_dataset.items():
        grouped[dataset][assignments[sample_id]].append(sample_id)
    for dataset, by_split in grouped.items():
        directory = output_root / dataset / "sample_ids"
        directory.mkdir(exist_ok=True)
        for split in SPLITS:
            ids = sorted(by_split[split])
            (directory / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)

    sample_to_dataset, trace_rows = scan_questions(args.trace_paths)
    assignments = make_assignments(
        sample_to_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    taus, threshold_summaries = train_thresholds(
        args.trace_paths, assignments, args.threshold_quantile
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries, output_paths = build_training_splits(args, assignments, taus, trace_rows)
    write_sample_id_files(args.output_root, assignments, sample_to_dataset)

    for dataset, summary in summaries.items():
        manifest = {
            "type": "rag2_filter_training_splits",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": dataset,
            "trace_paths": [str(path) for path in args.trace_paths],
            "split_unit": "sample_id",
            "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
            "seed": args.seed,
            "label_protocol": {
                "ppl_scope": (
                    "for fixed-target traces: teacher-forced PPL of the identical no-RAG rationale-only span; "
                    "for paper-exact free-response traces: independently generated complete visible response PPL "
                    "(reasoning plus expressed answer), because the published prompt defines no stable delimiter; "
                    "other legacy free-generation traces retain their recorded PPL unit"
                ),
                "delta_ppl": (
                    "no-document PPL - one-document PPL over the same row-level recorded PPL unit; positive means "
                    "the document lowered PPL. No answer or rationale is teacher-forced for paper-exact traces."
                ),
                "threshold": "train-only per-dataset quantile",
                "threshold_quantile": args.threshold_quantile,
                "tau": taus[dataset],
                "branches": {
                    "wrong_to_correct": "Helpful",
                    "correct_to_wrong": "Not Helpful",
                    "correct_to_correct": "Helpful if Delta-PPL >= tau, otherwise Discard",
                    "wrong_to_wrong": "Not Helpful if Delta-PPL >= tau, otherwise Discard",
                },
            },
            "quality_policy": {
                "exclude": [
                    "teacher-forced PPL/hash failures",
                    "technical or malformed free-generation outcomes",
                    "missing answer transition or rationale-only PPL",
                    "output truncated at max tokens",
                ],
                "retain": (
                    "minor terminal sentence formatting deviations when parsing succeeds; explicit documented "
                    "out-of-options/refusal responses are retained as non-correct harmful outcomes without coercion"
                ),
            },
            "filter_input": {
                "format": "official RAG2 evidence-then-question template",
                "document_character_cap": args.max_doc_chars,
                "encoder_overlength_policy": "handled by the training script: exclude > max_seq_length without truncation/windows",
            },
            "threshold_summary": threshold_summaries[dataset],
            "summary": summary,
            "files": {name: str(path) for name, path in output_paths[dataset].items()},
            "sample_id_files": {
                split: str(args.output_root / dataset / "sample_ids" / f"{split}.txt") for split in SPLITS
            },
        }
        (args.output_root / dataset / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logging.info("[%s] split summary: %s", dataset, summary["splits"])


if __name__ == "__main__":
    main()

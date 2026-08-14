#!/usr/bin/env python3
"""Audit completed RAG2 single-document generations without copying large traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from array import array
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

from tqdm import tqdm

try:
    import msgspec

    _DECODER = msgspec.json.Decoder(dict)

    def decode_json(line: bytes) -> dict[str, Any]:
        return _DECODER.decode(line)

except ImportError:

    def decode_json(line: bytes) -> dict[str, Any]:
        return json.loads(line)


ANSWER_PATTERN = re.compile(
    r"\btherefore\s*,?\s+the\s+answer\s+is\s*\(?\s*([A-Za-z])\s*\)?",
    re.IGNORECASE,
)
SELF_CORRECTION_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)(?:wait\s*[,!:]|re-?evaluat(?:e|ing)\b|"
    r"reconsider(?:ing)?\b|on second thought\b|correction\s*:|"
    r"let(?:'|’)s (?:recheck|reconsider|re-evaluate)\b|actually,?\s+no\b)",
    re.IGNORECASE,
)
REFUSAL_PATTERN = re.compile(
    r"\b(?:if forced to (?:choose|select)|"
    r"question (?:is|appears|seems)(?: to be)? (?:ambiguous|flawed|incorrect)|"
    r"question (?:contains|appears to contain) (?:an error|a typo)|"
    r"none of the (?:provided )?(?:options|choices) (?:is|are|can be) "
    r"(?:correct|accurate|supported|definitively supported)|"
    r"no (?:correct|valid|accurate) (?:answer|option) (?:is|appears)|"
    r"not enough information (?:is provided )?to (?:answer|choose|determine))\b",
    re.IGNORECASE,
)
META_DOCUMENT_PATTERN = re.compile(
    r"\b(?:provided|retrieved|given|supplied|reference) (?:document|text|context)|"
    r"\b(?:document|text) (?:states|mentions|describes|indicates|suggests|does not)\b",
    re.IGNORECASE,
)
HEADING_PATTERN = re.compile(
    r"^\s*(?:analysis|explanation|answer|final answer|reasoning)\s*:|"
    r"^\s*(?:[-*]|\d+[.)])\s+",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["medmcqa", "medqa"])
    parser.add_argument("--valid-sample-size", type=int, default=40)
    return parser.parse_args()


def finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def percentile(values: array, fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return int(ordered[round((len(ordered) - 1) * fraction)])


def compact_record(
    row: dict[str, Any], reasons: list[str], *, kind: str, line_number: int
) -> dict[str, Any]:
    document = row.get("doc") if isinstance(row.get("doc"), dict) else {}
    raw = str(row.get("with_doc_raw_generation") or "")
    rationale = str(row.get("with_doc_rationale_only") or "")
    return {
        "kind": kind,
        "dataset": row.get("dataset"),
        "line_number": line_number,
        "pair_id": row.get("pair_id"),
        "sample_id": row.get("sample_id"),
        "doc_rank": row.get("doc_rank"),
        "source": document.get("source"),
        "stable_id": document.get("stable_id"),
        "reasons": reasons,
        "prediction": row.get("with_doc_prediction"),
        "gold_answer": row.get("answer"),
        "finish_reason": row.get("with_doc_finish_reason"),
        "generation_attempts": row.get("with_doc_generation_attempts"),
        "raw_generation_preview": raw[:1600],
        "rationale_only_preview": rationale[:1200],
    }


def write_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_exceptions(path: Path) -> tuple[dict[str, dict[str, Any]], Counter, Counter]:
    rows: dict[str, dict[str, Any]] = {}
    quality_counts: Counter = Counter()
    parse_counts: Counter = Counter()
    if not path.exists():
        return rows, quality_counts, parse_counts
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = decode_json(line)
            pair_id = str(row.get("pair_id") or "")
            if pair_id:
                rows[pair_id] = row
            quality_counts.update(row.get("quality_issues") or [])
            parse_counts.update(row.get("parse_errors") or [])
    return rows, quality_counts, parse_counts


def audit_dataset(
    dataset: str,
    config: dict[str, Any],
    summary_row: dict[str, Any],
    output_dir: Path,
    valid_sample_size: int,
) -> dict[str, Any]:
    trace_path = Path(summary_row["output_path"])
    exception_path = Path(summary_row["exception_output_path"])
    expected_pairs = int(summary_row["expected_pairs"])
    expected_questions = int(summary_row["selected_questions"])
    expected_prompt = str(config.get("document_prompt_version") or "")
    exceptions, exception_quality, exception_parse = load_exceptions(exception_path)

    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    regeneration_handle = (dataset_dir / "regeneration_targets.jsonl").open("w", encoding="utf-8")
    ppl_handle = (dataset_dir / "ppl_rescore_targets.jsonl").open("w", encoding="utf-8")
    review_handle = (dataset_dir / "manual_review_targets.jsonl").open("w", encoding="utf-8")
    incomplete_handle = (dataset_dir / "incomplete_samples.jsonl").open("w", encoding="utf-8")

    generation_counts: Counter = Counter()
    ppl_counts: Counter = Counter()
    suspicious_counts: Counter = Counter()
    prediction_counts: Counter = Counter()
    finish_counts: Counter = Counter()
    token_counts = array("I")
    rationale_token_counts = array("I")
    valid_samples: list[dict[str, Any]] = []
    randomizer = random.Random(1729 if dataset == "medmcqa" else 2718)
    structurally_valid_seen = 0
    audited_exception_ids: set[str] = set()
    generation_invalid_ids: set[str] = set()
    ppl_only_ids: set[str] = set()
    sample_ids_seen: set[str] = set()
    malformed_json_lines = 0
    duplicate_pair_rows = 0
    pair_ids_seen: set[str] = set()
    total_rows = 0

    current_sample = ""
    current_ranks: list[int] = []

    def finish_sample() -> None:
        nonlocal current_sample, current_ranks
        if not current_sample:
            return
        reasons: list[str] = []
        if len(current_ranks) != 10:
            reasons.append(f"row_count_{len(current_ranks)}")
        if sorted(current_ranks) != list(range(1, 11)):
            reasons.append("ranks_not_1_to_10")
        if reasons:
            write_jsonl(
                incomplete_handle,
                {"dataset": dataset, "sample_id": current_sample, "ranks": current_ranks, "reasons": reasons},
            )

    try:
        with trace_path.open("rb") as handle, tqdm(
            total=expected_pairs,
            desc=f"Audit:{dataset}",
            unit="pair",
            dynamic_ncols=True,
            mininterval=2.0,
        ) as progress:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = decode_json(line)
                except Exception as exc:  # keep scanning after a corrupt line
                    malformed_json_lines += 1
                    write_jsonl(
                        regeneration_handle,
                        {
                            "kind": "generation",
                            "dataset": dataset,
                            "line_number": line_number,
                            "pair_id": None,
                            "reasons": ["malformed_json"],
                            "error": str(exc),
                        },
                    )
                    progress.update(1)
                    continue

                total_rows += 1
                pair_id = str(row.get("pair_id") or "")
                sample_id = str(row.get("sample_id") or "")
                if sample_id != current_sample:
                    finish_sample()
                    if sample_id in sample_ids_seen:
                        write_jsonl(
                            incomplete_handle,
                            {
                                "dataset": dataset,
                                "sample_id": sample_id,
                                "reasons": ["sample_rows_not_contiguous"],
                            },
                        )
                    sample_ids_seen.add(sample_id)
                    current_sample = sample_id
                    current_ranks = []
                try:
                    rank = int(row.get("doc_rank"))
                except (TypeError, ValueError):
                    rank = -1
                current_ranks.append(rank)

                generation_reasons: list[str] = []
                ppl_reasons: list[str] = []
                suspicious_reasons: list[str] = []

                if not pair_id:
                    generation_reasons.append("missing_pair_id")
                elif pair_id in pair_ids_seen:
                    duplicate_pair_rows += 1
                    generation_reasons.append("duplicate_pair_id")
                pair_ids_seen.add(pair_id)
                if not sample_id:
                    generation_reasons.append("missing_sample_id")
                if rank not in range(1, 11):
                    generation_reasons.append("invalid_doc_rank")
                if row.get("dataset") != dataset:
                    generation_reasons.append("dataset_mismatch")
                if expected_prompt and row.get("prompt_version") != expected_prompt:
                    generation_reasons.append("prompt_version_mismatch")

                options = row.get("options") if isinstance(row.get("options"), dict) else {}
                valid_options = {str(key).upper() for key in options}
                prediction = str(row.get("with_doc_prediction") or "").upper()
                prediction_counts[prediction or "<missing>"] += 1
                raw = str(row.get("with_doc_raw_generation") or "").strip()
                rationale = str(row.get("with_doc_rationale") or "").strip()
                rationale_only = str(row.get("with_doc_rationale_only") or "").strip()
                conclusion = str(row.get("with_doc_answer_conclusion") or "").strip()
                parse_errors = [str(value) for value in (row.get("with_doc_parse_errors") or [])]
                stored_quality = [str(value) for value in (row.get("with_doc_quality_issues") or [])]
                finish_reason = str(row.get("with_doc_finish_reason") or "<missing>")
                finish_counts[finish_reason] += 1

                if not raw:
                    generation_reasons.append("empty_raw_generation")
                if not rationale:
                    generation_reasons.append("missing_rationale")
                if not rationale_only:
                    generation_reasons.append("missing_rationale_only")
                if not prediction or prediction not in valid_options:
                    generation_reasons.append("missing_or_invalid_prediction")
                if parse_errors:
                    generation_reasons.extend(f"parse:{value}" for value in parse_errors)

                generation_issue_names = {
                    "parse_errors",
                    "missing_rationale",
                    "missing_final_answer",
                    "max_tokens_exhausted",
                    "unexpected_answer_conclusion_count",
                    "empty_raw_generation",
                }
                generation_reasons.extend(value for value in stored_quality if value in generation_issue_names)
                ppl_reasons.extend(
                    value for value in stored_quality if value in {"missing_rationale_ppl", "missing_rationale_only_ppl"}
                )
                if finish_reason == "length" or bool(row.get("with_doc_truncated_by_max_tokens")):
                    generation_reasons.append("truncated_by_max_tokens")

                answer_matches = ANSWER_PATTERN.findall(raw)
                if len(answer_matches) != 1:
                    generation_reasons.append(f"answer_conclusion_count_{len(answer_matches)}")
                elif prediction and answer_matches[0].upper() != prediction:
                    generation_reasons.append("conclusion_prediction_mismatch")
                if not conclusion:
                    generation_reasons.append("missing_answer_conclusion")
                elif raw and not raw.endswith(conclusion):
                    generation_reasons.append("answer_conclusion_not_terminal")

                stats = row.get("with_doc_generation_stats")
                stats = stats if isinstance(stats, dict) else {}
                rationale_stats = stats.get("rationale") if isinstance(stats.get("rationale"), dict) else {}
                rationale_only_stats = (
                    stats.get("rationale_only") if isinstance(stats.get("rationale_only"), dict) else {}
                )
                full_stats = stats.get("full_generation") if isinstance(stats.get("full_generation"), dict) else {}
                if not finite_positive(rationale_stats.get("ppl")):
                    ppl_reasons.append("invalid_rationale_ppl")
                if not finite_positive(rationale_only_stats.get("ppl")):
                    ppl_reasons.append("invalid_rationale_only_ppl")
                full_tokens = int(full_stats.get("token_count") or 0)
                rationale_tokens = int(rationale_only_stats.get("token_count") or 0)
                if full_tokens > 0:
                    token_counts.append(full_tokens)
                if rationale_tokens > 0:
                    rationale_token_counts.append(rationale_tokens)

                if rationale_tokens and rationale_tokens < 20:
                    suspicious_reasons.append("very_short_rationale")
                if full_tokens >= 350 and finish_reason != "length":
                    suspicious_reasons.append("near_token_limit")
                if SELF_CORRECTION_PATTERN.search(raw):
                    suspicious_reasons.append("self_correction_language")
                if REFUSAL_PATTERN.search(raw):
                    suspicious_reasons.append("refusal_or_ambiguous_question_language")
                if HEADING_PATTERN.search(raw):
                    suspicious_reasons.append("extra_heading_or_list_format")
                if META_DOCUMENT_PATTERN.search(raw):
                    suspicious_counts["explicit_document_meta_reference"] += 1

                if conclusion and prediction in valid_options:
                    expected_option = normalized_text(options.get(prediction))
                    normalized_conclusion = normalized_text(conclusion)
                    if expected_option and expected_option not in normalized_conclusion:
                        suspicious_counts["non_exact_answer_option_text"] += 1
                        other_option_matches = [
                            label
                            for label, option_text in options.items()
                            if str(label).upper() != prediction
                            and len(normalized_text(option_text)) >= 4
                            and normalized_text(option_text) in normalized_conclusion
                        ]
                        if other_option_matches:
                            suspicious_reasons.append("conclusion_mentions_other_option_text")

                generation_reasons = sorted(set(generation_reasons))
                ppl_reasons = sorted(set(ppl_reasons))
                suspicious_reasons = sorted(set(suspicious_reasons))
                generation_counts.update(generation_reasons)
                ppl_counts.update(ppl_reasons)
                suspicious_counts.update(suspicious_reasons)

                if pair_id in exceptions:
                    audited_exception_ids.add(pair_id)
                if generation_reasons:
                    generation_invalid_ids.add(pair_id)
                    write_jsonl(
                        regeneration_handle,
                        compact_record(row, generation_reasons, kind="generation", line_number=line_number),
                    )
                elif ppl_reasons:
                    ppl_only_ids.add(pair_id)
                    write_jsonl(
                        ppl_handle,
                        compact_record(row, ppl_reasons, kind="ppl_rescore", line_number=line_number),
                    )
                else:
                    structurally_valid_seen += 1
                    if len(valid_samples) < valid_sample_size:
                        valid_samples.append(compact_record(row, [], kind="valid_sample", line_number=line_number))
                    else:
                        sample_index = randomizer.randrange(structurally_valid_seen)
                        if sample_index < valid_sample_size:
                            valid_samples[sample_index] = compact_record(
                                row, [], kind="valid_sample", line_number=line_number
                            )
                if suspicious_reasons and not generation_reasons:
                    write_jsonl(
                        review_handle,
                        compact_record(row, suspicious_reasons, kind="manual_review", line_number=line_number),
                    )
                progress.update(1)
        finish_sample()
    finally:
        regeneration_handle.close()
        ppl_handle.close()
        review_handle.close()
        incomplete_handle.close()

    with (dataset_dir / "valid_random_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in valid_samples:
            write_jsonl(handle, row)

    exception_ids = set(exceptions)
    detected_issue_ids = generation_invalid_ids | ppl_only_ids
    result = {
        "dataset": dataset,
        "trace_path": str(trace_path),
        "expected_questions": expected_questions,
        "audited_questions": len(sample_ids_seen),
        "expected_pairs": expected_pairs,
        "audited_rows": total_rows,
        "malformed_json_lines": malformed_json_lines,
        "duplicate_pair_rows": duplicate_pair_rows,
        "generation_invalid_pairs": len(generation_invalid_ids),
        "generation_invalid_percent": round(100 * len(generation_invalid_ids) / max(total_rows, 1), 5),
        "ppl_only_invalid_pairs": len(ppl_only_ids - generation_invalid_ids),
        "ppl_only_invalid_percent": round(100 * len(ppl_only_ids - generation_invalid_ids) / max(total_rows, 1), 5),
        "structurally_valid_pairs": total_rows - len(generation_invalid_ids),
        "generation_reason_counts": dict(generation_counts.most_common()),
        "ppl_reason_counts": dict(ppl_counts.most_common()),
        "suspicious_reason_counts": dict(suspicious_counts.most_common()),
        "prediction_counts": dict(prediction_counts),
        "finish_reason_counts": dict(finish_counts),
        "full_generation_tokens": {
            "count": len(token_counts),
            "min": min(token_counts) if token_counts else None,
            "median": percentile(token_counts, 0.5),
            "p95": percentile(token_counts, 0.95),
            "p99": percentile(token_counts, 0.99),
            "max": max(token_counts) if token_counts else None,
            "mean": round(sum(token_counts) / len(token_counts), 2) if token_counts else None,
        },
        "rationale_only_tokens": {
            "count": len(rationale_token_counts),
            "min": min(rationale_token_counts) if rationale_token_counts else None,
            "median": percentile(rationale_token_counts, 0.5),
            "p95": percentile(rationale_token_counts, 0.95),
            "p99": percentile(rationale_token_counts, 0.99),
            "max": max(rationale_token_counts) if rationale_token_counts else None,
            "mean": round(sum(rationale_token_counts) / len(rationale_token_counts), 2)
            if rationale_token_counts
            else None,
        },
        "generator_exception_pairs": len(exception_ids),
        "generator_exception_quality_counts": dict(exception_quality.most_common()),
        "generator_exception_parse_counts": dict(exception_parse.most_common()),
        "exceptions_not_found_in_trace": len(exception_ids - audited_exception_ids),
        "exceptions_without_detected_issue": len(exception_ids - detected_issue_ids),
        "detected_issues_missing_from_exception_file": len(detected_issue_ids - exception_ids),
        "artifacts": {
            "regeneration_targets": str(dataset_dir / "regeneration_targets.jsonl"),
            "ppl_rescore_targets": str(dataset_dir / "ppl_rescore_targets.jsonl"),
            "manual_review_targets": str(dataset_dir / "manual_review_targets.jsonl"),
            "incomplete_samples": str(dataset_dir / "incomplete_samples.jsonl"),
            "valid_random_sample": str(dataset_dir / "valid_random_sample.jsonl"),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        generation_summary = json.load(handle)
    with (run_dir / "run_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (run_dir / "quality_audit" / timestamp)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset in args.datasets:
        if dataset not in generation_summary:
            raise KeyError(f"Dataset missing from generation summary: {dataset}")
        results.append(
            audit_dataset(
                dataset,
                config,
                generation_summary[dataset],
                output_dir,
                args.valid_sample_size,
            )
        )

    aggregate = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "datasets": results,
        "totals": {
            "expected_pairs": sum(row["expected_pairs"] for row in results),
            "audited_rows": sum(row["audited_rows"] for row in results),
            "generation_invalid_pairs": sum(row["generation_invalid_pairs"] for row in results),
            "ppl_only_invalid_pairs": sum(row["ppl_only_invalid_pairs"] for row in results),
            "generator_exception_pairs": sum(row["generator_exception_pairs"] for row in results),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    lines = [
        "# RAG2 Document Trace Quality Audit",
        "",
        f"- Run: `{run_dir}`",
        f"- Audited pairs: {aggregate['totals']['audited_rows']:,}",
        f"- Generation regeneration targets: {aggregate['totals']['generation_invalid_pairs']:,}",
        f"- PPL-only rescore targets: {aggregate['totals']['ppl_only_invalid_pairs']:,}",
        "",
        "| Dataset | Pairs | Generation invalid | PPL-only | Valid |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        valid = row["audited_rows"] - row["generation_invalid_pairs"]
        lines.append(
            f"| {row['dataset']} | {row['audited_rows']:,} | "
            f"{row['generation_invalid_pairs']:,} ({row['generation_invalid_percent']:.5f}%) | "
            f"{row['ppl_only_invalid_pairs']:,} | {valid:,} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["totals"], ensure_ascii=False, indent=2))
    print(f"Audit output: {output_dir}")


if __name__ == "__main__":
    main()

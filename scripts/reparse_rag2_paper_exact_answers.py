from __future__ import annotations

"""Re-extract MCQ answers from saved RAG2 paper-exact generations.

This utility deliberately does not load an LLM, change a prompt, regenerate a
response, or alter the retrieval query text.  It only replaces the parsed
answer metadata beside the already-saved raw model output.  The result is
written to a new artifact root so the original generation artifact remains a
reproducible immutable record.
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.io_utils import iter_jsonl, read_json, write_json
from medrag.rag2_mcq import PAPER_EXACT_PROMPT_VERSION, parse_paper_exact_mcq_output


DATASETS = [
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]
ANSWER_EXTRACTION_VERSION = "paper_exact_terminal_decision_sentence_no_rewrite_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reparse answers in existing paper-exact no-RAG generations. "
            "No LLM or GPU is used; model_raw_generation and retrieval query text are preserved verbatim."
        )
    )
    parser.add_argument("--input-artifact-root", required=True, type=Path)
    parser.add_argument("--output-artifact-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input-file", default="no_rag_generations.jsonl")
    parser.add_argument("--output-file", default="no_rag_generations.jsonl")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSONL under --output-artifact-root. The input root can never be overwritten.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def artifact_dir(root: Path, dataset: str, split: str) -> Path:
    return root / "no_rag" / dataset / split


def row_gold_answers(row: dict[str, Any]) -> set[str]:
    values = row.get("gold_answers")
    if not isinstance(values, list) or not values:
        values = [row.get("gold_answer")]
    return {str(value).upper() for value in values if str(value or "").strip()}


def update_row(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    raw_text = str(result.get("model_raw_generation") or result.get("no_rag_generation") or "")
    parsed = parse_paper_exact_mcq_output(raw_text, result.get("options"))
    answer = parsed.final_answer

    # The paper-exact retrieval protocol always embeds the full visible raw
    # response.  Do not replace it with the parser's convenience field or a
    # canonicalized version.
    rationale_query = parsed.visible_text
    result["schema_version"] = max(int(result.get("schema_version") or 0), 3)
    result["answer_extraction_method"] = ANSWER_EXTRACTION_VERSION
    result["retrieval_query_policy"] = "complete_visible_response_including_expressed_answer_no_rewrite_v1"
    result["answer_reparsed_from_raw_generation"] = True
    result["parsed"] = {
        "visible_text": parsed.visible_text,
        "rationale": parsed.rationale,
        "rationale_only": parsed.rationale_only,
        "rationale_query": rationale_query,
        "answer_conclusion": parsed.answer_conclusion,
        "rationale_query_normalized": False,
        "final_answer": answer,
        "final_answer_correct": answer in row_gold_answers(result) if answer is not None else False,
        "parse_errors": parsed.parse_errors,
    }
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    answers = 0
    correct = 0
    for row in rows:
        parsed = row["parsed"]
        answers += int(parsed.get("final_answer") is not None)
        correct += int(bool(parsed.get("final_answer_correct")))
        errors.update(str(error) for error in parsed.get("parse_errors") or [])
    total = len(rows)
    return {
        "rows": total,
        "answers_extracted": answers,
        "answer_extraction_rate": (answers / total) if total else None,
        "correct_answers": correct,
        "accuracy_over_all_rows": (correct / total) if total else None,
        "accuracy_among_extracted": (correct / answers) if answers else None,
        "remaining_parse_errors": dict(sorted(errors.items())),
    }


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reparse_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    input_dir = artifact_dir(args.input_artifact_root, dataset, args.split)
    output_dir = artifact_dir(args.output_artifact_root, dataset, args.split)
    input_path = input_dir / args.input_file
    output_path = output_dir / args.output_file
    manifest_path = input_dir / "manifest.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input generation artifact: {input_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing input manifest: {manifest_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output-artifact-root must differ from --input-artifact-root; raw generations are immutable.")
    source_manifest = read_json(manifest_path)
    if source_manifest.get("prompt_profile") != "paper_exact":
        raise ValueError(
            f"{manifest_path} is not a paper_exact artifact: prompt_profile={source_manifest.get('prompt_profile')!r}"
        )
    if source_manifest.get("prompt_version") != PAPER_EXACT_PROMPT_VERSION:
        raise ValueError(
            f"Unexpected paper-exact prompt version in {manifest_path}: {source_manifest.get('prompt_version')!r}"
        )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace only this derived artifact.")

    rows = [update_row(row) for row in iter_jsonl(input_path)]
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")
    report = {
        "dataset": dataset,
        "split": args.split,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_manifest_path": str(manifest_path),
        **summarize(rows),
    }
    if args.dry_run:
        return report

    write_rows_atomic(output_path, rows)
    output_manifest = copy.deepcopy(source_manifest)
    output_manifest.update(
        {
            # Keep this type unchanged: downstream embedding/evaluation code
            # validates it as a compatible no-RAG artifact.
            "type": "rag2_no_rag_rationale_artifact",
            "created_or_updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "output_path": str(output_path),
            "answer_extraction": (
                "Post-generation reparse of the original raw response. Markdown final-answer markers and "
                "explicit decision cues are accepted; the response and retrieval query are never rewritten."
            ),
            "answer_extraction_version": ANSWER_EXTRACTION_VERSION,
            "retrieval_query_policy": "complete_visible_response_including_expressed_answer_no_rewrite_v1",
            "source_artifact_path": str(input_path),
            "source_manifest_path": str(manifest_path),
            "reparse_report": report,
        }
    )
    write_json(output_dir / "manifest.json", output_manifest)
    return report


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.input_artifact_root = args.input_artifact_root.resolve()
    args.output_artifact_root = args.output_artifact_root.resolve()
    reports = [reparse_dataset(args, dataset) for dataset in args.datasets]
    total_rows = sum(int(report["rows"]) for report in reports)
    total_answers = sum(int(report["answers_extracted"]) for report in reports)
    total_correct = sum(int(report["correct_answers"]) for report in reports)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "dry_run" if args.dry_run else "write",
        "input_artifact_root": str(args.input_artifact_root),
        "output_artifact_root": str(args.output_artifact_root),
        "answer_extraction_version": ANSWER_EXTRACTION_VERSION,
        "rows": total_rows,
        "answers_extracted": total_answers,
        "answer_extraction_rate": (total_answers / total_rows) if total_rows else None,
        "correct_answers": total_correct,
        "accuracy_over_all_rows": (total_correct / total_rows) if total_rows else None,
        "datasets": reports,
    }
    if not args.dry_run:
        write_json(args.output_artifact_root / "reparse_summary.json", summary)
    logging.info(
        "Reparse complete: extracted=%s/%s (%.2f%%), raw generations unchanged%s",
        total_answers,
        total_rows,
        100 * total_answers / total_rows if total_rows else 0.0,
        " [dry run]" if args.dry_run else "",
    )
    for report in reports:
        logging.info(
            "[%s] extracted=%s/%s (%.2f%%), accuracy=%s",
            report["dataset"],
            report["answers_extracted"],
            report["rows"],
            100 * report["answer_extraction_rate"] if report["answer_extraction_rate"] is not None else 0.0,
            "n/a"
            if report["accuracy_over_all_rows"] is None
            else f"{100 * report['accuracy_over_all_rows']:.2f}%",
        )


if __name__ == "__main__":
    main()

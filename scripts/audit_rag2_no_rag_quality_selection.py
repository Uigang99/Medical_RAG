from __future__ import annotations

"""Create a conservative, auditable selection of usable RAG2 no-RAG traces.

This utility never changes a raw LLM generation.  It writes only row-index
selection metadata and, for a small set of clearly expressed free-form
answers, the answer that can be recovered deterministically from the raw
response.  Gold labels are retained for auditing but are never read by the
selection logic.
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "filtering"
    / "rag2"
    / "llama3_8b_paper_exact_free_response_v2"
    / "no_rag_rationales_train"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "filtering"
    / "rag2"
    / "llama3_8b_paper_exact_free_response_v2"
    / "no_rag_quality_selection_v1"
)
DEFAULT_QUERY_ENCODER = Path("/home/user/Uiheon/models/MedCPT-Query-Encoder")


MARKDOWN_MARKERS = re.compile(r"(?:\*{1,3}|_{1,3}|`)")
REFUSAL_OR_NON_OPTION = re.compile(
    r"(?is)\b(?:"
    r"none\s+of\s+(?:the\s+)?(?:above|options?|choices?)|"
    r"all\s+(?:the\s+)?options?\s+(?:are\s+)?correct|"
    r"not\s+(?:among|one\s+of)\s+(?:the\s+)?(?:options?|choices?)|"
    r"no\s+(?:correct\s+)?(?:option|answer|choice)\s+(?:is\s+)?"
    r"(?:among|provided|given|available|mentioned)|"
    r"cannot\s+(?:choose|select|determine|provide|answer)|"
    r"(?:unable|impossible)\s+to\s+(?:choose|select|determine|provide|answer)|"
    r"insufficient\s+information|question\s+(?:is|seems)\s+(?:flawed|ambiguous)"
    r")\b"
)
MULTIPLE_EXPLICIT_ANSWER = re.compile(
    r"(?is)\b(?:the\s+)?(?:final\s+)?(?:correct|best)?\s*"
    r"(?:answer|option|choice)\b\s*(?:is\s*:?|:|-)?\s*(?:the\s+)?"
    r"(?:correct\s+)?(?:option\s*)?[\(\[]?\s*([A-Za-z])\s*[\)\]]?"
    r"\s*(?:and|or|,)\s*(?:option\s*)?[\(\[]?\s*([A-Za-z])"
    r"(?=\s*(?:[\)\].,:;]|$))"
)
DECISION_ANCHOR = re.compile(
    r"(?is)\b(?:"
    r"final\s+answer|correct\s+answer|(?:the\s+)?answer|"
    r"(?:the\s+)?correct\s+(?:option|choice)|conclusion|"
    r"most\s+(?:appropriate|likely|probable|important|common)\b[^\n]{0,100}|"
    r"best\s+(?:answer|option|choice)|next\s+step|"
    r"treatment\s+of\s+choice|first\s+investigation"
    r")\b"
)
BARE_OPTION_LABEL = re.compile(
    r"(?is)^\s*(?:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)"
    r"\s*(?:is\s*:?|:|-)?\s*)?[\(\[]?\s*([A-Za-z])"
    r"(?=\s*(?:[\)\].,:;\-]|$))"
)
INLINE_DECISION_LABEL = re.compile(
    r"(?is)\b(?:final\s+answer|correct\s+answer|(?:the\s+)?answer|"
    r"(?:the\s+)?correct\s+(?:option|choice)|conclusion|"
    r"most\s+(?:appropriate|likely|probable|important|common)|"
    r"best\s+(?:answer|option|choice)|next\s+step|"
    r"treatment\s+of\s+choice|first\s+investigation)\b.{0,260}?"
    r"(?:\bis\b|:|-)\s*(?:the\s+)?(?:correct\s+)?(?:option\s*)?"
    r"[\(\[]?\s*([A-Za-z])(?=\s*[\)\].,:;\-])"
)
TERMINAL_OPTION_PARENTHETICAL = re.compile(
    r"(?is)\b(?:therefore|thus|hence|based\s+on\s+(?:the\s+)?"
    r"(?:information|findings)|most\s+(?:likely|appropriate)|best)\b.{0,360}?"
    r"\(\s*(?:option|choice)\s*([A-Za-z])\s*\)"
)
SEMANTIC_RISK_PATTERNS = {
    "question_flawed_or_ambiguous": re.compile(
        r"\b(?:question (?:is|seems) (?:flawed|ambiguous)|ambiguous question|"
        r"no correct (?:option|answer)|all (?:the )?options are correct)\b",
        re.IGNORECASE,
    ),
    "forced_choice_after_refusal": re.compile(
        r"\b(?:if i had to choose|if forced to choose|although (?:the )?question "
        r".*?(?:flawed|ambiguous))\b",
        re.IGNORECASE,
    ),
    "insufficient_information": re.compile(
        r"\b(?:cannot determine|can't determine|not enough information|"
        r"insufficient information|unable to determine)\b",
        re.IGNORECASE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and conservatively select usable paper-exact no-RAG traces."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=["medmcqa", "medqa"], default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--query-encoder-path", type=Path, default=DEFAULT_QUERY_ENCODER)
    parser.add_argument("--max-query-tokens", type=int, default=512)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def artifact_path(root: Path, dataset: str, split: str) -> Path:
    return root / "no_rag" / dataset / split / "no_rag_generations.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL: {path}:{line_no}") from exc


def raw_response(row: dict[str, Any]) -> str:
    return str(row.get("model_raw_generation") or row.get("no_rag_generation") or "")


def has_generation_ppl(row: dict[str, Any]) -> bool:
    rationale_stats = ((row.get("generation_stats") or {}).get("rationale") or {})
    return bool(rationale_stats.get("token_count")) and rationale_stats.get("ppl") is not None


def recover_explicit_free_form_answer(
    response: str,
    options: dict[str, Any],
) -> tuple[str | None, str]:
    """Recover only high-confidence choices expressed in the raw response.

    Priority is deliberate: a label immediately after a decision heading wins
    over a later explanation which may enumerate rejected options.  The
    function does not use gold labels or rewrite the original text.
    """
    visible = MARKDOWN_MARKERS.sub("", response)
    valid_options = {str(label).upper() for label in options}
    if REFUSAL_OR_NON_OPTION.search(visible):
        return None, "refusal_or_non_option"
    multiple = MULTIPLE_EXPLICIT_ANSWER.search(visible)
    if (
        multiple is not None
        and multiple.group(1).upper() in valid_options
        and multiple.group(2).upper() in valid_options
    ):
        return None, "multiple_explicit_options"

    candidates: list[tuple[int, str, str]] = []
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    for line_index, line in enumerate(lines):
        if not DECISION_ANCHOR.search(line):
            continue
        is_heading_or_decision = bool(
            re.search(r"(?is)(?:\bis\b|:|-)\s*$", line)
            or re.search(r"(?is)\b(?:final\s+answer|correct\s+answer|conclusion)\s*:?\s*$", line)
            or re.search(r"(?is)\bmost\s+(?:appropriate|likely|probable|important|common)\b", line)
        )
        if not is_heading_or_decision:
            continue
        for next_line in lines[line_index + 1 : line_index + 4]:
            match = BARE_OPTION_LABEL.match(next_line)
            if match is not None and match.group(1).upper() in valid_options:
                candidates.append(
                    (line_index, match.group(1).upper(), "decision_heading_followed_label")
                )
                break

    for match in INLINE_DECISION_LABEL.finditer(visible):
        label = match.group(1).upper()
        if label in valid_options:
            candidates.append((match.start(), label, "inline_decision_label"))
    for match in TERMINAL_OPTION_PARENTHETICAL.finditer(visible):
        label = match.group(1).upper()
        if label in valid_options:
            candidates.append((match.start(), label, "terminal_option_parenthetical"))

    if not candidates:
        return None, "unresolved"
    priority = {
        "decision_heading_followed_label": 3,
        "inline_decision_label": 2,
        "terminal_option_parenthetical": 1,
    }
    highest_priority = max(priority[method] for _, _, method in candidates)
    best = [candidate for candidate in candidates if priority[candidate[2]] == highest_priority]
    final_position = max(position for position, _, _ in best)
    labels = {label for position, label, _ in best if position == final_position}
    if len(labels) != 1:
        return None, "ambiguous_recovery_candidates"
    method = next(method for position, _, method in best if position == final_position)
    return next(iter(labels)), method


def semantic_risk_reasons(response: str) -> list[str]:
    return [
        f"semantic_risk:{name}"
        for name, pattern in SEMANTIC_RISK_PATTERNS.items()
        if pattern.search(response)
    ]


def selection_record(
    row: dict[str, Any],
    *,
    selected_answer: str,
    answer_source: str,
    query_token_count: int,
) -> dict[str, Any]:
    parsed = row.get("parsed") or {}
    return {
        "row_idx": int(row["row_idx"]),
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "selected_no_rag_answer": selected_answer,
        "answer_source": answer_source,
        "query_token_count_original": query_token_count,
        "retrieval_query_policy": row.get("retrieval_query_policy"),
        "prompt_version": row.get("prompt_version"),
        "generation_policy_version": row.get("generation_policy_version"),
        "generation_ppl": ((row.get("generation_stats") or {}).get("rationale") or {}).get("ppl"),
        "source_parse_errors": parsed.get("parse_errors") or [],
    }


def process_dataset(
    *,
    args: argparse.Namespace,
    dataset: str,
    tokenizer: Any,
) -> dict[str, Any]:
    input_path = artifact_path(args.input_root, dataset, args.split)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing no-RAG artifact: {input_path}")
    rows = list(iter_jsonl(input_path))
    row_indices = [int(row.get("row_idx", -1)) for row in rows]
    if len(set(row_indices)) != len(row_indices) or min(row_indices, default=0) < 0:
        raise ValueError(f"{dataset}: row_idx must be unique and non-negative.")

    output_dir = args.output_root / dataset / args.split
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}; use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    usable_path = output_dir / "usable_rows.jsonl"
    recovered_path = output_dir / "recovered_answers.jsonl"
    excluded_path = output_dir / "excluded_rows.jsonl"
    for path in (usable_path, recovered_path, excluded_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace it.")

    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    recovery_gold_correct = 0
    recovery_gold_total = 0
    with (
        usable_path.open("w", encoding="utf-8") as usable_out,
        recovered_path.open("w", encoding="utf-8") as recovered_out,
        excluded_path.open("w", encoding="utf-8") as excluded_out,
    ):
        for start in range(0, len(rows), args.tokenizer_batch_size):
            batch = rows[start : start + args.tokenizer_batch_size]
            queries = [str(((row.get("parsed") or {}).get("rationale_query") or raw_response(row))) for row in batch]
            encoded = tokenizer(
                queries,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            token_lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
            for row, query_tokens in zip(batch, token_lengths):
                counts["rows"] += 1
                sample_id = str(row.get("sample_id") or f"row_idx:{row.get('row_idx')}")
                if sample_id in seen_ids:
                    raise ValueError(f"{dataset}: duplicate sample_id={sample_id}")
                seen_ids.add(sample_id)
                parsed = row.get("parsed") or {}
                response = raw_response(row)

                selected_answer: str | None = None
                answer_source: str | None = None
                if parsed.get("final_answer") and not parsed.get("parse_errors"):
                    selected_answer = str(parsed["final_answer"]).upper()
                    answer_source = "stored_parser"
                    counts["stored_parser_answer"] += 1
                else:
                    selected_answer, answer_source = recover_explicit_free_form_answer(
                        response, row.get("options") or {}
                    )
                    if selected_answer is not None:
                        counts["recovered_answer"] += 1
                        counts[f"recovered_method:{answer_source}"] += 1
                        recovery_gold_total += 1
                        recovery_gold_correct += int(selected_answer == str(row.get("gold_answer") or "").upper())
                    else:
                        counts[f"no_reliable_answer:{answer_source}"] += 1

                exclusion_reasons: list[str] = []
                if selected_answer is None:
                    exclusion_reasons.append(f"no_reliable_answer:{answer_source}")
                if row.get("finish_reason") == "length" or row.get("truncated_by_max_tokens"):
                    exclusion_reasons.append("generation_max_tokens_exhausted")
                if query_tokens > args.max_query_tokens:
                    exclusion_reasons.append("medcpt_query_exceeds_max_length")
                if not has_generation_ppl(row):
                    exclusion_reasons.append("missing_generation_ppl")
                exclusion_reasons.extend(semantic_risk_reasons(response))

                if exclusion_reasons:
                    counts["excluded_rows"] += 1
                    for reason in exclusion_reasons:
                        counts[f"excluded:{reason}"] += 1
                    excluded_out.write(
                        json.dumps(
                            {
                                "row_idx": int(row["row_idx"]),
                                "sample_id": row.get("sample_id"),
                                "dataset": row.get("dataset"),
                                "split": row.get("split"),
                                "recovered_answer": selected_answer,
                                "answer_source": answer_source,
                                "query_token_count_original": query_tokens,
                                "reasons": exclusion_reasons,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                record = selection_record(
                    row,
                    selected_answer=selected_answer,
                    answer_source=str(answer_source),
                    query_token_count=query_tokens,
                )
                usable_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["usable_rows"] += 1
                if answer_source != "stored_parser":
                    recovered_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": args.split,
        "input_path": str(input_path),
        "selection_contract": {
            "answer": (
                "A stored answer without parse errors is used. Otherwise, only a deterministic, explicit "
                "decision-heading/label or terminal '(option X)' recovery is accepted; no gold label is used."
            ),
            "generation": "finish_reason must not be length and generation-time rationale PPL must be present.",
            "retrieval": f"MedCPT original query length must be <= {args.max_query_tokens} tokens.",
            "semantic": "Explicit refusal, ambiguity, insufficient-information, and forced-choice risk outputs are excluded.",
            "raw_generation": "The source model response and retrieval query are never rewritten.",
        },
        "counts": dict(counts),
        "recovered_answer_gold_accuracy_for_audit_only": (
            recovery_gold_correct / recovery_gold_total if recovery_gold_total else None
        ),
        "outputs": {
            "usable_rows": str(usable_path),
            "recovered_answers": str(recovered_path),
            "excluded_rows": str(excluded_path),
        },
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.max_query_tokens <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("--max-query-tokens and --tokenizer-batch-size must be positive.")
    if not args.query_encoder_path.exists():
        raise FileNotFoundError(f"Missing MedCPT query encoder: {args.query_encoder_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path, local_files_only=True, use_fast=True)
    reports = [process_dataset(args=args, dataset=dataset, tokenizer=tokenizer) for dataset in args.datasets]
    # A user may run the two datasets separately to keep an audit invocation
    # short.  Preserve a root summary covering every already-completed split,
    # rather than replacing it with the most recently processed dataset.
    reports_by_dataset = {str(report["dataset"]): report for report in reports}
    for dataset_dir in args.output_root.iterdir() if args.output_root.exists() else []:
        report_path = dataset_dir / args.split / "selection_report.json"
        if dataset_dir.is_dir() and report_path.exists() and dataset_dir.name not in reports_by_dataset:
            reports_by_dataset[dataset_dir.name] = json.loads(report_path.read_text(encoding="utf-8"))
    reports = [reports_by_dataset[dataset] for dataset in sorted(reports_by_dataset)]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "datasets": reports,
        "total_rows": sum(report["counts"].get("rows", 0) for report in reports),
        "total_usable_rows": sum(report["counts"].get("usable_rows", 0) for report in reports),
    }
    (args.output_root / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for report in reports:
        counts = report["counts"]
        print(
            f"{report['dataset']}: usable={counts.get('usable_rows', 0)}/{counts.get('rows', 0)} "
            f"recovered={counts.get('recovered_answer', 0)} excluded={counts.get('excluded_rows', 0)}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .core import CaseResult, RetrievedDocument

MCQ_DATASET_ORDER = [
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]


def _escape_md(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", "<br>")


def _short(text: Any, max_len: int = 120) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def _doc_ids(docs: list[RetrievedDocument], limit: int | None = None, prefer_rerank: bool = False) -> str:
    selected = docs if limit is None else docs[:limit]
    if not selected:
        return "-"
    parts = []
    for doc in selected:
        rank = doc.rerank_rank if prefer_rerank and doc.rerank_rank is not None else doc.retrieval_rank
        parts.append(f"{rank or ''}:{_escape_md(doc.stable_id)}")
    return "<br>".join(parts)


def _aggregate_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    if _is_mcq_results(results):
        return _aggregate_mcq_rows(results)

    buckets: dict[tuple[str, str], list[CaseResult]] = {}
    for result in results:
        key = (result.case_id, result.sample.dataset)
        buckets.setdefault(key, []).append(result)

    rows = []
    for (case_id, dataset), items in sorted(buckets.items()):
        total = len(items)
        if dataset == "pubmedqa":
            accuracy_value = _pubmedqa_label_accuracy(items)
            correct = int(round((accuracy_value or 0.0) * total))
            accuracy = accuracy_value if accuracy_value is not None else 0.0
            em = accuracy_value
            f1 = _label_macro_f1(items)
            bertscore_f1 = None
            bleurt = None
            prometheus2_0to1 = None
        else:
            correct = sum(1 for item in items if bool(item.evaluation.get("correct")))
            accuracy = correct / total if total else 0.0
            em_values = [float(item.evaluation["em"]) for item in items if "em" in item.evaluation]
            f1_values = [float(item.evaluation["f1"]) for item in items if "f1" in item.evaluation]
            em = sum(em_values) / len(em_values) if em_values else None
            f1 = sum(f1_values) / len(f1_values) if f1_values else None
            bertscore_f1 = _mean_optional(items, "bertscore_f1")
            bleurt = _mean_optional(items, "bleurt")
            prometheus2_0to1 = _mean_optional(items, "prometheus2_0to1")
        rows.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "total": total,
                "evaluable": total,
                "unlabeled": 0,
                "correct": correct,
                "accuracy": accuracy,
                "invalid": 0,
                "em": em,
                "f1": f1,
                "bertscore_f1": bertscore_f1,
                "bleurt": bleurt,
                "prometheus2_0to1": prometheus2_0to1,
            }
        )
    return rows


def _mean_optional(items: list[CaseResult], key: str) -> float | None:
    values = []
    for item in items:
        value = item.evaluation.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except Exception:
            continue
    return sum(values) / len(values) if values else None


def _fmt_optional(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _fmt_count(value: Any) -> str:
    return "-" if value is None else str(value)


def _is_mcq_results(results: list[CaseResult]) -> bool:
    return bool(results) and all(result.sample.task == "mcq" for result in results)


def _ordered_mcq_datasets(datasets: Iterable[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(MCQ_DATASET_ORDER)}
    return sorted(datasets, key=lambda name: (order.get(name, len(order)), name))


def _case_order(results: list[CaseResult]) -> list[str]:
    seen: dict[str, None] = {}
    for result in results:
        seen.setdefault(result.case_id, None)
    return list(seen)


def _mcq_macro_accuracy_row(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["accuracy"])
        for row in dataset_rows
        if row.get("accuracy") is not None
    ]
    return {
        "dataset": "dataset_avg",
        "n": len(values),
        "total": len(values),
        "evaluable": None,
        "correct": None,
        "accuracy": sum(values) / len(values) if values else None,
        "invalid_pred": None,
        "invalid": None,
        "unlabeled": None,
        "summary_kind": "macro_avg",
    }


def _aggregate_mcq_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    by_case: dict[str, list[CaseResult]] = {}
    by_case_dataset: dict[tuple[str, str], list[CaseResult]] = {}
    for result in results:
        by_case.setdefault(result.case_id, []).append(result)
        key = (result.case_id, result.sample.dataset)
        by_case_dataset.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for case_id in _case_order(results):
        datasets = _ordered_mcq_datasets(
            dataset for (case, dataset) in by_case_dataset if case == case_id
        )
        dataset_rows = []
        for dataset in datasets:
            row = _mcq_summary_row(dataset, by_case_dataset[(case_id, dataset)])
            row["case_id"] = case_id
            row["summary_kind"] = "dataset"
            dataset_rows.append(row)

        overall = _mcq_summary_row("overall", by_case[case_id])
        overall["case_id"] = case_id
        overall["summary_kind"] = "overall"
        macro = _mcq_macro_accuracy_row(dataset_rows)
        macro["case_id"] = case_id
        rows.extend([overall, macro, *dataset_rows])
    return rows


def _label_macro_f1(items: list[CaseResult], labels: tuple[str, ...] = ("yes", "no", "maybe")) -> float | None:
    golds: list[str] = []
    preds: list[str] = []
    for item in items:
        evaluation = item.evaluation
        gold = evaluation.get("pubmedqa_label_gold")
        pred = evaluation.get("pubmedqa_label_pred")
        if gold is None:
            gold_answers = [str(answer).strip().lower() for answer in item.sample.answers if str(answer or "").strip()]
            gold = gold_answers[0] if gold_answers else None
        if pred is None:
            pred = evaluation.get("predicted_label") or item.prediction
        gold = str(gold or "").strip().lower()
        pred = str(pred or "").strip().lower()
        if not gold:
            continue
        golds.append(gold)
        preds.append(pred)

    if not golds:
        return None

    scores: list[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _pubmedqa_label_accuracy(items: list[CaseResult]) -> float | None:
    values = []
    for item in items:
        value = item.evaluation.get("pubmedqa_label_correct")
        if value is None:
            value = item.evaluation.get("correct")
        if value is not None:
            values.append(float(bool(value)))
    return sum(values) / len(values) if values else None


def _extract_pmids(value: Any) -> set[str]:
    pmids: set[str] = set()
    if value is None:
        return pmids
    text = str(value)
    for match in re.finditer(r"(?:PMID[:\s]*)?(\d{6,9})", text, flags=re.IGNORECASE):
        pmids.add(match.group(1))
    return pmids


def _gold_alias_groups(result: CaseResult) -> list[set[str]]:
    raw = result.sample.raw or {}
    groups: list[set[str]] = []
    for gold_id in raw.get("gold_corpus_ids") or []:
        gold_id = str(gold_id)
        aliases = {gold_id}
        for pmid in _extract_pmids(gold_id):
            aliases.update({pmid, f"PMID:{pmid}", f"pubmed:{pmid}"})
        groups.append(aliases)

    metadata = raw.get("metadata") or {}
    pmid_values = [
        metadata.get("pubid"),
        metadata.get("pmid"),
        metadata.get("source_qid"),
        metadata.get("original_qid"),
        raw.get("id"),
        result.sample.id,
    ]
    metadata_aliases: set[str] = set()
    for value in pmid_values:
        for pmid in _extract_pmids(value):
            metadata_aliases.update({pmid, f"PMID:{pmid}", f"pubmed:{pmid}"})

    if metadata_aliases:
        if groups:
            groups[0].update(metadata_aliases)
        else:
            groups.append(metadata_aliases)
    return groups


def _doc_aliases(doc: RetrievedDocument) -> set[str]:
    aliases = {str(value) for value in [doc.stable_id, doc.corpus_id, doc.chunk_id, doc.db_id, doc.doc_id] if value}
    metadata = doc.metadata or {}
    for key in (
        "corpus_id",
        "chunk_id",
        "db_id",
        "doc_id",
        "source_doc_id",
        "source_chunk_id",
        "pmid",
        "pubmed_id",
        "id",
    ):
        value = metadata.get(key)
        if value:
            aliases.add(str(value))
    for alias in list(aliases):
        for pmid in _extract_pmids(alias):
            aliases.update({pmid, f"PMID:{pmid}", f"pubmed:{pmid}"})
    return aliases


def _retrieval_metrics(items: list[CaseResult]) -> dict[str, float | None]:
    rows_with_gold = 0
    rows_with_docs = 0
    hit_count = 0
    recall_sum = 0.0
    for item in items:
        groups = _gold_alias_groups(item)
        if not groups:
            continue
        docs = item.final_documents or item.initial_documents
        if not docs:
            continue
        rows_with_gold += 1
        rows_with_docs += 1
        doc_aliases: set[str] = set()
        for doc in docs:
            doc_aliases.update(_doc_aliases(doc))
        matched = [group for group in groups if group & doc_aliases]
        if matched:
            hit_count += 1
        recall_sum += len(matched) / len(groups)

    if rows_with_gold == 0 or rows_with_docs == 0:
        return {"retr_hit": None, "gold_hit": None, "gold_recall": None}
    return {
        "retr_hit": hit_count / rows_with_gold,
        "gold_hit": hit_count / rows_with_gold,
        "gold_recall": recall_sum / rows_with_gold,
    }


def _pretty_table_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    if _is_mcq_results(results):
        return _mcq_pretty_table_rows(results)

    by_dataset: dict[str, list[CaseResult]] = {}
    for result in results:
        by_dataset.setdefault(result.sample.dataset, []).append(result)

    preferred_order = ["bioasq", "covidqa", "mashqa", "pubmedqa"]
    ordered_datasets = [name for name in preferred_order if name in by_dataset]
    ordered_datasets.extend(sorted(name for name in by_dataset if name not in set(ordered_datasets)))

    rows: list[dict[str, Any]] = []
    non_pubmedqa = [item for item in results if item.sample.dataset != "pubmedqa"]
    if non_pubmedqa:
        row = _summary_row("__overall__", non_pubmedqa, include_extra=True)
        rows.append(row)

    for dataset in ordered_datasets:
        items = by_dataset[dataset]
        rows.append(_summary_row(dataset, items, include_extra=dataset != "pubmedqa"))
    return rows


def _mcq_summary_row(dataset: str, items: list[CaseResult]) -> dict[str, Any]:
    evaluable_items = [item for item in items if bool(item.evaluation.get("evaluable"))]
    evaluable = len(evaluable_items)
    correct = sum(1 for item in evaluable_items if bool(item.evaluation.get("correct")))
    invalid = sum(1 for item in items if item.evaluation.get("predicted_choice") is None)
    return {
        "dataset": dataset,
        "n": len(items),
        "total": len(items),
        "evaluable": evaluable,
        "correct": correct,
        "accuracy": correct / evaluable if evaluable else None,
        "invalid_pred": invalid,
        "invalid": invalid,
        "unlabeled": len(items) - evaluable,
        "summary_kind": "dataset",
    }


def _mcq_pretty_table_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[CaseResult]] = {}
    for result in results:
        by_dataset.setdefault(result.sample.dataset, []).append(result)

    ordered = _ordered_mcq_datasets(set(by_dataset))
    dataset_rows = [_mcq_summary_row(dataset, by_dataset[dataset]) for dataset in ordered]
    overall = _mcq_summary_row("overall", results)
    overall["summary_kind"] = "overall"
    macro = _mcq_macro_accuracy_row(dataset_rows)
    return [overall, macro, *dataset_rows]


def _summary_row(dataset: str, items: list[CaseResult], include_extra: bool) -> dict[str, Any]:
    if dataset == "pubmedqa":
        em = _pubmedqa_label_accuracy(items)
        f1 = _label_macro_f1(items)
    else:
        em = _mean_optional(items, "em")
        f1 = _mean_optional(items, "f1")
    retrieval = _retrieval_metrics(items)
    return {
        "dataset": dataset,
        "n": len(items),
        "em": em,
        "f1": f1,
        "bert_f1": _mean_optional(items, "bertscore_f1") if include_extra else None,
        "bleurt": _mean_optional(items, "bleurt") if include_extra else None,
        "prom_0to1": _mean_optional(items, "prometheus2_0to1") if include_extra else None,
        "retr_hit": retrieval["retr_hit"],
        "gold_hit": retrieval["gold_hit"],
        "gold_recall": retrieval["gold_recall"],
    }


def _format_pretty_table(rows: list[dict[str, Any]]) -> str:
    if rows and "accuracy" in rows[0] and "evaluable" in rows[0]:
        columns = [
            ("dataset", "dataset", 28, "left"),
            ("n", "n", 8, "right"),
            ("evaluable", "evaluable", 10, "right"),
            ("correct", "correct", 8, "right"),
            ("accuracy", "accuracy", 10, "right"),
            ("invalid_pred", "invalid", 8, "right"),
            ("unlabeled", "unlabeled", 10, "right"),
        ]
    else:
        columns = [
        ("dataset", "dataset", 16, "left"),
        ("n", "n", 8, "right"),
        ("em", "em", 8, "right"),
        ("f1", "f1", 8, "right"),
        ("bert_f1", "bert_f1", 10, "right"),
        ("bleurt", "bleurt", 8, "right"),
        ("prom_0to1", "prom_0to1", 10, "right"),
        ("retr_hit", "retr_hit", 10, "right"),
        ("gold_hit", "gold_hit", 10, "right"),
        ("gold_recall", "gold_recall", 12, "right"),
        ]

    formatted_rows: list[list[str]] = []
    for row in rows:
        values = []
        for key, _header, _width, _align in columns:
            value = row.get(key)
            if key in {"n", "evaluable", "correct", "invalid_pred", "unlabeled"}:
                values.append(_fmt_count(value))
            elif key == "dataset":
                values.append(str(value))
            else:
                values.append(_fmt_optional(value))
        formatted_rows.append(values)

    widths = []
    for idx, (_key, header, min_width, _align) in enumerate(columns):
        widths.append(max(min_width, len(header), *(len(row[idx]) for row in formatted_rows)))

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(values: list[str]) -> str:
        cells = []
        for value, width, (_key, _header, _min_width, align) in zip(values, widths, columns):
            padded = value.ljust(width) if align == "left" else value.rjust(width)
            cells.append(f" {padded} ")
        return "|" + "|".join(cells) + "|"

    output = [border(), line([header for _key, header, _width, _align in columns]), border()]
    for idx, row in enumerate(formatted_rows):
        if idx > 0 and rows[idx - 1].get("summary_kind") in {"overall", "macro_avg"} and rows[idx].get("summary_kind") == "dataset":
            output.append(border())
        output.append(line(row))
    output.append(border())
    return "\n".join(output) + "\n"


def write_pretty_summary_table(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_pretty_table(_pretty_table_rows(results)), encoding="utf-8")


def write_markdown_report(path: Path, results: list[CaseResult], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Basic RAG Evaluation Report")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for key, value in config.items():
        lines.append(f"| `{_escape_md(key)}` | `{_escape_md(value)}` |")
    lines.append("")
    lines.append("## Aggregate Summary")
    lines.append("")
    if _is_mcq_results(results):
        lines.append("| Case | Dataset | N | Evaluable | Correct | Accuracy | Invalid Pred | Unlabeled |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in _aggregate_rows(results):
            lines.append(
                f"| {_escape_md(row['case_id'])} | {_escape_md(row['dataset'])} | {_fmt_count(row['total'])} | "
                f"{_fmt_count(row['evaluable'])} | {_fmt_count(row['correct'])} | {_fmt_optional(row['accuracy'])} | "
                f"{_fmt_count(row['invalid'])} | {_fmt_count(row['unlabeled'])} |"
            )
    else:
        lines.append("| Case | Dataset | N | Correct | Accuracy | EM | F1 | BERTScore F1 | BLEURT | Prometheus 0-1 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in _aggregate_rows(results):
            lines.append(
                f"| {_escape_md(row['case_id'])} | {_escape_md(row['dataset'])} | {row['total']} | "
                f"{row['correct']} | {_fmt_optional(row['accuracy'])} | {_fmt_optional(row['em'])} | {_fmt_optional(row['f1'])} | "
                f"{_fmt_optional(row['bertscore_f1'])} | {_fmt_optional(row['bleurt'])} | "
                f"{_fmt_optional(row['prometheus2_0to1'])} |"
            )
    lines.append("")
    lines.append("## Sample Summary")
    lines.append("")
    lines.append("| # | Case | Dataset | Sample ID | Gold | Prediction | Correct | EM | F1 | BERTScore F1 | BLEURT | Prometheus | Initial IDs | Final IDs |")
    lines.append("|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for idx, result in enumerate(results, start=1):
        gold = result.sample.answers or ([result.sample.answer] if result.sample.answer else [])
        em = result.evaluation.get("em")
        f1 = result.evaluation.get("f1")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    _escape_md(result.case_id),
                    _escape_md(result.sample.dataset),
                    _escape_md(result.sample.id),
                    _escape_md(", ".join(gold)),
                    _escape_md(_short(result.prediction, 160)),
                    "-" if result.evaluation.get("correct") is None else str(bool(result.evaluation.get("correct"))),
                    _fmt_optional(em),
                    _fmt_optional(f1),
                    _fmt_optional(result.evaluation.get("bertscore_f1")),
                    _fmt_optional(result.evaluation.get("bleurt")),
                    _fmt_optional(result.evaluation.get("prometheus2_0to1")),
                    _doc_ids(result.initial_documents, limit=10),
                    _doc_ids(result.final_documents, limit=None, prefer_rerank=True),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Retrieved Document IDs")
    for idx, result in enumerate(results, start=1):
        if not result.initial_documents and not result.final_documents:
            continue
        lines.append("")
        lines.append(f"### {idx}. {result.case_id} / {result.sample.id}")
        lines.append("")
        lines.append("Initial retrieval:")
        lines.append("")
        lines.append("| Rank | Source | ID | Score |")
        lines.append("|---:|---|---|---:|")
        for doc in result.initial_documents:
            lines.append(
                f"| {doc.retrieval_rank or ''} | {_escape_md(doc.source)} | "
                f"`{_escape_md(doc.stable_id)}` | {doc.retrieval_score:.6f} |"
            )
        lines.append("")
        lines.append("Final context:")
        lines.append("")
        lines.append("| Rank | Initial Rank | Source | ID | Retrieval Score | Rerank Score |")
        lines.append("|---:|---:|---|---|---:|---:|")
        for doc in result.final_documents:
            rerank_score = "" if doc.rerank_score is None else f"{doc.rerank_score:.6f}"
            lines.append(
                f"| {doc.rerank_rank or doc.retrieval_rank or ''} | {doc.retrieval_rank or ''} | "
                f"{_escape_md(doc.source)} | `{_escape_md(doc.stable_id)}` | "
                f"{doc.retrieval_score:.6f} | {rerank_score} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text_report(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["#", "case", "dataset", "sample_id", "correct", "em", "f1", "bertscore", "bleurt", "prom", "gold", "prediction"]
    widths = [4, 16, 24, 34, 8, 8, 8, 10, 8, 8, 24, 72]
    lines = []
    lines.append("Aggregate summary")
    if _is_mcq_results(results):
        lines.append(
            "case".ljust(16)
            + "dataset".ljust(28)
            + "n".rjust(8)
            + "evaluable".rjust(12)
            + "correct".rjust(10)
            + "accuracy".rjust(12)
            + "invalid".rjust(10)
            + "unlabeled".rjust(12)
        )
        lines.append("-" * 108)
        for row in _aggregate_rows(results):
            lines.append(
                row["case_id"].ljust(16)
                + row["dataset"].ljust(28)
                + _fmt_count(row["total"]).rjust(8)
                + _fmt_count(row["evaluable"]).rjust(12)
                + _fmt_count(row["correct"]).rjust(10)
                + _fmt_optional(row["accuracy"]).rjust(12)
                + _fmt_count(row["invalid"]).rjust(10)
                + _fmt_count(row["unlabeled"]).rjust(12)
            )
    else:
        lines.append(
            "case".ljust(16)
            + "dataset".ljust(24)
            + "n".rjust(8)
            + "correct".rjust(10)
            + "accuracy".rjust(12)
            + "em".rjust(10)
            + "f1".rjust(10)
            + "bertscore".rjust(12)
            + "bleurt".rjust(10)
            + "prom0-1".rjust(10)
        )
        lines.append("-" * 122)
        for row in _aggregate_rows(results):
            lines.append(
                row["case_id"].ljust(16)
                + row["dataset"].ljust(24)
                + str(row["total"]).rjust(8)
                + str(row["correct"]).rjust(10)
                + _fmt_optional(row["accuracy"]).rjust(12)
                + _fmt_optional(row["em"]).rjust(10)
                + _fmt_optional(row["f1"]).rjust(10)
                + _fmt_optional(row["bertscore_f1"]).rjust(12)
                + _fmt_optional(row["bleurt"]).rjust(10)
                + _fmt_optional(row["prometheus2_0to1"]).rjust(10)
            )
    lines.append("")
    lines.append("Sample summary")
    lines.append(" ".join(header.ljust(width) for header, width in zip(headers, widths)))
    lines.append(" ".join("-" * width for width in widths))
    for idx, result in enumerate(results, start=1):
        gold = ", ".join(result.sample.answers or ([result.sample.answer] if result.sample.answer else []))
        em = result.evaluation.get("em")
        f1 = result.evaluation.get("f1")
        values = [
            str(idx),
            result.case_id,
            result.sample.dataset,
            result.sample.id,
            "-" if result.evaluation.get("correct") is None else str(bool(result.evaluation.get("correct"))),
            _fmt_optional(em),
            _fmt_optional(f1),
            _fmt_optional(result.evaluation.get("bertscore_f1")),
            _fmt_optional(result.evaluation.get("bleurt")),
            _fmt_optional(result.evaluation.get("prometheus2_0to1")),
            _short(gold, widths[10]),
            _short(result.prediction, widths[11]),
        ]
        lines.append(" ".join(value.ljust(width)[:width] for value, width in zip(values, widths)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

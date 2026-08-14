from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import BenchmarkSample
from .io_utils import iter_jsonl


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _options_from_row(row: dict[str, Any]) -> dict[str, str] | None:
    options = row.get("options")
    if isinstance(options, dict):
        return {str(key): _clean_text(value) for key, value in options.items()}
    choices = row.get("choices")
    if isinstance(choices, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {labels[idx]: _clean_text(choice) for idx, choice in enumerate(choices)}
    return None


def _answers_from_row(row: dict[str, Any]) -> list[str]:
    answers = row.get("answers")
    if isinstance(answers, list):
        return [str(answer) for answer in answers]
    answer = row.get("answer")
    return [str(answer)] if answer is not None else []


def load_benchmark_samples(
    path: Path,
    task: str,
    collection: str,
    dataset: str,
    split: str,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for row_idx, row in enumerate(iter_jsonl(path, limit=limit)):
        samples.append(
            BenchmarkSample(
                row_idx=row_idx,
                id=str(row.get("id") or f"{dataset}:{split}:{row_idx:06d}"),
                task=task,
                collection=collection,
                dataset=str(row.get("dataset") or dataset),
                split=str(row.get("split") or split),
                question=_clean_text(row.get("question")),
                options=_options_from_row(row),
                answer=str(row["answer"]) if row.get("answer") is not None else None,
                answers=_answers_from_row(row),
                raw=row,
            )
        )
    return samples


def resolve_benchmark_path(
    benchmark_root: Path,
    task: str,
    collection: str,
    dataset: str,
    split: str,
) -> Path:
    if task == "mcq":
        return benchmark_root / "mcq" / collection / dataset / f"{split}.jsonl"
    if task == "open_ended":
        return benchmark_root / "open_ended" / collection / f"{dataset}.jsonl"
    raise ValueError(f"Unsupported task: {task}")


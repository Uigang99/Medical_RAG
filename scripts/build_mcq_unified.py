from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "mcq" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "mcq" / "unified"

OPTION_LABELS = ["A", "B", "C", "D"]
MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]
MMLU_DATASET_PREFIX = "mmlu"
SPLIT_ORDER = ["train", "dev", "validation", "test"]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def answer_from_index(index: int | None) -> str | None:
    if index is None or index < 0 or index >= len(OPTION_LABELS):
        return None
    return OPTION_LABELS[index]


def ordered_choices(options: dict[str, str]) -> list[str]:
    return [options[label] for label in OPTION_LABELS]


def answer_text(options: dict[str, str], answer: str | None) -> str | None:
    if answer is None:
        return None
    return options.get(answer)


def build_record(
    *,
    qid: str,
    dataset: str,
    source_dataset: str,
    split: str,
    subject: str | None,
    question: str,
    options: dict[str, str],
    answer: str | None,
    source_file: Path,
    source_index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    text_answer = answer_text(options, answer)
    return {
        "id": qid,
        "dataset": dataset,
        "split": split,
        "subject": subject,
        "question": question,
        "options": options,
        "choices": ordered_choices(options),
        "answer": answer,
        "answers": [] if answer is None else [answer],
        "answer_text": text_answer,
        "gold_corpus_ids": [],
        "metadata": {
            "source_dataset": source_dataset,
            "source_file": rel(source_file),
            "source_index": source_index,
            "has_gold_answer": answer is not None,
            "question_type": "multiple_choice",
            **metadata,
        },
    }


def iter_medqa(raw_dir: Path) -> Iterator[dict[str, Any]]:
    specs = [
        ("train", raw_dir / "medqa" / "phrases_no_exclude_train.jsonl"),
        ("test", raw_dir / "medqa" / "phrases_no_exclude_test.jsonl"),
    ]
    for split, path in specs:
        for idx, row in enumerate(read_jsonl(path)):
            options = {label: str(row["options"][label]) for label in OPTION_LABELS}
            answer = row.get("answer_idx")
            if answer not in OPTION_LABELS:
                answer = None
            yield build_record(
                qid=f"medqa:{split}:{idx:06d}",
                dataset="medqa",
                source_dataset="medqa",
                split=split,
                subject=row.get("meta_info"),
                question=row["question"],
                options=options,
                answer=answer,
                source_file=path,
                source_index=idx,
                metadata={
                    "source_qid": None,
                    "exam_step": row.get("meta_info"),
                    "raw_answer": row.get("answer"),
                    "metamap_phrases": row.get("metamap_phrases", []),
                },
            )


def iter_medmcqa(raw_dir: Path) -> Iterator[dict[str, Any]]:
    specs = [
        ("train", raw_dir / "medmcqa" / "data" / "train-00000-of-00001.jsonl"),
        ("validation", raw_dir / "medmcqa" / "data" / "validation-00000-of-00001.jsonl"),
        ("test", raw_dir / "medmcqa" / "data" / "test-00000-of-00001.jsonl"),
    ]
    for split, path in specs:
        for idx, row in enumerate(read_jsonl(path)):
            options = {
                "A": row.get("opa") or "",
                "B": row.get("opb") or "",
                "C": row.get("opc") or "",
                "D": row.get("opd") or "",
            }
            answer = answer_from_index(row.get("cop"))
            source_id = row.get("id") or f"{split}:{idx:06d}"
            yield build_record(
                qid=f"medmcqa:{source_id}",
                dataset="medmcqa",
                source_dataset="medmcqa",
                split=split,
                subject=row.get("subject_name"),
                question=row["question"],
                options=options,
                answer=answer,
                source_file=path,
                source_index=idx,
                metadata={
                    "source_qid": row.get("id"),
                    "choice_type": row.get("choice_type"),
                    "topic": row.get("topic_name"),
                    "explanation": row.get("exp"),
                    "raw_cop": row.get("cop"),
                },
            )


def mmlu_dataset_name(subject: str) -> str:
    return f"{MMLU_DATASET_PREFIX}_{subject}"


def iter_mmlu_subject(raw_dir: Path, subject: str) -> Iterator[dict[str, Any]]:
    dataset = mmlu_dataset_name(subject)
    for split in ["dev", "validation", "test"]:
        path = raw_dir / "mmlu_medical" / subject / f"{split}-00000-of-00001.jsonl"
        for idx, row in enumerate(read_jsonl(path)):
            options = {
                label: str(choice)
                for label, choice in zip(OPTION_LABELS, row["choices"], strict=True)
            }
            answer = answer_from_index(row.get("answer"))
            yield build_record(
                qid=f"{dataset}:{split}:{idx:06d}",
                dataset=dataset,
                source_dataset="cais/mmlu",
                split=split,
                subject=subject,
                question=row["question"],
                options=options,
                answer=answer,
                source_file=path,
                source_index=idx,
                metadata={
                    "source_qid": None,
                    "mmlu_subject": row.get("subject", subject),
                    "raw_answer": row.get("answer"),
                },
            )


def summarize(path: Path) -> dict[str, Any]:
    rows = 0
    labeled = 0
    splits: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    for row in read_jsonl(path):
        rows += 1
        if row["answer"] is not None:
            labeled += 1
        splits[row["split"]] += 1
        if row.get("subject") is not None:
            subjects[str(row["subject"])] += 1
    return {
        "path": rel(path),
        "rows": rows,
        "labeled_rows": labeled,
        "unlabeled_rows": rows - labeled,
        "splits": dict(sorted(splits.items(), key=lambda item: SPLIT_ORDER.index(item[0]))),
        "subjects": dict(sorted(subjects.items())),
    }


def build_unified(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stale_mmlu_merged = output_dir / "mmlu_medical.jsonl"
    if stale_mmlu_merged.exists():
        stale_mmlu_merged.unlink()

    dataset_builders: list[tuple[str, Iterable[dict[str, Any]]]] = [
        ("medqa", iter_medqa(raw_dir)),
        ("medmcqa", iter_medmcqa(raw_dir)),
        *[
            (mmlu_dataset_name(subject), iter_mmlu_subject(raw_dir, subject))
            for subject in MMLU_MEDICAL_SUBJECTS
        ],
    ]

    dataset_summaries: list[dict[str, Any]] = []
    for dataset, rows in dataset_builders:
        output_path = output_dir / f"{dataset}.jsonl"
        write_jsonl(output_path, rows)
        dataset_summaries.append({"dataset": dataset, **summarize(output_path)})

    all_path = output_dir / "all.jsonl"
    def all_rows() -> Iterator[dict[str, Any]]:
        for dataset, _ in dataset_builders:
            yield from read_jsonl(output_dir / f"{dataset}.jsonl")

    write_jsonl(all_path, all_rows())
    all_summary = summarize(all_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": rel(raw_dir),
        "output_dir": rel(output_dir),
        "schema": {
            "id": "stable unified id, namespaced by dataset",
            "dataset": "medqa | medmcqa | mmlu_<subject>",
            "split": "train | dev | validation | test",
            "subject": "dataset-specific subject or exam step",
            "question": "question stem",
            "options": "dict of A/B/C/D option text",
            "choices": "A/B/C/D option text as an ordered list",
            "answer": "gold option label A/B/C/D, or null when unavailable",
            "answers": "list containing answer when available, otherwise empty",
            "answer_text": "gold option text, or null when unavailable",
            "gold_corpus_ids": "empty for MCQ benchmarks without gold retrieval corpus",
            "metadata": "source provenance and dataset-specific fields",
        },
        "datasets": dataset_summaries,
        "merged": all_summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified MCQ benchmark JSONL files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_unified(args.raw_dir, args.output_dir)
    for dataset in manifest["datasets"]:
        print(
            f"{dataset['dataset']}: {dataset['rows']} rows "
            f"({dataset['labeled_rows']} labeled, {dataset['unlabeled_rows']} unlabeled)",
            flush=True,
        )
    print(
        f"all: {manifest['merged']['rows']} rows -> {manifest['merged']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

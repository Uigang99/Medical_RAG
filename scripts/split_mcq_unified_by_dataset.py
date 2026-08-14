from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "mcq" / "unified"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "mcq" / "unified_split"
SPLIT_NAME_MAP = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
    "dev": "dev",
}
SPLIT_ORDER = ["train", "dev", "val", "test"]
SKIP_FILES = {"all.jsonl"}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_dataset_files(unified_dir: Path) -> Iterator[Path]:
    for path in sorted(unified_dir.glob("*.jsonl")):
        if path.name not in SKIP_FILES:
            yield path


def write_split_files(unified_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for source_path in iter_dataset_files(unified_dir):
        dataset = source_path.stem
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        handles: dict[str, Any] = {}
        split_counts: Counter[str] = Counter()
        labeled_counts: Counter[str] = Counter()
        unlabeled_counts: Counter[str] = Counter()

        try:
            for row in read_jsonl(source_path):
                raw_split = row.get("split")
                split = SPLIT_NAME_MAP.get(raw_split)
                if split is None:
                    raise ValueError(
                        f"Unsupported split {raw_split!r} in {source_path}"
                    )

                if split not in handles:
                    handles[split] = (dataset_dir / f"{split}.jsonl").open(
                        "w", encoding="utf-8"
                    )

                handles[split].write(json.dumps(row, ensure_ascii=False))
                handles[split].write("\n")
                split_counts[split] += 1
                if row.get("answer") is None:
                    unlabeled_counts[split] += 1
                else:
                    labeled_counts[split] += 1
        finally:
            for handle in handles.values():
                handle.close()

        split_files = {
            split: {
                "path": rel(dataset_dir / f"{split}.jsonl"),
                "rows": split_counts[split],
                "labeled_rows": labeled_counts[split],
                "unlabeled_rows": unlabeled_counts[split],
            }
            for split in sorted(split_counts, key=SPLIT_ORDER.index)
        }
        summaries.append(
            {
                "dataset": dataset,
                "source_path": rel(source_path),
                "output_dir": rel(dataset_dir),
                "rows": sum(split_counts.values()),
                "splits": split_files,
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": rel(unified_dir),
        "output_dir": rel(output_dir),
        "split_name_map": SPLIT_NAME_MAP,
        "datasets": summaries,
    }


def build_global_split_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    global_dir = output_dir / "all"
    global_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[Path]] = defaultdict(list)
    for dataset in manifest["datasets"]:
        for split, info in dataset["splits"].items():
            by_split[split].append(PROJECT_ROOT / info["path"])

    manifest["merged"] = {"output_dir": rel(global_dir), "splits": {}}
    for split in sorted(by_split, key=SPLIT_ORDER.index):
        output_path = global_dir / f"{split}.jsonl"
        rows = 0
        labeled = 0
        with output_path.open("w", encoding="utf-8") as out:
            for input_path in by_split[split]:
                for row in read_jsonl(input_path):
                    out.write(json.dumps(row, ensure_ascii=False))
                    out.write("\n")
                    rows += 1
                    if row.get("answer") is not None:
                        labeled += 1
        manifest["merged"]["splits"][split] = {
            "path": rel(output_path),
            "rows": rows,
            "labeled_rows": labeled,
            "unlabeled_rows": rows - labeled,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split flat unified MCQ JSONL files into per-dataset split files."
    )
    parser.add_argument("--unified-dir", type=Path, default=DEFAULT_UNIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--with-merged-splits",
        action="store_true",
        help="Also write all/{train,dev,val,test}.jsonl merged split files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unified_dir = args.unified_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = write_split_files(unified_dir, output_dir)
    if args.with_merged_splits:
        build_global_split_files(output_dir, manifest)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for dataset in manifest["datasets"]:
        split_summary = ", ".join(
            f"{split}={info['rows']}"
            for split, info in dataset["splits"].items()
        )
        print(f"{dataset['dataset']}: {split_summary}", flush=True)


if __name__ == "__main__":
    main()

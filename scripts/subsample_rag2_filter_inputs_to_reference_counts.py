#!/usr/bin/env python3
from __future__ import annotations

"""Deterministically downsample filter inputs to another run's split sizes.

This is intended for a controlled PPL-versus-Codex RAG² document-filter
comparison.  It preserves the new dataset's class/source/rerank-rank mixture
while matching the previous PPL document-filter run's number of training,
validation, and test question-document pairs exactly.
"""

import argparse
import hashlib
import heapq
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm


SPLITS = ("train", "val", "test")
SAMPLING_VERSION = "rag2_filter_pair_count_matched_stratified_hash_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample RAG² filter JSONLs to reference split row counts without changing rows."
    )
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Full Codex Top-8 filter-input root produced by build_rag2_codex_semantic_filter_inputs.py.",
    )
    parser.add_argument(
        "--reference-count-root",
        type=Path,
        required=True,
        help="Existing PPL document-filter split root; only split row counts are read.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-fields",
        nargs="+",
        default=("target", "source", "doc_rank"),
        choices=("target", "source", "doc_rank"),
        help="Fields whose source-dataset proportions are preserved in each split.",
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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from error


def require_file(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def count_jsonl_rows(path: Path) -> int:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        return sum(1 for line in handle if line.strip())


def group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        value = str(row.get(field) or "")
        if not value:
            raise ValueError(f"Missing stratification field {field!r} for pair {row.get('pair_id')}")
        values.append(value)
    return tuple(values)


def allocate_quotas(group_counts: Counter[tuple[str, ...]], target_total: int) -> dict[tuple[str, ...], int]:
    source_total = sum(group_counts.values())
    if target_total > source_total:
        raise ValueError(f"Cannot sample {target_total} rows from only {source_total} source rows.")
    exact = {key: count * target_total / source_total for key, count in group_counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = target_total - sum(quotas.values())
    ordering = sorted(
        group_counts,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in ordering[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != target_total or any(quotas[key] > group_counts[key] for key in quotas):
        raise RuntimeError("Invalid largest-remainder stratum allocation.")
    return quotas


def deterministic_score(pair_id: str, seed: int) -> int:
    value = f"{seed}:{pair_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=16).digest(), byteorder="big")


def scan_source_split(path: Path, fields: tuple[str, ...]) -> tuple[int, Counter[tuple[str, ...]]]:
    count = 0
    groups: Counter[tuple[str, ...]] = Counter()
    seen: set[str] = set()
    for row in tqdm(iter_jsonl(path), desc=f"count:{path.parent.name}/{path.stem}", unit="pair"):
        pair_id = str(row.get("pair_id") or row.get("id") or "")
        if not pair_id:
            raise ValueError(f"Missing pair_id in {path}")
        if pair_id in seen:
            raise ValueError(f"Duplicate input pair_id in {path}: {pair_id}")
        seen.add(pair_id)
        groups[group_key(row, fields)] += 1
        count += 1
    return count, groups


def select_pair_ids(
    path: Path,
    fields: tuple[str, ...],
    quotas: dict[tuple[str, ...], int],
    seed: int,
) -> set[str]:
    """Select exactly each stratum quota using the lowest stable hash values."""

    heaps: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    for row in tqdm(iter_jsonl(path), desc=f"sample:{path.parent.name}/{path.stem}", unit="pair"):
        pair_id = str(row.get("pair_id") or row.get("id") or "")
        key = group_key(row, fields)
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue
        # Heap root is the largest original score because its stored score is negative.
        entry = (-deterministic_score(pair_id, seed), pair_id)
        heap = heaps[key]
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected: set[str] = set()
    for key, quota in quotas.items():
        heap = heaps[key]
        if len(heap) != quota:
            raise RuntimeError(f"Sampling shortfall in stratum {key}: {len(heap)} != {quota}")
        for _, pair_id in heap:
            if pair_id in selected:
                raise RuntimeError(f"Selected duplicate pair_id: {pair_id}")
            selected.add(pair_id)
    return selected


def write_selected_split(path: Path, output_path: Path, selected: set[str]) -> dict[str, Any]:
    counters: dict[str, Any] = {
        "rows": 0,
        "sample_ids": set(),
        "target": Counter(),
        "semantic_label": Counter(),
        "source": Counter(),
        "doc_rank": Counter(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with output_path.open("w", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for row in tqdm(iter_jsonl(path), desc=f"write:{path.parent.name}/{path.stem}", unit="pair"):
            pair_id = str(row.get("pair_id") or row.get("id") or "")
            if pair_id not in selected:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            found.add(pair_id)
            counters["rows"] += 1
            counters["sample_ids"].add(str(row.get("sample_id") or ""))
            counters["target"][str(row.get("target") or "unknown")] += 1
            counters["semantic_label"][str(row.get("codex_semantic_label") or "unknown")] += 1
            counters["source"][str(row.get("source") or "unknown")] += 1
            counters["doc_rank"][str(row.get("doc_rank") or "unknown")] += 1
    if found != selected:
        missing = selected - found
        raise RuntimeError(f"Failed to write {len(missing)} selected pairs; first={sorted(missing)[:1]}")
    return {
        "rows": counters["rows"],
        "sample_ids": len(counters["sample_ids"]),
        "target": dict(sorted(counters["target"].items())),
        "semantic_label": dict(sorted(counters["semantic_label"].items())),
        "source": dict(sorted(counters["source"].items())),
        "doc_rank": dict(sorted(counters["doc_rank"].items(), key=lambda item: int(item[0]))),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    fields = tuple(args.stratify_fields)
    if len(set(fields)) != len(fields):
        raise ValueError("--stratify-fields cannot contain duplicates")

    input_dir = (args.input_root / args.dataset).resolve()
    reference_dir = (args.reference_count_root / args.dataset).resolve()
    output_dir = args.output_root / args.dataset
    input_paths = {split: require_file(input_dir / f"{split}.jsonl", f"input {split} split") for split in SPLITS}
    reference_paths = {
        split: require_file(reference_dir / f"{split}.jsonl", f"reference {split} split") for split in SPLITS
    }
    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output dataset directory is not empty: {output_dir}")

    result: dict[str, Any] = {
        "type": "rag2_filter_count_matched_subsample",
        "sampling_version": SAMPLING_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "input_root": str(args.input_root.resolve()),
        "reference_count_root": str(args.reference_count_root.resolve()),
        "seed": args.seed,
        "stratify_fields": list(fields),
        "splits": {},
        "dry_run": bool(args.dry_run),
    }
    for split in SPLITS:
        source_rows, group_counts = scan_source_split(input_paths[split], fields)
        target_rows = count_jsonl_rows(reference_paths[split])
        quotas = allocate_quotas(group_counts, target_rows)
        selected = select_pair_ids(input_paths[split], fields, quotas, args.seed)
        if len(selected) != target_rows:
            raise RuntimeError(f"Selected {len(selected)} instead of {target_rows} rows for {split}")
        result["splits"][split] = {
            "source_rows": source_rows,
            "reference_target_rows": target_rows,
            "stratum_count": len(group_counts),
            "allocated_quotas": {"|".join(key): value for key, value in sorted(quotas.items())},
        }
        if not args.dry_run:
            result["splits"][split]["sampled"] = write_selected_split(
                input_paths[split], output_dir / f"{split}.jsonl", selected
            )
        logging.info(
            "[%s] sampled %s/%s Codex rows to reference PPL count",
            split,
            target_rows,
            source_rows,
        )
    if not args.dry_run:
        result["files"] = {split: str((output_dir / f"{split}.jsonl").resolve()) for split in SPLITS}
        (output_dir / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

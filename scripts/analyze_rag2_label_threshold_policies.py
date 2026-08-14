from __future__ import annotations

import argparse
import json
import math
import random
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
from tqdm import tqdm


LABEL_NAMES = ("Helpful", "Not Helpful", "Discard", "Excluded")
LABEL_TO_CODE = {label: index for index, label in enumerate(LABEL_NAMES)}
DATASET_NAMES = ("medmcqa", "medqa")
DATASET_TO_CODE = {dataset: index for index, dataset in enumerate(DATASET_NAMES)}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RAG2 pseudo-label counts under alternative delta-PPL threshold policies."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def assign_label(no_correct: bool, with_correct: bool, delta: float, tau: float) -> int:
    meets_threshold = math.isfinite(delta) and delta >= tau
    if no_correct and with_correct:
        return LABEL_TO_CODE["Helpful"] if meets_threshold else LABEL_TO_CODE["Discard"]
    if no_correct and not with_correct:
        return LABEL_TO_CODE["Not Helpful"]
    if not no_correct and with_correct:
        return LABEL_TO_CODE["Helpful"]
    if not no_correct and not with_correct:
        return LABEL_TO_CODE["Not Helpful"] if meets_threshold else LABEL_TO_CODE["Discard"]
    return LABEL_TO_CODE["Discard"]


def branch_name(no_correct: bool, with_correct: bool) -> str:
    return ("C" if no_correct else "W") + "->" + ("C" if with_correct else "W")


def nested_counts() -> dict[str, Counter[str]]:
    return {dataset: Counter() for dataset in DATASET_NAMES}


def serialize_counts(counts: Counter[str], total: int) -> dict[str, Any]:
    return {
        label: {
            "count": int(counts[label]),
            "percent": (100.0 * counts[label] / total) if total else 0.0,
        }
        for label in LABEL_NAMES[:3]
    }


def main() -> None:
    args = parse_args()
    decoder = msgspec.json.Decoder()

    dataset_codes = array("B")
    sample_indexes = array("I")
    correctness_flags = array("B")
    old_labels = array("B")
    delta_ra = array("d")
    delta_ro = array("d")

    sample_to_index: dict[str, int] = {}
    sample_ids: list[str] = []
    sample_dataset_codes = array("B")
    excluded_rows = 0
    scanned_rows = 0

    size = args.input_file.stat().st_size
    with args.input_file.open("rb", buffering=16 * 1024 * 1024) as handle:
        progress = tqdm(total=size, desc="Reading label rows", unit="B", unit_scale=True)
        for line in handle:
            progress.update(len(line))
            if not line.strip():
                continue
            row = decoder.decode(line)
            scanned_rows += 1
            dataset = str(row.get("dataset") or "")
            if dataset not in DATASET_TO_CODE:
                continue
            dataset_code = DATASET_TO_CODE[dataset]
            sample_id = str(row.get("sample_id") or "")
            sample_index = sample_to_index.get(sample_id)
            if sample_index is None:
                sample_index = len(sample_ids)
                sample_to_index[sample_id] = sample_index
                sample_ids.append(sample_id)
                sample_dataset_codes.append(dataset_code)

            if row.get("quality_pass") is not True:
                excluded_rows += 1
                continue

            no_correct = bool(row.get("no_doc_correct"))
            with_correct = bool(row.get("with_doc_correct"))
            old_label = str(row.get("pseudo_label") or "Discard")
            dataset_codes.append(dataset_code)
            sample_indexes.append(sample_index)
            correctness_flags.append((1 if no_correct else 0) | (2 if with_correct else 0))
            old_labels.append(LABEL_TO_CODE.get(old_label, LABEL_TO_CODE["Discard"]))
            delta_ra.append(finite_float(row.get("generated_output_delta_ppl_rationale_with_answer")))
            delta_ro.append(finite_float(row.get("generated_output_delta_ppl_rationale_only")))
        progress.close()

    dataset_np = np.frombuffer(dataset_codes, dtype=np.uint8)
    sample_np = np.frombuffer(sample_indexes, dtype=np.uint32)
    flags_np = np.frombuffer(correctness_flags, dtype=np.uint8)
    old_np = np.frombuffer(old_labels, dtype=np.uint8)
    delta_ra_np = np.frombuffer(delta_ra, dtype=np.float64)
    delta_ro_np = np.frombuffer(delta_ro, dtype=np.float64)
    same_outcome = ((flags_np & 1) > 0) == ((flags_np & 2) > 0)

    sample_split = np.full(len(sample_ids), 2, dtype=np.uint8)
    rng = random.Random(args.seed)
    for dataset in DATASET_NAMES:
        dataset_code = DATASET_TO_CODE[dataset]
        indexes = [
            index for index, code in enumerate(sample_dataset_codes) if int(code) == dataset_code
        ]
        indexes.sort(key=lambda index: sample_ids[index])
        rng.shuffle(indexes)
        n_train = int(round(len(indexes) * args.train_ratio))
        n_val = int(round(len(indexes) * args.val_ratio))
        if n_train + n_val > len(indexes):
            n_val = max(0, len(indexes) - n_train)
        sample_split[np.asarray(indexes[:n_train], dtype=np.int64)] = 0
        sample_split[np.asarray(indexes[n_train : n_train + n_val], dtype=np.int64)] = 1
    row_split = sample_split[sample_np]

    valid_ro = np.isfinite(delta_ro_np)
    valid_ra = np.isfinite(delta_ra_np)
    policies: dict[str, dict[str, Any]] = {}

    global_all_ro_tau = float(np.quantile(delta_ro_np[valid_ro], args.quantile))
    global_same_ro_tau = float(np.quantile(delta_ro_np[valid_ro & same_outcome], args.quantile))
    global_all_ra_tau = float(np.quantile(delta_ra_np[valid_ra], args.quantile))
    policies["legacy_recomputed"] = {
        "description": "Global all-pair rationale+answer delta-PPL quantile (legacy policy).",
        "tau": {"all": global_all_ra_tau},
        "delta": delta_ra_np,
        "tau_by_row": np.full(len(dataset_np), global_all_ra_tau, dtype=np.float64),
    }
    policies["paper_literal_global_all_rationale_only"] = {
        "description": "Global all-pair rationale-only delta-PPL quantile.",
        "tau": {"all": global_all_ro_tau},
        "delta": delta_ro_np,
        "tau_by_row": np.full(len(dataset_np), global_all_ro_tau, dtype=np.float64),
    }
    policies["global_same_outcome_rationale_only"] = {
        "description": "Global C->C/W->W rationale-only delta-PPL quantile.",
        "tau": {"all": global_same_ro_tau},
        "delta": delta_ro_np,
        "tau_by_row": np.full(len(dataset_np), global_same_ro_tau, dtype=np.float64),
    }

    train_all_tau_by_dataset: dict[str, float] = {}
    train_all_tau_by_row = np.empty(len(dataset_np), dtype=np.float64)
    tau_by_dataset: dict[str, float] = {}
    tau_by_row = np.empty(len(dataset_np), dtype=np.float64)
    for dataset in DATASET_NAMES:
        dataset_code = DATASET_TO_CODE[dataset]
        train_all_mask = (
            (dataset_np == dataset_code)
            & (row_split == 0)
            & valid_ro
        )
        train_all_tau = float(np.quantile(delta_ro_np[train_all_mask], args.quantile))
        train_all_tau_by_dataset[dataset] = train_all_tau
        train_all_tau_by_row[dataset_np == dataset_code] = train_all_tau

        mask = (
            (dataset_np == dataset_code)
            & (row_split == 0)
            & same_outcome
            & valid_ro
        )
        tau = float(np.quantile(delta_ro_np[mask], args.quantile))
        tau_by_dataset[dataset] = tau
        tau_by_row[dataset_np == dataset_code] = tau
    policies["paper_literal_train_all_per_dataset_rationale_only"] = {
        "description": "Per-dataset train-only all-pair rationale-only delta-PPL quantile.",
        "tau": train_all_tau_by_dataset,
        "delta": delta_ro_np,
        "tau_by_row": train_all_tau_by_row,
    }
    policies["recommended_train_same_outcome_per_dataset_rationale_only"] = {
        "description": "Per-dataset train-only C->C/W->W rationale-only delta-PPL quantile.",
        "tau": tau_by_dataset,
        "delta": delta_ro_np,
        "tau_by_row": tau_by_row,
    }

    report_policies: dict[str, Any] = {}
    old_counts = Counter(LABEL_NAMES[int(code)] for code in old_np)
    for policy_name, policy in policies.items():
        labels = np.empty(len(dataset_np), dtype=np.uint8)
        delta_values = policy["delta"]
        tau_values = policy["tau_by_row"]
        for index in range(len(labels)):
            flags = int(flags_np[index])
            labels[index] = assign_label(
                bool(flags & 1),
                bool(flags & 2),
                float(delta_values[index]),
                float(tau_values[index]),
            )

        counts = Counter(LABEL_NAMES[int(code)] for code in labels)
        per_dataset = nested_counts()
        per_branch: dict[str, Counter[str]] = defaultdict(Counter)
        transitions: Counter[str] = Counter()
        changed = 0
        for index, new_code in enumerate(labels):
            dataset = DATASET_NAMES[int(dataset_np[index])]
            label = LABEL_NAMES[int(new_code)]
            per_dataset[dataset][label] += 1
            flags = int(flags_np[index])
            per_branch[branch_name(bool(flags & 1), bool(flags & 2))][label] += 1
            old_label = LABEL_NAMES[int(old_np[index])]
            transitions[f"{old_label} -> {label}"] += 1
            changed += int(new_code != old_np[index])

        report_policies[policy_name] = {
            "description": policy["description"],
            "tau": policy["tau"],
            "counts": serialize_counts(counts, len(labels)),
            "per_dataset": {
                dataset: serialize_counts(per_dataset[dataset], int(np.sum(dataset_np == code)))
                for dataset, code in DATASET_TO_CODE.items()
            },
            "per_branch": {branch: dict(counter) for branch, counter in sorted(per_branch.items())},
            "changed_from_stored_legacy": {
                "count": changed,
                "percent": 100.0 * changed / len(labels),
            },
            "transitions_from_stored_legacy": dict(transitions),
        }

    report = {
        "input_file": str(args.input_file),
        "quantile": args.quantile,
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        },
        "rows": {
            "scanned": scanned_rows,
            "quality_valid": len(dataset_np),
            "excluded": excluded_rows,
            "sample_ids": len(sample_ids),
        },
        "stored_legacy_counts": serialize_counts(old_counts, len(old_np)),
        "policies": report_policies,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combine Direct-Choice and Rationale+Answer dynamic Top-k reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOP_K_VALUES = (1, 2, 4, 8, 16, 32)
CONDITIONS = ("no_rag", "base_rag", "pced_rerank", "pced_semantic")
LABELS = {
    "no_rag": "No-RAG",
    "base_rag": "Base-RAG",
    "pced_rerank": "PCED rerank prior",
    "pced_semantic": "PCED semantic prior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100*value:.2f}"


def main() -> None:
    args = parse_args()
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    missing: list[Path] = []
    for mode in ("direct_choice", "rationale_answer"):
        for top_k in TOP_K_VALUES:
            path = args.root / mode / f"top{top_k}" / "summary.json"
            if not path.is_file():
                missing.append(path)
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if int(value.get("top_k", -1)) != top_k:
                raise RuntimeError(f"Summary Top-k mismatch: {path}")
            summaries[(mode, top_k)] = value
    if missing:
        raise FileNotFoundError("Missing sweep summaries:\n" + "\n".join(map(str, missing)))

    lines = [
        "PCED dynamic Top-k comparison", "",
        "| Answer mode | k | Condition | Mean docs available | N | MedMCQA | MedQA | MMLU pooled | Micro | Macro-8 | Macro-3 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    machine_rows: list[dict[str, Any]] = []
    for mode in ("direct_choice", "rationale_answer"):
        for top_k in TOP_K_VALUES:
            summary = summaries[(mode, top_k)]
            for condition in CONDITIONS:
                metrics = summary["conditions"][condition]
                lines.append(
                    f"| {mode.replace('_', ' ')} | {top_k} | {LABELS[condition]} | "
                    f"{metrics.get('mean_documents_available', 0.0):.2f} | {metrics['questions']} | "
                    f"{pct(metrics['medmcqa_accuracy'])} | {pct(metrics['medqa_accuracy'])} | "
                    f"{pct(metrics['mmlu_pooled_accuracy'])} | {pct(metrics['micro_accuracy'])} | "
                    f"{pct(metrics['macro8_accuracy'])} | {pct(metrics['macro3_accuracy'])} |"
                )
                machine_rows.append({
                    "answer_mode": mode, "top_k": top_k, "condition": condition, **metrics,
                })
    lines.extend(["", "Micro-accuracy deltas on each identical Top-k cohort:", "",
                  "| Answer mode | k | PCED rerank − Base-RAG | PCED semantic − PCED rerank |",
                  "|---|---:|---:|---:|"])
    for mode in ("direct_choice", "rationale_answer"):
        for top_k in TOP_K_VALUES:
            comparisons = summaries[(mode, top_k)]["paired_comparisons"]
            rerank = comparisons["PCED rerank vs Base-RAG"]["accuracy_delta"]
            semantic = comparisons["PCED semantic vs PCED rerank"]["accuracy_delta"]
            lines.append(f"| {mode.replace('_', ' ')} | {top_k} | {100*rerank:+.2f}%p | {100*semantic:+.2f}%p |")
    table = "\n".join(lines) + "\n"
    output = args.root / "combined_summary_table.txt"
    output.write_text(table, encoding="utf-8")
    (args.root / "combined_summary.json").write_text(
        json.dumps({"top_k_values": list(TOP_K_VALUES), "rows": machine_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(table, flush=True)
    print(f"[combined report complete] {output}", flush=True)


if __name__ == "__main__":
    main()

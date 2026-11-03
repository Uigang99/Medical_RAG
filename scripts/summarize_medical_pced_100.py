#!/usr/bin/env python3
"""Audit and summarize matched medical PCED 100-question pilot outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_loft_pced_paper as paper
from scripts import run_medical_pced_100 as medical

ARMS = ("no-rag", "single", "all", "all-ranked", "pced")


def main():
    import numpy as np
    from scipy.stats import binomtest

    root = ROOT / "results/medical_pced_seeded100_v1"
    baselines = ROOT / "results/medical_pced_seeded100_v1_baselines"
    report = {"version": "medical_pced_seeded100_comparison_v1", "datasets": {},
              "n": 300, "primary_baseline": "all-ranked", "metric": "Subspan EM",
              "bootstrap": "paired within dataset, 20000 draws, seed 20261103",
              "method_note": "All is official seed-125 shuffle; All-ranked preserves PCED CE Top-90 order"}
    paired = []
    for dataset in medical.DATASETS:
        ref = json.loads((root / dataset / "manifest.json").read_text())
        selector = argparse.Namespace(dataset=dataset, fetch_k=ref["fetch_k"],
                                      sample_seed=ref["sample_seed"])
        _, _, _, questions = medical.selected_inputs(selector)
        qids = [str(q["qid"]) for q in questions]
        if qids != ref["sample_qids"] or len(qids) != 100:
            raise RuntimeError(f"Sample contract mismatch: {dataset}")
        predictions = {}
        for arm in ARMS:
            folder = (root / dataset if arm == "pced" else
                      baselines / dataset / arm) / "predictions"
            rows = [json.loads(paper.result_file(folder, qid).read_text()) for qid in qids]
            if any(str(row["qid"]) != qid or row["subspan_em"] not in (0, 1)
                   for row, qid in zip(rows, qids)):
                raise RuntimeError(f"Invalid predictions: {dataset}/{arm}")
            predictions[arm] = rows
        for qid, pced, ranked, shuffled, single in zip(
                qids, predictions["pced"], predictions["all-ranked"],
                predictions["all"], predictions["single"]):
            ids = pced["top90_pids"]
            if (ranked["document_pids"] != ids or len(ids) != 90 or
                    set(shuffled["document_pids"]) != set(ids) or
                    single["document_pids"] != ids[:1]):
                raise RuntimeError(f"Matched evidence contract failed: {dataset}/{qid}")
        correct = {arm: sum(row["subspan_em"] for row in rows)
                   for arm, rows in predictions.items()}
        pced = np.array([row["subspan_em"] for row in predictions["pced"]], dtype=int)
        ranked = np.array([row["subspan_em"] for row in predictions["all-ranked"]], dtype=int)
        diff = pced - ranked
        paired.append(diff)
        gain = int((diff == 1).sum())
        harm = int((diff == -1).sum())
        rng = np.random.default_rng(20261103)
        bootstrap = diff[rng.integers(0, 100, (20000, 100))].mean(axis=1) * 100
        report["datasets"][dataset] = {
            "n": 100, "missing": 0, "correct": correct,
            "subspan_em": {arm: correct[arm] / 100 for arm in ARMS},
            "pced_vs_all_ranked": {
                "delta_pp": float(diff.mean() * 100),
                "all_wrong_pced_correct": gain,
                "all_correct_pced_wrong": harm,
                "paired_bootstrap_ci95_pp": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                "mcnemar_exact_p": float(binomtest(min(gain, harm), gain + harm, 0.5).pvalue)
                    if gain + harm else 1.0},
            "reference_manifest_sha256": paper.sha(root / dataset / "manifest.json"),
        }
    stacked = np.stack(paired)
    rng = np.random.default_rng(20261103)
    bootstrap = np.stack([stacked[j][rng.integers(0, 100, (20000, 100))].mean(axis=1)
                          for j in range(3)], axis=1).mean(axis=1) * 100
    report["macro3_subspan_em"] = {
        arm: sum(report["datasets"][name]["subspan_em"][arm]
                 for name in medical.DATASETS) / 3 for arm in ARMS}
    report["micro_subspan_em"] = report["macro3_subspan_em"]  # exactly 100 per dataset
    report["pced_vs_all_ranked_macro3"] = {
        "delta_pp": float(stacked.mean() * 100),
        "paired_bootstrap_ci95_pp": np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    path = root / "comparison.json"
    paper.atomic_json(path, report)
    print(json.dumps({"path": str(path), "macro3": report["macro3_subspan_em"],
                      "pced_vs_all_ranked": report["pced_vs_all_ranked_macro3"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

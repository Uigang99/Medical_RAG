#!/usr/bin/env python3
"""Re-score relation-oracle outputs on the same frozen core ontology as v5.

The original oracle report counted auxiliary v5 relations as false-positive
core edges while its full-pipeline baseline excluded them.  This analysis keeps
the raw generation immutable and reports comparable validated and choice-only
metrics, plus output-contract failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_rag2_paired_raw_proposition_kg_pilot import atomic_json  # noqa: E402
from evaluate_rag2_statement_graph_relation_oracle import f1_counts  # noqa: E402


DEFAULT_ORACLE = ROOT / "datasets/kg/rag2_statement_graph_relation_oracle_1k_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(rows: Sequence[Mapping[str, Any]], *, choice_only: bool, shared: set[str]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        pair_by_id = {str(pair["pair_id"]): pair for pair in row["candidate_pairs"]}
        decision_by_id = {str(value["pair_id"]): value for value in row["pair_decisions"]}
        for pair_id, pair in pair_by_id.items():
            decision = decision_by_id.get(pair_id)
            gold = list(pair["gold_relations"])
            counts["pairs"] += 1
            counts["gold_edges"] += len(gold)
            counts["gold_positive_pairs"] += int(bool(gold))
            counts["negative_pairs"] += int(not gold)
            if not decision:
                counts["invalid"] += 1
                continue
            choice = str(decision.get("choice") or "")
            allowed = choice in pair["allowed_choices"]
            valid = allowed if choice_only else bool(decision.get("classification_valid"))
            counts["valid"] += int(valid)
            if not valid:
                counts["invalid"] += 1
                continue
            is_positive = choice not in {"NO_RELATION", "AMBIGUOUS"}
            relation_type = choice.split(":", 1)[1] if is_positive and ":" in choice else None
            shared_positive = bool(is_positive and relation_type in shared)
            counts["predicted_edges"] += int(shared_positive)
            counts["detection_tp"] += int(shared_positive and bool(gold))
            counts["exact_tp"] += int(shared_positive and choice in {value["choice"] for value in gold})
            counts["negative_no_relation"] += int(not gold and choice == "NO_RELATION")
            counts["ambiguous"] += int(choice == "AMBIGUOUS")
    relation_tpr = counts["detection_tp"] / max(counts["gold_positive_pairs"], 1)
    negative_tnr = counts["negative_no_relation"] / max(counts["negative_pairs"], 1)
    return {
        "pairs": counts["pairs"],
        "valid_output_rate": counts["valid"] / max(counts["pairs"], 1),
        "invalid_rate": counts["invalid"] / max(counts["pairs"], 1),
        "exact_core_relation": f1_counts(counts["exact_tp"], counts["predicted_edges"], counts["gold_edges"]),
        "core_relation_detection": f1_counts(
            counts["detection_tp"], counts["predicted_edges"], counts["gold_positive_pairs"]
        ),
        "hard_negative_no_relation_accuracy": negative_tnr,
        "binary_balanced_accuracy": (relation_tpr + negative_tnr) / 2,
        "ambiguous_rate": counts["ambiguous"] / max(counts["pairs"], 1),
    }


def main() -> None:
    args = parse_args()
    raw_path = args.oracle_root / "oracle_classifications.jsonl"
    summary_path = args.oracle_root / "summary.json"
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Completed relation oracle outputs are required")
    rows = read_jsonl(raw_path)
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    shared = {
        str(value["relation_type"])
        for row in rows
        for pair in row["candidate_pairs"]
        for value in pair["gold_relations"]
    }
    errors = Counter(
        error
        for row in rows
        for decision in row["pair_decisions"]
        if not decision.get("classification_valid")
        for error in decision.get("validation_errors", [])
    )
    parse_success = sum(bool(row.get("classification_parse_success")) for row in rows)
    validated = score(rows, choice_only=False, shared=shared)
    choice_only = score(rows, choice_only=True, shared=shared)
    baseline = original["full_pipeline_baseline"]
    result = {
        "contract": {
            "raw_generation_unchanged": True,
            "shared_core_relations": sorted(shared),
            "comparison_scope": "same 24 core relation types for full v5 baseline and oracle",
        },
        "counts": {
            "propositions": len(rows),
            "parse_success": parse_success,
            "parse_success_rate": parse_success / max(len(rows), 1),
            "validation_error_counts_overlapping": dict(errors.most_common()),
        },
        "full_pipeline_baseline": baseline,
        "gold_endpoint_validated": validated,
        "gold_endpoint_choice_only": choice_only,
        "delta_validated_f1_vs_full_pipeline": validated["exact_core_relation"]["f1"] - baseline["f1"],
        "delta_choice_only_f1_vs_validated": (
            choice_only["exact_core_relation"]["f1"] - validated["exact_core_relation"]["f1"]
        ),
        "decision": {
            "threshold_exact_relation_f1": 0.75,
            "validated_pass": validated["exact_core_relation"]["f1"] >= 0.75,
            "choice_only_pass": choice_only["exact_core_relation"]["f1"] >= 0.75,
        },
    }
    atomic_json(args.oracle_root / "posthoc_summary.json", result)
    b = baseline
    v = validated["exact_core_relation"]
    c = choice_only["exact_core_relation"]
    report = f"""# Comparable post-hoc relation-oracle analysis

Raw Qwen generations are unchanged. All conditions below use the same frozen 24-relation core ontology.

| Condition | Precision | Recall | F1 |
|---|---:|---:|---:|
| Full v5 pipeline | {100*b['precision']:.2f}% | {100*b['recall']:.2f}% | {100*b['f1']:.2f}% |
| Gold endpoints + full validation | {100*v['precision']:.2f}% | {100*v['recall']:.2f}% | {100*v['f1']:.2f}% |
| Gold endpoints + relation choice only | {100*c['precision']:.2f}% | {100*c['recall']:.2f}% | {100*c['f1']:.2f}% |

- JSON parse success: {parse_success}/{len(rows)} ({100*parse_success/max(len(rows),1):.2f}%)
- Pair output valid after evidence checks: {100*validated['valid_output_rate']:.2f}%
- Choice-only hard-negative NO_RELATION accuracy: {100*choice_only['hard_negative_no_relation_accuracy']:.2f}%
- Choice-only binary balanced accuracy: {100*choice_only['binary_balanced_accuracy']:.2f}%
- Prespecified exact relation F1 threshold: 75%; result: FAIL
"""
    tmp = args.oracle_root / "posthoc_report.md.tmp"
    tmp.write_text(report, encoding="utf-8")
    tmp.replace(args.oracle_root / "posthoc_report.md")
    print(report)


if __name__ == "__main__":
    main()

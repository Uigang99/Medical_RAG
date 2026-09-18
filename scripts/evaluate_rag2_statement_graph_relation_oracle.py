#!/usr/bin/env python3
"""Measure the current v5 relation classifier with gold entity endpoints.

The only experimental change from v5 is the endpoint source: entity spans,
types, and positive relation pairs come from the frozen source-only reference.
The Qwen model, prompt, relation signatures, decoding, parser, and evidence
validation are reused unchanged.  Type-compatible non-relation pairs from the
same propositions are retained as hard negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

import extract_rag2_statement_graph_pair_classifier_v5 as v5  # noqa: E402


RUN_VERSION = "rag2_statement_graph_relation_oracle_1k_v1"
DEFAULT_REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_relation_oracle_1k_v1"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"
DEFAULT_BASELINE = DEFAULT_REFERENCE / "automatic_kg_comparison/summary.json"
PROGRESS_PREFIX = "@@RELATION_ORACLE_PROGRESS@@"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "classify", "aggregate"), default="orchestrate")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(PROGRESS_PREFIX + v5.canonical_json({"event": event, **payload}), flush=True)


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def proportional_quotas(counts: Mapping[tuple[str, bool], int], target: int) -> dict[tuple[str, bool], int]:
    total = sum(counts.values())
    raw = {key: target * count / total for key, count in counts.items()}
    result = {key: int(value) for key, value in raw.items()}
    remaining = target - sum(result.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def frozen_splits(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    """Create exact 512/128/128 splits, stratified by source and relation presence."""
    strata: dict[tuple[str, bool], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["source"]), bool(row["relations"]))].append(row)
    counts = {key: len(values) for key, values in strata.items()}
    val_quota = proportional_quotas(counts, 128)
    test_quota = proportional_quotas(counts, 128)
    assignments: dict[str, str] = {}
    for key, values in strata.items():
        ordered = sorted(
            values,
            key=lambda row: v5.sha256_text(f"{seed}|{row['proposition_id']}"),
        )
        val_end = val_quota[key]
        test_end = val_end + test_quota[key]
        for row in ordered[:val_end]:
            assignments[str(row["proposition_id"])] = "validation"
        for row in ordered[val_end:test_end]:
            assignments[str(row["proposition_id"])] = "test"
        for row in ordered[test_end:]:
            assignments[str(row["proposition_id"])] = "development"
    actual = Counter(assignments.values())
    if actual != {"development": 512, "validation": 128, "test": 128}:
        raise RuntimeError(f"Unexpected frozen split sizes: {dict(actual)}")
    return assignments


def load_reference(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.reference_root / "reference_statements.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [dict(row) for row in v5.read_jsonl(path) if row.get("curation_confidence") == "high"]
    if len(rows) != 768:
        raise RuntimeError(f"Expected 768 high-confidence propositions; found {len(rows)}")
    identifiers = [str(row["proposition_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Duplicate proposition IDs in reference")
    return rows


def as_mention(entity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mention_id": entity["entity_id"],
        "surface": entity["surface"],
        "kind": entity["entity_type"],
        "start": entity["start"],
        "end": entity["end"],
        "umls_cui": None,
        "umls_name": None,
    }


def target_choice(relation: Mapping[str, Any], left_id: str) -> str:
    relation_type = str(relation["relation_type"])
    if relation_type in v5.SYMMETRIC_RELATIONS:
        return f"SYMMETRIC:{relation_type}"
    orientation = "L2R" if relation["subject_entity_id"] == left_id else "R2L"
    return f"{orientation}:{relation_type}"


def materialize_oracle_pairs(
    rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counts = Counter()
    relation_counts = Counter()
    unsupported_counts = Counter()
    for row in rows:
        entities = sorted(
            (dict(value) for value in row["entities"]),
            key=lambda value: (int(value["start"]), int(value["end"]), str(value["entity_id"])),
        )
        relation_by_pair: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
        for relation in row["relations"]:
            relation_by_pair[frozenset((relation["subject_entity_id"], relation["object_entity_id"]))].append(dict(relation))
            relation_counts[str(relation["relation_type"])] += 1
            counts["gold_relation_instances"] += 1
        pairs: list[dict[str, Any]] = []
        for left_index, left_entity in enumerate(entities):
            for right_entity in entities[left_index + 1 :]:
                pair_gold = relation_by_pair.get(
                    frozenset((str(left_entity["entity_id"]), str(right_entity["entity_id"]))), []
                )
                options = v5.directed_options(str(left_entity["entity_type"]), str(right_entity["entity_type"]))
                # Keep every positive pair even when the current type signature makes
                # the gold label unavailable.  Keep only type-compatible negatives,
                # matching the current v5 classifier's actual candidate universe.
                if not pair_gold and not options:
                    continue
                gold_relations = []
                for relation in pair_gold:
                    choice = target_choice(relation, str(left_entity["entity_id"]))
                    supported = choice in options
                    gold_relations.append(
                        {
                            "relation_id": relation["relation_id"],
                            "choice": choice,
                            "relation_type": relation["relation_type"],
                            "polarity": relation["polarity"],
                            "schema_supported": supported,
                        }
                    )
                    counts["schema_supported_gold_relations"] += int(supported)
                    counts["schema_unsupported_gold_relations"] += int(not supported)
                    if not supported:
                        unsupported_counts[str(relation["relation_type"])] += 1
                pairs.append(
                    {
                        "pair_id": "",
                        "left": as_mention(left_entity),
                        "right": as_mention(right_entity),
                        "allowed_choices": ["NO_RELATION", "AMBIGUOUS", *options],
                        "gold_relations": gold_relations,
                        "gold_positive": bool(gold_relations),
                    }
                )
        pairs.sort(key=v5.pair_priority)
        for index, pair in enumerate(pairs, start=1):
            pair["pair_id"] = f"{row['proposition_id']}::PAIR{index:03d}"
            counts["pairs"] += 1
            counts["positive_pairs"] += int(pair["gold_positive"])
            counts["hard_negative_pairs"] += int(not pair["gold_positive"])
        output.append(
            {
                "proposition_id": row["proposition_id"],
                "source": row["source"],
                "split": assignments[str(row["proposition_id"])],
                "claim": row["claim"],
                "mentions": [as_mention(value) for value in entities],
                "candidate_pairs": pairs,
                "truncated_pairs": 0,
            }
        )
    counts["propositions"] = len(output)
    counts["propositions_with_pairs"] = sum(bool(row["candidate_pairs"]) for row in output)
    summary = {
        **dict(counts),
        "split_counts": dict(Counter(row["split"] for row in output)),
        "source_counts": dict(Counter(row["source"] for row in output)),
        "gold_relation_types": dict(relation_counts.most_common()),
        "schema_unsupported_relation_types": dict(unsupported_counts.most_common()),
        "schema_ceiling_relation_recall": counts["schema_supported_gold_relations"]
        / max(counts["gold_relation_instances"], 1),
    }
    return output, summary


def contract(
    args: argparse.Namespace,
    pair_summary: Mapping[str, Any],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    reference_path = args.reference_root / "reference_statements.jsonl"
    value = {
        "run_version": RUN_VERSION,
        "purpose": "Isolate current v5 relation classification by replacing predicted endpoints with frozen gold endpoints.",
        "changed_variable": "entity spans, types, and positive endpoint pairs are gold; v5 prompt/model/signatures/decoding/parser are unchanged",
        "reference": file_identity(reference_path),
        "baseline_summary": file_identity(args.baseline_summary),
        "cohort": {"confidence": "high", "propositions": 768, "splits": dict(pair_summary["split_counts"])},
        "split_sha256": v5.sha256_text(v5.canonical_json(dict(sorted(assignments.items())))),
        "pairs": dict(pair_summary),
        "model": str(args.qwen_model.resolve()),
        "prompt_sha256": v5.sha256_text(v5.PAIR_SYSTEM),
        "relation_signatures": {
            key: [sorted(value[0]), sorted(value[1])] for key, value in v5.RELATION_SIGNATURES.items()
        },
        "generation": {
            "temperature": 0.0,
            "batch_size": args.generation_batch_size,
            "shard_size": args.generation_shard_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
        },
        "primary_metric": "exact directed relation edge F1 over gold positive relations and within-proposition hard negatives",
        "decision_rule": {"oracle_pass_f1": 0.75},
        "code": {
            "script": file_identity(Path(__file__).resolve()),
            "v5_classifier": file_identity(Path(v5.__file__).resolve()),
        },
        "uses_previous_relation_predictions": False,
        "uses_question_answer_gold": False,
    }
    value["contract_sha256"] = v5.sha256_text(v5.canonical_json(value))
    return value


def ensure_manifest(args: argparse.Namespace, value: Mapping[str, Any]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != value.get("contract_sha256"):
            durable = (
                args.output_root / "classification_shards",
                args.output_root / "oracle_classifications.jsonl",
                args.output_root / "summary.json",
                args.output_root / "COMPLETE.json",
            )
            if any(item.exists() for item in durable):
                raise RuntimeError("Relation-oracle resume contract mismatch; use a new output root")
    v5.atomic_json(path, value)


def classify_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(v5.read_jsonl(args.output_root / "oracle_pairs.jsonl"))
    shard_root = args.output_root / "classification_shards"
    cached = v5.cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen gold-endpoint relation/NO_RELATION classification")
    if len(cached) < len(rows):
        tokenizer, llm = v5.init_vllm(args.qwen_model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)
        for start in range(0, len(rows), args.generation_shard_size):
            source_batch = rows[start : start + args.generation_shard_size]
            missing = [row for row in source_batch if row["proposition_id"] not in cached]
            prompt_rows = [row for row in missing if row["candidate_pairs"]]
            if prompt_rows:
                prompts = [v5.chat_prompt(tokenizer, v5.PAIR_SYSTEM, v5.pair_prompt(row)) for row in prompt_rows]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(prompt_rows, outputs):
                    raw = output.outputs[0].text.strip() if output.outputs else ""
                    value = v5.validate_decisions(row, raw)
                    cached[str(value["proposition_id"])] = value
            for row in missing:
                if not row["candidate_pairs"]:
                    cached[str(row["proposition_id"])] = {
                        **row,
                        "classification_parse_success": True,
                        "pair_decisions": [],
                        "unexpected_pair_ids": [],
                        "classification_raw_output": "",
                    }
            values = [cached[str(row["proposition_id"])] for row in source_batch]
            v5.atomic_jsonl(shard_root / f"batch_{start:05d}_{start + len(source_batch):05d}.jsonl", values)
            emit("progress", total=len(rows), completed=len(cached))
    if len(cached) != len(rows):
        raise RuntimeError(f"Oracle classification incomplete: {len(cached)}/{len(rows)}")
    v5.atomic_jsonl(
        args.output_root / "oracle_classifications.jsonl",
        (cached[str(row["proposition_id"])] for row in rows),
    )
    emit("complete", total=len(rows), completed=len(rows))


def f1_counts(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def predicted_polarity(decision: Mapping[str, Any]) -> str:
    if decision.get("negated"):
        return "negated"
    if decision.get("uncertain"):
        return "uncertain"
    return "asserted"


def pair_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        pair_by_id = {str(pair["pair_id"]): pair for pair in row["candidate_pairs"]}
        decision_by_id = {str(value["pair_id"]): value for value in row["pair_decisions"]}
        for pair_id, pair in pair_by_id.items():
            decision = decision_by_id.get(pair_id)
            valid = bool(decision and decision.get("classification_valid"))
            prediction = str(decision["choice"]) if valid else "INVALID"
            predicted_positive = bool(valid and decision.get("positive"))
            gold_relations = list(pair["gold_relations"])
            gold_choices = {str(value["choice"]) for value in gold_relations}
            supported_gold = [value for value in gold_relations if value["schema_supported"]]
            exact = bool(predicted_positive and prediction in gold_choices)
            predicted_type = str(decision.get("relation_type")) if predicted_positive else None
            supported_types = {str(value["relation_type"]) for value in supported_gold}
            type_correct = bool(predicted_type and predicted_type in supported_types)
            polarity_correct = False
            if exact and decision:
                polarity = predicted_polarity(decision)
                polarity_correct = any(
                    value["choice"] == prediction and value["polarity"] == polarity for value in gold_relations
                )
            records.append(
                {
                    "proposition_id": row["proposition_id"],
                    "source": row["source"],
                    "split": row["split"],
                    "claim": row["claim"],
                    "pair_id": pair_id,
                    "left": pair["left"],
                    "right": pair["right"],
                    "gold_relations": gold_relations,
                    "gold_positive": bool(gold_relations),
                    "schema_supported_positive": bool(supported_gold),
                    "prediction": prediction,
                    "predicted_positive": predicted_positive,
                    "prediction_valid": valid,
                    "exact_directed_relation": exact,
                    "supported_relation_type_correct": type_correct,
                    "polarity_correct_given_exact": polarity_correct,
                    "validation_errors": list(decision.get("validation_errors", [])) if decision else ["missing_decision"],
                }
            )
    return records


def macro_relation_f1(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = sorted(
        {"NO_RELATION"}
        | {str(gold["relation_type"]) for row in records for gold in row["gold_relations"]}
    )
    by_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = predicted = gold_count = 0
        for row in records:
            gold_labels = {str(value["relation_type"]) for value in row["gold_relations"]} or {"NO_RELATION"}
            pred_label = "NO_RELATION" if row["prediction"] == "NO_RELATION" else None
            if row["predicted_positive"]:
                pred_label = str(row["prediction"]).split(":", 1)[-1]
            gold_count += int(label in gold_labels)
            predicted += int(pred_label == label)
            tp += int(pred_label == label and label in gold_labels)
        by_label[label] = f1_counts(tp, predicted, gold_count)
    supported = [value["f1"] for value in by_label.values() if value["gold"]]
    return {"macro_f1": sum(supported) / len(supported) if supported else 0.0, "by_label": by_label}


def metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold_edges = sum(len(row["gold_relations"]) for row in records)
    gold_positive_pairs = sum(bool(row["gold_relations"]) for row in records)
    negatives = sum(not row["gold_relations"] for row in records)
    predicted_edges = sum(bool(row["predicted_positive"]) for row in records)
    exact = sum(bool(row["exact_directed_relation"]) for row in records)
    detection_tp = sum(bool(row["gold_positive"] and row["predicted_positive"]) for row in records)
    negative_correct = sum(not row["gold_positive"] and row["prediction"] == "NO_RELATION" for row in records)
    supported_pairs = sum(bool(row["schema_supported_positive"]) for row in records)
    type_correct = sum(bool(row["supported_relation_type_correct"]) for row in records)
    exact_supported = sum(
        bool(row["schema_supported_positive"] and row["exact_directed_relation"]) for row in records
    )
    polarity_denom = exact
    polarity_correct = sum(bool(row["polarity_correct_given_exact"]) for row in records)
    relation_tpr = detection_tp / gold_positive_pairs if gold_positive_pairs else 0.0
    negative_tnr = negative_correct / negatives if negatives else 0.0
    exact_pair_correct = exact + negative_correct
    return {
        "pairs": len(records),
        "gold_positive_pairs": gold_positive_pairs,
        "gold_relation_instances": gold_edges,
        "hard_negative_pairs": negatives,
        "valid_output_rate": sum(bool(row["prediction_valid"]) for row in records) / max(len(records), 1),
        "pair_exact_accuracy": exact_pair_correct / max(len(records), 1),
        "majority_no_relation_accuracy": negatives / max(len(records), 1),
        "exact_directed_relation": f1_counts(exact, predicted_edges, gold_edges),
        "relation_detection": f1_counts(detection_tp, predicted_edges, gold_positive_pairs),
        "hard_negative_no_relation_accuracy": negative_tnr,
        "binary_balanced_accuracy": (relation_tpr + negative_tnr) / 2,
        "schema_supported_positive_pairs": supported_pairs,
        "schema_supported_exact_accuracy": exact_supported / max(supported_pairs, 1),
        "relation_type_accuracy_on_schema_supported_pairs": type_correct / max(supported_pairs, 1),
        "direction_accuracy_given_correct_supported_type": exact_supported / max(type_correct, 1),
        "polarity_accuracy_given_exact_relation": polarity_correct / max(polarity_denom, 1),
        "ambiguous_rate": sum(row["prediction"] == "AMBIGUOUS" for row in records) / max(len(records), 1),
        "invalid_rate": sum(row["prediction"] == "INVALID" for row in records) / max(len(records), 1),
        "relation_type_macro": macro_relation_f1(records),
    }


def report_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["oracle_results"]["overall"]
    exact = overall["exact_directed_relation"]
    baseline = summary["full_pipeline_baseline"]
    schema = summary["oracle_input"]
    passed = summary["decision"]["passed"]
    lines = [
        "# Gold-endpoint relation-classifier oracle",
        "",
        "> Gold entity spans, types, and positive endpoint pairs replace predicted endpoints. "
        "The v5 Qwen prompt, relation signatures, decoding, parser, and evidence validation are unchanged.",
        "",
        "## Input",
        "",
        f"- High-confidence propositions: {schema['propositions']:,}",
        f"- Gold relation instances: {schema['gold_relation_instances']:,}",
        f"- Gold positive pairs: {schema['positive_pairs']:,}",
        f"- Within-proposition hard negatives: {schema['hard_negative_pairs']:,}",
        f"- Current signature schema ceiling: {100 * schema['schema_ceiling_relation_recall']:.2f}%",
        "",
        "## Primary result",
        "",
        "| Condition | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
        f"| Full v5 pipeline | {100 * baseline['precision']:.2f}% | {100 * baseline['recall']:.2f}% | {100 * baseline['f1']:.2f}% |",
        f"| Gold endpoints + current classifier | {100 * exact['precision']:.2f}% | {100 * exact['recall']:.2f}% | {100 * exact['f1']:.2f}% |",
        "",
        f"- Hard-negative NO_RELATION accuracy: {100 * overall['hard_negative_no_relation_accuracy']:.2f}%",
        f"- Relation type accuracy on schema-supported gold pairs: {100 * overall['relation_type_accuracy_on_schema_supported_pairs']:.2f}%",
        f"- Direction accuracy conditional on correct supported relation type: {100 * overall['direction_accuracy_given_correct_supported_type']:.2f}%",
        f"- Polarity accuracy conditional on exact relation: {100 * overall['polarity_accuracy_given_exact_relation']:.2f}%",
        "",
        "## Decision",
        "",
        f"- Prespecified oracle pass threshold: exact directed relation F1 >= 75%",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Implication: {summary['decision']['implication']}",
        "",
    ]
    return "\n".join(lines)


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(v5.read_jsonl(args.output_root / "oracle_classifications.jsonl"))
    emit("start", total=len(rows), completed=0, name="score classifier against frozen gold endpoints")
    records = pair_records(rows)
    v5.atomic_jsonl(args.output_root / "per_pair_results.jsonl", records)
    for completed in range(64, len(rows), 64):
        emit("progress", total=len(rows), completed=completed)
    baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    baseline = baseline_summary["v5"]["high_confidence"]["relations"]["accepted_edges"][
        "relaxed_typed_without_polarity"
    ]
    pair_summary = json.loads((args.output_root / "preflight_summary.json").read_text(encoding="utf-8"))[
        "oracle_pairs"
    ]
    result = metrics(records)
    by_split = {
        split: metrics([row for row in records if row["split"] == split])
        for split in ("development", "validation", "test")
    }
    by_source = {
        source: metrics([row for row in records if row["source"] == source])
        for source in sorted({str(row["source"]) for row in records})
    }
    passed = result["exact_directed_relation"]["f1"] >= 0.75
    implication = (
        "The current relation classifier is adequate under gold endpoints; prioritize entity and pair generation."
        if passed
        else "The current relation decision layer remains inadequate even with gold endpoints; fix it before upstream scaling."
    )
    summary = {
        "run_version": RUN_VERSION,
        "reference_scope": "Codex-curated source-only structural reference; not physician-annotated medical truth",
        "oracle_input": pair_summary,
        "full_pipeline_baseline": baseline,
        "oracle_results": {"overall": result, "by_split": by_split, "by_source": by_source},
        "delta_vs_full_pipeline_f1": result["exact_directed_relation"]["f1"] - baseline["f1"],
        "decision": {
            "threshold": {"exact_directed_relation_f1_min": 0.75},
            "passed": passed,
            "implication": implication,
        },
    }
    v5.atomic_json(args.output_root / "summary.json", summary)
    tmp = args.output_root / "report.md.tmp"
    tmp.write_text(report_markdown(summary), encoding="utf-8")
    tmp.replace(args.output_root / "report.md")
    v5.atomic_json(
        args.output_root / "COMPLETE.json",
        {"run_version": RUN_VERSION, "summary": str((args.output_root / "summary.json").resolve())},
    )
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python),
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--reference-root",
        str(args.reference_root),
        "--baseline-summary",
        str(args.baseline_summary),
        "--output-root",
        str(args.output_root),
        "--qwen-model",
        str(args.qwen_model),
        "--llm-python",
        str(args.llm_python),
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--generation-shard-size",
        str(args.generation_shard_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--seed",
        str(args.seed),
    ]


def run_worker(args: argparse.Namespace, mode: str, progress: Any, stage: int) -> None:
    environment = dict(os.environ)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    environment.setdefault("TRANSFORMERS_VERBOSITY", "error")
    process = subprocess.Popen(
        worker_command(args, mode),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    started = False
    for line in process.stdout:
        value = line.rstrip()
        if value.startswith(PROGRESS_PREFIX):
            event = json.loads(value[len(PROGRESS_PREFIX) :])
            if event["event"] == "start":
                progress.start(stage, int(event["total"]), int(event.get("completed", 0)))
                started = True
            elif event["event"] in {"progress", "complete"}:
                if not started:
                    progress.start(stage, int(event["total"]), int(event.get("completed", 0)))
                    started = True
                progress.update(int(event["completed"]), int(event["total"]))
        elif value and not value.startswith("Loading weights:"):
            progress.log(value)
    code = process.wait()
    if code:
        raise RuntimeError(f"Worker {mode} failed with exit code {code}")


def orchestrate(args: argparse.Namespace) -> None:
    for path in (
        args.reference_root / "reference_statements.jsonl",
        args.baseline_summary,
        args.qwen_model,
        args.llm_python,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.generation_batch_size <= 0 or args.generation_shard_size <= 0:
        raise ValueError("Batch and shard sizes must be positive")
    rows = load_reference(args)
    assignments = frozen_splits(rows, args.seed)
    pairs, pair_summary = materialize_oracle_pairs(rows, assignments)
    ensure_manifest(args, contract(args, pair_summary, assignments))
    v5.atomic_jsonl(args.output_root / "oracle_pairs.jsonl", pairs)
    v5.atomic_json(args.output_root / "split_assignments.json", dict(sorted(assignments.items())))
    prompt_audit = v5.prompt_token_audit(args, pairs)
    v5.atomic_json(
        args.output_root / "preflight_summary.json",
        {"oracle_pairs": pair_summary, "prompt_tokens": prompt_audit},
    )

    progress = v5.WorkflowProgress(
        [
            "freeze gold endpoints, hard negatives, and split contract",
            "Qwen relation/NO_RELATION classification",
            "score oracle and compare with full v5 pipeline",
        ],
        [20, 360, 20],
    )
    progress.log(
        f"[workflow plan] high-confidence propositions={len(rows):,} pairs={pair_summary['pairs']:,} "
        f"positive_pairs={pair_summary['positive_pairs']:,} hard_negatives={pair_summary['hard_negative_pairs']:,}"
    )
    progress.log(
        f"[fixed variable] same Qwen/v5 prompt/signatures/parser; changed only predicted endpoints -> gold endpoints; "
        f"schema_ceiling={100 * pair_summary['schema_ceiling_relation_recall']:.2f}%"
    )
    progress.start(1, len(rows), len(rows))
    progress.log(
        f"[stage 1/3 complete] splits={pair_summary['split_counts']} "
        f"max_prompt_plus_output={prompt_audit['max_prompt_plus_output_tokens']:,} "
        f"output={args.output_root / 'oracle_pairs.jsonl'}"
    )
    if args.preflight_only:
        progress.close()
        progress.log(
            "[preflight complete] reference, exact split, gold endpoint pairs, hard negatives, prompt length, manifest, and resume contract passed; no model loaded"
        )
        return
    run_worker(args, "classify", progress, 2)
    progress.log(f"[stage 2/3 complete] output={args.output_root / 'oracle_classifications.jsonl'}")
    run_worker(args, "aggregate", progress, 3)
    progress.log(f"[stage 3/3 complete] report={args.output_root / 'report.md'}")
    progress.close()
    progress.log(f"[workflow complete] result={args.output_root}")


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate":
        orchestrate(args)
    elif args.mode == "classify":
        classify_worker(args)
    elif args.mode == "aggregate":
        aggregate_worker(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

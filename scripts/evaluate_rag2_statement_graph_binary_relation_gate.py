#!/usr/bin/env python3
"""Test a binary core-relation gate on frozen gold endpoint pairs.

This is the single changed variable after the relation oracle: instead of
choosing a relation type and direction, Qwen chooses only R (a direct frozen
core relation exists), N (no direct core relation), or A (genuinely
ambiguous).  Entity spans/types, pair cohort, model, and decoding stay fixed.
The frozen test split is not generated or reported in this development run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

import extract_rag2_statement_graph_pair_classifier_v5 as v5  # noqa: E402


RUN_VERSION = "rag2_statement_graph_binary_relation_gate_gold_endpoints_bounded256_v3_pair_isolated"
DEFAULT_ORACLE = ROOT / "datasets/kg/rag2_statement_graph_relation_oracle_1k_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_binary_relation_gate_bounded256_v3"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"
PROGRESS_PREFIX = "@@BINARY_RELATION_GATE_PROGRESS@@"
CORE_RELATIONS = tuple(
    sorted(
        {
            "IS_A", "PART_OF", "LOCATED_IN", "TREATS", "MANAGES", "PREVENTS",
            "CONTRAINDICATED_FOR", "ADMINISTERED_TO", "CAUSES", "ASSOCIATED_WITH",
            "INCREASES_RISK_OF", "DECREASES_RISK_OF", "INCREASES", "DECREASES",
            "INHIBITS", "ACTIVATES", "BINDS_OR_INTERACTS_WITH", "EXPRESSES",
            "ENCODES", "DIAGNOSES", "DETECTS", "INDICATES", "HAS_SIGN_OR_SYMPTOM",
            "COMPARED_WITH",
        }
    )
)

BINARY_SYSTEM_MINIMAL = f"""Determine whether each fixed entity pair has a direct core relation explicitly asserted by the source proposition.

R means the proposition directly asserts at least one relation from this frozen ontology between the exact two endpoints, in either direction:
{', '.join(CORE_RELATIONS)}

N means no such direct core relation is asserted. Co-occurrence, broad topical association, and relations to a third entity are N. Dose, duration, numeric value, and population qualifiers alone are not core relations. A means the wording is genuinely insufficient to decide. Do not replace endpoints or use outside knowledge.

Return one compact decision for every pair and nothing else:
{{"d":[{{"p":"P001","c":"R|N|A"}}]}}"""

BINARY_SYSTEM_DEFINED = f"""Determine whether each fixed entity pair has a direct core relation asserted by the source proposition. Ordinary paraphrases count; the exact ontology word need not appear.

R means the proposition directly asserts at least one relation from this frozen ontology between the exact two endpoints, in either direction:
{', '.join(CORE_RELATIONS)}

Use these fixed meanings:
- IS_A: one endpoint is described as a type, class, role, or form of the other.
- PART_OF / LOCATED_IN: a component-membership or physical-location relation.
- TREATS / MANAGES / PREVENTS / CONTRAINDICATED_FOR / ADMINISTERED_TO: an intervention is used for, manages, prevents, is contraindicated for, or is given to the other endpoint.
- CAUSES: one endpoint produces or leads to the other.
- ASSOCIATED_WITH: an explicit association, presentation-with, accompaniment, or absence/presence relation that has no more specific core type.
- INCREASES_RISK_OF / DECREASES_RISK_OF: one endpoint changes the risk of the other.
- INCREASES / DECREASES / INHIBITS / ACTIVATES: one endpoint changes the amount, activity, or process of the other.
- BINDS_OR_INTERACTS_WITH / EXPRESSES / ENCODES: the stated molecular relation.
- DIAGNOSES / DETECTS / INDICATES: a test, sign, or finding diagnoses, detects, measures, or indicates the other endpoint.
- HAS_SIGN_OR_SYMPTOM: a disease or condition presents with the sign, symptom, or finding.
- COMPARED_WITH: the proposition explicitly compares the two endpoints or reports a difference between them.

N means no such direct core relation is asserted. Co-occurrence, broad topical association, and relations to a third entity are N. Dose, duration, numeric value, and population qualifiers alone are not core relations. A means the wording is genuinely insufficient to decide. Do not replace endpoints or use outside knowledge.

Return one compact decision for every pair and nothing else:
{{"d":[{{"p":"P001","c":"R|N|A"}}]}}"""


def binary_system(version: str) -> str:
    return BINARY_SYSTEM_DEFINED if version == "defined" else BINARY_SYSTEM_MINIMAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "classify", "aggregate"), default="orchestrate")
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--development-limit", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--prompt-version", choices=("minimal", "defined"), default="minimal")
    parser.add_argument("--pair-isolated", action="store_true")
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


def _source_stratified_subset(rows: Sequence[dict[str, Any]], limit: int, split: str) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)
    sources = sorted(by_source)
    base, remainder = divmod(limit, len(sources))
    selected: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        quota = base + int(source_index < remainder)
        ranked = sorted(
            by_source[source],
            key=lambda row: hashlib.sha256(
                f"bounded-binary-gate-v1::{split}::{row['proposition_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < quota:
            raise RuntimeError(f"Not enough {split}/{source} rows for quota={quota}: {len(ranked)}")
        selected.extend(ranked[:quota])
    return sorted(selected, key=lambda row: str(row["proposition_id"]))


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.oracle_root / "oracle_pairs.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [dict(row) for row in v5.read_jsonl(path) if row.get("split") in {"development", "validation"}]
    split_counts = Counter(str(row["split"]) for row in rows)
    if split_counts != {"development": 512, "validation": 128}:
        raise RuntimeError(f"Expected frozen development/validation split 512/128; got {dict(split_counts)}")
    development = _source_stratified_subset(
        [row for row in rows if row["split"] == "development"], args.development_limit, "development"
    )
    validation = _source_stratified_subset(
        [row for row in rows if row["split"] == "validation"], args.validation_limit, "validation"
    )
    return development + validation


def isolate_candidate_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    isolated: list[dict[str, Any]] = []
    for row in rows:
        for pair in row["candidate_pairs"]:
            value = dict(row)
            value["original_proposition_id"] = row["proposition_id"]
            value["proposition_id"] = f"{row['proposition_id']}::{pair['pair_id'].rsplit('::', 1)[-1]}"
            value["candidate_pairs"] = [pair]
            isolated.append(value)
    return isolated


def binary_prompt(row: Mapping[str, Any]) -> str:
    pairs = [
        {
            "p": str(pair["pair_id"]).rsplit("::", 1)[-1].replace("PAIR", "P"),
            "l": [pair["left"]["surface"], pair["left"]["kind"]],
            "r": [pair["right"]["surface"], pair["right"]["kind"]],
        }
        for pair in row["candidate_pairs"]
    ]
    return (
        "SOURCE PROPOSITION:\n"
        + str(row["claim"])
        + "\n\nFIXED ENTITY PAIRS:\n"
        + json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    )


def validate_output(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = v5.parse_json_object(raw)
    decisions = parsed.get("d") if isinstance(parsed, dict) else None
    parse_success = isinstance(decisions, list)
    local_to_full = {
        str(pair["pair_id"]).rsplit("::", 1)[-1].replace("PAIR", "P"): str(pair["pair_id"])
        for pair in row["candidate_pairs"]
    }
    supplied: dict[str, list[Mapping[str, Any]]] = {}
    if parse_success:
        for item in decisions:
            if not isinstance(item, dict):
                continue
            returned = str(item.get("p") or "")
            pair_id = local_to_full.get(returned, returned)
            supplied.setdefault(pair_id, []).append(item)
    values = []
    expected = {str(pair["pair_id"]) for pair in row["candidate_pairs"]}
    for pair in row["candidate_pairs"]:
        pair_id = str(pair["pair_id"])
        candidates = supplied.get(pair_id, [])
        choice = str(candidates[0].get("c") or "").upper() if len(candidates) == 1 else ""
        errors = []
        if len(candidates) != 1:
            errors.append("missing_or_duplicate_decision")
        if choice not in {"R", "N", "A"}:
            errors.append("invalid_choice")
        values.append(
            {
                "pair_id": pair_id,
                "choice": choice,
                "valid": not errors,
                "errors": errors,
                "gold_positive": bool(pair["gold_relations"]),
                "gold_relations": pair["gold_relations"],
                "left": pair["left"],
                "right": pair["right"],
            }
        )
    unexpected = sorted(key for key in supplied if key and key not in expected)
    return {
        **row,
        "parse_success": parse_success,
        "binary_decisions": values,
        "unexpected_pair_ids": unexpected,
        "raw_output": raw,
    }


def prompt_audit(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_model), local_files_only=True, trust_remote_code=True)
    maximum = 0
    over = 0
    for row in rows:
        if not row["candidate_pairs"]:
            continue
        prompt = v5.chat_prompt(tokenizer, binary_system(args.prompt_version), binary_prompt(row))
        length = len(tokenizer.encode(prompt, add_special_tokens=False)) + args.max_output_tokens
        maximum = max(maximum, length)
        over += int(length > args.max_model_len)
    if over:
        raise RuntimeError(f"Binary prompts exceed context limit: {over} rows; maximum={maximum}")
    return {"max_prompt_plus_output_tokens": maximum, "over_limit": over}


def ensure_manifest(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    source = args.oracle_root / "oracle_pairs.jsonl"
    split_counts = Counter(str(row["split"]) for row in rows)
    pairs = sum(len(row["candidate_pairs"]) for row in rows)
    gold_pairs = sum(bool(pair["gold_relations"]) for row in rows for pair in row["candidate_pairs"])
    value = {
        "run_version": RUN_VERSION,
        "hypothesis": "Separating direct core-relation existence from relation typing improves NO_RELATION discrimination on gold endpoints.",
        "single_changed_variable": "multiclass relation+direction choice is replaced by binary R/N/A core-relation gate",
        "frozen": "gold endpoint pairs, Qwen3.5-9B, temperature=0, proposition text, entity types, development/validation split",
        "input": file_identity(source),
        "cohort": {"splits": dict(split_counts), "propositions": len(rows), "pairs": pairs, "gold_positive_pairs": gold_pairs},
        "bounded_iteration_budget": {
            "development_limit": args.development_limit,
            "validation_limit": args.validation_limit,
            "development_sampling": "deterministic SHA256 sampling, exactly balanced across the four corpora",
            "maximum_single_variable_attempts_for_this_bottleneck": 3,
        },
        "test_split_status": "not generated and not evaluated",
        "model": str(args.qwen_model.resolve()),
        "prompt_version": args.prompt_version,
        "pair_isolated": args.pair_isolated,
        "prompt_sha256": v5.sha256_text(binary_system(args.prompt_version)),
        "generation": {
            "temperature": 0.0,
            "seed": args.seed,
            "batch_size": args.generation_batch_size,
            "shard_size": args.generation_shard_size,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "prespecified_validation_pass": {
            "binary_balanced_accuracy_min": 0.75,
            "positive_recall_min": 0.75,
            "negative_specificity_min": 0.70,
        },
        "code": {"script": file_identity(Path(__file__).resolve())},
    }
    value["contract_sha256"] = v5.sha256_text(v5.canonical_json(value))
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != value["contract_sha256"]:
            durable = (
                args.output_root / "classification_shards",
                args.output_root / "binary_classifications.jsonl",
                args.output_root / "summary.json",
                args.output_root / "COMPLETE.json",
            )
            if any(item.exists() for item in durable):
                raise RuntimeError("Binary-gate resume contract mismatch; use a new output root")
    v5.atomic_json(path, value)


def classify_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(v5.read_jsonl(args.output_root / "binary_gate_inputs.jsonl"))
    shard_root = args.output_root / "classification_shards"
    cached = v5.cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen binary core-relation gate")
    if len(cached) < len(rows):
        tokenizer, llm = v5.init_vllm(args.qwen_model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)
        for start in range(0, len(rows), args.generation_shard_size):
            source_batch = rows[start : start + args.generation_shard_size]
            missing = [row for row in source_batch if row["proposition_id"] not in cached]
            prompt_rows = [row for row in missing if row["candidate_pairs"]]
            if prompt_rows:
                prompts = [v5.chat_prompt(tokenizer, binary_system(args.prompt_version), binary_prompt(row)) for row in prompt_rows]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(prompt_rows, outputs):
                    raw = output.outputs[0].text.strip() if output.outputs else ""
                    value = validate_output(row, raw)
                    cached[str(value["proposition_id"])] = value
            for row in missing:
                if not row["candidate_pairs"]:
                    cached[str(row["proposition_id"])] = {
                        **row,
                        "parse_success": True,
                        "binary_decisions": [],
                        "unexpected_pair_ids": [],
                        "raw_output": "",
                    }
            v5.atomic_jsonl(
                shard_root / f"batch_{start:05d}_{start + len(source_batch):05d}.jsonl",
                (cached[str(row["proposition_id"])] for row in source_batch),
            )
            emit("progress", total=len(rows), completed=len(cached))
    if len(cached) != len(rows):
        raise RuntimeError(f"Binary classification incomplete: {len(cached)}/{len(rows)}")
    v5.atomic_jsonl(
        args.output_root / "binary_classifications.jsonl",
        (cached[str(row["proposition_id"])] for row in rows),
    )
    emit("complete", total=len(rows), completed=len(rows))


def binary_metrics(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for value in decisions:
        counts["pairs"] += 1
        counts["positive"] += int(value["gold_positive"])
        counts["negative"] += int(not value["gold_positive"])
        counts["valid"] += int(value["valid"])
        if not value["valid"]:
            counts["invalid"] += 1
            continue
        choice = value["choice"]
        counts["predicted_positive"] += int(choice == "R")
        counts["tp"] += int(choice == "R" and value["gold_positive"])
        counts["tn"] += int(choice == "N" and not value["gold_positive"])
        counts["ambiguous"] += int(choice == "A")
    precision = counts["tp"] / counts["predicted_positive"] if counts["predicted_positive"] else 0.0
    recall = counts["tp"] / counts["positive"] if counts["positive"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = counts["tn"] / counts["negative"] if counts["negative"] else 0.0
    balanced = (recall + specificity) / 2
    return {
        "pairs": counts["pairs"],
        "gold_positive_pairs": counts["positive"],
        "gold_negative_pairs": counts["negative"],
        "valid_output_rate": counts["valid"] / max(counts["pairs"], 1),
        "invalid_rate": counts["invalid"] / max(counts["pairs"], 1),
        "positive_relation": {
            "true_positive": counts["tp"],
            "predicted": counts["predicted_positive"],
            "gold": counts["positive"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "negative_specificity": specificity,
        "balanced_accuracy": balanced,
        "ambiguous_rate": counts["ambiguous"] / max(counts["pairs"], 1),
        "majority_no_relation_accuracy": counts["negative"] / max(counts["pairs"], 1),
    }


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(v5.read_jsonl(args.output_root / "binary_classifications.jsonl"))
    emit("start", total=len(rows), completed=0, name="score binary gate on development and validation")
    records = []
    error_counts = Counter()
    for index, row in enumerate(rows, start=1):
        for value in row["binary_decisions"]:
            for error in value["errors"]:
                error_counts[error] += 1
            records.append(
                {
                    "proposition_id": row["proposition_id"],
                    "source": row["source"],
                    "split": row["split"],
                    "claim": row["claim"],
                    **value,
                }
            )
        if index % 64 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)
    v5.atomic_jsonl(args.output_root / "per_pair_results.jsonl", records)
    overall = binary_metrics(records)
    by_split = {
        split: binary_metrics([value for value in records if value["split"] == split])
        for split in ("development", "validation")
    }
    by_source = {
        source: binary_metrics([value for value in records if value["source"] == source])
        for source in sorted({str(value["source"]) for value in records})
    }
    validation = by_split["validation"]
    passed = (
        validation["balanced_accuracy"] >= 0.75
        and validation["positive_relation"]["recall"] >= 0.75
        and validation["negative_specificity"] >= 0.70
    )
    summary = {
        "run_version": RUN_VERSION,
        "test_split_status": "not generated and not evaluated",
        "metrics": {"development_and_validation": overall, "by_split": by_split, "by_source": by_source},
        "output_errors": dict(error_counts.most_common()),
        "decision": {
            "validation_thresholds": {
                "balanced_accuracy_min": 0.75,
                "positive_recall_min": 0.75,
                "negative_specificity_min": 0.70,
            },
            "passed": passed,
            "next_action": (
                "Keep the binary gate and test relation typing only on gate-positive pairs."
                if passed
                else "Reject this binary-gate formulation; do not add relation typing or modify entity extraction yet."
            ),
        },
    }
    v5.atomic_json(args.output_root / "summary.json", summary)
    v = validation
    p = v["positive_relation"]
    report = f"""# Binary core-relation gate on gold endpoints

The frozen test split was not generated or evaluated.

## Validation result

- Pairs: {v['pairs']:,} (positive={v['gold_positive_pairs']:,}, negative={v['gold_negative_pairs']:,})
- Positive relation: P {100*p['precision']:.2f}%, R {100*p['recall']:.2f}%, F1 {100*p['f1']:.2f}%
- Negative specificity: {100*v['negative_specificity']:.2f}%
- Balanced accuracy: {100*v['balanced_accuracy']:.2f}%
- Valid output rate: {100*v['valid_output_rate']:.2f}%
- Decision: {'PASS' if passed else 'FAIL'}
"""
    tmp = args.output_root / "report.md.tmp"
    tmp.write_text(report, encoding="utf-8")
    tmp.replace(args.output_root / "report.md")
    v5.atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "passed": passed})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--oracle-root", str(args.oracle_root), "--output-root", str(args.output_root),
        "--qwen-model", str(args.qwen_model), "--llm-python", str(args.llm_python),
        "--generation-batch-size", str(args.generation_batch_size),
        "--generation-shard-size", str(args.generation_shard_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--max-output-tokens", str(args.max_output_tokens),
        "--seed", str(args.seed),
        "--development-limit", str(args.development_limit),
        "--validation-limit", str(args.validation_limit),
        "--prompt-version", args.prompt_version,
    ]
    if args.pair_isolated:
        command.append("--pair-isolated")
    return command


def run_worker(args: argparse.Namespace, mode: str, progress: Any, stage: int) -> None:
    environment = dict(os.environ)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    environment.setdefault("TRANSFORMERS_VERBOSITY", "error")
    process = subprocess.Popen(
        worker_command(args, mode), cwd=ROOT, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    started = False
    for line in process.stdout:
        value = line.rstrip()
        if value.startswith(PROGRESS_PREFIX):
            event = json.loads(value[len(PROGRESS_PREFIX):])
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
    for path in (args.oracle_root / "oracle_pairs.jsonl", args.qwen_model, args.llm_python):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = load_rows(args)
    if args.pair_isolated:
        rows = isolate_candidate_pairs(rows)
    original_propositions = len({str(row.get("original_proposition_id") or row["proposition_id"]) for row in rows})
    ensure_manifest(args, rows)
    v5.atomic_jsonl(args.output_root / "binary_gate_inputs.jsonl", rows)
    audit = prompt_audit(args, rows)
    v5.atomic_json(
        args.output_root / "preflight_summary.json",
        {
            "propositions": original_propositions,
            "prompt_units": len(rows),
            "pairs": sum(len(row["candidate_pairs"]) for row in rows),
            "split_counts": dict(Counter(str(row["split"]) for row in rows)),
            "prompt_tokens": audit,
        },
    )
    progress = v5.WorkflowProgress(
        ["freeze development/validation gold pairs and binary prompt", "Qwen binary relation gate", "score gate and apply stop rule"],
        [15, 240, 15],
    )
    progress.log(
        f"[workflow plan] propositions={original_propositions:,} prompt_units={len(rows):,} "
        f"pairs={sum(len(row['candidate_pairs']) for row in rows):,} "
        f"splits={dict(Counter(str(row['split']) for row in rows))}; frozen test split is not generated"
    )
    progress.log("[single changed variable] multiclass relation+direction selection -> binary R/N/A gate; gold endpoints/model/decoding stay fixed")
    progress.start(1, len(rows), len(rows))
    progress.log(
        f"[stage 1/3 complete] max_prompt_plus_output={audit['max_prompt_plus_output_tokens']:,} "
        f"output={args.output_root / 'binary_gate_inputs.jsonl'}"
    )
    if args.preflight_only:
        progress.close()
        progress.log("[preflight complete] cohort, split exclusion, prompt length, manifest, and resume contract passed; no model loaded")
        return
    run_worker(args, "classify", progress, 2)
    progress.log(f"[stage 2/3 complete] output={args.output_root / 'binary_classifications.jsonl'}")
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

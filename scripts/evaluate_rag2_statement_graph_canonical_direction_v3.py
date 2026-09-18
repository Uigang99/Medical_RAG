#!/usr/bin/env python3
"""Re-normalize cached statement relations by emitting canonical argument IDs directly.

This bounded pilot reuses the immutable raw relation cache produced by the v2
Statement Graph run.  It removes the ambiguous preserved/reversed label: Qwen
must return the actual canonical subject and object mention IDs.  No Llama
verifier is used.  A deterministic audit evaluates clear active/passive forms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_rag2_paired_raw_proposition_kg_pilot import (  # noqa: E402
    WorkflowProgress,
    atomic_json,
    atomic_jsonl,
    cached_rows,
    canonical_json,
    chat_prompt,
    init_vllm,
    read_jsonl,
    sha256_text,
)
from build_rag2_open_statement_kg_pilot import RELATION_TYPES, parse_json_object  # noqa: E402
from build_rag2_statement_graph_v2_pilot import emit  # noqa: E402


RUN_VERSION = "rag2_statement_graph_canonical_direction_pilot_1k_v3"
DEFAULT_V2_ROOT = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_canonical_direction_pilot_1k_v3"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"

SYMMETRIC_RELATIONS = {"ASSOCIATED_WITH", "BINDS_OR_INTERACTS_WITH", "COMPARED_WITH"}

CANONICAL_DIRECTIONS = {
    "IS_A": "specific concept -> broader class",
    "PART_OF": "part -> whole",
    "LOCATED_IN": "located entity -> location",
    "HAS_ATTRIBUTE": "entity -> attribute",
    "HAS_VALUE": "entity or measurement -> value",
    "HAS_MEASUREMENT": "measured entity -> measurement",
    "HAS_DOSE": "drug or intervention -> dose",
    "HAS_DURATION": "event, condition, or intervention -> duration",
    "OCCURS_IN": "event or condition -> time, population, or location",
    "APPLIES_TO_POPULATION": "statement subject, recommendation, or intervention -> population",
    "ADMINISTERED_TO": "drug or intervention -> recipient population",
    "TREATS": "treatment -> treated condition",
    "MANAGES": "management intervention -> managed condition",
    "PREVENTS": "preventive intervention or factor -> prevented outcome",
    "CONTRAINDICATED_FOR": "intervention -> contraindicated condition or population",
    "CAUSES": "cause -> effect",
    "INCREASES_RISK_OF": "risk factor -> outcome",
    "DECREASES_RISK_OF": "protective factor -> outcome",
    "INCREASES": "factor or intervention -> increased entity or measurement",
    "DECREASES": "factor or intervention -> decreased entity or measurement",
    "INHIBITS": "inhibitor -> inhibited target",
    "ACTIVATES": "activator -> activated target",
    "PRODUCES": "producer -> product",
    "SECRETES": "secretor -> secreted substance",
    "EXPRESSES": "cell, organism, or tissue -> expressed gene or protein",
    "ENCODES": "gene -> encoded product",
    "DIAGNOSES": "diagnostic procedure -> diagnosed condition",
    "DETECTS": "test or procedure -> detected finding",
    "INDICATES": "finding -> indicated condition",
    "HAS_SIGN_OR_SYMPTOM": "condition -> sign or symptom",
    "RESULTS_IN": "antecedent event or condition -> result",
    "RECOMMENDS": "recommendation source or recommended action -> target action, condition, or population",
    "REPORTS": "study or document -> reported finding",
    "DESCRIBES": "study or document -> described topic",
    "USES": "agent, process, or intervention -> used method or resource",
    "HAS_OUTCOME": "study, process, condition, or intervention -> outcome",
}

NORMALIZATION_SYSTEM = """Normalize each source-grounded raw statement into a canonical relation triple.

Return the ACTUAL ordered canonical subject and object mention IDs. Never output an abstract direction flag. Use only the two raw argument IDs supplied for that statement; you may keep or swap them. Follow the relation-direction definitions exactly. Examples:
- Source 'A causes B' -> A CAUSES B.
- Source 'B is caused by A' -> A CAUSES B.
- Source 'A prevents B' -> A PREVENTS B.
- Source 'B is prevented by A' -> A PREVENTS B.
- Source 'Disease A is characterized by symptom B' -> Disease A HAS_SIGN_OR_SYMPTOM symptom B.
- Source 'Substance A is a vitamin' -> Substance A IS_A vitamin.

Choose OTHER rather than forcing an unsuitable ontology relation. Use status 'symmetric' only for ASSOCIATED_WITH, BINDS_OR_INTERACTS_WITH, or COMPARED_WITH. Use status 'uncertain' when neither ordering is faithfully supported, the raw arguments are wrong, or the relation cannot be normalized. Do not infer outside knowledge.

Return JSON only:
{"mappings":[{"statement_id":"S1","relation_type":"ALLOWED_TYPE","canonical_subject_mention_id":"ID","canonical_object_mention_id":"ID","status":"normalized|symmetric|uncertain","confidence":0.0,"explanation":"brief source-based reason"}]}"""

# These high-precision patterns audit orientation only; they do not establish
# complete relation-type accuracy.  First matching rule wins.
DIRECTION_RULES: Sequence[tuple[str, re.Pattern[str], str]] = (
    ("passive_cause", re.compile(r"\b(?:is|are|was|were|be|been) caused by\b|\bresults? from\b|\bconsequence of\b", re.I), "swapped"),
    ("active_cause", re.compile(r"\b(?:can |may |must )?causes?\b|\bleads? to\b|\bresults? in\b|\binduces?\b", re.I), "kept"),
    ("passive_prevent", re.compile(r"\b(?:is|are|was|were|be|been) prevented by\b", re.I), "swapped"),
    ("active_prevent", re.compile(r"\bprevents?\b|\bis used to prevent\b", re.I), "kept"),
    ("characterized_or_includes", re.compile(r"\bis characterized by\b|\b(?:includes?|included)\b", re.I), "kept"),
    ("is_a", re.compile(r"^is an?$", re.I), "kept"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "normalize", "audit"), default="orchestrate")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def contract(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = {
        "run_version": RUN_VERSION,
        "hypothesis": "Direct canonical subject/object IDs remove the ambiguity of preserved/reversed labels.",
        "input": content_identity(args.v2_root / "raw_statements_v2.jsonl"),
        "cohort": {
            "propositions": len(rows),
            "structurally_valid_raw_statements": sum(s["structurally_valid"] for r in rows for s in r["statements"]),
            "selection_sha256": sha256_text("\n".join(str(r["proposition_id"]) for r in rows)),
        },
        "model": str(args.qwen_model.resolve()),
        "prompt_sha256": sha256_text(NORMALIZATION_SYSTEM),
        "relation_types": list(RELATION_TYPES),
        "canonical_directions": CANONICAL_DIRECTIONS,
        "symmetric_relations": sorted(SYMMETRIC_RELATIONS),
        "audit_rules": [(name, pattern.pattern, expected) for name, pattern, expected in DIRECTION_RULES],
        "primary_metric": "clear-direction end-to-end orientation accuracy",
        "success_criterion": {"minimum_audit_n": 50, "minimum_resolved_coverage": 0.90, "minimum_accuracy": 0.95, "maximum_inversion_rate": 0.05},
        "generation": {
            "temperature": 0.0,
            "batch_size": args.generation_batch_size,
            "shard_size": args.generation_shard_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
        },
        "code": {"script": content_identity(Path(__file__).resolve())},
        "uses_llama_verifier": False,
        "uses_questions_answers_gold_or_semantic_labels": False,
    }
    value["contract_sha256"] = sha256_text(canonical_json(value))
    return value


def ensure_manifest(args: argparse.Namespace, value: Mapping[str, Any]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != value.get("contract_sha256"):
            raise RuntimeError("Canonical-direction v3 resume contract mismatch; use a new output root")
    else:
        atomic_json(path, value)


def validate_source(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = args.v2_root / "raw_statements_v2.jsonl"
    complete = args.v2_root / "COMPLETE.json"
    if not source.is_file() or not complete.is_file():
        raise FileNotFoundError(f"Completed v2 source required: {source}")
    rows = list(read_jsonl(source))
    ids = [str(row.get("proposition_id") or "") for row in rows]
    if len(rows) != 1000 or any(not x for x in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Expected 1,000 unique v2 rows; found rows={len(rows)} unique={len(set(ids))}")
    valid = sum(statement["structurally_valid"] for row in rows for statement in row["statements"])
    if valid != 972:
        raise RuntimeError(f"Expected immutable v2 structural cohort of 972 statements; found {valid}")
    return rows


def normalization_prompt(row: Mapping[str, Any]) -> str:
    mentions = {m["mention_id"]: {"text": m["surface"], "kind": m["kind"]} for m in row["mentions"]}
    statements = []
    for item in row["statements"]:
        if not item["structurally_valid"]:
            continue
        statements.append({
            "statement_id": item["statement_id"],
            "raw_subject_mention_id": item["raw_subject_mention_id"],
            "raw_subject": mentions[item["raw_subject_mention_id"]],
            "raw_predicate": item["predicate_text"],
            "raw_object_mention_id": item["raw_object_mention_id"],
            "raw_object": mentions[item["raw_object_mention_id"]],
            "negated": item["negated"],
            "uncertain": item["uncertain"],
            "modality": item["modality"],
        })
    return (
        "ALLOWED RELATION TYPES:\n" + ", ".join(RELATION_TYPES)
        + "\n\nCANONICAL DIRECTION DEFINITIONS:\n" + json.dumps(CANONICAL_DIRECTIONS, ensure_ascii=False)
        + "\n\nSYMMETRIC RELATIONS:\n" + ", ".join(sorted(SYMMETRIC_RELATIONS))
        + "\n\nSOURCE PROPOSITION:\n" + row["claim"]
        + "\n\nRAW STATEMENTS:\n" + json.dumps(statements, ensure_ascii=False)
    )


def validate_output(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    valid = {s["statement_id"]: s for s in row["statements"] if s["structurally_valid"]}
    mappings: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    parse_success = parsed is not None and isinstance(parsed.get("mappings"), list)
    if parse_success:
        for index, item in enumerate(parsed["mappings"]):
            if not isinstance(item, dict):
                errors.append(f"mapping_{index}:not_object")
                continue
            sid = str(item.get("statement_id") or "")
            if sid not in valid:
                errors.append(f"mapping_{index}:unknown_statement")
                continue
            relation = str(item.get("relation_type") or "OTHER").upper()
            if relation not in RELATION_TYPES:
                relation = "OTHER"
                errors.append(f"mapping_{index}:invalid_relation_type")
            status = str(item.get("status") or "uncertain")
            if status not in {"normalized", "symmetric", "uncertain"}:
                status = "uncertain"
                errors.append(f"mapping_{index}:invalid_status")
            raw_pair = {valid[sid]["raw_subject_mention_id"], valid[sid]["raw_object_mention_id"]}
            subject = str(item.get("canonical_subject_mention_id") or "")
            obj = str(item.get("canonical_object_mention_id") or "")
            if {subject, obj} != raw_pair or subject == obj:
                status = "uncertain"
                errors.append(f"mapping_{index}:canonical_arguments_not_raw_pair")
                subject, obj = valid[sid]["raw_subject_mention_id"], valid[sid]["raw_object_mention_id"]
            if relation == "OTHER":
                status = "uncertain"
            elif relation in SYMMETRIC_RELATIONS:
                status = "symmetric"
                subject, obj = sorted(raw_pair)
            elif status == "symmetric":
                status = "uncertain"
                errors.append(f"mapping_{index}:asymmetric_relation_marked_symmetric")
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            mappings[sid] = {
                "relation_type_v3": relation,
                "canonical_subject_mention_id_v3": subject,
                "canonical_object_mention_id_v3": obj,
                "orientation_status_v3": status,
                "normalization_confidence_v3": confidence,
                "normalization_explanation_v3": str(item.get("explanation") or ""),
            }
    statements = []
    for statement in row["statements"]:
        mapping = mappings.get(statement["statement_id"])
        if not statement["structurally_valid"]:
            mapping = {"relation_type_v3": None, "canonical_subject_mention_id_v3": None, "canonical_object_mention_id_v3": None, "orientation_status_v3": "invalid_raw_statement", "normalization_confidence_v3": None, "normalization_explanation_v3": ""}
        elif mapping is None:
            mapping = {"relation_type_v3": "OTHER", "canonical_subject_mention_id_v3": statement["raw_subject_mention_id"], "canonical_object_mention_id_v3": statement["raw_object_mention_id"], "orientation_status_v3": "uncertain", "normalization_confidence_v3": None, "normalization_explanation_v3": "missing mapping"}
            errors.append(f"{statement['statement_id']}:missing_mapping")
        statements.append({**statement, **mapping})
    return {**row, "v3_parse_success": parse_success, "v3_errors": errors if parse_success else ["invalid_json_or_schema"], "statements": statements, "v3_raw_output": raw}


def clear_direction_rule(predicate: str) -> tuple[str, str] | None:
    for name, pattern, expected in DIRECTION_RULES:
        if pattern.search(predicate):
            return name, expected
    return None


def normalize_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(read_jsonl(args.v2_root / "raw_statements_v2.jsonl"))
    shard_root = args.output_root / "normalization_shards"
    cached = cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen canonical subject--relation--object normalization")
    if len(cached) < len(rows):
        tokenizer, llm = init_vllm(args.qwen_model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)
        done = len(cached)
        for start in range(0, len(rows), args.generation_shard_size):
            batch = rows[start:start + args.generation_shard_size]
            missing = [row for row in batch if row["proposition_id"] not in cached]
            if not missing:
                continue
            prompts = [chat_prompt(tokenizer, NORMALIZATION_SYSTEM, normalization_prompt(row)) for row in missing]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            values = [validate_output(row, output.outputs[0].text.strip() if output.outputs else "") for row, output in zip(missing, outputs)]
            atomic_jsonl(shard_root / f"batch_{start:05d}_{start + len(batch):05d}.jsonl", values)
            for value in values:
                cached[value["proposition_id"]] = value
            done += len(values)
            emit("progress", total=len(rows), completed=done)
    if len(cached) != len(rows):
        raise RuntimeError(f"Normalization incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / "canonical_triples_v3.jsonl", (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def audit_worker(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.output_root / "canonical_triples_v3.jsonl"))
    emit("start", total=len(rows), completed=0, name="deterministic clear-direction audit and report")
    counts = Counter()
    status_counts = Counter()
    relation_counts = Counter()
    rule_counts: dict[str, Counter[str]] = {}
    mismatches: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        counts["propositions"] += 1
        counts["parse_success"] += int(row["v3_parse_success"])
        mentions = {m["mention_id"]: m["surface"] for m in row["mentions"]}
        for statement in row["statements"]:
            if not statement["structurally_valid"]:
                continue
            counts["structurally_valid"] += 1
            status = statement["orientation_status_v3"]
            status_counts[status] += 1
            relation_counts[str(statement["relation_type_v3"])] += 1
            if status in {"normalized", "symmetric"} and statement["relation_type_v3"] != "OTHER":
                edges.append({
                    "statement_id": statement["statement_id"], "proposition_id": row["proposition_id"], "source": row["source"],
                    "subject_mention_id": statement["canonical_subject_mention_id_v3"], "relation_type": statement["relation_type_v3"],
                    "object_mention_id": statement["canonical_object_mention_id_v3"], "orientation_status": status,
                })
            rule = clear_direction_rule(statement["predicate_text"])
            if rule is None:
                continue
            name, expected = rule
            bucket = rule_counts.setdefault(name, Counter())
            bucket["total"] += 1
            counts["clear_total"] += 1
            if status != "normalized":
                bucket["unresolved"] += 1
                counts["clear_unresolved"] += 1
                if len(uncertain) < 200:
                    uncertain.append({"rule": name, "expected": expected, "proposition": row["claim"], "statement": statement})
                continue
            counts["clear_resolved"] += 1
            predicted = "kept" if statement["canonical_subject_mention_id_v3"] == statement["raw_subject_mention_id"] else "swapped"
            if predicted == expected:
                bucket["correct"] += 1
                counts["clear_correct"] += 1
            else:
                bucket["wrong"] += 1
                counts["clear_wrong"] += 1
                if len(mismatches) < 200:
                    mismatches.append({
                        "rule": name, "expected": expected, "predicted": predicted, "proposition": row["claim"],
                        "raw_subject": mentions.get(statement["raw_subject_mention_id"]), "raw_predicate": statement["predicate_text"],
                        "raw_object": mentions.get(statement["raw_object_mention_id"]),
                        "canonical_subject": mentions.get(statement["canonical_subject_mention_id_v3"]),
                        "relation_type": statement["relation_type_v3"],
                        "canonical_object": mentions.get(statement["canonical_object_mention_id_v3"]),
                        "explanation": statement["normalization_explanation_v3"],
                    })
        if index % 50 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)

    total = counts["clear_total"]
    resolved = counts["clear_resolved"]
    correct = counts["clear_correct"]
    wrong = counts["clear_wrong"]
    coverage = resolved / total if total else 0.0
    resolved_accuracy = correct / resolved if resolved else 0.0
    end_to_end_accuracy = correct / total if total else 0.0
    inversion_rate = wrong / total if total else 0.0
    passed = total >= 50 and coverage >= 0.90 and end_to_end_accuracy >= 0.95 and inversion_rate <= 0.05
    summary = {
        "run_version": RUN_VERSION,
        "scope": "Direction-format pilot only; this does not establish full relation-type or entity accuracy.",
        "input": {"propositions": counts["propositions"], "structurally_valid_raw_statements": counts["structurally_valid"]},
        "normalization": {
            "parse_success": counts["parse_success"], "parse_success_rate": counts["parse_success"] / counts["propositions"],
            "orientation_status": dict(status_counts), "relation_types": dict(relation_counts), "candidate_edges": len(edges),
        },
        "clear_direction_audit": {
            "total": total, "resolved": resolved, "correct": correct, "wrong": wrong, "unresolved": counts["clear_unresolved"],
            "resolved_coverage": coverage, "accuracy_among_resolved": resolved_accuracy,
            "end_to_end_accuracy": end_to_end_accuracy, "inversion_rate": inversion_rate,
            "by_rule": {name: dict(value) for name, value in rule_counts.items()},
            "success_criterion_passed": passed,
        },
        "interpretation": "Scale only if the preregistered clear-direction criterion passes; inspect saved mismatch and uncertain cases before broader KG use.",
    }
    atomic_jsonl(args.output_root / "candidate_semantic_edges_v3.jsonl", edges)
    atomic_jsonl(args.output_root / "diagnostics/direction_mismatches.jsonl", mismatches)
    atomic_jsonl(args.output_root / "diagnostics/direction_uncertain.jsonl", uncertain)
    atomic_json(args.output_root / "summary.json", summary)
    atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "summary": str((args.output_root / "summary.json").resolve())})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--v2-root", str(args.v2_root), "--output-root", str(args.output_root),
        "--qwen-model", str(args.qwen_model), "--llm-python", str(args.llm_python),
        "--generation-batch-size", str(args.generation_batch_size), "--generation-shard-size", str(args.generation_shard_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-model-len", str(args.max_model_len),
        "--max-output-tokens", str(args.max_output_tokens), "--seed", str(args.seed),
    ]


def run_worker(args: argparse.Namespace, mode: str, progress: WorkflowProgress, stage: int) -> None:
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    process = subprocess.Popen(worker_command(args, mode), cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    started = False
    for line in process.stdout:
        text = line.rstrip()
        if text.startswith("@@STATEMENT_GRAPH_PROGRESS@@"):
            event = json.loads(text[len("@@STATEMENT_GRAPH_PROGRESS@@"):])
            if event["event"] == "start":
                progress.start(stage, int(event["total"]), int(event.get("completed", 0)))
                started = True
            elif event["event"] in {"progress", "complete"}:
                if not started:
                    progress.start(stage, int(event["total"]), int(event.get("completed", 0)))
                    started = True
                progress.update(int(event["completed"]), int(event["total"]))
        elif text and not text.startswith("Loading weights:"):
            progress.log(text)
    code = process.wait()
    if code:
        raise RuntimeError(f"Worker {mode} failed with exit code {code}")


def orchestrate(args: argparse.Namespace) -> None:
    for path in (args.v2_root, args.qwen_model, args.llm_python):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = validate_source(args)
    ensure_manifest(args, contract(args, rows))
    progress = WorkflowProgress(
        ["validate immutable v2 raw-relation cohort", "Qwen canonical triple normalization", "clear-direction audit and report"],
        [5, 105, 5],
    )
    progress.log("[workflow plan] reuse 972 v2 structural statements -> direct canonical IDs with Qwen -> deterministic active/passive audit")
    progress.log("[changed variable] normalization prompt/schema only; no Llama verifier and no relation re-extraction")
    progress.start(1, len(rows), len(rows))
    progress.log(f"[stage 1/3 complete] propositions={len(rows):,} structural_statements=972 input={args.v2_root / 'raw_statements_v2.jsonl'}")
    if args.preflight_only:
        progress.close()
        progress.log("[preflight complete] immutable cohort, model path, manifest, and output contract passed; no model loaded")
        return
    run_worker(args, "normalize", progress, 2)
    progress.log(f"[stage 2/3 complete] output={args.output_root / 'canonical_triples_v3.jsonl'}")
    run_worker(args, "audit", progress, 3)
    progress.log(f"[stage 3/3 complete] summary={args.output_root / 'summary.json'} diagnostics={args.output_root / 'diagnostics'}")
    progress.close()
    progress.log(f"[workflow complete] result={args.output_root}")


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate":
        orchestrate(args)
    elif args.mode == "normalize":
        normalize_worker(args)
    elif args.mode == "audit":
        audit_worker(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

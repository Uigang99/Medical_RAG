#!/usr/bin/env python3
"""Classify type-constrained mention pairs in standalone propositions.

This precision-first pilot fixes each candidate pair before inference.  Qwen
may choose only a pair-specific directed ontology relation, NO_RELATION, or
AMBIGUOUS.  It cannot invent endpoints or reuse v2-v4 relation predictions.
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
from build_rag2_open_statement_kg_pilot import parse_json_object  # noqa: E402
from build_rag2_statement_graph_v2_pilot import all_exact_spans, normalized_surface  # noqa: E402


RUN_VERSION = "rag2_statement_graph_pair_classifier_pilot_1k_v5"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
DEFAULT_V2_ROOT = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_V4_ROOT = ROOT / "datasets/kg/rag2_statement_graph_semantic_frames_pilot_1k_v4"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_pair_classifier_pilot_1k_v5"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"

CORE_KINDS = frozenset(
    {
        "condition", "drug_or_chemical", "procedure_or_intervention",
        "finding_or_outcome", "anatomy", "organism", "gene_or_protein",
        "population", "event", "measurement", "numeric_literal", "dose",
        "time", "location",
    }
)
CONDITIONS = frozenset({"condition", "finding_or_outcome"})
TREATMENTS = frozenset({"drug_or_chemical", "procedure_or_intervention"})
FINDINGS = frozenset({"finding_or_outcome", "measurement"})
POPULATIONS = frozenset({"population", "organism"})
LOCATIONS = frozenset({"anatomy", "location"})
BIOLOGICAL = frozenset(
    {
        "condition", "drug_or_chemical", "procedure_or_intervention",
        "finding_or_outcome", "anatomy", "organism", "gene_or_protein",
        "event", "measurement",
    }
)
VALUES = frozenset({"measurement", "numeric_literal", "dose", "time"})
SYMMETRIC_RELATIONS = frozenset({"ASSOCIATED_WITH", "BINDS_OR_INTERACTS_WITH", "COMPARED_WITH"})

# Relation signatures define valid semantic roles, not merely co-occurring
# mention types. Generic HAS_ATTRIBUTE/DESCRIBES/REPORTS are intentionally
# excluded from the precision-first edge layer.
RELATION_SIGNATURES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "IS_A": (BIOLOGICAL, BIOLOGICAL),
    "PART_OF": (frozenset({"anatomy", "gene_or_protein"}), frozenset({"anatomy", "gene_or_protein"})),
    "LOCATED_IN": (BIOLOGICAL, LOCATIONS),
    "HAS_VALUE": (frozenset({"condition", "finding_or_outcome", "measurement", "event"}), VALUES),
    "HAS_MEASUREMENT": (frozenset({"condition", "finding_or_outcome", "population", "event"}), frozenset({"measurement", "numeric_literal"})),
    "HAS_DOSE": (TREATMENTS, frozenset({"dose", "numeric_literal"})),
    "HAS_DURATION": (frozenset({"condition", "finding_or_outcome", "procedure_or_intervention", "event"}), frozenset({"time", "numeric_literal"})),
    "OCCURS_IN": (frozenset({"condition", "finding_or_outcome", "event"}), POPULATIONS | LOCATIONS),
    "APPLIES_TO_POPULATION": (TREATMENTS | CONDITIONS | frozenset({"event"}), POPULATIONS),
    "ADMINISTERED_TO": (TREATMENTS, POPULATIONS),
    "TREATS": (TREATMENTS, CONDITIONS),
    "MANAGES": (TREATMENTS, CONDITIONS),
    "PREVENTS": (TREATMENTS, CONDITIONS),
    "CONTRAINDICATED_FOR": (TREATMENTS, CONDITIONS | POPULATIONS),
    "CAUSES": (BIOLOGICAL, CONDITIONS | frozenset({"event", "measurement"})),
    "ASSOCIATED_WITH": (BIOLOGICAL, BIOLOGICAL),
    "INCREASES_RISK_OF": (BIOLOGICAL, CONDITIONS),
    "DECREASES_RISK_OF": (BIOLOGICAL, CONDITIONS),
    "INCREASES": (BIOLOGICAL, CONDITIONS | frozenset({"event", "measurement"})),
    "DECREASES": (BIOLOGICAL, CONDITIONS | frozenset({"event", "measurement"})),
    "INHIBITS": (BIOLOGICAL, BIOLOGICAL),
    "ACTIVATES": (BIOLOGICAL, BIOLOGICAL),
    "BINDS_OR_INTERACTS_WITH": (frozenset({"drug_or_chemical", "gene_or_protein"}), frozenset({"drug_or_chemical", "gene_or_protein"})),
    "PRODUCES": (BIOLOGICAL, frozenset({"drug_or_chemical", "gene_or_protein", "finding_or_outcome"})),
    "SECRETES": (frozenset({"anatomy", "organism", "gene_or_protein"}), frozenset({"drug_or_chemical", "gene_or_protein"})),
    "EXPRESSES": (frozenset({"anatomy", "organism", "condition", "gene_or_protein"}), frozenset({"gene_or_protein"})),
    "ENCODES": (frozenset({"gene_or_protein"}), frozenset({"gene_or_protein"})),
    "DIAGNOSES": (frozenset({"procedure_or_intervention"}), CONDITIONS),
    "DETECTS": (frozenset({"procedure_or_intervention"}), CONDITIONS | FINDINGS),
    "INDICATES": (FINDINGS, frozenset({"condition"})),
    "HAS_SIGN_OR_SYMPTOM": (frozenset({"condition"}), frozenset({"finding_or_outcome"})),
    "COMPARED_WITH": (BIOLOGICAL | VALUES, BIOLOGICAL | VALUES),
    "RESULTS_IN": (BIOLOGICAL, CONDITIONS | frozenset({"event", "measurement"})),
}

MODALITIES = frozenset({"asserted", "may", "should", "must", "can", "reported"})

PAIR_SYSTEM = """Classify every supplied entity pair independently using only the source proposition.

The endpoints and allowed choices are fixed. Do not replace endpoints, invent entities, or infer outside medical knowledge. Choose a positive relation only when the proposition explicitly asserts that exact semantic relation between the two endpoints. Co-occurrence is NO_RELATION. Prefer NO_RELATION over a broad or approximate relation. AMBIGUOUS means the proposition is genuinely insufficient to choose between listed options. Preserve direction, negation, uncertainty, and modality. A positive decision requires the shortest exact contiguous evidence span that supports both endpoints and the relation.

Choice codes are N=NO_RELATION, A=AMBIGUOUS, L:RELATION=left-to-right,
R:RELATION=right-to-left, and S:RELATION=symmetric. Return one decision for
every local pair ID and compact JSON only:
{"d":[{"p":"P001","c":"one supplied choice code","e":["exact contiguous source span"],"n":false,"u":false,"m":"asserted|may|should|must|can|reported"}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "classify", "aggregate"), default="orchestrate")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    parser.add_argument("--max-pairs-per-proposition", type=int, default=80)
    parser.add_argument("--review-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def emit(event: str, **payload: Any) -> None:
    print("@@PAIR_V5_PROGRESS@@" + canonical_json({"event": event, **payload}), flush=True)


def same_concept(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_cui, right_cui = left.get("umls_cui"), right.get("umls_cui")
    if left_cui and right_cui and left_cui == right_cui:
        return True
    return normalized_surface(str(left["surface"])) == normalized_surface(str(right["surface"]))


def directed_options(left_kind: str, right_kind: str) -> list[str]:
    choices: set[str] = set()
    for relation, (heads, tails) in RELATION_SIGNATURES.items():
        left_to_right = left_kind in heads and right_kind in tails
        right_to_left = right_kind in heads and left_kind in tails
        if relation in SYMMETRIC_RELATIONS:
            if left_to_right or right_to_left:
                choices.add(f"SYMMETRIC:{relation}")
        else:
            if left_to_right:
                choices.add(f"L2R:{relation}")
            if right_to_left:
                choices.add(f"R2L:{relation}")
    return sorted(choices)


def pair_priority(pair: Mapping[str, Any]) -> tuple[int, int, str]:
    # Specific choices before broad symmetric choices, then close spans.
    choices = pair["allowed_choices"]
    specific = sum(not choice.endswith(("ASSOCIATED_WITH", "COMPARED_WITH")) for choice in choices)
    distance = abs(int(pair["right"]["start"]) - int(pair["left"]["end"]))
    return (-specific, distance, str(pair["pair_id"]))


def construct_pairs(row: Mapping[str, Any], max_pairs: int) -> tuple[list[dict[str, Any]], int]:
    mentions = [dict(value) for value in row["mentions"] if str(value.get("kind")) in CORE_KINDS]
    mentions.sort(key=lambda value: (int(value["start"]), int(value["end"]), str(value["mention_id"])))
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(mentions):
        for right in mentions[left_index + 1 :]:
            if same_concept(left, right):
                continue
            options = directed_options(str(left["kind"]), str(right["kind"]))
            if not options:
                continue
            pairs.append(
                {
                    "pair_id": "",
                    "left": {key: left.get(key) for key in ("mention_id", "surface", "kind", "start", "end", "umls_cui", "umls_name")},
                    "right": {key: right.get(key) for key in ("mention_id", "surface", "kind", "start", "end", "umls_cui", "umls_name")},
                    "allowed_choices": ["NO_RELATION", "AMBIGUOUS", *options],
                }
            )
    pairs.sort(key=pair_priority)
    untruncated = len(pairs)
    pairs = pairs[:max_pairs]
    for index, pair in enumerate(pairs, start=1):
        pair["pair_id"] = f"{row['proposition_id']}::PAIR{index:03d}"
    return pairs, max(0, untruncated - len(pairs))


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = args.v2_root / "raw_statements_v2.jsonl"
    if not source.is_file() or not (args.v2_root / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"Completed v2 source required: {source}")
    rows = list(read_jsonl(source))
    identifiers = [str(row.get("proposition_id") or "") for row in rows]
    if len(rows) != 1000 or any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Expected 1,000 unique propositions; rows={len(rows)} unique={len(set(identifiers))}")
    if Counter(str(row.get("source")) for row in rows) != {source: 250 for source in SOURCES}:
        raise RuntimeError("Expected 250 propositions from each of four sources")
    return rows


def materialize_pairs(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values: list[dict[str, Any]] = []
    counts = Counter()
    option_counts = Counter()
    for row in rows:
        pairs, dropped = construct_pairs(row, args.max_pairs_per_proposition)
        counts["propositions"] += 1
        counts["candidate_pairs"] += len(pairs)
        counts["truncated_pairs"] += dropped
        counts["propositions_with_pairs"] += int(bool(pairs))
        counts["truncated_propositions"] += int(dropped > 0)
        for pair in pairs:
            for choice in pair["allowed_choices"]:
                if ":" in choice:
                    option_counts[choice.split(":", 1)[1]] += 1
        values.append(
            {
                "proposition_id": row["proposition_id"], "source": row["source"], "claim": row["claim"],
                "mentions": row["mentions"], "candidate_pairs": pairs, "truncated_pairs": dropped,
            }
        )
    summary = {**dict(counts), "relation_option_counts": dict(option_counts)}
    return values, summary


def pair_prompt(row: Mapping[str, Any]) -> str:
    def compact_choice(choice: str) -> str:
        if choice == "NO_RELATION":
            return "N"
        if choice == "AMBIGUOUS":
            return "A"
        orientation, relation = choice.split(":", 1)
        short_orientation = {"L2R": "L", "R2L": "R", "SYMMETRIC": "S"}[orientation]
        return f"{short_orientation}:{relation}"

    choice_sets: dict[tuple[str, ...], str] = {}
    schemas: dict[str, list[str]] = {}
    pairs = []
    for pair in row["candidate_pairs"]:
        choices = tuple(compact_choice(value) for value in pair["allowed_choices"])
        if choices not in choice_sets:
            schema_id = f"S{len(choice_sets) + 1:02d}"
            choice_sets[choices] = schema_id
            schemas[schema_id] = list(choices)
        pairs.append(
            {
                "p": pair["pair_id"].rsplit("::", 1)[-1].replace("PAIR", "P"),
                "l": [pair["left"]["surface"], pair["left"]["kind"]],
                "r": [pair["right"]["surface"], pair["right"]["kind"]],
                "s": choice_sets[choices],
            }
        )
    payload = {"choice_sets": schemas, "pairs": pairs}
    return "SOURCE PROPOSITION:\n" + str(row["claim"]) + "\n\nFIXED PAIR TASK:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def prompt_token_audit(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_model), local_files_only=True, trust_remote_code=True)
    maximum = 0
    over_limit = 0
    for row in rows:
        if not row["candidate_pairs"]:
            continue
        prompt = chat_prompt(tokenizer, PAIR_SYSTEM, pair_prompt(row))
        length = len(tokenizer.encode(prompt, add_special_tokens=False)) + args.max_output_tokens
        maximum = max(maximum, length)
        over_limit += int(length > args.max_model_len)
    if over_limit:
        raise RuntimeError(f"Pair prompts exceed max model length: rows={over_limit} max_tokens={maximum} limit={args.max_model_len}")
    return {"max_prompt_plus_output_tokens": maximum, "over_limit": over_limit}


def contract(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], pair_summary: Mapping[str, Any], review_ids: Sequence[str]) -> dict[str, Any]:
    value = {
        "run_version": RUN_VERSION,
        "hypothesis": "Fixing type-compatible entity pairs and classifying a restricted relation or NO_RELATION improves exact semantic-edge precision over free semantic-frame generation.",
        "changed_variable": "free endpoint/frame generation is replaced by fixed-pair constrained classification",
        "input": file_identity(args.v2_root / "raw_statements_v2.jsonl"),
        "cohort": {"propositions": len(rows), "by_source": dict(Counter(row["source"] for row in rows))},
        "candidate_pairs": dict(pair_summary),
        "model": str(args.qwen_model.resolve()),
        "prompt_sha256": sha256_text(PAIR_SYSTEM),
        "relation_signatures": {key: [sorted(value[0]), sorted(value[1])] for key, value in RELATION_SIGNATURES.items()},
        "generation": {
            "temperature": 0.0, "batch_size": args.generation_batch_size, "shard_size": args.generation_shard_size,
            "gpu_memory_utilization": args.gpu_memory_utilization, "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens, "max_pairs_per_proposition": args.max_pairs_per_proposition,
            "seed": args.seed,
        },
        "evaluation": {
            "fixed_v4_review_ids_sha256": sha256_text("\n".join(review_ids)), "review_size": len(review_ids),
            "pilot_pass": {"exact_triple_precision": 0.75, "direction_accuracy": 0.90, "wrong_edge_rate_max": 0.10, "correct_edge_proposition_coverage": 0.20},
            "scale_pass": {"exact_triple_precision": 0.85, "direction_accuracy": 0.95, "entity_normalization_accuracy": 0.90},
            "automatic_status": "pending_manual_audit",
        },
        "code": {"script": file_identity(Path(__file__).resolve())},
        "uses_v2_v3_v4_relation_predictions_as_model_input": False,
        "uses_questions_answers_gold_or_semantic_labels": False,
    }
    value["contract_sha256"] = sha256_text(canonical_json(value))
    return value


def ensure_manifest(args: argparse.Namespace, value: Mapping[str, Any]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != value.get("contract_sha256"):
            durable = (
                args.output_root / "classification_shards", args.output_root / "pair_classifications_v5.jsonl",
                args.output_root / "summary.json", args.output_root / "COMPLETE.json",
            )
            if any(item.exists() for item in durable):
                raise RuntimeError("Pair-classifier v5 resume contract mismatch; use a new output root")
            atomic_json(path, value)
    else:
        atomic_json(path, value)


def fixed_review_ids(args: argparse.Namespace) -> list[str]:
    path = args.v4_root / "review_sample_200.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Fixed v4 review sample required: {path}")
    identifiers = [str(row["proposition_id"]) for row in read_jsonl(path)]
    if len(identifiers) != args.review_size or len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Expected {args.review_size} unique fixed review IDs; found {len(identifiers)}")
    return identifiers


def validate_decisions(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    decisions = (parsed.get("d") or parsed.get("decisions")) if isinstance(parsed, dict) else None
    parse_success = isinstance(decisions, list)
    supplied: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    canonical_ids = {str(pair["pair_id"]) for pair in row["candidate_pairs"]}
    local_to_canonical = {
        str(pair["pair_id"]).rsplit("::", 1)[-1].replace("PAIR", "P"): str(pair["pair_id"])
        for pair in row["candidate_pairs"]
    }
    if parse_success:
        for decision in decisions:
            if isinstance(decision, dict):
                returned_id = str(decision.get("p") or decision.get("pair_id") or "")
                supplied[local_to_canonical.get(returned_id, returned_id)].append(decision)
    validated: list[dict[str, Any]] = []
    for pair in row["candidate_pairs"]:
        errors: list[str] = []
        candidates = supplied.get(pair["pair_id"], [])
        if len(candidates) != 1:
            errors.append("missing_or_duplicate_pair_decision")
        item = candidates[0] if candidates else {}
        choice = str(item.get("c") or item.get("choice") or "")
        if choice == "N":
            choice = "NO_RELATION"
        elif choice == "A":
            choice = "AMBIGUOUS"
        elif ":" in choice and choice.split(":", 1)[0] in {"L", "R", "S"}:
            short_orientation, relation = choice.split(":", 1)
            orientation = {"L": "L2R", "R": "R2L", "S": "SYMMETRIC"}[short_orientation]
            choice = f"{orientation}:{relation}"
        if choice not in pair["allowed_choices"]:
            errors.append("choice_not_allowed_for_pair")
        positive = choice not in {"NO_RELATION", "AMBIGUOUS", ""}
        evidence: list[dict[str, Any]] = []
        for evidence_text in item.get("e") or item.get("evidence_spans") or []:
            value = str(evidence_text or "").strip()
            spans = all_exact_spans(str(row["claim"]), value)
            if value and spans:
                evidence.append({"text": value, "spans": spans})
            else:
                errors.append("nonexact_evidence_span")
        if positive and not evidence:
            errors.append("positive_without_exact_evidence")
        if positive and evidence:
            evidence_text = " ".join(value["text"] for value in evidence).casefold()
            left_text = str(pair["left"]["surface"]).casefold()
            right_text = str(pair["right"]["surface"]).casefold()
            if left_text not in evidence_text or right_text not in evidence_text:
                errors.append("evidence_does_not_contain_both_endpoints")
        modality = str(item.get("m") or item.get("modality") or "asserted")
        if modality not in MODALITIES:
            errors.append("invalid_modality")
        subject, obj, relation = None, None, None
        if positive and choice in pair["allowed_choices"]:
            orientation, relation = choice.split(":", 1)
            if orientation == "L2R":
                subject, obj = pair["left"], pair["right"]
            elif orientation == "R2L":
                subject, obj = pair["right"], pair["left"]
            elif orientation == "SYMMETRIC":
                subject, obj = pair["left"], pair["right"]
            else:
                errors.append("invalid_orientation")
        validated.append(
            {
                **pair, "choice": choice or "MISSING", "classification_valid": not errors,
                "positive": positive and not errors, "subject": subject, "relation_type": relation, "object": obj,
                "evidence": evidence, "negated": bool(item.get("n", item.get("negated", False))),
                "uncertain": bool(item.get("u", item.get("uncertain", False))), "modality": modality,
                "validation_errors": errors,
            }
        )
    unexpected = sorted(pair_id for pair_id in supplied if pair_id and pair_id not in canonical_ids)
    return {
        **row, "classification_parse_success": parse_success, "pair_decisions": validated,
        "unexpected_pair_ids": unexpected, "classification_raw_output": raw,
    }


def classify_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(read_jsonl(args.output_root / "candidate_pairs_v5.jsonl"))
    shard_root = args.output_root / "classification_shards"
    cached = cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen fixed-pair relation/NO_RELATION classification")
    if len(cached) < len(rows):
        tokenizer, llm = init_vllm(args.qwen_model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)
        # Stable source-indexed shards make interrupted runs exactly resumable.
        for start in range(0, len(rows), args.generation_shard_size):
            source_batch = rows[start : start + args.generation_shard_size]
            missing = [row for row in source_batch if row["proposition_id"] not in cached]
            pair_rows = [row for row in missing if row["candidate_pairs"]]
            if pair_rows:
                prompts = [chat_prompt(tokenizer, PAIR_SYSTEM, pair_prompt(row)) for row in pair_rows]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(pair_rows, outputs):
                    value = validate_decisions(row, output.outputs[0].text.strip() if output.outputs else "")
                    cached[value["proposition_id"]] = value
            for row in missing:
                if not row["candidate_pairs"]:
                    cached[row["proposition_id"]] = {
                        **row, "classification_parse_success": True, "pair_decisions": [],
                        "unexpected_pair_ids": [], "classification_raw_output": "",
                    }
            shard_values = [cached[row["proposition_id"]] for row in source_batch]
            atomic_jsonl(
                shard_root / f"batch_{start:05d}_{start + len(source_batch):05d}.jsonl",
                shard_values,
            )
            emit("progress", total=len(rows), completed=len(cached))
    if len(cached) != len(rows):
        raise RuntimeError(f"Pair classification incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / "pair_classifications_v5.jsonl", (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def compact_v4(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    mentions = {value["mention_id"]: value["surface"] for value in row["mentions"]}
    return [
        {
            "subject": mentions.get(frame.get("subject_mention_id")), "relation": frame.get("relation_type"),
            "object": mentions.get(frame.get("object_mention_id")), "structurally_valid": frame.get("structurally_valid"),
        }
        for frame in row["semantic_frames"]
    ]


def compact_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": value["pair_id"], "left": value["left"]["surface"], "right": value["right"]["surface"],
        "choice": value["choice"], "classification_valid": value["classification_valid"],
        "evidence": [item["text"] for item in value["evidence"]], "validation_errors": value["validation_errors"],
    }


def balanced_positive_sample(edges: Sequence[Mapping[str, Any]], size: int, seed: int) -> list[Mapping[str, Any]]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        buckets[(str(edge["source"]), str(edge["relation_type"]))].append(edge)
    for key in buckets:
        buckets[key].sort(key=lambda row: sha256_text(f"{seed}|{row['pair_id']}"))
    selected: list[Mapping[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < size and keys:
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < size:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return selected


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.output_root / "pair_classifications_v5.jsonl"))
    emit("start", total=len(rows), completed=0, name="aggregate pair decisions and fixed review material")
    counts = Counter()
    choice_counts = Counter()
    relation_counts = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    edges: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        counts["propositions"] += 1
        counts["parse_success"] += int(row["classification_parse_success"])
        counts["propositions_with_pairs"] += int(bool(row["candidate_pairs"]))
        row_positive = False
        for decision in row["pair_decisions"]:
            counts["pairs"] += 1
            counts["valid_classifications"] += int(decision["classification_valid"])
            choice_counts[decision["choice"]] += 1
            if decision["positive"]:
                row_positive = True
                relation_counts[decision["relation_type"]] += 1
                source_counts[row["source"]][decision["relation_type"]] += 1
                edges.append(
                    {
                        "pair_id": decision["pair_id"], "proposition_id": row["proposition_id"], "source": row["source"],
                        "proposition": row["claim"], "subject_mention_id": decision["subject"]["mention_id"],
                        "subject_text": decision["subject"]["surface"], "subject_kind": decision["subject"]["kind"],
                        "relation_type": decision["relation_type"], "object_mention_id": decision["object"]["mention_id"],
                        "object_text": decision["object"]["surface"], "object_kind": decision["object"]["kind"],
                        "negated": decision["negated"], "uncertain": decision["uncertain"], "modality": decision["modality"],
                        "evidence": decision["evidence"],
                    }
                )
        counts["propositions_with_positive_edge"] += int(row_positive)
        if index % 50 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)

    ids = fixed_review_ids(args)
    by_id = {str(row["proposition_id"]): row for row in rows}
    v4_path = args.v4_root / "semantic_frames_v4.jsonl"
    v4 = {str(row["proposition_id"]): row for row in read_jsonl(v4_path)} if v4_path.is_file() else {}
    review = []
    for proposition_id in ids:
        row = by_id[proposition_id]
        review.append(
            {
                "proposition_id": proposition_id, "source": row["source"], "proposition": row["claim"],
                "v4_frames": compact_v4(v4[proposition_id]) if proposition_id in v4 else [],
                "v5_pair_decisions": [compact_decision(value) for value in row["pair_decisions"]],
                "manual_gold": {
                    "eligible_relations": [], "missing_candidate_relations": [],
                    "entity_mentions_correct": None, "entity_normalization_correct": None, "notes": "",
                },
            }
        )
    positive_review = []
    for edge in balanced_positive_sample(edges, args.review_size, args.seed):
        positive_review.append(
            {
                **edge,
                "manual_audit": {"entity_pair_correct": None, "relation_correct": None, "direction_correct": None, "polarity_correct": None, "notes": ""},
            }
        )

    summary = {
        "run_version": RUN_VERSION,
        "scope": "Automatic schema/grounding metrics only; semantic precision and recall require the fixed manual audit files.",
        "input": {"propositions": counts["propositions"], "by_source": dict(Counter(row["source"] for row in rows))},
        "classification": {
            "parse_success": counts["parse_success"], "parse_success_rate": counts["parse_success"] / counts["propositions"],
            "propositions_with_candidate_pairs": counts["propositions_with_pairs"],
            "candidate_pair_proposition_coverage": counts["propositions_with_pairs"] / counts["propositions"],
            "pairs": counts["pairs"], "valid_classifications": counts["valid_classifications"],
            "valid_classification_rate": counts["valid_classifications"] / max(counts["pairs"], 1),
            "choice_counts": dict(choice_counts),
        },
        "positive_edges": {
            "edges": len(edges), "propositions": counts["propositions_with_positive_edge"],
            "proposition_coverage": counts["propositions_with_positive_edge"] / counts["propositions"],
            "relations": dict(relation_counts), "relations_by_source": {key: dict(value) for key, value in source_counts.items()},
        },
        "manual_evaluation": {
            "fixed_proposition_review": str((args.output_root / "fixed_proposition_review_200.jsonl").resolve()),
            "positive_edge_review": str((args.output_root / "positive_edge_review_200.jsonl").resolve()),
            "pilot_pass": {"exact_triple_precision": 0.75, "direction_accuracy": 0.90, "wrong_edge_rate_max": 0.10, "correct_edge_proposition_coverage": 0.20},
            "status": "pending_manual_audit",
        },
    }
    atomic_jsonl(args.output_root / "candidate_edges_v5.jsonl", edges)
    atomic_jsonl(args.output_root / "fixed_proposition_review_200.jsonl", review)
    atomic_jsonl(args.output_root / "positive_edge_review_200.jsonl", positive_review)
    atomic_json(args.output_root / "summary.json", summary)
    atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "summary": str((args.output_root / "summary.json").resolve())})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--v2-root", str(args.v2_root), "--v4-root", str(args.v4_root), "--output-root", str(args.output_root),
        "--qwen-model", str(args.qwen_model), "--llm-python", str(args.llm_python),
        "--generation-batch-size", str(args.generation_batch_size), "--generation-shard-size", str(args.generation_shard_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-model-len", str(args.max_model_len),
        "--max-output-tokens", str(args.max_output_tokens), "--max-pairs-per-proposition", str(args.max_pairs_per_proposition),
        "--review-size", str(args.review_size), "--seed", str(args.seed),
    ]


def run_worker(args: argparse.Namespace, mode: str, progress: WorkflowProgress, stage: int) -> None:
    environment = dict(os.environ)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    environment.setdefault("TRANSFORMERS_VERBOSITY", "error")
    process = subprocess.Popen(worker_command(args, mode), cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    started = False
    for line in process.stdout:
        value = line.rstrip()
        if value.startswith("@@PAIR_V5_PROGRESS@@"):
            event = json.loads(value[len("@@PAIR_V5_PROGRESS@@") :])
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
    if args.max_pairs_per_proposition <= 0 or args.generation_batch_size <= 0 or args.generation_shard_size <= 0:
        raise ValueError("Pair and batch limits must be positive")
    for path in (args.qwen_model, args.llm_python, args.v4_root / "review_sample_200.jsonl"):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = load_rows(args)
    pairs, pair_summary = materialize_pairs(args, rows)
    review_ids = fixed_review_ids(args)
    ensure_manifest(args, contract(args, rows, pair_summary, review_ids))
    atomic_jsonl(args.output_root / "candidate_pairs_v5.jsonl", pairs)
    prompt_audit = prompt_token_audit(args, pairs)
    atomic_json(args.output_root / "preflight_summary.json", {"candidate_pairs": pair_summary, "prompt_tokens": prompt_audit})

    progress = WorkflowProgress(
        ["freeze type-constrained pairs and audit prompts", "Qwen pair relation/NO_RELATION classification", "aggregate edges and fixed reviews"],
        [20, 420, 20],
    )
    progress.log(
        f"[workflow plan] propositions={len(rows):,} pairs={pair_summary['candidate_pairs']:,} "
        f"paired_propositions={pair_summary['propositions_with_pairs']:,} max_tokens={prompt_audit['max_prompt_plus_output_tokens']:,}"
    )
    progress.log("[changed variable] endpoints are fixed before inference; Qwen chooses only an allowed directed relation, NO_RELATION, or AMBIGUOUS")
    progress.start(1, len(rows), len(rows))
    progress.log(
        f"[stage 1/3 complete] pairs={pair_summary['candidate_pairs']:,} truncated_pairs={pair_summary['truncated_pairs']:,} "
        f"output={args.output_root / 'candidate_pairs_v5.jsonl'}"
    )
    if args.preflight_only:
        progress.close()
        progress.log("[preflight complete] source, pair schema, fixed review cohort, tokenizer length, manifest, and output contract passed; no model loaded")
        return
    run_worker(args, "classify", progress, 2)
    progress.log(f"[stage 2/3 complete] output={args.output_root / 'pair_classifications_v5.jsonl'}")
    run_worker(args, "aggregate", progress, 3)
    progress.log(f"[stage 3/3 complete] summary={args.output_root / 'summary.json'}")
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

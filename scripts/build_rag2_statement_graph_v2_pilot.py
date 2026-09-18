#!/usr/bin/env python3
"""Build a conservative Mention--Concept--Statement graph from 1,000 propositions.

This v2 pilot reuses the immutable propositions and mention/UMLS candidates from
the open-statement v1 pilot.  It deliberately separates:

* all source-grounded mentions (provenance layer),
* conservatively resolved canonical concepts (identity layer), and
* independently verified statements (semantic relation layer).

Qwen extracts and normalizes relations.  Llama verifies only whether the
resulting statement preserves the source proposition; neither model judges the
medical truth of the source.  All raw outputs and rejected cases are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

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
from build_rag2_open_statement_kg_pilot import (  # noqa: E402
    NODE_KINDS,
    RELATION_TYPES,
    parse_json_object,
)


RUN_VERSION = "rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_V1_ROOT = ROOT / "datasets/kg/rag2_open_statement_kg_pilot_1k_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_LLAMA = WORKSPACE / "models/Llama-3-8B-Instruct"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"

PROPOSITION_CLASSES = {"factual", "recommendation", "study_finding", "metadata", "incomplete"}
QUALIFIER_ROLES = {"population", "condition", "time", "dose", "amount", "comparison", "location", "other"}
MODALITIES = {"asserted", "may", "should", "must", "can", "reported"}

GENERIC_GLOBAL_SURFACES = {
    "a patient", "patient", "patients", "person", "persons", "people",
    "a study", "the study", "study", "studies", "the author", "author",
    "condition", "conditions", "effect", "effects", "level", "levels",
    "result", "results", "finding", "findings", "group", "groups",
    "treatment", "treatments", "procedure", "procedures", "disease",
    "disorder", "disorders", "symptom", "symptoms", "value", "values",
}

KIND_PRIORITY = {
    "condition": 9,
    "drug_or_chemical": 9,
    "procedure_or_intervention": 9,
    "anatomy": 9,
    "organism": 9,
    "gene_or_protein": 9,
    "finding_or_outcome": 8,
    "population": 8,
    "measurement": 8,
    "dose": 8,
    "time": 8,
    "numeric_literal": 8,
    "location": 7,
    "event": 6,
    "study_or_document": 5,
    "biomedical_concept": 4,
    "other_concept": 1,
}

UMLS_KIND_CATEGORIES = {
    "condition": {"DiseaseOrDisorder", "FindingOrSymptom"},
    "drug_or_chemical": {"DrugOrChemical"},
    "procedure_or_intervention": {"ProcedureOrTreatment", "DiagnosticTest"},
    "finding_or_outcome": {"FindingOrSymptom", "DiagnosticTest"},
    "anatomy": {"Anatomy"},
    "organism": {"Organism"},
    "gene_or_protein": {"GeneOrProtein"},
    "population": {"Organism", "OtherBiomedical"},
    "location": {"Anatomy", "OtherBiomedical"},
}

RELATION_DIRECTION = {
    "IS_A": "specific concept -> broader class",
    "PART_OF": "part -> whole",
    "LOCATED_IN": "located entity -> location",
    "ADMINISTERED_TO": "drug or intervention -> recipient population",
    "TREATS": "treatment -> condition",
    "MANAGES": "management intervention -> condition",
    "PREVENTS": "preventive intervention -> prevented condition",
    "CONTRAINDICATED_FOR": "intervention -> contraindicated condition or population",
    "CAUSES": "cause -> effect",
    "INCREASES_RISK_OF": "risk factor -> outcome",
    "DECREASES_RISK_OF": "protective factor -> outcome",
    "DIAGNOSES": "diagnostic procedure -> condition",
    "DETECTS": "test or procedure -> detected finding",
    "INDICATES": "finding -> indicated condition",
    "HAS_SIGN_OR_SYMPTOM": "condition -> sign or symptom",
    "APPLIES_TO_POPULATION": "statement subject -> population",
}

RELATION_SYSTEM = """Extract only semantic statements explicitly expressed by one atomic proposition.

Candidate mentions are aids, not mandatory outputs. Do not connect every mention pair. Return the main subject--relation--object assertion and any additional assertion explicitly stated by a conjunction or list. Preserve raw wording, negation, uncertainty, recommendation modality, and qualifiers. Use a supplied mention_id when it matches the exact argument. If no supplied mention covers an argument, use null and provide its exact contiguous text and node kind. Do not infer outside knowledge. A factual-looking proposition may legitimately have no extractable relation.

Do not calculate character offsets. Return JSON only:
{"status":"extracted|no_explicit_relation|metadata_only|incomplete","statements":[{"subject_mention_id":"ID or null","subject_text":"exact span","subject_kind":"ALLOWED_KIND","predicate_text":"exact contiguous span","object_mention_id":"ID or null","object_text":"exact span","object_kind":"ALLOWED_KIND","qualifiers":[{"role":"population|condition|time|dose|amount|comparison|location|other","mention_id":"ID or null","text":"exact span","kind":"ALLOWED_KIND"}],"negated":false,"uncertain":false,"modality":"asserted|may|should|must|can|reported"}]}"""

NORMALIZATION_SYSTEM = """Normalize source-grounded raw statements while preserving their meaning.

Choose an allowed relation type only when it fits. Use OTHER rather than forcing a broad or incorrect relation. Normalize direction according to the supplied direction conventions, not merely grammatical subject order. For example, 'A is caused by B' and 'A is a consequence of B' must become B CAUSES A; 'A is composed of B' must become B PART_OF A. Preserve the original raw predicate, negation, uncertainty, modality, and qualifiers outside this mapping.

Return JSON only:
{"mappings":[{"statement_id":"S1","relation_type":"ALLOWED_TYPE","direction":"preserved|reversed","mapping_status":"exact|other","confidence":0.0,"explanation":"brief source-based reason"}]}"""

VERIFIER_SYSTEM = """Verify whether each extracted statement faithfully preserves its source proposition.

Do not judge whether the medical source is true. Use no outside medical knowledge. Check only textual support: argument identity, relation direction, negation, uncertainty/modality, qualifiers, and whether the normalized relation changes meaning. For relation type OTHER, judge the raw predicate rather than requiring an ontology label. Reject a statement if it adds a causal, diagnostic, treatment, or other claim absent from the proposition.

Return JSON only:
{"checks":[{"statement_id":"S1","arguments_supported":true,"raw_relation_supported":true,"direction_correct":true,"negation_correct":true,"modality_correct":true,"qualifiers_supported":true,"normalized_relation_supported":true,"verdict":"verified|uncertain|rejected","issues":["brief issue"]}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "prepare", "relations", "normalize", "verify", "canonicalize", "aggregate"), default="orchestrate")
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-relation-output-tokens", type=int, default=1024)
    parser.add_argument("--max-normalization-output-tokens", type=int, default=768)
    parser.add_argument("--max-verifier-output-tokens", type=int, default=1024)
    parser.add_argument("--canonical-umls-threshold", type=float, default=0.90)
    parser.add_argument("--canonical-umls-min-margin", type=float, default=0.05)
    parser.add_argument("--canonical-lexical-overlap", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print("@@STATEMENT_GRAPH_PROGRESS@@" + canonical_json({"event": event, **payload}), flush=True)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def normalized_surface(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    value = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", value)
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    value = re.sub(r"[^a-z0-9%+./-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def lexical_tokens(value: str) -> set[str]:
    return {token for token in normalized_surface(value).split() if len(token) > 1}


def lexical_overlap(left: str, right: str) -> float:
    a, b = lexical_tokens(left), lexical_tokens(right)
    if not a or not b:
        return float(normalized_surface(left) == normalized_surface(right))
    return len(a & b) / min(len(a), len(b))


def all_exact_spans(text: str, value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    spans = [{"text": value, "start": match.start(), "end": match.end()} for match in re.finditer(re.escape(value), text)]
    if spans:
        return spans
    return [{"text": text[match.start():match.end()], "start": match.start(), "end": match.end()} for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE)]


def proposition_inputs(args: argparse.Namespace) -> Path:
    return args.v1_root / "normalized_nodes.jsonl"


def validate_source_contract(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = proposition_inputs(args)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = list(read_jsonl(source))
    ids = [str(row.get("proposition_id") or "") for row in rows]
    if len(rows) != 1000 or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Expected 1,000 unique v1 proposition rows; found rows={len(rows)} unique_ids={len(set(ids))}")
    by_source = dict(Counter(str(row.get("source") or "") for row in rows))
    if by_source != {"cpg": 250, "pmc": 250, "pubmed": 250, "textbooks": 250}:
        raise RuntimeError(f"Unexpected source cohort: {by_source}")
    metadata = {
        "propositions": len(rows),
        "by_source": by_source,
        "selection_sha256": sha256_text("\n".join(f"{row['proposition_id']}\t{sha256_text(row['claim'])}" for row in rows)),
    }
    return rows, metadata


def experiment_contract(args: argparse.Namespace, cohort: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "run_version": RUN_VERSION,
        "purpose": "Build a source-grounded Mention--Concept--Statement graph without forcing relations between co-occurring entities.",
        "scope": "Structural and source-fidelity pilot; source medical truth is assumed and not evaluated.",
        "input": file_identity(proposition_inputs(args)),
        "cohort": dict(cohort),
        "models": {"extractor_and_normalizer": str(args.qwen_model.resolve()), "source_fidelity_verifier": str(args.llama_model.resolve())},
        "prompts": {
            "relation_sha256": sha256_text(RELATION_SYSTEM),
            "normalization_sha256": sha256_text(NORMALIZATION_SYSTEM),
            "verifier_sha256": sha256_text(VERIFIER_SYSTEM),
        },
        "relation_types": list(RELATION_TYPES),
        "direction_conventions": RELATION_DIRECTION,
        "canonicalization": {
            "umls_threshold": args.canonical_umls_threshold,
            "umls_min_margin": args.canonical_umls_min_margin,
            "lexical_overlap": args.canonical_lexical_overlap,
            "fallback": "explicit acronym, then exact normalized surface plus compatible node kind; generic/event mentions stay local",
            "no_embedding_or_gold_labels": True,
        },
        "generation": {
            "temperature": 0.0,
            "batch_size": args.generation_batch_size,
            "shard_size": args.generation_shard_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_relation_output_tokens": args.max_relation_output_tokens,
            "max_normalization_output_tokens": args.max_normalization_output_tokens,
            "max_verifier_output_tokens": args.max_verifier_output_tokens,
        },
        "code": {
            "script": file_identity(Path(__file__).resolve()),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or None,
        },
        "seed": args.seed,
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
            raise RuntimeError("Statement Graph v2 resume contract mismatch; use a new output root")
    else:
        atomic_json(path, value)


def clean_mentions(row: Mapping[str, Any]) -> dict[str, Any]:
    groups: list[list[dict[str, Any]]] = []
    for source_node in sorted(row.get("nodes", []), key=lambda item: (int(item["start"]), int(item["end"]), -KIND_PRIORITY.get(str(item.get("kind")), 0))):
        node = dict(source_node)
        node["surface_normalized"] = normalized_surface(str(node["surface"]))
        found: list[dict[str, Any]] | None = None
        for group in groups:
            exemplar = group[0]
            overlaps = max(int(node["start"]), int(exemplar["start"])) < min(int(node["end"]), int(exemplar["end"]))
            same_span = int(node["start"]) == int(exemplar["start"]) and int(node["end"]) == int(exemplar["end"])
            same_surface = node["surface_normalized"] == exemplar["surface_normalized"]
            if same_span or (overlaps and same_surface):
                found = group
                break
        if found is None:
            groups.append([node])
        else:
            found.append(node)

    cleaned: list[dict[str, Any]] = []
    for group in groups:
        primary = max(
            group,
            key=lambda item: (
                KIND_PRIORITY.get(str(item.get("kind")), 0),
                int(item.get("umls_status") == "linked_confident"),
                -len(str(item.get("surface") or "")),
            ),
        )
        aliases = [
            {"source_node_id": item["node_id"], "surface": item["surface"], "start": item["start"], "end": item["end"], "kind": item["kind"]}
            for item in group
        ]
        cleaned.append({
            **primary,
            "mention_id": primary["node_id"],
            "source_node_ids": sorted({item["node_id"] for item in group}),
            "aliases": aliases,
            "surface_normalized": normalized_surface(str(primary["surface"])),
            "globally_resolvable": normalized_surface(str(primary["surface"])) not in GENERIC_GLOBAL_SURFACES and str(primary.get("kind")) not in {"event", "study_or_document"},
        })
    cleaned.sort(key=lambda item: (item["start"], item["end"], item["mention_id"]))
    return {**row, "mentions": cleaned, "node_cleaning": {"input_nodes": len(row.get("nodes", [])), "cleaned_mentions": len(cleaned), "collapsed_nodes": len(row.get("nodes", [])) - len(cleaned)}}


def prepare_worker(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(proposition_inputs(args)))
    output = args.output_root / "prepare_shards"
    cached = cached_rows(output, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="clean mention candidates and preserve aliases")
    done = len(cached)
    for start in range(0, len(rows), 100):
        batch = rows[start:start + 100]
        values = [clean_mentions(row) for row in batch if row["proposition_id"] not in cached]
        if values:
            atomic_jsonl(output / f"batch_{start:05d}_{start + len(batch):05d}.jsonl", values)
            for value in values:
                cached[value["proposition_id"]] = value
            done += len(values)
            emit("progress", total=len(rows), completed=done)
    if len(cached) != len(rows):
        raise RuntimeError(f"Mention preparation incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / "prepared_mentions.jsonl", (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def relation_prompt(row: Mapping[str, Any]) -> str:
    mentions = [
        {"mention_id": item["mention_id"], "text": item["surface"], "kind": item["kind"]}
        for item in row["mentions"]
    ]
    return (
        "ALLOWED NODE KINDS:\n" + ", ".join(NODE_KINDS)
        + "\n\nPROPOSITION CLASS:\n" + str(row.get("proposition_class") or "factual")
        + "\n\nPROPOSITION:\n" + row["claim"]
        + "\n\nMENTION CANDIDATES:\n" + json.dumps(mentions, ensure_ascii=False)
    )


def resolve_argument(
    row: dict[str, Any], mention_id: Any, text: Any, kind: Any, added: list[dict[str, Any]], role: str
) -> tuple[str | None, str | None]:
    value = str(text or "").strip()
    identifier = str(mention_id or "")
    mentions = list(row["mentions"]) + added
    by_id = {item["mention_id"]: item for item in mentions}
    if identifier in by_id and (not value or by_id[identifier]["surface"].casefold() == value.casefold()):
        return identifier, None
    exact = [item for item in mentions if value and item["surface"].casefold() == value.casefold()]
    if len(exact) == 1:
        return exact[0]["mention_id"], "id_repaired_from_text" if identifier else None
    spans = all_exact_spans(row["claim"], value)
    if len(spans) != 1:
        return None, f"{role}_text_not_unique_or_absent"
    node_kind = str(kind or "other_concept")
    if node_kind not in NODE_KINDS:
        node_kind = "other_concept"
    span = spans[0]
    new_id = f"{row['proposition_id']}::R{len(added)+1}"
    added.append({
        "mention_id": new_id,
        "node_id": new_id,
        "surface": span["text"],
        "surface_normalized": normalized_surface(span["text"]),
        "start": span["start"],
        "end": span["end"],
        "kind": node_kind,
        "origin": "relation_added",
        "source_node_ids": [],
        "aliases": [],
        "globally_resolvable": normalized_surface(span["text"]) not in GENERIC_GLOBAL_SURFACES and node_kind not in {"event", "study_or_document"},
        "umls_status": "not_evaluated",
        "umls_cui": None,
        "umls_name": None,
        "umls_score": None,
        "umls_candidates": [],
    })
    return new_id, "mention_added_from_exact_text"


def validate_relation_output(source_row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    row = dict(source_row)
    parsed = parse_json_object(raw)
    if parsed is None or not isinstance(parsed.get("statements"), list):
        return {**row, "relation_parse_success": False, "relation_status": "parse_failure", "statements": [], "relation_added_mentions": [], "relation_validation_errors": ["invalid_json_or_schema"], "relation_raw_output": raw}
    status = str(parsed.get("status") or "incomplete")
    if status not in {"extracted", "no_explicit_relation", "metadata_only", "incomplete"}:
        status = "incomplete"
    added: list[dict[str, Any]] = []
    statements: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for index, item in enumerate(parsed["statements"]):
        if not isinstance(item, dict):
            all_errors.append(f"statement_{index}:not_object")
            continue
        errors: list[str] = []
        repairs: list[str] = []
        subject, note = resolve_argument(row, item.get("subject_mention_id"), item.get("subject_text"), item.get("subject_kind"), added, "subject")
        if note:
            (errors if subject is None else repairs).append(note)
        obj, note = resolve_argument(row, item.get("object_mention_id"), item.get("object_text"), item.get("object_kind"), added, "object")
        if note:
            (errors if obj is None else repairs).append(note)
        predicate = str(item.get("predicate_text") or "").strip()
        predicate_spans = all_exact_spans(row["claim"], predicate)
        if not predicate_spans:
            errors.append("predicate_text_absent")
        if subject and obj and subject == obj:
            errors.append("self_relation")
        qualifiers: list[dict[str, Any]] = []
        seen_qualifiers: set[tuple[str, str]] = set()
        for q_index, qualifier in enumerate(item.get("qualifiers") or []):
            if not isinstance(qualifier, dict):
                errors.append(f"qualifier_{q_index}:not_object")
                continue
            qid, note = resolve_argument(row, qualifier.get("mention_id"), qualifier.get("text"), qualifier.get("kind"), added, f"qualifier_{q_index}")
            if note:
                (errors if qid is None else repairs).append(note)
            if qid:
                qrole = str(qualifier.get("role") or "other")
                qrole = qrole if qrole in QUALIFIER_ROLES else "other"
                if qid in {subject, obj}:
                    repairs.append(f"qualifier_{q_index}:duplicates_core_argument")
                    continue
                qualifier_key = (qrole, qid)
                if qualifier_key not in seen_qualifiers:
                    qualifiers.append({"role": qrole, "mention_id": qid})
                    seen_qualifiers.add(qualifier_key)
        modality = str(item.get("modality") or "asserted")
        if modality not in MODALITIES:
            modality = "asserted"
            repairs.append("invalid_modality_repaired")
        statement_id = f"{row['proposition_id']}::S{len(statements)+1}"
        statements.append({
            "statement_id": statement_id,
            "raw_subject_mention_id": subject,
            "raw_object_mention_id": obj,
            "predicate_text": predicate,
            "predicate_spans": predicate_spans,
            "qualifiers": qualifiers,
            "negated": bool(item.get("negated", False)),
            "uncertain": bool(item.get("uncertain", False)),
            "modality": modality,
            "structurally_valid": not errors and bool(subject) and bool(obj),
            "validation_errors": errors,
            "repairs": repairs,
        })
        all_errors.extend(f"statement_{index}:{error}" for error in errors)
    row["mentions"] = list(row["mentions"]) + added
    return {**row, "relation_parse_success": True, "relation_status": status, "statements": statements, "relation_added_mentions": added, "relation_validation_errors": all_errors, "relation_raw_output": raw}


def normalization_prompt(row: Mapping[str, Any]) -> str:
    mention_map = {item["mention_id"]: {"text": item["surface"], "kind": item["kind"]} for item in row["mentions"]}
    statements = []
    for item in row["statements"]:
        if not item["structurally_valid"]:
            continue
        statements.append({
            "statement_id": item["statement_id"],
            "raw_subject": mention_map[item["raw_subject_mention_id"]],
            "raw_predicate": item["predicate_text"],
            "raw_object": mention_map[item["raw_object_mention_id"]],
            "negated": item["negated"],
            "uncertain": item["uncertain"],
            "modality": item["modality"],
        })
    return (
        "ALLOWED RELATION TYPES:\n" + ", ".join(RELATION_TYPES)
        + "\n\nDIRECTION CONVENTIONS:\n" + json.dumps(RELATION_DIRECTION, ensure_ascii=False)
        + "\n\nPROPOSITION:\n" + row["claim"]
        + "\n\nRAW STATEMENTS:\n" + json.dumps(statements, ensure_ascii=False)
    )


def validate_normalization_output(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    valid = {item["statement_id"]: item for item in row["statements"] if item["structurally_valid"]}
    mappings: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    parse_success = parsed is not None and isinstance(parsed.get("mappings"), list)
    if parse_success:
        for index, item in enumerate(parsed["mappings"]):
            if not isinstance(item, dict):
                errors.append(f"mapping_{index}:not_object")
                continue
            statement_id = str(item.get("statement_id") or "")
            if statement_id not in valid:
                errors.append(f"mapping_{index}:unknown_statement")
                continue
            relation_type = str(item.get("relation_type") or "OTHER").upper()
            if relation_type not in RELATION_TYPES:
                relation_type = "OTHER"
                errors.append(f"mapping_{index}:invalid_relation_type")
            direction = str(item.get("direction") or "preserved")
            if direction not in {"preserved", "reversed"}:
                direction = "preserved"
                errors.append(f"mapping_{index}:invalid_direction")
            mapping_status = str(item.get("mapping_status") or "other")
            if mapping_status not in {"exact", "other"}:
                mapping_status = "other"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            mappings[statement_id] = {
                "relation_type": relation_type,
                "direction": direction,
                "mapping_status": mapping_status,
                "normalization_confidence": confidence,
                "normalization_explanation": str(item.get("explanation") or ""),
            }
    normalized: list[dict[str, Any]] = []
    for statement in row["statements"]:
        mapping = mappings.get(statement["statement_id"])
        if mapping is None:
            mapping = {
                "relation_type": "OTHER",
                "direction": "preserved",
                "mapping_status": "missing_mapping" if statement["structurally_valid"] else "invalid_statement",
                "normalization_confidence": None,
                "normalization_explanation": "",
            }
            if statement["structurally_valid"]:
                errors.append(f"{statement['statement_id']}:missing_mapping")
        subject = statement["raw_subject_mention_id"]
        obj = statement["raw_object_mention_id"]
        if mapping["direction"] == "reversed":
            subject, obj = obj, subject
        normalized.append({
            **statement,
            **mapping,
            "normalized_subject_mention_id": subject,
            "normalized_object_mention_id": obj,
        })
    return {**row, "normalization_parse_success": parse_success, "normalization_errors": errors if parse_success else ["invalid_json_or_schema"], "statements": normalized, "normalization_raw_output": raw}


def verifier_prompt(row: Mapping[str, Any]) -> str:
    mention_map = {item["mention_id"]: {"text": item["surface"], "kind": item["kind"]} for item in row["mentions"]}
    statements = []
    for item in row["statements"]:
        if not item["structurally_valid"]:
            continue
        statements.append({
            "statement_id": item["statement_id"],
            "raw_subject": mention_map[item["raw_subject_mention_id"]],
            "raw_predicate": item["predicate_text"],
            "raw_object": mention_map[item["raw_object_mention_id"]],
            "normalized_subject": mention_map[item["normalized_subject_mention_id"]],
            "normalized_relation": item["relation_type"],
            "normalized_object": mention_map[item["normalized_object_mention_id"]],
            "qualifiers": [{"role": q["role"], **mention_map[q["mention_id"]]} for q in item["qualifiers"]],
            "negated": item["negated"],
            "uncertain": item["uncertain"],
            "modality": item["modality"],
        })
    return "SOURCE PROPOSITION:\n" + row["claim"] + "\n\nSTATEMENTS TO VERIFY:\n" + json.dumps(statements, ensure_ascii=False)


VERIFY_FIELDS = (
    "arguments_supported", "raw_relation_supported", "direction_correct", "negation_correct",
    "modality_correct", "qualifiers_supported", "normalized_relation_supported",
)


def validate_verifier_output(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    valid = {item["statement_id"]: item for item in row["statements"] if item["structurally_valid"]}
    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    parse_success = parsed is not None and isinstance(parsed.get("checks"), list)
    if parse_success:
        for index, item in enumerate(parsed["checks"]):
            if not isinstance(item, dict):
                errors.append(f"check_{index}:not_object")
                continue
            statement_id = str(item.get("statement_id") or "")
            if statement_id not in valid:
                errors.append(f"check_{index}:unknown_statement")
                continue
            flags = {field: bool(item.get(field, False)) for field in VERIFY_FIELDS}
            verdict = str(item.get("verdict") or "uncertain")
            if verdict not in {"verified", "uncertain", "rejected"}:
                verdict = "uncertain"
            accepted = verdict == "verified" and all(flags.values())
            checks[statement_id] = {**flags, "verifier_verdict": verdict, "verifier_issues": [str(x) for x in item.get("issues", [])], "accepted": accepted}
    checked: list[dict[str, Any]] = []
    for statement in row["statements"]:
        check = checks.get(statement["statement_id"])
        if check is None:
            check = {**{field: False for field in VERIFY_FIELDS}, "verifier_verdict": "not_checked" if not statement["structurally_valid"] else "missing_check", "verifier_issues": [], "accepted": False}
            if statement["structurally_valid"]:
                errors.append(f"{statement['statement_id']}:missing_check")
        checked.append({**statement, **check})
    return {**row, "verifier_parse_success": parse_success, "verifier_errors": errors if parse_success else ["invalid_json_or_schema"], "statements": checked, "verifier_raw_output": raw}


def generate_worker(args: argparse.Namespace, mode: str) -> None:
    from vllm import SamplingParams

    if mode == "relations":
        model, system = args.qwen_model, RELATION_SYSTEM
        input_path, output_name, shard_name = args.output_root / "prepared_mentions.jsonl", "raw_statements_v2.jsonl", "relation_shards"
        prompt_fn, validator, max_tokens = relation_prompt, validate_relation_output, args.max_relation_output_tokens
    elif mode == "normalize":
        model, system = args.qwen_model, NORMALIZATION_SYSTEM
        input_path, output_name, shard_name = args.output_root / "raw_statements_v2.jsonl", "normalized_statements_v2.jsonl", "normalization_shards"
        prompt_fn, validator, max_tokens = normalization_prompt, validate_normalization_output, args.max_normalization_output_tokens
    elif mode == "verify":
        model, system = args.llama_model, VERIFIER_SYSTEM
        input_path, output_name, shard_name = args.output_root / "normalized_statements_v2.jsonl", "verified_statements_v2.jsonl", "verifier_shards"
        prompt_fn, validator, max_tokens = verifier_prompt, validate_verifier_output, args.max_verifier_output_tokens
    else:
        raise AssertionError(mode)

    rows = list(read_jsonl(input_path))
    output = args.output_root / shard_name
    cached = cached_rows(output, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name=mode)
    missing_count = sum(row["proposition_id"] not in cached for row in rows)
    if missing_count:
        tokenizer, llm = init_vllm(model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        done = len(cached)
        for start in range(0, len(rows), args.generation_shard_size):
            batch = rows[start:start + args.generation_shard_size]
            missing = [row for row in batch if row["proposition_id"] not in cached]
            if not missing:
                continue
            prompts = [chat_prompt(tokenizer, system, prompt_fn(row)) for row in missing]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            values = []
            for row, generated in zip(missing, outputs):
                raw = generated.outputs[0].text.strip() if generated.outputs else ""
                values.append(validator(row, raw))
            atomic_jsonl(output / f"batch_{start:05d}_{start + len(batch):05d}.jsonl", values)
            for value in values:
                cached[value["proposition_id"]] = value
            done += len(values)
            emit("progress", total=len(rows), completed=done)
    if len(cached) != len(rows):
        raise RuntimeError(f"{mode} incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / output_name, (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def acronym_alias_map(row: Mapping[str, Any]) -> dict[str, str]:
    mentions = row["mentions"]
    aliases: dict[str, str] = {}
    for short in mentions:
        surface = str(short["surface"])
        if not re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", surface):
            continue
        left = row["claim"][:int(short["start"])]
        if not re.search(r"\(\s*$", left):
            continue
        candidates = [
            item for item in mentions
            if int(item["end"]) <= int(short["start"]) and int(short["start"]) - int(item["end"]) <= 3
            and normalized_surface(str(item["surface"])) != normalized_surface(surface)
        ]
        if candidates:
            long = max(candidates, key=lambda item: len(str(item["surface"])))
            aliases[short["mention_id"]] = long["mention_id"]
    return aliases


def umls_compatible(mention: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if mention.get("umls_status") != "linked_confident" or not mention.get("umls_cui"):
        return False
    try:
        score = float(mention.get("umls_score") or 0.0)
    except (TypeError, ValueError):
        return False
    if score < args.canonical_umls_threshold:
        return False
    candidates = list(mention.get("umls_candidates") or [])
    if len(candidates) > 1 and float(candidates[0].get("score") or 0.0) - float(candidates[1].get("score") or 0.0) < args.canonical_umls_min_margin:
        return False
    allowed = UMLS_KIND_CATEGORIES.get(str(mention.get("kind")))
    top_category = str(candidates[0].get("category") or "") if candidates else ""
    if allowed is None or top_category not in allowed:
        return False
    return lexical_overlap(str(mention["surface"]), str(mention.get("umls_name") or "")) >= args.canonical_lexical_overlap


def local_concept_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:20]
    return f"LOCAL:{kind}:{digest}"


def resolve_concepts(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    alias_map = acronym_alias_map(row)
    by_id = {item["mention_id"]: item for item in row["mentions"]}
    resolved: list[dict[str, Any]] = []
    assigned: dict[str, tuple[str, str]] = {}
    for mention in row["mentions"]:
        mention_id = mention["mention_id"]
        long_id = alias_map.get(mention_id)
        if long_id and long_id in assigned:
            concept_id, method = assigned[long_id][0], "explicit_acronym"
        elif umls_compatible(mention, args):
            concept_id, method = f"UMLS:{mention['umls_cui']}", "high_precision_umls"
        elif bool(mention.get("globally_resolvable")):
            key = normalized_surface(str(mention["surface"]))
            concept_id, method = local_concept_id(str(mention["kind"]), key), "exact_typed_surface"
        else:
            concept_id, method = f"MENTION:{hashlib.sha256(mention_id.encode()).hexdigest()[:20]}", "mention_local"
        assigned[mention_id] = (concept_id, method)
        resolved.append({**mention, "canonical_concept_id": concept_id, "canonicalization_method": method})
    # Acronyms can precede a resolved long form only in unusual input order; repair deterministically.
    resolved_by_id = {item["mention_id"]: item for item in resolved}
    for short_id, long_id in alias_map.items():
        if short_id in resolved_by_id and long_id in resolved_by_id:
            resolved_by_id[short_id]["canonical_concept_id"] = resolved_by_id[long_id]["canonical_concept_id"]
            resolved_by_id[short_id]["canonicalization_method"] = "explicit_acronym"
    return {**row, "mentions": [resolved_by_id[item["mention_id"]] for item in row["mentions"]]}


def canonicalize_worker(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.output_root / "verified_statements_v2.jsonl"))
    output = args.output_root / "canonicalization_shards"
    cached = cached_rows(output, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="resolve mentions to conservative canonical concepts")
    done = len(cached)
    for start in range(0, len(rows), 100):
        batch = rows[start:start + 100]
        values = [resolve_concepts(row, args) for row in batch if row["proposition_id"] not in cached]
        if values:
            atomic_jsonl(output / f"batch_{start:05d}_{start + len(batch):05d}.jsonl", values)
            for value in values:
                cached[value["proposition_id"]] = value
            done += len(values)
            emit("progress", total=len(rows), completed=done)
    if len(cached) != len(rows):
        raise RuntimeError(f"Canonicalization incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / "canonicalized_statements_v2.jsonl", (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.output_root / "canonicalized_statements_v2.jsonl"))
    emit("start", total=len(rows), completed=0, name="write provenance and verified semantic graphs")
    mentions: list[dict[str, Any]] = []
    mention_edges: list[dict[str, Any]] = []
    concept_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_statements: list[dict[str, Any]] = []
    semantic_edges: list[dict[str, Any]] = []
    qualifier_edges: list[dict[str, Any]] = []
    counts = Counter()
    relation_types = Counter()
    canonical_methods = Counter()
    diagnostics: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, row in enumerate(rows, start=1):
        counts["propositions"] += 1
        counts["input_nodes"] += int(row["node_cleaning"]["input_nodes"])
        counts["cleaned_mentions"] += len(row["mentions"])
        counts["collapsed_nodes"] += int(row["node_cleaning"]["collapsed_nodes"])
        counts["relation_parse_success"] += int(row["relation_parse_success"])
        counts["normalization_parse_success"] += int(row["normalization_parse_success"])
        counts["verifier_parse_success"] += int(row["verifier_parse_success"])
        counts["raw_statements"] += len(row["statements"])
        counts["structurally_valid"] += sum(item["structurally_valid"] for item in row["statements"])
        counts["verified_statements"] += sum(item["accepted"] for item in row["statements"])
        counts["propositions_with_verified"] += int(any(item["accepted"] for item in row["statements"]))

        mention_map: dict[str, dict[str, Any]] = {}
        for mention in row["mentions"]:
            record = {
                "proposition_id": row["proposition_id"], "stable_id": row["stable_id"], "source": row["source"],
                **mention,
            }
            mentions.append(record)
            mention_map[mention["mention_id"]] = record
            concept_members[mention["canonical_concept_id"]].append(record)
            canonical_methods[mention["canonicalization_method"]] += 1
            mention_edges.append({"source_id": row["proposition_id"], "edge_type": "MENTIONS", "target_id": mention["mention_id"]})
            mention_edges.append({"source_id": mention["mention_id"], "edge_type": "DENOTES", "target_id": mention["canonical_concept_id"]})

        for statement in row["statements"]:
            if not statement["accepted"]:
                if len(diagnostics["rejected_statements"]) < 200 and statement["structurally_valid"]:
                    diagnostics["rejected_statements"].append({"claim": row["claim"], "source": row["source"], "statement": statement, "mentions": row["mentions"]})
                continue
            subject = mention_map[statement["normalized_subject_mention_id"]]["canonical_concept_id"]
            obj = mention_map[statement["normalized_object_mention_id"]]["canonical_concept_id"]
            record = {
                "proposition_id": row["proposition_id"], "stable_id": row["stable_id"], "source": row["source"],
                "proposition": row["claim"], **statement,
                "subject_concept_id": subject, "object_concept_id": obj,
            }
            verified_statements.append(record)
            relation_types[statement["relation_type"]] += 1
            counts["direction_reversed"] += int(statement["direction"] == "reversed")
            semantic_edges.append({
                "subject_concept_id": subject, "relation_type": statement["relation_type"],
                "raw_relation": statement["predicate_text"], "object_concept_id": obj,
                "statement_id": statement["statement_id"], "proposition_id": row["proposition_id"],
                "negated": statement["negated"], "uncertain": statement["uncertain"], "modality": statement["modality"],
            })
            for qualifier in statement["qualifiers"]:
                target = mention_map[qualifier["mention_id"]]["canonical_concept_id"]
                qualifier_edges.append({"statement_id": statement["statement_id"], "role": qualifier["role"], "target_concept_id": target})
            if statement["relation_type"] == "OTHER" and len(diagnostics["verified_other"]) < 200:
                diagnostics["verified_other"].append(record)
            if statement["direction"] == "reversed" and len(diagnostics["direction_reversed"]) < 200:
                diagnostics["direction_reversed"].append(record)

        if index % 50 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)

    concepts: list[dict[str, Any]] = []
    for concept_id, values in sorted(concept_members.items()):
        surfaces = Counter(value["surface"] for value in values)
        proposition_ids = {value["proposition_id"] for value in values}
        concepts.append({
            "concept_id": concept_id,
            "preferred_name": surfaces.most_common(1)[0][0],
            "kinds": sorted({value["kind"] for value in values}),
            "surfaces": [surface for surface, _ in surfaces.most_common()],
            "mention_count": len(values),
            "proposition_count": len(proposition_ids),
            "sources": sorted({value["source"] for value in values}),
            "cross_proposition": len(proposition_ids) > 1,
        })

    graph = args.output_root / "graph"
    atomic_jsonl(graph / "mentions.jsonl", mentions)
    atomic_jsonl(graph / "concepts.jsonl", concepts)
    atomic_jsonl(graph / "statements.jsonl", verified_statements)
    atomic_jsonl(graph / "semantic_edges.jsonl", semantic_edges)
    atomic_jsonl(graph / "qualifier_edges.jsonl", qualifier_edges)
    atomic_jsonl(graph / "provenance_edges.jsonl", mention_edges)
    diagnostic_root = args.output_root / "diagnostics"
    for name, values in diagnostics.items():
        atomic_jsonl(diagnostic_root / f"{name}.jsonl", values)

    cross = sum(item["cross_proposition"] for item in concepts)
    summary = {
        "run_version": RUN_VERSION,
        "scope": "Automated source-fidelity and structural audit; source medical truth and human-gold KG precision are not evaluated.",
        "sample": {"propositions": len(rows), "by_source": dict(Counter(row["source"] for row in rows))},
        "mention_cleaning": {
            "input_nodes": counts["input_nodes"], "cleaned_mentions": counts["cleaned_mentions"],
            "collapsed_nodes": counts["collapsed_nodes"], "collapse_rate": ratio(counts["collapsed_nodes"], counts["input_nodes"]),
        },
        "relation_extraction": {
            "parse_success_rate": ratio(counts["relation_parse_success"], len(rows)),
            "raw_statements": counts["raw_statements"], "structurally_valid_statements": counts["structurally_valid"],
            "structural_validity_rate": ratio(counts["structurally_valid"], counts["raw_statements"]),
        },
        "relation_normalization": {
            "parse_success_rate": ratio(counts["normalization_parse_success"], len(rows)),
            "direction_reversed_verified": counts["direction_reversed"],
        },
        "source_fidelity_verification": {
            "parse_success_rate": ratio(counts["verifier_parse_success"], len(rows)),
            "verified_statements": counts["verified_statements"],
            "verified_rate_among_structurally_valid": ratio(counts["verified_statements"], counts["structurally_valid"]),
            "proposition_verified_statement_coverage": ratio(counts["propositions_with_verified"], len(rows)),
        },
        "canonicalization": {
            "methods": dict(canonical_methods), "canonical_concepts": len(concepts),
            "mention_to_concept_reduction": 1.0 - ratio(len(concepts), len(mentions)),
            "cross_proposition_concepts": cross,
        },
        "verified_relation_types": dict(relation_types.most_common()),
        "graph": {
            "mentions": len(mentions), "concepts": len(concepts), "verified_statements": len(verified_statements),
            "semantic_edges": len(semantic_edges), "qualifier_edges": len(qualifier_edges), "provenance_edges": len(mention_edges),
        },
        "interpretation_limits": [
            "Llama agreement is a source-fidelity proxy, not human-gold semantic accuracy.",
            "Conservative exact/UMLS/acronym canonicalization favors precision and leaves unresolved synonyms separate.",
            "Nodes not used by a verified statement remain in the provenance mention layer and are not semantic facts.",
        ],
    }
    atomic_json(args.output_root / "summary.json", summary)
    atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "summary": str((args.output_root / "summary.json").resolve())})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--v1-root", str(args.v1_root), "--output-root", str(args.output_root),
        "--qwen-model", str(args.qwen_model), "--llama-model", str(args.llama_model),
        "--llm-python", str(args.llm_python), "--generation-batch-size", str(args.generation_batch_size),
        "--generation-shard-size", str(args.generation_shard_size), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len), "--max-relation-output-tokens", str(args.max_relation_output_tokens),
        "--max-normalization-output-tokens", str(args.max_normalization_output_tokens),
        "--max-verifier-output-tokens", str(args.max_verifier_output_tokens),
        "--canonical-umls-threshold", str(args.canonical_umls_threshold),
        "--canonical-umls-min-margin", str(args.canonical_umls_min_margin),
        "--canonical-lexical-overlap", str(args.canonical_lexical_overlap), "--seed", str(args.seed),
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
                if event["event"] == "complete":
                    progress.render(True)
        elif text and not text.startswith("Loading weights:"):
            progress.log(text)
    code = process.wait()
    if code:
        raise RuntimeError(f"Worker {mode} failed with exit code {code}")


def orchestrate(args: argparse.Namespace) -> None:
    for path in (proposition_inputs(args), args.qwen_model, args.llama_model, args.llm_python):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.generation_batch_size <= 0 or args.generation_shard_size <= 0:
        raise ValueError("Generation batch and shard sizes must be positive")
    if not 0 < args.canonical_umls_threshold <= 1 or not 0 <= args.canonical_umls_min_margin <= 1:
        raise ValueError("Invalid UMLS canonicalization thresholds")
    rows, cohort = validate_source_contract(args)
    ensure_manifest(args, experiment_contract(args, cohort))
    names = [
        "clean v1 mentions and preserve provenance",
        "extract explicit raw statements with Qwen",
        "normalize relation type and direction with Qwen",
        "verify source fidelity with frozen Llama",
        "resolve conservative canonical concepts",
        "write Mention--Concept--Statement graphs and diagnostics",
    ]
    progress = WorkflowProgress(names, [30, 720, 540, 600, 45, 75])
    progress.log("[workflow plan] reuse 1,000 v1 propositions -> clean mentions -> explicit statements -> direction-aware normalization -> independent source-fidelity verification -> dual-layer graph")
    progress.log("[scope] no QA/gold/semantic labels; all mentions retained, only verified statements enter the semantic graph")
    plan = [("prepare", 1, "prepared_mentions.jsonl")]
    if not args.preflight_only:
        plan.extend([
            ("relations", 2, "raw_statements_v2.jsonl"),
            ("normalize", 3, "normalized_statements_v2.jsonl"),
            ("verify", 4, "verified_statements_v2.jsonl"),
            ("canonicalize", 5, "canonicalized_statements_v2.jsonl"),
            ("aggregate", 6, "summary.json"),
        ])
    for mode, stage, output in plan:
        run_worker(args, mode, progress, stage)
        progress.log(f"[stage {stage}/6 complete] output={args.output_root / output}")
    progress.close()
    if args.preflight_only:
        progress.log(f"[preflight complete] cohort={len(rows):,}; cache contract and deterministic mention cleaning passed; no model loaded")
    else:
        progress.log(f"[workflow complete] summary={args.output_root / 'summary.json'} graph={args.output_root / 'graph'} diagnostics={args.output_root / 'diagnostics'}")


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate":
        orchestrate(args)
    elif args.mode == "prepare":
        prepare_worker(args)
    elif args.mode in {"relations", "normalize", "verify"}:
        generate_worker(args, args.mode)
    elif args.mode == "canonicalize":
        canonicalize_worker(args)
    elif args.mode == "aggregate":
        aggregate_worker(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

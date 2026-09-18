#!/usr/bin/env python3
"""Extract ontology-grounded semantic frames from 1,000 standalone propositions.

Unlike v2/v3, this pilot does not expose grammatical raw triples to Qwen.  The
model receives the proposition and all grounded mention candidates, interprets
the asserted meaning, and emits canonical semantic roles directly.  Code only
checks source spans, IDs, schema, and soft ontology type compatibility.  No
Llama verifier or QA/gold information is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
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
from build_rag2_open_statement_kg_pilot import NODE_KINDS, RELATION_TYPES, parse_json_object  # noqa: E402
from build_rag2_statement_graph_v2_pilot import all_exact_spans, emit, normalized_surface  # noqa: E402
from evaluate_rag2_statement_graph_canonical_direction_v3 import (  # noqa: E402
    CANONICAL_DIRECTIONS,
    SYMMETRIC_RELATIONS,
)


RUN_VERSION = "rag2_statement_graph_semantic_frames_pilot_1k_v4"
DEFAULT_V2_ROOT = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_V3_ROOT = ROOT / "datasets/kg/rag2_statement_graph_canonical_direction_pilot_1k_v3"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_semantic_frames_pilot_1k_v4"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"

QUALIFIER_ROLES = {"population", "condition", "time", "dose", "amount", "comparison", "location", "provenance", "other"}
MODALITIES = {"asserted", "may", "should", "must", "can", "reported"}

# These are soft diagnostic expectations, not hard filters, because upstream
# mention types are themselves imperfect.
ROLE_KIND_EXPECTATIONS: dict[str, tuple[set[str], set[str]]] = {
    "ADMINISTERED_TO": ({"drug_or_chemical", "procedure_or_intervention"}, {"population", "organism"}),
    "TREATS": ({"drug_or_chemical", "procedure_or_intervention"}, {"condition"}),
    "MANAGES": ({"drug_or_chemical", "procedure_or_intervention"}, {"condition", "finding_or_outcome"}),
    "CONTRAINDICATED_FOR": ({"drug_or_chemical", "procedure_or_intervention"}, {"condition", "population"}),
    "DIAGNOSES": ({"procedure_or_intervention"}, {"condition"}),
    "DETECTS": ({"procedure_or_intervention"}, {"condition", "finding_or_outcome", "measurement"}),
    "INDICATES": ({"finding_or_outcome", "measurement"}, {"condition"}),
    "HAS_SIGN_OR_SYMPTOM": ({"condition"}, {"finding_or_outcome"}),
    "HAS_DOSE": ({"drug_or_chemical", "procedure_or_intervention"}, {"dose", "numeric_literal"}),
    "HAS_DURATION": ({"event", "condition", "procedure_or_intervention"}, {"time", "numeric_literal"}),
}

ONTOLOGY_GUIDE = {
    relation: {
        "direction": CANONICAL_DIRECTIONS.get(relation, "interpret the semantic roles explicitly; do not follow grammatical order blindly"),
        "symmetric": relation in SYMMETRIC_RELATIONS,
    }
    for relation in RELATION_TYPES if relation != "OTHER"
}

FRAME_SYSTEM = """Convert one standalone proposition into the semantic fact frames it actually asserts.

Interpret meaning, not grammatical dependency order. Candidate mentions are grounding anchors, not a command to connect adjacent nouns. Select the entities that fill the ontology's semantic roles. A grammatical subject such as 'patients', 'symptoms', 'study', 'history', or 'figure' must not replace the underlying medical subject or compared items when the proposition's meaning identifies them. Study/document entities may instead be provenance qualifiers.

Examples:
1) 'The main symptoms of Sjogren syndrome are xerostomia.'
   Sjogren syndrome HAS_SIGN_OR_SYMPTOM xerostomia; 'main symptoms' is not the semantic subject.
2) 'Penicillin-allergic patients can receive erythromycin.'
   erythromycin ADMINISTERED_TO penicillin-allergic patients.
3) 'A review compared treatment A with treatment B.'
   treatment A COMPARED_WITH treatment B; the review is provenance, not a compared item.
4) 'Intestinal pseudoobstruction causes marked delays in small-bowel transit.'
   intestinal pseudoobstruction CAUSES marked delays; small-bowel transit is a condition/location qualifier.

Use only meaning entailed by the source. Preserve negation, uncertainty, modality, and qualifiers. Emit multiple frames only when the proposition explicitly asserts multiple facts. Use an existing mention ID whenever it denotes the intended argument. If no candidate covers an argument, set the ID to null and provide one exact contiguous source span plus an allowed node kind. Do not invent entities or medical facts. Use OTHER only when no allowed relation preserves the meaning; OTHER frames will not enter the semantic edge layer. Return 'ambiguous' instead of guessing.

Return JSON only:
{"status":"frames|no_semantic_frame|metadata_only|ambiguous","frames":[{"asserted_fact":"short canonical paraphrase entailed by the source","subject_mention_id":"ID or null","subject_text":"exact source span","subject_kind":"ALLOWED_KIND","relation_type":"ALLOWED_TYPE","object_mention_id":"ID or null","object_text":"exact source span","object_kind":"ALLOWED_KIND","qualifiers":[{"role":"population|condition|time|dose|amount|comparison|location|provenance|other","mention_id":"ID or null","text":"exact source span","kind":"ALLOWED_KIND"}],"evidence_spans":["exact contiguous source span"],"negated":false,"uncertain":false,"modality":"asserted|may|should|must|can|reported","frame_status":"grounded|ambiguous"}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "extract", "aggregate"), default="orchestrate")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument("--review-per-source", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_source(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = args.v2_root / "raw_statements_v2.jsonl"
    if not source.is_file() or not (args.v2_root / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"Completed v2 source required: {source}")
    rows = list(read_jsonl(source))
    ids = [str(row.get("proposition_id") or "") for row in rows]
    if len(rows) != 1000 or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Expected 1,000 unique v2 rows; found rows={len(rows)} unique={len(set(ids))}")
    sources = Counter(str(row.get("source") or "") for row in rows)
    if sources != {"cpg": 250, "pmc": 250, "pubmed": 250, "textbooks": 250}:
        raise RuntimeError(f"Unexpected source cohort: {dict(sources)}")
    return rows


def review_ids(rows: Sequence[Mapping[str, Any]], per_source: int, seed: int) -> list[str]:
    result: list[str] = []
    for source in ("cpg", "pmc", "pubmed", "textbooks"):
        candidates = [row for row in rows if row["source"] == source]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}|{row['proposition_id']}".encode()).hexdigest())
        result.extend(str(row["proposition_id"]) for row in candidates[:per_source])
    return result


def experiment_contract(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = review_ids(rows, args.review_per_source, args.seed)
    value = {
        "run_version": RUN_VERSION,
        "hypothesis": "Ontology-aware semantic-frame interpretation selects more accurate relation arguments than grammatical raw triples.",
        "changed_variable": "semantic-frame prompt/schema; the v2 proposition and mention inventory are unchanged",
        "input": content_identity(args.v2_root / "raw_statements_v2.jsonl"),
        "cohort": {
            "propositions": len(rows), "by_source": dict(Counter(row["source"] for row in rows)),
            "review_sample": len(selected), "review_selection_sha256": sha256_text("\n".join(selected)),
        },
        "model": str(args.qwen_model.resolve()),
        "prompt_sha256": sha256_text(FRAME_SYSTEM),
        "relation_ontology": ONTOLOGY_GUIDE,
        "node_kinds": list(NODE_KINDS),
        "evaluation": {
            "primary_metric": "manual exact semantic-frame precision on the fixed 200-proposition review sample",
            "frame_correct_if": "subject, relation, object, source entailment, and modifiers are all correct",
            "success_criterion": {"precision": 0.85, "usable_edge_coverage": 0.60, "direction_error_rate": 0.05},
            "automatic_status": "pending_manual_audit",
        },
        "generation": {
            "temperature": 0.0, "batch_size": args.generation_batch_size, "shard_size": args.generation_shard_size,
            "gpu_memory_utilization": args.gpu_memory_utilization, "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens, "seed": args.seed,
        },
        "code": {"script": content_identity(Path(__file__).resolve())},
        "uses_v2_raw_relation_as_model_input": False,
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
            durable_outputs = (
                args.output_root / "semantic_frame_shards",
                args.output_root / "semantic_frames_v4.jsonl",
                args.output_root / "summary.json",
                args.output_root / "COMPLETE.json",
            )
            if any(output.exists() for output in durable_outputs):
                raise RuntimeError("Semantic-frame v4 resume contract mismatch; use a new output root")
            # A prior preflight may have written only a manifest. With no model
            # outputs to reinterpret, replacing that manifest is safe.
            atomic_json(path, value)
    else:
        atomic_json(path, value)


def semantic_frame_prompt(row: Mapping[str, Any]) -> str:
    candidates = [
        {
            "mention_id": item["mention_id"], "text": item["surface"], "kind": item["kind"],
            "start": item["start"], "end": item["end"],
        }
        for item in row["mentions"]
    ]
    return (
        "ALLOWED NODE KINDS:\n" + ", ".join(NODE_KINDS)
        + "\n\nRELATION ONTOLOGY AND SEMANTIC ROLES:\n" + json.dumps(ONTOLOGY_GUIDE, ensure_ascii=False)
        + "\n\nSOURCE PROPOSITION:\n" + row["claim"]
        + "\n\nGROUNDING CANDIDATES:\n" + json.dumps(candidates, ensure_ascii=False)
    )


def resolve_argument(
    row: Mapping[str, Any], mention_id: Any, text: Any, kind: Any,
    added: list[dict[str, Any]], label: str,
) -> tuple[str | None, str | None]:
    candidates = list(row["mentions"]) + added
    by_id = {item["mention_id"]: item for item in candidates}
    mid = str(mention_id or "")
    value = str(text or "").strip()
    if mid in by_id:
        if value and normalized_surface(value) != normalized_surface(str(by_id[mid]["surface"])):
            return None, f"{label}:id_text_mismatch"
        return mid, None
    if not value:
        return None, f"{label}:missing_id_and_text"
    exact = [item for item in candidates if normalized_surface(str(item["surface"])) == normalized_surface(value)]
    if len(exact) == 1:
        return str(exact[0]["mention_id"]), f"{label}:id_repaired_from_text"
    spans = all_exact_spans(str(row["claim"]), value)
    if len(spans) != 1:
        return None, f"{label}:text_not_unique_or_absent"
    node_kind = str(kind or "other_concept")
    if node_kind not in NODE_KINDS:
        node_kind = "other_concept"
    new_id = f"{row['proposition_id']}::SF{len(added)+1}"
    mention = {
        "mention_id": new_id, "node_id": new_id, "surface": spans[0]["text"], "start": spans[0]["start"], "end": spans[0]["end"],
        "kind": node_kind, "origin": "semantic_frame_added", "globally_resolvable": node_kind not in {"event", "other_concept"},
        "surface_normalized": normalized_surface(spans[0]["text"]), "source_node_ids": [], "aliases": [],
        "umls_status": "not_attempted", "umls_cui": None, "umls_name": None, "umls_score": None, "umls_candidates": [],
    }
    added.append(mention)
    return new_id, f"{label}:mention_added_from_exact_text"


def type_compatibility(relation: str, subject: Mapping[str, Any], obj: Mapping[str, Any]) -> str:
    expected = ROLE_KIND_EXPECTATIONS.get(relation)
    if expected is None:
        return "not_checked"
    subject_ok = str(subject["kind"]) in expected[0]
    object_ok = str(obj["kind"]) in expected[1]
    return "compatible" if subject_ok and object_ok else "warning"


def validate_output(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    parse_success = parsed is not None and isinstance(parsed.get("frames"), list)
    status = str(parsed.get("status") or "ambiguous") if parsed else "parse_failure"
    if status not in {"frames", "no_semantic_frame", "metadata_only", "ambiguous"}:
        status = "ambiguous"
    errors: list[str] = []
    added: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    if parse_success:
        for index, item in enumerate(parsed["frames"]):
            if not isinstance(item, dict):
                errors.append(f"frame_{index}:not_object")
                continue
            local_errors: list[str] = []
            repairs: list[str] = []
            subject, note = resolve_argument(row, item.get("subject_mention_id"), item.get("subject_text"), item.get("subject_kind"), added, f"frame_{index}:subject")
            if note:
                (local_errors if subject is None else repairs).append(note)
            obj, note = resolve_argument(row, item.get("object_mention_id"), item.get("object_text"), item.get("object_kind"), added, f"frame_{index}:object")
            if note:
                (local_errors if obj is None else repairs).append(note)
            relation = str(item.get("relation_type") or "OTHER").upper()
            if relation not in RELATION_TYPES:
                relation = "OTHER"
                local_errors.append("invalid_relation_type")
            frame_status = str(item.get("frame_status") or "ambiguous")
            if frame_status not in {"grounded", "ambiguous"}:
                frame_status = "ambiguous"
                local_errors.append("invalid_frame_status")
            if subject and obj and subject == obj:
                local_errors.append("self_relation")
            asserted_fact = str(item.get("asserted_fact") or "").strip()
            if not asserted_fact:
                local_errors.append("missing_asserted_fact")
            evidence = []
            for e_index, evidence_text in enumerate(item.get("evidence_spans") or []):
                value = str(evidence_text or "").strip()
                spans = all_exact_spans(str(row["claim"]), value)
                if not value or not spans:
                    local_errors.append(f"evidence_{e_index}:not_exact_source_span")
                else:
                    evidence.append({"text": value, "spans": spans})
            if not evidence:
                local_errors.append("missing_exact_evidence")
            qualifiers: list[dict[str, Any]] = []
            for q_index, qualifier in enumerate(item.get("qualifiers") or []):
                if not isinstance(qualifier, dict):
                    local_errors.append(f"qualifier_{q_index}:not_object")
                    continue
                qid, note = resolve_argument(row, qualifier.get("mention_id"), qualifier.get("text"), qualifier.get("kind"), added, f"frame_{index}:qualifier_{q_index}")
                if note:
                    (local_errors if qid is None else repairs).append(note)
                if qid and qid not in {subject, obj}:
                    role = str(qualifier.get("role") or "other")
                    qualifiers.append({"role": role if role in QUALIFIER_ROLES else "other", "mention_id": qid})
            modality = str(item.get("modality") or "asserted")
            if modality not in MODALITIES:
                modality = "asserted"
                repairs.append("invalid_modality_repaired")
            mention_map = {m["mention_id"]: m for m in list(row["mentions"]) + added}
            compatibility = "not_checked"
            if subject in mention_map and obj in mention_map:
                compatibility = type_compatibility(relation, mention_map[subject], mention_map[obj])
            structurally_valid = bool(subject and obj and relation != "OTHER" and frame_status == "grounded" and not local_errors)
            frames.append({
                "frame_id": f"{row['proposition_id']}::F{len(frames)+1}",
                "asserted_fact": asserted_fact,
                "subject_mention_id": subject, "relation_type": relation, "object_mention_id": obj,
                "qualifiers": qualifiers, "evidence": evidence,
                "negated": bool(item.get("negated", False)), "uncertain": bool(item.get("uncertain", False)), "modality": modality,
                "frame_status": frame_status, "structurally_valid": structurally_valid,
                "type_compatibility": compatibility, "validation_errors": local_errors, "repairs": repairs,
            })
            errors.extend(f"frame_{index}:{error}" for error in local_errors)
    base = {key: value for key, value in row.items() if key not in {"statements", "relation_raw_output", "relation_validation_errors"}}
    base["mentions"] = list(row["mentions"]) + added
    return {
        **base, "semantic_frame_parse_success": parse_success, "semantic_frame_status": status,
        "semantic_frames": frames, "semantic_frame_added_mentions": added,
        "semantic_frame_validation_errors": errors if parse_success else ["invalid_json_or_schema"],
        "semantic_frame_raw_output": raw,
    }


def extract_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(read_jsonl(args.v2_root / "raw_statements_v2.jsonl"))
    shard_root = args.output_root / "semantic_frame_shards"
    cached = cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen ontology-aware semantic-frame extraction")
    if len(cached) < len(rows):
        tokenizer, llm = init_vllm(args.qwen_model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)
        done = len(cached)
        for start in range(0, len(rows), args.generation_shard_size):
            batch = rows[start:start + args.generation_shard_size]
            missing = [row for row in batch if row["proposition_id"] not in cached]
            if not missing:
                continue
            prompts = [chat_prompt(tokenizer, FRAME_SYSTEM, semantic_frame_prompt(row)) for row in missing]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            values = [validate_output(row, output.outputs[0].text.strip() if output.outputs else "") for row, output in zip(missing, outputs)]
            atomic_jsonl(shard_root / f"batch_{start:05d}_{start + len(batch):05d}.jsonl", values)
            for value in values:
                cached[value["proposition_id"]] = value
            done += len(values)
            emit("progress", total=len(rows), completed=done)
    if len(cached) != len(rows):
        raise RuntimeError(f"Semantic-frame extraction incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / "semantic_frames_v4.jsonl", (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def concise_v3(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    mention_map = {m["mention_id"]: m["surface"] for m in row["mentions"]}
    return [
        {
            "subject": mention_map.get(s.get("canonical_subject_mention_id_v3")), "relation": s.get("relation_type_v3"),
            "object": mention_map.get(s.get("canonical_object_mention_id_v3")), "status": s.get("orientation_status_v3"),
        }
        for s in row["statements"] if s.get("structurally_valid")
    ]


def concise_v4(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    mention_map = {m["mention_id"]: m["surface"] for m in row["mentions"]}
    return [
        {
            "asserted_fact": f["asserted_fact"], "subject": mention_map.get(f["subject_mention_id"]),
            "relation": f["relation_type"], "object": mention_map.get(f["object_mention_id"]),
            "qualifiers": [{"role": q["role"], "text": mention_map.get(q["mention_id"])} for q in f["qualifiers"]],
            "negated": f["negated"], "uncertain": f["uncertain"], "modality": f["modality"],
            "frame_status": f["frame_status"], "structurally_valid": f["structurally_valid"],
            "type_compatibility": f["type_compatibility"], "validation_errors": f["validation_errors"],
        }
        for f in row["semantic_frames"]
    ]


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.output_root / "semantic_frames_v4.jsonl"))
    emit("start", total=len(rows), completed=0, name="structural audit and fixed manual-review materialization")
    counts = Counter()
    relation_counts = Counter()
    status_counts = Counter()
    type_counts = Counter()
    edges: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    by_id = {row["proposition_id"]: row for row in rows}
    for index, row in enumerate(rows, start=1):
        counts["propositions"] += 1
        counts["parse_success"] += int(row["semantic_frame_parse_success"])
        status_counts[row["semantic_frame_status"]] += 1
        counts["added_mentions"] += len(row["semantic_frame_added_mentions"])
        mention_map = {m["mention_id"]: m for m in row["mentions"]}
        row_has_usable_frame = False
        for frame in row["semantic_frames"]:
            counts["frames"] += 1
            relation_counts[frame["relation_type"]] += 1
            type_counts[frame["type_compatibility"]] += 1
            if frame["structurally_valid"]:
                counts["structurally_valid_frames"] += 1
                row_has_usable_frame = True
                edges.append({
                    "frame_id": frame["frame_id"], "proposition_id": row["proposition_id"], "source": row["source"],
                    "subject_mention_id": frame["subject_mention_id"], "subject_text": mention_map[frame["subject_mention_id"]]["surface"],
                    "relation_type": frame["relation_type"],
                    "object_mention_id": frame["object_mention_id"], "object_text": mention_map[frame["object_mention_id"]]["surface"],
                    "qualifiers": frame["qualifiers"], "negated": frame["negated"], "uncertain": frame["uncertain"], "modality": frame["modality"],
                    "asserted_fact": frame["asserted_fact"], "evidence": frame["evidence"], "type_compatibility": frame["type_compatibility"],
                })
            elif len(rejected) < 300:
                rejected.append({"proposition": row["claim"], "source": row["source"], "frame": frame})
        counts["propositions_with_usable_frame"] += int(row_has_usable_frame)
        if index % 50 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)

    v3_path = args.v3_root / "canonical_triples_v3.jsonl"
    v3 = {row["proposition_id"]: row for row in read_jsonl(v3_path)} if v3_path.is_file() else {}
    selected = review_ids(rows, args.review_per_source, args.seed)
    review = []
    for proposition_id in selected:
        row = by_id[proposition_id]
        review.append({
            "proposition_id": proposition_id, "source": row["source"], "proposition": row["claim"],
            "v3_triples": concise_v3(v3[proposition_id]) if proposition_id in v3 else [],
            "v4_semantic_frames": concise_v4(row),
            "manual_audit": {"v3_exact": None, "v4_exact": None, "v4_error_types": [], "notes": ""},
        })
    usable_edge_rate = counts["structurally_valid_frames"] / max(counts["frames"], 1)
    usable_proposition_coverage = counts["propositions_with_usable_frame"] / counts["propositions"]
    summary = {
        "run_version": RUN_VERSION,
        "scope": "Automated grounding/structure metrics only; semantic precision requires the fixed manual review sample.",
        "input": {"propositions": counts["propositions"], "by_source": dict(Counter(row["source"] for row in rows))},
        "extraction": {
            "parse_success": counts["parse_success"], "parse_success_rate": counts["parse_success"] / counts["propositions"],
            "status": dict(status_counts), "frames": counts["frames"], "structurally_valid_frames": counts["structurally_valid_frames"],
            "usable_edge_rate": usable_edge_rate,
            "propositions_with_usable_frame": counts["propositions_with_usable_frame"],
            "usable_proposition_coverage": usable_proposition_coverage,
            "usable_edges_per_proposition": counts["structurally_valid_frames"] / counts["propositions"],
            "added_mentions": counts["added_mentions"],
            "relation_types": dict(relation_counts), "soft_type_compatibility": dict(type_counts),
        },
        "manual_evaluation": {
            "sample_propositions": len(review), "sample_per_source": args.review_per_source,
            "path": str((args.output_root / "review_sample_200.jsonl").resolve()),
            "primary_metric": "exact semantic-frame precision", "status": "pending_manual_audit",
            "success_criterion": {"precision": 0.85, "usable_edge_coverage": 0.60, "direction_error_rate": 0.05},
        },
        "interpretation_limit": "Structural validity means grounded spans and valid schema, not semantic correctness.",
    }
    atomic_jsonl(args.output_root / "candidate_semantic_edges_v4.jsonl", edges)
    atomic_jsonl(args.output_root / "diagnostics/rejected_frames.jsonl", rejected)
    atomic_jsonl(args.output_root / "review_sample_200.jsonl", review)
    atomic_json(args.output_root / "summary.json", summary)
    atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "summary": str((args.output_root / "summary.json").resolve())})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--v2-root", str(args.v2_root), "--v3-root", str(args.v3_root), "--output-root", str(args.output_root),
        "--qwen-model", str(args.qwen_model), "--llm-python", str(args.llm_python),
        "--generation-batch-size", str(args.generation_batch_size), "--generation-shard-size", str(args.generation_shard_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-model-len", str(args.max_model_len),
        "--max-output-tokens", str(args.max_output_tokens), "--review-per-source", str(args.review_per_source), "--seed", str(args.seed),
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
    ensure_manifest(args, experiment_contract(args, rows))
    progress = WorkflowProgress(
        ["validate proposition/mention cohort and freeze review sample", "Qwen semantic-frame extraction", "grounding audit and review report"],
        [5, 135, 10],
    )
    progress.log("[workflow plan] 1,000 propositions + all mentions -> Qwen meaning-based frames -> deterministic grounding checks -> fixed 200-case review")
    progress.log("[changed variable] semantic interpretation replaces grammatical raw triples; no v2 relation, Llama verifier, QA, or gold input")
    progress.start(1, len(rows), len(rows))
    progress.log(f"[stage 1/3 complete] propositions={len(rows):,} sources=250x4 review_sample={args.review_per_source*4}")
    if args.preflight_only:
        progress.close()
        progress.log("[preflight complete] source, model, review cohort, manifest, and output contract passed; no model loaded")
        return
    run_worker(args, "extract", progress, 2)
    progress.log(f"[stage 2/3 complete] output={args.output_root / 'semantic_frames_v4.jsonl'}")
    run_worker(args, "aggregate", progress, 3)
    progress.log(f"[stage 3/3 complete] summary={args.output_root / 'summary.json'} review={args.output_root / 'review_sample_200.jsonl'}")
    progress.close()
    progress.log(f"[workflow complete] result={args.output_root}")


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate":
        orchestrate(args)
    elif args.mode == "extract":
        extract_worker(args)
    elif args.mode == "aggregate":
        aggregate_worker(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

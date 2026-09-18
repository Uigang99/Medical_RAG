#!/usr/bin/env python3
"""Validate and materialize the source-only Codex-curated 1K KG reference.

The compact manual file intentionally contains semantic decisions only.  This
validator joins them to the immutable proposition source, resolves exact spans,
builds local canonical concept IDs, validates relation endpoints/direction, and
writes a comparison-ready Statement Graph without reading v4/v5/v6 predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2/raw_statements_v2.jsonl"
DEFAULT_REFERENCE_ROOT = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1"

ENTITY_TYPES = {
    "condition", "drug_or_chemical", "procedure_or_intervention",
    "finding_or_outcome", "anatomy", "organism", "gene_or_protein",
    "population",
}

IN_SCOPE_RELATIONS = {
    "IS_A", "PART_OF", "LOCATED_IN", "TREATS", "MANAGES", "PREVENTS",
    "CONTRAINDICATED_FOR", "ADMINISTERED_TO", "CAUSES", "ASSOCIATED_WITH",
    "INCREASES_RISK_OF", "DECREASES_RISK_OF", "INCREASES", "DECREASES",
    "INHIBITS", "ACTIVATES", "BINDS_OR_INTERACTS_WITH", "EXPRESSES",
    "ENCODES", "DIAGNOSES", "DETECTS", "INDICATES", "HAS_SIGN_OR_SYMPTOM",
    "COMPARED_WITH",
}

POLARITIES = {"asserted", "negated", "uncertain"}
STATUSES = {"usable", "no_core_relation", "metadata", "incomplete"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def canonical_surface(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def exact_spans(text: str, surface: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(re.escape(surface), text)]


def source_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "rows": len(payload.splitlines()), "sha256": hashlib.sha256(payload).hexdigest()}


def validate_row(index: int, source: dict[str, Any], annotation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if annotation.get("n") != index:
        errors.append(f"row_number_mismatch:{annotation.get('n')}!= {index}")
    status = str(annotation.get("status") or "")
    if status not in STATUSES:
        errors.append(f"invalid_status:{status}")
    entities = annotation.get("e")
    relations = annotation.get("r")
    outside = annotation.get("x", [])
    if not isinstance(entities, list) or not isinstance(relations, list) or not isinstance(outside, list):
        return {}, [*errors, "e_r_x_must_be_lists"]

    materialized_entities = []
    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, list) or len(entity) not in {3, 4}:
            errors.append(f"entity_{entity_index}:expected_[surface,type,canonical,optional_occurrence]")
            continue
        surface, entity_type, canonical = (str(entity[0]), str(entity[1]), str(entity[2]))
        occurrence = int(entity[3]) if len(entity) == 4 else 0
        if entity_type not in ENTITY_TYPES:
            errors.append(f"entity_{entity_index}:invalid_type:{entity_type}")
        spans = exact_spans(str(source["claim"]), surface)
        if not spans or occurrence < 0 or occurrence >= len(spans):
            errors.append(f"entity_{entity_index}:surface_not_unique_or_missing:{surface}:{occurrence}")
            start, end = -1, -1
        else:
            start, end = spans[occurrence]
        canonical_key = canonical_surface(canonical)
        if not canonical_key:
            errors.append(f"entity_{entity_index}:empty_canonical")
        materialized_entities.append(
            {
                "entity_id": f"{source['proposition_id']}::E{entity_index + 1}",
                "surface": surface,
                "start": start,
                "end": end,
                "entity_type": entity_type,
                "canonical_name": canonical,
                "concept_id": f"LOCAL::{entity_type}::{canonical_key}",
            }
        )

    materialized_relations = []
    for relation_index, relation in enumerate(relations):
        if not isinstance(relation, list) or len(relation) != 4:
            errors.append(f"relation_{relation_index}:expected_[subject_index,type,object_index,polarity]")
            continue
        try:
            subject_index, relation_type, object_index, polarity = int(relation[0]), str(relation[1]), int(relation[2]), str(relation[3])
        except (TypeError, ValueError):
            errors.append(f"relation_{relation_index}:invalid_fields")
            continue
        if relation_type not in IN_SCOPE_RELATIONS:
            errors.append(f"relation_{relation_index}:invalid_relation:{relation_type}")
        if polarity not in POLARITIES:
            errors.append(f"relation_{relation_index}:invalid_polarity:{polarity}")
        if subject_index == object_index or min(subject_index, object_index) < 0 or max(subject_index, object_index) >= len(materialized_entities):
            errors.append(f"relation_{relation_index}:invalid_endpoints:{subject_index}:{object_index}")
            continue
        materialized_relations.append(
            {
                "relation_id": f"{source['proposition_id']}::R{relation_index + 1}",
                "subject_entity_id": materialized_entities[subject_index]["entity_id"],
                "relation_type": relation_type,
                "object_entity_id": materialized_entities[object_index]["entity_id"],
                "polarity": polarity,
                "evidence": str(source["claim"]),
            }
        )
    for outside_index, value in enumerate(outside):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"out_of_scope_{outside_index}:must_be_nonempty_string")
    return {
        "proposition_id": source["proposition_id"],
        "source": source["source"],
        "claim": source["claim"],
        "status": status,
        "entities": materialized_entities,
        "relations": materialized_relations,
        "out_of_scope_relations": outside,
        "curation_confidence": annotation.get("confidence", "high"),
        "notes": annotation.get("notes", ""),
    }, errors


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source)
    annotation_path = args.reference_root / "source_only_annotations.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    annotation_rows = read_jsonl(annotation_path)
    if not args.allow_partial and len(annotation_rows) != len(source_rows):
        raise RuntimeError(f"Reference incomplete: annotations={len(annotation_rows)} source={len(source_rows)}")
    if len(annotation_rows) > len(source_rows):
        raise RuntimeError("Reference has more rows than source")

    materialized = []
    all_errors: dict[int, list[str]] = {}
    for index, annotation in enumerate(annotation_rows, start=1):
        value, errors = validate_row(index, source_rows[index - 1], annotation)
        if errors:
            all_errors[index] = errors
        if value:
            materialized.append(value)
    if all_errors:
        write_json(args.reference_root / "validation_errors.json", all_errors)
        raise RuntimeError(f"Reference validation failed for {len(all_errors)} rows; see validation_errors.json")

    entities = [entity for row in materialized for entity in row["entities"]]
    relations = [relation for row in materialized for relation in row["relations"]]
    concept_mentions: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        concept_mentions[entity["concept_id"]].append(entity["entity_id"])
    concept_nodes = [
        {"concept_id": concept_id, "mention_entity_ids": mention_ids}
        for concept_id, mention_ids in sorted(concept_mentions.items())
    ]
    summary = {
        "reference_type": "Codex-curated source-only structural reference; not physician-annotated medical truth",
        "source": source_identity(args.source),
        "annotation_rows": len(annotation_rows),
        "status_counts": dict(Counter(row["status"] for row in materialized)),
        "entities": len(entities),
        "entity_types": dict(Counter(entity["entity_type"] for entity in entities)),
        "local_concepts": len(concept_nodes),
        "relations": len(relations),
        "relation_types": dict(Counter(relation["relation_type"] for relation in relations)),
        "polarity": dict(Counter(relation["polarity"] for relation in relations)),
        "out_of_scope_relation_descriptions": sum(len(row["out_of_scope_relations"]) for row in materialized),
    }
    write_jsonl(args.reference_root / "reference_statements.jsonl", materialized)
    write_jsonl(args.reference_root / "reference_entities.jsonl", entities)
    write_jsonl(args.reference_root / "reference_relations.jsonl", relations)
    write_jsonl(args.reference_root / "reference_concepts.jsonl", concept_nodes)
    write_json(args.reference_root / "summary.json", summary)
    write_json(args.reference_root / "manifest.json", {
        "source": source_identity(args.source),
        "annotation_protocol": "source-only first pass; no v4/v5/v6 predictions visible; second-pass source review before comparison",
        "scope": {"entity_types": sorted(ENTITY_TYPES), "in_scope_relations": sorted(IN_SCOPE_RELATIONS)},
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

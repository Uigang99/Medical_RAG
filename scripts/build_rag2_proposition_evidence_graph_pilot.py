#!/usr/bin/env python3
"""Build a source-grounded proposition Evidence Graph without unverified semantic triples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2 = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_V6 = ROOT / "datasets/kg/rag2_statement_graph_trigger_classifier_pilot_1k_v6_2"
DEFAULT_REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_proposition_evidence_graph_pilot_1k_v1"
RUN_VERSION = "rag2_proposition_evidence_graph_pilot_1k_v1"
ALLOWED_UMLS_CATEGORIES = {
    "DiseaseOrDisorder", "FindingOrSymptom", "DrugOrChemical", "ProcedureOrTreatment",
    "DiagnosticTest", "Anatomy", "GeneOrProtein", "Organism",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2)
    parser.add_argument("--v6-root", type=Path, default=DEFAULT_V6)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--umls-score-min", type=float, default=0.90)
    parser.add_argument("--umls-margin-min", type=float, default=0.05)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def strict_umls_link(mention: Mapping[str, Any], score_min: float, margin_min: float) -> dict[str, Any] | None:
    if mention.get("umls_status") != "linked_confident" or not mention.get("umls_cui"):
        return None
    candidates = list(mention.get("umls_candidates") or [])
    if not candidates:
        return None
    top = candidates[0]
    score = float(top.get("score") or 0.0)
    second = float(candidates[1].get("score") or 0.0) if len(candidates) > 1 else 0.0
    if str(top.get("cui") or "") != str(mention["umls_cui"]):
        return None
    if score < score_min or score - second < margin_min:
        return None
    if str(top.get("category") or "") not in ALLOWED_UMLS_CATEGORIES:
        return None
    return {
        "concept_id": f"UMLS::{top['cui']}", "cui": top["cui"],
        "name": top.get("canonical_name") or mention.get("umls_name") or mention["surface"],
        "category": top["category"], "score": score, "margin": score - second,
    }


def main() -> None:
    args = parse_args()
    source_path = args.v2_root / "raw_statements_v2.jsonl"
    semantic_path = args.v6_root / "candidate_edges_v6.jsonl"
    reference_path = args.reference_root / "reference_statements.jsonl"
    for path in (source_path, semantic_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_jsonl(source_path)
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1,000 propositions; got {len(rows)}")
    print("[stage 1/3] validate 1,000 cached propositions and build deterministic provenance backbone", flush=True)

    documents: dict[str, dict[str, Any]] = {}
    statements: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    concepts: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    quality = Counter()
    for row in rows:
        proposition_id = str(row["proposition_id"])
        document_id = str(row["stable_id"])
        claim = str(row["claim"])
        documents.setdefault(document_id, {"node_id": document_id, "node_type": "Document", "source": row["source"]})
        statements.append({
            "node_id": proposition_id, "node_type": "Statement", "document_id": document_id,
            "source": row["source"], "text": claim, "evidence": row.get("evidence") or "",
            "proposition_class": row.get("proposition_class"), "source_text_sha256": row.get("source_text_sha256"),
        })
        edges.append({"source": document_id, "relation": "HAS_STATEMENT", "target": proposition_id, "verified": True})
        quality["statements"] += 1
        quality["nonempty_statement"] += int(bool(claim.strip()))
        for mention in row.get("mentions") or []:
            start, end = int(mention["start"]), int(mention["end"])
            surface = str(mention["surface"])
            grounded = 0 <= start < end <= len(claim) and claim[start:end] == surface
            quality["mentions"] += 1
            quality["grounded_mentions"] += int(grounded)
            if not grounded:
                continue
            mention_id = str(mention["mention_id"])
            mentions.append({
                "node_id": mention_id, "node_type": "Mention", "statement_id": proposition_id,
                "surface": surface, "start": start, "end": end, "kind": mention.get("kind"),
            })
            edges.append({"source": proposition_id, "relation": "MENTIONS", "target": mention_id, "verified": True})
            link = strict_umls_link(mention, args.umls_score_min, args.umls_margin_min)
            if link:
                concepts.setdefault(link["concept_id"], {"node_id": link["concept_id"], "node_type": "Concept", **link})
                edges.append({
                    "source": mention_id, "relation": "NORMALIZED_TO", "target": link["concept_id"],
                    "verified": True, "score": link["score"], "margin": link["margin"],
                })
                quality["strict_normalized_mentions"] += 1

    print(f"[stage 2/3] core graph built: statements={len(statements):,} mentions={len(mentions):,} strict_concepts={len(concepts):,}", flush=True)
    reference_status = Counter(str(row["status"]) for row in read_jsonl(reference_path))
    experimental = read_jsonl(semantic_path)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "documents.jsonl", documents.values())
    write_jsonl(output / "statements.jsonl", statements)
    write_jsonl(output / "mentions.jsonl", mentions)
    write_jsonl(output / "concepts.jsonl", concepts.values())
    write_jsonl(output / "core_edges.jsonl", edges)
    write_jsonl(output / "experimental_semantic_edges.jsonl", experimental)

    span_rate = quality["grounded_mentions"] / max(quality["mentions"], 1)
    nonempty_rate = quality["nonempty_statement"] / max(quality["statements"], 1)
    reference_factual_rate = (reference_status["usable"] + reference_status["no_core_relation"]) / max(sum(reference_status.values()), 1)
    passed = nonempty_rate == 1.0 and span_rate == 1.0 and len(concepts) > 0
    summary = {
        "run_version": RUN_VERSION,
        "scope": "source-grounded Evidence Graph; entity-to-entity semantic triples are excluded from the core graph",
        "counts": {
            "documents": len(documents), "statements": len(statements), "mentions": len(mentions),
            "strict_umls_concepts": len(concepts), "core_edges": len(edges),
            "experimental_semantic_edges": len(experimental),
        },
        "quality": {
            "nonempty_statement_rate": nonempty_rate, "exact_mention_span_grounding_rate": span_rate,
            "strict_normalized_mentions": quality["strict_normalized_mentions"],
            "normalization_contract": {
                "umls_score_min": args.umls_score_min, "top1_margin_min": args.umls_margin_min,
                "allowed_categories": sorted(ALLOWED_UMLS_CATEGORIES),
            },
            "reference_status": dict(reference_status),
            "reference_factual_or_no_core_relation_rate": reference_factual_rate,
        },
        "semantic_edge_policy": {
            "core_graph": "no automatically inferred entity-to-entity semantic relation",
            "experimental_layer": str((output / "experimental_semantic_edges.jsonl").resolve()),
            "reason": "bounded relation extraction failed coverage and wrong-edge thresholds",
        },
        "passed_structural_gate": passed,
        "limitations": [
            "Mention edges assert textual occurrence, not biomedical salience.",
            "UMLS score/margin filtering is a structural confidence rule, not an independently measured normalization precision.",
            "This is an Evidence Graph, not a complete biomedical entity-relation KG.",
        ],
        "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[stage 3/3] structural gate={'PASS' if passed else 'FAIL'} span_grounding={100*span_rate:.2f}% "
        f"core_semantic_triples=0 experimental_triples={len(experimental):,} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a precision-first biomedical Statement Graph from atomic propositions.

This pilot deliberately narrows the task before asking an LLM to judge it:
1. spaCy keeps noun-headed, graph-worthy core mentions and records every rejection.
2. Code proposes a directed relation only when a compatible type signature and an
   explicit lexical trigger are both present.
3. Qwen performs binary source-fidelity verification of each fixed assertion.  It
   cannot invent endpoints, reverse their direction, or choose from a broad ontology.
4. The pipeline preserves polarity, uncertainty, source proposition, trigger, and
   all intermediate decisions for a fixed manual precision/coverage audit.

The goal is not maximal recall.  It is a decisive test of whether this project can
reach usable semantic-edge precision before any larger KG is built.
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
from build_rag2_statement_graph_v2_pilot import normalized_surface  # noqa: E402


RUN_VERSION = "rag2_statement_graph_trigger_classifier_pilot_1k_v6_2"
PROGRESS_PREFIX = "@@TRIGGER_V6_PROGRESS@@"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
DEFAULT_V2_ROOT = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_V4_ROOT = ROOT / "datasets/kg/rag2_statement_graph_semantic_frames_pilot_1k_v4"
DEFAULT_V5_ROOT = ROOT / "datasets/kg/rag2_statement_graph_pair_classifier_pilot_1k_v5"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_trigger_classifier_pilot_1k_v6_2"
DEFAULT_QWEN = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_ENTITY_PYTHON = WORKSPACE / ".venv_rag2_kg_mvp_py312/bin/python"
DEFAULT_LLM_PYTHON = WORKSPACE / ".venv_vllm/bin/python"

CORE_KINDS = frozenset(
    {
        "condition",
        "drug_or_chemical",
        "procedure_or_intervention",
        "finding_or_outcome",
        "anatomy",
        "organism",
        "gene_or_protein",
        "population",
    }
)
CONDITIONS = frozenset({"condition", "finding_or_outcome"})
INTERVENTIONS = frozenset({"drug_or_chemical", "procedure_or_intervention"})
BIOLOGICAL = frozenset(CORE_KINDS - {"population"})
ALL_CORE = frozenset(CORE_KINDS)

UMLS_CATEGORY_TO_KIND = {
    "DiseaseOrDisorder": "condition",
    "FindingOrSymptom": "finding_or_outcome",
    "DrugOrChemical": "drug_or_chemical",
    "ProcedureOrTreatment": "procedure_or_intervention",
    "DiagnosticTest": "procedure_or_intervention",
    "Anatomy": "anatomy",
    "GeneOrProtein": "gene_or_protein",
    "Organism": "organism",
}

# These words are valid English nouns but are too generic to be reusable KG nodes
# without a modifier.  Specific multiword phrases such as "insulin treatment" remain.
GENERIC_SINGLETONS = frozenset(
    {
        "analysis", "approach", "case", "condition", "data", "disease", "effect",
        "effects", "finding", "group", "information", "intervention", "level", "method",
        "outcome", "patient", "patients", "procedure", "result", "results", "risk",
        "people", "study", "subject", "subjects", "symptom", "symptoms", "therapy", "treatment",
        "value", "values",
    }
)


def relation_spec(
    subjects: Iterable[str],
    objects: Iterable[str],
    patterns: Sequence[tuple[str, str]],
    description: str,
    symmetric: bool = False,
) -> dict[str, Any]:
    return {
        "subjects": frozenset(subjects),
        "objects": frozenset(objects),
        "patterns": tuple(patterns),
        "description": description,
        "symmetric": symmetric,
    }


# Pattern orientation is relative to the trigger: SO requires subject before and
# object after it; OS requires the reverse; ANY is used only when semantic types
# themselves determine roles (for example Drug TREATS Condition).
RELATION_SPECS: dict[str, dict[str, Any]] = {
    "IS_A": relation_spec(ALL_CORE, ALL_CORE, ((r"\b(?:is|are|was|were)\s+(?:an?|the)\s+", "SO"), (r"\b(?:is|are|was|were)\s+(?:a\s+)?(?:type|form|subtype|kind|class)\s+of\b", "SO"), (r"\b(?:classified|defined)\s+as\b", "SO")), "subject is a subtype or instance of object"),
    "PART_OF": relation_spec(frozenset({"anatomy", "gene_or_protein"}), frozenset({"anatomy", "gene_or_protein"}), ((r"\b(?:part|component|subunit)\s+of\b", "SO"), (r"\b(?:contains?|comprises?|includes?)\b", "OS")), "subject is a physical or biological part of object"),
    "LOCATED_IN": relation_spec(BIOLOGICAL, frozenset({"anatomy"}), ((r"\b(?:located|found|present|expressed)\s+(?:in|within|at)\b", "SO"), (r"\bwithin\b", "SO")), "subject is located in object"),
    "TREATS": relation_spec(INTERVENTIONS, CONDITIONS, ((r"\b(?:treats?|treating|used\s+to\s+treat|therapy\s+for|treatment\s+of)\b", "SO"), (r"\btreated\s+(?:with|by)\b", "OS")), "intervention treats condition"),
    "MANAGES": relation_spec(INTERVENTIONS, CONDITIONS, ((r"\b(?:manages?|management\s+of|controls?)\b", "SO"), (r"\b(?:managed|controlled)\s+(?:with|by)\b", "OS")), "intervention manages condition"),
    "PREVENTS": relation_spec(INTERVENTIONS, CONDITIONS, ((r"\b(?:prevents?|prevention\s+of|protects?\s+against)\b", "SO"), (r"\bprevented\s+(?:with|by)\b", "OS")), "intervention prevents condition"),
    "CONTRAINDICATED_FOR": relation_spec(INTERVENTIONS, CONDITIONS | frozenset({"population"}), ((r"\bcontraindicat(?:ed|ion|ions)\b", "ANY"), (r"\bshould\s+not\s+(?:be\s+)?(?:use|administer|prescribe)\w*\b", "ANY")), "intervention is contraindicated for condition or population"),
    "ADMINISTERED_TO": relation_spec(INTERVENTIONS, frozenset({"population"}), ((r"\b(?:administered|given|prescribed|provided)\s+to\b", "SO"), (r"\b(?:receive|receives|received)\b", "OS")), "intervention is administered to population"),
    "CAUSES": relation_spec(BIOLOGICAL, CONDITIONS, ((r"\b(?:causes?|induces?|leads?\s+to|results?\s+in|produces?)\b", "SO"), (r"\b(?:caused|induced|produced)\b(?!\s+by)", "SO"), (r"\b(?:caused|induced|produced)\s+by\b", "OS"), (r"\b(?:due|attributable)\s+to\b", "OS")), "subject causes object"),
    "ASSOCIATED_WITH": relation_spec(BIOLOGICAL, BIOLOGICAL, ((r"\b(?:associated|correlated|linked)\s+with\b", "ANY"), (r"\bassociation\s+(?:with|between)\b", "ANY")), "subject is explicitly associated with object", symmetric=True),
    "INCREASES_RISK_OF": relation_spec(BIOLOGICAL, CONDITIONS, ((r"\b(?:increases?|increased|elevates?|elevated)\s+(?:the\s+)?risk\s+of\b", "SO"), (r"\brisk\s+factor\s+for\b", "SO"), (r"\bpredisposes?\s+to\b", "SO")), "subject increases risk of object"),
    "DECREASES_RISK_OF": relation_spec(BIOLOGICAL, CONDITIONS, ((r"\b(?:decreases?|decreased|reduces?|reduced|lowers?|lowered)\s+(?:the\s+)?risk\s+of\b", "SO"), (r"\bprotects?\s+against\b", "SO")), "subject decreases risk of object"),
    "INCREASES": relation_spec(BIOLOGICAL, CONDITIONS, ((r"\b(?:increases?|increased|elevates?|elevated|raises?|raised|enhances?|enhanced)\b(?!\s+(?:the\s+)?risk)", "SO"),), "subject increases object"),
    "DECREASES": relation_spec(BIOLOGICAL, CONDITIONS, ((r"\b(?:decreases?|decreased|reduces?|reduced|lowers?|lowered|suppresses?|suppressed)\b(?!\s+(?:the\s+)?risk)", "SO"),), "subject decreases object"),
    "INHIBITS": relation_spec(BIOLOGICAL, BIOLOGICAL, ((r"\b(?:inhibits?|inhibited|blocks?|blocked)\b", "SO"),), "subject inhibits object"),
    "ACTIVATES": relation_spec(BIOLOGICAL, BIOLOGICAL, ((r"\b(?:activates?|activated|stimulates?|stimulated)\b", "SO"),), "subject activates object"),
    "BINDS_OR_INTERACTS_WITH": relation_spec(frozenset({"drug_or_chemical", "gene_or_protein"}), frozenset({"drug_or_chemical", "gene_or_protein"}), ((r"\b(?:binds?|bound)\s+to\b", "SO"), (r"\binteracts?\s+with\b", "ANY")), "subject binds or interacts with object", symmetric=True),
    "EXPRESSES": relation_spec(frozenset({"anatomy", "organism", "condition", "gene_or_protein"}), frozenset({"gene_or_protein"}), ((r"\b(?:expresses?|expression\s+of)\b", "SO"), (r"\bexpressed\s+(?:in|by)\b", "OS")), "subject expresses gene or protein object"),
    "ENCODES": relation_spec(frozenset({"gene_or_protein"}), frozenset({"gene_or_protein"}), ((r"\b(?:encodes?|encoded)\b", "SO"),), "subject encodes object"),
    "DIAGNOSES": relation_spec(frozenset({"procedure_or_intervention"}), CONDITIONS, ((r"\b(?:diagnoses?|diagnosis\s+of|diagnostic\s+for)\b", "SO"), (r"\bdiagnosed\s+(?:with|by)\b", "OS")), "test or procedure diagnoses condition"),
    "DETECTS": relation_spec(frozenset({"procedure_or_intervention"}), CONDITIONS, ((r"\b(?:detects?|screens?\s+for|screening\s+for|test\s+for)\b", "SO"), (r"\bdetected\s+by\b", "OS")), "test or procedure detects condition or finding"),
    "INDICATES": relation_spec(frozenset({"finding_or_outcome"}), frozenset({"condition"}), ((r"\b(?:indicates?|suggests?|marker\s+of|predicts?)\b", "SO"), (r"\b(?:indicated|suggested|predicted)\s+by\b", "OS")), "finding indicates condition"),
    "HAS_SIGN_OR_SYMPTOM": relation_spec(frozenset({"condition"}), frozenset({"finding_or_outcome"}), ((r"\b(?:presents?|presented)\s+with\b", "ANY"), (r"\b(?:characteri[sz]ed|manifested)\s+by\b", "ANY"), (r"\b(?:signs?|symptoms?|manifestations?)\b.*\b(?:include|includes|included)\b", "ANY")), "condition has sign, symptom, or manifestation object"),
    "COMPARED_WITH": relation_spec(ALL_CORE, ALL_CORE, ((r"\b(?:compared|versus|vs\.?|relative)\s+(?:with|to)?\b", "ANY"),), "subject is explicitly compared with object", symmetric=True),
}

VERIFY_SYSTEM = """Verify fixed biomedical relation assertions against one source proposition.

For every candidate, judge only whether the proposition explicitly states the exact directed assertion shown. Do not use outside medical knowledge. Entity co-occurrence is not support. Similar, broader, or reversed relations are not support. The trigger is only a candidate-generation hint and is not proof. Use S only when subject, relation meaning, and object direction all match the proposition; N when not stated; A only when the wording is genuinely unresolved. Preserve whether a supported assertion is negated or uncertain. Do not invent or replace endpoints.

Return exactly one item for every supplied ID and JSON only:
{"d":[{"i":"A001","j":"S|N|A","p":"asserted|negated|uncertain"}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "candidates", "classify", "aggregate"), default="orchestrate")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--v5-root", type=Path, default=DEFAULT_V5_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--entity-python", type=Path, default=DEFAULT_ENTITY_PYTHON)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_LLM_PYTHON)
    parser.add_argument("--spacy-model", default="en_core_sci_sm")
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument("--max-candidates-per-proposition", type=int, default=32)
    parser.add_argument("--max-pair-distance-chars", type=int, default=180)
    parser.add_argument("--review-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(PROGRESS_PREFIX + canonical_json({"event": event, **payload}), flush=True)


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = args.v2_root / "raw_statements_v2.jsonl"
    if not source.is_file() or not (args.v2_root / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"Completed v2 source required: {source}")
    rows = list(read_jsonl(source))
    identifiers = [str(row.get("proposition_id") or "") for row in rows]
    if len(rows) != 1000 or any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Expected 1,000 unique propositions; rows={len(rows)} unique={len(set(identifiers))}")
    if Counter(str(row.get("source")) for row in rows) != {source_name: 250 for source_name in SOURCES}:
        raise RuntimeError("Expected 250 propositions from each of four sources")
    return rows


def fixed_review_ids(args: argparse.Namespace) -> list[str]:
    path = args.v4_root / "review_sample_200.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Fixed v4 review sample required: {path}")
    identifiers = [str(row["proposition_id"]) for row in read_jsonl(path)]
    if len(identifiers) != args.review_size or len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Expected {args.review_size} unique fixed review IDs; found {len(identifiers)}")
    return identifiers


def confident_umls_kind(mention: Mapping[str, Any]) -> str | None:
    if mention.get("umls_status") != "linked_confident" or not mention.get("umls_cui"):
        return None
    candidates = mention.get("umls_candidates") or []
    category = str(candidates[0].get("category") or "") if candidates else ""
    return UMLS_CATEGORY_TO_KIND.get(category)


def resolve_kind(mention: Mapping[str, Any]) -> tuple[str | None, str]:
    current = str(mention.get("kind") or "")
    umls_kind = confident_umls_kind(mention)
    if current in CORE_KINDS:
        if umls_kind and current != umls_kind:
            return umls_kind, "umls_confident_type_override"
        return current, "existing_core_type"
    if umls_kind:
        return umls_kind, "umls_confident_type_mapping"
    return None, "noncore_or_unresolved_type"


def mention_decision(mention: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict[str, Any]:
    kind, type_reason = resolve_kind(mention)
    surface = str(mention.get("surface") or "").strip()
    normalized = normalized_surface(surface)
    reasons: list[str] = []
    if kind is None:
        reasons.append(type_reason)
    if not re.search(r"[A-Za-z]", surface):
        reasons.append("no_alphabetic_content")
    root_pos = str(annotation.get("root_pos") or "")
    if root_pos not in {"NOUN", "PROPN"}:
        reasons.append("not_noun_headed")
    tokens = [value for value in re.split(r"\s+", normalized) if value]
    if re.match(r"^(?:after|before|by|due\s+to|for|from|in|of|on|to|under|with|without)\b", normalized):
        reasons.append("prepositional_or_relational_fragment")
    if re.match(r"^(?:her|his|its|our|their|your)\b", normalized) and not mention.get("umls_cui"):
        reasons.append("unresolved_possessive_fragment")
    if len(tokens) == 1 and normalized in GENERIC_SINGLETONS and not mention.get("umls_cui"):
        reasons.append("generic_unmodified_noun")
    if kind == "population" and len(tokens) == 1 and not mention.get("umls_cui"):
        reasons.append("generic_population_without_modifier")
    if len(surface) > 120:
        reasons.append("overlong_mention")
    return {
        **dict(mention),
        "resolved_kind": kind,
        "type_resolution": type_reason,
        "syntax": dict(annotation),
        "keep": not reasons,
        "rejection_reasons": reasons,
    }


def mention_rank(value: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(bool(value.get("umls_cui"))),
        int(bool(value.get("globally_resolvable"))),
        int(str(value.get("kind")) != "biomedical_concept"),
        int(value["end"]) - int(value["start"]),
        str(value.get("mention_id")),
    )


def overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def same_concept(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_cui, right_cui = left.get("umls_cui"), right.get("umls_cui")
    if left_cui and right_cui and left_cui == right_cui:
        return True
    return normalized_surface(str(left["surface"])) == normalized_surface(str(right["surface"]))


def deduplicate_mentions(claim: str, decisions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = [dict(value) for value in decisions if value["keep"]]
    kept.sort(key=lambda value: mention_rank(value), reverse=True)
    selected: list[dict[str, Any]] = []
    deduplicated: list[dict[str, Any]] = []
    for mention in kept:
        duplicate = None
        for previous in selected:
            same_cui = bool(mention.get("umls_cui") and mention.get("umls_cui") == previous.get("umls_cui"))
            same_surface = normalized_surface(str(mention["surface"])) == normalized_surface(str(previous["surface"]))
            nested_same_kind = overlap(mention, previous) and mention["resolved_kind"] == previous["resolved_kind"]
            between = claim[min(int(mention["end"]), int(previous["end"])) : max(int(mention["start"]), int(previous["start"]))]
            acronym_pair = (
                "(" in between
                and (str(mention["surface"]).isupper() or str(previous["surface"]).isupper())
                and min(len(str(mention["surface"])), len(str(previous["surface"]))) <= 12
            )
            if same_cui or same_surface or nested_same_kind or acronym_pair:
                duplicate = previous
                break
        if duplicate is None:
            selected.append(mention)
        else:
            deduplicated.append({**mention, "keep": False, "rejection_reasons": ["duplicate_or_alias"], "merged_into": duplicate["mention_id"]})
            duplicate.setdefault("merged_mentions", []).append(
                {key: mention.get(key) for key in ("mention_id", "surface", "start", "end", "resolved_kind", "umls_cui")}
            )
    selected.sort(key=lambda value: (int(value["start"]), int(value["end"]), str(value["mention_id"])))
    for index, value in enumerate(selected, start=1):
        value["core_id"] = f"C{index:03d}"
        value["kind"] = value.pop("resolved_kind")
    return selected, deduplicated


def trigger_is_near(subject: Mapping[str, Any], obj: Mapping[str, Any], match: re.Match[str], orientation: str, max_distance: int) -> bool:
    subject_start, subject_end = int(subject["start"]), int(subject["end"])
    object_start, object_end = int(obj["start"]), int(obj["end"])
    if max(subject_start, object_start) - min(subject_end, object_end) > max_distance:
        return False
    if orientation == "SO":
        return (
            subject_end <= match.start()
            and match.end() <= object_start
            and match.start() - subject_end <= max_distance
            and object_start - match.end() <= max_distance
        )
    if orientation == "OS":
        return (
            object_end <= match.start()
            and match.end() <= subject_start
            and match.start() - object_end <= max_distance
            and subject_start - match.end() <= max_distance
        )
    envelope_start = min(subject_start, object_start) - 40
    envelope_end = max(subject_end, object_end) + 40
    return match.start() >= envelope_start and match.end() <= envelope_end


def pair_trigger_distance(subject: Mapping[str, Any], obj: Mapping[str, Any], match: re.Match[str]) -> int:
    def span_distance(value: Mapping[str, Any]) -> int:
        start, end = int(value["start"]), int(value["end"])
        if end <= match.start():
            return match.start() - end
        if start >= match.end():
            return start - match.end()
        return 0

    return span_distance(subject) + span_distance(obj)


def construct_relation_candidates(row: Mapping[str, Any], max_candidates: int, max_distance: int) -> tuple[list[dict[str, Any]], int]:
    claim = str(row["claim"])
    mentions = list(row["core_mentions"])
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation, spec in RELATION_SPECS.items():
        matches: list[tuple[re.Match[str], str, str]] = []
        for pattern, orientation in spec["patterns"]:
            for match in re.finditer(pattern, claim, flags=re.IGNORECASE):
                matches.append((match, orientation, pattern))
        if not matches:
            continue
        for match, orientation, pattern in matches:
            compatible_pairs: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
            for subject in mentions:
                if subject["kind"] not in spec["subjects"]:
                    continue
                for obj in mentions:
                    if subject["mention_id"] == obj["mention_id"] or obj["kind"] not in spec["objects"]:
                        continue
                    if same_concept(subject, obj):
                        continue
                    if spec["symmetric"] and str(subject["mention_id"]) > str(obj["mention_id"]):
                        continue
                    if trigger_is_near(subject, obj, match, orientation, max_distance):
                        compatible_pairs.append((pair_trigger_distance(subject, obj, match), subject, obj))
            # One lexical relation expression normally has one core argument pair.
            # Keep the two closest candidates only so coordination remains possible,
            # while preventing one copula/verb from producing every pair in a clause.
            compatible_pairs.sort(key=lambda value: (value[0], int(value[1]["start"]), int(value[2]["start"])))
            for _, subject, obj in compatible_pairs[:2]:
                key = (str(subject["mention_id"]), relation, str(obj["mention_id"]))
                candidates[key] = {
                    "assertion_id": "",
                    "subject": {key_name: subject.get(key_name) for key_name in ("core_id", "mention_id", "surface", "kind", "start", "end", "umls_cui", "umls_name")},
                    "relation_type": relation,
                    "relation_description": spec["description"],
                    "object": {key_name: obj.get(key_name) for key_name in ("core_id", "mention_id", "surface", "kind", "start", "end", "umls_cui", "umls_name")},
                    "trigger": {"text": match.group(0), "start": match.start(), "end": match.end(), "orientation_rule": orientation, "pattern": pattern},
                }
    ordered = sorted(
        candidates.values(),
        key=lambda value: (
            int(value["trigger"]["start"]),
            int(value["subject"]["start"]),
            int(value["object"]["start"]),
            str(value["relation_type"]),
        ),
    )
    untruncated = len(ordered)
    ordered = ordered[:max_candidates]
    for index, value in enumerate(ordered, start=1):
        value["assertion_id"] = f"{row['proposition_id']}::A{index:03d}"
    return ordered, max(0, untruncated - len(ordered))


def annotate_mentions(doc: Any, mentions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for mention in mentions:
        span = doc.char_span(int(mention["start"]), int(mention["end"]), alignment_mode="expand")
        if span is None or len(span) == 0:
            annotation = {"root_text": "", "root_pos": "", "root_lemma": "", "token_count": 0}
        else:
            annotation = {
                "root_text": span.root.text,
                "root_pos": span.root.pos_,
                "root_lemma": span.root.lemma_,
                "token_count": len(span),
            }
        values.append(mention_decision(mention, annotation))
    return values


def candidate_worker(args: argparse.Namespace) -> None:
    import spacy

    rows = load_rows(args)
    output = args.output_root / "relation_candidates_v6.jsonl"
    if output.is_file():
        cached = list(read_jsonl(output))
        if len(cached) == len(rows) and {row["proposition_id"] for row in cached} == {row["proposition_id"] for row in rows}:
            emit("start", total=len(rows), completed=len(rows), name="spaCy core-node and trigger candidate construction")
            emit("complete", total=len(rows), completed=len(rows))
            return
    emit("start", total=len(rows), completed=0, name="spaCy core-node and trigger candidate construction")
    nlp = spacy.load(args.spacy_model, disable=["ner"])
    results: list[dict[str, Any]] = []
    for index, (row, doc) in enumerate(zip(rows, nlp.pipe((str(value["claim"]) for value in rows), batch_size=128)), start=1):
        decisions = annotate_mentions(doc, row["mentions"])
        core_mentions, deduplicated = deduplicate_mentions(str(row["claim"]), decisions)
        candidate_row = {
            "proposition_id": row["proposition_id"],
            "source": row["source"],
            "claim": row["claim"],
            "mention_decisions": [*decisions, *deduplicated],
            "core_mentions": core_mentions,
        }
        assertions, dropped = construct_relation_candidates(candidate_row, args.max_candidates_per_proposition, args.max_pair_distance_chars)
        candidate_row["relation_candidates"] = assertions
        candidate_row["truncated_candidates"] = dropped
        results.append(candidate_row)
        if index % 50 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)
    atomic_jsonl(output, results)
    emit("complete", total=len(rows), completed=len(rows))


def candidate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    rejection = Counter()
    kinds = Counter()
    relations = Counter()
    for row in rows:
        counts["propositions"] += 1
        counts["input_mentions"] += sum(1 for value in row["mention_decisions"] if "merged_into" not in value)
        counts["core_mentions"] += len(row["core_mentions"])
        counts["propositions_with_core_mentions"] += int(bool(row["core_mentions"]))
        counts["relation_candidates"] += len(row["relation_candidates"])
        counts["propositions_with_candidates"] += int(bool(row["relation_candidates"]))
        counts["truncated_candidates"] += int(row.get("truncated_candidates", 0))
        for value in row["mention_decisions"]:
            if not value.get("keep"):
                rejection.update(value.get("rejection_reasons") or [])
        kinds.update(value["kind"] for value in row["core_mentions"])
        relations.update(value["relation_type"] for value in row["relation_candidates"])
    return {**dict(counts), "rejection_reasons": dict(rejection), "core_kinds": dict(kinds), "candidate_relations": dict(relations)}


def compact_local_id(assertion_id: str) -> str:
    return assertion_id.rsplit("::", 1)[-1]


def verifier_prompt(row: Mapping[str, Any]) -> str:
    payload = [
        {
            "i": compact_local_id(value["assertion_id"]),
            "s": value["subject"]["surface"],
            "r": value["relation_type"],
            "meaning": value["relation_description"],
            "o": value["object"]["surface"],
            "trigger": value["trigger"]["text"],
        }
        for value in row["relation_candidates"]
    ]
    return "SOURCE PROPOSITION:\n" + str(row["claim"]) + "\n\nFIXED ASSERTIONS TO VERIFY:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_verification(row: Mapping[str, Any], raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    decisions = (parsed.get("d") or parsed.get("decisions")) if isinstance(parsed, dict) else None
    parse_success = isinstance(decisions, list)
    supplied: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    local_to_full = {compact_local_id(value["assertion_id"]): value["assertion_id"] for value in row["relation_candidates"]}
    if parse_success:
        for decision in decisions:
            if isinstance(decision, dict):
                returned = str(decision.get("i") or decision.get("assertion_id") or "")
                supplied[local_to_full.get(returned, returned)].append(decision)
    verified = []
    for candidate in row["relation_candidates"]:
        errors: list[str] = []
        choices = supplied.get(candidate["assertion_id"], [])
        if len(choices) != 1:
            errors.append("missing_or_duplicate_decision")
        item = choices[0] if choices else {}
        judgement = str(item.get("j") or item.get("judgement") or "").upper()
        judgement = {"SUPPORTED": "S", "NO_RELATION": "N", "AMBIGUOUS": "A"}.get(judgement, judgement)
        if judgement not in {"S", "N", "A"}:
            errors.append("invalid_judgement")
        polarity = str(item.get("p") or item.get("polarity") or "asserted").lower()
        if polarity not in {"asserted", "negated", "uncertain"}:
            errors.append("invalid_polarity")
        verified.append(
            {
                **candidate,
                "judgement": judgement or "MISSING",
                "polarity": polarity,
                "classification_valid": not errors,
                "positive": judgement == "S" and not errors,
                "validation_errors": errors,
            }
        )
    known = {value["assertion_id"] for value in row["relation_candidates"]}
    unexpected = sorted(value for value in supplied if value and value not in known)
    return {
        **row,
        "verification_parse_success": parse_success,
        "verified_assertions": verified,
        "unexpected_assertion_ids": unexpected,
        "verification_raw_output": raw,
    }


def prompt_token_audit(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_model), local_files_only=True, trust_remote_code=True)
    maximum = 0
    over_limit = 0
    for row in rows:
        if not row["relation_candidates"]:
            continue
        prompt = chat_prompt(tokenizer, VERIFY_SYSTEM, verifier_prompt(row))
        length = len(tokenizer.encode(prompt, add_special_tokens=False)) + args.max_output_tokens
        maximum = max(maximum, length)
        over_limit += int(length > args.max_model_len)
    if over_limit:
        raise RuntimeError(f"Verifier prompts exceed max model length: rows={over_limit} max_tokens={maximum} limit={args.max_model_len}")
    return {"max_prompt_plus_output_tokens": maximum, "over_limit": over_limit}


def classify_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(read_jsonl(args.output_root / "relation_candidates_v6.jsonl"))
    shard_root = args.output_root / "verification_shards"
    cached = cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen fixed assertion source-fidelity verification")
    if len(cached) < len(rows):
        tokenizer, llm = init_vllm(args.qwen_model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)
        for start in range(0, len(rows), args.generation_shard_size):
            source_batch = rows[start : start + args.generation_shard_size]
            missing = [row for row in source_batch if row["proposition_id"] not in cached]
            model_rows = [row for row in missing if row["relation_candidates"]]
            if model_rows:
                prompts = [chat_prompt(tokenizer, VERIFY_SYSTEM, verifier_prompt(row)) for row in model_rows]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(model_rows, outputs):
                    raw = output.outputs[0].text.strip() if output.outputs else ""
                    value = validate_verification(row, raw)
                    cached[value["proposition_id"]] = value
            for row in missing:
                if not row["relation_candidates"]:
                    cached[row["proposition_id"]] = {
                        **row,
                        "verification_parse_success": True,
                        "verified_assertions": [],
                        "unexpected_assertion_ids": [],
                        "verification_raw_output": "",
                    }
            atomic_jsonl(
                shard_root / f"batch_{start:05d}_{start + len(source_batch):05d}.jsonl",
                (cached[row["proposition_id"]] for row in source_batch),
            )
            emit("progress", total=len(rows), completed=len(cached))
    if len(cached) != len(rows):
        raise RuntimeError(f"Verification incomplete: {len(cached)}/{len(rows)}")
    atomic_jsonl(args.output_root / "assertion_verifications_v6.jsonl", (cached[row["proposition_id"]] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def balanced_positive_sample(edges: Sequence[Mapping[str, Any]], size: int, seed: int) -> list[Mapping[str, Any]]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        buckets[(str(edge["source"]), str(edge["relation_type"]))].append(edge)
    for key in buckets:
        buckets[key].sort(key=lambda row: sha256_text(f"{seed}|{row['assertion_id']}"))
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
    rows = list(read_jsonl(args.output_root / "assertion_verifications_v6.jsonl"))
    emit("start", total=len(rows), completed=0, name="aggregate edges and fixed audit material")
    counts = Counter()
    judgements = Counter()
    relations = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    edges: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        counts["propositions"] += 1
        counts["parse_success"] += int(row["verification_parse_success"])
        row_positive = False
        for decision in row["verified_assertions"]:
            counts["candidates"] += 1
            counts["valid"] += int(decision["classification_valid"])
            judgements[decision["judgement"]] += 1
            if decision["positive"]:
                row_positive = True
                relation = decision["relation_type"]
                relations[relation] += 1
                sources[str(row["source"])][relation] += 1
                edges.append(
                    {
                        "assertion_id": decision["assertion_id"],
                        "proposition_id": row["proposition_id"],
                        "source": row["source"],
                        "proposition": row["claim"],
                        "subject_mention_id": decision["subject"]["mention_id"],
                        "subject_text": decision["subject"]["surface"],
                        "subject_kind": decision["subject"]["kind"],
                        "subject_umls_cui": decision["subject"].get("umls_cui"),
                        "relation_type": relation,
                        "object_mention_id": decision["object"]["mention_id"],
                        "object_text": decision["object"]["surface"],
                        "object_kind": decision["object"]["kind"],
                        "object_umls_cui": decision["object"].get("umls_cui"),
                        "polarity": decision["polarity"],
                        "negated": decision["polarity"] == "negated",
                        "uncertain": decision["polarity"] == "uncertain",
                        "trigger": decision["trigger"],
                        "evidence": {"text": row["claim"], "start": 0, "end": len(str(row["claim"]))},
                    }
                )
        counts["propositions_with_positive_edge"] += int(row_positive)
        if index % 50 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)

    by_id = {str(row["proposition_id"]): row for row in rows}
    fixed_review = []
    for proposition_id in fixed_review_ids(args):
        row = by_id[proposition_id]
        fixed_review.append(
            {
                "proposition_id": proposition_id,
                "source": row["source"],
                "proposition": row["claim"],
                "core_mentions": row["core_mentions"],
                "rejected_mentions": [
                    {
                        "surface": value.get("surface"),
                        "kind": value.get("kind"),
                        "reasons": value.get("rejection_reasons"),
                    }
                    for value in row["mention_decisions"]
                    if not value.get("keep")
                ],
                "v6_assertions": [
                    {
                        "subject": value["subject"]["surface"],
                        "relation": value["relation_type"],
                        "object": value["object"]["surface"],
                        "trigger": value["trigger"]["text"],
                        "judgement": value["judgement"],
                        "polarity": value["polarity"],
                        "valid": value["classification_valid"],
                    }
                    for value in row["verified_assertions"]
                ],
                "manual_gold": {
                    "core_entities_complete": None,
                    "core_entities_precise": None,
                    "gold_relations": [],
                    "missed_relations": [],
                    "notes": "",
                },
            }
        )

    positive_review = [
        {
            **edge,
            "manual_audit": {
                "subject_correct": None,
                "relation_correct": None,
                "object_correct": None,
                "direction_correct": None,
                "polarity_correct": None,
                "exact_edge_correct": None,
                "notes": "",
            },
        }
        for edge in balanced_positive_sample(edges, args.review_size, args.seed)
    ]
    candidates = list(read_jsonl(args.output_root / "relation_candidates_v6.jsonl"))
    preflight = json.loads((args.output_root / "preflight_summary.json").read_text(encoding="utf-8"))
    summary = {
        "run_version": RUN_VERSION,
        "purpose": "Precision-first feasibility test; not a final KG and not a medical-truth validator.",
        "input": {"propositions": len(rows), "by_source": dict(Counter(str(row["source"]) for row in rows))},
        "candidate_construction": preflight["candidate_construction"],
        "verification": {
            "parse_success": counts["parse_success"],
            "parse_success_rate": counts["parse_success"] / max(counts["propositions"], 1),
            "candidates": counts["candidates"],
            "valid_decisions": counts["valid"],
            "valid_decision_rate": counts["valid"] / max(counts["candidates"], 1),
            "judgements": dict(judgements),
        },
        "positive_edges": {
            "edges": len(edges),
            "propositions": counts["propositions_with_positive_edge"],
            "proposition_coverage": counts["propositions_with_positive_edge"] / max(counts["propositions"], 1),
            "relations": dict(relations),
            "relations_by_source": {key: dict(value) for key, value in sources.items()},
        },
        "manual_evaluation": {
            "fixed_proposition_review": str((args.output_root / "fixed_proposition_review_200.jsonl").resolve()),
            "positive_edge_review": str((args.output_root / "positive_edge_review_200.jsonl").resolve()),
            "pilot_pass": {
                "exact_edge_precision_min": 0.75,
                "direction_accuracy_min": 0.90,
                "wrong_edge_rate_max": 0.10,
                "correct_edge_candidate_cohort_coverage_min": 0.60,
                "correct_edge_all_proposition_coverage_min": 0.12,
            },
            "decision_rule": "Do not expand relation coverage or scale the KG if any threshold fails on the fixed audit.",
            "status": "pending_manual_audit",
        },
        "comparison_note": {
            "v5_root": str(args.v5_root.resolve()),
            "same_source_cohort": True,
            "same_fixed_200_proposition_ids": True,
            "changed_variable": "core-node and trigger-constrained fixed assertions replace broad all-pair relation classification",
        },
    }
    if len(candidates) != len(rows):
        raise RuntimeError("Candidate and verification row counts differ")
    atomic_jsonl(args.output_root / "candidate_edges_v6.jsonl", edges)
    atomic_jsonl(args.output_root / "fixed_proposition_review_200.jsonl", fixed_review)
    atomic_jsonl(args.output_root / "positive_edge_review_200.jsonl", positive_review)
    atomic_json(args.output_root / "summary.json", summary)
    atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "summary": str((args.output_root / "summary.json").resolve())})
    emit("complete", total=len(rows), completed=len(rows))


def relation_specs_for_manifest() -> dict[str, Any]:
    return {
        relation: {
            "subjects": sorted(spec["subjects"]),
            "objects": sorted(spec["objects"]),
            "patterns": list(spec["patterns"]),
            "description": spec["description"],
            "symmetric": spec["symmetric"],
        }
        for relation, spec in RELATION_SPECS.items()
    }


def contract(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], review_ids: Sequence[str]) -> dict[str, Any]:
    value = {
        "run_version": RUN_VERSION,
        "hypothesis": "Noun-headed core entities plus explicit trigger/type-constrained fixed assertions can reach at least 75% exact semantic-edge precision and preserve usable coverage within the trigger-eligible cohort.",
        "changed_variable": "v5 broad all-pair relation classification is replaced by precision-first core-node filtering, trigger/type candidate generation, and binary assertion verification",
        "input": file_identity(args.v2_root / "raw_statements_v2.jsonl"),
        "cohort": {"propositions": len(rows), "by_source": dict(Counter(str(row["source"]) for row in rows))},
        "models": {"syntax": args.spacy_model, "verifier": str(args.qwen_model.resolve())},
        "prompts": {"verifier_sha256": sha256_text(VERIFY_SYSTEM)},
        "relation_specs": relation_specs_for_manifest(),
        "generation": {
            "temperature": 0.0,
            "batch_size": args.generation_batch_size,
            "shard_size": args.generation_shard_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "max_candidates_per_proposition": args.max_candidates_per_proposition,
            "max_pair_distance_chars": args.max_pair_distance_chars,
            "seed": args.seed,
        },
        "evaluation": {
            "fixed_v4_review_ids_sha256": sha256_text("\n".join(review_ids)),
            "review_size": len(review_ids),
            "pilot_pass": {"exact_edge_precision_min": 0.75, "direction_accuracy_min": 0.90, "wrong_edge_rate_max": 0.10, "correct_edge_candidate_cohort_coverage_min": 0.60, "correct_edge_all_proposition_coverage_min": 0.12},
        },
        "code": {"script": file_identity(Path(__file__).resolve())},
        "uses_prior_relation_predictions_as_model_input": False,
        "uses_questions_answers_gold_or_semantic_labels": False,
        "scope": "core biomedical entity relations only; literals, measurements, dose, time, and study metadata remain qualifiers outside this edge pilot",
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
                args.output_root / "relation_candidates_v6.jsonl",
                args.output_root / "verification_shards",
                args.output_root / "assertion_verifications_v6.jsonl",
                args.output_root / "COMPLETE.json",
            )
            if any(item.exists() for item in durable):
                raise RuntimeError("Trigger-classifier v6 resume contract mismatch; use a new output root")
            atomic_json(path, value)
    else:
        atomic_json(path, value)


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    python = args.entity_python if mode == "candidates" else args.llm_python
    return [
        str(python),
        str(Path(__file__).resolve()),
        "--mode", mode,
        "--v2-root", str(args.v2_root),
        "--v4-root", str(args.v4_root),
        "--v5-root", str(args.v5_root),
        "--output-root", str(args.output_root),
        "--qwen-model", str(args.qwen_model),
        "--entity-python", str(args.entity_python),
        "--llm-python", str(args.llm_python),
        "--spacy-model", args.spacy_model,
        "--generation-batch-size", str(args.generation_batch_size),
        "--generation-shard-size", str(args.generation_shard_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--max-output-tokens", str(args.max_output_tokens),
        "--max-candidates-per-proposition", str(args.max_candidates_per_proposition),
        "--max-pair-distance-chars", str(args.max_pair_distance_chars),
        "--review-size", str(args.review_size),
        "--seed", str(args.seed),
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
        elif value and "FutureWarning" not in value and not value.startswith("Loading weights:"):
            progress.log(value)
    code = process.wait()
    if code:
        raise RuntimeError(f"Worker {mode} failed with exit code {code}")


def orchestrate(args: argparse.Namespace) -> None:
    if min(args.generation_batch_size, args.generation_shard_size, args.max_candidates_per_proposition, args.max_pair_distance_chars) <= 0:
        raise ValueError("Batch, candidate, and distance limits must be positive")
    for path in (args.qwen_model, args.entity_python, args.llm_python, args.v4_root / "review_sample_200.jsonl"):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = load_rows(args)
    review_ids = fixed_review_ids(args)
    ensure_manifest(args, contract(args, rows, review_ids))
    progress = WorkflowProgress(
        [
            "spaCy core-node filtering and trigger candidate construction",
            "audit candidates and verifier prompts",
            "Qwen fixed assertion source-fidelity verification",
            "aggregate edges and fixed audit material",
        ],
        [45, 20, 300, 20],
    )
    progress.log("[workflow plan] same 1,000 propositions -> core noun entities -> explicit trigger/type assertions -> Qwen binary verification -> fixed 200-case audit")
    progress.log("[success gate] exact precision >=75%, direction >=90%, wrong edges <=10%, correct-edge coverage >=60% of eligible and >=12% of all propositions; otherwise stop")
    run_worker(args, "candidates", progress, 1)
    progress.log(f"[stage 1/4 complete] output={args.output_root / 'relation_candidates_v6.jsonl'}")
    candidate_rows = list(read_jsonl(args.output_root / "relation_candidates_v6.jsonl"))
    if len(candidate_rows) != len(rows):
        raise RuntimeError(f"Candidate row mismatch: {len(candidate_rows)}/{len(rows)}")
    summary = candidate_summary(candidate_rows)
    prompt_audit = prompt_token_audit(args, candidate_rows)
    atomic_json(args.output_root / "preflight_summary.json", {"candidate_construction": summary, "prompt_tokens": prompt_audit})
    progress.start(2, len(candidate_rows), len(candidate_rows))
    progress.log(
        f"[stage 2/4 complete] core_mentions={summary['core_mentions']:,} candidates={summary['relation_candidates']:,} "
        f"covered_props={summary['propositions_with_candidates']:,}/1,000 max_tokens={prompt_audit['max_prompt_plus_output_tokens']:,}"
    )
    if args.preflight_only:
        progress.close()
        progress.log("[preflight complete] exact command paths, source cohort, manifest, spaCy filtering, relation candidates, and prompt lengths passed; Qwen was not loaded")
        return
    run_worker(args, "classify", progress, 3)
    progress.log(f"[stage 3/4 complete] output={args.output_root / 'assertion_verifications_v6.jsonl'}")
    run_worker(args, "aggregate", progress, 4)
    progress.log(f"[stage 4/4 complete] summary={args.output_root / 'summary.json'}")
    progress.close()
    progress.log(f"[workflow complete] result={args.output_root}")


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate":
        orchestrate(args)
    elif args.mode == "candidates":
        candidate_worker(args)
    elif args.mode == "classify":
        classify_worker(args)
    elif args.mode == "aggregate":
        aggregate_worker(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

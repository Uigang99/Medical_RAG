"""Pure helpers for the biomedical-KG MVP.

The expensive NLP pipeline lives in the command-line extractor.  This module
keeps the relation schema, semantic-type mapping, and aggregation logic free of
spaCy/scispaCy imports so they can be tested in the repository environment.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


SPACE_RE = re.compile(r"\s+")


# UMLS semantic-type groups.  The mapping is intentionally coarse: the MVP is
# testing whether stable biomedical entities and evidence-backed relations can
# be recovered, not claiming a complete biomedical ontology.
TUI_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "DiagnosticTest",
        frozenset({"T034", "T059", "T060"}),
    ),
    (
        "GeneOrProtein",
        frozenset({"T028", "T085", "T086", "T087", "T088", "T116", "T126"}),
    ),
    (
        "DrugOrChemical",
        frozenset(
            {
                "T103", "T104", "T109", "T110", "T111", "T114", "T115",
                "T116", "T118", "T119", "T120", "T121", "T122", "T123",
                "T124", "T125", "T126", "T127", "T129", "T130", "T131",
                "T192", "T195", "T196", "T197", "T200", "T203",
            }
        ),
    ),
    (
        "DiseaseOrDisorder",
        frozenset({"T019", "T020", "T037", "T046", "T047", "T048", "T049", "T050", "T190", "T191"}),
    ),
    (
        "FindingOrSymptom",
        frozenset({"T033", "T034", "T040", "T041", "T042", "T043", "T044", "T045", "T184"}),
    ),
    (
        "ProcedureOrTreatment",
        frozenset({"T058", "T059", "T060", "T061", "T062", "T063", "T064", "T065"}),
    ),
    (
        "Anatomy",
        frozenset({"T017", "T021", "T022", "T023", "T024", "T025", "T026", "T029", "T030"}),
    ),
    (
        "Organism",
        frozenset({"T001", "T002", "T004", "T005", "T007", "T008", "T010", "T011", "T012", "T013", "T014", "T015", "T016", "T194", "T204"}),
    ),
)


TREATMENT = frozenset({"DrugOrChemical", "ProcedureOrTreatment"})
CONDITION = frozenset({"DiseaseOrDisorder", "FindingOrSymptom"})
FINDING = frozenset({"FindingOrSymptom", "DiagnosticTest", "DiseaseOrDisorder"})
CAUSAL_SUBJECT = frozenset(
    {
        "DrugOrChemical",
        "ProcedureOrTreatment",
        "DiseaseOrDisorder",
        "FindingOrSymptom",
        "GeneOrProtein",
        "Organism",
    }
)


NEGATION_RE = re.compile(
    r"\b(no|not|never|neither|without|absence of|absent|fails? to|unlikely)\b",
    flags=re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(may|might|could|possibly|potentially|suggests?|suspected|probable|likely)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RelationAssertion:
    subject_id: str
    predicate: str
    object_id: str
    polarity: str
    certainty: str
    rule_id: str
    rule_score: float
    symmetric: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_surface(value: str) -> str:
    """Normalize an unlinked surface without pretending it is a UMLS entity."""

    return SPACE_RE.sub(" ", value.strip().casefold())


def local_entity_id(surface: str) -> str:
    normalized = normalize_surface(surface)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"LOCAL:{digest}"


def category_from_tuis(tuis: Iterable[str]) -> str:
    values = frozenset(str(value) for value in tuis)
    for category, members in TUI_GROUPS:
        if values & members:
            return category
    return "OtherBiomedical"


def _orientation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    subject_categories: frozenset[str],
    object_categories: frozenset[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    if left["category"] in subject_categories and right["category"] in object_categories:
        return left, right
    if right["category"] in subject_categories and left["category"] in object_categories:
        return right, left
    return None


def _assertion(
    subject: Mapping[str, Any],
    predicate: str,
    obj: Mapping[str, Any],
    *,
    sentence_text: str,
    rule_id: str,
    score: float,
    symmetric: bool = False,
) -> RelationAssertion:
    subject_id, object_id = str(subject["entity_id"]), str(obj["entity_id"])
    if symmetric and object_id < subject_id:
        subject_id, object_id = object_id, subject_id
    return RelationAssertion(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        polarity="negated" if NEGATION_RE.search(sentence_text) else "affirmed",
        certainty="uncertain" if UNCERTAINTY_RE.search(sentence_text) else "asserted",
        rule_id=rule_id,
        rule_score=float(score),
        symmetric=symmetric,
    )


def infer_relation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    sentence_text: str,
    between_text: str,
) -> RelationAssertion | None:
    """Infer one conservative typed relation for a linked same-sentence pair.

    The score is a fixed rule strength, not a calibrated probability.  If no
    typed rule fires, callers may emit a separate CO_OCCURS_WITH assertion.
    """

    between = normalize_surface(between_text)
    full = normalize_surface(sentence_text)

    oriented = _orientation(left, right, TREATMENT, CONDITION)
    if oriented and re.search(r"\b(treat|treats|treated|therapy|therapeutic|management)\b", between):
        return _assertion(oriented[0], "TREATS", oriented[1], sentence_text=sentence_text, rule_id="treat_between_v1", score=0.90)
    if oriented and re.search(r"\b(prevent|prevents|prevented|prevention|prophylaxis|protects?)\b", between):
        return _assertion(oriented[0], "PREVENTS", oriented[1], sentence_text=sentence_text, rule_id="prevent_between_v1", score=0.90)
    if oriented and re.search(r"\b(contraindicated|contraindication|avoid|avoided)\b", full):
        return _assertion(oriented[0], "CONTRAINDICATED_FOR", oriented[1], sentence_text=sentence_text, rule_id="contraindication_sentence_v1", score=0.86)

    chemical_pair = left["category"] == right["category"] == "DrugOrChemical"
    if chemical_pair and re.search(r"\b(interact|interacts|interaction|coadministration|co-administered)\b", full):
        return _assertion(left, "INTERACTS_WITH", right, sentence_text=sentence_text, rule_id="interaction_sentence_v1", score=0.88, symmetric=True)

    diagnostic = _orientation(left, right, FINDING, frozenset({"DiseaseOrDisorder"}))
    if diagnostic and re.search(r"\b(diagnos|detect|screen|test)\w*\b", between):
        return _assertion(diagnostic[0], "DIAGNOSES_OR_DETECTS", diagnostic[1], sentence_text=sentence_text, rule_id="diagnostic_between_v1", score=0.84)
    if diagnostic and re.search(r"\b(indicat|marker|predict)\w*\b", between):
        return _assertion(diagnostic[0], "INDICATES", diagnostic[1], sentence_text=sentence_text, rule_id="indicates_between_v1", score=0.82)

    # Direction for causal/association language follows entity order, except
    # for an explicit passive "... caused by ..." construction.
    if re.search(r"\b(cause|causes|caused|induce|induces|induced|lead(?:s)? to|result(?:s)? in)\b", between):
        subject, obj = (right, left) if re.search(r"\b(caused|induced) by\b", between) else (left, right)
        if subject["category"] in CAUSAL_SUBJECT and obj["category"] in CONDITION:
            return _assertion(subject, "CAUSES", obj, sentence_text=sentence_text, rule_id="causal_between_v1", score=0.84)

    if re.search(r"\b(risk factor|increases? (?:the )?risk|predispos\w*)\b", between):
        return _assertion(left, "RISK_FACTOR_FOR", right, sentence_text=sentence_text, rule_id="risk_between_v1", score=0.84)

    if re.search(r"\b(adverse effect|side effect|toxicity|toxic|complication)\b", full):
        oriented = _orientation(left, right, TREATMENT, CONDITION)
        if oriented:
            return _assertion(oriented[0], "HAS_ADVERSE_EFFECT", oriented[1], sentence_text=sentence_text, rule_id="adverse_sentence_v1", score=0.78)

    if re.search(r"\b(associated with|association with|correlated with|linked to)\b", between):
        return _assertion(left, "ASSOCIATED_WITH", right, sentence_text=sentence_text, rule_id="association_between_v1", score=0.76, symmetric=True)
    return None


def cooccurrence_assertion(
    left: Mapping[str, Any], right: Mapping[str, Any], *, sentence_text: str
) -> RelationAssertion:
    return _assertion(
        left,
        "CO_OCCURS_WITH",
        right,
        sentence_text=sentence_text,
        rule_id="same_sentence_cooccurrence_v1",
        score=0.25,
        symmetric=True,
    )


def aggregate_edge_assertions(assertions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate evidence without collapsing polarity or certainty conflicts."""

    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in assertions:
        key = (
            str(row["subject_id"]),
            str(row["predicate"]),
            str(row["object_id"]),
            str(row["polarity"]),
            str(row["certainty"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        chunks = sorted({str(row["chunk_id"]) for row in rows})
        sources = Counter(str(row["source"]) for row in rows)
        rules = Counter(str(row["rule_id"]) for row in rows)
        output.append(
            {
                "subject_id": key[0],
                "predicate": key[1],
                "object_id": key[2],
                "polarity": key[3],
                "certainty": key[4],
                "assertion_count": len(rows),
                "distinct_chunk_count": len(chunks),
                "chunk_ids": chunks,
                "source_counts": dict(sorted(sources.items())),
                "rule_counts": dict(sorted(rules.items())),
                "max_rule_score": max(float(row["rule_score"]) for row in rows),
            }
        )
    return output

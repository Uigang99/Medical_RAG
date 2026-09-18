#!/usr/bin/env python3
"""Evaluate statement-graph stages against the source-only Codex reference.

This evaluator deliberately separates mention extraction, candidate generation,
and accepted relations.  The reference is a structural annotation of what the
source proposition says; it is not a physician-verified medical truth set.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1"
DEFAULT_SOURCE = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2/raw_statements_v2.jsonl"
DEFAULT_V5 = ROOT / "datasets/kg/rag2_statement_graph_pair_classifier_pilot_1k_v5"
DEFAULT_V6 = ROOT / "datasets/kg/rag2_statement_graph_trigger_classifier_pilot_1k_v6_2"
DEFAULT_OUTPUT = DEFAULT_REFERENCE / "automatic_kg_comparison"

CORE_TYPES = {
    "condition",
    "drug_or_chemical",
    "procedure_or_intervention",
    "finding_or_outcome",
    "anatomy",
    "organism",
    "gene_or_protein",
    "population",
}
SYMMETRIC = {"ASSOCIATED_WITH", "COMPARED_WITH", "BINDS_OR_INTERACTS_WITH"}
SHARED_RELATIONS = {
    "IS_A", "PART_OF", "LOCATED_IN", "TREATS", "MANAGES", "PREVENTS",
    "CONTRAINDICATED_FOR", "ADMINISTERED_TO", "CAUSES", "ASSOCIATED_WITH",
    "INCREASES_RISK_OF", "DECREASES_RISK_OF", "INCREASES", "DECREASES",
    "INHIBITS", "ACTIVATES", "BINDS_OR_INTERACTS_WITH", "EXPRESSES",
    "ENCODES", "DIAGNOSES", "DETECTS", "INDICATES", "HAS_SIGN_OR_SYMPTOM",
    "COMPARED_WITH",
}
TYPE_MAP = {
    "condition": "condition",
    "drug_or_chemical": "drug_or_chemical",
    "procedure_or_intervention": "procedure_or_intervention",
    "finding_or_outcome": "finding_or_outcome",
    "anatomy": "anatomy",
    "location": "anatomy",
    "organism": "organism",
    "gene_or_protein": "gene_or_protein",
    "population": "population",
}
RELATION_ALIAS = {"PRODUCES": "CAUSES", "RESULTS_IN": "CAUSES", "SECRETES": "EXPRESSES"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


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


def overlap_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
    if not inter:
        return 0.0
    shorter = min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"]))
    return inter / shorter if shorter else 0.0


def compatible_type(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    return gold["entity_type"] == TYPE_MAP.get(pred.get("kind"))


def align_entities(
    gold: list[dict[str, Any]], pred: list[dict[str, Any]], *, exact: bool, require_type: bool
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if require_type and not compatible_type(g, p):
                continue
            same = int(g["start"]) == int(p["start"]) and int(g["end"]) == int(p["end"])
            score = 1.0 if same else (0.0 if exact else overlap_score(g, p))
            if score >= (1.0 if exact else 0.5):
                candidates.append((score + (0.1 if same else 0.0), gi, pi))
    matches: list[tuple[int, int]] = []
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _, gi, pi in sorted(candidates, reverse=True):
        if gi not in used_g and pi not in used_p:
            matches.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)
    return matches, sorted(set(range(len(gold))) - used_g), sorted(set(range(len(pred))) - used_p)


def mention_key(mention: dict[str, Any]) -> tuple[int, int]:
    return int(mention["start"]), int(mention["end"])


def endpoint_matches(gold: dict[str, Any], pred: dict[str, Any], *, exact: bool, typed: bool) -> bool:
    if typed and not compatible_type(gold, pred):
        return False
    if exact:
        return mention_key(gold) == mention_key(pred)
    return overlap_score(gold, pred) >= 0.5


def polarity_of(edge: dict[str, Any]) -> str:
    if edge.get("negated") or edge.get("modality") == "negated":
        return "negated"
    if edge.get("uncertain") or edge.get("modality") in {"may", "might", "possible", "uncertain"}:
        return "uncertain"
    return "asserted"


def relation_type(edge: dict[str, Any], aliases: bool) -> str:
    value = str(edge["relation_type"]).upper()
    return RELATION_ALIAS.get(value, value) if aliases else value


def relation_equal(
    gold_rel: dict[str, Any], pred_rel: dict[str, Any], gold_entities: dict[str, dict[str, Any]],
    mention_by_id: dict[str, dict[str, Any]], *, exact_endpoint: bool, typed_endpoint: bool,
    require_relation: bool, require_polarity: bool, aliases: bool,
) -> bool:
    subject = mention_by_id.get(pred_rel.get("subject_mention_id", ""))
    obj = mention_by_id.get(pred_rel.get("object_mention_id", ""))
    if not subject or not obj:
        return False
    gs = gold_entities[gold_rel["subject_entity_id"]]
    go = gold_entities[gold_rel["object_entity_id"]]
    normal = endpoint_matches(gs, subject, exact=exact_endpoint, typed=typed_endpoint) and endpoint_matches(
        go, obj, exact=exact_endpoint, typed=typed_endpoint
    )
    reverse = endpoint_matches(gs, obj, exact=exact_endpoint, typed=typed_endpoint) and endpoint_matches(
        go, subject, exact=exact_endpoint, typed=typed_endpoint
    )
    gtype = gold_rel["relation_type"]
    ptype = relation_type(pred_rel, aliases)
    if gtype in SYMMETRIC:
        endpoints = normal or reverse
    else:
        endpoints = normal
    if not endpoints:
        return False
    if require_relation and gtype != ptype:
        return False
    if require_polarity and gold_rel["polarity"] != polarity_of(pred_rel):
        return False
    return True


def greedy_relation_score(
    gold: list[dict[str, Any]], pred: list[dict[str, Any]], entities: dict[str, dict[str, Any]],
    mentions: dict[str, dict[str, Any]], **kwargs: Any,
) -> tuple[dict[str, Any], list[int], list[int]]:
    possible: list[tuple[int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if relation_equal(g, p, entities, mentions, **kwargs):
                possible.append((gi, pi))
    used_g: set[int] = set()
    used_p: set[int] = set()
    for gi, pi in possible:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    return f1_counts(len(used_g), len(pred), len(gold)), sorted(set(range(len(gold))) - used_g), sorted(
        set(range(len(pred))) - used_p
    )


def flatten_v5_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for pair in row.get("candidate_pairs", []):
            result.append({
                "proposition_id": row["proposition_id"],
                "subject_mention_id": pair["left"]["mention_id"],
                "object_mention_id": pair["right"]["mention_id"],
                "relation_type": "CANDIDATE",
            })
    return result


def flatten_v6_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for candidate in row.get("relation_candidates", []):
            result.append({
                "proposition_id": row["proposition_id"],
                "subject_mention_id": candidate["subject"]["mention_id"],
                "object_mention_id": candidate["object"]["mention_id"],
                "relation_type": candidate["relation_type"],
            })
    return result


def entity_metrics(
    reference_rows: list[dict[str, Any]], pred_by_prop: dict[str, list[dict[str, Any]]], cohort: set[str]
) -> dict[str, Any]:
    totals: dict[str, list[int]] = {name: [0, 0, 0] for name in ["exact_typed", "relaxed_typed", "exact_span", "relaxed_span"]}
    excluded_noncore = 0
    matched_for_norm: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in reference_rows:
        pid = row["proposition_id"]
        if pid not in cohort:
            continue
        gold = row["entities"]
        all_pred = pred_by_prop.get(pid, [])
        excluded_noncore += sum(1 for p in all_pred if TYPE_MAP.get(p.get("kind")) not in CORE_TYPES)
        pred = [p for p in all_pred if TYPE_MAP.get(p.get("kind")) in CORE_TYPES]
        relaxed_typed_matches, _, _ = align_entities(gold, pred, exact=False, require_type=True)
        for entity in gold:
            by_type[entity["entity_type"]][2] += 1
        for entity in pred:
            by_type[TYPE_MAP[entity["kind"]]][1] += 1
        for gi, _ in relaxed_typed_matches:
            by_type[gold[gi]["entity_type"]][0] += 1
        for name, exact, typed in [
            ("exact_typed", True, True), ("relaxed_typed", False, True),
            ("exact_span", True, False), ("relaxed_span", False, False),
        ]:
            matches, _, _ = align_entities(gold, pred, exact=exact, require_type=typed)
            totals[name][0] += len(matches)
            totals[name][1] += len(pred)
            totals[name][2] += len(gold)
            if name == "exact_typed":
                matched_for_norm.extend((gold[gi], pred[pi]) for gi, pi in matches)
    norm_covered = [pair for pair in matched_for_norm if pair[1].get("umls_cui")]
    name_matches = sum(
        normalized(g["canonical_name"]) == normalized(p.get("umls_name") or "") for g, p in norm_covered
    )
    surface_matches = sum(
        normalized(g["canonical_name"]) == normalized(p.get("surface_normalized") or p.get("surface") or "")
        for g, p in matched_for_norm
    )
    gold_groups = Counter()
    pred_groups = Counter()
    intersections = Counter()
    for gold_entity, pred_entity in matched_for_norm:
        gold_key = f"{gold_entity['entity_type']}::{normalized(gold_entity['canonical_name'])}"
        pred_type = TYPE_MAP.get(pred_entity.get("kind"), pred_entity.get("kind", "unknown"))
        pred_key = (
            f"UMLS::{pred_entity['umls_cui']}" if pred_entity.get("umls_cui")
            else f"LOCAL::{pred_type}::{normalized(pred_entity.get('surface_normalized') or pred_entity.get('surface') or '')}"
        )
        gold_groups[gold_key] += 1
        pred_groups[pred_key] += 1
        intersections[(gold_key, pred_key)] += 1
    choose2 = lambda count: count * (count - 1) // 2
    gold_pairs = sum(choose2(value) for value in gold_groups.values())
    pred_pairs = sum(choose2(value) for value in pred_groups.values())
    shared_pairs = sum(choose2(value) for value in intersections.values())
    return {
        name: f1_counts(*values) for name, values in totals.items()
    } | {
        "predicted_noncore_mentions_excluded_from_core_metric": excluded_noncore,
        "by_type_relaxed_typed": {key: f1_counts(*value) for key, value in sorted(by_type.items())},
        "normalization_proxy": {
            "exact_typed_matches": len(matched_for_norm),
            "umls_linked": len(norm_covered),
            "umls_link_coverage": len(norm_covered) / len(matched_for_norm) if matched_for_norm else 0.0,
            "gold_canonical_equals_umls_name": name_matches,
            "gold_canonical_equals_umls_name_rate": name_matches / len(norm_covered) if norm_covered else 0.0,
            "gold_canonical_equals_predicted_surface": surface_matches,
            "pairwise_cluster": f1_counts(shared_pairs, pred_pairs, gold_pairs),
            "gold_concepts_with_repeated_matched_mentions": sum(value >= 2 for value in gold_groups.values()),
            "note": "Lexical proxy only; reference does not assign authoritative UMLS CUIs.",
        },
    }


def relation_metrics(
    reference_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], mention_by_id: dict[str, dict[str, Any]],
    cohort: set[str], *, candidates: bool,
) -> dict[str, Any]:
    ref_by_prop = {row["proposition_id"]: row for row in reference_rows if row["proposition_id"] in cohort}
    pred_by_prop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in pred_rows:
        if edge.get("proposition_id") in cohort:
            pred_by_prop[edge["proposition_id"]].append(edge)
    aggregate: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    nonshared = Counter()
    errors = Counter()
    per_relation: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for pid, row in ref_by_prop.items():
        gold_entities = {e["entity_id"]: e for e in row["entities"]}
        gold = row["relations"]
        pred_all = pred_by_prop.get(pid, [])
        if candidates:
            pred = pred_all
            metric_specs = {
                "strict_endpoint_candidate_recall": dict(exact_endpoint=True, typed_endpoint=True, require_relation=False, require_polarity=False, aliases=False),
                "relaxed_endpoint_candidate_recall": dict(exact_endpoint=False, typed_endpoint=True, require_relation=False, require_polarity=False, aliases=False),
                "relaxed_untyped_candidate_recall": dict(exact_endpoint=False, typed_endpoint=False, require_relation=False, require_polarity=False, aliases=False),
            }
        else:
            pred = [p for p in pred_all if relation_type(p, False) in SHARED_RELATIONS]
            pred_alias = [p for p in pred_all if relation_type(p, True) in SHARED_RELATIONS]
            for p in pred_all:
                if relation_type(p, False) not in SHARED_RELATIONS:
                    nonshared[relation_type(p, False)] += 1
            metric_specs = {
                "strict_exact": dict(exact_endpoint=True, typed_endpoint=True, require_relation=True, require_polarity=True, aliases=False),
                "relaxed_typed_with_polarity": dict(exact_endpoint=False, typed_endpoint=True, require_relation=True, require_polarity=True, aliases=False),
                "relaxed_typed_without_polarity": dict(exact_endpoint=False, typed_endpoint=True, require_relation=True, require_polarity=False, aliases=False),
                "relaxed_untyped_without_polarity": dict(exact_endpoint=False, typed_endpoint=False, require_relation=True, require_polarity=False, aliases=False),
                "relaxed_alias_without_polarity": dict(exact_endpoint=False, typed_endpoint=True, require_relation=True, require_polarity=False, aliases=True),
            }
        for name, kwargs in metric_specs.items():
            metric_pred = pred_alias if (not candidates and kwargs.get("aliases")) else pred
            score, _, _ = greedy_relation_score(gold, metric_pred, gold_entities, mention_by_id, **kwargs)
            aggregate[name][0] += score["true_positive"]
            aggregate[name][1] += score["predicted"]
            aggregate[name][2] += score["gold"]
        if not candidates:
            _, missing_gold, wrong_pred = greedy_relation_score(
                gold, pred, gold_entities, mention_by_id, exact_endpoint=False, typed_endpoint=True,
                require_relation=True, require_polarity=False, aliases=False,
            )
            for gi in missing_gold:
                g = gold[gi]
                same_endpoints = any(relation_equal(
                    g, p, gold_entities, mention_by_id, exact_endpoint=False, typed_endpoint=True,
                    require_relation=False, require_polarity=False, aliases=False,
                ) for p in pred)
                errors["gold_missed_with_endpoints_present"] += int(same_endpoints)
                errors["gold_missed_endpoint_absent"] += int(not same_endpoints)
            for pi in wrong_pred:
                p = pred[pi]
                same_endpoints = any(relation_equal(
                    g, p, gold_entities, mention_by_id, exact_endpoint=False, typed_endpoint=True,
                    require_relation=False, require_polarity=False, aliases=False,
                ) for g in gold)
                errors["prediction_wrong_relation_on_gold_endpoints"] += int(same_endpoints)
                errors["prediction_endpoint_not_in_gold_relation"] += int(not same_endpoints)
            for rel_type in SHARED_RELATIONS:
                gsub = [g for g in gold if g["relation_type"] == rel_type]
                psub = [p for p in pred if relation_type(p, False) == rel_type]
                score, _, _ = greedy_relation_score(
                    gsub, psub, gold_entities, mention_by_id, exact_endpoint=False, typed_endpoint=True,
                    require_relation=True, require_polarity=False, aliases=False,
                )
                per_relation[rel_type][0] += score["true_positive"]
                per_relation[rel_type][1] += score["predicted"]
                per_relation[rel_type][2] += score["gold"]
    result = {name: f1_counts(*values) for name, values in aggregate.items()}
    if candidates:
        for value in result.values():
            value["candidate_recall"] = value["recall"]
        return result
    result["predicted_nonshared_relations_excluded"] = dict(nonshared)
    result["error_taxonomy_relaxed_typed"] = dict(errors)
    result["by_relation_relaxed_typed_without_polarity"] = {
        key: f1_counts(*value) for key, value in sorted(per_relation.items()) if value[1] or value[2]
    }
    return result


def relation_diagnostics(
    reference_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], mention_by_id: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ref = {row["proposition_id"]: row for row in reference_rows}
    pred_by_prop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in pred_rows:
        pred_by_prop[edge["proposition_id"]].append(edge)
    predicted_diagnostics: list[dict[str, Any]] = []
    missed_gold: list[dict[str, Any]] = []
    for pid, row in ref.items():
        entities = {e["entity_id"]: e for e in row["entities"]}
        gold = row["relations"]
        predictions = pred_by_prop.get(pid, [])
        for pred in predictions:
            ptype = relation_type(pred, False)
            if ptype not in SHARED_RELATIONS:
                category = "nonshared_relation_not_scored"
                endpoint_gold: list[dict[str, Any]] = []
            else:
                endpoint_gold = [g for g in gold if relation_equal(
                    g, pred, entities, mention_by_id, exact_endpoint=False, typed_endpoint=True,
                    require_relation=False, require_polarity=False, aliases=False,
                )]
                exact_gold = [g for g in endpoint_gold if g["relation_type"] == ptype]
                if exact_gold:
                    category = "correct_relaxed_typed_without_polarity"
                elif endpoint_gold:
                    category = "correct_endpoints_wrong_relation"
                else:
                    category = "endpoint_or_type_mismatch"
            predicted_diagnostics.append({
                "proposition_id": pid,
                "claim": row["claim"],
                "curation_confidence": row["curation_confidence"],
                "category": category,
                "predicted": {
                    "subject": pred.get("subject_text"), "relation": ptype, "object": pred.get("object_text"),
                    "polarity": polarity_of(pred),
                },
                "gold_on_same_endpoints": [
                    {
                        "subject": entities[g["subject_entity_id"]]["surface"],
                        "relation": g["relation_type"],
                        "object": entities[g["object_entity_id"]]["surface"],
                        "polarity": g["polarity"],
                    } for g in endpoint_gold
                ],
            })
        for g in gold:
            found = any(relation_equal(
                g, pred, entities, mention_by_id, exact_endpoint=False, typed_endpoint=True,
                require_relation=True, require_polarity=False, aliases=False,
            ) for pred in predictions if relation_type(pred, False) in SHARED_RELATIONS)
            if not found:
                missed_gold.append({
                    "proposition_id": pid,
                    "claim": row["claim"],
                    "curation_confidence": row["curation_confidence"],
                    "gold": {
                        "subject": entities[g["subject_entity_id"]]["surface"],
                        "relation": g["relation_type"],
                        "object": entities[g["object_entity_id"]]["surface"],
                        "polarity": g["polarity"],
                    },
                })
    return predicted_diagnostics, missed_gold


def reference_audit(reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    self_edges = 0
    duplicate_edges = 0
    relation_confidence = Counter()
    for row in reference_rows:
        seen = set()
        for rel in row["relations"]:
            key = (rel["subject_entity_id"], rel["relation_type"], rel["object_entity_id"], rel["polarity"])
            self_edges += rel["subject_entity_id"] == rel["object_entity_id"]
            duplicate_edges += key in seen
            seen.add(key)
            relation_confidence[row["curation_confidence"]] += 1
    return {
        "rows": len(reference_rows),
        "entities": sum(len(row["entities"]) for row in reference_rows),
        "relations": sum(len(row["relations"]) for row in reference_rows),
        "relation_count_by_statement_confidence": dict(relation_confidence),
        "self_edges": self_edges,
        "duplicate_edges": duplicate_edges,
        "limitations": [
            "Source-only structural reference, not physician-adjudicated medical truth.",
            "Canonical names are local normalizations, not authoritative UMLS assignments.",
            "Primary relation evaluation covers only the frozen shared relation ontology.",
        ],
    }


def render_report(summary: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Statement graph evaluation against Codex-curated reference",
        "",
        "> This reference checks source-faithful structural extraction; it is not physician-annotated medical truth.",
        "",
        "## Reference",
        "",
        f"- Propositions: {summary['reference_audit']['rows']:,}",
        f"- Entities: {summary['reference_audit']['entities']:,}",
        f"- In-ontology relations: {summary['reference_audit']['relations']:,}",
        f"- Self edges: {summary['reference_audit']['self_edges']}; duplicate edges: {summary['reference_audit']['duplicate_edges']}",
        "",
    ]
    for cohort in ["high_confidence", "all"]:
        result = summary["v5"][cohort]
        lines += [f"## v5 — {cohort.replace('_', ' ')}", "", "### Entity extraction", ""]
        for name in ["candidate_mentions", "final_graph_nodes"]:
            metric = result["entities"][name]["relaxed_typed"]
            lines.append(
                f"- {name}: P {pct(metric['precision'])}, R {pct(metric['recall'])}, F1 {pct(metric['f1'])} "
                f"(TP={metric['true_positive']:,}, predicted={metric['predicted']:,}, gold={metric['gold']:,})"
            )
        rel = result["relations"]["accepted_edges"]["relaxed_typed_without_polarity"]
        cand = result["relations"]["candidate_pairs"]["relaxed_endpoint_candidate_recall"]
        lines += [
            "", "### Relation extraction", "",
            f"- Candidate-pair recall: {pct(cand['recall'])} ({cand['true_positive']:,}/{cand['gold']:,} gold relations)",
            f"- Accepted relation: P {pct(rel['precision'])}, R {pct(rel['recall'])}, F1 {pct(rel['f1'])} "
            f"(TP={rel['true_positive']:,}, predicted={rel['predicted']:,}, gold={rel['gold']:,})",
            "",
        ]
    v6 = summary.get("v6_preflight")
    if v6:
        lines += ["## v6.2 preflight (no accepted-edge generation)", ""]
        for cohort in ["high_confidence", "all"]:
            cand = v6[cohort]["relations"]["candidate_relations"]["relaxed_endpoint_candidate_recall"]
            ent = v6[cohort]["entities"]["core_mentions"]["relaxed_typed"]
            lines.append(
                f"- {cohort}: core entity F1 {pct(ent['f1'])}; trigger-candidate recall {pct(cand['recall'])} "
                f"({cand['true_positive']:,}/{cand['gold']:,})"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--v5-root", type=Path, default=DEFAULT_V5)
    parser.add_argument("--v6-root", type=Path, default=DEFAULT_V6)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference_rows = load_jsonl(args.reference_root / "reference_statements.jsonl")
    if len(reference_rows) != 1000:
        raise RuntimeError(f"Expected 1000 validated reference rows, found {len(reference_rows)}")
    source_rows = load_jsonl(args.source)
    if len(source_rows) != 1000:
        raise RuntimeError(f"Expected 1000 source rows, found {len(source_rows)}")
    source_by_prop = {row["proposition_id"]: row for row in source_rows}
    mention_by_id = {
        mention["mention_id"]: mention for row in source_rows for mention in row.get("mentions", [])
    }
    source_mentions = {pid: row.get("mentions", []) for pid, row in source_by_prop.items()}

    v5_edges = load_jsonl(args.v5_root / "candidate_edges_v5.jsonl")
    v5_pair_rows = load_jsonl(args.v5_root / "candidate_pairs_v5.jsonl")
    if not v5_edges or not v5_pair_rows:
        raise RuntimeError("Completed v5 edges and candidate pairs are required")
    v5_pairs = flatten_v5_pairs(v5_pair_rows)
    v5_node_ids = {
        edge[key] for edge in v5_edges for key in ["subject_mention_id", "object_mention_id"] if edge.get(key)
    }
    v5_graph_mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention_id in v5_node_ids:
        mention = mention_by_id.get(mention_id)
        if mention:
            pid = mention_id.split("::N", 1)[0]
            v5_graph_mentions[pid].append(mention)

    confidence = {row["proposition_id"]: row["curation_confidence"] for row in reference_rows}
    cohorts = {
        "high_confidence": {pid for pid, value in confidence.items() if value == "high"},
        "all": set(confidence),
    }
    source_of = {row["proposition_id"]: row["source"] for row in reference_rows}
    summary: dict[str, Any] = {
        "reference_audit": reference_audit(reference_rows),
        "contracts": {
            "primary_cohort": "high_confidence",
            "primary_entity_metric": "relaxed_typed one-to-one span matching (>=50% of shorter span)",
            "primary_relation_metric": "shared ontology, relaxed typed endpoints, correct direction/type, polarity ignored",
            "sensitivity_cohort": "all including medium/low confidence",
        },
        "v5": {},
    }
    for cohort_name, cohort in cohorts.items():
        summary["v5"][cohort_name] = {
            "cohort_propositions": len(cohort),
            "entities": {
                "candidate_mentions": entity_metrics(reference_rows, source_mentions, cohort),
                "final_graph_nodes": entity_metrics(reference_rows, v5_graph_mentions, cohort),
            },
            "relations": {
                "candidate_pairs": relation_metrics(reference_rows, v5_pairs, mention_by_id, cohort, candidates=True),
                "accepted_edges": relation_metrics(reference_rows, v5_edges, mention_by_id, cohort, candidates=False),
            },
        }
    summary["v5"]["high_confidence_by_source"] = {}
    for source in sorted(set(source_of.values())):
        cohort = {pid for pid in cohorts["high_confidence"] if source_of[pid] == source}
        summary["v5"]["high_confidence_by_source"][source] = {
            "cohort_propositions": len(cohort),
            "candidate_entity": entity_metrics(reference_rows, source_mentions, cohort)["relaxed_typed"],
            "final_node_entity": entity_metrics(reference_rows, v5_graph_mentions, cohort)["relaxed_typed"],
            "candidate_relation": relation_metrics(
                reference_rows, v5_pairs, mention_by_id, cohort, candidates=True
            )["relaxed_endpoint_candidate_recall"],
            "accepted_relation": relation_metrics(
                reference_rows, v5_edges, mention_by_id, cohort, candidates=False
            )["relaxed_typed_without_polarity"],
        }

    v6_rows = load_jsonl(args.v6_root / "relation_candidates_v6.jsonl")
    if v6_rows:
        v6_mentions = {row["proposition_id"]: row.get("core_mentions", []) for row in v6_rows}
        v6_candidates = flatten_v6_candidates(v6_rows)
        summary["v6_preflight"] = {}
        for cohort_name, cohort in cohorts.items():
            summary["v6_preflight"][cohort_name] = {
                "cohort_propositions": len(cohort),
                "entities": {"core_mentions": entity_metrics(reference_rows, v6_mentions, cohort)},
                "relations": {
                    "candidate_relations": relation_metrics(
                        reference_rows, v6_candidates, mention_by_id, cohort, candidates=True
                    )
                },
                "limitation": "Only deterministic preflight candidates exist; accepted relation precision is not available.",
            }

    write_json(args.output_root / "summary.json", summary)
    diagnostics, missed = relation_diagnostics(reference_rows, v5_edges, mention_by_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v5_relation_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostics), encoding="utf-8"
    )
    (args.output_root / "v5_missed_gold_relations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in missed), encoding="utf-8"
    )
    report = render_report(summary)
    (args.output_root / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

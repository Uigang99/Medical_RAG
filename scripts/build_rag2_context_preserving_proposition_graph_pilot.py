#!/usr/bin/env python3
"""Build and evaluate a context-preserving proposition factor graph.

Every source-grounded proposition is retained as an evidence occurrence.  The
model extracts only exact biomedical participant spans; it never decides that a
patient/study-specific claim is universally true or unusable.  Context remains
losslessly available in the full proposition text and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

import extract_rag2_statement_graph_pair_classifier_v5 as common  # noqa: E402
import evaluate_rag2_proposition_participant_graph_pilot as prior  # noqa: E402
from build_rag2_proposition_evidence_graph_pilot import strict_umls_link  # noqa: E402


RUN_VERSION = "rag2_context_preserving_proposition_graph_highconf768_v3"
DEFAULT_SOURCE = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_context_preserving_proposition_graph_pilot_768_v3"
DEFAULT_PREDICTION_CACHE = ROOT / "datasets/kg/rag2_context_preserving_proposition_graph_pilot_768_v1/predictions.jsonl"
DEFAULT_MODEL = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"
PROGRESS_PREFIX = "@@CONTEXT_PROPOSITION_GRAPH_PROGRESS@@"
GENERIC_BRIDGES = {
    "article", "case", "cases", "condition", "data", "effect", "effects", "finding", "findings",
    "group", "groups", "level", "levels", "measurement", "measurements", "patient", "patients",
    "population", "result", "results", "sample", "samples", "study", "studies", "subject", "subjects",
    "test", "tests", "treatment", "treatments",
}


SYSTEM_PROMPT = """Extract every graph-worthy biomedical participant from one atomic proposition.

The proposition is an evidence occurrence, not a universal law. Keep participants from patient-specific observations, cohorts, studies, measurements, and recommendations; do not reject a proposition because its scope is narrow.

Participants may be anatomy, condition/disease, drug/chemical, finding/outcome/state, gene/protein, organism, population, or procedure/intervention. Include entities and states needed to preserve what the proposition reports. Copy exact spans from PROP. Prefer one informative occurrence of an abbreviation/long-form concept rather than duplicate aliases. A numeric value alone is not a participant; when essential, keep it inside the complete measurement or threshold span.

Exclude relation words by themselves, citation/table/article vocabulary, and generic fragments that do not identify biomedical content. Candidate hints are noisy and incomplete: trim, combine, or ignore them. Never add text absent from PROP.

Return only compact JSON:
{"e":["exact span",...]}

Examples:
PROP: The serum troponin I level of the admitted woman was 0.12 micrograms/L.
OUTPUT: {"e":["serum troponin I level","admitted woman"]}

PROP: Aspirin reduces platelet aggregation in high-risk adults.
OUTPUT: {"e":["Aspirin","platelet aggregation","high-risk adults"]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "extract", "aggregate"), default="orchestrate")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-predictions-from", type=Path, default=DEFAULT_PREDICTION_CACHE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--umls-score-min", type=float, default=0.90)
    parser.add_argument("--umls-margin-min", type=float, default=0.05)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(PROGRESS_PREFIX + common.canonical_json({"event": event, **payload}), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_path = args.source_root / "raw_statements_v2.jsonl"
    reference_path = args.reference_root / "reference_statements.jsonl"
    source = {str(row["proposition_id"]): row for row in common.read_jsonl(source_path)}
    reference = list(common.read_jsonl(reference_path))
    rows: list[dict[str, Any]] = []
    for gold in reference:
        if str(gold.get("curation_confidence")) != "high":
            continue
        proposition_id = str(gold["proposition_id"])
        raw = source[proposition_id]
        rows.append(
            {
                "proposition_id": proposition_id,
                "document_id": str(raw["stable_id"]),
                "source": str(gold["source"]),
                "claim": str(gold["claim"]),
                "evidence": str(raw.get("evidence") or ""),
                "source_text_sha256": raw.get("source_text_sha256"),
                "candidate_hints": prior.deduplicate_surfaces(mention["surface"] for mention in raw.get("mentions") or []),
                "raw_mentions": list(raw.get("mentions") or []),
                "gold_status": str(gold["status"]),
                "gold_entities": [
                    {
                        "surface": str(entity["surface"]), "start": int(entity["start"]), "end": int(entity["end"]),
                        "entity_type": str(entity["entity_type"]), "concept_id": str(entity["concept_id"]),
                    }
                    for entity in gold.get("entities") or []
                ],
            }
        )
    if len(rows) != 768:
        raise RuntimeError(f"Expected 768 high-confidence rows; got {len(rows)}")
    provisional = [{**row, "gold_use": True, "gold_status": row["gold_status"]} for row in rows]
    split = prior.split_rows(provisional)
    return [{key: value for key, value in row.items() if key != "gold_use"} for row in split]


def user_prompt(row: Mapping[str, Any]) -> str:
    hints = [{"id": f"H{index:02d}", "s": surface} for index, surface in enumerate(row["candidate_hints"], start=1)]
    return "PROP:\n" + str(row["claim"]) + "\n\nNOISY SPAN HINTS:\n" + json.dumps(hints, ensure_ascii=False, separators=(",", ":"))


def parse_prediction(row: Mapping[str, Any], raw_output: str) -> dict[str, Any]:
    candidate_output = raw_output.strip()
    # Deterministic repair for the only observed schema truncation: the model
    # closed the entity list but omitted the final object brace.
    if candidate_output.startswith('{"e":[') and candidate_output.endswith("]"):
        candidate_output += "}"
    parsed = common.parse_json_object(candidate_output)
    errors: list[str] = []
    schema_valid = True
    if not isinstance(parsed, dict):
        parsed = {}
        errors.append("invalid_json")
        schema_valid = False
    supplied = parsed.get("e")
    if not isinstance(supplied, list):
        supplied = []
        errors.append("invalid_entity_list")
        schema_valid = False
    selected = prior.deduplicate_surfaces(str(value) for value in supplied if isinstance(value, str))
    entities = []
    for surface in selected:
        occurrences = prior.exact_occurrences(str(row["claim"]), surface)
        if not occurrences:
            errors.append("non_extractable_span")
            continue
        start, end = occurrences[0]
        entities.append({"surface": str(row["claim"])[start:end], "start": start, "end": end})
    return {**row, "predicted_entities": entities, "valid_output": schema_valid, "output_errors": sorted(set(errors)), "raw_output": raw_output}


def prompt_audit(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    maximum = 0
    over = 0
    for row in rows:
        prompt = common.chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt(row))
        length = len(tokenizer.encode(prompt, add_special_tokens=False)) + args.max_output_tokens
        maximum = max(maximum, length)
        over += int(length > args.max_model_len)
    if over:
        raise RuntimeError(f"Prompts exceed context: {over}; max={maximum}")
    return {"max_prompt_plus_output_tokens": maximum, "over_limit": over}


def ensure_manifest(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    source = args.source_root / "raw_statements_v2.jsonl"
    reference = args.reference_root / "reference_statements.jsonl"
    value = {
        "run_version": RUN_VERSION,
        "hypothesis": "Keeping every grounded proposition as an evidence occurrence while extracting only participants yields a stable relation-label-free graph without treating context as a necessary condition.",
        "changed_from_prior": "Remove graph-use/generalizability gate; retain full proposition text and provenance; normalize only exact-span strict UMLS anchors; retain lexical keys as non-identity indexes.",
        "cohort": {"rows": len(rows), "splits": dict(Counter(row["split"] for row in rows)), "sources": dict(Counter(row["source"] for row in rows)), "reference": "Codex-curated source-only high-confidence structural reference"},
        "inputs": {"source_sha256": sha256_file(source), "reference_sha256": sha256_file(reference), "reused_prediction_sha256": sha256_file(args.reuse_predictions_from) if args.reuse_predictions_from.is_file() else None},
        "model": str(args.model.resolve()), "prompt_sha256": common.sha256_text(SYSTEM_PROMPT),
        "generation": {"temperature": 0.0, "seed": args.seed, "batch_size": args.generation_batch_size, "shard_size": args.generation_shard_size, "max_model_len": args.max_model_len, "max_output_tokens": args.max_output_tokens},
        "graph_contract": {
            "statement_semantics": "full proposition text; context is evidence metadata, never a hard applicability condition",
            "cross_document_merge": "only strict UMLS links whose raw mention span exactly equals the extracted participant span",
            "lexical_index": "exact normalized surface is a retrieval index only and never an identity assertion",
            "typed_entity_relations": "none",
        },
        "prespecified_gate": {"participant_precision_min": 0.85, "participant_recall_min": 0.80, "clean_two_participant_rate_min": 0.70, "valid_output_rate_min": 0.99, "all_normalized_mentions_exact_span": True},
        "evaluation_note": "The prior frozen split is reused after observing the v1 eligibility result; this is a developmental pilot, not an independent confirmatory test.",
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    value["contract_sha256"] = common.sha256_text(common.canonical_json(value))
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("contract_sha256") != value["contract_sha256"] and any((args.output_root / item).exists() for item in ("extraction_shards", "predictions.jsonl", "summary.json")):
            raise RuntimeError("Context-preserving graph resume contract mismatch; use a new output root")
    common.atomic_json(path, value)


def extract_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(common.read_jsonl(args.output_root / "inputs.jsonl"))
    shard_root = args.output_root / "extraction_shards"
    cached = common.cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen participant extraction without applicability filtering")
    if len(cached) < len(rows):
        tokenizer, llm = common.init_vllm(args.model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed)
        for start in range(0, len(rows), args.generation_shard_size):
            batch = rows[start : start + args.generation_shard_size]
            missing = [row for row in batch if str(row["proposition_id"]) not in cached]
            if missing:
                prompts = [common.chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt(row)) for row in missing]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(missing, outputs):
                    raw = output.outputs[0].text.strip() if output.outputs else ""
                    parsed = parse_prediction(row, raw)
                    cached[str(parsed["proposition_id"])] = parsed
            common.atomic_jsonl(shard_root / f"batch_{start:04d}_{start + len(batch):04d}.jsonl", (cached[str(row["proposition_id"])] for row in batch))
            emit("progress", total=len(rows), completed=len(cached))
    if len(cached) != len(rows):
        raise RuntimeError(f"Incomplete participant cache: {len(cached)}/{len(rows)}")
    common.atomic_jsonl(args.output_root / "predictions.jsonl", (cached[str(row["proposition_id"])] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def score_participants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        predicted, gold = list(row["predicted_entities"]), list(row["gold_entities"])
        exact, _, _ = prior.maximum_match(predicted, gold, relaxed=False)
        relaxed, unmatched_pred, _ = prior.maximum_match(predicted, gold, relaxed=True)
        counts["rows"] += 1
        counts["valid"] += int(row["valid_output"])
        counts["pred"] += len(predicted); counts["gold"] += len(gold)
        counts["exact"] += exact; counts["relaxed"] += relaxed
        counts["exact_set"] += int(exact == len(predicted) == len(gold))
        counts["relaxed_set"] += int(relaxed == len(predicted) == len(gold))
        if len(gold) >= 2:
            counts["gold_connected"] += 1
            counts["clean_connected"] += int(relaxed >= 2 and not unmatched_pred)
            counts["two_recalled"] += int(relaxed >= 2)

    def metric(matched: int) -> dict[str, Any]:
        precision = matched / max(counts["pred"], 1); recall = matched / max(counts["gold"], 1)
        return {"matched": matched, "predicted": counts["pred"], "gold": counts["gold"], "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
    return {
        "rows": counts["rows"], "valid_output_rate": counts["valid"] / max(counts["rows"], 1),
        "exact": metric(counts["exact"]), "boundary_tolerant": metric(counts["relaxed"]),
        "exact_set_rate": counts["exact_set"] / max(counts["rows"], 1), "boundary_tolerant_set_rate": counts["relaxed_set"] / max(counts["rows"], 1),
        "gold_two_plus_statements": counts["gold_connected"], "clean_two_participant_rate": counts["clean_connected"] / max(counts["gold_connected"], 1), "two_participant_recall_rate": counts["two_recalled"] / max(counts["gold_connected"], 1),
    }


def exact_raw_mention(entity: Mapping[str, Any], raw_mentions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return an upstream mention only when its source span is identical.

    Contained-span linking is unsafe here: e.g. a participant such as
    ``patients with partial renal recovery`` must not inherit the UMLS concept
    for the contained phrase ``renal function``.
    """
    for mention in raw_mentions:
        if int(mention.get("start", -1)) == int(entity["start"]) and int(mention.get("end", -1)) == int(entity["end"]):
            return mention
    return None


def lexical_key(surface: str, kind: str) -> str | None:
    normalized = prior.normalize(surface)
    if not normalized or normalized in GENERIC_BRIDGES or len(normalized) < 4 or normalized.replace(" ", "").isdigit():
        return None
    if kind in {"", "biomedical_concept", "unknown", "None"}:
        return None
    return f"LEX::{kind}::{normalized}"


def build_graph(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    statements: list[dict[str, Any]] = []
    participants: list[dict[str, Any]] = []
    concepts: dict[str, dict[str, Any]] = {}
    lexical_indexes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    predicted_to_gold: list[tuple[str, str, str]] = []
    preliminary: list[tuple[dict[str, Any], str | None, dict[str, Any] | None]] = []

    for row in rows:
        document_id = str(row["document_id"]); proposition_id = str(row["proposition_id"])
        documents.setdefault(document_id, {"node_id": document_id, "node_type": "Document", "source": row["source"]})
        statements.append({"node_id": proposition_id, "node_type": "EvidenceStatement", "document_id": document_id, "source": row["source"], "text": row["claim"], "evidence": row["evidence"], "context_policy": "preserved_in_full_text_not_a_hard_condition"})
        edges.append({"source": document_id, "relation": "HAS_EVIDENCE_STATEMENT", "target": proposition_id})
        for index, entity in enumerate(row["predicted_entities"], start=1):
            mention_id = f"{proposition_id}::ARG{index}"
            raw = exact_raw_mention(entity, row["raw_mentions"])
            kind = str(raw.get("kind") if raw else "unknown")
            link = strict_umls_link(raw, args.umls_score_min, args.umls_margin_min) if raw else None
            lex = lexical_key(str(entity["surface"]), kind)
            mention = {"node_id": mention_id, "node_type": "ParticipantMention", "statement_id": proposition_id, "document_id": document_id, "surface": entity["surface"], "start": entity["start"], "end": entity["end"], "kind": kind}
            participants.append(mention)
            preliminary.append((mention, lex, link))

        matched, _, _ = prior.maximum_match(row["predicted_entities"], row["gold_entities"], relaxed=True)
        # Build deterministic matched pairs only for concept-cluster diagnostics.
        remaining = list(range(len(row["gold_entities"])))
        for pred_index, entity in enumerate(row["predicted_entities"]):
            choices = [gold_index for gold_index in remaining if prior.compatible(entity, row["gold_entities"][gold_index])]
            if choices:
                gold_index = choices[0]; remaining.remove(gold_index)
                predicted_to_gold.append((f"{proposition_id}::ARG{pred_index+1}", str(row["gold_entities"][gold_index]["concept_id"]), document_id))

    mention_to_concept: dict[str, str] = {}
    for mention, lex, link in preliminary:
        concept_id = None; method = None
        if link:
            concept_id = str(link["concept_id"]); method = "strict_umls"
            concepts.setdefault(concept_id, {"node_id": concept_id, "node_type": "CanonicalEntity", "name": link["name"], "category": link["category"], "normalization": method})
        else:
            concept_id = f"LOCAL::{mention['node_id']}"; method = "local_unmerged"
            concepts[concept_id] = {"node_id": concept_id, "node_type": "CanonicalEntity", "name": mention["surface"], "category": mention["kind"], "normalization": method}
        mention_to_concept[str(mention["node_id"])] = concept_id
        edges.append({"source": mention["statement_id"], "relation": "HAS_ARGUMENT", "target": mention["node_id"]})
        edges.append({"source": mention["node_id"], "relation": "DENOTES", "target": concept_id, "normalization": method})
        if lex:
            lexical_indexes.setdefault(lex, {"node_id": lex, "node_type": "LexicalIndex", "surface_normalized": lex.rsplit("::", 1)[-1], "category": mention["kind"], "identity_assertion": False})
            edges.append({"source": mention["node_id"], "relation": "HAS_LEXICAL_INDEX", "target": lex, "identity_assertion": False})

    gold_by_mention = {mention_id: (gold, document_id) for mention_id, gold, document_id in predicted_to_gold}
    pred_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for mention_id, concept_id in mention_to_concept.items():
        if mention_id in gold_by_mention and not concept_id.startswith("LOCAL::"):
            gold, document_id = gold_by_mention[mention_id]
            pred_groups[concept_id].append((gold, document_id))
    pred_pairs = correct_pred_pairs = 0
    for members in pred_groups.values():
        for left, right in itertools.combinations(members, 2):
            if left[1] == right[1]: continue
            pred_pairs += 1; correct_pred_pairs += int(left[0] == right[0])

    concept_docs: dict[str, set[str]] = defaultdict(set)
    concept_statements: dict[str, set[str]] = defaultdict(set)
    for mention in participants:
        concept_id = mention_to_concept[mention["node_id"]]
        if not concept_id.startswith("LOCAL::"):
            concept_docs[concept_id].add(str(mention["document_id"])); concept_statements[concept_id].add(str(mention["statement_id"]))
    shared = {concept for concept, docs in concept_docs.items() if len(docs) >= 2}
    covered_statements = set().union(*(concept_statements[concept] for concept in shared)) if shared else set()
    stats = {
        "documents": len(documents), "evidence_statements": len(statements), "participant_mentions": len(participants), "canonical_entities": len(concepts),
        "strict_umls_entities": sum(value["normalization"] == "strict_umls" for value in concepts.values()), "lexical_indexes": len(lexical_indexes),
        "cross_document_shared_entities": len(shared), "cross_document_statement_coverage": len(covered_statements) / max(len(statements), 1),
        "cross_document_normalized_pairs": pred_pairs,
        "same_reference_role_id_pairs": correct_pred_pairs,
        "reference_role_id_agreement": correct_pred_pairs / pred_pairs if pred_pairs else None,
        "reference_role_id_note": "Diagnostic only. Reference IDs include proposition-specific roles and are not canonical-identity gold labels.",
        "normalized_links_require_exact_span": True,
        "edges": len(edges),
    }
    return {"documents": list(documents.values()), "statements": statements, "participants": participants, "concepts": list(concepts.values()), "lexical_indexes": list(lexical_indexes.values()), "edges": edges}, stats


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(common.read_jsonl(args.output_root / "predictions.jsonl"))
    emit("start", total=len(rows), completed=0, name="score participants and build context-preserving graph")
    by_split = {split: score_participants([row for row in rows if row["split"] == split]) for split in prior.SPLITS}
    baseline_rows = [{**row, "predicted_entities": [{"surface": mention["surface"], "start": mention["start"], "end": mention["end"]} for mention in row["raw_mentions"]], "valid_output": True} for row in rows]
    baseline = {split: score_participants([row for row in baseline_rows if row["split"] == split]) for split in prior.SPLITS}
    graph, topology = build_graph(args, rows)
    for index in range(0, len(rows), 64): emit("progress", total=len(rows), completed=min(index + 64, len(rows)))
    for name, values in graph.items(): common.atomic_jsonl(args.output_root / f"{name}.jsonl", values)
    common.atomic_jsonl(args.output_root / "scored_predictions.jsonl", rows)
    validation = by_split["validation"]; test = by_split["test"]
    def passes(metrics: Mapping[str, Any]) -> bool:
        return metrics["valid_output_rate"] >= .99 and metrics["boundary_tolerant"]["precision"] >= .85 and metrics["boundary_tolerant"]["recall"] >= .80 and metrics["clean_two_participant_rate"] >= .70
    summary = {
        "run_version": RUN_VERSION, "context_policy": "All grounded propositions are evidence occurrences; patient/study/demographic context is preserved in full text and is never interpreted as a necessary condition.",
        "metrics": {"by_split": by_split}, "baseline_all_mentions": {"by_split": baseline}, "graph_topology": topology,
        "output_errors": dict(Counter(error for row in rows for error in row["output_errors"])),
        "decision": {"validation_passed": passes(validation), "developmental_test_passed": passes(test), "normalization_contract_passed": topology["normalized_links_require_exact_span"], "overall_passed": passes(validation) and passes(test) and topology["normalized_links_require_exact_span"], "independent_confirmation_required": True, "canonical_identity_accuracy_gold_evaluated": False},
    }
    common.atomic_json(args.output_root / "summary.json", summary)
    p = test["boundary_tolerant"]
    report = f"""# Context-preserving proposition graph pilot

- Developmental test N: {test['rows']}
- Participant precision/recall/F1: {100*p['precision']:.2f}% / {100*p['recall']:.2f}% / {100*p['f1']:.2f}%
- Exact-span F1: {100*test['exact']['f1']:.2f}%
- Clean two-participant rate: {100*test['clean_two_participant_rate']:.2f}%
- Valid output rate: {100*test['valid_output_rate']:.2f}%
- Strict exact-span UMLS concepts: {topology['strict_umls_entities']}
- Cross-document shared UMLS concepts: {topology['cross_document_shared_entities']}
- Cross-document statement coverage: {100*topology['cross_document_statement_coverage']:.2f}%
- Canonical identity accuracy: not gold-evaluated; reference role IDs are not canonical concept labels
- Decision: {'PASS' if summary['decision']['overall_passed'] else 'FAIL'} (developmental; independent fresh-sample confirmation required)
"""
    temp=args.output_root/'report.md.tmp'; temp.write_text(report,encoding='utf-8'); temp.replace(args.output_root/'report.md')
    common.atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "passed": summary["decision"]["overall_passed"]})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode, "--source-root", str(args.source_root), "--reference-root", str(args.reference_root), "--output-root", str(args.output_root), "--model", str(args.model), "--llm-python", str(args.llm_python), "--generation-batch-size", str(args.generation_batch_size), "--generation-shard-size", str(args.generation_shard_size), "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-model-len", str(args.max_model_len), "--max-output-tokens", str(args.max_output_tokens), "--seed", str(args.seed), "--umls-score-min", str(args.umls_score_min), "--umls-margin-min", str(args.umls_margin_min)]


def run_worker(args: argparse.Namespace, mode: str, progress: Any, stage: int) -> None:
    environment=dict(os.environ); environment.setdefault("TOKENIZERS_PARALLELISM","false"); environment.setdefault("VLLM_LOGGING_LEVEL","WARNING"); environment.setdefault("TRANSFORMERS_VERBOSITY","error")
    process=subprocess.Popen(worker_command(args,mode),cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    assert process.stdout is not None
    started=False
    for line in process.stdout:
        value=line.rstrip()
        if value.startswith(PROGRESS_PREFIX):
            event=json.loads(value[len(PROGRESS_PREFIX):])
            if event["event"]=="start": progress.start(stage,int(event["total"]),int(event.get("completed",0))); started=True
            elif event["event"] in {"progress","complete"}:
                if not started: progress.start(stage,int(event["total"]),int(event.get("completed",0))); started=True
                progress.update(int(event["completed"]),int(event["total"]))
        elif value and not value.startswith("Loading weights:"): progress.log(value)
    code=process.wait()
    if code: raise RuntimeError(f"Worker {mode} failed with exit code {code}")


def orchestrate(args: argparse.Namespace) -> None:
    for path in (args.source_root/'raw_statements_v2.jsonl',args.reference_root/'reference_statements.jsonl',args.model,args.llm_python):
        if not path.exists(): raise FileNotFoundError(path)
    rows=load_rows(args); ensure_manifest(args,rows); common.atomic_jsonl(args.output_root/'inputs.jsonl',rows); audit=prompt_audit(args,rows)
    common.atomic_json(args.output_root/'preflight_summary.json',{"rows":len(rows),"splits":dict(Counter(row['split'] for row in rows)),"prompt_tokens":audit})
    progress=common.WorkflowProgress(["freeze context-preserving graph contract","extract participants for every proposition","score and build proposition factor graph"],[15,180,20])
    progress.log(f"[workflow plan] propositions={len(rows)} hard_applicability_filter=none relation_labels=none full_context_preserved=true")
    progress.log("[success gate] participant precision>=85%, recall>=80%, clean two-participant>=70%, valid output>=99%; normalization requires exact-span strict UMLS")
    progress.start(1,len(rows),len(rows)); progress.log(f"[stage 1/3 complete] max_prompt_plus_output={audit['max_prompt_plus_output_tokens']} output={args.output_root/'inputs.jsonl'}")
    if args.preflight_only: progress.close(); progress.log("[preflight complete] no model loaded"); return
    if args.reuse_predictions_from.is_file():
        cached_raw = {str(row["proposition_id"]): row for row in common.read_jsonl(args.reuse_predictions_from)}
        expected = {str(row["proposition_id"]): row for row in rows}
        if set(cached_raw) != set(expected):
            raise RuntimeError(f"Reusable prediction IDs differ: cached={len(cached_raw)} expected={len(expected)}")
        cached = {}
        for proposition_id, row in expected.items():
            old = cached_raw[proposition_id]
            if old.get("claim") != row.get("claim") or old.get("source_text_sha256") != row.get("source_text_sha256"):
                raise RuntimeError(f"Reusable prediction contract mismatch: {proposition_id}")
            cached[proposition_id] = parse_prediction(row, str(old.get("raw_output") or ""))
        progress.start(2,len(rows),len(rows))
        common.atomic_jsonl(args.output_root/'predictions.jsonl',(cached[str(row['proposition_id'])] for row in rows))
        progress.log(f"[stage 2/3 complete] reused immutable predictions source={args.reuse_predictions_from}")
    else:
        run_worker(args,"extract",progress,2); progress.log(f"[stage 2/3 complete] output={args.output_root/'predictions.jsonl'}")
    run_worker(args,"aggregate",progress,3); progress.log(f"[stage 3/3 complete] report={args.output_root/'report.md'}"); progress.close(); progress.log(f"[workflow complete] result={args.output_root}")


def main() -> None:
    args=parse_args()
    if args.mode=="orchestrate": orchestrate(args)
    elif args.mode=="extract": extract_worker(args)
    elif args.mode=="aggregate": aggregate_worker(args)
    else: raise AssertionError(args.mode)


if __name__ == "__main__": main()

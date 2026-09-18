#!/usr/bin/env python3
"""Evaluate a relation-label-free proposition graph on a frozen 768-row reference.

The only learned decision is extractive: decide whether a proposition is useful
as a graph factor and select its graph-worthy biomedical participant spans.  No
subject/object direction or relation ontology is predicted.  The proposition
text itself remains the relation semantics.
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

import extract_rag2_statement_graph_pair_classifier_v5 as common  # noqa: E402
from build_rag2_proposition_evidence_graph_pilot import strict_umls_link  # noqa: E402


RUN_VERSION = "rag2_proposition_participant_graph_highconf768_v1"
DEFAULT_SOURCE = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2"
DEFAULT_REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_proposition_participant_graph_pilot_768_v1"
DEFAULT_MODEL = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"
PROGRESS_PREFIX = "@@PROPOSITION_PARTICIPANT_PROGRESS@@"
SPLITS = ("development", "validation", "test")
ELIGIBLE_STATUSES = {"usable", "no_core_relation"}


SYSTEM_PROMPT = """You extract a source-grounded proposition graph from one atomic biomedical proposition.

Return whether the proposition should be a knowledge factor and all graph-worthy biomedical entity spans.

Use = Y for a self-contained biomedical fact, finding, comparison, definition, mechanism, or recommendation. Use = N for bibliographic metadata, study-administration metadata without a biomedical claim, or a context-incomplete fragment.

Entity spans may be anatomy, condition/disease, drug/chemical, finding/outcome/state, gene/protein, organism, population, or procedure/intervention. Select every entity needed to preserve the biomedical content of the proposition. Findings and states expressed as adjectives may be entities when they are central to the claim. Exclude relation words by themselves, bare quantities or durations, citation/study/result vocabulary, and generic fragments that do not identify a biomedical concept.

Every returned span must be copied exactly from PROP. Candidate hints are noisy and incomplete: trim an overlong hint, combine adjacent words, or ignore hints as needed, but never add text absent from PROP. Return each concept once; prefer its most informative occurrence.

Return only compact JSON:
{"u":"Y|N","e":["exact span",...]}

Examples:
PROP: Aspirin reduces platelet aggregation in high-risk adults.
OUTPUT: {"u":"Y","e":["Aspirin","platelet aggregation","high-risk adults"]}

PROP: The article contains 12 references and 4 tables.
OUTPUT: {"u":"N","e":[]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "classify", "aggregate"), default="orchestrate")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-shard-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
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


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def exact_occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    if not surface:
        return []
    return [(match.start(), match.end()) for match in re.finditer(re.escape(surface), text, flags=re.IGNORECASE)]


def deduplicate_surfaces(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        surface = str(value).strip()
        key = normalize(surface)
        if key and key not in seen:
            seen.add(key)
            output.append(surface)
    return output


def split_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic source/status stratification with globally balanced thirds."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source"]), str(row["gold_status"]))].append(row)
    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    global_counts = Counter()
    for group_key in sorted(groups):
        ranked = sorted(
            groups[group_key],
            key=lambda row: hashlib.sha256(
                f"participant-graph-v1::{row['proposition_id']}".encode("utf-8")
            ).hexdigest(),
        )
        start = min(range(3), key=lambda index: global_counts[SPLITS[index]])
        for offset, row in enumerate(ranked):
            split = SPLITS[(start + offset) % 3]
            assigned[split].append({**row, "split": split})
            global_counts[split] += 1
    flattened = [row for split in SPLITS for row in sorted(assigned[split], key=lambda value: value["proposition_id"])]
    if len(flattened) != len(rows) or len({row["proposition_id"] for row in flattened}) != len(rows):
        raise RuntimeError("Participant split lost or duplicated propositions")
    if max(global_counts.values()) - min(global_counts.values()) > 1:
        raise RuntimeError(f"Participant splits are imbalanced: {dict(global_counts)}")
    return flattened


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
        raw = source.get(proposition_id)
        if raw is None:
            raise RuntimeError(f"Missing source proposition: {proposition_id}")
        hints = deduplicate_surfaces(mention["surface"] for mention in raw.get("mentions") or [])
        rows.append(
            {
                "proposition_id": proposition_id,
                "source": str(gold["source"]),
                "claim": str(gold["claim"]),
                "candidate_hints": hints,
                "raw_mentions": list(raw.get("mentions") or []),
                "proposition_class": raw.get("proposition_class"),
                "gold_status": str(gold["status"]),
                "gold_use": str(gold["status"]) in ELIGIBLE_STATUSES,
                "gold_entities": [
                    {
                        "surface": str(entity["surface"]),
                        "start": int(entity["start"]),
                        "end": int(entity["end"]),
                        "entity_type": str(entity["entity_type"]),
                        "concept_id": str(entity["concept_id"]),
                    }
                    for entity in gold.get("entities") or []
                ],
            }
        )
    if len(rows) != 768:
        raise RuntimeError(f"Expected 768 high-confidence reference propositions; got {len(rows)}")
    return split_rows(rows)


def user_prompt(row: Mapping[str, Any]) -> str:
    hints = [{"id": f"H{index:02d}", "s": surface} for index, surface in enumerate(row["candidate_hints"], start=1)]
    return "PROP:\n" + str(row["claim"]) + "\n\nNOISY SPAN HINTS:\n" + json.dumps(hints, ensure_ascii=False, separators=(",", ":"))


def parse_prediction(row: Mapping[str, Any], raw_output: str) -> dict[str, Any]:
    parsed = common.parse_json_object(raw_output)
    errors: list[str] = []
    if not isinstance(parsed, dict):
        parsed = {}
        errors.append("invalid_json")
    use = str(parsed.get("u") or "").upper()
    if use not in {"Y", "N"}:
        errors.append("invalid_use")
        use = "N"
    supplied = parsed.get("e")
    if not isinstance(supplied, list):
        errors.append("invalid_entity_list")
        supplied = []
    selected = deduplicate_surfaces(str(value) for value in supplied if isinstance(value, str))
    spans: list[dict[str, Any]] = []
    for surface in selected:
        occurrences = exact_occurrences(str(row["claim"]), surface)
        if not occurrences:
            errors.append("non_extractable_span")
            continue
        start, end = occurrences[0]
        spans.append({"surface": str(row["claim"])[start:end], "start": start, "end": end})
    if use == "N" and spans:
        errors.append("entities_with_use_n")
        spans = []
    if use == "Y" and not spans:
        errors.append("use_y_without_entities")
    return {
        **row,
        "predicted_use": use == "Y",
        "predicted_entities": spans,
        "valid_output": not errors,
        "output_errors": sorted(set(errors)),
        "raw_output": raw_output,
    }


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
        raise RuntimeError(f"Participant prompts exceed context: {over}; max={maximum}")
    return {"max_prompt_plus_output_tokens": maximum, "over_limit": over}


def ensure_manifest(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    source_path = args.source_root / "raw_statements_v2.jsonl"
    reference_path = args.reference_root / "reference_statements.jsonl"
    split_counts = Counter(str(row["split"]) for row in rows)
    status_counts = Counter(str(row["gold_status"]) for row in rows)
    value = {
        "run_version": RUN_VERSION,
        "purpose": "Test whether relation-label-free proposition factors and extractive participant selection can form a stable graph.",
        "single_changed_task": "Qwen predicts graph usability and exact participant spans; no relation label, direction, or answer label is available.",
        "cohort": {
            "reference_type": "Codex-curated source-only structural reference; not physician medical truth",
            "rows": len(rows),
            "splits": dict(split_counts),
            "statuses": dict(status_counts),
            "selection": "all high-confidence rows, deterministically stratified by corpus and status",
        },
        "inputs": {
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path)},
            "reference": {"path": str(reference_path.resolve()), "sha256": sha256_file(reference_path)},
        },
        "model": str(args.model.resolve()),
        "prompt_sha256": common.sha256_text(SYSTEM_PROMPT),
        "generation": {
            "temperature": 0.0,
            "seed": args.seed,
            "batch_size": args.generation_batch_size,
            "shard_size": args.generation_shard_size,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "prespecified_validation_gate": {
            "use_balanced_accuracy_min": 0.80,
            "participant_relaxed_precision_min": 0.85,
            "participant_relaxed_recall_min": 0.80,
            "clean_two_participant_statement_rate_min": 0.70,
            "valid_output_rate_min": 0.99,
        },
        "test_policy": "Prompt, parser, thresholds, and split are frozen before any aggregate is computed; test is reported without retuning.",
        "code": {"script": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    value["contract_sha256"] = common.sha256_text(common.canonical_json(value))
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != value["contract_sha256"]:
            durable = [args.output_root / "prediction_shards", args.output_root / "predictions.jsonl", args.output_root / "summary.json"]
            if any(item.exists() for item in durable):
                raise RuntimeError("Participant-graph resume contract mismatch; use a new output root")
    common.atomic_json(path, value)


def classify_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(common.read_jsonl(args.output_root / "inputs.jsonl"))
    shard_root = args.output_root / "prediction_shards"
    cached = common.cached_rows(shard_root, "proposition_id")
    emit("start", total=len(rows), completed=len(cached), name="Qwen extractive graph participant selection")
    if len(cached) < len(rows):
        tokenizer, llm = common.init_vllm(args.model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed)
        for start in range(0, len(rows), args.generation_shard_size):
            source_batch = rows[start : start + args.generation_shard_size]
            missing = [row for row in source_batch if str(row["proposition_id"]) not in cached]
            if missing:
                prompts = [common.chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt(row)) for row in missing]
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(missing, outputs):
                    raw = output.outputs[0].text.strip() if output.outputs else ""
                    value = parse_prediction(row, raw)
                    cached[str(value["proposition_id"])] = value
            common.atomic_jsonl(
                shard_root / f"batch_{start:04d}_{start + len(source_batch):04d}.jsonl",
                (cached[str(row["proposition_id"])] for row in source_batch),
            )
            emit("progress", total=len(rows), completed=len(cached))
    if len(cached) != len(rows):
        raise RuntimeError(f"Participant selection incomplete: {len(cached)}/{len(rows)}")
    common.atomic_jsonl(args.output_root / "predictions.jsonl", (cached[str(row["proposition_id"])] for row in rows))
    emit("complete", total=len(rows), completed=len(rows))


def token_set(text: str) -> set[str]:
    return {token for token in normalize(text).split() if token}


def compatible(prediction: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    pred_text = normalize(str(prediction["surface"]))
    gold_text = normalize(str(gold["surface"]))
    if pred_text == gold_text:
        return True
    pred_tokens, gold_tokens = token_set(pred_text), token_set(gold_text)
    if not pred_tokens or not gold_tokens:
        return False
    overlap = len(pred_tokens & gold_tokens)
    token_f1 = 2 * overlap / (len(pred_tokens) + len(gold_tokens))
    contained = pred_text in gold_text or gold_text in pred_text
    containment_coverage = overlap / min(len(pred_tokens), len(gold_tokens))
    return (contained and containment_coverage == 1.0) or (overlap >= 2 and token_f1 >= (2.0 / 3.0))


def maximum_match(predictions: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]], relaxed: bool) -> tuple[int, list[int], list[int]]:
    edges: dict[int, list[int]] = {}
    for pred_index, prediction in enumerate(predictions):
        values = []
        for gold_index, target in enumerate(gold):
            okay = compatible(prediction, target) if relaxed else normalize(str(prediction["surface"])) == normalize(str(target["surface"]))
            if okay:
                values.append(gold_index)
        edges[pred_index] = values
    matched_gold: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], visited):
                matched_gold[gold_index] = pred_index
                return True
        return False

    matched = sum(augment(pred_index, set()) for pred_index in range(len(predictions)))
    matched_predictions = set(matched_gold.values())
    return matched, [index for index in range(len(predictions)) if index not in matched_predictions], [index for index in range(len(gold)) if index not in matched_gold]


def score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        gold_use = bool(row["gold_use"])
        predicted_use = bool(row["predicted_use"])
        counts["rows"] += 1
        counts["valid"] += int(row["valid_output"])
        counts["tp_use"] += int(gold_use and predicted_use)
        counts["tn_use"] += int(not gold_use and not predicted_use)
        counts["gold_use"] += int(gold_use)
        counts["gold_not_use"] += int(not gold_use)
        counts["pred_use"] += int(predicted_use)
        if not gold_use:
            continue
        predicted = list(row["predicted_entities"]) if predicted_use else []
        gold = list(row["gold_entities"])
        exact, _, _ = maximum_match(predicted, gold, relaxed=False)
        relaxed, unmatched_pred, unmatched_gold = maximum_match(predicted, gold, relaxed=True)
        counts["pred_entities"] += len(predicted)
        counts["gold_entities"] += len(gold)
        counts["exact_match"] += exact
        counts["relaxed_match"] += relaxed
        counts["exact_statement"] += int(exact == len(predicted) == len(gold))
        counts["relaxed_statement"] += int(relaxed == len(predicted) == len(gold))
        if len(gold) >= 2:
            counts["gold_connected"] += 1
            counts["clean_connected"] += int(relaxed >= 2 and not unmatched_pred)
            counts["two_recalled"] += int(relaxed >= 2)
    use_recall = counts["tp_use"] / max(counts["gold_use"], 1)
    specificity = counts["tn_use"] / max(counts["gold_not_use"], 1)
    use_precision = counts["tp_use"] / max(counts["pred_use"], 1)

    def participant(prefix: str) -> dict[str, float | int]:
        matched = counts[f"{prefix}_match"]
        precision = matched / max(counts["pred_entities"], 1)
        recall = matched / max(counts["gold_entities"], 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"matched": matched, "predicted": counts["pred_entities"], "gold": counts["gold_entities"], "precision": precision, "recall": recall, "f1": f1}

    return {
        "rows": counts["rows"],
        "valid_output_rate": counts["valid"] / max(counts["rows"], 1),
        "use_decision": {
            "gold_use": counts["gold_use"], "gold_not_use": counts["gold_not_use"],
            "predicted_use": counts["pred_use"], "true_positive": counts["tp_use"], "true_negative": counts["tn_use"],
            "precision": use_precision, "recall": use_recall, "specificity": specificity,
            "balanced_accuracy": (use_recall + specificity) / 2,
        },
        "participants_exact": participant("exact"),
        "participants_boundary_tolerant": participant("relaxed"),
        "exact_participant_set_rate": counts["exact_statement"] / max(counts["gold_use"], 1),
        "boundary_tolerant_participant_set_rate": counts["relaxed_statement"] / max(counts["gold_use"], 1),
        "gold_two_plus_participant_statements": counts["gold_connected"],
        "clean_two_participant_statement_rate": counts["clean_connected"] / max(counts["gold_connected"], 1),
        "two_participant_recall_rate": counts["two_recalled"] / max(counts["gold_connected"], 1),
    }


def baseline_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        seen = set()
        predictions = []
        for mention in row["raw_mentions"]:
            key = normalize(str(mention["surface"]))
            if not key or key in seen:
                continue
            seen.add(key)
            predictions.append({"surface": mention["surface"], "start": mention["start"], "end": mention["end"]})
        proposition_class = str(row.get("proposition_class") or "")
        predicted_use = proposition_class not in {"metadata", "incomplete"}
        output.append({**row, "predicted_use": predicted_use, "predicted_entities": predictions if predicted_use else [], "valid_output": True})
    return output


def graph_rows(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    counts = Counter()
    for row in rows:
        if not row["predicted_use"]:
            continue
        proposition_id = str(row["proposition_id"])
        counts["propositions"] += 1
        strict_concepts: set[str] = set()
        for index, entity in enumerate(row["predicted_entities"], start=1):
            edge = {
                "proposition_id": proposition_id,
                "participant_id": f"{proposition_id}::ARG{index}",
                "surface": entity["surface"], "start": entity["start"], "end": entity["end"],
                "relation": "HAS_ARGUMENT", "normalization": None,
            }
            candidates = [
                mention for mention in row["raw_mentions"]
                if compatible(entity, {"surface": mention["surface"]})
            ]
            candidates.sort(key=lambda mention: (abs(len(str(mention["surface"])) - len(str(entity["surface"]))), int(mention["start"])))
            for mention in candidates:
                link = strict_umls_link(mention, args.umls_score_min, args.umls_margin_min)
                if link:
                    edge["normalization"] = link
                    strict_concepts.add(str(link["concept_id"]))
                    counts["normalized_arguments"] += 1
                    break
            edges.append(edge)
            counts["arguments"] += 1
        counts["two_plus_arguments"] += int(len(row["predicted_entities"]) >= 2)
        counts["two_plus_strict_concepts"] += int(len(strict_concepts) >= 2)
    return edges, {
        "predicted_graph_propositions": counts["propositions"],
        "argument_edges": counts["arguments"],
        "strict_normalized_argument_edges": counts["normalized_arguments"],
        "two_plus_argument_propositions": counts["two_plus_arguments"],
        "two_plus_strict_concept_propositions": counts["two_plus_strict_concepts"],
    }


def aggregate_worker(args: argparse.Namespace) -> None:
    rows = list(common.read_jsonl(args.output_root / "predictions.jsonl"))
    emit("start", total=len(rows), completed=0, name="score held-out participant graph and build diagnostics")
    for index, row in enumerate(rows, start=1):
        predicted = list(row["predicted_entities"]) if row["predicted_use"] else []
        gold = list(row["gold_entities"]) if row["gold_use"] else []
        _, unmatched_pred, unmatched_gold = maximum_match(predicted, gold, relaxed=True)
        row["unmatched_predictions"] = [predicted[value] for value in unmatched_pred]
        row["missed_gold_entities"] = [gold[value] for value in unmatched_gold]
        if index % 64 == 0 or index == len(rows):
            emit("progress", total=len(rows), completed=index)
    baseline = baseline_rows(rows)
    by_split = {split: score_rows([row for row in rows if row["split"] == split]) for split in SPLITS}
    baseline_by_split = {split: score_rows([row for row in baseline if row["split"] == split]) for split in SPLITS}
    test_rows = [row for row in rows if row["split"] == "test"]
    by_source_test = {source: score_rows([row for row in test_rows if row["source"] == source]) for source in sorted({row["source"] for row in test_rows})}
    graph_edges, graph_counts = graph_rows(args, rows)
    validation = by_split["validation"]
    validation_passed = (
        validation["use_decision"]["balanced_accuracy"] >= 0.80
        and validation["participants_boundary_tolerant"]["precision"] >= 0.85
        and validation["participants_boundary_tolerant"]["recall"] >= 0.80
        and validation["clean_two_participant_statement_rate"] >= 0.70
        and validation["valid_output_rate"] >= 0.99
    )
    test = by_split["test"]
    test_passed = (
        test["use_decision"]["balanced_accuracy"] >= 0.80
        and test["participants_boundary_tolerant"]["precision"] >= 0.85
        and test["participants_boundary_tolerant"]["recall"] >= 0.80
        and test["clean_two_participant_statement_rate"] >= 0.70
        and test["valid_output_rate"] >= 0.99
    )
    error_counts = Counter(error for row in rows for error in row["output_errors"])
    summary = {
        "run_version": RUN_VERSION,
        "reference_scope": "source-only structural entity selection; not physician medical-truth evaluation",
        "metrics": {"by_split": by_split, "test_by_source": by_source_test},
        "baseline_all_v2_mentions": {"by_split": baseline_by_split},
        "graph_structure": graph_counts,
        "output_errors": dict(error_counts),
        "decision": {
            "thresholds": {
                "use_balanced_accuracy_min": 0.80,
                "participant_relaxed_precision_min": 0.85,
                "participant_relaxed_recall_min": 0.80,
                "clean_two_participant_statement_rate_min": 0.70,
                "valid_output_rate_min": 0.99,
            },
            "validation_passed": validation_passed,
            "test_passed": test_passed,
            "overall_passed": validation_passed and test_passed,
            "next_action": (
                "Participant extraction generalizes on the frozen cohort; proceed to question-local graph connectivity evaluation."
                if validation_passed and test_passed
                else "Do not scale the factor graph; participant selection or graph eligibility is not stable enough."
            ),
        },
    }
    common.atomic_jsonl(args.output_root / "scored_predictions.jsonl", rows)
    common.atomic_jsonl(args.output_root / "argument_edges.jsonl", graph_edges)
    common.atomic_json(
        args.output_root / "error_examples.json",
        {
            "false_use": [row for row in rows if row["predicted_use"] and not row["gold_use"]][:30],
            "missed_use": [row for row in rows if not row["predicted_use"] and row["gold_use"]][:30],
            "participant_errors": [row for row in rows if row["gold_use"] and (row["unmatched_predictions"] or row["missed_gold_entities"])][:60],
        },
    )
    common.atomic_json(args.output_root / "summary.json", summary)
    p = test["participants_boundary_tolerant"]
    report = f"""# Proposition participant graph pilot

## Purpose

Test relation-label-free graph construction: proposition text is the factor; Qwen predicts only graph usability and exact participant spans.

## Frozen test result

- Test propositions: {test['rows']:,}
- Valid output rate: {100*test['valid_output_rate']:.2f}%
- Graph-use balanced accuracy: {100*test['use_decision']['balanced_accuracy']:.2f}%
- Participant boundary-tolerant precision/recall/F1: {100*p['precision']:.2f}% / {100*p['recall']:.2f}% / {100*p['f1']:.2f}%
- Exact-span participant F1: {100*test['participants_exact']['f1']:.2f}%
- Clean two-participant statement rate: {100*test['clean_two_participant_statement_rate']:.2f}%
- Validation gate: {'PASS' if validation_passed else 'FAIL'}
- Test gate: {'PASS' if test_passed else 'FAIL'}
- Final decision: {'PASS' if validation_passed and test_passed else 'FAIL'}
"""
    temp = args.output_root / "report.md.tmp"
    temp.write_text(report, encoding="utf-8")
    temp.replace(args.output_root / "report.md")
    common.atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "passed": validation_passed and test_passed})
    emit("complete", total=len(rows), completed=len(rows))


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--source-root", str(args.source_root), "--reference-root", str(args.reference_root),
        "--output-root", str(args.output_root), "--model", str(args.model), "--llm-python", str(args.llm_python),
        "--generation-batch-size", str(args.generation_batch_size), "--generation-shard-size", str(args.generation_shard_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-model-len", str(args.max_model_len),
        "--max-output-tokens", str(args.max_output_tokens), "--seed", str(args.seed),
        "--umls-score-min", str(args.umls_score_min), "--umls-margin-min", str(args.umls_margin_min),
    ]


def run_worker(args: argparse.Namespace, mode: str, progress: Any, stage: int) -> None:
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
            event = json.loads(value[len(PROGRESS_PREFIX):])
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
    required = [
        args.source_root / "raw_statements_v2.jsonl",
        args.reference_root / "reference_statements.jsonl",
        args.model,
        args.llm_python,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    rows = load_rows(args)
    ensure_manifest(args, rows)
    common.atomic_jsonl(args.output_root / "inputs.jsonl", rows)
    audit = prompt_audit(args, rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    source_counts = Counter(str(row["source"]) for row in rows)
    common.atomic_json(args.output_root / "preflight_summary.json", {"rows": len(rows), "splits": dict(split_counts), "sources": dict(source_counts), "prompt_tokens": audit})
    progress = common.WorkflowProgress(
        ["freeze high-confidence participant cohort and prompt", "Qwen graph usability and participant extraction", "score validation/test and materialize proposition graph"],
        [15, 180, 15],
    )
    progress.log(f"[workflow plan] high-confidence propositions={len(rows):,} splits={dict(split_counts)} model=Qwen3.5-9B relation_labels=none")
    progress.log("[success gate] validation/test each require use BA>=80%, participant P>=85%, R>=80%, clean connected>=70%, valid output>=99%")
    progress.start(1, len(rows), len(rows))
    progress.log(f"[stage 1/3 complete] max_prompt_plus_output={audit['max_prompt_plus_output_tokens']:,} output={args.output_root / 'inputs.jsonl'}")
    if args.preflight_only:
        progress.close()
        progress.log("[preflight complete] paths, cohort, split, manifest, prompt length, and output contract passed; no model loaded")
        return
    run_worker(args, "classify", progress, 2)
    progress.log(f"[stage 2/3 complete] output={args.output_root / 'predictions.jsonl'}")
    run_worker(args, "aggregate", progress, 3)
    progress.log(f"[stage 3/3 complete] report={args.output_root / 'report.md'}")
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

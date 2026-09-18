#!/usr/bin/env python3
"""Build a budgeted context-preserving Evidence Graph from existing propositions.

This workflow never regenerates propositions.  It selects already validated
Standalone-v2 propositions, extracts exact participant spans, normalizes only
high-confidence UMLS surfaces, and writes a proposition factor graph.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_rag2_context_preserving_proposition_graph_pilot as pilot  # noqa: E402
import build_rag2_paired_raw_proposition_kg_pilot as common  # noqa: E402


RUN_VERSION = "rag2_context_graph_existing_propositions_budget600k_v1"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
DEFAULT_QUALITY_ROOT = ROOT / "datasets/kg/rag2_train60k_test6545_standalone_v2_usable_exact_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_context_graph_existing_propositions_600k_v1"
DEFAULT_MODEL = WORKSPACE / "models/Qwen3.5-9B"
DEFAULT_PYTHON = WORKSPACE / ".venv_vllm/bin/python"
DEFAULT_REFERENCE = ROOT / "datasets/kg/rag2_context_preserving_proposition_graph_pilot_768_v3/inputs.jsonl"
PROGRESS_PREFIX = "@@BUDGET_CONTEXT_GRAPH_PROGRESS@@"
GENERIC_SURFACES = pilot.GENERIC_BRIDGES | {
    "information", "method", "methods", "process", "processes", "system", "systems",
    "value", "values", "variable", "variables",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orchestrate", "calibrate", "select", "extract", "normalize", "graph"), default="orchestrate")
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--reference-input", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--target-propositions", type=int, default=600_000)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--generation-shard-size", type=int, default=256)
    parser.add_argument("--normalization-batch-size", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=192)
    parser.add_argument("--umls-score-min", type=float, default=0.90)
    parser.add_argument("--umls-margin-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260918)
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


def stable_rank(seed: int, identifier: str) -> int:
    return int(hashlib.sha256(f"{seed}|{identifier}".encode()).hexdigest()[:16], 16)


def normalized_surface(value: str) -> str:
    return " ".join("".join(character.casefold() if character.isalnum() else " " for character in value).split())


def iter_batches(path: Path, size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in common.read_jsonl(path):
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def compact_input(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "proposition_id": str(row["proposition_id"]),
        "document_id": str(row["stable_id"]),
        "source": str(row["source"]),
        "origin_split": split,
        "claim": str(row["claim"]),
        "evidence": str(row["evidence"]),
        "text_sha256": str(row.get("text_sha256") or ""),
    }


def source_targets(args: argparse.Namespace, test_counts: Mapping[str, int]) -> dict[str, int]:
    if args.target_propositions < sum(test_counts.values()):
        raise RuntimeError("Target is smaller than the mandatory test proposition cohort")
    train_summary = json.loads((args.quality_root / "train/summary.json").read_text(encoding="utf-8"))
    test_summary = json.loads((args.quality_root / "test/summary.json").read_text(encoding="utf-8"))
    weights = {
        source: int(train_summary["by_source"][source]["usable"]) + int(test_summary["by_source"][source]["usable"])
        for source in SOURCES
    }
    total_weight = sum(weights.values())
    raw = {source: args.target_propositions * weights[source] / total_weight for source in SOURCES}
    targets = {source: max(test_counts[source], math.floor(raw[source])) for source in SOURCES}
    while sum(targets.values()) < args.target_propositions:
        candidates = sorted(SOURCES, key=lambda source: (raw[source] - math.floor(raw[source]), source), reverse=True)
        for source in candidates:
            if sum(targets.values()) >= args.target_propositions:
                break
            targets[source] += 1
    while sum(targets.values()) > args.target_propositions:
        candidates = sorted(SOURCES, key=lambda source: (targets[source] - test_counts[source], source), reverse=True)
        changed = False
        for source in candidates:
            if sum(targets.values()) <= args.target_propositions:
                break
            if targets[source] > test_counts[source]:
                targets[source] -= 1
                changed = True
        if not changed:
            raise RuntimeError("Target is smaller than the mandatory test proposition cohort")
    return targets


def selection_metadata(args: argparse.Namespace) -> dict[str, Any]:
    test_summary = json.loads((args.quality_root / "test/summary.json").read_text(encoding="utf-8"))
    test_counts = {source: int(test_summary["by_source"][source]["usable"]) for source in SOURCES}
    test_total = sum(test_counts.values())
    if args.target_propositions < test_total:
        raise RuntimeError(f"target_propositions={args.target_propositions} is below mandatory test cohort={test_total}")
    targets = source_targets(args, test_counts)
    return {
        "target": args.target_propositions,
        "test_total": test_total,
        "test_by_source": test_counts,
        "target_by_source": targets,
        "train_additions_by_source": {source: targets[source] - test_counts[source] for source in SOURCES},
    }


def ensure_manifest(args: argparse.Namespace, selection: Mapping[str, Any]) -> None:
    train_path = args.quality_root / "train/usable_propositions.jsonl"
    test_path = args.quality_root / "test/usable_propositions.jsonl"
    contract = {
        "run_version": RUN_VERSION,
        "purpose": "Build a context-preserving proposition factor graph from existing usable propositions; proposition generation is not repeated.",
        "cohort": dict(selection),
        "inputs": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "test": {"path": str(test_path), "sha256": sha256_file(test_path)},
            "quality_manifest_sha256": sha256_file(args.quality_root / "manifest.json"),
        },
        "model": str(args.model.resolve()),
        "prompt_sha256": common.sha256_text(pilot.SYSTEM_PROMPT),
        "generation": {
            "batch_size": args.generation_batch_size, "shard_size": args.generation_shard_size,
            "max_output_tokens": args.max_output_tokens, "temperature": 0.0, "seed": args.seed,
        },
        "normalization": {
            "surface_cache": True, "umls_score_min": args.umls_score_min,
            "umls_margin_min": args.umls_margin_min, "embedding_only_merge": False,
        },
        "graph": {
            "context_policy": "full proposition text is retained; demographic/study context is not a hard condition",
            "typed_entity_relations": "none", "lexical_index_is_identity": False,
        },
        "leakage": "Uses retrieved source documents and proposition text only; no question, answer, semantic label, or behavioral label.",
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract["contract_sha256"] = common.sha256_text(common.canonical_json(contract))
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract["contract_sha256"] and any((args.output_root / name).exists() for name in ("selected_propositions.jsonl", "participant_shards", "umls_cache.sqlite3", "graph")):
            raise RuntimeError("Budgeted context graph resume contract mismatch; use a new output root")
    common.atomic_json(path, contract)


def select_worker(args: argparse.Namespace) -> None:
    metadata = selection_metadata(args)
    test_path = args.quality_root / "test/usable_propositions.jsonl"
    train_path = args.quality_root / "train/usable_propositions.jsonl"
    emit("start", total=args.target_propositions, completed=0, name="select existing propositions without regeneration")
    test_rows: list[dict[str, Any]] = []
    test_ids: set[str] = set()
    for row in common.read_jsonl(test_path):
        proposition_id = str(row["proposition_id"])
        if proposition_id in test_ids:
            continue
        test_ids.add(proposition_id)
        test_rows.append(compact_input(row, "test"))
    emit("progress", total=args.target_propositions, completed=len(test_rows))
    additions = metadata["train_additions_by_source"]
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {source: [] for source in SOURCES}
    for row in common.read_jsonl(train_path):
        proposition_id = str(row["proposition_id"])
        source = str(row["source"])
        quota = int(additions.get(source, 0))
        if quota <= 0 or proposition_id in test_ids:
            continue
        rank = stable_rank(args.seed, proposition_id)
        item = (-rank, proposition_id, compact_input(row, "train"))
        heap = heaps[source]
        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    train_rows = [item[2] for source in SOURCES for item in heaps[source]]
    if len(test_rows) + len(train_rows) != args.target_propositions:
        raise RuntimeError(f"Selection incomplete: test={len(test_rows)} train={len(train_rows)} target={args.target_propositions}")
    test_rows.sort(key=lambda row: (row["source"], stable_rank(args.seed, row["proposition_id"])))
    train_rows.sort(key=lambda row: (row["source"], stable_rank(args.seed, row["proposition_id"])))
    output = args.output_root / "selected_propositions.jsonl"
    common.atomic_jsonl(output, [*test_rows, *train_rows])
    common.atomic_json(args.output_root / "selection_summary.json", {**metadata, "selected_test": len(test_rows), "selected_train": len(train_rows)})
    emit("complete", total=args.target_propositions, completed=args.target_propositions)


def prompt_for_claim(claim: str) -> str:
    return "PROP:\n" + claim + "\n\nNOISY SPAN HINTS:\n[]"


def parse_output(row: Mapping[str, Any], raw_output: str) -> dict[str, Any]:
    base = {**row, "candidate_hints": [], "gold_entities": []}
    parsed = pilot.parse_prediction(base, raw_output)
    return {
        **row,
        "participants": parsed["predicted_entities"],
        "schema_valid": parsed["valid_output"],
        "output_errors": parsed["output_errors"],
        "raw_output": raw_output,
    }


def valid_shard(path: Path, expected_ids: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = list(common.read_jsonl(path))
    except (OSError, ValueError):
        return False
    return [str(row.get("proposition_id")) for row in rows] == list(expected_ids)


def extraction_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    selected = args.output_root / "selected_propositions.jsonl"
    shard_root = args.output_root / "participant_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, batch in enumerate(iter_batches(selected, args.generation_shard_size)):
        path = shard_root / f"batch_{index:06d}.jsonl"
        ids = [str(row["proposition_id"]) for row in batch]
        if valid_shard(path, ids):
            completed += len(batch)
    emit("start", total=args.target_propositions, completed=completed, name="Qwen participant extraction from existing propositions")
    tokenizer = llm = None
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed)
    for index, batch in enumerate(iter_batches(selected, args.generation_shard_size)):
        path = shard_root / f"batch_{index:06d}.jsonl"
        ids = [str(row["proposition_id"]) for row in batch]
        if valid_shard(path, ids):
            continue
        if llm is None:
            tokenizer, llm = common.init_vllm(args.model, args)
        prompts = [common.chat_prompt(tokenizer, pilot.SYSTEM_PROMPT, prompt_for_claim(str(row["claim"]))) for row in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        values = [parse_output(row, output.outputs[0].text.strip() if output.outputs else "") for row, output in zip(batch, outputs)]
        common.atomic_jsonl(path, values)
        completed += len(batch)
        emit("progress", total=args.target_propositions, completed=completed)
    emit("complete", total=args.target_propositions, completed=args.target_propositions)


def calibration_worker(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    rows = list(common.read_jsonl(args.reference_input))
    root = args.output_root / "no_hint_quality_calibration"
    output = root / "predictions.jsonl"
    emit("start", total=len(rows), completed=len(rows) if output.is_file() else 0, name="verify no-hint participant extraction quality")
    if not output.is_file():
        tokenizer, llm = common.init_vllm(args.model, args)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed)
        predictions = []
        for start in range(0, len(rows), args.generation_batch_size):
            batch = rows[start:start + args.generation_batch_size]
            prompts = [common.chat_prompt(tokenizer, pilot.SYSTEM_PROMPT, prompt_for_claim(str(row["claim"]))) for row in batch]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for row, generated in zip(batch, outputs):
                predictions.append(pilot.parse_prediction(row, generated.outputs[0].text.strip() if generated.outputs else ""))
            emit("progress", total=len(rows), completed=len(predictions))
        common.atomic_jsonl(output, predictions)
    predictions = list(common.read_jsonl(output))
    metrics = pilot.score_participants(predictions)
    passed = metrics["boundary_tolerant"]["precision"] >= .85 and metrics["boundary_tolerant"]["recall"] >= .80 and metrics["clean_two_participant_rate"] >= .70 and metrics["valid_output_rate"] >= .99
    common.atomic_json(root / "summary.json", {"rows": len(rows), "metrics": metrics, "passed": passed})
    if not passed:
        raise RuntimeError(f"No-hint participant extraction quality gate failed: {metrics}")
    emit("complete", total=len(rows), completed=len(rows))


def iter_prediction_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    for path in sorted((args.output_root / "participant_shards").glob("batch_*.jsonl")):
        yield from common.read_jsonl(path)


def normalization_worker(args: argparse.Namespace) -> None:
    from build_rag2_medical_kg_mvp import load_pipeline
    from medrag.kg.biomedical_mvp import category_from_tuis

    surfaces: dict[str, str] = {}
    for row in iter_prediction_rows(args):
        for participant in row.get("participants") or []:
            key = normalized_surface(str(participant["surface"]))
            if key:
                surfaces.setdefault(key, str(participant["surface"]))
    database = args.output_root / "umls_cache.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS surfaces (surface_key TEXT PRIMARY KEY, surface TEXT NOT NULL, cui TEXT, canonical_name TEXT, category TEXT, score REAL, margin REAL, status TEXT NOT NULL)")
    existing = {str(row[0]) for row in connection.execute("SELECT surface_key FROM surfaces")}
    missing = [(key, surface) for key, surface in surfaces.items() if key not in existing]
    emit("start", total=len(surfaces), completed=len(existing & surfaces.keys()), name="cache strict UMLS normalization by unique surface")
    if missing:
        _, linker = load_pipeline("en_core_sci_sm", "umls", 0.70, 5)
        done = len(existing & surfaces.keys())
        for start in range(0, len(missing), args.normalization_batch_size):
            batch = missing[start:start + args.normalization_batch_size]
            generated = linker.candidate_generator([surface for _, surface in batch], k=5)
            records = []
            for (key, surface), candidates in zip(batch, generated):
                values = []
                for candidate in candidates:
                    entity = linker.kb.cui_to_entity.get(candidate.concept_id)
                    if entity is None:
                        continue
                    score = max((float(value) for value in candidate.similarities), default=0.0)
                    values.append((score, candidate.concept_id, entity))
                values.sort(key=lambda value: (-value[0], value[1]))
                top = values[0][0] if values else 0.0
                second = values[1][0] if len(values) > 1 else 0.0
                margin = top - second
                accepted = bool(values and top >= args.umls_score_min and (len(values) == 1 or margin >= args.umls_margin_min))
                if accepted:
                    _, cui, entity = values[0]
                    tuis = sorted(str(value) for value in entity.types)
                    records.append((key, surface, cui, entity.canonical_name, category_from_tuis(tuis), top, margin, "linked_confident"))
                else:
                    records.append((key, surface, None, None, None, top if values else None, margin if values else None, "not_linked"))
            connection.executemany("INSERT OR REPLACE INTO surfaces VALUES (?,?,?,?,?,?,?,?)", records)
            connection.commit()
            done += len(batch)
            emit("progress", total=len(surfaces), completed=done)
    connection.close()
    emit("complete", total=len(surfaces), completed=len(surfaces))


def jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    handle = temporary.open("w", encoding="utf-8")
    return handle, temporary


def write_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def graph_worker(args: argparse.Namespace) -> None:
    connection = sqlite3.connect(args.output_root / "umls_cache.sqlite3")
    normalizations = {
        str(row[0]): {"cui": row[1], "name": row[2], "category": row[3], "score": row[4], "status": row[5]}
        for row in connection.execute("SELECT surface_key,cui,canonical_name,category,score,status FROM surfaces")
    }
    connection.close()
    graph_root = args.output_root / "graph"
    names = ("statements", "participants", "concepts", "lexical_indexes", "edges")
    writers = {name: jsonl_writer(graph_root / f"{name}.jsonl") for name in names}
    documents: dict[str, dict[str, Any]] = {}
    umls_concepts: dict[str, dict[str, Any]] = {}
    lexical: dict[str, dict[str, Any]] = {}
    counts = Counter()
    emit("start", total=args.target_propositions, completed=0, name="write context-preserving proposition factor graph")
    try:
        for row in iter_prediction_rows(args):
            proposition_id = str(row["proposition_id"]); document_id = str(row["document_id"])
            documents.setdefault(document_id, {"node_id": document_id, "node_type": "Document", "source": row["source"]})
            write_row(writers["statements"][0], {"node_id": proposition_id, "node_type": "EvidenceStatement", "document_id": document_id, "text": row["claim"], "evidence": row["evidence"], "context_policy": "preserved_not_hard_condition"})
            write_row(writers["edges"][0], {"source": document_id, "relation": "HAS_EVIDENCE_STATEMENT", "target": proposition_id})
            counts["statements"] += 1
            for index, participant in enumerate(row.get("participants") or [], start=1):
                mention_id = f"{proposition_id}::ARG{index}"
                surface = str(participant["surface"]); key = normalized_surface(surface)
                normalization = normalizations.get(key) or {"status": "not_linked"}
                write_row(writers["participants"][0], {"node_id": mention_id, "node_type": "ParticipantMention", "statement_id": proposition_id, "document_id": document_id, "surface": surface, "start": participant["start"], "end": participant["end"]})
                write_row(writers["edges"][0], {"source": proposition_id, "relation": "HAS_ARGUMENT", "target": mention_id})
                if normalization.get("status") == "linked_confident" and normalization.get("cui"):
                    concept_id = f"UMLS::{normalization['cui']}"
                    umls_concepts.setdefault(concept_id, {"node_id": concept_id, "node_type": "CanonicalEntity", "name": normalization["name"], "category": normalization["category"], "normalization": "strict_umls_surface", "score": normalization["score"]})
                    counts["umls_mentions"] += 1
                else:
                    concept_id = f"LOCAL::{mention_id}"
                    write_row(writers["concepts"][0], {"node_id": concept_id, "node_type": "CanonicalEntity", "name": surface, "normalization": "local_unmerged"})
                    counts["local_mentions"] += 1
                write_row(writers["edges"][0], {"source": mention_id, "relation": "DENOTES", "target": concept_id})
                if key and key not in GENERIC_SURFACES and len(key) >= 4:
                    lexical_id = f"LEX::{key}"
                    lexical.setdefault(lexical_id, {"node_id": lexical_id, "node_type": "LexicalIndex", "surface_normalized": key, "identity_assertion": False})
                    write_row(writers["edges"][0], {"source": mention_id, "relation": "HAS_LEXICAL_INDEX", "target": lexical_id, "identity_assertion": False})
                counts["participants"] += 1
            if counts["statements"] % 4096 == 0:
                emit("progress", total=args.target_propositions, completed=counts["statements"])
        for value in umls_concepts.values():
            write_row(writers["concepts"][0], value)
        for value in lexical.values():
            write_row(writers["lexical_indexes"][0], value)
    finally:
        for handle, _ in writers.values():
            handle.close()
    for name, (_, temporary) in writers.items():
        os.replace(temporary, graph_root / f"{name}.jsonl")
    common.atomic_jsonl(graph_root / "documents.jsonl", documents.values())
    summary = {
        "run_version": RUN_VERSION, "target_propositions": args.target_propositions,
        "documents": len(documents), "statements": counts["statements"], "participants": counts["participants"],
        "strict_umls_mentions": counts["umls_mentions"], "local_mentions": counts["local_mentions"],
        "strict_umls_concepts": len(umls_concepts), "lexical_indexes": len(lexical),
        "context_policy": "All proposition text and evidence retained; patient/study/demographic context is not interpreted as a necessary condition.",
    }
    common.atomic_json(args.output_root / "summary.json", summary)
    common.atomic_json(args.output_root / "COMPLETE.json", {"run_version": RUN_VERSION, "complete": counts["statements"] == args.target_propositions})
    emit("complete", total=args.target_propositions, completed=counts["statements"])


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.llm_python), str(Path(__file__).resolve()), "--mode", mode,
        "--quality-root", str(args.quality_root), "--output-root", str(args.output_root),
        "--model", str(args.model), "--llm-python", str(args.llm_python),
        "--reference-input", str(args.reference_input), "--target-propositions", str(args.target_propositions),
        "--generation-batch-size", str(args.generation_batch_size), "--generation-shard-size", str(args.generation_shard_size),
        "--normalization-batch-size", str(args.normalization_batch_size), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len), "--max-output-tokens", str(args.max_output_tokens),
        "--umls-score-min", str(args.umls_score_min), "--umls-margin-min", str(args.umls_margin_min), "--seed", str(args.seed),
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
                progress.start(stage, int(event["total"]), int(event.get("completed", 0))); started = True
            elif event["event"] in {"progress", "complete"}:
                if not started:
                    progress.start(stage, int(event["total"]), int(event.get("completed", 0))); started = True
                progress.update(int(event["completed"]), int(event["total"]))
        elif value and not value.startswith("Loading weights:"):
            progress.log(value)
    code = process.wait()
    if code:
        raise RuntimeError(f"Worker {mode} failed with exit code {code}")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.quality_root / "manifest.json", args.quality_root / "train/usable_propositions.jsonl",
        args.quality_root / "test/usable_propositions.jsonl", args.quality_root / "train/summary.json",
        args.quality_root / "test/summary.json", args.model, args.llm_python, args.reference_input,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    selection = selection_metadata(args)
    ensure_manifest(args, selection)
    usage = os.statvfs(args.output_root)
    free_bytes = usage.f_bavail * usage.f_frsize
    if free_bytes < 60 * 1024**3:
        raise RuntimeError(f"At least 60 GiB free space required; available={free_bytes/1024**3:.1f} GiB")
    value = {"selection": selection, "free_gib": free_bytes / 1024**3, "propositions_already_exist": True}
    common.atomic_json(args.output_root / "preflight.json", value)
    return value


def orchestrate(args: argparse.Namespace) -> None:
    info = preflight(args)
    stages = [
        "verify no-hint participant quality", "select existing usable propositions", "extract participant spans",
        "normalize unique surfaces", "write proposition factor graph",
    ]
    estimates = [75, 90, 7.5 * 3600, 2.5 * 3600, 45 * 60]
    progress = common.WorkflowProgress(stages, estimates)
    selection = info["selection"]
    progress.log(f"[workflow plan] existing propositions only; target={args.target_propositions} test={selection['test_total']} train_additions={args.target_propositions-selection['test_total']} free={info['free_gib']:.1f}GiB")
    progress.log("[runtime estimate] H200 GPU 1: approximately 6-12 hours after empty-cache calibration; first 10k extraction updates the measured ETA")
    if args.preflight_only:
        progress.start(1, 1, 1); progress.close(); progress.log("[preflight complete] no model loaded and no long job started"); return
    for stage, mode in enumerate(("calibrate", "select", "extract", "normalize", "graph"), start=1):
        run_worker(args, mode, progress, stage)
        progress.log(f"[stage {stage}/5 complete] mode={mode} output={args.output_root}")
    progress.close(); progress.log(f"[workflow complete] result={args.output_root}")


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate": orchestrate(args)
    elif args.mode == "calibrate": calibration_worker(args)
    elif args.mode == "select": select_worker(args)
    elif args.mode == "extract": extraction_worker(args)
    elif args.mode == "normalize": normalization_worker(args)
    elif args.mode == "graph": graph_worker(args)
    else: raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

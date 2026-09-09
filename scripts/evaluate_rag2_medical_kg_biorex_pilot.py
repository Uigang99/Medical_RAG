#!/usr/bin/env python3
"""Evaluate official BioREx relation classification on frozen KG-MVP entities.

The pilot intentionally reuses the existing scispaCy/UMLS mentions.  It changes
only the relation extractor, making comparison with the rule/hybrid pilots
interpretable.  BioREx-native labels are retained without mapping them to the
project's rule relation schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_rag2_medical_kg_mvp import (
    TwoLineProgress,
    analysis_text,
    atomic_json,
    atomic_jsonl,
    canonical_json,
    read_jsonl,
    sha256_text,
)

RUN_VERSION = "rag2_medical_kg_biorex_relation_pilot_v1"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
BIOREX_REPO_COMMIT = "1f199d8c359eaceb5469538f4daa24aac44c4eae"
TYPE_MAP = {
    "DrugOrChemical": "ChemicalEntity",
    "GeneOrProtein": "GeneOrGeneProduct",
    "DiseaseOrDisorder": "DiseaseOrPhenotypicFeature",
    "FindingOrSymptom": "DiseaseOrPhenotypicFeature",
}
SUPPORTED_PAIRS = {
    tuple(sorted(pair))
    for pair in (
        ("ChemicalEntity", "ChemicalEntity"),
        ("ChemicalEntity", "DiseaseOrPhenotypicFeature"),
        ("ChemicalEntity", "GeneOrGeneProduct"),
        ("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
    )
}
PRIMARY_LABELS = {
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
}
OFFICIAL_LABELS = [
    "None", "Association", "Bind", "Comparison", "Conversion", "Cotreatment",
    "Drug_Interaction", "Negative_Correlation", "Positive_Correlation", "None-CID",
    "CID", "None-PPIm", "PPIm", "None-AIMED", "None-DDI", "None-BC7",
    "None-phargkb", "None-GDA", "None-DISGENET", "None-EMU_BC", "None-EMU_PC",
    "None-HPRD50", "None-PHARMGKB", "ACTIVATOR", "AGONIST", "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR", "ANTAGONIST", "DIRECT-REGULATOR",
    "INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR", "INHIBITOR", "PART-OF",
    "PRODUCT-OF", "SUBSTRATE", "SUBSTRATE_PRODUCT-OF", "mechanism", "int",
    "effect", "advise", "AIMED-Association", "HPRD-Association", "EUADR-Association",
    "None-EUADR", "Indirect_conversion", "Non_conversion",
]
GENERIC_SURFACES = {
    "background", "conclusion", "conclusions", "data", "day", "days", "figure",
    "figures", "method", "methods", "patient", "patients", "result", "results",
    "study", "studies", "table", "tables", "year", "years",
}


def parse_args() -> argparse.Namespace:
    home = Path("/home/user/Uiheon")
    root = home / "Medical_RAG"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "datasets/kg/rag2_candidate_medical_kg_mvp_v2")
    parser.add_argument("--output-root", type=Path, default=root / "datasets/kg/rag2_candidate_medical_kg_biorex_pilot_v1")
    parser.add_argument("--biorex-repo", type=Path, default=home / "tools/BioREx")
    parser.add_argument("--model", type=Path, default=home / "models/BioREx-BioLinkBERT")
    parser.add_argument("--documents-per-source", type=int, default=100)
    parser.add_argument("--minimum-link-score", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--audit-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def iter_shards(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted((root / "extraction_shards").glob("shard_*.jsonl")):
        yield from read_jsonl(path)


def stable_sample(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        values = by_source[source]
        if len(values) < count:
            raise RuntimeError(f"Source {source} has {len(values)} chunks; requested {count}")
        values = sorted(
            values,
            key=lambda value: hashlib.sha256(f"{seed}\0{value['stable_id']}".encode()).hexdigest(),
        )[:count]
        selected.extend(dict(value) for value in values)
    return sorted(selected, key=lambda value: (value["source"], value["stable_id"]))


def eligible_mentions(
    row: Mapping[str, Any], minimum_score: float
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return non-overlapping, BioREx-supported, linked mentions."""
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for raw in row.get("mentions", []):
        mention = dict(raw)
        if not mention.get("linked"):
            rejected["unlinked"] += 1
            continue
        if float(mention.get("link_score") or 0.0) < minimum_score:
            rejected["low_link_score"] += 1
            continue
        mapped_type = TYPE_MAP.get(str(mention.get("category")))
        if mapped_type is None:
            rejected[f"unsupported_type:{mention.get('category')}"] += 1
            continue
        surface = str(mention.get("surface") or "").strip()
        compact = "".join(character for character in surface if character.isalnum())
        if len(compact) <= 2:
            rejected["too_short"] += 1
            continue
        if surface.casefold() in GENERIC_SURFACES:
            rejected["generic_surface"] += 1
            continue
        entity_id = str(mention.get("entity_id") or "").strip()
        if not entity_id:
            rejected["missing_entity_id"] += 1
            continue
        mention["biorex_type"] = mapped_type
        candidates.append(mention)

    # Exact duplicate mentions are collapsed first. Overlaps cannot be encoded
    # safely by the official IOB conversion, so highest-link-score/longest wins.
    unique: dict[tuple[int, int, str], dict[str, Any]] = {}
    for mention in candidates:
        key = (int(mention["start"]), int(mention["end"]), str(mention["entity_id"]))
        if key not in unique or float(mention["link_score"]) > float(unique[key]["link_score"]):
            unique[key] = mention
    ranked = sorted(
        unique.values(),
        key=lambda item: (-float(item["link_score"]), -(int(item["end"]) - int(item["start"])), int(item["start"])),
    )
    accepted: list[dict[str, Any]] = []
    for mention in ranked:
        start, end = int(mention["start"]), int(mention["end"])
        if any(start < int(other["end"]) and int(other["start"]) < end for other in accepted):
            rejected["overlapping_span"] += 1
            continue
        accepted.append(mention)

    # One concept ID must have one entity type inside a PubTator document.
    best_type: dict[str, str] = {}
    best_score: dict[str, float] = {}
    for mention in accepted:
        entity_id = str(mention["entity_id"])
        score = float(mention["link_score"])
        if score > best_score.get(entity_id, -1.0):
            best_score[entity_id] = score
            best_type[entity_id] = str(mention["biorex_type"])
    final = []
    for mention in accepted:
        if str(mention["biorex_type"]) != best_type[str(mention["entity_id"])]:
            rejected["concept_type_conflict"] += 1
            continue
        final.append(mention)
    return sorted(final, key=lambda item: (int(item["start"]), int(item["end"]))), rejected


def has_supported_pair(mentions: Sequence[Mapping[str, Any]]) -> bool:
    entities: dict[str, str] = {}
    for mention in mentions:
        entities[str(mention["entity_id"])] = str(mention["biorex_type"])
    values = list(entities.items())
    return any(tuple(sorted((left[1], right[1]))) in SUPPORTED_PAIRS for i, left in enumerate(values) for right in values[i + 1 :])


def write_pubtator(
    selected: Sequence[Mapping[str, Any]], path: Path, minimum_score: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    mapping: list[dict[str, Any]] = []
    rejected_total: Counter[str] = Counter()
    supported_pair_docs = 0
    accepted_mention_count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected, start=1):
            doc_id = f"R2KG{index:06d}"
            text, _ = analysis_text(row)
            # PubTator requires one physical line per text field. Replacing each
            # control character one-for-one preserves all entity offsets.
            text = text.translate(str.maketrans({"\n": " ", "\r": " ", "\t": " "}))
            mentions, rejected = eligible_mentions(row, minimum_score)
            accepted_mention_count += len(mentions)
            rejected_total.update(rejected)
            if has_supported_pair(mentions):
                supported_pair_docs += 1
            handle.write(f"{doc_id}|t|{text}\n")
            entity_metadata: dict[str, dict[str, Any]] = {}
            for mention in mentions:
                entity_id = str(mention["entity_id"])
                handle.write(
                    "\t".join(
                        [doc_id, str(mention["start"]), str(mention["end"]), str(mention["surface"]),
                         str(mention["biorex_type"]), entity_id]
                    ) + "\n"
                )
                meta = entity_metadata.setdefault(
                    entity_id,
                    {"entity_id": entity_id, "type": mention["biorex_type"], "surfaces": [], "canonical_names": []},
                )
                if mention["surface"] not in meta["surfaces"]:
                    meta["surfaces"].append(mention["surface"])
                if mention.get("canonical_name") and mention["canonical_name"] not in meta["canonical_names"]:
                    meta["canonical_names"].append(mention["canonical_name"])
            handle.write("\n")
            mapping.append(
                {
                    "biorex_doc_id": doc_id,
                    "stable_id": row["stable_id"],
                    "source": row["source"],
                    "title": row.get("title", ""),
                    "text": text,
                    "entities": entity_metadata,
                }
            )
    os.replace(temp, path)
    return mapping, {
        "documents": len(selected),
        "documents_with_supported_pair": supported_pair_docs,
        "accepted_mentions": accepted_mention_count,
        "accepted_unique_concepts": sum(len(row["entities"]) for row in mapping),
        "rejected_mentions": dict(sorted(rejected_total.items())),
    }


def preflight(args: argparse.Namespace) -> str:
    required = [
        args.input_root / "COMPLETE.json",
        args.input_root / "selected_chunks.jsonl",
        args.biorex_repo / "src/dataset_format_converter/convert_pubtator_2_tsv.py",
        args.biorex_repo / "src/run_ncbi_rel_exp.py",
        args.model / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing BioREx pilot dependency:\n  " + "\n  ".join(missing))
    commit = subprocess.check_output(["git", "-C", str(args.biorex_repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != BIOREX_REPO_COMMIT:
        raise RuntimeError(f"BioREx repo contract mismatch: expected {BIOREX_REPO_COMMIT}, got {commit}")
    if args.documents_per_source <= 0 or args.batch_size <= 0:
        raise ValueError("documents-per-source and batch-size must be positive")
    return commit


def repository_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def model_contract(path: Path) -> dict[str, Any]:
    files = sorted(
        (
            {"name": item.name, "size": item.stat().st_size}
            for item in path.iterdir()
            if item.is_file()
        ),
        key=lambda item: item["name"],
    )
    return {
        "path": str(path.resolve()),
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "files": files,
    }


def run_converter(args: argparse.Namespace, pubtator: Path, processed: Path) -> None:
    command = [
        sys.executable,
        str(args.biorex_repo / "src/dataset_format_converter/convert_pubtator_2_tsv.py"),
        "--exp_option", "biored_pred",
        "--in_pubtator_file", str(pubtator),
        "--out_tsv_file", str(processed),
    ]
    completed = subprocess.run(command, cwd=args.biorex_repo, text=True, capture_output=True)
    (args.output_root / "converter.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.output_root / "converter.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Official BioREx converter failed; inspect {args.output_root}/converter.stderr.log")
    if not processed.is_file():
        raise RuntimeError("Official BioREx converter did not create processed TSV")


def read_processed(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not values:
                continue
            if len(values) != 10:
                raise ValueError(f"Unexpected BioREx TSV width at {path}:{line_number}: {len(values)}")
            rows.append(
                dict(zip(
                    ("doc_id", "type1", "type2", "id1", "id2", "same_section", "min_section_window", "text", "neighbors", "gold_label"),
                    values,
                ))
            )
    return rows


def softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def load_biorex(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Load the official TensorFlow checkpoint and return model/tokenizer."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow sees no GPU. Run with CUDA_VISIBLE_DEVICES=1 after setup preflight.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model)
    return model, tokenizer, {"tensorflow": tf.__version__, "gpu_devices": [str(gpu) for gpu in gpus]}


def classify(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, str]],
    progress: TwoLineProgress,
    model: Any,
    tokenizer: Any,
) -> list[list[float]]:

    logits_all: list[list[float]] = []
    first_batch_started = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="tf",
        )
        logits = model(**encoded, training=False).logits.numpy().tolist()
        if logits and len(logits[0]) != len(OFFICIAL_LABELS):
            raise RuntimeError(f"BioREx checkpoint has {len(logits[0])} labels; expected {len(OFFICIAL_LABELS)}")
        logits_all.extend(logits)
        progress.update(len(batch))
        if start == 0:
            elapsed = max(time.time() - first_batch_started, 1e-6)
            rate = len(batch) / elapsed
            progress.stage_estimates[3] = len(rows) / max(rate, 1e-6)
            progress.render(force=True)
    return logits_all


def prediction_rows(
    processed: Sequence[Mapping[str, str]],
    logits: Sequence[Sequence[float]],
    doc_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(processed) != len(logits):
        raise RuntimeError(f"Prediction alignment mismatch: rows={len(processed)} logits={len(logits)}")
    output: list[dict[str, Any]] = []
    for pair, values in zip(processed, logits):
        probabilities = softmax(values)
        order = sorted(range(len(values)), key=lambda index: probabilities[index], reverse=True)
        winner = order[0]
        label = OFFICIAL_LABELS[winner]
        document = doc_map[pair["doc_id"]]
        left = document["entities"].get(pair["id1"], {})
        right = document["entities"].get(pair["id2"], {})
        output.append(
            {
                "biorex_doc_id": pair["doc_id"],
                "stable_id": document["stable_id"],
                "source": document["source"],
                "title": document["title"],
                "entity1_id": pair["id1"],
                "entity1_type": pair["type1"],
                "entity1_surfaces": left.get("surfaces", []),
                "entity2_id": pair["id2"],
                "entity2_type": pair["type2"],
                "entity2_surfaces": right.get("surfaces", []),
                "same_section": pair["same_section"].casefold() == "true",
                "minimum_section_window": int(pair["min_section_window"]),
                "predicted_label": label,
                "label_scope": "primary" if label in PRIMARY_LABELS else ("negative" if label.startswith("None") else "auxiliary_training_label"),
                "confidence": probabilities[winner],
                "top2_margin": probabilities[winner] - probabilities[order[1]],
                "top3": [{"label": OFFICIAL_LABELS[index], "probability": probabilities[index]} for index in order[:3]],
            }
        )
    return output


def summarize(
    selected: Sequence[Mapping[str, Any]], prep: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    primary = [row for row in predictions if row["label_scope"] == "primary"]
    confidence_bins = Counter()
    for row in primary:
        confidence = float(row["confidence"])
        confidence_bins[">=0.95" if confidence >= 0.95 else ">=0.80" if confidence >= 0.80 else ">=0.50" if confidence >= 0.50 else "<0.50"] += 1
    positive_docs = {row["stable_id"] for row in primary}
    return {
        "scope": {
            "documents": len(selected),
            "documents_by_source": dict(sorted(Counter(row["source"] for row in selected).items())),
            "candidate_pairs": len(predictions),
        },
        "entity_preparation": prep,
        "predictions": {
            "all_label_counts": dict(sorted(Counter(row["predicted_label"] for row in predictions).items())),
            "scope_counts": dict(sorted(Counter(row["label_scope"] for row in predictions).items())),
            "primary_relation_counts": dict(sorted(Counter(row["predicted_label"] for row in primary).items())),
            "primary_confidence_bins": dict(sorted(confidence_bins.items())),
            "documents_with_primary_relation": len(positive_docs),
            "document_primary_relation_coverage": len(positive_docs) / max(1, len(selected)),
            "pair_type_counts": dict(sorted(Counter(f"{row['entity1_type']}--{row['entity2_type']}" for row in predictions).items())),
        },
        "quality_status": "manual audit required; confidence is not correctness",
    }


def audit_sample(predictions: Sequence[Mapping[str, Any]], documents: Mapping[str, Mapping[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    positives = [dict(row) for row in predictions if row["label_scope"] == "primary"]
    ordered = sorted(
        positives,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['stable_id']}\0{row['entity1_id']}\0{row['entity2_id']}".encode()).hexdigest(),
    )[:size]
    for row in ordered:
        row["text"] = documents[row["biorex_doc_id"]]["text"]
        row["review_entity_pair_valid"] = None
        row["review_relation_correct"] = None
        row["review_direction_correct"] = None
        row["review_notes"] = ""
    return ordered


def main() -> None:
    args = parse_args()
    commit = preflight(args)
    if args.preflight_only:
        print(
            f"[preflight complete] input={args.input_root} model={args.model} "
            f"BioREx_commit={commit} documents={args.documents_per_source * len(SOURCES)}"
        )
        return

    names = [
        "freeze entity cohort and write PubTator",
        "official BioREx pair conversion",
        "load official BioREx checkpoint",
        "classify entity pairs",
        "summarize predictions and write audit",
    ]
    progress = TwoLineProgress(names)
    progress.plan([
        f"400 frozen chunks ({args.documents_per_source}/source)",
        "BioRED-supported entity pairs only",
        "BioLinkBERT TensorFlow checkpoint on visible GPU",
        f"batch={args.batch_size} max_length={args.max_length}",
        f"native labels + {args.audit_size}-pair manual audit",
    ])
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    selected_documents = stable_sample(
        list(read_jsonl(args.input_root / "selected_chunks.jsonl")),
        args.documents_per_source,
        args.seed,
    )
    selected_ids = {str(row["stable_id"]) for row in selected_documents}
    extraction_by_id = {
        str(row["stable_id"]): row
        for row in iter_shards(args.input_root)
        if str(row["stable_id"]) in selected_ids
    }
    missing_extractions = sorted(selected_ids - set(extraction_by_id))
    if missing_extractions:
        raise RuntimeError(f"Missing frozen entity extraction for {len(missing_extractions)} selected chunks")
    selected = [
        {**document, "mentions": extraction_by_id[str(document["stable_id"])].get("mentions", [])}
        for document in selected_documents
    ]
    contract = {
        "run_version": RUN_VERSION,
        "hypothesis": "On the identical frozen entity mentions, BioREx can recover more specific biomedical relation categories than lexical rules at acceptable manually adjudicated precision.",
        "scope_limit": "Relation-classification feasibility only; entity extraction is frozen and no BioREx label is remapped to the project rule schema.",
        "evaluation": {
            "primary_metric": "manual relation precision over the generated audit sample",
            "minimum_worthwhile_result": "precision >= 0.75 with non-zero primary-relation coverage",
            "automatic_outputs": ["native label distribution", "confidence distribution", "document coverage"],
            "warning": "Model confidence is not relation correctness; manual adjudication is required.",
            "reference_baselines": [
                "datasets/kg/rag2_candidate_medical_kg_mvp_v2",
                "datasets/kg/rag2_candidate_medical_kg_hybrid_pilot_v3",
            ],
        },
        "input_root": str(args.input_root.resolve()),
        "input_complete": json.loads((args.input_root / "COMPLETE.json").read_text()),
        "selected_ids_sha256": sha256_text("\n".join(row["stable_id"] for row in selected)),
        "documents_per_source": args.documents_per_source,
        "minimum_link_score": args.minimum_link_score,
        "biorex_repo": str(args.biorex_repo.resolve()),
        "biorex_repo_commit": commit,
        "model": model_contract(args.model),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "supported_type_map": TYPE_MAP,
        "primary_labels": sorted(PRIMARY_LABELS),
        "code_commit": repository_commit(REPOSITORY_ROOT),
        "output_root": str(args.output_root.resolve()),
    }
    contract["contract_sha256"] = sha256_text(canonical_json(contract))
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("BioREx pilot contract mismatch; use a new versioned output root")
    else:
        atomic_json(manifest_path, contract)

    pubtator = args.output_root / "input.pubtator"
    mapping_path = args.output_root / "document_mapping.jsonl"
    progress.start_stage(1, total=len(selected), unit="chunk")
    mapping, prep = write_pubtator(selected, pubtator, args.minimum_link_score)
    atomic_jsonl(mapping_path, mapping)
    progress.update(len(selected))
    progress.complete(f"PubTator={pubtator} supported_pair_docs={prep['documents_with_supported_pair']}")

    processed_path = args.output_root / "processed_pairs.tsv"
    progress.start_stage(2, total=1, unit="conversion")
    run_converter(args, pubtator, processed_path)
    processed = read_processed(processed_path)
    progress.update(1)
    progress.complete(f"candidate_pairs={len(processed)} output={processed_path}")
    if not processed:
        raise RuntimeError("No BioREx-supported entity pair was generated")

    progress.start_stage(3, total=1, unit="checkpoint")
    model, tokenizer, environment = load_biorex(args)
    progress.update(1)
    progress.complete(f"loaded={args.model} TensorFlow={environment['tensorflow']}")

    progress.start_stage(4, total=len(processed), unit="pair")
    progress.stage_estimates[4] = 5.0
    logits = classify(args, processed, progress, model, tokenizer)
    progress.complete(f"classified={len(logits)} TensorFlow={environment['tensorflow']}")

    progress.start_stage(5, total=3, unit="artifact", estimate=5.0)
    doc_map = {row["biorex_doc_id"]: row for row in mapping}
    predictions = prediction_rows(processed, logits, doc_map)
    atomic_jsonl(args.output_root / "predictions.jsonl", predictions)
    progress.update(1)
    positives = [row for row in predictions if row["label_scope"] == "primary"]
    atomic_jsonl(args.output_root / "primary_relations.jsonl", positives)
    with (args.output_root / "raw_logits.tsv").open("w", encoding="utf-8") as handle:
        for values in logits:
            handle.write("\t".join(f"{value:.9g}" for value in values) + "\n")
    progress.update(1)
    summary = summarize(selected, prep, predictions)
    summary["environment"] = environment
    atomic_json(args.output_root / "summary.json", summary)
    atomic_jsonl(args.output_root / "manual_audit_sample.jsonl", audit_sample(predictions, doc_map, args.audit_size, args.seed))
    atomic_json(args.output_root / "COMPLETE.json", {"contract_sha256": contract["contract_sha256"], "candidate_pairs": len(predictions), "primary_relations": len(positives)})
    progress.update(1)
    progress.complete(f"summary={args.output_root / 'summary.json'} audit={args.output_root / 'manual_audit_sample.jsonl'}")
    progress.close(force_complete=True)
    print(
        f"[workflow complete] documents={len(selected)} pairs={len(predictions)} "
        f"primary_relations={len(positives)} output={args.output_root}"
    )


if __name__ == "__main__":
    main()

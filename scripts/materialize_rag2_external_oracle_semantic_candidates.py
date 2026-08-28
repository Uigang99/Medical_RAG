#!/usr/bin/env python3
"""Materialize and verify semantic-label inputs for the external MCQ oracle set.

The evaluation cache stores 128 globally reranked documents for each of the
6,545 external MCQ questions, but it intentionally omits the benchmark text and
gold answer.  This utility joins that cache back to the immutable benchmark
files and materializes rerank ranks 1--32 in the exact input contract consumed
by ``label_rag2_candidates_with_codex.py``.

For transport parity with the original training annotation run, every Top-32
question is represented as four rows of eight documents.  Consequently one
Codex call still receives ten rows / at most 80 question-document pairs, rather
than an unsafe 320-pair prompt.  The semantic unit, pair ID, and original global
rerank rank remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm


DATASETS = (
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)
EXPECTED_DATASET_QUESTIONS = {
    "medmcqa": 4183,
    "medqa": 1273,
    "mmlu_anatomy": 135,
    "mmlu_clinical_knowledge": 265,
    "mmlu_college_biology": 144,
    "mmlu_college_medicine": 173,
    "mmlu_medical_genetics": 100,
    "mmlu_professional_medicine": 272,
}
ANNOTATION_VERSION = "rag2_codex_evidence_utility_label_v2"
PROMPT_VERSION = "rag2_codex_evidence_utility_prompt_v3_compact_item_index"
PREPARATION_VERSION = "rag2_external_oracle_top32_semantic_candidates_v1"
VALID_LABELS = {
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the 6,545-question external-oracle semantic annotation run."
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidate-cache-path", type=Path, required=True)
    prepare.add_argument("--candidate-cache-manifest-path", type=Path, required=True)
    prepare.add_argument("--benchmark-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    prepare.add_argument("--top-k", type=int, default=32)
    prepare.add_argument("--documents-per-block", type=int, default=8)
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--prepared-root", type=Path, required=True)
    verify.add_argument("--label-root", type=Path, required=True)
    verify.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    verify.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def canonical_options(row: dict[str, Any], sample_id: str) -> dict[str, str]:
    options = row.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(f"Missing options for {sample_id}")
    result = {str(key).strip().upper(): clean_text(value) for key, value in options.items()}
    if not all(result) or not all(result.values()):
        raise ValueError(f"Invalid options for {sample_id}")
    return dict(sorted(result.items()))


def canonical_answers(row: dict[str, Any], options: dict[str, str], sample_id: str) -> list[str]:
    values = row.get("answers")
    if not isinstance(values, list):
        values = [row.get("answer")]
    answers = sorted({str(value or "").strip().upper() for value in values if str(value or "").strip()})
    if not answers or any(answer not in options for answer in answers):
        raise ValueError(f"Invalid answers for {sample_id}: {answers}")
    return answers


def stable_document_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    value = str(value or "").strip()
    if not value:
        raise ValueError("Candidate document has no stable ID")
    return value


def benchmark_paths(benchmark_root: Path, datasets: Iterable[str]) -> dict[str, Path]:
    paths = {dataset: benchmark_root / dataset / "test.jsonl" for dataset in datasets}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark files: " + ", ".join(missing))
    return paths


def expected_question_counts(datasets: Iterable[str]) -> dict[str, int]:
    return {dataset: EXPECTED_DATASET_QUESTIONS[dataset] for dataset in datasets}


def input_contract(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "preparation_version": PREPARATION_VERSION,
        "candidate_cache": path_identity(args.candidate_cache_path),
        "candidate_cache_manifest": path_identity(args.candidate_cache_manifest_path),
        "benchmarks": {dataset: path_identity(path) for dataset, path in paths.items()},
        "datasets": list(args.datasets),
        "top_k": args.top_k,
        "documents_per_block": args.documents_per_block,
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
    }


def output_files_match(manifest: dict[str, Any], output_root: Path, datasets: Iterable[str]) -> bool:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return False
    for dataset in datasets:
        expected = outputs.get(dataset)
        path = output_root / "candidates" / f"{dataset}.jsonl"
        if not isinstance(expected, dict) or not path.is_file():
            return False
        identity = path_identity(path)
        if identity["size"] != expected.get("size") or identity["mtime_ns"] != expected.get("mtime_ns"):
            return False
    return True


def validate_cache_manifest(path: Path, expected_questions: int, top_k: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "type": "rag2_mcq_eval_candidates",
        "rows": expected_questions,
        "candidate_layout": "source_balanced",
        "rerank_top_k": 128,
        "dense_query_mode": "rationale",
        "prompt_profile": "paper_compatible_three_anchor",
    }
    mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if int(value.get("rerank_top_k", 0)) < top_k:
        mismatches["rerank_top_k_minimum"] = {"expected": top_k, "actual": value.get("rerank_top_k")}
    if mismatches:
        raise ValueError(f"Candidate cache manifest is incompatible: {mismatches}")
    return value


def load_benchmark_index(
    paths: dict[str, Path], expected_counts: dict[str, int], overall: tqdm
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    total = sum(expected_counts.values())
    stage = tqdm(total=total, desc="Stage 1/2 benchmark gold/index", unit="question", position=1, dynamic_ncols=True)
    observed = Counter()
    for dataset, path in paths.items():
        for row_idx, row in enumerate(iter_jsonl(path)):
            sample_id = str(row.get("id") or f"{dataset}:test:{row_idx:06d}")
            row_dataset = str(row.get("dataset") or dataset).lower()
            if row_dataset != dataset:
                raise ValueError(f"Benchmark dataset mismatch at {path}:{row_idx + 1}: {row_dataset}")
            options = canonical_options(row, sample_id)
            answers = canonical_answers(row, options, sample_id)
            question = clean_text(row.get("question"))
            if not question:
                raise ValueError(f"Empty benchmark question for {sample_id}")
            key = (dataset, sample_id)
            if key in index:
                raise ValueError(f"Duplicate benchmark sample: {key}")
            index[key] = {
                "row_idx": row_idx,
                "sample_id": sample_id,
                "dataset": dataset,
                "split": str(row.get("split") or "test"),
                "question": question,
                "options": options,
                "answer": answers[0],
                "answers": answers,
            }
            observed[dataset] += 1
            stage.update(1)
            overall.update(1)
    stage.close()
    if dict(observed) != expected_counts:
        raise ValueError(f"Benchmark counts do not match expected external cohort: {dict(observed)} != {expected_counts}")
    return index


def select_top_documents(row: dict[str, Any], sample_id: str, top_k: int) -> list[dict[str, Any]]:
    raw_documents = row.get("reranked_documents")
    if not isinstance(raw_documents, list):
        raise ValueError(f"Missing reranked_documents for {sample_id}")
    ordered = sorted(
        (document for document in raw_documents if isinstance(document, dict)),
        key=lambda document: int(document.get("rerank_rank") or 10**9),
    )
    selected = ordered[:top_k]
    ranks = [int(document.get("rerank_rank") or 0) for document in selected]
    if ranks != list(range(1, top_k + 1)):
        raise ValueError(f"Non-contiguous Top-{top_k} ranks for {sample_id}: {ranks}")
    seen_stable_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for document in selected:
        stable_id = stable_document_id(document)
        if stable_id in seen_stable_ids:
            raise ValueError(f"Duplicate Top-{top_k} document for {sample_id}: {stable_id}")
        seen_stable_ids.add(stable_id)
        text = clean_text(document.get("text"))
        if not text:
            raise ValueError(f"Empty document text for {sample_id} rank={document.get('rerank_rank')}")
        result.append(
            {
                "rerank_rank": int(document["rerank_rank"]),
                "stable_id": stable_id,
                "source": clean_text(document.get("source")) or "unknown",
                "title": clean_text(document.get("title")),
                "text": text,
                "retrieval_score": document.get("retrieval_score"),
                "rerank_score": document.get("rerank_score"),
            }
        )
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.top_k <= 0 or args.documents_per_block <= 0 or args.top_k % args.documents_per_block:
        raise ValueError("--top-k must be positive and exactly divisible by --documents-per-block")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate --datasets entry")
    for path in (args.candidate_cache_path, args.candidate_cache_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_counts = expected_question_counts(args.datasets)
    expected_questions = sum(expected_counts.values())
    expected_pairs = expected_questions * args.top_k
    blocks_per_question = args.top_k // args.documents_per_block
    paths = benchmark_paths(args.benchmark_root, args.datasets)
    cache_manifest = validate_cache_manifest(args.candidate_cache_manifest_path, expected_questions, args.top_k)
    contract = input_contract(args, paths)
    input_fingerprint = fingerprint(contract)
    manifest_path = args.output_root / "prepare_manifest.json"
    if args.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("input_fingerprint") == input_fingerprint
            and output_files_match(existing, args.output_root, args.datasets)
        ):
            logging.info(
                "Prepared external semantic candidates already complete: questions=%d pairs=%d root=%s",
                expected_questions,
                expected_pairs,
                args.output_root,
            )
            return existing

    overall_total = expected_questions * 2
    overall = tqdm(total=overall_total, desc="ExternalSemanticPrepareOverall", unit="question", position=0, dynamic_ncols=True)
    benchmark_index = load_benchmark_index(paths, expected_counts, overall)

    candidates_dir = args.output_root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {dataset: candidates_dir / f".{dataset}.jsonl.tmp" for dataset in args.datasets}
    block_temporary_paths = {
        (dataset, block_index): candidates_dir / f".{dataset}.block_{block_index:02d}.jsonl.tmp"
        for dataset in args.datasets
        for block_index in range(blocks_per_question)
    }
    final_paths = {dataset: candidates_dir / f"{dataset}.jsonl" for dataset in args.datasets}
    handles = {key: path.open("w", encoding="utf-8") for key, path in block_temporary_paths.items()}
    observed = Counter()
    pair_counts = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_samples: set[tuple[str, str]] = set()
    stage = tqdm(
        total=expected_questions,
        desc=f"Stage 2/2 cache Top-{args.top_k} materialization",
        unit="question",
        position=1,
        dynamic_ncols=True,
    )
    try:
        for cache_row in iter_jsonl(args.candidate_cache_path):
            dataset = str(cache_row.get("dataset") or "").lower()
            if dataset not in expected_counts:
                raise ValueError(f"Unexpected cache dataset: {dataset}")
            sample_id = str(cache_row.get("sample_id") or "")
            key = (dataset, sample_id)
            if key in seen_samples:
                raise ValueError(f"Duplicate cache sample: {key}")
            benchmark = benchmark_index.get(key)
            if benchmark is None:
                raise ValueError(f"Cache sample is absent from benchmark: {key}")
            if int(cache_row.get("row_idx", -1)) != int(benchmark["row_idx"]):
                raise ValueError(
                    f"Cache row_idx mismatch for {key}: {cache_row.get('row_idx')} != {benchmark['row_idx']}"
                )
            documents = select_top_documents(cache_row, sample_id, args.top_k)
            for block_index in range(blocks_per_question):
                start = block_index * args.documents_per_block
                block = documents[start : start + args.documents_per_block]
                output_row = {
                    **benchmark,
                    "semantic_transport_block_index": block_index,
                    "semantic_transport_block_count": blocks_per_question,
                    "semantic_transport_rank_start": int(block[0]["rerank_rank"]),
                    "semantic_transport_rank_end": int(block[-1]["rerank_rank"]),
                    "candidate_documents": block,
                    "candidate_cache_key": cache_row.get("key"),
                    "candidate_cache_dense_query_mode": cache_row.get("dense_query_mode"),
                }
                handles[(dataset, block_index)].write(
                    json.dumps(output_row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            seen_samples.add(key)
            observed[dataset] += 1
            pair_counts[dataset] += len(documents)
            source_counts[dataset].update(str(document["source"]) for document in documents)
            stage.update(1)
            overall.update(1)
    finally:
        for handle in handles.values():
            handle.close()
        stage.close()
        overall.close()

    if dict(observed) != expected_counts:
        raise ValueError(f"Candidate cache counts are incomplete: {dict(observed)} != {expected_counts}")
    missing = sorted(set(benchmark_index) - seen_samples)
    if missing:
        raise ValueError(f"Candidate cache omitted {len(missing)} benchmark samples; first={missing[0]}")
    if sum(pair_counts.values()) != expected_pairs:
        raise ValueError(f"Pair total mismatch: {sum(pair_counts.values())} != {expected_pairs}")

    # Preserve the original annotation transport: one Top-8 block from each of
    # ten distinct questions per Codex request whenever a full batch exists.
    # Writing all block-0 rows before block-1 rows prevents the four blocks of
    # one question from sharing a model request and influencing one another.
    for dataset in args.datasets:
        with temporary_paths[dataset].open("wb") as output_handle:
            for block_index in range(blocks_per_question):
                block_path = block_temporary_paths[(dataset, block_index)]
                with block_path.open("rb") as input_handle:
                    shutil.copyfileobj(input_handle, output_handle)
                block_path.unlink()
        os.replace(temporary_paths[dataset], final_paths[dataset])
    outputs = {dataset: path_identity(path) for dataset, path in final_paths.items()}
    manifest = {
        "preparation_version": PREPARATION_VERSION,
        "created_at": utc_now(),
        "status": "complete",
        "input_fingerprint": input_fingerprint,
        "input_contract": contract,
        "candidate_cache_contract": cache_manifest,
        "transport_contract": {
            "semantic_unit": "one question-document pair",
            "top_k_per_question": args.top_k,
            "documents_per_transport_row": args.documents_per_block,
            "transport_rows_per_question": blocks_per_question,
            "questions_per_codex_call": 10,
            "maximum_pairs_per_codex_call": 10 * args.documents_per_block,
            "row_order": "block-major so a full request contains ten distinct questions whenever possible",
            "reason": "Preserve the original Top-8 / 80-pair Codex request size while covering rerank Top-32.",
        },
        "annotation_contract": {
            "annotation_version": ANNOTATION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "web_search_enabled": False,
            "max_doc_chars": 0,
            "workers": 8,
        },
        "totals": {
            "questions": expected_questions,
            "pairs": expected_pairs,
            "transport_rows": expected_questions * blocks_per_question,
            "planned_codex_batches": sum(
                (count * blocks_per_question + 9) // 10 for count in expected_counts.values()
            ),
        },
        "datasets": {
            dataset: {
                "questions": observed[dataset],
                "pairs": pair_counts[dataset],
                "transport_rows": observed[dataset] * blocks_per_question,
                "source_distribution_top32": dict(sorted(source_counts[dataset].items())),
            }
            for dataset in args.datasets
        },
        "outputs": outputs,
    }
    atomic_json(manifest_path, manifest)
    logging.info(
        "External semantic candidates ready: questions=%d pairs=%d transport_rows=%d batches=%d root=%s",
        expected_questions,
        expected_pairs,
        manifest["totals"]["transport_rows"],
        manifest["totals"]["planned_codex_batches"],
        args.output_root,
    )
    return manifest


def validate_label_manifest(path: Path, prepared: dict[str, Any], datasets: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "annotation_version": ANNOTATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "docs_per_question": int(prepared["transport_contract"]["documents_per_transport_row"]),
        "allow_fewer_documents": False,
        "questions_per_batch": 10,
        "max_doc_chars": 0,
        "codex_bin": "codex",
        "codex_model_request": "gpt-5.6-terra",
        "codex_reasoning_effort": "medium",
        "web_search_enabled": False,
        "worker_count": 8,
    }
    mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    expected_datasets = prepared["datasets"]
    manifest_datasets = value.get("datasets") if isinstance(value.get("datasets"), dict) else {}
    for dataset in datasets:
        wanted = expected_datasets[dataset]
        actual = manifest_datasets.get(dataset, {})
        for key, expected_key in (("questions", "transport_rows"), ("pairs", "pairs")):
            if actual.get(key) != wanted[expected_key]:
                mismatches[f"datasets.{dataset}.{key}"] = {
                    "expected": wanted[expected_key],
                    "actual": actual.get(key),
                }
    if mismatches:
        raise ValueError(f"Semantic label run is incompatible: {mismatches}")
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    prepared_path = args.prepared_root / "prepare_manifest.json"
    if not prepared_path.is_file():
        raise FileNotFoundError(prepared_path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("status") != "complete" or prepared.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError(f"Prepared candidate manifest is incomplete or incompatible: {prepared_path}")
    for dataset in args.datasets:
        if dataset not in prepared.get("datasets", {}):
            raise ValueError(f"Prepared candidates omit dataset={dataset}")
    label_manifest = validate_label_manifest(args.label_root / "manifest.json", prepared, args.datasets)
    expected_pairs = sum(int(prepared["datasets"][dataset]["pairs"]) for dataset in args.datasets)
    overall = tqdm(total=expected_pairs * 2, desc="ExternalSemanticVerifyOverall", unit="pair", position=0, dynamic_ncols=True)

    expected: dict[tuple[str, str, int], str] = {}
    expected_samples: dict[str, set[str]] = defaultdict(set)
    stage = tqdm(total=expected_pairs, desc="Stage 1/2 index prepared Top-32", unit="pair", position=1, dynamic_ncols=True)
    for dataset in args.datasets:
        candidate_path = args.prepared_root / "candidates" / f"{dataset}.jsonl"
        for row in iter_jsonl(candidate_path):
            sample_id = str(row.get("sample_id") or "")
            expected_samples[dataset].add(sample_id)
            documents = row.get("candidate_documents")
            if not isinstance(documents, list):
                raise ValueError(f"Invalid prepared documents for {sample_id}")
            for document in documents:
                rank = int(document.get("rerank_rank") or 0)
                key = (dataset, sample_id, rank)
                if key in expected:
                    raise ValueError(f"Duplicate prepared pair: {key}")
                expected[key] = stable_document_id(document)
                stage.update(1)
                overall.update(1)
    stage.close()
    if len(expected) != expected_pairs:
        raise ValueError(f"Prepared pair count mismatch: {len(expected)} != {expected_pairs}")

    seen: set[tuple[str, str, int]] = set()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    stage = tqdm(total=expected_pairs, desc="Stage 2/2 verify semantic labels", unit="pair", position=1, dynamic_ncols=True)
    for dataset in args.datasets:
        label_path = args.label_root / dataset / "codex_semantic_labels.jsonl"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        for row in iter_jsonl(label_path):
            row_dataset = str(row.get("dataset") or "").lower()
            sample_id = str(row.get("sample_id") or "")
            rank = int(row.get("doc_rank") or 0)
            key = (row_dataset, sample_id, rank)
            if row_dataset != dataset:
                raise ValueError(f"Label dataset mismatch: expected={dataset} actual={row_dataset}")
            if key in seen:
                raise ValueError(f"Duplicate semantic label: {key}")
            wanted_stable_id = expected.get(key)
            if wanted_stable_id is None:
                raise ValueError(f"Unexpected semantic label pair: {key}")
            if str(row.get("doc_stable_id") or "") != wanted_stable_id:
                raise ValueError(f"Semantic label document mismatch for {key}")
            wanted_pair_id = f"{sample_id}::{rank}::{wanted_stable_id}"
            if str(row.get("pair_id") or "") != wanted_pair_id:
                raise ValueError(f"Semantic label pair_id mismatch for {key}")
            label = str(row.get("semantic_label") or "")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid semantic label for {key}: {label}")
            seen.add(key)
            label_counts[dataset][label] += 1
            stage.update(1)
            overall.update(1)
    stage.close()
    overall.close()
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"Semantic labels omit {len(missing)} pairs; first={missing[0] if missing else None}")

    for dataset, sample_ids in expected_samples.items():
        ranks_by_sample: dict[str, set[int]] = defaultdict(set)
        for row_dataset, sample_id, rank in seen:
            if row_dataset == dataset:
                ranks_by_sample[sample_id].add(rank)
        for sample_id in sample_ids:
            if ranks_by_sample[sample_id] != set(range(1, 33)):
                raise ValueError(f"Incomplete Top-32 label ranks for {sample_id}: {sorted(ranks_by_sample[sample_id])}")

    report = {
        "verification_version": "rag2_external_oracle_top32_semantic_verification_v1",
        "created_at": utc_now(),
        "status": "complete",
        "prepared_manifest": str(prepared_path.resolve()),
        "label_manifest": str((args.label_root / "manifest.json").resolve()),
        "questions": sum(len(values) for values in expected_samples.values()),
        "pairs": len(seen),
        "annotation_version": label_manifest["annotation_version"],
        "prompt_version": label_manifest["prompt_version"],
        "model": label_manifest["codex_model_request"],
        "reasoning_effort": label_manifest["codex_reasoning_effort"],
        "datasets": {
            dataset: {
                "questions": len(expected_samples[dataset]),
                "pairs": sum(label_counts[dataset].values()),
                "label_distribution": dict(sorted(label_counts[dataset].items())),
            }
            for dataset in args.datasets
        },
    }
    output_path = args.output_path or args.label_root / "external_oracle_top32_verification_report.json"
    atomic_json(output_path, report)
    logging.info("External semantic labels verified: questions=%d pairs=%d report=%s", report["questions"], report["pairs"], output_path)
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.command == "prepare":
        prepare(args)
    elif args.command == "verify":
        verify(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

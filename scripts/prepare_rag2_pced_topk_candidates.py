#!/usr/bin/env python3
"""Materialize the exact dynamic Top-k documents used for semantic labeling.

This script never retrieves, reranks, or reconstructs candidates.  It reads
``selected_document_ids_by_top_k`` from the immutable three-anchor semantic
candidate union and copies those exact document IDs in their stored order.
The exported semantic-label file is audited against the same union so a PCED
run cannot silently use a different question-document cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)
DEFAULT_UNION_ROOT = (
    DATA_ROOT
    / "external_test_dynamic_topk_rag2_oracle_v1/candidates_topk_union"
)
DEFAULT_SEMANTIC_LABELS = (
    DATA_ROOT
    / "external_test_dynamic_topk_semantic_oracle_v1/dynamic_semantic_oracle_labels.jsonl"
)
DEFAULT_SEMANTIC_MANIFEST = DEFAULT_SEMANTIC_LABELS.with_name(
    "dynamic_semantic_oracle_labels_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2"
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
SOURCES = ("pubmed", "pmc", "cpg", "textbooks")
MATERIALIZATION_VERSION = "rag2_pced_exact_semantic_labeled_dynamic_topk_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-union-root", type=Path, default=DEFAULT_UNION_ROOT)
    parser.add_argument("--semantic-labels", type=Path, default=DEFAULT_SEMANTIC_LABELS)
    parser.add_argument("--semantic-label-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--expected-rows", type=int, default=6545)
    parser.add_argument("--expected-pairs", type=int, default=211875)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object: {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_id(document: dict[str, Any]) -> str:
    value = (
        document.get("stable_id")
        or document.get("corpus_id")
        or document.get("chunk_id")
        or document.get("db_id")
    )
    if not value:
        raise RuntimeError("Candidate document has no stable ID")
    return str(value)


def pair_identity(sample_id: str, document_id: str) -> str:
    return f"{sample_id}\x1f{document_id}"


def selected_from_union(row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    selected_by_k = row.get("selected_document_ids_by_top_k")
    documents = row.get("candidate_documents")
    if not isinstance(selected_by_k, dict) or not isinstance(documents, list):
        raise RuntimeError(f"Invalid candidate-union row: {row.get('key')}")
    selected_ids = selected_by_k.get(str(top_k))
    if not isinstance(selected_ids, list) or len(selected_ids) != top_k:
        raise RuntimeError(
            f"Missing exact stored Top-{top_k} selection: {row.get('key')}"
        )
    if len(set(map(str, selected_ids))) != top_k:
        raise RuntimeError(f"Duplicate stored Top-{top_k} document ID: {row.get('key')}")
    by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise RuntimeError(f"Non-object candidate document: {row.get('key')}")
        identifier = stable_id(document)
        if identifier in by_id:
            raise RuntimeError(f"Duplicate union document ID: {row.get('key')} {identifier}")
        by_id[identifier] = document

    result: list[dict[str, Any]] = []
    for rank, raw_identifier in enumerate(selected_ids, 1):
        identifier = str(raw_identifier)
        original = by_id.get(identifier)
        if original is None:
            raise RuntimeError(
                f"Stored Top-{top_k} ID is absent from candidate union: "
                f"{row.get('key')} {identifier}"
            )
        metadata = original.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        memberships = {int(value) for value in metadata.get("oracle_dynamic_top_k_membership", [])}
        ranks = metadata.get("oracle_dynamic_rerank_rank_by_top_k")
        ranks = ranks if isinstance(ranks, dict) else {}
        if top_k not in memberships or int(ranks.get(str(top_k), -1)) != rank:
            raise RuntimeError(
                f"Stored membership/rank disagreement: {row.get('key')} "
                f"k={top_k} rank={rank} document={identifier}"
            )
        item = dict(original)
        item["semantic_union_rank"] = int(original.get("oracle_union_rank", -1))
        item["rerank_rank"] = rank
        result.append(item)
    return result


def validate_source_contract(manifest: dict[str, Any], args: argparse.Namespace, values: list[int]) -> None:
    expected = {
        "type": "rag2_paper_balanced_dynamic_oracle_candidate_union",
        "questions": args.expected_rows,
        "pairs": args.expected_pairs,
        "dynamic_top_k_values": values,
        "sources": list(SOURCES),
        "master_per_source_top_k": 32,
    }
    mismatch = {
        key: {"expected": wanted, "actual": manifest.get(key)}
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    }
    if mismatch:
        raise RuntimeError(f"Semantic candidate-union contract mismatch: {mismatch}")


def validate_semantic_manifest(
    manifest: dict[str, Any], args: argparse.Namespace, union_manifest_path: Path, values: list[int]
) -> None:
    contract = manifest.get("input_contract")
    contract = contract if isinstance(contract, dict) else {}
    recorded = contract.get("candidate_union_manifest")
    recorded = recorded if isinstance(recorded, dict) else {}
    expected = {
        "status": "complete",
        "questions": args.expected_rows,
        "pairs": args.expected_pairs,
        "dynamic_top_k_values": values,
    }
    mismatch = {
        key: {"expected": wanted, "actual": manifest.get(key)}
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    }
    if Path(str(recorded.get("path", ""))).resolve() != union_manifest_path.resolve():
        mismatch["candidate_union_manifest_path"] = {
            "expected": str(union_manifest_path.resolve()),
            "actual": recorded.get("path"),
        }
    stat = union_manifest_path.stat()
    if recorded.get("size") != stat.st_size or recorded.get("mtime_ns") != stat.st_mtime_ns:
        mismatch["candidate_union_manifest_identity"] = {
            "expected": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "actual": {"size": recorded.get("size"), "mtime_ns": recorded.get("mtime_ns")},
        }
    if mismatch:
        raise RuntimeError(f"Pseudo-semantic label contract mismatch: {mismatch}")


def complete_cache(
    path: Path,
    top_k: int,
    expected_rows: int,
    union_hash: str,
    semantic_hash: str,
) -> bool:
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        value.get("type") == MATERIALIZATION_VERSION
        and int(value.get("rows", -1)) == expected_rows
        and int(value.get("evaluation_top_k", -1)) == top_k
        and value.get("candidate_union_manifest_sha256") == union_hash
        and value.get("semantic_labels_sha256") == semantic_hash
        and value.get("output_sha256") == sha256_file(path)
    )


def main() -> None:
    args = parse_args()
    values = sorted(set(args.top_k_values))
    if values != [1, 2, 4, 8, 16, 32]:
        raise ValueError("This frozen comparison requires Top-k = 1,2,4,8,16,32")

    union_manifest_path = args.candidate_union_root / "manifest.json"
    required = [union_manifest_path, args.semantic_labels, args.semantic_label_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen semantic-candidate artifacts: " + ", ".join(missing))
    union_manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
    semantic_manifest = json.loads(args.semantic_label_manifest.read_text(encoding="utf-8"))
    validate_source_contract(union_manifest, args, values)
    validate_semantic_manifest(semantic_manifest, args, union_manifest_path, values)
    union_hash = sha256_file(union_manifest_path)
    semantic_hash = sha256_file(args.semantic_labels)

    pending = [
        top_k
        for top_k in values
        if not complete_cache(
            args.output_root / f"top{top_k}" / "candidates.jsonl",
            top_k,
            args.expected_rows,
            union_hash,
            semantic_hash,
        )
    ]
    print(
        f"[stage 1/2 | exact semantic-labeled candidate materialization] "
        f"questions={args.expected_rows} semantic_pairs={args.expected_pairs} "
        f"k={values} cached={len(values)-len(pending)} pending={len(pending)}",
        flush=True,
    )

    handles: dict[int, Any] = {}
    temporary_paths: dict[int, Path] = {}
    expected_semantic_pairs: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    selected_counts: Counter[int] = Counter()
    started = time.time()
    try:
        for top_k in pending:
            output = args.output_root / f"top{top_k}" / "candidates.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + f".partial.{os.getpid()}")
            temporary_paths[top_k] = temporary
            handles[top_k] = temporary.open("w", encoding="utf-8")

        progress = tqdm(
            total=args.expected_rows,
            desc="Stage 1/2 - copy stored Top-k ID/order (no retrieval/reranking)",
            unit="question",
            dynamic_ncols=True,
        )
        rows = 0
        for dataset in DATASETS:
            source_path = args.candidate_union_root / dataset / "test/candidates_topk_union.jsonl"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            for row in iter_jsonl(source_path):
                rows += 1
                dataset_counts[dataset] += 1
                if row.get("dataset") != dataset:
                    raise RuntimeError(f"Candidate dataset mismatch: {row.get('key')}")
                sample_id = str(row.get("sample_id") or "")
                documents = row.get("candidate_documents")
                if not sample_id or not isinstance(documents, list):
                    raise RuntimeError(f"Invalid candidate-union row: {row.get('key')}")
                for document in documents:
                    identity = pair_identity(sample_id, stable_id(document))
                    if identity in expected_semantic_pairs:
                        raise RuntimeError(f"Duplicate semantic pair in union: {identity}")
                    expected_semantic_pairs.add(identity)
                for top_k in values:
                    selected = selected_from_union(row, top_k)
                    selected_counts[top_k] += len(selected)
                    if top_k not in pending:
                        continue
                    value = {
                        "key": row.get("key"),
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "row_idx": row.get("row_idx"),
                        "prompt_profile": "paper_compatible_three_anchor",
                        "candidate_protocol": row.get("candidate_protocol"),
                        "retrieval_query_text": row.get("retrieval_query_text"),
                        "rerank_query_text": row.get("rerank_query_text"),
                        "reranked_documents": selected,
                    }
                    handles[top_k].write(
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                progress.update(1)
                elapsed = max(time.time() - started, 1e-6)
                rate = rows / elapsed
                progress.set_postfix_str(
                    f"{rate:.1f} q/s ETA={(args.expected_rows-rows)/max(rate,1e-9)/60:.1f}m",
                    refresh=False,
                )
        progress.close()
        if rows != args.expected_rows or len(expected_semantic_pairs) != args.expected_pairs:
            raise RuntimeError(
                f"Union count mismatch: questions={rows}/{args.expected_rows} "
                f"pairs={len(expected_semantic_pairs)}/{args.expected_pairs}"
            )
        for top_k in values:
            if selected_counts[top_k] != args.expected_rows * top_k:
                raise RuntimeError(
                    f"Top-{top_k} selected-pair count mismatch: {selected_counts[top_k]}"
                )

        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for top_k in pending:
            os.replace(
                temporary_paths[top_k],
                args.output_root / f"top{top_k}" / "candidates.jsonl",
            )
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    print(
        f"[stage 2/2 | exact pseudo-semantic pair audit] expected={args.expected_pairs}",
        flush=True,
    )
    observed: set[str] = set()
    semantic_membership_counts: Counter[int] = Counter()
    audit = tqdm(
        total=args.expected_pairs,
        desc="Stage 2/2 - verify pseudo-semantic labels match candidate IDs",
        unit="pair",
        dynamic_ncols=True,
    )
    for row in iter_jsonl(args.semantic_labels):
        identity = pair_identity(str(row.get("sample_id") or ""), str(row.get("doc_stable_id") or ""))
        if identity in observed:
            raise RuntimeError(f"Duplicate pseudo-semantic label pair: {identity}")
        if identity not in expected_semantic_pairs:
            raise RuntimeError(f"Pseudo-semantic label has non-candidate pair: {identity}")
        observed.add(identity)
        for top_k in row.get("dynamic_top_k_membership") or []:
            semantic_membership_counts[int(top_k)] += 1
        audit.update(1)
    audit.close()
    missing_pairs = expected_semantic_pairs - observed
    if missing_pairs or len(observed) != args.expected_pairs:
        first = next(iter(missing_pairs), None)
        raise RuntimeError(
            f"Pseudo-semantic pair audit failed: observed={len(observed)} "
            f"missing={len(missing_pairs)} first_missing={first}"
        )
    for top_k in values:
        expected = args.expected_rows * top_k
        if semantic_membership_counts[top_k] != expected:
            raise RuntimeError(
                f"Pseudo-semantic Top-{top_k} membership mismatch: "
                f"expected={expected} actual={semantic_membership_counts[top_k]}"
            )

    for top_k in pending:
        output = args.output_root / f"top{top_k}" / "candidates.jsonl"
        atomic_json(
            output.parent / "manifest.json",
            {
                "type": MATERIALIZATION_VERSION,
                "rows": args.expected_rows,
                "selected_pairs": args.expected_rows * top_k,
                "sources": list(SOURCES),
                "per_source_top_k": top_k,
                "candidate_pool_top_k": 4 * top_k,
                "candidate_layout": "source_balanced",
                "rerank_top_k": top_k,
                "evaluation_top_k": top_k,
                "prompt_profile": "paper_compatible_three_anchor",
                "candidate_protocol": "rag2_paper_balanced_dynamic_topk_union_v1",
                "selection": (
                    "exact selected_document_ids_by_top_k from the pseudo-semantic-label "
                    "candidate union; no retrieval, reranking, or prefix reconstruction"
                ),
                "candidate_union_root": str(args.candidate_union_root.resolve()),
                "candidate_union_manifest": str(union_manifest_path.resolve()),
                "candidate_union_manifest_sha256": union_hash,
                "semantic_labels": str(args.semantic_labels.resolve()),
                "semantic_labels_sha256": semantic_hash,
                "semantic_label_manifest": str(args.semantic_label_manifest.resolve()),
                "semantic_label_manifest_sha256": sha256_file(args.semantic_label_manifest),
                "output_path": str(output.resolve()),
                "output_sha256": sha256_file(output),
            },
        )
    elapsed = time.time() - started
    print(
        f"[stage 2/2 complete | elapsed={elapsed/60:.1f}m] "
        f"exact candidate-label identity verified: questions={args.expected_rows} "
        f"union_pairs={args.expected_pairs} memberships="
        f"{dict(sorted(semantic_membership_counts.items()))} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()

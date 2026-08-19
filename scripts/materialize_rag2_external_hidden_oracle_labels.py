#!/usr/bin/env python3
from __future__ import annotations

"""Compute gold-direction scores for frozen external-test h0/hD feature shards."""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extract_rag2_preanswer_hidden_pilot import (
    PROMPT_VERSION,
    FeatureExtractor,
    choice_index,
)
from medrag.io_utils import write_json, write_jsonl


FEATURE_CACHE_TYPE = "rag2_mcq_eval_preanswer_hidden_features"
FEATURE_CACHE_VERSION = "rag2_preanswer_hidden_states_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates-path", type=Path, required=True)
    p.add_argument("--feature-cache-dir", type=Path, required=True)
    p.add_argument("--model-name-or-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.4])
    p.add_argument("--question-batch-size", type=int, default=32)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument(
        "--h0-max-abs-tolerance",
        type=float,
        default=0.5,
        help="Catastrophic per-coordinate mismatch guard; BF16 batch-size changes can differ by one ULP.",
    )
    p.add_argument("--h0-max-relative-l2-tolerance", type=float, default=0.02)
    p.add_argument("--h0-min-cosine-similarity", type=float, default=0.999)
    p.add_argument(
        "--h0-validation-mode",
        choices=["warn", "strict"],
        default="warn",
        help=(
            "The feature-cache manifest, model identity, sample keys, and document keys are "
            "always validated strictly. Numerical h0 drift from BF16 batch-size changes is "
            "diagnostic-only by default; use strict only for deterministic FP32 audits."
        ),
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL: {path}:{number}") from error


def stable_id(doc: dict[str, Any]) -> str:
    return str(doc.get("stable_id") or doc.get("corpus_id") or doc.get("chunk_id") or doc.get("db_id") or f"{doc.get('source')}:{doc.get('local_id')}")


def threshold_name(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def validate_feature_cache_contract(
    args: argparse.Namespace,
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Hard-fail on semantic cache mismatch without relying on BF16 equality.

    h0/hD were created by the evaluation pipeline and stored in BF16.  Recomputing
    h0 here is necessary to obtain the gold gradient direction c, but eager-attention
    batches may be adaptively subdivided after an OOM.  BF16 hidden coordinates can
    then differ by one ULP even though the prompt and model are identical.  The
    durable compatibility contract is therefore the cache manifest plus exact
    sample/document identities, not coordinate-wise equality across batch shapes.
    """

    manifest_path = args.feature_cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature-cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = manifest.get("settings") or {}
    expected_pairs = sum(len(row.get("candidate_documents") or []) for row in candidates)
    expected = {
        "type": FEATURE_CACHE_TYPE,
        "version": FEATURE_CACHE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_path": str(args.model_name_or_path.resolve()),
        "hidden_layer": int(args.layer),
        "hidden_max_input_tokens": int(args.max_input_tokens),
        "hidden_dtype": args.dtype,
        "hidden_attn_implementation": args.attn_implementation,
        "questions": len(candidates),
        "documents": expected_pairs,
    }
    actual = {
        "type": manifest.get("type"),
        "version": manifest.get("version"),
        "prompt_version": settings.get("prompt_version"),
        "model_path": str(Path(settings.get("state_model", {}).get("path", "")).resolve()),
        "hidden_layer": settings.get("hidden_layer"),
        "hidden_max_input_tokens": settings.get("hidden_max_input_tokens"),
        "hidden_dtype": settings.get("hidden_dtype"),
        "hidden_attn_implementation": settings.get("hidden_attn_implementation"),
        "questions": manifest.get("questions"),
        "documents": manifest.get("documents"),
    }
    mismatches = [
        f"{key}: cache={actual[key]!r} requested={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]

    # Verify that the model files have not changed in place since h0/hD were cached.
    model_identity = settings.get("state_model") or {}
    model_path = args.model_name_or_path.resolve()
    for entry in [*(model_identity.get("files") or []), *(model_identity.get("weight_shards") or [])]:
        path = model_path / str(entry.get("name") or "")
        if not path.is_file():
            mismatches.append(f"model_file_missing: {path}")
            continue
        stat = path.stat()
        if int(entry.get("size", -1)) != stat.st_size:
            mismatches.append(
                f"model_file_size: {path.name} cache={entry.get('size')} current={stat.st_size}"
            )
        if int(entry.get("mtime_ns", -1)) != stat.st_mtime_ns:
            mismatches.append(
                f"model_file_mtime_ns: {path.name} cache={entry.get('mtime_ns')} "
                f"current={stat.st_mtime_ns}"
            )
    if mismatches:
        raise RuntimeError(
            "Feature cache is incompatible with this hidden-label invocation:\n- "
            + "\n- ".join(mismatches)
        )
    return manifest


def h0_exceeds_numerical_tolerance(
    *,
    max_abs: float,
    max_relative_l2: float,
    min_cosine: float,
    args: argparse.Namespace,
) -> bool:
    return (
        max_abs > args.h0_max_abs_tolerance
        or max_relative_l2 > args.h0_max_relative_l2_tolerance
        or min_cosine < args.h0_min_cosine_similarity
    )


def adaptive_question_feature_batches(
    extractor: FeatureExtractor,
    batch: Sequence[dict[str, Any]],
    *,
    absolute_start: int,
) -> Iterator[tuple[int, Sequence[dict[str, Any]], list[str], Any]]:
    """Retry only an OOM batch at half size, then return to the caller's normal size.

    Eager attention has quadratic peak memory, so one unusually long MCQ can
    make a batch of 32 much larger than adjacent batches.  Successful shards
    keep the requested throughput; only the offending batch is subdivided.
    """

    sequences, _ = extractor.encode_questions(batch, [None] * len(batch))
    gold = [
        str(row.get("answer") or (row.get("answers") or [""])[0]).upper()
        for row in batch
    ]
    if any(value not in {"A", "B", "C", "D"} for value in gold):
        raise ValueError(f"Invalid gold answer at question offset {absolute_start}: {gold}")
    oom = False
    try:
        features = extractor.no_document_features(
            sequences,
            [choice_index(value) for value in gold],
        )
    except torch.OutOfMemoryError:
        oom = True
    if oom:
        # Perform cleanup after leaving the except block so the traceback no
        # longer retains tensors from the failed attention forward pass.
        maximum_tokens = max(len(value) for value in sequences)
        del sequences
        gc.collect()
        torch.cuda.empty_cache()
        if len(batch) <= 1:
            raise RuntimeError(
                "CUDA OOM even for one no-document question: "
                f"offset={absolute_start} max_tokens={maximum_tokens}"
            )
        midpoint = len(batch) // 2
        logging.warning(
            "Question-feature OOM at offset=%s batch=%s max_tokens=%s; retrying this batch as %s + %s",
            absolute_start,
            len(batch),
            maximum_tokens,
            midpoint,
            len(batch) - midpoint,
        )
        yield from adaptive_question_feature_batches(
            extractor,
            batch[:midpoint],
            absolute_start=absolute_start,
        )
        yield from adaptive_question_feature_batches(
            extractor,
            batch[midpoint:],
            absolute_start=absolute_start + midpoint,
        )
        return
    yield absolute_start, batch, gold, features


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    candidates = list(rows(args.candidates_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_output = args.output_dir / "shards"
    shard_output.mkdir(parents=True, exist_ok=True)
    metadata_paths = sorted((args.feature_cache_dir / "shards").glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No feature shard metadata: {args.feature_cache_dir}")
    cache_manifest = validate_feature_cache_contract(args, candidates)
    logging.info(
        "Feature-cache contract valid: questions=%s pairs=%s shards=%s prompt=%s layer=%s",
        cache_manifest["questions"],
        cache_manifest["documents"],
        cache_manifest["shards"],
        cache_manifest["settings"]["prompt_version"],
        cache_manifest["settings"]["hidden_layer"],
    )

    extractor_args = SimpleNamespace(
        device=args.device, dtype=args.dtype, model_name_or_path=args.model_name_or_path,
        trust_remote_code=False, attn_implementation=args.attn_implementation,
        layers=[str(args.layer)], max_input_tokens=args.max_input_tokens,
    )
    extractor = FeatureExtractor(extractor_args)
    processed = 0
    max_h0_difference = 0.0
    max_h0_relative_l2 = 0.0
    min_h0_cosine = 1.0
    h0_numerical_warning_batches = 0
    try:
        for shard_number, metadata_path in enumerate(metadata_paths, 1):
            output_path = shard_output / f"{metadata_path.stem}.jsonl"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if args.resume and output_path.is_file():
                completed_rows = sum(1 for _ in rows(output_path))
                if completed_rows == int(metadata["documents"]):
                    processed += completed_rows
                    continue
                logging.warning(
                    "Ignoring incomplete completed-shard file %s: rows=%s expected=%s",
                    output_path,
                    completed_rows,
                    metadata["documents"],
                )
            indices = [int(value) for value in metadata["sample_indices"]]
            shard_rows = [candidates[index] for index in indices]
            actual_keys = [str(row["key"]) for row in shard_rows]
            if actual_keys != list(metadata["sample_keys"]):
                raise RuntimeError(f"Candidate/feature sample order mismatch: {metadata_path}")
            tensor_path = Path(metadata.get("tensor_path") or metadata_path.with_suffix(".safetensors"))
            if not tensor_path.is_absolute():
                tensor_path = metadata_path.parent / tensor_path.name
            tensors = load_file(str(tensor_path), device="cpu")
            cached_h0 = tensors["h0"].float()
            cached_hD = tensors["hD"].float()
            offsets = tensors["document_offsets"].to(torch.int64).tolist()
            output_rows: list[dict[str, Any]] = []
            for start in range(0, len(shard_rows), args.question_batch_size):
                requested_batch = shard_rows[start : start + args.question_batch_size]
                feature_batches = adaptive_question_feature_batches(
                    extractor,
                    requested_batch,
                    absolute_start=start,
                )
                for batch_start, batch, gold, question_features in feature_batches:
                    recomputed_h0 = question_features.h0[:, 0, :]
                    comparison = cached_h0[batch_start : batch_start + len(batch)]
                    residual = recomputed_h0 - comparison
                    difference = float(torch.max(torch.abs(residual)).item())
                    relative_l2 = torch.linalg.vector_norm(residual, dim=-1) / torch.linalg.vector_norm(
                        comparison, dim=-1
                    ).clamp_min(1e-12)
                    cosine = torch.nn.functional.cosine_similarity(
                        recomputed_h0,
                        comparison,
                        dim=-1,
                        eps=1e-12,
                    )
                    batch_max_relative_l2 = float(relative_l2.max().item())
                    batch_min_cosine = float(cosine.min().item())
                    max_h0_difference = max(max_h0_difference, difference)
                    max_h0_relative_l2 = max(max_h0_relative_l2, batch_max_relative_l2)
                    min_h0_cosine = min(min_h0_cosine, batch_min_cosine)
                    if h0_exceeds_numerical_tolerance(
                        max_abs=difference,
                        max_relative_l2=batch_max_relative_l2,
                        min_cosine=batch_min_cosine,
                        args=args,
                    ):
                        message = (
                            f"Cached/recomputed BF16 h0 drift in {metadata_path.name}: "
                            f"max_abs={difference:.6f} (diagnostic={args.h0_max_abs_tolerance}), "
                            f"max_relative_l2={batch_max_relative_l2:.6f} "
                            f"(diagnostic={args.h0_max_relative_l2_tolerance}), "
                            f"min_cosine={batch_min_cosine:.8f} "
                            f"(diagnostic={args.h0_min_cosine_similarity})"
                        )
                        if args.h0_validation_mode == "strict":
                            raise RuntimeError(message)
                        h0_numerical_warning_batches += 1
                        if h0_numerical_warning_batches == 1:
                            logging.warning(
                                "%s; continuing because the cache manifest, model files, "
                                "sample keys, and document keys match exactly. Further BF16 "
                                "drift messages are summarized in the final manifest.",
                                message,
                            )
                        else:
                            logging.debug(message)
                    c_unit = question_features.c_unit[:, 0, :]
                    for local, row in enumerate(batch):
                        position = batch_start + local
                        documents = list(row.get("candidate_documents") or [])
                        begin, end = offsets[position], offsets[position + 1]
                        if end - begin != len(documents):
                            raise RuntimeError(f"Document offset mismatch for {row['key']}")
                        document_keys = metadata["document_keys"][position]
                        for doc_index, (doc, hD) in enumerate(zip(documents, cached_hD[begin:end])):
                            expected_key = [str(doc.get("db_id") or ""), int(doc.get("local_id", -1))]
                            if list(document_keys[doc_index]) != expected_key:
                                raise RuntimeError(f"Document identity mismatch for {row['key']} rank={doc_index + 1}")
                            score = float(torch.dot(hD - comparison[local], c_unit[local]).item())
                            labels = {
                                threshold_name(value): ("Helpful" if score > value else "Not Helpful")
                                for value in args.thresholds
                            }
                            output_rows.append({
                                "schema_version": 1,
                                "policy": "preanswer_gold_direction_external_test",
                                "dataset": row["dataset"], "sample_key": row["key"],
                                "sample_id": row["sample_id"], "row_idx": row["row_idx"],
                                "pair_id": f"{row['sample_id']}::{doc_index + 1}::{stable_id(doc)}",
                                "doc_rank": doc_index + 1, "doc_stable_id": stable_id(doc),
                                "db_id": doc.get("db_id"), "local_id": doc.get("local_id"),
                                "source": doc.get("source"), "projection_score": score,
                                "labels": labels, "gold_answer": gold[local],
                                "layer": args.layer,
                            })
                    del question_features, recomputed_h0, comparison, c_unit
            temporary = output_path.with_suffix(".jsonl.partial")
            write_jsonl(temporary, output_rows)
            os.replace(temporary, output_path)
            processed += len(output_rows)
            logging.info("Hidden gold direction shard %s/%s: %s pairs (total=%s)", shard_number, len(metadata_paths), len(output_rows), processed)
            del tensors, cached_h0, cached_hD
    finally:
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_path = args.output_dir / "hidden_oracle_labels.jsonl"
    temporary = combined_path.with_suffix(".jsonl.partial")
    with temporary.open("w", encoding="utf-8") as destination:
        total = 0
        for path in sorted(shard_output.glob("*.jsonl")):
            for row in rows(path):
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    os.replace(temporary, combined_path)
    expected = sum(len(row.get("candidate_documents") or []) for row in candidates)
    if total != expected:
        raise RuntimeError(f"Hidden output incomplete: {total}/{expected}")
    write_json(args.output_dir / "manifest.json", {
        "type": "rag2_external_hidden_oracle_labels", "questions": len(candidates), "pairs": total,
        "thresholds": args.thresholds, "layer": args.layer, "feature_cache_dir": str(args.feature_cache_dir.resolve()),
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "recomputed_h0_validation_new_shards": {
            "mode": args.h0_validation_mode,
            "numerical_warning_batches": h0_numerical_warning_batches,
            "max_abs_difference": max_h0_difference,
            "max_relative_l2": max_h0_relative_l2,
            "min_cosine_similarity": min_h0_cosine,
            "limits": {
                "max_abs_difference": args.h0_max_abs_tolerance,
                "max_relative_l2": args.h0_max_relative_l2_tolerance,
                "min_cosine_similarity": args.h0_min_cosine_similarity,
            },
        },
    })


if __name__ == "__main__":
    main()

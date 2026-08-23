#!/usr/bin/env python3
"""Replay anchored document traces and save selected-layer anchor features.

The generated rationale is never regenerated or rewritten.  Each stored
``question + options + one document + generated rationale`` prefix is replayed
through the same causal LM, and hidden states are captured at the same three
anchors used by the no-RAG feature cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_rag2_anchored_no_rag_features import (  # noqa: E402
    SelectedLayerExtractor,
    adaptive_extract,
    atomic_save_safetensors,
    atomic_write_json,
    atomic_write_jsonl,
    chunks,
    read_jsonl,
)
from generate_rag2_anchored_document_traces import (  # noqa: E402
    RUN_VERSION as GENERATION_RUN_VERSION,
)
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    ANCHOR_NAMES,
    CHOICES,
    PROMPT_VERSION,
    TRACE_VERSION,
    encode_to_pre_choice,
    normalized_mcq_row,
)

RUN_VERSION = "rag2_anchored_independent_document_selected_layer_features_v1"
NO_RAG_FEATURE_RUN_VERSION = "rag2_anchored_no_rag_selected_layer_features_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_NO_RAG_FEATURE_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "train_no_rag_anchored_features_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-rag-feature-root", type=Path, default=DEFAULT_NO_RAG_FEATURE_ROOT)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", choices=["medmcqa", "medqa"], default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 12, 20, 28, 31])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--minimum-free-space-gib", type=float, default=20.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def feature_paths(root: Path, dataset: str, split: str, shard_name: str) -> dict[str, Path]:
    base = root / "with_document_features" / dataset / split / "shards" / shard_name
    return {
        "root": base,
        "meta": base / "pairs.jsonl",
        "tensor": base / "features.safetensors",
        "complete": base / "COMPLETE.json",
    }


def complete_valid(paths: dict[str, Path], expected: int, layers: Sequence[int], trace_size: int) -> bool:
    if any(not paths[key].is_file() for key in ("meta", "tensor", "complete")):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and int(marker.get("pair_count", -1)) == expected
        and marker.get("layers") == list(layers)
        and int(marker.get("source_trace_size_bytes", -1)) == trace_size
    )


class DocumentSelectedLayerExtractor(SelectedLayerExtractor):
    def encode(self, row: dict[str, Any]) -> Any:
        normalized = normalized_mcq_row(
            {
                "question": row["question"],
                "options": row["options"],
                "answer": row["gold_answer"],
            }
        )
        encoding = encode_to_pre_choice(
            self.tokenizer,
            normalized,
            str(row.get("document_text_used") or ""),
            str(row.get("rationale") or ""),
        )
        if len(encoding.input_ids) > self.args.max_input_tokens:
            raise ValueError(
                f"Input exceeds max tokens for {row['pair_id']}: "
                f"{len(encoding.input_ids)} > {self.args.max_input_tokens}"
            )
        if encoding.prompt_sha256 != row.get("user_prompt_sha256"):
            raise RuntimeError(f"Prompt hash mismatch for {row['pair_id']}")
        return encoding


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def process_shard(
    args: argparse.Namespace,
    extractor: DocumentSelectedLayerExtractor,
    trace_path: Path,
    paths: dict[str, Path],
) -> int:
    rows = read_jsonl(trace_path)
    hidden_rows: list[torch.Tensor] = []
    logits_rows: list[torch.Tensor] = []
    probability_rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for batch in chunks(rows, args.batch_size):
        for features, encodings, actual_rows in adaptive_extract(extractor, batch):
            for index, (row, encoding) in enumerate(zip(actual_rows, encodings)):
                probabilities = features["choice_probabilities"][index]
                hf_answer = CHOICES[int(torch.argmax(probabilities).item())]
                document = row.get("document") or {}
                tensor_row = len(hidden_rows)
                hidden_rows.append(features["anchor_hidden"][index].half())
                logits_rows.append(features["choice_logits"][index])
                probability_rows.append(probabilities)
                metadata.append(
                    {
                        "run_version": RUN_VERSION,
                        "trace_version": row["trace_version"],
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "sample_id": row["sample_id"],
                        "pair_id": row["pair_id"],
                        "row_idx": int(row["row_idx"]),
                        "doc_rank": int(row["doc_rank"]),
                        "document_source": document.get("source"),
                        "document_stable_id": document.get("stable_id"),
                        "document_text_sha256": sha256_text(str(row.get("document_text_used") or "")),
                        "gold_answer": row["gold_answer"],
                        "generated_answer": row["answer"],
                        "generated_answer_correct": bool(row["answer_correct"]),
                        "hf_replay_answer": hf_answer,
                        "hf_replay_correct": hf_answer == row["gold_answer"],
                        "generated_hf_answer_match": hf_answer == row["answer"],
                        "rationale_ppl": (row.get("rationale_stats") or {}).get("ppl"),
                        "quality_flags": row.get("quality_flags") or [],
                        "input_token_count": len(encoding.input_ids),
                        "anchor_indices": encoding.anchor_indices,
                        "anchor_token_ids": encoding.anchor_token_ids,
                        "anchor_token_text": encoding.anchor_token_text,
                        "tensor_row": tensor_row,
                    }
                )
    if not hidden_rows:
        raise RuntimeError(f"Trace shard contains no pairs: {trace_path}")
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_save_safetensors(
        paths["tensor"],
        {
            "anchor_hidden": torch.stack(hidden_rows),
            "choice_logits": torch.stack(logits_rows).float(),
            "choice_probabilities": torch.stack(probability_rows).float(),
        },
        {
            "run_version": RUN_VERSION,
            "trace_version": TRACE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "layer_order": json.dumps(extractor.layer_names),
            "anchor_order": json.dumps(list(ANCHOR_NAMES)),
            "choice_order": json.dumps(list(CHOICES)),
            "anchor_hidden_layout": "[pair, selected_decoder_block, anchor, hidden]",
        },
    )
    atomic_write_jsonl(paths["meta"], metadata)
    atomic_write_json(
        paths["complete"],
        {
            "run_version": RUN_VERSION,
            "completed_at": utc_now(),
            "pair_count": len(metadata),
            "layers": list(args.layers),
            "source_trace_size_bytes": trace_path.stat().st_size,
            "feature_size_bytes": paths["tensor"].stat().st_size,
        },
    )
    return len(metadata)


def validate_no_rag_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = args.no_rag_feature_root / "feature_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing no-RAG feature manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("run_version") != NO_RAG_FEATURE_RUN_VERSION:
        raise RuntimeError(f"No-RAG feature version mismatch: {manifest.get('run_version')}")
    if manifest.get("trace_version") != TRACE_VERSION or manifest.get("prompt_version") != PROMPT_VERSION:
        raise RuntimeError("No-RAG and with-document anchor prompt contracts differ")
    no_rag_model = Path(str(manifest.get("model_name_or_path") or "")).resolve()
    if no_rag_model != args.model_name_or_path.resolve():
        raise RuntimeError(
            "No-RAG and with-document feature models differ: "
            f"no_rag={no_rag_model} with_document={args.model_name_or_path.resolve()}"
        )
    available_layers = [int(value) for value in manifest.get("layers") or []]
    missing = [layer for layer in args.layers if layer not in available_layers]
    if missing:
        raise RuntimeError(
            f"Requested with-document layers are absent from no-RAG features: {missing}; "
            f"available={available_layers}"
        )
    return manifest


def estimate_storage(args: argparse.Namespace, total_pairs: int, hidden_size: int) -> dict[str, float]:
    tensor_bytes = total_pairs * len(args.layers) * len(ANCHOR_NAMES) * hidden_size * 2
    tensor_bytes += total_pairs * 4 * 4 * 2  # logits and probabilities, float32
    free_bytes = shutil.disk_usage(args.output_root).free
    reserve_bytes = int(args.minimum_free_space_gib * (1024**3))
    return {
        "estimated_tensor_gib": tensor_bytes / (1024**3),
        "free_gib": free_bytes / (1024**3),
        "reserve_gib": reserve_bytes / (1024**3),
        "fits": float(free_bytes >= tensor_bytes + reserve_bytes),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.batch_size <= 0 or args.max_input_tokens <= 0:
        raise ValueError("Batch size and token limit must be positive")
    generation_manifest_path = args.trace_root / "generation_manifest.json"
    if not generation_manifest_path.is_file():
        raise FileNotFoundError(generation_manifest_path)
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if generation_manifest.get("run_version") != GENERATION_RUN_VERSION:
        raise RuntimeError(f"Generation version mismatch: {generation_manifest.get('run_version')}")
    if generation_manifest.get("trace_version") != TRACE_VERSION:
        raise RuntimeError("Generation trace contract does not match feature extraction")
    no_rag_manifest = validate_no_rag_contract(args)
    pair_counts = {
        dataset: int(generation_manifest["pairs_by_dataset"][dataset])
        for dataset in args.datasets
    }
    total_pairs = sum(pair_counts.values())
    hidden_size = int(no_rag_manifest["hidden_size"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    trace_shards: list[tuple[str, Path, int]] = []
    completed = 0
    observed_by_dataset = {dataset: 0 for dataset in args.datasets}
    for dataset in args.datasets:
        roots = sorted((args.trace_root / "trace_shards" / dataset / args.split).glob("shard_*"))
        for root in roots:
            marker_path = root / "COMPLETE.json"
            trace_path = root / "pairs.jsonl"
            if not marker_path.is_file() or not trace_path.is_file():
                raise RuntimeError(f"Incomplete generation shard: {root}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            expected = int(marker["pair_count"])
            if int(marker.get("pairs_size_bytes", -1)) != trace_path.stat().st_size:
                raise RuntimeError(f"Generation shard size mismatch: {root}")
            observed_by_dataset[dataset] += expected
            paths = feature_paths(args.output_root, dataset, args.split, root.name)
            trace_shards.append((dataset, root, expected))
            if args.resume and complete_valid(paths, expected, args.layers, trace_path.stat().st_size):
                completed += expected
        if observed_by_dataset[dataset] != pair_counts[dataset]:
            raise RuntimeError(
                f"Trace coverage mismatch for {dataset}: "
                f"{observed_by_dataset[dataset]} != {pair_counts[dataset]}"
            )
    total_storage = estimate_storage(args, total_pairs, hidden_size)
    remaining_storage = estimate_storage(args, total_pairs - completed, hidden_size)
    storage = {
        "total_estimated_tensor_gib": total_storage["estimated_tensor_gib"],
        "remaining_estimated_tensor_gib": remaining_storage["estimated_tensor_gib"],
        "completed_pairs": completed,
        "remaining_pairs": total_pairs - completed,
        "free_gib": remaining_storage["free_gib"],
        "reserve_gib": remaining_storage["reserve_gib"],
        "remaining_fits": bool(remaining_storage["fits"]),
    }
    logging.info(
        "Document feature plan: pairs=%s completed=%d remaining=%d layers=%s anchors=%s "
        "total_tensor=%.2f GiB remaining_tensor=%.2f GiB free=%.2f GiB reserve=%.2f GiB",
        pair_counts,
        completed,
        total_pairs - completed,
        args.layers,
        list(ANCHOR_NAMES),
        storage["total_estimated_tensor_gib"],
        storage["remaining_estimated_tensor_gib"],
        storage["free_gib"],
        storage["reserve_gib"],
    )
    if not storage["remaining_fits"]:
        raise RuntimeError(
            "Insufficient disk space for remaining hidden features: "
            f"estimate={storage['remaining_estimated_tensor_gib']:.2f} GiB "
            f"free={storage['free_gib']:.2f} GiB reserve={storage['reserve_gib']:.2f} GiB"
        )
    if args.dry_run:
        logging.info("Dry run complete: generated trace contract and disk budget are valid")
        return

    final_manifest_path = args.output_root / "document_feature_manifest.json"
    if completed == total_pairs and final_manifest_path.is_file():
        final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if (
            final_manifest.get("run_version") == RUN_VERSION
            and int(final_manifest.get("total_pairs", -1)) == total_pairs
            and final_manifest.get("layers") == list(args.layers)
        ):
            logging.info("All with-document feature shards are already complete; skipping model load.")
            return

    extractor = DocumentSelectedLayerExtractor(args)
    progress = PipelineProgress(
        overall_total=2 * total_pairs,
        overall_initial=total_pairs + completed,
        desc="AnchoredDocumentPipeline",
    )
    progress.set_stage(
        "2/2 exact replay and anchor feature extraction",
        total=total_pairs,
        initial=completed,
    )
    newly_written = 0
    try:
        for dataset, trace_root, expected in trace_shards:
            progress.set_detail(f"dataset={dataset} shard={trace_root.name}")
            trace_path = trace_root / "pairs.jsonl"
            paths = feature_paths(args.output_root, dataset, args.split, trace_root.name)
            if args.resume and complete_valid(paths, expected, args.layers, trace_path.stat().st_size):
                continue
            written = process_shard(args, extractor, trace_path, paths)
            if written != expected:
                raise RuntimeError(f"Feature count mismatch in {trace_root}: {written} != {expected}")
            newly_written += written
            progress.update(written)
        atomic_write_json(
            args.output_root / "document_feature_manifest.json",
            {
                "run_version": RUN_VERSION,
                "generation_run_version": GENERATION_RUN_VERSION,
                "trace_version": TRACE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "created_at": utc_now(),
                "trace_root": str(args.trace_root.resolve()),
                "no_rag_feature_root": str(args.no_rag_feature_root.resolve()),
                "model_name_or_path": str(args.model_name_or_path.resolve()),
                "datasets": pair_counts,
                "total_pairs": total_pairs,
                "layers": args.layers,
                "layer_order": extractor.layer_names,
                "anchor_order": list(ANCHOR_NAMES),
                "choice_token_ids": extractor.choice_token_ids,
                "hidden_size": extractor.hidden_size,
                "storage_dtype": "float16",
                "compute_dtype": args.dtype,
                "max_input_tokens": args.max_input_tokens,
                "newly_written": newly_written,
                "estimated_storage": storage,
                "feature_semantics": {
                    "pre_rationale": "with-document state at final token of Rationale: before free reasoning",
                    "post_rationale": "with-document state at final token of fixed end-of-reasoning marker",
                    "pre_choice": "with-document state at opening parenthesis immediately before A/B/C/D",
                    "document_delta": "join by sample_id with no-RAG feature and compute h_D - h_0 at matching layer/anchor",
                },
            },
        )
        logging.info("With-document anchor feature extraction complete: %s", args.output_root)
    finally:
        progress.close()
        extractor.close()


if __name__ == "__main__":
    main()

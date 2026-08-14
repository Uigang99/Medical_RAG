#!/usr/bin/env python3
"""Extract all-layer behavioral scalars and selected full pre-answer vectors.

This is the storage-efficient full-data companion to
``extract_rag2_preanswer_hidden_full.py``.  For every question--document pair it
computes, at each requested hidden-state probe,

    delta_h = hD - h0
    utility = delta_h dot c_unit
    cosine = utility / ||delta_h||

where ``c_unit`` is the unit negative gradient of the gold-choice NLL at the
no-document state.  Scalar metrics are retained for every probe while full
4096-dimensional vectors are retained only for explicitly selected probes.

For Llama-3-8B, ``--scalar-layers all`` means layer_1..layer_31 plus ``final``.
The input embedding state is deliberately omitted: the final-answer marker is
the same token with and without a document, so its contextual delta is zero at
the embedding input.  ``final`` is the normalized state after all 32 decoder
blocks and immediately before the LM head.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import AutoConfig

from extract_rag2_preanswer_hidden_full import (
    count_lines,
    iter_selected_rows,
    selected_question_count,
    stream_chunks,
)
from extract_rag2_preanswer_hidden_pilot import (
    CHOICES,
    FINAL_ANSWER_PREFILL,
    PROMPT_VERSION,
    FeatureExtractor,
    atomic_save_safetensors,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    chunks,
    choice_index,
    float_list,
    parse_layer_specs,
    predicted_choice,
    sha256_text,
    utc_now,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
FORMAT_VERSION = "rag2_preanswer_hidden_multilayer_scalar_selected_vector_v1"
RUN_VERSION = "rag2_preanswer_hidden_multilayer_full_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract behavioral scalar metrics at all requested layers and full vectors "
            "only at selected layers for the complete rerank Top-k dataset."
        )
    )
    parser.add_argument("--dataset", choices=["medmcqa", "medqa"], required=True)
    parser.add_argument("--candidates-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument(
        "--scalar-layers",
        nargs="+",
        default=["all"],
        help=(
            "Layers for projection/cosine/norm scalars. 'all' expands to every "
            "contextual intermediate state plus final."
        ),
    )
    parser.add_argument(
        "--vector-layers",
        nargs="+",
        default=["16", "24", "28", "final"],
        help=(
            "Subset of scalar layers for full h0, c_unit, c_norm, and hD vectors. "
            "Use 'none' for scalar-only storage."
        ),
    )
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=32)
    parser.add_argument("--questions-per-shard", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--start-question", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/layer/storage contracts and write no feature shards.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_layer_contract(args: argparse.Namespace) -> None:
    config = AutoConfig.from_pretrained(
        str(args.model_name_or_path), trust_remote_code=args.trust_remote_code
    )
    num_hidden_layers = int(config.num_hidden_layers)
    scalar_raw = [str(value).strip().lower() for value in args.scalar_layers]
    if "all" in scalar_raw:
        if scalar_raw != ["all"]:
            raise ValueError("--scalar-layers all cannot be combined with explicit layers")
        # The historical extractor labels hidden_states[i] as layer_i and uses
        # the final normalized output for the state after the last decoder block.
        scalar_raw = [str(index) for index in range(1, num_hidden_layers)] + ["final"]
    scalar_names, _ = parse_layer_specs(scalar_raw, num_hidden_layers)

    vector_raw = [str(value).strip().lower() for value in args.vector_layers]
    if vector_raw == ["none"]:
        vector_names: list[str] = []
    else:
        if "none" in vector_raw or "all" in vector_raw:
            raise ValueError("--vector-layers accepts explicit layers or the single value 'none'")
        vector_names, _ = parse_layer_specs(vector_raw, num_hidden_layers)
    missing = [name for name in vector_names if name not in scalar_names]
    if missing:
        raise ValueError(f"Vector layers must be included in scalar layers: {missing}")

    # FeatureExtractor consumes ``args.layers``.  It returns states in this
    # exact scalar order; selected vector indices are applied only at storage.
    args.layers = scalar_raw
    args.scalar_layer_names = scalar_names
    args.vector_layer_names = vector_names
    args.vector_layer_indices = [scalar_names.index(name) for name in vector_names]
    args.num_hidden_layers = num_hidden_layers


def shard_paths(output_dir: Path, shard_index: int) -> dict[str, Path]:
    root = output_dir / "shards" / f"shard_{shard_index:05d}"
    return {
        "root": root,
        "questions_meta": root / "questions.jsonl",
        "pairs_meta": root / "pairs.jsonl",
        "questions_tensor": root / "question_features.safetensors",
        "pairs_tensor": root / "pair_features.safetensors",
        "complete": root / "COMPLETE.json",
    }


def complete_shard_valid(
    paths: dict[str, Path],
    expected_questions: int,
    expected_pairs: int,
    scalar_layer_names: Sequence[str],
    vector_layer_names: Sequence[str],
) -> bool:
    if not paths["complete"].is_file():
        return False
    required = ("questions_meta", "pairs_meta", "questions_tensor", "pairs_tensor")
    if any(not paths[name].is_file() for name in required):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("format_version") == FORMAT_VERSION
        and marker.get("question_count") == expected_questions
        and marker.get("pair_count") == expected_pairs
        and marker.get("scalar_layer_order") == list(scalar_layer_names)
        and marker.get("vector_layer_order") == list(vector_layer_names)
    )


def select_layers(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        raise ValueError("select_layers requires at least one index")
    return tensor.index_select(0, torch.tensor(indices, dtype=torch.long))


def process_shard(
    args: argparse.Namespace,
    extractor: FeatureExtractor,
    shard_index: int,
    rows: Sequence[dict[str, Any]],
) -> None:
    paths = shard_paths(args.output_dir, shard_index)
    paths["root"].mkdir(parents=True, exist_ok=True)
    legacy_layer_index = extractor.layer_names.index("layer_28")
    legacy_layer_indices = [legacy_layer_index]

    question_h0_vector: list[torch.Tensor] = []
    question_c_unit_vector: list[torch.Tensor] = []
    question_c_norm_vector: list[torch.Tensor] = []
    # Exact legacy layer-28 tensors.  Keep these names and singleton layer
    # dimensions so existing readers can consume this run unchanged.
    question_h0_legacy: list[torch.Tensor] = []
    question_c_unit_legacy: list[torch.Tensor] = []
    question_c_norm_legacy: list[torch.Tensor] = []
    question_c_norm_all: list[torch.Tensor] = []
    question_logits: list[torch.Tensor] = []
    question_probs: list[torch.Tensor] = []
    question_metadata: list[dict[str, Any]] = []
    question_runtime: list[dict[str, Any]] = []

    for batch_rows in chunks(rows, args.question_batch_size):
        sequences, prompts = extractor.encode_questions(batch_rows, [None] * len(batch_rows))
        gold_indices = [choice_index(row["gold_answer"]) for row in batch_rows]
        features = extractor.no_document_features(sequences, gold_indices)
        for local_index, (row, sequence, prompt) in enumerate(zip(batch_rows, sequences, prompts)):
            h0_all = features.h0[local_index]
            c_unit_all = features.c_unit[local_index]
            c_norm_all = features.c_norm[local_index]
            logits = features.choice_logits[local_index]
            probabilities = features.choice_probs[local_index]
            prediction = predicted_choice(probabilities)
            shard_question_index = len(question_metadata)
            if args.vector_layer_indices:
                question_h0_vector.append(
                    select_layers(h0_all, args.vector_layer_indices).half()
                )
                question_c_unit_vector.append(
                    select_layers(c_unit_all, args.vector_layer_indices).half()
                )
                question_c_norm_vector.append(
                    select_layers(c_norm_all, args.vector_layer_indices).float()
                )
            question_h0_legacy.append(select_layers(h0_all, legacy_layer_indices).half())
            question_c_unit_legacy.append(
                select_layers(c_unit_all, legacy_layer_indices).half()
            )
            question_c_norm_legacy.append(
                select_layers(c_norm_all, legacy_layer_indices).float()
            )
            question_c_norm_all.append(c_norm_all.float())
            question_logits.append(logits)
            question_probs.append(probabilities)
            question_runtime.append(
                {
                    "h0_all": h0_all,
                    "c_unit_all": c_unit_all,
                    "gold_choice_logprob": features.gold_choice_logprob[local_index],
                    "prediction": prediction,
                    "shard_question_index": shard_question_index,
                }
            )
            question_metadata.append(
                {
                    "format_version": FORMAT_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "shard_index": shard_index,
                    "shard_question_index": shard_question_index,
                    "global_question_index": row["question_index"],
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "source_split": row["source_split"],
                    "source_row_index": row["source_row_index"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold_answer": row["gold_answer"],
                    "no_document_answer": prediction,
                    "no_document_correct": prediction == row["gold_answer"],
                    "no_document_visible_output": f"{FINAL_ANSWER_PREFILL} {prediction}",
                    "no_document_choice_logits": float_list(logits),
                    "no_document_choice_probabilities": float_list(probabilities),
                    "no_document_gold_choice_logprob": float(
                        features.gold_choice_logprob[local_index].item()
                    ),
                    "input_token_count": len(sequence),
                    "user_prompt_sha256": sha256_text(prompt),
                    "feature_file": "question_features.safetensors",
                    "feature_tensor_row": shard_question_index,
                }
            )

    pair_hD_vector: list[torch.Tensor] = []
    pair_hD_legacy: list[torch.Tensor] = []
    pair_logits: list[torch.Tensor] = []
    pair_probs: list[torch.Tensor] = []
    pair_utility: list[torch.Tensor] = []
    pair_delta_norm: list[torch.Tensor] = []
    pair_cosine: list[torch.Tensor] = []
    pair_gold_logprob_delta: list[torch.Tensor] = []
    pair_question_rows: list[int] = []
    pair_metadata: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for local_question_index, row in enumerate(rows):
        for document in row["documents"]:
            pending.append((local_question_index, row, document))

    for batch_items in chunks(pending, args.document_batch_size):
        batch_rows = [item[1] for item in batch_items]
        document_texts = [item[2]["text"] for item in batch_items]
        sequences, prompts = extractor.encode_questions(batch_rows, document_texts)
        features = extractor.document_features(sequences)
        for local_index, ((question_local, row, document), sequence, prompt) in enumerate(
            zip(batch_items, sequences, prompts)
        ):
            runtime = question_runtime[question_local]
            hD_all = features.hD[local_index]
            logits = features.choice_logits[local_index]
            probabilities = features.choice_probs[local_index]
            delta_all = hD_all - runtime["h0_all"]
            delta_norm = torch.linalg.vector_norm(delta_all, dim=-1)
            utility = torch.sum(delta_all * runtime["c_unit_all"], dim=-1)
            cosine = utility / delta_norm.clamp_min(1e-12)
            if args.vector_layer_indices:
                # Preserve the historical storage contract exactly for the
                # selected probes: store hD itself (not a rounded delta), so
                # layer_28 can be compared directly with the prior full run.
                pair_hD_vector.append(
                    select_layers(hD_all, args.vector_layer_indices).half()
                )
            pair_hD_legacy.append(select_layers(hD_all, legacy_layer_indices).half())
            gold_index = choice_index(row["gold_answer"])
            gold_logprob = F.log_softmax(logits, dim=-1)[gold_index]
            gold_delta = gold_logprob - runtime["gold_choice_logprob"]
            prediction = predicted_choice(probabilities)
            no_doc_prediction = runtime["prediction"]
            transition = (
                ("C" if no_doc_prediction == row["gold_answer"] else "W")
                + "->"
                + ("C" if prediction == row["gold_answer"] else "W")
            )
            pair_index = len(pair_metadata)
            pair_logits.append(logits)
            pair_probs.append(probabilities)
            pair_utility.append(utility)
            pair_delta_norm.append(delta_norm)
            pair_cosine.append(cosine)
            pair_gold_logprob_delta.append(gold_delta)
            pair_question_rows.append(runtime["shard_question_index"])
            pair_metadata.append(
                {
                    "format_version": FORMAT_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "shard_index": shard_index,
                    "shard_pair_index": pair_index,
                    "shard_question_index": runtime["shard_question_index"],
                    "global_question_index": row["question_index"],
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "pair_id": document["pair_id"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold_answer": row["gold_answer"],
                    "document": document,
                    "no_document_answer": no_doc_prediction,
                    "with_document_answer": prediction,
                    "with_document_correct": prediction == row["gold_answer"],
                    "answer_transition": transition,
                    "with_document_visible_output": f"{FINAL_ANSWER_PREFILL} {prediction}",
                    "with_document_choice_logits": float_list(logits),
                    "with_document_choice_probabilities": float_list(probabilities),
                    "gold_choice_logprob_delta": float(gold_delta.item()),
                    # Retain the exact historical JSON contract for layer_28.
                    # All-layer scalars live losslessly in pair_features.safetensors
                    # to avoid expanding pairs.jsonl by tens of gigabytes.
                    "utility_projection_by_layer": {
                        "layer_28": float(utility[legacy_layer_index].item())
                    },
                    "delta_h_norm_by_layer": {
                        "layer_28": float(delta_norm[legacy_layer_index].item())
                    },
                    "delta_c_cosine_by_layer": {
                        "layer_28": float(cosine[legacy_layer_index].item())
                    },
                    "all_layer_scalar_feature_file": "pair_features.safetensors",
                    "all_layer_scalar_tensor_row": pair_index,
                    "input_token_count": len(sequence),
                    "user_prompt_sha256": sha256_text(prompt),
                    "feature_file": "pair_features.safetensors",
                    "feature_tensor_row": pair_index,
                    "h0_and_c_question_tensor_row": runtime["shard_question_index"],
                }
            )

    question_tensors: dict[str, torch.Tensor] = {
        # Historical layer-28 contract (same keys, dtypes and shapes).
        "h0": torch.stack(question_h0_legacy),
        "c_unit": torch.stack(question_c_unit_legacy),
        "c_norm": torch.stack(question_c_norm_legacy).float(),
        # Extended all-layer/selected-layer contract.
        "c_norm_all": torch.stack(question_c_norm_all).float(),
        "choice_logits": torch.stack(question_logits).float(),
        "choice_probabilities": torch.stack(question_probs).float(),
    }
    if args.vector_layer_indices:
        question_tensors.update(
            {
                "h0_selected": torch.stack(question_h0_vector),
                "c_unit_selected": torch.stack(question_c_unit_vector),
                "c_norm_selected": torch.stack(question_c_norm_vector).float(),
            }
        )
    pair_tensors: dict[str, torch.Tensor] = {
        # Historical layer-28 contract (same keys, dtypes and shapes).
        "hD": torch.stack(pair_hD_legacy),
        "choice_logits": torch.stack(pair_logits).float(),
        "choice_probabilities": torch.stack(pair_probs).float(),
        "utility_projection": torch.stack(pair_utility).float()[:, legacy_layer_indices],
        "delta_h_norm": torch.stack(pair_delta_norm).float()[:, legacy_layer_indices],
        "delta_c_cosine": torch.stack(pair_cosine).float()[:, legacy_layer_indices],
        # New all-layer scalar arrays; columns follow scalar_layer_order.
        "utility_projection_all": torch.stack(pair_utility).float(),
        "delta_h_norm_all": torch.stack(pair_delta_norm).float(),
        "delta_c_cosine_all": torch.stack(pair_cosine).float(),
        "gold_choice_logprob_delta": torch.stack(pair_gold_logprob_delta).float(),
        "question_tensor_row": torch.tensor(pair_question_rows, dtype=torch.int64),
    }
    if args.vector_layer_indices:
        pair_tensors["hD_selected"] = torch.stack(pair_hD_vector)

    tensor_metadata = {
        "format_version": FORMAT_VERSION,
        "layer_order": canonical_json(["layer_28"]),
        "scalar_layer_order": canonical_json(extractor.layer_names),
        "vector_layer_order": canonical_json(args.vector_layer_names),
        "choice_order": canonical_json(CHOICES),
        "vector_note": (
            "For selected layers delta_h is reconstructed as hD_selected - h0_selected; "
            "c_raw is reconstructed as c_unit_selected * c_norm_selected"
        ),
    }
    atomic_save_safetensors(paths["questions_tensor"], question_tensors, tensor_metadata)
    atomic_save_safetensors(paths["pairs_tensor"], pair_tensors, tensor_metadata)
    atomic_write_jsonl(paths["questions_meta"], question_metadata)
    atomic_write_jsonl(paths["pairs_meta"], pair_metadata)
    atomic_write_json(
        paths["complete"],
        {
            "format_version": FORMAT_VERSION,
            "completed_at": utc_now(),
            "shard_index": shard_index,
            "question_count": len(question_metadata),
            "pair_count": len(pair_metadata),
            "scalar_layer_order": list(extractor.layer_names),
            "vector_layer_order": list(args.vector_layer_names),
        },
    )


def estimate_storage_bytes(args: argparse.Namespace, total_questions: int) -> dict[str, int]:
    total_pairs = total_questions * args.docs_per_question
    scalar_layers = len(args.scalar_layer_names)
    vector_layers = len(args.vector_layer_names)
    hidden_size = 4096
    return {
        "pair_scalar_float32": total_pairs * scalar_layers * 3 * 4,
        "pair_selected_hD_float16": total_pairs * vector_layers * hidden_size * 2,
        "question_selected_h0_c_float16": total_questions * vector_layers * hidden_size * 2 * 2,
        "question_scalar_c_norm_float32": total_questions * scalar_layers * 4,
        "legacy_layer28_question_h0_c_float16": total_questions * hidden_size * 2 * 2,
        "legacy_layer28_pair_hD_float16": total_pairs * hidden_size * 2,
        "legacy_layer28_question_c_norm_float32": total_questions * 4,
        "legacy_layer28_pair_scalars_float32": total_pairs * 3 * 4,
    }


def invocation_contract(args: argparse.Namespace, total_questions: int) -> dict[str, Any]:
    stat = args.candidates_path.stat()
    return {
        "run_version": RUN_VERSION,
        "format_version": FORMAT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": args.dataset,
        "candidates_path": str(args.candidates_path.resolve()),
        "candidates_size": stat.st_size,
        "candidates_mtime_ns": stat.st_mtime_ns,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "total_questions": total_questions,
        "total_pairs": total_questions * args.docs_per_question,
        "docs_per_question": args.docs_per_question,
        "num_hidden_layers": args.num_hidden_layers,
        "scalar_layer_order": list(args.scalar_layer_names),
        "vector_layer_order": list(args.vector_layer_names),
        "questions_per_shard": args.questions_per_shard,
        "max_input_tokens": args.max_input_tokens,
        "start_question": args.start_question,
        "limit_questions": args.limit_questions,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "answer_contract": "one constrained argmax token over A/B/C/D after Final answer:",
        "direction_contract": "c = -gradient of gold choice-normalized NLL with respect to h0",
        "storage_contract": {
            "legacy_layer28_question": "h0, c_unit, c_norm, choice_logits, choice_probabilities",
            "legacy_layer28_pair": (
                "hD, utility_projection, delta_h_norm, delta_c_cosine, "
                "gold_choice_logprob_delta, choice_logits, choice_probabilities, question_tensor_row"
            ),
            "all_scalar_layers": (
                "utility_projection_all, delta_h_norm_all, delta_c_cosine_all, c_norm_all"
            ),
            "selected_vector_layers_question": "h0_selected, c_unit_selected, c_norm_selected",
            "selected_vector_layers_pair": "hD_selected",
            "legacy_json_layer": "layer_28",
            "vector_dtype": "float16",
            "scalar_dtype": "float32",
        },
        "estimated_core_tensor_bytes": estimate_storage_bytes(args, total_questions),
    }


def validate_manifest(args: argparse.Namespace, total_questions: int) -> None:
    path = args.output_dir / "run_manifest.json"
    contract = invocation_contract(args, total_questions)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in contract}
        if comparable != contract:
            raise RuntimeError(
                "Existing run manifest does not match this invocation; use a new output directory"
            )
        if not args.resume:
            raise FileExistsError(f"Output exists with --no-resume: {args.output_dir}")
        return
    manifest = dict(contract)
    manifest["created_at"] = utc_now()
    atomic_write_json(path, manifest)


def consolidate_outputs(args: argparse.Namespace, shard_count: int) -> dict[str, Any]:
    question_output = args.output_dir / "questions.jsonl"
    pair_output = args.output_dir / "pairs.jsonl"
    question_temp = question_output.with_name(question_output.name + ".partial")
    pair_temp = pair_output.with_name(pair_output.name + ".partial")
    counters: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    projection_sum = torch.zeros(len(args.scalar_layer_names), dtype=torch.float64)
    question_count = 0
    pair_count = 0
    with question_temp.open("w", encoding="utf-8") as question_handle, pair_temp.open(
        "w", encoding="utf-8"
    ) as pair_handle:
        for shard_index in range(shard_count):
            paths = shard_paths(args.output_dir, shard_index)
            with paths["questions_meta"].open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        question_handle.write(line)
                        row = json.loads(line)
                        question_count += 1
                        counters["no_document_correct"] += int(row["no_document_correct"])
            with paths["pairs_meta"].open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        pair_handle.write(line)
                        row = json.loads(line)
                        pair_count += 1
                        counters["with_document_correct"] += int(row["with_document_correct"])
                        transitions[row["answer_transition"]] += 1
            with safe_open(paths["pairs_tensor"], framework="pt", device="cpu") as tensors:
                projection = tensors.get_tensor("utility_projection_all").double()
                projection_sum += projection.sum(dim=0)
        question_handle.flush()
        pair_handle.flush()
        os.fsync(question_handle.fileno())
        os.fsync(pair_handle.fileno())
    os.replace(question_temp, question_output)
    os.replace(pair_temp, pair_output)
    return {
        "format_version": FORMAT_VERSION,
        "completed_at": utc_now(),
        "dataset": args.dataset,
        "questions": question_count,
        "pairs": pair_count,
        "no_document_correct": counters["no_document_correct"],
        "no_document_accuracy": counters["no_document_correct"] / question_count,
        "with_document_correct": counters["with_document_correct"],
        "with_document_pair_accuracy": counters["with_document_correct"] / pair_count,
        "transitions": dict(sorted(transitions.items())),
        "scalar_layer_order": list(args.scalar_layer_names),
        "vector_layer_order": list(args.vector_layer_names),
        "mean_utility_projection_by_layer": {
            name: float(projection_sum[index].item() / pair_count)
            for index, name in enumerate(args.scalar_layer_names)
        },
        "question_metadata": str(question_output.resolve()),
        "pair_metadata": str(pair_output.resolve()),
        "vector_shards": str((args.output_dir / "shards").resolve()),
    }


def run(args: argparse.Namespace) -> None:
    if not args.candidates_path.is_file():
        raise FileNotFoundError(args.candidates_path)
    if args.docs_per_question < 1 or args.questions_per_shard < 1:
        raise ValueError("Document and shard sizes must be positive")
    if args.start_question < 0 or (args.limit_questions is not None and args.limit_questions < 1):
        raise ValueError("Invalid question range")
    resolve_layer_contract(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_questions = count_lines(args.candidates_path)
    total_questions = selected_question_count(
        source_questions, args.start_question, args.limit_questions
    )
    if total_questions == 0:
        raise RuntimeError("Selected question range is empty")
    total_pairs = total_questions * args.docs_per_question
    shard_count = math.ceil(total_questions / args.questions_per_shard)
    validate_manifest(args, total_questions)

    estimate = estimate_storage_bytes(args, total_questions)
    logging.info(
        "Multilayer extraction contract: dataset=%s questions=%d pairs=%d scalar_layers=%d "
        "vector_layers=%s estimated_core_tensors=%.2f GiB",
        args.dataset,
        total_questions,
        total_pairs,
        len(args.scalar_layer_names),
        args.vector_layer_names,
        sum(estimate.values()) / (1024**3),
    )
    if args.dry_run:
        dry_run_summary = invocation_contract(args, total_questions)
        dry_run_summary.update(
            {
                "status": "dry_run_complete",
                "estimated_core_tensor_gib": sum(estimate.values()) / (1024**3),
            }
        )
        atomic_write_json(args.output_dir / "dry_run_summary.json", dry_run_summary)
        logging.info("Dry run complete: %s", json.dumps(dry_run_summary, ensure_ascii=False))
        return
    completed_pairs = 0
    pending_shards = 0
    for shard_index in range(shard_count):
        questions = min(
            args.questions_per_shard,
            total_questions - shard_index * args.questions_per_shard,
        )
        pairs = questions * args.docs_per_question
        if args.resume and complete_shard_valid(
            shard_paths(args.output_dir, shard_index),
            questions,
            pairs,
            args.scalar_layer_names,
            args.vector_layer_names,
        ):
            completed_pairs += pairs
        else:
            pending_shards += 1
    logging.info(
        "Resume audit: shards=%d completed_pairs=%d pending_shards=%d",
        shard_count,
        completed_pairs,
        pending_shards,
    )

    extractor: FeatureExtractor | None = None
    progress = tqdm(
        total=total_pairs,
        initial=completed_pairs,
        desc=f"PreAnswerHiddenMulti:{args.dataset}",
        unit="pair",
        dynamic_ncols=True,
    )
    observed_questions = 0
    for shard_index, shard_rows in enumerate(
        stream_chunks(iter_selected_rows(args), args.questions_per_shard)
    ):
        observed_questions += len(shard_rows)
        expected_pairs = len(shard_rows) * args.docs_per_question
        paths = shard_paths(args.output_dir, shard_index)
        if args.resume and complete_shard_valid(
            paths,
            len(shard_rows),
            expected_pairs,
            args.scalar_layer_names,
            args.vector_layer_names,
        ):
            continue
        if extractor is None:
            extractor = FeatureExtractor(args)
            if extractor.layer_names != args.scalar_layer_names:
                raise RuntimeError(
                    f"Extractor layer order mismatch: {extractor.layer_names} != "
                    f"{args.scalar_layer_names}"
                )
        process_shard(args, extractor, shard_index, shard_rows)
        progress.update(expected_pairs)
        progress.set_postfix(shard=shard_index)
    progress.close()
    if observed_questions != total_questions:
        raise RuntimeError(
            f"Observed {observed_questions} selected questions, expected {total_questions}"
        )
    summary = consolidate_outputs(args, shard_count)
    summary.update(
        {
            "run_version": RUN_VERSION,
            "expected_questions": total_questions,
            "expected_pairs": total_pairs,
            "estimated_core_tensor_bytes": estimate,
        }
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    logging.info("Multilayer extraction complete: %s", json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()

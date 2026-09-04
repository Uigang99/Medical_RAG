#!/usr/bin/env python3
"""Bounded PCW semantic-influence learnability pilot.

This pilot tests one narrowly defined claim before any full-data run:

    question-conditioned parallel document windows plus a document-level
    fusion-odds gate can make Semantic-Support document channels exert more
    causal influence on a frozen Llama output than non-support channels.

The target Llama is always frozen.  A small permutation-equivariant router
receives inference-time Semantic classifier probabilities and document length
features.  Its log-gates are added to document-key attention logits only for
post-document output queries.  A PCW mask prevents every document token from
attending to any other document, so changing one gate cannot alter another
document's cached representation.

Document influence is the full-vocabulary Jensen-Shannon divergence between
the Direct-Choice next-token distribution with all channels and the same
distribution with one channel hard-blocked.  Gold answers are used only for a
secondary accuracy guardrail.  They are never an input, target, loss, or
checkpoint-selection signal.

The workflow stops automatically if either (a) the PCW/gate mechanism test or
(b) the 32-question same-set overfit test fails.  A held-out pilot is run only
after both feasibility gates pass.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rag2_direct_choice_document_attribution import (  # noqa: E402
    HierarchicalProgress,
    choice_token_ids,
    direct_sequence,
    spearman,
)
from evaluate_rag2_direct_choice_document_mask_validity import (  # noqa: E402
    canonical_hash,
    jsd_from_logits,
    model_identity,
    sha256_file,
)
from medrag.filtering.rag2_official import resolve_label_token_ids  # noqa: E402
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_pcw_semantic_influence_learnability_pilot_v4"
PCW_VERSION = "shared_question_parallel_symmetric_document_windows_reused_positions_v2"
GATE_VERSION = "post_document_query_document_attention_odds_gate_all_layers_v1"
INFLUENCE_VERSION = "direct_choice_full_vocabulary_jsd_full_vs_pcw_channel_drop_v1"
SUPPORT_LABELS = frozenset({"direct_support", "supporting_evidence"})
NON_SUPPORT_LABELS = frozenset({"no_evidence", "misleading_evidence"})
SPLITS = ("train", "val", "test")
# Document-token IDs use -1 for every non-document token.  Therefore -1
# cannot mean "block nothing" in the shared exact-mask implementation.
NO_BLOCK_DOCUMENT_ID = -2

DEFAULT_BASE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)
DEFAULT_DATA_ROOT = (
    DEFAULT_BASE
    / "semantic_influence_bounded_pilot_v1/medqa/"
    "rag2_semantic_influence_mixed_top8_256_64_64_v1"
)
DEFAULT_SEMANTIC_ROOT = (
    DEFAULT_BASE / "filter_training_inputs_semantic_top8_four_class_v1/medqa"
)
DEFAULT_LLAMA = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_SEMANTIC_MODEL = (
    WORKSPACE_ROOT
    / "models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa/"
    "medqa_semantic_top8_binary_support_epoch8_len1280_fullpair/"
    "20260830_170945/final_model"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "models/RAG2-PCW-Semantic-Influence-Pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medqa",), default="medqa")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--semantic-model", type=Path, default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="medqa_pcw_mixed128_64_64_v4")
    parser.add_argument("--train-questions", type=int, default=128)
    parser.add_argument("--val-questions", type=int, default=64)
    parser.add_argument("--test-questions", type=int, default=64)
    parser.add_argument("--mechanism-questions", type=int, default=32)
    parser.add_argument("--overfit-questions", type=int, default=32)
    parser.add_argument("--overfit-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument("--variant-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--semantic-max-input-tokens", type=int, default=1280)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--router-heads", type=int, default=4)
    parser.add_argument("--router-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-gate", type=float, default=0.05)
    parser.add_argument("--max-gate", type=float, default=1.50)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--non-support-weight", type=float, default=0.5)
    parser.add_argument("--support-floor-weight", type=float, default=1.0)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--gate-supervision-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--mechanism-only",
        action="store_true",
        help="Stop after the PCW/gate forward-mechanism audit; intended for a GPU smoke test.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def stop_progress(progress: HierarchicalProgress, detail: str) -> None:
    """Close progress bars without falsely reporting unfinished stages as 100%."""

    if progress.stage_bar is not None:
        progress.stage_bar.close()
        progress.stage_bar = None
    progress.overall.refresh()
    progress.overall.close()
    print(f"[workflow stopped after stage {progress.stage_index}/{len(progress.stages)}] {detail}", flush=True)


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def load_selected_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    limits = {
        "train": args.train_questions,
        "val": args.val_questions,
        "test": args.test_questions,
    }
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        path = args.data_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = list(iter_jsonl(path))[: limits[split]]
        if len(rows) != limits[split]:
            raise RuntimeError(
                f"Insufficient {split} rows: requested={limits[split]} found={len(rows)}"
            )
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            documents = list(row.get("documents") or [])
            labels = [str(value) for value in row.get("semantic_labels") or []]
            if not sample_id or sample_id in seen:
                raise RuntimeError(f"Missing, duplicate, or cross-split sample_id: {sample_id}")
            seen.add(sample_id)
            if len(documents) != 8 or len(labels) != 8:
                raise RuntimeError(f"Expected eight documents and labels: {sample_id}")
            if not any(label in SUPPORT_LABELS for label in labels):
                raise RuntimeError(f"Mixed cohort has no Support document: {sample_id}")
            if not any(label in NON_SUPPORT_LABELS for label in labels):
                raise RuntimeError(f"Mixed cohort has no non-support document: {sample_id}")
            if any(label not in SUPPORT_LABELS | NON_SUPPORT_LABELS for label in labels):
                raise RuntimeError(f"Unsupported Semantic label: {sample_id}")
            if str(row.get("gold_answer")) not in CHOICES:
                raise RuntimeError(f"Invalid gold answer: {sample_id}")
            pair_ids = [str(document.get("pair_id") or "") for document in documents]
            if len(set(pair_ids)) != 8 or any(not pair_id for pair_id in pair_ids):
                raise RuntimeError(f"Invalid document pair IDs: {sample_id}")
        result[split] = rows
    return result


def semantic_inputs_for_pairs(
    semantic_root: Path,
    selected_pair_ids: set[str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for split in SPLITS:
        path = semantic_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            pair_id = str(row.get("pair_id") or row.get("id") or "")
            if pair_id not in selected_pair_ids:
                continue
            text = str(row.get("input") or "")
            if not text:
                raise RuntimeError(f"Missing Semantic classifier input: {pair_id}")
            previous = values.setdefault(pair_id, text)
            if previous != text:
                raise RuntimeError(f"Conflicting Semantic classifier inputs: {pair_id}")
    missing = selected_pair_ids - values.keys()
    if missing:
        raise RuntimeError(f"Missing Semantic inputs for {len(missing)} pairs; first={next(iter(missing))}")
    return values


@torch.inference_mode()
def score_semantic_probabilities(
    args: argparse.Namespace,
    rows: dict[str, list[dict[str, Any]]],
    output_path: Path,
    progress: HierarchicalProgress,
) -> dict[str, float]:
    selected_pairs = {
        str(document["pair_id"])
        for split_rows in rows.values()
        for row in split_rows
        for document in row["documents"]
    }
    cached: dict[str, float] = {}
    if args.resume and output_path.is_file():
        for row in iter_jsonl(output_path):
            pair_id = str(row.get("pair_id") or "")
            probability = float(row.get("prob_support"))
            if pair_id in selected_pairs and math.isfinite(probability) and 0 <= probability <= 1:
                cached[pair_id] = probability
    missing = sorted(selected_pairs - cached.keys())
    progress.set(len(cached))
    if not missing:
        return cached

    inputs = semantic_inputs_for_pairs(args.semantic_root, set(missing))
    tokenizer = AutoTokenizer.from_pretrained(
        args.semantic_model, local_files_only=True, use_fast=True
    )
    label_ids = resolve_label_token_ids(tokenizer)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.semantic_model,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id
    if decoder_start is None:
        raise RuntimeError("Semantic classifier has no decoder start token")

    new_rows: list[dict[str, Any]] = []
    for start in range(0, len(missing), args.semantic_batch_size):
        pair_ids = missing[start : start + args.semantic_batch_size]
        encoded = tokenizer(
            [inputs[pair_id] for pair_id in pair_ids],
            truncation=True,
            max_length=args.semantic_max_input_tokens,
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        decoder_input_ids = torch.full(
            (len(pair_ids), 1), int(decoder_start),
            dtype=torch.long, device=args.device,
        )
        logits = model(
            **encoded, decoder_input_ids=decoder_input_ids, return_dict=True
        ).logits[:, 0].float()
        candidate = logits[:, [label_ids["helpful"], label_ids["not helpful"]]]
        probabilities = torch.softmax(candidate, dim=-1)[:, 0].cpu().tolist()
        for pair_id, probability in zip(pair_ids, probabilities):
            cached[pair_id] = float(probability)
            new_rows.append({"pair_id": pair_id, "prob_support": float(probability)})
        progress.update(len(pair_ids))

    ordered = [
        {"pair_id": pair_id, "prob_support": cached[pair_id]}
        for pair_id in sorted(selected_pairs)
    ]
    atomic_jsonl(output_path, ordered)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cached


def build_pcw_layout(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Build symmetric document channels around the unchanged Direct-Choice task.

    The source prompt has one separator before middle documents but no matching
    separator before the first/after the last document.  Reusing those raw
    spans would make channel identity depend on its list position.  We retain
    the original question/instruction prefix and assistant answer suffix, then
    encode each document independently behind the same separator token(s).
    Thus a document receives the same token IDs and local positions under any
    permutation.
    """

    sequence = direct_sequence(tokenizer, row, list(row["documents"]))
    document_indices = [list(map(int, values)) for values in sequence["document_token_indices"]]
    if len(document_indices) != 8 or any(not values for values in document_indices):
        raise RuntimeError(f"Invalid Direct-Choice document spans: {row['sample_id']}")
    starts = [values[0] for values in document_indices]
    text_ends = [values[-1] + 1 for values in document_indices]
    if starts != sorted(starts):
        raise RuntimeError(f"Document spans do not follow prompt order: {row['sample_id']}")
    original_ids = list(map(int, sequence["input_ids"]))
    prefix_ids = original_ids[: starts[0]]
    suffix_ids = original_ids[text_ends[-1] :]
    separator_ids = original_ids[text_ends[0] : starts[1]]
    if not separator_ids:
        separator_ids = list(tokenizer.encode("\n\n", add_special_tokens=False))
    if not prefix_ids or not suffix_ids or not separator_ids:
        raise RuntimeError(f"Invalid PCW prefix/output boundary: {row['sample_id']}")

    window_token_ids: list[list[int]] = []
    for document in row["documents"]:
        text_ids = list(
            tokenizer.encode(str(document["text"]).strip(), add_special_tokens=False)
        )
        if not text_ids:
            raise RuntimeError(f"Empty PCW document encoding: {row['sample_id']}")
        window_token_ids.append([*separator_ids, *text_ids])

    flattened_windows = [token for window in window_token_ids for token in window]
    input_ids = torch.tensor([*prefix_ids, *flattened_windows, *suffix_ids], dtype=torch.long)
    prefix_end = len(prefix_ids)
    output_start = prefix_end + len(flattened_windows)
    document_ids = torch.full((len(input_ids),), -1, dtype=torch.long)
    window_ranges: list[tuple[int, int]] = []
    cursor = prefix_end
    for index, window in enumerate(window_token_ids):
        start, end = cursor, cursor + len(window)
        document_ids[start:end] = index
        window_ranges.append((start, end))
        cursor = end
    if cursor != output_start:
        raise RuntimeError(f"PCW window assembly mismatch: {row['sample_id']}")

    maximum_window = max(end - start for start, end in window_ranges)
    position_ids = torch.empty_like(input_ids)
    position_ids[:prefix_end] = torch.arange(prefix_end)
    for start, end in window_ranges:
        position_ids[start:end] = torch.arange(prefix_end, prefix_end + end - start)
    position_ids[output_start:] = torch.arange(
        prefix_end + maximum_window,
        prefix_end + maximum_window + len(input_ids) - output_start,
    )
    query_mask = torch.zeros_like(input_ids)
    query_mask[output_start:] = 1
    if int(position_ids.max()) >= 8192:
        raise RuntimeError(f"PCW position exceeds Llama context: {row['sample_id']}")
    return {
        "sample_id": str(row["sample_id"]),
        "input_ids": input_ids,
        "position_ids": position_ids,
        "document_ids": document_ids,
        "query_mask": query_mask,
        "prefix_end": prefix_end,
        "output_start": output_start,
        "window_ranges": window_ranges,
        "document_lengths": [end - start for start, end in window_ranges],
        "gold_answer": str(row["gold_answer"]),
        "semantic_labels": list(row["semantic_labels"]),
        "documents": list(row["documents"]),
    }


def pcw_additive_mask(layout: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    document_ids = layout["document_ids"].to(device)
    length = int(document_ids.numel())
    positions = torch.arange(length, device=device)
    causal = positions[:, None] >= positions[None, :]
    query_document = document_ids[:, None]
    key_document = document_ids[None, :]
    query_is_document = query_document.ge(0)
    key_is_shared_prefix = positions[None, :].lt(int(layout["prefix_end"]))
    same_document = query_document.eq(key_document) & query_document.ge(0)
    allowed = causal & (~query_is_document | key_is_shared_prefix | same_document)
    # A valid causal row must always retain at least its own/shared-prefix key.
    if not bool(allowed.any(dim=1).all()):
        raise RuntimeError(f"PCW mask contains an empty query row: {layout['sample_id']}")
    mask = torch.zeros((1, 1, length, length), dtype=dtype, device=device)
    return mask.masked_fill(~allowed[None, None], torch.finfo(dtype).min)


def validate_layout_mask(layout: dict[str, Any]) -> None:
    mask = pcw_additive_mask(layout, torch.device("cpu"), torch.float32)[0, 0]
    document_ids = layout["document_ids"]
    for document_index in range(8):
        query_rows = document_ids.eq(document_index)
        other_document_keys = document_ids.ge(0) & document_ids.ne(document_index)
        if bool(mask[query_rows][:, other_document_keys].eq(0).any()):
            raise RuntimeError(
                f"PCW document isolation failed: {layout['sample_id']} "
                f"document={document_index}"
            )
        own_keys = document_ids.eq(document_index)
        if not bool(mask[query_rows][:, own_keys].eq(0).any(dim=1).all()):
            raise RuntimeError(
                f"PCW document cannot see its own causal channel: "
                f"{layout['sample_id']} document={document_index}"
            )
    output_start = int(layout["output_start"])
    visible_at_last = {
        int(document_ids[key])
        for key in range(len(document_ids))
        if float(mask[-1, key]) == 0.0 and int(document_ids[key]) >= 0
    }
    if visible_at_last != set(range(8)) or output_start >= len(document_ids):
        raise RuntimeError(f"PCW output cannot see all document channels: {layout['sample_id']}")


def document_key_bias(log_gates: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:
    if log_gates.ndim != 2 or document_ids.ndim != 2:
        raise ValueError("log_gates and document_ids must be rank-2 tensors")
    safe_ids = document_ids.clamp(min=0, max=log_gates.shape[1] - 1)
    values = log_gates.gather(1, safe_ids)
    return values.masked_fill(document_ids.lt(0), 0.0)


def forward_pcw_logits(
    model: Any,
    layout: dict[str, Any],
    log_gates: torch.Tensor,
    blocked_document_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if log_gates.ndim != 2 or log_gates.shape[1] != 8:
        raise ValueError("log_gates must have shape [batch, 8]")
    batch = int(log_gates.shape[0])
    if blocked_document_ids.shape != (batch,):
        raise ValueError("blocked_document_ids must have shape [batch]")
    input_ids = layout["input_ids"].to(device).unsqueeze(0).expand(batch, -1)
    position_ids = layout["position_ids"].to(device).unsqueeze(0).expand(batch, -1)
    document_ids = layout["document_ids"].to(device).unsqueeze(0).expand(batch, -1)
    query_mask = layout["query_mask"].to(device).unsqueeze(0).expand(batch, -1)
    attention_mask = pcw_additive_mask(layout, device, model.dtype).expand(batch, -1, -1, -1)
    token_bias = document_key_bias(log_gates, document_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
        semantic_token_bias=token_bias,
        semantic_query_mask=query_mask,
        semantic_layer_start=0,
        semantic_token_document_ids=document_ids,
        semantic_blocked_document_ids=blocked_document_ids.to(device),
        semantic_document_block_layer_start=0,
    )
    logits = outputs.logits[:, -1].float()
    del outputs
    return logits


@torch.inference_mode()
def forward_concat_logits(model: Any, layout: dict[str, Any], device: torch.device) -> torch.Tensor:
    input_ids = layout["input_ids"].to(device).unsqueeze(0)
    attention = torch.ones_like(input_ids)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=positions,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    logits = outputs.logits[0, -1].float()
    del outputs
    return logits


def run_variant_batches(
    model: Any,
    layout: dict[str, Any],
    log_gates: torch.Tensor,
    blocked: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(log_gates), batch_size):
        outputs.append(
            forward_pcw_logits(
                model,
                layout,
                log_gates[start : start + batch_size].to(device),
                blocked[start : start + batch_size].to(device),
                device,
            ).detach().cpu()
        )
    return torch.cat(outputs, dim=0)


def choice_index(logits: torch.Tensor, choice_ids_cpu: torch.Tensor) -> int:
    return int(logits.index_select(0, choice_ids_cpu).argmax().item())


def influence_values(full: torch.Tensor, removals: torch.Tensor) -> list[float]:
    return [float(value) for value in jsd_from_logits(full.unsqueeze(0), removals).cpu()]


@torch.inference_mode()
def mechanism_audit(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    layouts: Sequence[dict[str, Any]],
    choice_ids_cpu: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    row_dir = output_dir / "mechanism_rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    values: list[dict[str, Any]] = []
    gate_values = [0.0, 0.25, 0.50, 0.75, 1.0, 1.50]
    for row, layout in zip(rows, layouts):
        sample_id = str(row["sample_id"])
        row_path = row_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"
        if args.resume and row_path.is_file():
            cached = json.loads(row_path.read_text(encoding="utf-8"))
            if cached.get("run_version") == RUN_VERSION and cached.get("sample_id") == sample_id:
                values.append(cached)
                progress.update()
                continue

        device = torch.device(args.device)
        full = forward_pcw_logits(
            model, layout, torch.zeros((1, 8), device=device),
            torch.tensor([NO_BLOCK_DOCUMENT_ID], device=device), device,
        )[0].cpu()
        concat = forward_concat_logits(model, layout, device).cpu()

        permutation = list(reversed(range(8)))
        permuted_row = dict(row)
        permuted_row["documents"] = [row["documents"][index] for index in permutation]
        permuted_row["semantic_labels"] = [row["semantic_labels"][index] for index in permutation]
        permuted_layout = build_pcw_layout(tokenizer, permuted_row)
        permuted = forward_pcw_logits(
            model, permuted_layout, torch.zeros((1, 8), device=device),
            torch.tensor([NO_BLOCK_DOCUMENT_ID], device=device), device,
        )[0].cpu()

        variants: list[tuple[int, float]] = []
        gate_rows: list[torch.Tensor] = []
        blocked_rows: list[int] = []
        for document_index in range(8):
            for gate in gate_values:
                gates = torch.ones(8)
                blocked_document = NO_BLOCK_DOCUMENT_ID
                if gate == 0:
                    blocked_document = document_index
                else:
                    gates[document_index] = gate
                gate_rows.append(gates.log())
                blocked_rows.append(blocked_document)
                variants.append((document_index, gate))
            near_zero = torch.ones(8)
            near_zero[document_index] = 1e-4
            gate_rows.append(near_zero.log())
            blocked_rows.append(NO_BLOCK_DOCUMENT_ID)
            variants.append((document_index, 1e-4))
        variant_logits = run_variant_batches(
            model, layout, torch.stack(gate_rows), torch.tensor(blocked_rows),
            device, args.variant_batch_size,
        )
        by_document: list[dict[str, Any]] = []
        for document_index in range(8):
            indexed = {
                gate: variant_logits[index]
                for index, (current_document, gate) in enumerate(variants)
                if current_document == document_index
            }
            dropped = indexed[0.0]
            curve = [float(jsd_from_logits(indexed[gate], dropped)) for gate in gate_values]
            monotonic = spearman(gate_values[:5], curve[:5])
            by_document.append({
                "document_index": document_index,
                "gate_values": gate_values,
                "influence_from_drop_jsd": curve,
                "gate_0_to_1_spearman": monotonic,
                "near_zero_vs_hard_drop_jsd": float(jsd_from_logits(indexed[1e-4], dropped)),
                "near_zero_vs_hard_drop_choice_agreement": choice_index(indexed[1e-4], choice_ids_cpu)
                == choice_index(dropped, choice_ids_cpu),
                "amplify_1_to_1p5_jsd": float(jsd_from_logits(indexed[1.0], indexed[1.5])),
            })
        value = {
            "run_version": RUN_VERSION,
            "sample_id": sample_id,
            "gold_answer": row["gold_answer"],
            "concat_prediction": CHOICES[choice_index(concat, choice_ids_cpu)],
            "pcw_prediction": CHOICES[choice_index(full, choice_ids_cpu)],
            "permuted_prediction": CHOICES[choice_index(permuted, choice_ids_cpu)],
            "pcw_vs_concat_jsd": float(jsd_from_logits(full, concat)),
            "permutation_jsd": float(jsd_from_logits(full, permuted)),
            "permutation_choice_agreement": choice_index(full, choice_ids_cpu)
            == choice_index(permuted, choice_ids_cpu),
            "documents": by_document,
        }
        atomic_json(row_path, value)
        values.append(value)
        progress.update()

    correlations = [
        float(document["gate_0_to_1_spearman"])
        for row in values for document in row["documents"]
        if document["gate_0_to_1_spearman"] is not None
        and math.isfinite(float(document["gate_0_to_1_spearman"]))
    ]
    near_zero_jsd = [
        float(document["near_zero_vs_hard_drop_jsd"])
        for row in values for document in row["documents"]
    ]
    near_zero_choice = [
        float(document["near_zero_vs_hard_drop_choice_agreement"])
        for row in values for document in row["documents"]
    ]
    permutation_jsd = [float(row["permutation_jsd"]) for row in values]
    permutation_choice = [float(row["permutation_choice_agreement"]) for row in values]
    concat_correct = np.mean([row["concat_prediction"] == row["gold_answer"] for row in values])
    pcw_correct = np.mean([row["pcw_prediction"] == row["gold_answer"] for row in values])
    amplify_jsd = [
        float(document["amplify_1_to_1p5_jsd"])
        for row in values for document in row["documents"]
    ]
    summary = {
        "questions": len(values),
        "documents": len(values) * 8,
        "accuracy_guardrail": {
            "concat_accuracy": float(concat_correct),
            "pcw_accuracy": float(pcw_correct),
            "pcw_minus_concat": float(pcw_correct - concat_correct),
            "meaning": "secondary only; gold answers are not used by the mechanism or loss",
        },
        "permutation": {
            "choice_agreement": float(np.mean(permutation_choice)),
            "mean_full_vocabulary_jsd": float(np.mean(permutation_jsd)),
        },
        "continuous_gate": {
            "defined_document_correlations": len(correlations),
            "median_gate_0_to_1_spearman": float(np.median(correlations)) if correlations else None,
            "mean_gate_0_to_1_spearman": float(np.mean(correlations)) if correlations else None,
            "median_amplify_1_to_1p5_jsd": float(np.median(amplify_jsd)),
        },
        "hard_drop_limit": {
            "mean_near_zero_vs_hard_drop_jsd": float(np.mean(near_zero_jsd)),
            "choice_agreement": float(np.mean(near_zero_choice)),
        },
    }
    decision = {
        "permutation_pass": (
            summary["permutation"]["choice_agreement"] >= 0.98
            and summary["permutation"]["mean_full_vocabulary_jsd"] <= 1e-3
        ),
        "continuous_gate_pass": (
            len(correlations) >= int(0.80 * len(values) * 8)
            and summary["continuous_gate"]["median_gate_0_to_1_spearman"] is not None
            and summary["continuous_gate"]["median_gate_0_to_1_spearman"] >= 0.80 - 1e-8
        ),
        "hard_drop_limit_pass": (
            summary["hard_drop_limit"]["choice_agreement"] >= 0.98
            and summary["hard_drop_limit"]["mean_near_zero_vs_hard_drop_jsd"] <= 1e-3
        ),
    }
    decision["overall_pass"] = bool(all(decision.values()))
    summary["pre_registered_decision"] = decision
    atomic_json(output_dir / "mechanism_summary.json", summary)
    return summary


@torch.inference_mode()
def frozen_baseline_for_split(
    model: Any,
    layouts: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    path: Path,
    progress: HierarchicalProgress,
) -> dict[str, Any]:
    if args.resume and path.is_file():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("run_version") == RUN_VERSION and len(value.get("sample_ids", [])) == len(layouts):
            progress.update(len(layouts))
            return value
    device = torch.device(args.device)
    sample_ids: list[str] = []
    full_logits: list[torch.Tensor] = []
    influences: list[torch.Tensor] = []
    for layout in layouts:
        gates = torch.zeros((9, 8))
        blocked = torch.tensor([NO_BLOCK_DOCUMENT_ID] + list(range(8)), dtype=torch.long)
        logits = run_variant_batches(
            model, layout, gates, blocked, device, args.variant_batch_size
        )
        sample_ids.append(str(layout["sample_id"]))
        full_logits.append(logits[0].to(torch.float16))
        influences.append(jsd_from_logits(logits[0].unsqueeze(0), logits[1:]).float())
        progress.update()
    value = {
        "run_version": RUN_VERSION,
        "sample_ids": sample_ids,
        "full_logits": torch.stack(full_logits),
        "influences": torch.stack(influences),
    }
    atomic_torch(path, value)
    return value


class SetGateRouter(nn.Module):
    """Small permutation-equivariant router initialized to gate=1."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        min_gate: float,
        max_gate: float,
    ) -> None:
        super().__init__()
        if not 0 < min_gate < 1 < max_gate:
            raise ValueError("Gate bounds must satisfy 0 < min_gate < 1 < max_gate")
        self.min_log_gate = math.log(min_gate)
        self.max_log_gate = math.log(max_gate)
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        target = (0.0 - self.min_log_gate) / (self.max_log_gate - self.min_log_gate)
        target = min(max(target, 1e-6), 1 - 1e-6)
        nn.init.constant_(self.head[-1].bias, math.log(target / (1 - target)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.projection(features))
        unit = torch.sigmoid(self.head(hidden).squeeze(-1))
        return self.min_log_gate + unit * (self.max_log_gate - self.min_log_gate)


def router_features(layout: dict[str, Any], semantic_probabilities: dict[str, float]) -> torch.Tensor:
    probabilities = torch.tensor(
        [semantic_probabilities[str(document["pair_id"])] for document in layout["documents"]],
        dtype=torch.float32,
    )
    eps = 1e-5
    logits = torch.logit(probabilities.clamp(eps, 1 - eps)).clamp(-8, 8) / 8.0
    lengths = torch.tensor(layout["document_lengths"], dtype=torch.float32)
    lengths = lengths / lengths.max().clamp_min(1.0)
    centered = probabilities - probabilities.mean()
    return torch.stack([probabilities, logits, lengths, centered], dim=-1)


def summarize_influence_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    support_values: list[float] = []
    non_values: list[float] = []
    pair_outcomes: list[float] = []
    drift: list[float] = []
    choice_preservation: list[float] = []
    mean_support_gate: list[float] = []
    mean_non_gate: list[float] = []
    for row in rows:
        influences = list(map(float, row["influences_jsd"]))
        labels = list(row["semantic_labels"])
        support = [influences[index] for index, label in enumerate(labels) if label in SUPPORT_LABELS]
        non = [influences[index] for index, label in enumerate(labels) if label in NON_SUPPORT_LABELS]
        support_values.extend(support)
        non_values.extend(non)
        pair_outcomes.extend(float(left > right) for left in support for right in non)
        drift.append(float(row["full_output_drift_jsd"]))
        choice_preservation.append(float(row["choice_preserved"]))
        gates = list(map(float, row["gates"]))
        mean_support_gate.extend(gates[index] for index, label in enumerate(labels) if label in SUPPORT_LABELS)
        mean_non_gate.extend(gates[index] for index, label in enumerate(labels) if label in NON_SUPPORT_LABELS)
    return {
        "questions": len(rows),
        "documents": len(support_values) + len(non_values),
        "support_over_non_pair_accuracy": float(np.mean(pair_outcomes)),
        "mean_support_influence_jsd": float(np.mean(support_values)),
        "mean_non_support_influence_jsd": float(np.mean(non_values)),
        "mean_full_output_drift_jsd": float(np.mean(drift)),
        "choice_preservation": float(np.mean(choice_preservation)),
        "mean_support_gate": float(np.mean(mean_support_gate)),
        "mean_non_support_gate": float(np.mean(mean_non_gate)),
    }


def baseline_metrics(
    layouts: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, layout in enumerate(layouts):
        rows.append({
            "sample_id": layout["sample_id"],
            "semantic_labels": layout["semantic_labels"],
            "influences_jsd": [float(value) for value in baseline["influences"][index]],
            "full_output_drift_jsd": 0.0,
            "choice_preserved": True,
            "gates": [1.0] * 8,
        })
    return summarize_influence_rows(rows), rows


@torch.inference_mode()
def evaluate_router(
    model: Any,
    router: SetGateRouter,
    layouts: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
    semantic_probabilities: dict[str, float],
    choice_ids_cpu: torch.Tensor,
    args: argparse.Namespace,
    progress: HierarchicalProgress | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(args.device)
    router.eval()
    rows: list[dict[str, Any]] = []
    for index, layout in enumerate(layouts):
        features = router_features(layout, semantic_probabilities).to(device).unsqueeze(0)
        log_gates = router(features)[0]
        variants = log_gates.unsqueeze(0).expand(9, -1)
        blocked = torch.tensor([NO_BLOCK_DOCUMENT_ID] + list(range(8)), device=device)
        logits = run_variant_batches(
            model, layout, variants.cpu(), blocked.cpu(), device, args.variant_batch_size
        )
        current_full = logits[0]
        base_full = baseline["full_logits"][index].float()
        rows.append({
            "sample_id": layout["sample_id"],
            "semantic_labels": layout["semantic_labels"],
            "semantic_prob_support": [
                semantic_probabilities[str(document["pair_id"])]
                for document in layout["documents"]
            ],
            "gates": log_gates.exp().cpu().tolist(),
            "influences_jsd": influence_values(current_full, logits[1:]),
            "full_output_drift_jsd": float(jsd_from_logits(current_full, base_full)),
            "choice_preserved": choice_index(current_full, choice_ids_cpu)
            == choice_index(base_full, choice_ids_cpu),
        })
        if progress is not None:
            progress.update()
    return summarize_influence_rows(rows), rows


def compare_metrics(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    baseline_non = max(float(baseline["mean_non_support_influence_jsd"]), 1e-12)
    baseline_support = max(float(baseline["mean_support_influence_jsd"]), 1e-12)
    return {
        "pair_accuracy_gain": float(current["support_over_non_pair_accuracy"])
        - float(baseline["support_over_non_pair_accuracy"]),
        "non_support_relative_reduction": 1.0
        - float(current["mean_non_support_influence_jsd"]) / baseline_non,
        "support_influence_retention": float(current["mean_support_influence_jsd"])
        / baseline_support,
        "full_output_drift_jsd": float(current["mean_full_output_drift_jsd"]),
        "choice_preservation": float(current["choice_preservation"]),
    }


def selection_score(comparison: dict[str, float]) -> float:
    penalty = (
        5.0 * max(0.0, 0.90 - comparison["support_influence_retention"])
        + 5.0 * max(0.0, comparison["full_output_drift_jsd"] - 0.01)
    )
    return (
        comparison["pair_accuracy_gain"]
        + 0.25 * max(-1.0, min(1.0, comparison["non_support_relative_reduction"]))
        - penalty
    )


def causal_router_losses(
    logits: torch.Tensor,
    log_gates: torch.Tensor,
    support_index: int,
    non_support_index: int,
    baseline_full: torch.Tensor,
    baseline_support_jsd: float,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if logits.shape[0] != 3:
        raise ValueError("Training logits must be full/support-drop/non-support-drop")
    full, support_drop, non_drop = logits
    eps = 1e-8
    support_distance = torch.sqrt(jsd_from_logits(full, support_drop) + eps)
    non_distance = torch.sqrt(jsd_from_logits(full, non_drop) + eps)
    baseline_support_distance = math.sqrt(max(float(baseline_support_jsd), 0.0) + eps)
    ranking = F.relu(args.ranking_margin - (support_distance - non_distance))
    non_support = non_distance
    support_floor = F.relu(
        torch.as_tensor(baseline_support_distance, device=logits.device) - support_distance
    )
    preservation = torch.sqrt(jsd_from_logits(full, baseline_full) + eps)
    gates = log_gates.exp()
    gate_rank = F.relu(0.50 - (gates[support_index] - gates[non_support_index]))
    non_gate = F.relu(gates[non_support_index] - 0.25)
    support_gate = F.relu(0.90 - gates[support_index])
    gate_supervision = gate_rank + 0.5 * non_gate + 0.5 * support_gate
    total = (
        args.ranking_weight * ranking
        + args.non_support_weight * non_support
        + args.support_floor_weight * support_floor
        + args.preservation_weight * preservation
        + args.gate_supervision_weight * gate_supervision
    )
    return {
        "loss": total,
        "ranking": ranking,
        "non_support": non_support,
        "support_floor": support_floor,
        "preservation": preservation,
        "gate_supervision": gate_supervision,
        "support_distance": support_distance,
        "non_support_distance": non_distance,
    }


def train_router_phase(
    *,
    phase: str,
    model: Any,
    router: SetGateRouter,
    train_layouts: Sequence[dict[str, Any]],
    validation_layouts: Sequence[dict[str, Any]],
    train_baseline: dict[str, Any],
    validation_baseline: dict[str, Any],
    semantic_probabilities: dict[str, float],
    choice_ids_cpu: torch.Tensor,
    args: argparse.Namespace,
    epochs: int,
    output_dir: Path,
    progress: HierarchicalProgress,
    stop_on_overfit_pass: bool,
    contract_fingerprint: str,
) -> dict[str, Any]:
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best_router.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = -math.inf
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_fingerprint") != contract_fingerprint:
            raise RuntimeError(f"{phase} checkpoint contract mismatch")
        router.load_state_dict(checkpoint["router"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
    progress.set((start_epoch - 1) * (len(train_layouts) + len(validation_layouts)))
    baseline_validation_metrics, _ = baseline_metrics(validation_layouts, validation_baseline)
    for epoch in range(start_epoch, epochs + 1):
        router.train()
        indices = list(range(len(train_layouts)))
        random.Random(args.seed + epoch + (0 if phase == "overfit" else 1000)).shuffle(indices)
        sums: Counter[str] = Counter()
        for ordinal in indices:
            layout = train_layouts[ordinal]
            labels = layout["semantic_labels"]
            support = [index for index, label in enumerate(labels) if label in SUPPORT_LABELS]
            non_support = [index for index, label in enumerate(labels) if label in NON_SUPPORT_LABELS]
            support_index = support[(epoch + ordinal) % len(support)]
            non_support_index = non_support[(epoch + ordinal) % len(non_support)]
            optimizer.zero_grad(set_to_none=True)
            features = router_features(layout, semantic_probabilities).to(device).unsqueeze(0)
            log_gates = router(features)[0]
            variants = log_gates.unsqueeze(0).expand(3, -1)
            blocked = torch.tensor(
                [NO_BLOCK_DOCUMENT_ID, support_index, non_support_index], device=device
            )
            logits = forward_pcw_logits(model, layout, variants, blocked, device)
            baseline_full = train_baseline["full_logits"][ordinal].float().to(device)
            losses = causal_router_losses(
                logits,
                log_gates,
                support_index,
                non_support_index,
                baseline_full,
                float(train_baseline["influences"][ordinal, support_index]),
                args,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                sums[key] += float(value.detach().item())
            progress.update()

        validation_metrics, _ = evaluate_router(
            model, router, validation_layouts, validation_baseline,
            semantic_probabilities, choice_ids_cpu, args, progress,
        )
        comparison = compare_metrics(validation_metrics, baseline_validation_metrics)
        score = selection_score(comparison)
        epoch_row = {
            "epoch": epoch,
            "train_loss": {key: value / len(train_layouts) for key, value in sums.items()},
            "validation": validation_metrics,
            "comparison_to_frozen": comparison,
            "selection_score": score,
        }
        history.append(epoch_row)
        if score > best_score:
            best_score = score
            atomic_torch(best_path, {
                "contract_fingerprint": contract_fingerprint,
                "epoch": epoch,
                "score": score,
                "router": router.state_dict(),
            })
        atomic_torch(checkpoint_path, {
            "contract_fingerprint": contract_fingerprint,
            "epoch": epoch,
            "best_score": best_score,
            "router": router.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        })
        print(
            f"[{phase} epoch {epoch}/{epochs}] "
            f"loss={epoch_row['train_loss']['loss']:.5f} "
            f"pair={validation_metrics['support_over_non_pair_accuracy']:.4f} "
            f"pair_gain={comparison['pair_accuracy_gain']:+.4f} "
            f"non_support_reduction={comparison['non_support_relative_reduction']:+.4f} "
            f"support_retention={comparison['support_influence_retention']:.4f} "
            f"output_drift={comparison['full_output_drift_jsd']:.6f}",
            flush=True,
        )
        if stop_on_overfit_pass and (
            validation_metrics["support_over_non_pair_accuracy"] >= 0.80
            and comparison["non_support_relative_reduction"] >= 0.30
            and comparison["support_influence_retention"] >= 0.90
            and comparison["full_output_drift_jsd"] <= 0.02
        ):
            break
    if not best_path.is_file():
        raise RuntimeError(f"No best router checkpoint was produced for {phase}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    router.load_state_dict(best["router"])
    metrics, rows = evaluate_router(
        model, router, validation_layouts, validation_baseline,
        semantic_probabilities, choice_ids_cpu, args, progress=None,
    )
    comparison = compare_metrics(metrics, baseline_validation_metrics)
    result = {
        "phase": phase,
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "metrics": metrics,
        "comparison_to_frozen": comparison,
        "history": history,
    }
    atomic_json(output_dir / "summary.json", result)
    atomic_jsonl(output_dir / "per_question.jsonl", rows)
    return result


def bootstrap_pair_gain(
    baseline_rows: Sequence[dict[str, Any]],
    final_rows: Sequence[dict[str, Any]],
    replicates: int,
    seed: int,
    progress: HierarchicalProgress,
) -> list[float] | None:
    if replicates <= 0 or len(final_rows) < 2:
        return None
    if [row["sample_id"] for row in baseline_rows] != [row["sample_id"] for row in final_rows]:
        raise RuntimeError("Bootstrap baseline/final row order mismatch")
    rng = np.random.default_rng(seed)
    gains: list[float] = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(final_rows), size=len(final_rows))
        baseline = summarize_influence_rows([baseline_rows[int(index)] for index in indices])
        final = summarize_influence_rows([final_rows[int(index)] for index in indices])
        gains.append(
            final["support_over_non_pair_accuracy"]
            - baseline["support_over_non_pair_accuracy"]
        )
        if (replicate + 1) % 25 == 0 or replicate + 1 == replicates:
            progress.set(len(final_rows) + replicate + 1)
    return [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))]


def semantic_classifier_diagnostic(
    layouts: Sequence[dict[str, Any]],
    semantic_probabilities: dict[str, float],
) -> dict[str, Any]:
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    for layout in layouts:
        for document, label in zip(layout["documents"], layout["semantic_labels"]):
            probability = semantic_probabilities[str(document["pair_id"])]
            targets.append(int(label in SUPPORT_LABELS))
            predictions.append(int(probability >= 0.5))
            probabilities.append(probability)
    return {
        "documents": len(targets),
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "target_support_rate": float(np.mean(targets)),
        "predicted_support_rate": float(np.mean(predictions)),
        "mean_predicted_support_probability": float(np.mean(probabilities)),
    }


def render_report(summary: dict[str, Any]) -> str:
    mechanism = summary["mechanism"]
    lines = [
        "# PCW Semantic influence learnability pilot",
        "",
        f"- Overall passed: **{summary['decision']['overall_pass']}**",
        f"- Stopped after: {summary['decision']['stopped_after']}",
        "- Gold answer in training loss: no",
        "- Target Llama parameters trained: no",
        "- Trainable component: set-conditioned document fusion-gate Router",
        "",
        "## Mechanism validity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Questions | {mechanism['questions']} |",
        f"| PCW minus concatenation accuracy (secondary) | {mechanism['accuracy_guardrail']['pcw_minus_concat']:+.4f} |",
        f"| Document permutation choice agreement | {mechanism['permutation']['choice_agreement']:.4f} |",
        f"| Document permutation mean JSD | {mechanism['permutation']['mean_full_vocabulary_jsd']:.6g} |",
        f"| Median gate-to-influence Spearman, gate 0→1 | {mechanism['continuous_gate']['median_gate_0_to_1_spearman']} |",
        f"| Near-zero gate vs hard drop choice agreement | {mechanism['hard_drop_limit']['choice_agreement']:.4f} |",
        f"| Mechanism passed | {mechanism['pre_registered_decision']['overall_pass']} |",
        "",
    ]
    if summary.get("overfit"):
        overfit = summary["overfit"]
        comparison = overfit["comparison_to_frozen"]
        lines.extend([
            "## Tiny-set overfit",
            "",
            "| Metric | Result | Pass threshold |",
            "|---|---:|---:|",
            f"| Support>non-support pair accuracy | {overfit['metrics']['support_over_non_pair_accuracy']:.4f} | ≥0.80 |",
            f"| Non-support influence reduction | {comparison['non_support_relative_reduction']:.4f} | ≥0.30 |",
            f"| Support influence retention | {comparison['support_influence_retention']:.4f} | ≥0.90 |",
            f"| Full-output JSD drift | {comparison['full_output_drift_jsd']:.6f} | ≤0.02 |",
            f"| Overfit passed | {overfit['passed']} |",
            "",
        ])
    if summary.get("test"):
        test = summary["test"]
        comparison = test["comparison_to_frozen"]
        lines.extend([
            "## Held-out test",
            "",
            "| Metric | Result | Pass threshold |",
            "|---|---:|---:|",
            f"| Questions | {test['metrics']['questions']} | — |",
            f"| Support>non-support pair accuracy gain | {comparison['pair_accuracy_gain']:+.4f} | ≥+0.08 |",
            f"| Non-support influence reduction | {comparison['non_support_relative_reduction']:.4f} | ≥0.20 |",
            f"| Support influence retention | {comparison['support_influence_retention']:.4f} | ≥0.90 |",
            f"| Full-output JSD drift | {comparison['full_output_drift_jsd']:.6f} | ≤0.01 |",
            f"| Choice preservation | {comparison['choice_preservation']:.4f} | reported |",
            f"| Pair-gain bootstrap 95% CI | {test['pair_gain_bootstrap_95ci']} | lower >0 |",
            "",
        ])
    lines.append(
        "Passing establishes only bounded MedQA Direct-Choice influence controllability and held-out learnability. "
        "It does not establish rationale-mode, MedMCQA, Top-K>8, free-response, or accuracy improvement."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    integer_values = (
        args.train_questions,
        args.val_questions,
        args.test_questions,
        args.mechanism_questions,
        args.overfit_questions,
        args.overfit_epochs,
        args.epochs,
        args.semantic_batch_size,
        args.variant_batch_size,
    )
    if min(integer_values) <= 0:
        raise ValueError("Question, epoch, and batch counts must be positive")
    if args.mechanism_questions > args.train_questions or args.overfit_questions > args.train_questions:
        raise ValueError("Mechanism/overfit counts cannot exceed train questions")
    if args.router_hidden_dim % args.router_heads:
        raise ValueError("Router hidden dimension must be divisible by router heads")
    if args.bootstrap_replicates < 0:
        raise ValueError("Bootstrap replicates cannot be negative")
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required after preflight")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_root / args.dataset / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [
        "preflight immutable data/model/mask contracts",
        "cache binary Semantic classifier probabilities",
        "tokenize question-conditioned parallel document windows",
        "audit PCW permutation, hard-drop limit, and continuous gate response",
        "cache frozen PCW full/drop influence baselines",
        "32-question same-set Router overfit",
        "held-out Router train and validation checkpoint selection",
        "one-time held-out test, bootstrap, and report",
    ]
    progress = HierarchicalProgress(
        stages,
        # Calibrated from the validated H200 one-question forward/backward
        # smoke.  Active-stage rolling throughput replaces these priors.
        [10.0, 30.0, 60.0, 300.0, 360.0, 1800.0, 1500.0, 300.0],
    )
    try:
        progress.start(1, sum((args.train_questions, args.val_questions, args.test_questions)), "question")
        rows = load_selected_rows(args)
        selected = [row for split in SPLITS for row in rows[split]]
        for _ in selected:
            progress.update()
        source_files = {
            split: file_identity(args.data_root / f"{split}.jsonl") for split in SPLITS
        }
        semantic_files = {
            split: file_identity(args.semantic_root / f"{split}.jsonl") for split in SPLITS
        }
        contract = {
            "run_version": RUN_VERSION,
            "hypothesis": (
                "PCW-isolated document fusion gates make Support channel influence "
                "rank above non-support influence on held-out questions"
            ),
            "dataset": args.dataset,
            "data": source_files,
            "semantic_rows": semantic_files,
            "models": {
                "llama": model_identity(args.llama_model),
                "semantic_classifier": model_identity(args.semantic_model),
            },
            "sizes": {
                "train": args.train_questions,
                "val": args.val_questions,
                "test": args.test_questions,
                "mechanism": args.mechanism_questions,
                "overfit": args.overfit_questions,
            },
            "pcw": PCW_VERSION,
            "gate": GATE_VERSION,
            "influence": INFLUENCE_VERSION,
            "prompt": "anchored question-first Direct-Choice without rationale",
            "semantic_classifier_input": "existing RAG2 evidence-question input",
            "semantic_gold_labels_in_prompt_or_router_input": False,
            "gold_answer_use": "secondary accuracy guardrail only; never training or selection",
            "target_llama_trainable": False,
            "router": {
                "input": "predicted Support probability/logit, relative document length, set-centered probability",
                "hidden_dim": args.router_hidden_dim,
                "heads": args.router_heads,
                "layers": args.router_layers,
                "gate_range": [args.min_gate, args.max_gate],
            },
            "optimization": {
                "overfit_epochs": args.overfit_epochs,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "ranking_margin": args.ranking_margin,
                "weights": {
                    "ranking": args.ranking_weight,
                    "non_support": args.non_support_weight,
                    "support_floor": args.support_floor_weight,
                    "preservation": args.preservation_weight,
                    "gate_supervision": args.gate_supervision_weight,
                },
            },
            "stop_gates": {
                "mechanism": {
                    "permutation_choice_agreement": 0.98,
                    "permutation_mean_jsd_max": 1e-3,
                    "median_gate_influence_spearman": 0.80,
                    "near_zero_vs_drop_choice_agreement": 0.98,
                    "near_zero_vs_drop_mean_jsd_max": 1e-3,
                },
                "overfit": {
                    "pair_accuracy": 0.80,
                    "non_support_reduction": 0.30,
                    "support_retention": 0.90,
                    "full_output_drift_max": 0.02,
                },
                "held_out": {
                    "pair_accuracy_gain": 0.08,
                    "non_support_reduction": 0.20,
                    "support_retention": 0.90,
                    "full_output_drift_max": 0.01,
                    "pair_gain_bootstrap_lower_bound": 0.0,
                },
            },
            "max_input_tokens": args.max_input_tokens,
            "dtype": args.dtype,
            "seed": args.seed,
            "code_commit": git_commit(),
        }
        fingerprint = canonical_hash(contract)
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract_fingerprint") != fingerprint:
                raise RuntimeError("PCW pilot resume contract mismatch; use a new RUN_NAME")
        else:
            atomic_json(manifest_path, {
                **contract,
                "contract_fingerprint": fingerprint,
                "created_at": utc_now(),
            })
        progress.complete(f"questions={len(selected)} manifest={manifest_path}")

        progress.start(2, len(selected) * 8, "question-document")
        if args.preflight_only:
            # Preflight must stay CPU-only and cheap.  It verifies that every
            # selected pair has the exact input expected by the frozen
            # Semantic classifier, but deliberately does not load either
            # model or create a probability cache.
            selected_pair_ids = {
                str(document["pair_id"])
                for row in selected
                for document in row["documents"]
            }
            semantic_inputs_for_pairs(args.semantic_root, selected_pair_ids)
            semantic_probabilities: dict[str, float] = {}
            semantic_diagnostic: dict[str, Any] = {
                "skipped_in_cpu_preflight": True,
                "documents_verified": len(selected_pair_ids),
            }
            progress.update(len(selected_pair_ids))
            progress.complete(
                f"pairs={len(selected_pair_ids)} inputs=verified "
                "classifier_forward=deferred_to_stage_2"
            )
        else:
            semantic_probabilities = score_semantic_probabilities(
                args, rows, output_dir / "semantic_probabilities.jsonl", progress
            )
            semantic_diagnostic = semantic_classifier_diagnostic(
                [
                    {
                        **row,
                        "document_lengths": [
                            len(str(document.get("text") or ""))
                            for document in row["documents"]
                        ],
                    }
                    for row in selected
                ],
                semantic_probabilities,
            )
            progress.complete(
                f"pairs={len(semantic_probabilities)} "
                f"cohort_accuracy={semantic_diagnostic['accuracy']:.4f} "
                f"cache={output_dir/'semantic_probabilities.jsonl'}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            args.llama_model, local_files_only=True, use_fast=True
        )
        if not tokenizer.is_fast:
            raise RuntimeError("Fast Llama tokenizer required for exact document spans")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        progress.start(3, len(selected), "question")
        layouts: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        maximum_tokens = 0
        for split in SPLITS:
            for row in rows[split]:
                layout = build_pcw_layout(tokenizer, row)
                validate_layout_mask(layout)
                if len(layout["input_ids"]) > args.max_input_tokens:
                    raise RuntimeError(
                        f"Input exceeds limit: {layout['sample_id']} "
                        f"tokens={len(layout['input_ids'])} limit={args.max_input_tokens}"
                    )
                maximum_tokens = max(maximum_tokens, len(layout["input_ids"]))
                layouts[split].append(layout)
                progress.update()
        progress.complete(f"questions={len(selected)} max_tokens={maximum_tokens} pcw_mask_contract=passed")
        if args.preflight_only:
            atomic_json(output_dir / "preflight_complete.json", {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "questions": len(selected),
                "max_tokens": maximum_tokens,
                "semantic_classifier_diagnostic": semantic_diagnostic,
                "completed_at": utc_now(),
            })
            stop_progress(progress, f"preflight-only passed; output={output_dir}")
            return

        attention_name = register_semantic_attention()
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        device = torch.device(args.device)
        print(
            f"[model load] frozen Llama={args.llama_model} device={device} "
            f"dtype={args.dtype} attention={attention_name}",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            attn_implementation=attention_name,
        ).to(device)
        model.eval()
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Target Llama must remain fully frozen")
        choice_ids_device = choice_token_ids(tokenizer, device)
        choice_ids_cpu = choice_ids_device.cpu()

        progress.start(4, args.mechanism_questions, "question")
        mechanism = mechanism_audit(
            model,
            tokenizer,
            rows["train"][: args.mechanism_questions],
            layouts["train"][: args.mechanism_questions],
            choice_ids_cpu,
            args,
            output_dir,
            progress,
        )
        progress.complete(
            f"passed={mechanism['pre_registered_decision']['overall_pass']} "
            f"median_gate_spearman={mechanism['continuous_gate']['median_gate_0_to_1_spearman']} "
            f"summary={output_dir/'mechanism_summary.json'}"
        )
        if args.mechanism_only:
            summary = {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "semantic_classifier_diagnostic": semantic_diagnostic,
                "mechanism": mechanism,
                "decision": {
                    "overall_pass": bool(mechanism["pre_registered_decision"]["overall_pass"]),
                    "stopped_after": "PCW continuous-gate mechanism audit by request",
                },
            }
            atomic_json(output_dir / "summary.json", summary)
            (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
            stop_progress(progress, f"mechanism-only complete; report={output_dir/'report.md'}")
            return
        if not mechanism["pre_registered_decision"]["overall_pass"]:
            summary = {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "semantic_classifier_diagnostic": semantic_diagnostic,
                "mechanism": mechanism,
                "decision": {
                    "overall_pass": False,
                    "stopped_after": "PCW continuous-gate mechanism audit",
                    "reason": "Mechanism validity failed; Router training was not run",
                },
            }
            atomic_json(output_dir / "summary.json", summary)
            (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
            stop_progress(progress, f"mechanism failed; training not run; report={output_dir/'report.md'}")
            return

        progress.start(5, len(selected), "question")
        baselines: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            baselines[split] = frozen_baseline_for_split(
                model,
                layouts[split],
                args,
                output_dir / "frozen_baseline" / f"{split}.pt",
                progress,
            )
        progress.complete(f"questions={len(selected)} baseline={output_dir/'frozen_baseline'}")

        router_kwargs = {
            "input_dim": 4,
            "hidden_dim": args.router_hidden_dim,
            "heads": args.router_heads,
            "layers": args.router_layers,
            "min_gate": args.min_gate,
            "max_gate": args.max_gate,
        }
        router = SetGateRouter(**router_kwargs).to(device)
        initial_router = copy.deepcopy(router.state_dict())
        overfit_layouts = layouts["train"][: args.overfit_questions]
        overfit_baseline = {
            "run_version": RUN_VERSION,
            "sample_ids": baselines["train"]["sample_ids"][: args.overfit_questions],
            "full_logits": baselines["train"]["full_logits"][: args.overfit_questions],
            "influences": baselines["train"]["influences"][: args.overfit_questions],
        }
        progress.start(
            6,
            args.overfit_epochs * (2 * args.overfit_questions),
            "question-pass",
        )
        overfit = train_router_phase(
            phase="overfit",
            model=model,
            router=router,
            train_layouts=overfit_layouts,
            validation_layouts=overfit_layouts,
            train_baseline=overfit_baseline,
            validation_baseline=overfit_baseline,
            semantic_probabilities=semantic_probabilities,
            choice_ids_cpu=choice_ids_cpu,
            args=args,
            epochs=args.overfit_epochs,
            output_dir=output_dir / "overfit",
            progress=progress,
            stop_on_overfit_pass=True,
            contract_fingerprint=fingerprint,
        )
        overfit_comparison = overfit["comparison_to_frozen"]
        overfit["passed"] = bool(
            overfit["metrics"]["support_over_non_pair_accuracy"] >= 0.80
            and overfit_comparison["non_support_relative_reduction"] >= 0.30
            and overfit_comparison["support_influence_retention"] >= 0.90
            and overfit_comparison["full_output_drift_jsd"] <= 0.02
        )
        atomic_json(output_dir / "overfit" / "summary.json", overfit)
        progress.complete(
            f"passed={overfit['passed']} best_epoch={overfit['best_epoch']} "
            f"pair={overfit['metrics']['support_over_non_pair_accuracy']:.4f}"
        )
        if not overfit["passed"]:
            summary = {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "semantic_classifier_diagnostic": semantic_diagnostic,
                "mechanism": mechanism,
                "overfit": overfit,
                "decision": {
                    "overall_pass": False,
                    "stopped_after": "32-question Router overfit",
                    "reason": "Same-set influence control failed; held-out training was not run",
                },
            }
            atomic_json(output_dir / "summary.json", summary)
            (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
            stop_progress(progress, f"overfit failed; held-out training not run; report={output_dir/'report.md'}")
            return

        router.load_state_dict(initial_router)
        progress.start(
            7,
            args.epochs * (args.train_questions + args.val_questions),
            "question-pass",
        )
        held_out = train_router_phase(
            phase="held_out",
            model=model,
            router=router,
            train_layouts=layouts["train"],
            validation_layouts=layouts["val"],
            train_baseline=baselines["train"],
            validation_baseline=baselines["val"],
            semantic_probabilities=semantic_probabilities,
            choice_ids_cpu=choice_ids_cpu,
            args=args,
            epochs=args.epochs,
            output_dir=output_dir / "held_out",
            progress=progress,
            stop_on_overfit_pass=False,
            contract_fingerprint=fingerprint,
        )
        progress.complete(
            f"best_epoch={held_out['best_epoch']} validation_pair_gain="
            f"{held_out['comparison_to_frozen']['pair_accuracy_gain']:+.4f}"
        )

        progress.start(8, args.test_questions + args.bootstrap_replicates, "evaluation-unit")
        baseline_test_metrics, baseline_test_rows = baseline_metrics(
            layouts["test"], baselines["test"]
        )
        test_metrics, test_rows = evaluate_router(
            model,
            router,
            layouts["test"],
            baselines["test"],
            semantic_probabilities,
            choice_ids_cpu,
            args,
            progress,
        )
        comparison = compare_metrics(test_metrics, baseline_test_metrics)
        confidence = bootstrap_pair_gain(
            baseline_test_rows,
            test_rows,
            args.bootstrap_replicates,
            args.seed + 2000,
            progress,
        )
        test = {
            "metrics": test_metrics,
            "frozen_baseline": baseline_test_metrics,
            "comparison_to_frozen": comparison,
            "pair_gain_bootstrap_95ci": confidence,
        }
        confidence_pass = confidence is not None and confidence[0] > 0
        decision = {
            "pair_gain_pass": comparison["pair_accuracy_gain"] >= 0.08,
            "non_support_pass": comparison["non_support_relative_reduction"] >= 0.20,
            "support_retention_pass": comparison["support_influence_retention"] >= 0.90,
            "full_output_drift_pass": comparison["full_output_drift_jsd"] <= 0.01,
            "pair_gain_confidence_pass": confidence_pass,
        }
        decision["overall_pass"] = bool(all(decision.values()))
        decision["stopped_after"] = "held-out test"
        summary = {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "semantic_classifier_diagnostic": semantic_diagnostic,
            "mechanism": mechanism,
            "overfit": overfit,
            "held_out_validation": held_out,
            "test": test,
            "decision": decision,
            "completed_at": utc_now(),
        }
        atomic_json(output_dir / "summary.json", summary)
        atomic_jsonl(output_dir / "test_per_question.jsonl", test_rows)
        (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
        atomic_json(output_dir / "COMPLETE.json", {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "overall_pass": decision["overall_pass"],
            "completed_at": utc_now(),
        })
        progress.complete(
            f"overall_pass={decision['overall_pass']} report={output_dir/'report.md'}"
        )
        progress.finish(f"complete; report={output_dir/'report.md'}")
    except Exception:
        print(
            f"[workflow FAILED] rerun the identical command to resume durable stages; "
            f"output={output_dir}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()

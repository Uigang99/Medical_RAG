#!/usr/bin/env python3
"""Question-held-out learnability audit for hidden-state RAG utility.

This diagnostic intentionally separates the observable inputs used by the
hidden-utility filter.  It asks whether the continuous projection score and
its tau-thresholded binary label are predictable on unseen questions from:

* ``h0``;
* ``delta_h = hD - h0``;
* frozen Question+Evidence text embeddings;
* text + delta_h (without h0); and
* text + h0 + delta_h.

Gold-derived ``c`` and the projection score are targets/audit metadata only.
They are never included in a probe input.  Feature caches are sharded and
resumable; no Llama forward pass is required because h0/hD already exist.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))


AUDIT_VERSION = "rag2_hidden_utility_learnability_mvp_v1"
FEATURE_VIEWS = (
    "h0",
    "delta",
    "text",
    "h0_delta",
    "text_delta",
    "text_h0_delta",
)
PROBE_TYPES = ("linear", "mlp")
TRAINING_REGIMES = ("natural", "no_rag_balanced")
SPLITS = ("train", "val", "test")
NO_RAG_STATES = ("no_rag_correct", "no_rag_wrong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--hidden-feature-root", type=Path, required=True)
    parser.add_argument(
        "--text-encoder-path",
        type=Path,
        default=WORKSPACE_ROOT / "models" / "Flan-T5-large",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("all", "cache", "probe"),
        default="all",
        help="Build compact features, train probes, or do both.",
    )
    parser.add_argument("--expected-label-threshold", type=float, default=0.4)
    parser.add_argument("--expected-label-mode", default="positive_vs_rest")
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--cache-shard-rows", type=int, default=8192)
    parser.add_argument("--hidden-projection-dim", type=int, default=256)
    parser.add_argument("--text-projection-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=1729)
    parser.add_argument("--hidden-shard-cache-size", type=int, default=32)
    parser.add_argument("--max-train-questions", type=int, default=0)
    parser.add_argument("--max-val-questions", type=int, default=0)
    parser.add_argument("--max-test-questions", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--feature-views",
        nargs="+",
        choices=FEATURE_VIEWS,
        default=list(FEATURE_VIEWS),
    )
    parser.add_argument(
        "--probe-types",
        nargs="+",
        choices=PROBE_TYPES,
        default=list(PROBE_TYPES),
    )
    parser.add_argument(
        "--probe-training-regimes",
        nargs="+",
        choices=TRAINING_REGIMES,
        default=list(TRAINING_REGIMES),
        help=(
            "Compare natural training with loss weighting that gives no-RAG "
            "correct/wrong groups equal total mass while preserving each "
            "group's natural Helpful/Not Helpful ratio."
        ),
    )
    parser.add_argument("--probe-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    parser.add_argument("--probe-dropout", type=float, default=0.1)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-patience", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--linear-learning-rate", type=float, default=3e-3)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--train-shuffle-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For the first seed, also train linear probes after shuffling the "
            "targets within each question. Question-level prevalence remains "
            "intact while document-target correspondence is destroyed."
        ),
    )
    parser.add_argument(
        "--test-correspondence-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate each fitted probe after permuting document-varying blocks within questions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_int(*values: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def no_rag_state(row: dict[str, Any]) -> int:
    transition = str(row["answer_transition_audit_only"])
    if transition.startswith("C->"):
        return 0
    if transition.startswith("W->"):
        return 1
    raise ValueError(f"Unknown answer transition: {transition}")


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.split_root / args.dataset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("materialization_version") != "rag2_hidden_utility_filter_inputs_v1":
        raise RuntimeError(f"Unsupported split materialization: {manifest_path}")
    if manifest.get("dataset") != args.dataset:
        raise RuntimeError("Dataset mismatch in split manifest")
    if not math.isclose(
        float(manifest.get("threshold")),
        float(args.expected_label_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Threshold mismatch: expected={args.expected_label_threshold} "
            f"actual={manifest.get('threshold')}"
        )
    if str(manifest.get("label_mode")) != args.expected_label_mode:
        raise RuntimeError(
            f"Label mode mismatch: expected={args.expected_label_mode} "
            f"actual={manifest.get('label_mode')}"
        )
    hidden_manifest_path = args.hidden_feature_root / "run_manifest.json"
    if not hidden_manifest_path.is_file():
        raise FileNotFoundError(hidden_manifest_path)
    hidden_manifest = json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
    layers = [str(value).removeprefix("layer_") for value in hidden_manifest.get("layers", [])]
    if layers != ["28"]:
        raise RuntimeError(f"MVP expects the existing layer-28 cache, found layers={layers}")
    return {"split": manifest, "hidden": hidden_manifest}


class HiddenFeatureStore:
    """Read only h0 and hD; gold-derived c/projection tensors are never opened."""

    def __init__(self, root: Path, cache_size: int) -> None:
        self.root = root
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
        first = root / "shards" / "shard_00000"
        with safe_open(str(first / "question_features.safetensors"), framework="pt", device="cpu") as handle:
            self.hidden_size = int(handle.get_slice("h0").get_shape()[-1])

    def _handles(self, shard_index: int) -> tuple[Any, Any]:
        cached = self.cache.pop(shard_index, None)
        if cached is not None:
            self.cache[shard_index] = cached
            return cached
        root = self.root / "shards" / f"shard_{shard_index:05d}"
        value = (
            safe_open(str(root / "question_features.safetensors"), framework="pt", device="cpu"),
            safe_open(str(root / "pair_features.safetensors"), framework="pt", device="cpu"),
        )
        self.cache[shard_index] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def get_many(self, batch: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        h0_values: list[torch.Tensor] = []
        hd_values: list[torch.Tensor] = []
        for row in batch:
            question, pair = self._handles(int(row["feature_shard_index"]))
            h0_values.append(
                question.get_slice("h0")[int(row["feature_question_row"]), 0, :].float()
            )
            hd_values.append(
                pair.get_slice("hD")[int(row["feature_pair_row"]), 0, :].float()
            )
        return torch.stack(h0_values), torch.stack(hd_values)


def stratified_questions(
    path: Path,
    limit: int,
    seed: int,
) -> tuple[set[str], dict[str, int], Counter[str]]:
    states: dict[str, int] = {}
    for row in rows(path):
        sample_id = str(row["sample_id"])
        value = no_rag_state(row)
        previous = states.setdefault(sample_id, value)
        if previous != value:
            raise RuntimeError(f"Question has inconsistent no-RAG state: {sample_id}")
    counts = Counter("no_rag_correct" if value == 0 else "no_rag_wrong" for value in states.values())
    if limit <= 0 or limit >= len(states):
        selected = set(states)
    else:
        grouped: dict[int, list[str]] = defaultdict(list)
        for sample_id, value in states.items():
            grouped[value].append(sample_id)
        allocations = {
            state: int(round(limit * len(values) / len(states)))
            for state, values in grouped.items()
        }
        while sum(allocations.values()) > limit:
            state = max(allocations, key=lambda key: allocations[key])
            allocations[state] -= 1
        while sum(allocations.values()) < limit:
            state = max(grouped, key=lambda key: len(grouped[key]) - allocations[key])
            allocations[state] += 1
        selected = set()
        for state, values in grouped.items():
            values.sort(key=lambda value: stable_int(seed, state, value))
            selected.update(values[: allocations[state]])
    selected_states = {sample_id: states[sample_id] for sample_id in selected}
    return selected, selected_states, counts


def projection_matrices(
    cache_dir: Path,
    hidden_size: int,
    text_size: int,
    hidden_dim: int,
    text_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    path = cache_dir / "projection_matrices.safetensors"
    contract_path = cache_dir / "projection_matrices.json"
    contract = {
        "hidden_size": hidden_size,
        "text_size": text_size,
        "hidden_dim": hidden_dim,
        "text_dim": text_dim,
        "seed": seed,
    }
    if path.is_file() and contract_path.is_file():
        actual = json.loads(contract_path.read_text(encoding="utf-8"))
        if actual != contract:
            raise RuntimeError(f"Projection cache contract mismatch: {contract_path}")
        return load_file(str(path), device="cpu")
    cache_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = {
        "hidden": torch.randn(hidden_size, hidden_dim, generator=generator, dtype=torch.float32)
        / math.sqrt(hidden_dim),
        "text": torch.randn(text_size, text_dim, generator=generator, dtype=torch.float32)
        / math.sqrt(text_dim),
    }
    temporary = path.with_name(f".{path.name}.partial")
    save_file(values, str(temporary))
    os.replace(temporary, path)
    atomic_json(contract_path, contract)
    return values


class FrozenTextEncoder:
    def __init__(self, path: Path, device: torch.device, bf16: bool) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        dtype = torch.bfloat16 if bf16 and device.type == "cuda" else torch.float32
        logging.info("Loading frozen text encoder: %s", path)
        self.model = T5EncoderModel.from_pretrained(path, dtype=dtype).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.d_model)

    def lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        return [len(value) for value in encoded["input_ids"]]

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        if int(encoded["input_ids"].shape[1]) > max_length:
            raise RuntimeError("Overlength text reached frozen encoder after filtering")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        return (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def shard_contract_hash(batch: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in batch:
        digest.update(str(row["pair_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_feature_shard(
    path: Path,
    meta_path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    atomic_json(meta_path, metadata)


def cache_split(
    args: argparse.Namespace,
    split: str,
    cache_dir: Path,
    selected: set[str],
    selected_states: dict[str, int],
    text_encoder: FrozenTextEncoder,
    hidden_store: HiddenFeatureStore,
    matrices: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source = args.split_root / args.dataset / f"{split}.jsonl"
    destination = cache_dir / split
    destination.mkdir(parents=True, exist_ok=True)
    q_indexes = {sample_id: index for index, sample_id in enumerate(sorted(selected))}
    pending: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    progress = tqdm(desc=f"LearnabilityCache:{args.dataset}:{split}", unit="pair")

    def flush(batch: list[dict[str, Any]], shard_index: int) -> None:
        if not batch:
            return
        data_path = destination / f"shard_{shard_index:05d}.safetensors"
        meta_path = destination / f"shard_{shard_index:05d}.json"
        pair_hash = shard_contract_hash(batch)
        expected = {
            "shard_index": shard_index,
            "rows": len(batch),
            "pair_ids_sha256": pair_hash,
        }
        if args.resume and data_path.is_file() and meta_path.is_file():
            actual = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(actual.get(key) == value for key, value in expected.items()):
                shard_reports.append(actual)
                progress.update(len(batch))
                return
            raise RuntimeError(f"Existing feature shard contract mismatch: {data_path}")

        outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        for offset in range(0, len(batch), args.text_batch_size):
            subset = batch[offset : offset + args.text_batch_size]
            texts = [str(row["input"]) for row in subset]
            text_vector = text_encoder.encode(texts, args.max_seq_length)
            h0, hd = hidden_store.get_many(subset)
            h0 = h0.to(text_encoder.device)
            delta = hd.to(text_encoder.device) - h0
            hidden_matrix = matrices["hidden"].to(text_encoder.device)
            text_matrix = matrices["text"].to(text_encoder.device)
            outputs["h0"].append((h0 @ hidden_matrix).to(torch.float16).cpu())
            outputs["delta"].append((delta @ hidden_matrix).to(torch.float16).cpu())
            outputs["text"].append((text_vector @ text_matrix).to(torch.float16).cpu())
        scores = torch.tensor(
            [float(row["hidden_projection_score_audit_only"]) for row in batch],
            dtype=torch.float32,
        )
        labels = torch.tensor([1 if row["target"] == "helpful" else 0 for row in batch], dtype=torch.uint8)
        expected_labels = scores > float(args.expected_label_threshold)
        if not torch.equal(labels.bool(), expected_labels):
            raise RuntimeError("Binary target does not match the declared projection threshold")
        tensor_values = {
            key: torch.cat(value, dim=0) for key, value in outputs.items()
        }
        tensor_values.update(
            {
                "score": scores,
                "label": labels,
                "state": torch.tensor([selected_states[str(row["sample_id"])] for row in batch], dtype=torch.uint8),
                "question": torch.tensor([q_indexes[str(row["sample_id"])] for row in batch], dtype=torch.int32),
                "doc_rank": torch.tensor([int(row["doc_rank"]) for row in batch], dtype=torch.int16),
            }
        )
        metadata = {
            **expected,
            "first_pair_id": str(batch[0]["pair_id"]),
            "last_pair_id": str(batch[-1]["pair_id"]),
            "helpful": int(labels.sum().item()),
            "no_rag_correct": int(sum(selected_states[str(row["sample_id"])] == 0 for row in batch)),
            "no_rag_wrong": int(sum(selected_states[str(row["sample_id"])] == 1 for row in batch)),
        }
        save_feature_shard(data_path, meta_path, tensor_values, metadata)
        shard_reports.append(metadata)
        progress.update(len(batch))

    shard_index = 0
    row_buffer: list[dict[str, Any]] = []
    for row in rows(source):
        if str(row["sample_id"]) not in selected:
            continue
        row_buffer.append(row)
        if len(row_buffer) >= args.text_batch_size:
            lengths = text_encoder.lengths([str(value["input"]) for value in row_buffer])
            for value, length in zip(row_buffer, lengths, strict=True):
                if length > args.max_seq_length:
                    counters["dropped_overlength"] += 1
                else:
                    pending.append(value)
                    if len(pending) >= args.cache_shard_rows:
                        flush(pending, shard_index)
                        shard_index += 1
                        pending = []
            row_buffer = []
    if row_buffer:
        lengths = text_encoder.lengths([str(value["input"]) for value in row_buffer])
        for value, length in zip(row_buffer, lengths, strict=True):
            if length > args.max_seq_length:
                counters["dropped_overlength"] += 1
            else:
                pending.append(value)
                if len(pending) >= args.cache_shard_rows:
                    flush(pending, shard_index)
                    shard_index += 1
                    pending = []
    flush(pending, shard_index)
    progress.close()
    report = {
        "split": split,
        "selected_questions": len(selected),
        "rows": sum(int(value["rows"]) for value in shard_reports),
        "dropped_overlength": int(counters["dropped_overlength"]),
        "shards": shard_reports,
    }
    atomic_json(destination / "manifest.json", report)
    return report


def build_cache(args: argparse.Namespace, contracts: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    device = torch.device(args.device)
    text_config = AutoConfig.from_pretrained(args.text_encoder_path)
    text_size = int(text_config.d_model)
    hidden_store = HiddenFeatureStore(args.hidden_feature_root, args.hidden_shard_cache_size)
    matrices = projection_matrices(
        cache_dir,
        hidden_store.hidden_size,
        text_size,
        args.hidden_projection_dim,
        args.text_projection_dim,
        args.projection_seed,
    )
    limits = {
        "train": args.max_train_questions,
        "val": args.max_val_questions,
        "test": args.max_test_questions,
    }
    selections: dict[str, tuple[set[str], dict[str, int], Counter[str]]] = {}
    for split in SPLITS:
        selections[split] = stratified_questions(
            args.split_root / args.dataset / f"{split}.jsonl",
            limits[split],
            stable_int(args.sampling_seed, args.dataset, split),
        )
        logging.info(
            "%s selected questions=%d source_distribution=%s",
            split,
            len(selections[split][0]),
            dict(selections[split][2]),
        )
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = selections[left][0] & selections[right][0]
            if overlap:
                examples = sorted(overlap)[:3]
                raise RuntimeError(
                    f"Question leakage between {left}/{right}: count={len(overlap)} "
                    f"examples={examples}"
                )
    if args.dry_run:
        return {
            split: {
                "selected_questions": len(value[0]),
                "source_distribution": dict(value[2]),
            }
            for split, value in selections.items()
        }

    text_encoder = FrozenTextEncoder(args.text_encoder_path, device, args.bf16)
    if text_encoder.hidden_size != text_size:
        raise RuntimeError("Text encoder dimension changed after loading")
    reports: dict[str, Any] = {}
    try:
        for split in SPLITS:
            selected, states, _ = selections[split]
            reports[split] = cache_split(
                args,
                split,
                cache_dir,
                selected,
                states,
                text_encoder,
                hidden_store,
                matrices,
            )
    finally:
        text_encoder.close()
    cache_manifest = {
        "audit_version": AUDIT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split_root": str(args.split_root.resolve()),
        "hidden_feature_root": str(args.hidden_feature_root.resolve()),
        "text_encoder_path": str(args.text_encoder_path.resolve()),
        "label_threshold": args.expected_label_threshold,
        "label_mode": args.expected_label_mode,
        "max_seq_length": args.max_seq_length,
        "hidden_projection_dim": args.hidden_projection_dim,
        "text_projection_dim": args.text_projection_dim,
        "projection_seed": args.projection_seed,
        "sampling_seed": args.sampling_seed,
        "question_limits": limits,
        "forbidden_probe_inputs": ["gold-derived c", "projection score", "gold answer", "answer transition"],
        "target_only": ["projection score", "tau-thresholded label"],
        "contracts": contracts,
        "splits": reports,
    }
    atomic_json(cache_dir / "manifest.json", cache_manifest)
    return cache_manifest


@dataclass
class SplitFeatures:
    h0: np.ndarray
    delta: np.ndarray
    text: np.ndarray
    score: np.ndarray
    label: np.ndarray
    state: np.ndarray
    question: np.ndarray


def load_cached_split(cache_dir: Path, split: str) -> SplitFeatures:
    manifest_path = cache_dir / split / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for shard in manifest["shards"]:
        path = cache_dir / split / f"shard_{int(shard['shard_index']):05d}.safetensors"
        tensors = load_file(str(path), device="cpu")
        for key in ("h0", "delta", "text", "score", "label", "state", "question"):
            values[key].append(tensors[key].numpy())
    arrays = {key: np.concatenate(value, axis=0) for key, value in values.items()}
    return SplitFeatures(
        h0=arrays["h0"].astype(np.float32),
        delta=arrays["delta"].astype(np.float32),
        text=arrays["text"].astype(np.float32),
        score=arrays["score"].astype(np.float32),
        label=arrays["label"].astype(np.int64),
        state=arrays["state"].astype(np.int64),
        question=arrays["question"].astype(np.int64),
    )


def feature_view(data: SplitFeatures, view: str) -> np.ndarray:
    if view == "h0":
        return data.h0
    if view == "delta":
        return data.delta
    if view == "text":
        return data.text
    if view == "h0_delta":
        return np.concatenate((data.h0, data.delta), axis=1)
    if view == "text_delta":
        return np.concatenate((data.text, data.delta), axis=1)
    if view == "text_h0_delta":
        return np.concatenate((data.text, data.h0, data.delta), axis=1)
    raise ValueError(view)


def standardize(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (
        (train - mean) / std,
        (validation - mean) / std,
        (test - mean) / std,
        {"mean_abs": float(np.mean(np.abs(mean))), "std_mean": float(np.mean(std))},
    )


class Probe(nn.Module):
    def __init__(self, input_dim: int, probe_type: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if probe_type == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif probe_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(probe_type)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def safe_roc_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else None


def safe_average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    return float(average_precision_score(target, score)) if len(np.unique(target)) == 2 else None


def safe_correlation(function: Any, target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return None
    value = function(target, prediction).statistic
    return float(value) if np.isfinite(value) else None


def binary_group_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        prediction,
        labels=[1, 0],
        zero_division=0,
    )
    return {
        "pairs": int(len(target)),
        "actual_helpful_rate": float(np.mean(target)),
        "predicted_helpful_rate": float(np.mean(prediction)),
        "roc_auc": safe_roc_auc(target, probability),
        "average_precision": safe_average_precision(target, probability),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "precision_helpful": float(precision[0]),
        "recall_helpful": float(recall[0]),
        "f1_helpful": float(f1[0]),
        "precision_not_helpful": float(precision[1]),
        "recall_not_helpful": float(recall[1]),
        "f1_not_helpful": float(f1[1]),
        "support_helpful": int(support[0]),
        "support_not_helpful": int(support[1]),
    }


def regression_group_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "pairs": int(len(target)),
        "pearson": safe_correlation(pearsonr, target, prediction),
        "spearman": safe_correlation(spearmanr, target, prediction),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
    }


def within_question_ranking(
    target: np.ndarray,
    prediction: np.ndarray,
    question: np.ndarray,
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, question_id in enumerate(question.tolist()):
        groups[int(question_id)].append(index)
    comparisons = 0
    correct = 0.0
    macro_values: list[float] = []
    for indexes in groups.values():
        index = np.asarray(indexes, dtype=np.int64)
        positive = prediction[index][target[index] == 1]
        negative = prediction[index][target[index] == 0]
        if len(positive) == 0 or len(negative) == 0:
            continue
        values = positive[:, None] - negative[None, :]
        local = float(np.mean((values > 0).astype(np.float32) + 0.5 * (values == 0)))
        count = int(values.size)
        comparisons += count
        correct += local * count
        macro_values.append(local)
    return {
        "mixed_questions": len(macro_values),
        "comparisons": comparisons,
        "pair_accuracy_micro": float(correct / comparisons) if comparisons else None,
        "pair_accuracy_macro": float(np.mean(macro_values)) if macro_values else None,
    }


def evaluate_predictions(
    label: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
    score_prediction: np.ndarray,
    state: np.ndarray,
    question: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "decision_threshold": threshold,
        "binary": {},
        "continuous": {},
        "ranking": {},
    }
    groups = {"overall": np.ones(len(label), dtype=bool)}
    groups.update({name: state == index for index, name in enumerate(NO_RAG_STATES)})
    for name, mask in groups.items():
        result["binary"][name] = binary_group_metrics(label[mask], probability[mask], threshold)
        result["continuous"][name] = regression_group_metrics(score[mask], score_prediction[mask])
        result["ranking"][name] = within_question_ranking(
            label[mask], probability[mask], question[mask]
        )
    return result


def threshold_from_validation(target: np.ndarray, probability: np.ndarray) -> float:
    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.01, 0.99, 197):
        prediction = probability >= threshold
        value = f1_score(target, prediction, average="macro", zero_division=0)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best[1]


def predict(model: nn.Module, features: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            values.append(model(features[offset : offset + batch_size]).float().cpu().numpy())
    return np.concatenate(values)


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    probe_type: str,
    task: str,
    args: argparse.Namespace,
    seed: int,
    train_weight: np.ndarray | None = None,
    validation_state: np.ndarray | None = None,
    subgroup_balanced_selection: bool = False,
) -> tuple[Probe, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)
    train_tensor = torch.from_numpy(train_x).to(device)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    train_target = torch.from_numpy(train_y.astype(np.float32)).to(device)
    train_weight_tensor = (
        torch.ones(len(train_y), dtype=torch.float32, device=device)
        if train_weight is None
        else torch.from_numpy(train_weight.astype(np.float32)).to(device)
    )
    validation_target = validation_y.astype(np.float32)
    model = Probe(train_x.shape[1], probe_type, args.probe_hidden_dim, args.probe_dropout).to(device)
    learning_rate = args.linear_learning_rate if probe_type == "linear" else args.mlp_learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    if task == "regression":
        target_mean = float(np.mean(train_y))
        target_std = float(np.std(train_y)) or 1.0
        train_target = (train_target - target_mean) / target_std
    else:
        target_mean, target_std = 0.0, 1.0
    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    remaining = args.probe_patience
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        order = torch.randperm(len(train_tensor), generator=generator, device=device)
        losses: list[float] = []
        for offset in range(0, len(order), args.probe_batch_size):
            index = order[offset : offset + args.probe_batch_size]
            output = model(train_tensor[index])
            if task == "classification":
                per_example = nn.functional.binary_cross_entropy_with_logits(
                    output, train_target[index], reduction="none"
                )
            else:
                per_example = nn.functional.huber_loss(
                    output, train_target[index], delta=1.0, reduction="none"
                )
            weights = train_weight_tensor[index]
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        raw = predict(model, validation_tensor, args.probe_batch_size)
        if task == "classification":
            prediction = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
            if subgroup_balanced_selection:
                if validation_state is None:
                    raise RuntimeError("Balanced selection requires validation_state")
                group_values = [
                    safe_roc_auc(
                        validation_target[validation_state == state].astype(np.int64),
                        prediction[validation_state == state],
                    )
                    for state in (0, 1)
                ]
                metric = mean_or_none(group_values)
            else:
                metric = safe_roc_auc(validation_target.astype(np.int64), prediction)
            metric = float(metric if metric is not None else 0.5)
        else:
            prediction = raw * target_std + target_mean
            if subgroup_balanced_selection:
                if validation_state is None:
                    raise RuntimeError("Balanced selection requires validation_state")
                value = mean_or_none(
                    safe_correlation(
                        spearmanr,
                        validation_target[validation_state == state],
                        prediction[validation_state == state],
                    )
                    for state in (0, 1)
                )
            else:
                value = safe_correlation(spearmanr, validation_target, prediction)
            metric = float(value if value is not None else -1.0)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_metric": metric})
        if metric > best_metric + 1e-6:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            remaining = args.probe_patience
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        raise RuntimeError("Probe training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_validation_metric": best_metric,
        "epochs_ran": len(history),
        "target_mean": target_mean,
        "target_std": target_std,
        "history": history,
        "subgroup_balanced_validation_selection": subgroup_balanced_selection,
    }


def no_rag_group_weights(state: np.ndarray) -> np.ndarray:
    """Give C/W equal total loss mass without altering labels within a group."""
    output = np.zeros(len(state), dtype=np.float32)
    for value in (0, 1):
        mask = state == value
        count = int(mask.sum())
        if count == 0:
            raise RuntimeError(f"Missing no-RAG state in probe training data: state={value}")
        output[mask] = len(state) / (2.0 * count)
    return output


def shuffled_targets_within_question(
    value: np.ndarray,
    question: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = value.copy()
    generator = np.random.default_rng(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, question_id in enumerate(question.tolist()):
        groups[int(question_id)].append(index)
    for indexes in groups.values():
        shuffled = np.asarray(indexes, dtype=np.int64)
        shuffled = generator.permutation(shuffled)
        output[np.asarray(indexes, dtype=np.int64)] = value[shuffled]
    return output


def permuted_document_features(
    features: np.ndarray,
    view: str,
    question: np.ndarray,
    text_dim: int,
    hidden_dim: int,
    seed: int,
) -> np.ndarray:
    if view == "h0":
        return features.copy()
    output = features.copy()
    groups: dict[int, list[int]] = defaultdict(list)
    for index, question_id in enumerate(question.tolist()):
        groups[int(question_id)].append(index)
    generator = np.random.default_rng(seed)
    for indexes in groups.values():
        destination = np.asarray(indexes, dtype=np.int64)
        source = generator.permutation(destination)
        if view in {"delta", "text", "h0_delta", "text_delta"}:
            if view == "h0_delta":
                output[destination, hidden_dim:] = features[source, hidden_dim:]
                continue
            output[destination] = features[source]
        elif view == "text_h0_delta":
            output[destination, :text_dim] = features[source, :text_dim]
            delta_start = text_dim + hidden_dim
            output[destination, delta_start:] = features[source, delta_start:]
        else:
            raise ValueError(view)
    return output


def probe_cache(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    cache_manifest_path = cache_dir / "manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(cache_manifest_path)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("audit_version") != AUDIT_VERSION:
        raise RuntimeError("Unsupported compact feature cache")
    expected_cache_contract = {
        "dataset": args.dataset,
        "hidden_feature_root": str(args.hidden_feature_root.resolve()),
        "text_encoder_path": str(args.text_encoder_path.resolve()),
        "label_threshold": args.expected_label_threshold,
        "label_mode": args.expected_label_mode,
        "max_seq_length": args.max_seq_length,
        "hidden_projection_dim": args.hidden_projection_dim,
        "text_projection_dim": args.text_projection_dim,
        "projection_seed": args.projection_seed,
        "sampling_seed": args.sampling_seed,
        "question_limits": {
            "train": args.max_train_questions,
            "val": args.max_val_questions,
            "test": args.max_test_questions,
        },
    }
    mismatches = {
        key: {"expected": value, "actual": cache_manifest.get(key)}
        for key, value in expected_cache_contract.items()
        if cache_manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Compact feature cache contract mismatch: {mismatches}")
    data = {split: load_cached_split(cache_dir, split) for split in SPLITS}

    def split_statistics(value: SplitFeatures) -> dict[str, Any]:
        groups = {
            "overall": np.ones(len(value.label), dtype=bool),
            **{
                name: value.state == index
                for index, name in enumerate(NO_RAG_STATES)
            },
        }
        return {
            "pairs": int(len(value.label)),
            "questions": int(len(np.unique(value.question))),
            "groups": {
                name: {
                    "pairs": int(mask.sum()),
                    "questions": int(len(np.unique(value.question[mask]))),
                    "helpful_rate": float(np.mean(value.label[mask])),
                    "score_mean": float(np.mean(value.score[mask])),
                    "score_std": float(np.std(value.score[mask])),
                }
                for name, mask in groups.items()
            },
        }

    split_summaries = {split: split_statistics(value) for split, value in data.items()}
    device = torch.device(args.device)
    if device.type != "cuda":
        logging.warning("Probe training is optimized for GPU; requested device=%s", device)
    results: list[dict[str, Any]] = []
    for view in args.feature_views:
        logging.info("Preparing feature view=%s", view)
        raw = {split: feature_view(data[split], view) for split in SPLITS}
        train_x, val_x, test_x, scaler = standardize(raw["train"], raw["val"], raw["test"])
        for training_regime in args.probe_training_regimes:
            if training_regime == "natural":
                training_weight = None
                balanced_selection = False
            elif training_regime == "no_rag_balanced":
                training_weight = no_rag_group_weights(data["train"].state)
                balanced_selection = True
            else:
                raise ValueError(training_regime)
            for probe_type in args.probe_types:
                for seed in args.probe_seeds:
                    logging.info(
                        "Training view=%s regime=%s probe=%s seed=%d",
                        view,
                        training_regime,
                        probe_type,
                        seed,
                    )
                    classifier, classifier_report = train_probe(
                        train_x,
                        data["train"].label,
                        val_x,
                        data["val"].label,
                        probe_type,
                        "classification",
                        args,
                        seed,
                        train_weight=training_weight,
                        validation_state=data["val"].state,
                        subgroup_balanced_selection=balanced_selection,
                    )
                    regressor, regressor_report = train_probe(
                        train_x,
                        data["train"].score,
                        val_x,
                        data["val"].score,
                        probe_type,
                        "regression",
                        args,
                        seed,
                        train_weight=training_weight,
                        validation_state=data["val"].state,
                        subgroup_balanced_selection=balanced_selection,
                    )
                    test_tensor = torch.from_numpy(test_x).to(device)
                    val_tensor = torch.from_numpy(val_x).to(device)
                    logits = predict(classifier, test_tensor, args.probe_batch_size)
                    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                    val_logits = predict(classifier, val_tensor, args.probe_batch_size)
                    val_probability = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30, 30)))
                    threshold = threshold_from_validation(data["val"].label, val_probability)
                    score_prediction = (
                        predict(regressor, test_tensor, args.probe_batch_size)
                        * float(regressor_report["target_std"])
                        + float(regressor_report["target_mean"])
                    )
                    evaluation = evaluate_predictions(
                        data["test"].label,
                        data["test"].score,
                        probability,
                        score_prediction,
                        data["test"].state,
                        data["test"].question,
                        threshold,
                    )
                    record: dict[str, Any] = {
                        "view": view,
                        "training_regime": training_regime,
                        "probe_type": probe_type,
                        "seed": seed,
                        "control": "observed",
                        "scaler": scaler,
                        "classifier_training": classifier_report,
                        "regressor_training": regressor_report,
                        "evaluation": evaluation,
                    }
                    results.append(record)

                    if args.test_correspondence_control:
                        permuted_raw = permuted_document_features(
                            raw["test"],
                            view,
                            data["test"].question,
                            data["test"].text.shape[1],
                            data["test"].h0.shape[1],
                            stable_int(seed, view, "test_document_permutation"),
                        )
                        mean = raw["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
                        std = raw["train"].std(axis=0, dtype=np.float64).astype(np.float32)
                        std[std < 1e-6] = 1.0
                        permuted_x = (permuted_raw - mean) / std
                        permuted_tensor = torch.from_numpy(permuted_x).to(device)
                        permuted_logits = predict(classifier, permuted_tensor, args.probe_batch_size)
                        permuted_probability = 1.0 / (1.0 + np.exp(-np.clip(permuted_logits, -30, 30)))
                        permuted_score = (
                            predict(regressor, permuted_tensor, args.probe_batch_size)
                            * float(regressor_report["target_std"])
                            + float(regressor_report["target_mean"])
                        )
                        results.append(
                            {
                                "view": view,
                                "training_regime": training_regime,
                                "probe_type": probe_type,
                                "seed": seed,
                                "control": "within_question_test_document_permutation",
                                "evaluation": evaluate_predictions(
                                    data["test"].label,
                                    data["test"].score,
                                    permuted_probability,
                                    permuted_score,
                                    data["test"].state,
                                    data["test"].question,
                                    threshold,
                                ),
                            }
                        )
                        del permuted_tensor, permuted_x, permuted_raw

                    if (
                        args.train_shuffle_control
                        and probe_type == "linear"
                        and seed == args.probe_seeds[0]
                    ):
                        shuffled_label = shuffled_targets_within_question(
                            data["train"].label,
                            data["train"].question,
                            stable_int(seed, view, "label_shuffle"),
                        )
                        shuffled_score = shuffled_targets_within_question(
                            data["train"].score,
                            data["train"].question,
                            stable_int(seed, view, "score_shuffle"),
                        )
                        shuffled_val_label = shuffled_targets_within_question(
                            data["val"].label,
                            data["val"].question,
                            stable_int(seed, view, "validation_label_shuffle"),
                        )
                        shuffled_val_score = shuffled_targets_within_question(
                            data["val"].score,
                            data["val"].question,
                            stable_int(seed, view, "validation_score_shuffle"),
                        )
                        shuffled_classifier, shuffled_classifier_report = train_probe(
                            train_x,
                            shuffled_label,
                            val_x,
                            shuffled_val_label,
                            probe_type,
                            "classification",
                            args,
                            stable_int(seed, view, "classifier_shuffle") % (2**31),
                            train_weight=training_weight,
                            validation_state=data["val"].state,
                            subgroup_balanced_selection=balanced_selection,
                        )
                        shuffled_regressor, shuffled_regressor_report = train_probe(
                            train_x,
                            shuffled_score,
                            val_x,
                            shuffled_val_score,
                            probe_type,
                            "regression",
                            args,
                            stable_int(seed, view, "regressor_shuffle") % (2**31),
                            train_weight=training_weight,
                            validation_state=data["val"].state,
                            subgroup_balanced_selection=balanced_selection,
                        )
                        shuffled_logits = predict(shuffled_classifier, test_tensor, args.probe_batch_size)
                        shuffled_probability = 1.0 / (1.0 + np.exp(-np.clip(shuffled_logits, -30, 30)))
                        shuffled_val_logits = predict(shuffled_classifier, val_tensor, args.probe_batch_size)
                        shuffled_val_probability = 1.0 / (
                            1.0 + np.exp(-np.clip(shuffled_val_logits, -30, 30))
                        )
                        shuffled_threshold = threshold_from_validation(
                            shuffled_val_label, shuffled_val_probability
                        )
                        shuffled_score_prediction = (
                            predict(shuffled_regressor, test_tensor, args.probe_batch_size)
                            * float(shuffled_regressor_report["target_std"])
                            + float(shuffled_regressor_report["target_mean"])
                        )
                        results.append(
                            {
                                "view": view,
                                "training_regime": training_regime,
                                "probe_type": probe_type,
                                "seed": seed,
                                "control": "within_question_train_target_shuffle",
                                "classifier_training": shuffled_classifier_report,
                                "regressor_training": shuffled_regressor_report,
                                "evaluation": evaluate_predictions(
                                    data["test"].label,
                                    data["test"].score,
                                    shuffled_probability,
                                    shuffled_score_prediction,
                                    data["test"].state,
                                    data["test"].question,
                                    shuffled_threshold,
                                ),
                            }
                        )
                        del shuffled_classifier, shuffled_regressor

                    del classifier, regressor, test_tensor, val_tensor
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        del train_x, val_x, test_x, raw
        gc.collect()

    report = {
        "audit_version": AUDIT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "cache_dir": str(cache_dir.resolve()),
        "feature_views": args.feature_views,
        "probe_types": args.probe_types,
        "probe_training_regimes": args.probe_training_regimes,
        "probe_seeds": args.probe_seeds,
        "primary_generalization_split": "question-disjoint test",
        "primary_subgroups": list(NO_RAG_STATES),
        "forbidden_probe_inputs": ["gold-derived c", "projection score", "gold answer", "answer transition"],
        "data": split_summaries,
        "results": results,
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    return report


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def render_summary(report: dict[str, Any]) -> str:
    observed = [value for value in report["results"] if value["control"] == "observed"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in observed:
        grouped[
            (value["view"], value["training_regime"], value["probe_type"])
        ].append(value)
    lines = [
        f"# Hidden Utility Learnability MVP — {report['dataset']}",
        "",
        "Scores are means over probe seeds. C/W denote no-RAG correct/wrong questions.",
        "The binary decision threshold is selected once on the full validation split.",
        "Within-Q is document-pair ranking on unseen test questions, not a probe fitted to the test question.",
        "",
        "## Data contract",
        "",
        "| Split | Questions | Pairs | C pairs | W pairs | C Helpful rate | W Helpful rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        value = report["data"][split]
        correct = value["groups"]["no_rag_correct"]
        wrong = value["groups"]["no_rag_wrong"]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(value["questions"]),
                    str(value["pairs"]),
                    str(correct["pairs"]),
                    str(wrong["pairs"]),
                    f"{correct['helpful_rate']:.4f}",
                    f"{wrong['helpful_rate']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Question-held-out predictability",
            "",
            "| Feature view | Train regime | Probe | C AUROC | W AUROC | C Macro-F1 | W Macro-F1 | C score rho | W score rho | C within-Q | W within-Q |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (view, training_regime, probe_type), values in grouped.items():
        def metric(section: str, group: str, key: str) -> float | None:
            return mean_or_none(
                value["evaluation"][section][group].get(key) for value in values
            )

        cells = [
            view,
            training_regime,
            probe_type,
            metric("binary", "no_rag_correct", "roc_auc"),
            metric("binary", "no_rag_wrong", "roc_auc"),
            metric("binary", "no_rag_correct", "macro_f1"),
            metric("binary", "no_rag_wrong", "macro_f1"),
            metric("continuous", "no_rag_correct", "spearman"),
            metric("continuous", "no_rag_wrong", "spearman"),
            metric("ranking", "no_rag_correct", "pair_accuracy_macro"),
            metric("ranking", "no_rag_wrong", "pair_accuracy_macro"),
        ]
        rendered = [cells[0], cells[1], cells[2]] + [
            "-" if value is None else f"{value:.4f}" for value in cells[3:]
        ]
        lines.append("| " + " | ".join(rendered) + " |")

    permuted = [
        value
        for value in report["results"]
        if value["control"] == "within_question_test_document_permutation"
    ]
    if permuted:
        controls: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for value in permuted:
            controls[
                (value["view"], value["training_regime"], value["probe_type"])
            ].append(value)
        lines.extend(
            [
                "",
                "## Document-correspondence control",
                "",
                "Document-varying feature blocks are permuted only within each unseen question; h0 is retained.",
                "A positive AUROC drop means that the observed probe used the correct document–feature correspondence.",
                "",
                "| Feature view | Train regime | Probe | C observed | C permuted | C drop | W observed | W permuted | W drop |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, control_values in controls.items():
            observed_values = grouped[key]
            row: list[Any] = [*key]
            for group in NO_RAG_STATES:
                observed_auc = mean_or_none(
                    value["evaluation"]["binary"][group]["roc_auc"]
                    for value in observed_values
                )
                control_auc = mean_or_none(
                    value["evaluation"]["binary"][group]["roc_auc"]
                    for value in control_values
                )
                row.extend(
                    [
                        observed_auc,
                        control_auc,
                        None
                        if observed_auc is None or control_auc is None
                        else observed_auc - control_auc,
                    ]
                )
            rendered = [str(row[0]), str(row[1]), str(row[2])] + [
                "-" if value is None else f"{value:.4f}" for value in row[3:]
            ]
            lines.append("| " + " | ".join(rendered) + " |")

    shuffled = [
        value
        for value in report["results"]
        if value["control"] == "within_question_train_target_shuffle"
    ]
    if shuffled:
        lines.extend(
            [
                "",
                "## Within-question target-shuffle control (linear, first seed)",
                "",
                "| Feature view | Train regime | C AUROC | W AUROC | C score rho | W score rho |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for value in shuffled:
            evaluation = value["evaluation"]
            row = [
                value["view"],
                value["training_regime"],
                evaluation["binary"]["no_rag_correct"]["roc_auc"],
                evaluation["binary"]["no_rag_wrong"]["roc_auc"],
                evaluation["continuous"]["no_rag_correct"]["spearman"],
                evaluation["continuous"]["no_rag_wrong"]["spearman"],
            ]
            rendered = [str(row[0]), str(row[1])] + [
                "-" if item is None else f"{item:.4f}" for item in row[2:]
            ]
            lines.append("| " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "- High h0 performance with weak within-question ranking indicates question-prevalence shortcut.",
            "- W AUROC/correlation above h0 and a large permutation drop support document-specific shared utility.",
            "- Strong within-question ranking but weak cross-question W AUROC indicates question-specific axes.",
            "- W AUROC near 0.5 and score rho near 0 across delta/hybrid views provide no evidence of an accessible shared structure.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    contracts = validate_contract(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "compact_feature_cache")
    if args.mode in {"all", "cache"}:
        cache_report = build_cache(args, contracts, cache_dir)
        if args.dry_run:
            atomic_json(args.output_dir / "dry_run.json", cache_report)
            logging.info("Dry run complete: %s", args.output_dir / "dry_run.json")
            return
    if args.mode in {"all", "probe"}:
        report = probe_cache(args, cache_dir)
        atomic_json(args.output_dir / "results.json", report)
        (args.output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
        logging.info("Learnability audit complete: %s", args.output_dir / "summary.md")


if __name__ == "__main__":
    main()

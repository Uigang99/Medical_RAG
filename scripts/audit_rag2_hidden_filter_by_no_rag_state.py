#!/usr/bin/env python3
"""Audit a trained text+hidden RAG2 filter by the no-document answer state.

The training run saves aggregate metrics only.  This utility re-scores a
saved train/val/test split from its cached *gold-independent* h0/hD features,
then reports pair-level Helpful precision/recall separately for questions
whose target LLM was correct or wrong without a document.

``answer_transition_audit_only`` is used only after scoring to form analysis
groups; it is never passed to the filter model.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_preanswer_text_hidden import TextHiddenRag2Filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--hidden-feature-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-shard-cache-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-pair-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class FeatureStore:
    """Read only h0/hD from the training cache; c is never opened."""

    def __init__(self, root: Path, cache_size: int) -> None:
        self.root = root
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
        first = root / "shards" / "shard_00000"
        with safe_open(str(first / "question_features.safetensors"), framework="pt", device="cpu") as handle:
            self.hidden_size = int(handle.get_slice("h0").get_shape()[-1])

    def _handles(self, shard_index: int) -> tuple[Any, Any]:
        value = self.cache.pop(shard_index, None)
        if value is not None:
            self.cache[shard_index] = value
            return value
        root = self.root / "shards" / f"shard_{shard_index:05d}"
        value = (
            safe_open(str(root / "question_features.safetensors"), framework="pt", device="cpu"),
            safe_open(str(root / "pair_features.safetensors"), framework="pt", device="cpu"),
        )
        self.cache[shard_index] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def get(self, shard_index: int, pair_row: int, question_row: int) -> tuple[torch.Tensor, torch.Tensor]:
        question, pair = self._handles(int(shard_index))
        h0 = question.get_slice("h0")[int(question_row), 0, :]
        h_d = pair.get_slice("hD")[int(pair_row), 0, :]
        return h0, h_d


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def no_rag_state(row: dict[str, Any]) -> str:
    transition = str(row["answer_transition_audit_only"])
    if transition.startswith("C->"):
        return "no_rag_correct"
    if transition.startswith("W->"):
        return "no_rag_wrong"
    raise ValueError(f"Unknown answer transition: {transition}")


def metric(counter: Counter[str], probabilities: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    tp, fp = int(counter["tp"]), int(counter["fp"])
    fn, tn = int(counter["fn"]), int(counter["tn"])
    support_h, support_nh = tp + fn, tn + fp
    predicted_h, predicted_nh = tp + fp, tn + fn
    precision_h = tp / predicted_h if predicted_h else 0.0
    recall_h = tp / support_h if support_h else 0.0
    precision_nh = tn / predicted_nh if predicted_nh else 0.0
    recall_nh = tn / support_nh if support_nh else 0.0
    f1_h = 2 * precision_h * recall_h / (precision_h + recall_h) if precision_h + recall_h else 0.0
    f1_nh = 2 * precision_nh * recall_nh / (precision_nh + recall_nh) if precision_nh + recall_nh else 0.0
    scores: dict[str, Any] = {}
    for target in ("helpful", "not_helpful"):
        values = np.asarray(probabilities["prob_helpful"][target], dtype=np.float32)
        scores[target] = {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "p10": float(np.quantile(values, 0.10)) if values.size else None,
            "p50": float(np.quantile(values, 0.50)) if values.size else None,
            "p90": float(np.quantile(values, 0.90)) if values.size else None,
        }
    return {
        "pairs": tp + fp + fn + tn,
        "actual_helpful": support_h,
        "actual_not_helpful": support_nh,
        "predicted_helpful": predicted_h,
        "predicted_not_helpful": predicted_nh,
        "actual_helpful_rate": support_h / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
        "predicted_helpful_rate": predicted_h / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
        "precision_helpful": precision_h,
        "recall_helpful": recall_h,
        "f1_helpful": f1_h,
        "precision_not_helpful": precision_nh,
        "recall_not_helpful": recall_nh,
        "f1_not_helpful": f1_nh,
        "macro_f1": (f1_h + f1_nh) / 2,
        "balanced_accuracy": (recall_h + recall_nh) / 2,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "prob_helpful_by_target": scores,
    }


def run(args: argparse.Namespace) -> None:
    split_dir = args.split_root / args.dataset
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_root = (args.hidden_feature_root or Path(str(manifest["hidden_feature_dir"]))).resolve()
    input_path = split_dir / f"{args.split}.jsonl"
    if not input_path.is_file() or not (feature_root / "shards").is_dir():
        raise FileNotFoundError(f"Missing split or feature cache: {input_path}, {feature_root}")
    primary_layer = str(manifest.get("primary_layer", ""))
    if primary_layer not in {"28", "layer_28"}:
        raise RuntimeError("This audit expects the layer-28 hidden-feature contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    pair_path = args.output_dir / "pair_scores.jsonl"
    if args.resume and summary_path.is_file() and (not args.write_pair_scores or pair_path.is_file()):
        logging.info("Reusing completed audit: %s", summary_path)
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    store = FeatureStore(feature_root, args.feature_shard_cache_size)
    scorer = TextHiddenRag2Filter(
        checkpoint_path=args.checkpoint_path,
        backbone_path=args.backbone_path,
        state_model_path=Path("/unused/state/model"),
        layer=28,
        filter_batch_size=args.batch_size,
        max_filter_input_length=args.max_seq_length,
        helpful_threshold=0.5,
        device=args.device,
        bf16=args.bf16,
        load_state_extractor=False,
    )
    if scorer.hidden_size != store.hidden_size:
        raise RuntimeError(f"Hidden-size mismatch: model={scorer.hidden_size}, cache={store.hidden_size}")

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    probs: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {"prob_helpful": {"helpful": [], "not_helpful": []}}
    )
    question_states: dict[str, str] = {}
    total_rows = sum(1 for _ in rows(input_path))
    dropped_by_state: Counter[str] = Counter()
    kept_by_state: Counter[str] = Counter()
    scores_handle = pair_path.open("w", encoding="utf-8") if args.write_pair_scores else None
    progress = tqdm(total=total_rows, desc=f"HiddenFilterAudit:{args.dataset}:{args.split}", unit="pair")
    try:
        batch: list[dict[str, Any]] = []
        for row in rows(input_path):
            batch.append(row)
            if len(batch) >= args.batch_size:
                _score_batch(args, scorer, store, batch, counters, probs, question_states, kept_by_state, dropped_by_state, scores_handle)
                progress.update(len(batch))
                batch = []
        if batch:
            _score_batch(args, scorer, store, batch, counters, probs, question_states, kept_by_state, dropped_by_state, scores_handle)
            progress.update(len(batch))
    finally:
        if scores_handle is not None:
            scores_handle.close()
        progress.close()
        scorer.close()

    result = {
        "audit_version": "rag2_hidden_filter_by_no_rag_state_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "backbone_path": str(args.backbone_path.resolve()),
        "hidden_feature_root": str(feature_root),
        "model_input_contract": {
            "included": ["official Question+Evidence text", "h0", "delta_h=hD-h0"],
            "excluded": ["gold answer", "gold-derived c", "projection score", "answer transition"],
            "posthoc_group_only": "answer_transition_audit_only",
        },
        "max_seq_length": args.max_seq_length,
        "raw_pairs": total_rows,
        "kept_pairs_by_no_rag_state": dict(kept_by_state),
        "dropped_overlength_by_no_rag_state": dict(dropped_by_state),
        "questions_by_no_rag_state": dict(Counter(question_states.values())),
        "metrics": {state: metric(counters[state], probs[state]) for state in sorted(counters)},
    }
    write_json(summary_path, result)
    logging.info("Audit complete: %s", summary_path)


def _score_batch(
    args: argparse.Namespace,
    scorer: TextHiddenRag2Filter,
    store: FeatureStore,
    batch: list[dict[str, Any]],
    counters: dict[str, Counter[str]],
    probs: dict[str, dict[str, dict[str, list[float]]]],
    question_states: dict[str, str],
    kept_by_state: Counter[str],
    dropped_by_state: Counter[str],
    scores_handle: Any | None,
) -> None:
    states = [no_rag_state(row) for row in batch]
    for row, state in zip(batch, states):
        previous = question_states.setdefault(str(row["sample_id"]), state)
        if previous != state:
            raise RuntimeError(f"Inconsistent no-RAG state for {row['sample_id']}")
    length_rows = scorer.tokenizer([str(row["input"]) for row in batch], truncation=False, padding=False)["input_ids"]
    keep_indices = [index for index, ids in enumerate(length_rows) if len(ids) <= args.max_seq_length]
    for index, state in enumerate(states):
        (kept_by_state if index in set(keep_indices) else dropped_by_state)[state] += 1
    if not keep_indices:
        return
    kept = [batch[index] for index in keep_indices]
    kept_states = [states[index] for index in keep_indices]
    encoded = scorer.tokenizer(
        [str(row["input"]) for row in kept],
        truncation=False,
        padding=True,
        return_tensors="pt",
    ).to(scorer.device)
    h0_values: list[torch.Tensor] = []
    h_d_values: list[torch.Tensor] = []
    for row in kept:
        h0, h_d = store.get(row["feature_shard_index"], row["feature_pair_row"], row["feature_question_row"])
        h0_values.append(h0)
        h_d_values.append(h_d)
    h0 = torch.stack(h0_values).to(device=scorer.device, dtype=torch.float32)
    h_d = torch.stack(h_d_values).to(device=scorer.device, dtype=torch.float32)
    decoder_start = int(scorer.model.config.decoder_start_token_id)
    decoder_input_ids = torch.full((len(kept), 1), decoder_start, dtype=torch.long, device=scorer.device)
    with torch.inference_mode(), torch.autocast(device_type=scorer.device.type, dtype=torch.bfloat16, enabled=scorer.use_bf16):
        outputs = scorer.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            h0=h0,
            delta_h=h_d - h0,
            decoder_input_ids=decoder_input_ids,
        )
        token_ids = [scorer.label_token_ids["helpful"], scorer.label_token_ids["not helpful"]]
        logits = outputs.logits[:, 0, token_ids].float()
        prob_helpful = torch.softmax(logits, dim=-1)[:, 0].cpu().tolist()
    for row, state, probability in zip(kept, kept_states, prob_helpful):
        target_helpful = str(row["target"]).lower() == "helpful"
        predicted_helpful = float(probability) >= 0.5
        counter = counters[state]
        counter["tp" if target_helpful and predicted_helpful else "fp" if predicted_helpful else "fn" if target_helpful else "tn"] += 1
        probs[state]["prob_helpful"]["helpful" if target_helpful else "not_helpful"].append(float(probability))
        if scores_handle is not None:
            scores_handle.write(json.dumps({
                "pair_id": row["pair_id"], "sample_id": row["sample_id"], "doc_rank": row["doc_rank"],
                "no_rag_state": state, "answer_transition": row["answer_transition_audit_only"],
                "target": "helpful" if target_helpful else "not helpful",
                "prediction": "helpful" if predicted_helpful else "not helpful",
                "prob_helpful": float(probability),
            }, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()

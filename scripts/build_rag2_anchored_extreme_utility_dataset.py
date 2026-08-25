#!/usr/bin/env python3
"""Join cached anchored states with gold directions and build curriculum data.

No LLM forward pass is performed here.  The script reads the cached no-RAG
``h0`` and with-document ``hD`` tensors, computes the selected-anchor utility
projection, audits a train/validation-only threshold grid, and materializes
question-disjoint pointer splits for extreme and neutral curriculum training.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_anchored_extreme_utility_dataset_v1"
SCORE_VERSION = "rag2_anchored_extreme_utility_scores_v1"
SPLITS = ("train", "val", "test")
CHOICES = ("A", "B", "C", "D")


try:
    import msgspec

    _DECODER = msgspec.json.Decoder()

    def decode_json(line: bytes) -> dict[str, Any]:
        return _DECODER.decode(line)

except ImportError:

    def decode_json(line: bytes) -> dict[str, Any]:
        return json.loads(line)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rag-root", type=Path, default=base / "train_no_rag_anchored_features_v1")
    parser.add_argument(
        "--document-root", type=Path, default=base / "document_traces_source_balanced32_rerank8_v1"
    )
    parser.add_argument(
        "--direction-root", type=Path, default=base / "hidden_utility_extreme_curriculum_v1/gold_directions"
    )
    parser.add_argument(
        "--reference-split-root", type=Path, default=base / "filter_training_inputs_rag2_paper_reproduction_v1"
    )
    parser.add_argument(
        "--output-root", type=Path, default=base / "hidden_utility_extreme_curriculum_v1/prepared"
    )
    parser.add_argument("--datasets", nargs="+", choices=("medmcqa", "medqa"), default=["medmcqa", "medqa"])
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--mode", choices=("all", "scores", "dataset"), default="all")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--anchor", default="pre_choice")
    parser.add_argument("--primary-threshold", type=float, default=0.4)
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--neutral-epsilon", type=float, default=0.05)
    parser.add_argument("--minimum-purity", type=float, default=0.95)
    parser.add_argument("--tensor-cache-shards", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield decode_json(line)
            except Exception as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(temporary),
        metadata=metadata,
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class QuestionRef:
    dataset: str
    shard_name: str
    tensor_row: int
    gold_index: int
    no_rag_correct: bool


class QuestionTensorCache:
    def __init__(
        self,
        no_rag_root: Path,
        direction_root: Path,
        source_split: str,
        layer_index: int,
        anchor_index: int,
        capacity: int,
    ) -> None:
        self.no_rag_root = no_rag_root
        self.direction_root = direction_root
        self.source_split = source_split
        self.layer_index = layer_index
        self.anchor_index = anchor_index
        self.capacity = max(1, capacity)
        self.cache: OrderedDict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()

    def _load(self, dataset: str, shard_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (dataset, shard_name)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        no_path = (
            self.no_rag_root
            / "no_rag_features"
            / dataset
            / self.source_split
            / "shards"
            / shard_name
            / "features.safetensors"
        )
        direction_path = (
            self.direction_root / dataset / self.source_split / "shards" / shard_name / "directions.safetensors"
        )
        with safe_open(no_path, framework="pt", device="cpu") as handle:
            h0 = handle.get_slice("anchor_hidden")[:, self.layer_index, self.anchor_index, :].float()
            logits = handle.get_tensor("choice_logits").float()
        with safe_open(direction_path, framework="pt", device="cpu") as handle:
            c_unit = handle.get_tensor("c_unit").float()
        if not (h0.shape[0] == logits.shape[0] == c_unit.shape[0]):
            raise RuntimeError(f"Question tensor count mismatch: {dataset}/{shard_name}")
        value = (h0, c_unit, logits)
        self.cache[key] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def rows(self, refs: list[QuestionRef]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0_rows: list[torch.Tensor] = []
        c_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        for ref in refs:
            h0, c_unit, logits = self._load(ref.dataset, ref.shard_name)
            h0_rows.append(h0[ref.tensor_row])
            c_rows.append(c_unit[ref.tensor_row])
            logits_rows.append(logits[ref.tensor_row])
        return torch.stack(h0_rows), torch.stack(c_rows), torch.stack(logits_rows)


def validate_args(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    if args.primary_threshold <= 0 or args.neutral_epsilon < 0:
        raise ValueError("Threshold must be positive and neutral epsilon non-negative")
    if any(value <= 0 for value in args.threshold_grid):
        raise ValueError("Every threshold-grid value must be positive")
    no_manifest = json.loads((args.no_rag_root / "feature_manifest.json").read_text(encoding="utf-8"))
    doc_manifest = json.loads((args.document_root / "document_feature_manifest.json").read_text(encoding="utf-8"))
    direction_manifest = json.loads((args.direction_root / "direction_manifest.json").read_text(encoding="utf-8"))
    for manifest in (no_manifest, doc_manifest):
        if manifest.get("trace_version") != "rag2_paper_compatible_three_anchor_v1":
            raise RuntimeError("Anchored feature trace contract mismatch")
    layers = [int(value) for value in no_manifest.get("layers") or []]
    anchors = [str(value) for value in no_manifest.get("anchor_order") or []]
    if args.layer not in layers or args.anchor not in anchors:
        raise RuntimeError(f"Missing layer/anchor in no-RAG manifest: {layers}/{anchors}")
    if doc_manifest.get("layers") != no_manifest.get("layers") or doc_manifest.get("anchor_order") != anchors:
        raise RuntimeError("No-RAG and document feature layouts differ")
    if int(direction_manifest.get("layer", -1)) != args.layer or direction_manifest.get("anchor") != args.anchor:
        raise RuntimeError("Direction layer/anchor mismatch")
    return no_manifest, doc_manifest, layers.index(args.layer), anchors.index(args.anchor)


def build_question_index(
    args: argparse.Namespace,
    layer_index: int,
    anchor_index: int,
    progress: PipelineProgress,
) -> dict[str, QuestionRef]:
    del layer_index, anchor_index
    result: dict[str, QuestionRef] = {}
    for dataset in args.datasets:
        feature_dirs = sorted(
            (args.no_rag_root / "no_rag_features" / dataset / args.source_split / "shards").glob("shard_*")
        )
        for feature_dir in feature_dirs:
            directions = args.direction_root / dataset / args.source_split / "shards" / feature_dir.name
            feature_rows = read_jsonl(feature_dir / "questions.jsonl")
            direction_rows = read_jsonl(directions / "questions.jsonl")
            if len(feature_rows) != len(direction_rows):
                raise RuntimeError(f"Question metadata count mismatch: {dataset}/{feature_dir.name}")
            for local, (feature, direction) in enumerate(zip(feature_rows, direction_rows)):
                sample_id = str(feature["sample_id"])
                if sample_id != direction["sample_id"] or int(feature["tensor_row"]) != local:
                    raise RuntimeError(f"Question alignment mismatch: {dataset}/{feature_dir.name}:{local}")
                gold = str(feature["gold_answer"]).upper()
                result[sample_id] = QuestionRef(
                    dataset=dataset,
                    shard_name=feature_dir.name,
                    tensor_row=local,
                    gold_index=CHOICES.index(gold),
                    no_rag_correct=bool(feature["hf_replay_correct"]),
                )
                progress.update(1)
    return result


def score_paths(output_root: Path, dataset: str, source_split: str, shard_name: str) -> dict[str, Path]:
    root = output_root / "score_shards" / dataset / source_split / shard_name
    return {
        "root": root,
        "rows": root / "rows.jsonl",
        "scores": root / "scores.safetensors",
        "complete": root / "COMPLETE.json",
    }


def score_complete(paths: dict[str, Path], expected: int, args: argparse.Namespace) -> bool:
    if any(not paths[name].is_file() for name in ("rows", "scores", "complete")):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("score_version") == SCORE_VERSION
        and int(marker.get("pair_count", -1)) == expected
        and int(marker.get("layer", -1)) == args.layer
        and marker.get("anchor") == args.anchor
    )


def neutral_band(score: float, threshold: float, epsilon: float) -> str:
    if score > threshold:
        return "extreme_helpful"
    if score < -threshold:
        return "extreme_harmful"
    if score > epsilon:
        return "neutral_positive"
    if score < -epsilon:
        return "neutral_negative"
    return "neutral_zero"


def process_score_shard(
    args: argparse.Namespace,
    dataset: str,
    trace_dir: Path,
    feature_dir: Path,
    output: dict[str, Path],
    question_index: dict[str, QuestionRef],
    question_cache: QuestionTensorCache,
    layer_index: int,
    anchor_index: int,
    progress: PipelineProgress,
) -> int:
    trace_rows = read_jsonl(trace_dir / "pairs.jsonl")
    feature_rows = read_jsonl(feature_dir / "pairs.jsonl")
    if len(trace_rows) != len(feature_rows):
        raise RuntimeError(f"Trace/feature pair count mismatch: {dataset}/{trace_dir.name}")
    refs: list[QuestionRef] = []
    for local, (trace, feature) in enumerate(zip(trace_rows, feature_rows)):
        if trace["pair_id"] != feature["pair_id"] or int(feature["tensor_row"]) != local:
            raise RuntimeError(f"Pair alignment mismatch: {dataset}/{trace_dir.name}:{local}")
        ref = question_index.get(str(trace["sample_id"]))
        if ref is None:
            raise RuntimeError(f"Unknown sample_id={trace['sample_id']}")
        refs.append(ref)
    h0, c_unit, no_logits = question_cache.rows(refs)
    with safe_open(feature_dir / "features.safetensors", framework="pt", device="cpu") as handle:
        hD = handle.get_slice("anchor_hidden")[:, layer_index, anchor_index, :].float()
        doc_logits = handle.get_tensor("choice_logits").float()
    delta = hD - h0
    projection = torch.sum(delta * c_unit, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    cosine = projection / delta_norm.clamp_min(1e-12)
    gold = torch.tensor([ref.gold_index for ref in refs], dtype=torch.long)
    no_gold_logprob = F.log_softmax(no_logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
    doc_gold_logprob = F.log_softmax(doc_logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
    exact_delta = doc_gold_logprob - no_gold_logprob
    doc_answers = torch.argmax(doc_logits, dim=-1)
    rows: list[dict[str, Any]] = []
    for local, (trace, feature, ref) in enumerate(zip(trace_rows, feature_rows, refs)):
        score = float(projection[local].item())
        doc_correct = int(doc_answers[local].item()) == ref.gold_index
        rows.append(
            {
                "score_version": SCORE_VERSION,
                "dataset": dataset,
                "source_split": args.source_split,
                "sample_id": trace["sample_id"],
                "pair_id": trace["pair_id"],
                "doc_rank": int(trace.get("doc_rank") or (trace.get("document") or {}).get("rerank_rank") or 0),
                "document_source": str((trace.get("document") or {}).get("source") or "unknown"),
                "trace_shard": trace_dir.name,
                "trace_pair_row": local,
                "document_feature_shard": feature_dir.name,
                "document_tensor_row": local,
                "question_feature_shard": ref.shard_name,
                "question_tensor_row": ref.tensor_row,
                "utility_projection": score,
                "delta_h_norm": float(delta_norm[local].item()),
                "delta_c_cosine": float(cosine[local].item()),
                "exact_gold_logprob_delta": float(exact_delta[local].item()),
                "no_rag_correct": ref.no_rag_correct,
                "with_document_correct": doc_correct,
                "answer_transition": ("C" if ref.no_rag_correct else "W") + "->" + ("C" if doc_correct else "W"),
                "curriculum_band": neutral_band(score, args.primary_threshold, args.neutral_epsilon),
                "usable": bool(trace.get("valid_for_layer_analysis", True)),
                "tensor_row": local,
            }
        )
    atomic_jsonl(output["rows"], rows)
    atomic_safetensors(
        output["scores"],
        {
            "utility_projection": projection.float(),
            "delta_h_norm": delta_norm.float(),
            "delta_c_cosine": cosine.float(),
            "exact_gold_logprob_delta": exact_delta.float(),
        },
        {
            "score_version": SCORE_VERSION,
            "dataset": dataset,
            "source_split": args.source_split,
            "layer": str(args.layer),
            "anchor": args.anchor,
        },
    )
    atomic_json(
        output["complete"],
        {
            "score_version": SCORE_VERSION,
            "completed_at": utc_now(),
            "dataset": dataset,
            "pair_count": len(rows),
            "usable_pairs": sum(row["usable"] for row in rows),
            "layer": args.layer,
            "anchor": args.anchor,
        },
    )
    progress.update(len(rows))
    progress.set_detail(f"dataset={dataset} shard={trace_dir.name}")
    return len(rows)


def load_split_assignments(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    for dataset in args.datasets:
        values: dict[str, str] = {}
        for split in SPLITS:
            path = args.reference_split_root / dataset / "sample_ids" / f"{split}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            for sample_id in path.read_text(encoding="utf-8").splitlines():
                if sample_id:
                    if sample_id in values:
                        raise RuntimeError(f"Split leakage for {sample_id}")
                    values[sample_id] = split
        assignments[dataset] = values
    return assignments


def score_row_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for dataset in args.datasets:
        for path in sorted((args.output_root / "score_shards" / dataset / args.source_split).glob("shard_*/rows.jsonl")):
            paths.append((dataset, path))
    return paths


def audit_thresholds(
    args: argparse.Namespace,
    assignments: dict[str, dict[str, str]],
    total_pairs: int,
    progress: PipelineProgress,
) -> dict[str, Any]:
    thresholds = sorted(set(args.threshold_grid + [args.primary_threshold]))
    counters: dict[tuple[str, str, str, float], Counter[str]] = defaultdict(Counter)
    mixed: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    usable_total = 0
    for dataset, path in score_row_paths(args):
        for row in iter_jsonl(path):
            progress.update(1)
            if not row.get("usable"):
                continue
            sample_id = str(row["sample_id"])
            split = assignments[dataset].get(sample_id)
            if split is None:
                raise RuntimeError(f"No reference split for {sample_id}")
            state = "no_rag_correct" if row["no_rag_correct"] else "no_rag_wrong"
            score = float(row["utility_projection"])
            exact = float(row["exact_gold_logprob_delta"])
            usable_total += 1
            for threshold in thresholds:
                for group_state in (state, "all"):
                    counter = counters[(dataset, split, group_state, threshold)]
                    counter["total"] += 1
                    if score > threshold:
                        counter["helpful"] += 1
                        counter["helpful_direction_correct"] += int(exact > 0)
                    elif score < -threshold:
                        counter["harmful"] += 1
                        counter["harmful_direction_correct"] += int(exact < 0)
                    else:
                        counter["neutral"] += 1
                if math.isclose(threshold, args.primary_threshold, abs_tol=1e-12):
                    if score > threshold:
                        mixed[(dataset, split)][sample_id] |= 1
                    elif score < -threshold:
                        mixed[(dataset, split)][sample_id] |= 2
    rows: list[dict[str, Any]] = []
    for key, counter in sorted(counters.items()):
        dataset, split, state, threshold = key
        helpful = counter["helpful"]
        harmful = counter["harmful"]
        extreme = helpful + harmful
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "no_rag_state": state,
                "threshold": threshold,
                "total": counter["total"],
                "helpful": helpful,
                "harmful": harmful,
                "neutral": counter["neutral"],
                "extreme_coverage": extreme / counter["total"] if counter["total"] else None,
                "helpful_purity": counter["helpful_direction_correct"] / helpful if helpful else None,
                "harmful_purity": counter["harmful_direction_correct"] / harmful if harmful else None,
                "direction_agreement": (
                    (counter["helpful_direction_correct"] + counter["harmful_direction_correct"]) / extreme
                    if extreme
                    else None
                ),
            }
        )
    mixed_summary = {
        f"{dataset}:{split}": {
            "questions_with_any_scored_pair": len(values),
            "questions_with_both_extremes": sum(value == 3 for value in values.values()),
            "mixed_fraction": sum(value == 3 for value in values.values()) / len(values) if values else 0.0,
        }
        for (dataset, split), values in sorted(mixed.items())
    }
    eligible: list[float] = []
    for threshold in thresholds:
        relevant = [
            row
            for row in rows
            if row["threshold"] == threshold
            and row["split"] in {"train", "val"}
            and row["no_rag_state"] != "all"
        ]
        if relevant and all(
            row["helpful_purity"] is not None
            and row["harmful_purity"] is not None
            and row["helpful_purity"] >= args.minimum_purity
            and row["harmful_purity"] >= args.minimum_purity
            for row in relevant
        ):
            eligible.append(threshold)
    recommendation = min(eligible) if eligible else None
    output = {
        "audit_version": "rag2_anchored_extreme_threshold_audit_v1",
        "created_at": utc_now(),
        "score_definition": "raw utility_projection = dot(hD-h0, unit negative gold CE gradient)",
        "primary_threshold": args.primary_threshold,
        "minimum_purity": args.minimum_purity,
        "recommended_threshold_train_val_only": recommendation,
        "usable_pairs": usable_total,
        "rows": rows,
        "mixed_question_coverage_at_primary_threshold": mixed_summary,
    }
    atomic_json(args.output_root / "threshold_audit.json", output)
    csv_path = args.output_root / "threshold_audit.csv"
    temporary = csv_path.with_name(csv_path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    if usable_total > total_pairs:
        raise RuntimeError("Usable score rows exceed expected total")
    return output


def materialize_dataset(
    args: argparse.Namespace,
    assignments: dict[str, dict[str, str]],
    audit: dict[str, Any],
    progress: PipelineProgress,
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Path]] = {}
    for dataset in args.datasets:
        dataset_root = args.output_root / "filter_inputs" / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        outputs[dataset] = {split: dataset_root / f"{split}.jsonl" for split in SPLITS}
        if not args.overwrite_dataset and any(path.exists() for path in outputs[dataset].values()):
            raise FileExistsError(
                f"Filter inputs already exist for {dataset}; pass --overwrite-dataset after checking them"
            )
    counters: dict[str, dict[str, Counter[str]]] = {
        dataset: {split: Counter() for split in SPLITS} for dataset in args.datasets
    }
    question_sets: dict[str, dict[str, set[str]]] = {
        dataset: {split: set() for split in SPLITS} for dataset in args.datasets
    }
    temporary_paths: dict[tuple[str, str], Path] = {}
    with ExitStack() as stack:
        handles = {}
        for dataset in args.datasets:
            handles[dataset] = {}
            for split in SPLITS:
                final = outputs[dataset][split]
                temporary = final.with_name(final.name + ".partial")
                temporary_paths[(dataset, split)] = temporary
                handles[dataset][split] = stack.enter_context(temporary.open("w", encoding="utf-8"))
        for dataset, path in score_row_paths(args):
            for row in iter_jsonl(path):
                progress.update(1)
                if not row.get("usable"):
                    counters[dataset]["train"]["excluded_unusable"] += 1
                    continue
                sample_id = str(row["sample_id"])
                split = assignments[dataset].get(sample_id)
                if split is None:
                    raise RuntimeError(f"No reference split for {sample_id}")
                score = float(row["utility_projection"])
                band = neutral_band(score, args.primary_threshold, args.neutral_epsilon)
                output = {
                    **row,
                    "assigned_split": split,
                    "curriculum_band": band,
                    "label_threshold": args.primary_threshold,
                    "neutral_epsilon": args.neutral_epsilon,
                }
                handles[dataset][split].write(json.dumps(output, ensure_ascii=False) + "\n")
                counters[dataset][split]["rows"] += 1
                counters[dataset][split][band] += 1
                counters[dataset][split]["no_rag_correct" if row["no_rag_correct"] else "no_rag_wrong"] += 1
                question_sets[dataset][split].add(sample_id)
        for dataset in args.datasets:
            for split in SPLITS:
                handles[dataset][split].flush()
                os.fsync(handles[dataset][split].fileno())
    for dataset in args.datasets:
        for split in SPLITS:
            os.replace(temporary_paths[(dataset, split)], outputs[dataset][split])
    summary: dict[str, Any] = {}
    for dataset in args.datasets:
        split_summary = {}
        for split in SPLITS:
            split_summary[split] = {
                "questions": len(question_sets[dataset][split]),
                **dict(counters[dataset][split]),
            }
        manifest = {
            "materialization_version": RUN_VERSION,
            "created_at": utc_now(),
            "dataset": dataset,
            "source_split": args.source_split,
            "no_rag_root": str(args.no_rag_root.resolve()),
            "document_root": str(args.document_root.resolve()),
            "direction_root": str(args.direction_root.resolve()),
            "score_root": str((args.output_root / "score_shards").resolve()),
            "reference_split_root": str(args.reference_split_root.resolve()),
            "layer": args.layer,
            "anchor": args.anchor,
            "threshold": args.primary_threshold,
            "neutral_epsilon": args.neutral_epsilon,
            "threshold_audit": str((args.output_root / "threshold_audit.json").resolve()),
            "recommended_threshold_train_val_only": audit["recommended_threshold_train_val_only"],
            "model_input_contract": {
                "included": [
                    "official Question+Options+Evidence text",
                    "normalized delta_h=hD-h0",
                    "log1p(norm(delta_h))",
                ],
                "forbidden": ["gold answer", "gold-derived c", "projection score", "answer transition"],
                "audit_only": ["projection score", "exact gold logprob delta", "no-RAG correctness"],
            },
            "splits": split_summary,
        }
        atomic_json(args.output_root / "filter_inputs" / dataset / "manifest.json", manifest)
        summary[dataset] = split_summary
    atomic_json(
        args.output_root / "dataset_manifest.json",
        {
            "materialization_version": RUN_VERSION,
            "created_at": utc_now(),
            "datasets": summary,
            "layer": args.layer,
            "anchor": args.anchor,
            "threshold": args.primary_threshold,
            "neutral_epsilon": args.neutral_epsilon,
        },
    )
    return summary


def dataset_complete(args: argparse.Namespace) -> bool:
    manifest = args.output_root / "dataset_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        value.get("materialization_version") != RUN_VERSION
        or int(value.get("layer", -1)) != args.layer
        or value.get("anchor") != args.anchor
        or not math.isclose(float(value.get("threshold", float("nan"))), args.primary_threshold, abs_tol=1e-12)
    ):
        return False
    return all(
        (args.output_root / "filter_inputs" / dataset / f"{split}.jsonl").is_file()
        and (args.output_root / "filter_inputs" / dataset / "manifest.json").is_file()
        for dataset in args.datasets
        for split in SPLITS
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    no_manifest, doc_manifest, layer_index, anchor_index = validate_args(args)
    question_total = sum(int(no_manifest["datasets"][dataset]) for dataset in args.datasets)
    pair_total = sum(int(doc_manifest["datasets"][dataset]) for dataset in args.datasets)
    score_plan: list[tuple[str, Path, Path, dict[str, Path], int]] = []
    score_completed = 0
    for dataset in args.datasets:
        trace_dirs = sorted((args.document_root / "trace_shards" / dataset / args.source_split).glob("shard_*"))
        for trace_dir in trace_dirs:
            feature_dir = args.document_root / "with_document_features" / dataset / args.source_split / "shards" / trace_dir.name
            marker = json.loads((feature_dir / "COMPLETE.json").read_text(encoding="utf-8"))
            count = int(marker["pair_count"])
            output = score_paths(args.output_root, dataset, args.source_split, trace_dir.name)
            if args.resume and score_complete(output, count, args):
                score_completed += count
            score_plan.append((dataset, trace_dir, feature_dir, output, count))
    if sum(item[-1] for item in score_plan) != pair_total:
        raise RuntimeError("Document shard totals do not match manifest")
    cached_dataset = args.resume and dataset_complete(args) and args.mode in {"all", "dataset"}
    stages = int(args.mode in {"all", "scores"}) + 2 * int(args.mode in {"all", "dataset"})
    overall_total = question_total + pair_total * stages
    overall_initial = (score_completed if args.mode in {"all", "scores"} else 0) + (
        2 * pair_total if cached_dataset else 0
    )
    logging.info(
        "Hidden utility preparation plan: questions=%d pairs=%d score_cached=%d mode=%s",
        question_total,
        pair_total,
        score_completed,
        args.mode,
    )
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = PipelineProgress(
        overall_total=overall_total,
        overall_initial=overall_initial,
        desc="ExtremeUtilityPrepare",
        enabled=args.show_progress,
    )
    progress.set_stage("1/4 index no-RAG questions", total=question_total)
    question_index = build_question_index(args, layer_index, anchor_index, progress)
    if len(question_index) != question_total:
        raise RuntimeError(f"Question index mismatch: {len(question_index)} != {question_total}")

    if args.mode in {"all", "scores"}:
        progress.set_stage("2/4 cached hD-h0 utility scoring", total=pair_total, initial=score_completed)
        cache = QuestionTensorCache(
            args.no_rag_root,
            args.direction_root,
            args.source_split,
            layer_index,
            anchor_index,
            args.tensor_cache_shards,
        )
        for dataset, trace_dir, feature_dir, output, count in score_plan:
            if args.resume and score_complete(output, count, args):
                continue
            output["root"].mkdir(parents=True, exist_ok=True)
            process_score_shard(
                args,
                dataset,
                trace_dir,
                feature_dir,
                output,
                question_index,
                cache,
                layer_index,
                anchor_index,
                progress,
            )
        atomic_json(
            args.output_root / "score_manifest.json",
            {
                "score_version": SCORE_VERSION,
                "created_at": utc_now(),
                "datasets": {dataset: int(doc_manifest["datasets"][dataset]) for dataset in args.datasets},
                "total_pairs": pair_total,
                "layer": args.layer,
                "anchor": args.anchor,
                "direction_root": str(args.direction_root.resolve()),
                "no_rag_root": str(args.no_rag_root.resolve()),
                "document_root": str(args.document_root.resolve()),
            },
        )

    if args.mode in {"all", "dataset"} and not cached_dataset:
        assignments = load_split_assignments(args)
        progress.set_stage("3/4 threshold and pair-coverage audit", total=pair_total)
        audit = audit_thresholds(args, assignments, pair_total, progress)
        progress.set_stage("4/4 materialize question-disjoint curriculum splits", total=pair_total)
        summary = materialize_dataset(args, assignments, audit, progress)
        logging.info("Curriculum dataset summary: %s", summary)
    elif cached_dataset:
        logging.info("Curriculum dataset already complete; retaining cached train/val/test materialization")
    progress.close()
    logging.info("Extreme utility preparation complete: %s", args.output_root)


if __name__ == "__main__":
    main()

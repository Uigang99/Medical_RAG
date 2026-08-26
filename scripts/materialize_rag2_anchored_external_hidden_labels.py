#!/usr/bin/env python3
"""Materialize three-class hidden-state oracle labels for the anchored MCQ test.

This script performs no language-model forward pass.  It joins the cached
Block/anchor states for the no-document and one-document anchored traces with
the cached gold-answer direction and computes

    utility = dot(h_D - h_0, c_gold).

Pairs above ``+tau`` are Helpful, pairs below ``-tau`` are Harmful, and the
closed interval between them is Neutral.  The existing RAG2 oracle labels are
used only as the authoritative candidate identity/cohort and for a diagnostic
cross-label table; they never affect the hidden label.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_anchored_external_hidden_oracle_labels_v1"
SCORE_VERSION = "rag2_anchored_external_hidden_utility_projection_v1"
TRACE_VERSION = "rag2_paper_compatible_three_anchor_v1"
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
CHOICES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rag-feature-root", type=Path, required=True)
    parser.add_argument("--document-feature-root", type=Path, required=True)
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--reference-rag2-labels-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--anchor", default="pre_choice")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--tensor-cache-shards", type=int, default=24)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
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
    with temporary.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hidden_label(score: float, threshold: float) -> str:
    if score > threshold:
        return "Helpful"
    if score < -threshold:
        return "Harmful"
    return "Neutral"


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
        split: str,
        layer_index: int,
        anchor_index: int,
        capacity: int,
    ) -> None:
        self.no_rag_root = no_rag_root
        self.direction_root = direction_root
        self.split = split
        self.layer_index = layer_index
        self.anchor_index = anchor_index
        self.capacity = max(1, capacity)
        self.cache: OrderedDict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()

    def _load(self, dataset: str, shard_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (dataset, shard_name)
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.cache[key] = cached
            return cached
        no_path = (
            self.no_rag_root
            / "no_rag_features"
            / dataset
            / self.split
            / "shards"
            / shard_name
            / "features.safetensors"
        )
        direction_path = (
            self.direction_root / dataset / self.split / "shards" / shard_name / "directions.safetensors"
        )
        with safe_open(no_path, framework="pt", device="cpu") as handle:
            h0 = handle.get_slice("anchor_hidden")[:, self.layer_index, self.anchor_index, :].float()
            logits = handle.get_tensor("choice_logits").float()
        with safe_open(direction_path, framework="pt", device="cpu") as handle:
            direction = handle.get_tensor("c_unit").float()
        if not (h0.shape[0] == logits.shape[0] == direction.shape[0]):
            raise RuntimeError(f"Question tensor mismatch: {dataset}/{shard_name}")
        value = (h0, direction, logits)
        self.cache[key] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def rows(self, refs: list[QuestionRef]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0_rows: list[torch.Tensor] = []
        direction_rows: list[torch.Tensor] = []
        logit_rows: list[torch.Tensor] = []
        for ref in refs:
            h0, direction, logits = self._load(ref.dataset, ref.shard_name)
            h0_rows.append(h0[ref.tensor_row])
            direction_rows.append(direction[ref.tensor_row])
            logit_rows.append(logits[ref.tensor_row])
        return torch.stack(h0_rows), torch.stack(direction_rows), torch.stack(logit_rows)


def validate_contracts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    if args.threshold <= 0 or not math.isfinite(args.threshold):
        raise ValueError("--threshold must be finite and positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets are not allowed")
    no_manifest = json.loads((args.no_rag_feature_root / "feature_manifest.json").read_text(encoding="utf-8"))
    doc_manifest = json.loads(
        (args.document_feature_root / "document_feature_manifest.json").read_text(encoding="utf-8")
    )
    direction_manifest = json.loads((args.direction_root / "direction_manifest.json").read_text(encoding="utf-8"))
    for manifest in (no_manifest, doc_manifest):
        if manifest.get("trace_version") != TRACE_VERSION:
            raise RuntimeError("No-RAG/document hidden features do not use the anchored trace contract")
    layers = [int(value) for value in no_manifest.get("layers") or []]
    anchors = [str(value) for value in no_manifest.get("anchor_order") or []]
    if args.layer not in layers or args.anchor not in anchors:
        raise RuntimeError(f"Requested layer/anchor absent: layers={layers} anchors={anchors}")
    if doc_manifest.get("layers") != no_manifest.get("layers") or doc_manifest.get("anchor_order") != anchors:
        raise RuntimeError("No-RAG and document hidden feature layouts differ")
    if int(direction_manifest.get("layer", -1)) != args.layer or direction_manifest.get("anchor") != args.anchor:
        raise RuntimeError("Gold direction layer/anchor differs from the hidden features")
    expected_questions = {dataset: int(no_manifest["datasets"][dataset]) for dataset in args.datasets}
    expected_pairs = {dataset: int(doc_manifest["datasets"][dataset]) for dataset in args.datasets}
    direction_questions = {dataset: int(direction_manifest["datasets"][dataset]) for dataset in args.datasets}
    if expected_questions != direction_questions:
        raise RuntimeError("No-RAG feature and gold-direction question counts differ")
    return no_manifest, doc_manifest, layers.index(args.layer), anchors.index(args.anchor)


def load_reference_labels(
    args: argparse.Namespace,
    expected_pairs: int,
    progress: PipelineProgress,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    allowed = set(args.datasets)
    progress.set_stage("1/4 index exact RAG2 oracle candidate identities", total=expected_pairs)
    for row in iter_jsonl(args.reference_rag2_labels_path):
        if row.get("dataset") not in allowed:
            continue
        pair_id = str(row.get("pair_id") or "")
        if not pair_id or pair_id in result:
            raise ValueError(f"Invalid/duplicate RAG2 oracle pair_id: {pair_id!r}")
        result[pair_id] = {
            key: row.get(key)
            for key in (
                "dataset",
                "sample_id",
                "sample_key",
                "row_idx",
                "doc_rank",
                "doc_stable_id",
                "source",
                "db_id",
                "local_id",
                "dynamic_top_k_membership",
                "dynamic_rerank_rank_by_top_k",
                "pseudo_label",
                "quality_pass",
            )
        }
        progress.update()
    if len(result) != expected_pairs:
        raise RuntimeError(f"RAG2 reference pair count mismatch: {len(result)} != {expected_pairs}")
    return result


def build_question_index(
    args: argparse.Namespace,
    expected_questions: int,
    progress: PipelineProgress,
) -> dict[str, QuestionRef]:
    result: dict[str, QuestionRef] = {}
    progress.set_stage("2/4 index no-RAG states and gold directions", total=expected_questions)
    for dataset in args.datasets:
        roots = sorted(
            (args.no_rag_feature_root / "no_rag_features" / dataset / args.split / "shards").glob("shard_*")
        )
        if not roots:
            raise FileNotFoundError(f"No no-RAG hidden shards for {dataset}")
        for root in roots:
            feature_rows = read_jsonl(root / "questions.jsonl")
            direction_root = args.direction_root / dataset / args.split / "shards" / root.name
            direction_rows = read_jsonl(direction_root / "questions.jsonl")
            if len(feature_rows) != len(direction_rows):
                raise RuntimeError(f"Question metadata mismatch: {dataset}/{root.name}")
            for local, (feature, direction) in enumerate(zip(feature_rows, direction_rows)):
                sample_id = str(feature.get("sample_id") or "")
                if (
                    not sample_id
                    or sample_id in result
                    or sample_id != direction.get("sample_id")
                    or int(feature.get("tensor_row", -1)) != local
                ):
                    raise RuntimeError(f"Question alignment failure: {dataset}/{root.name}:{local}")
                gold = str(feature.get("gold_answer") or "").upper()
                result[sample_id] = QuestionRef(
                    dataset=dataset,
                    shard_name=root.name,
                    tensor_row=local,
                    gold_index=CHOICES.index(gold),
                    no_rag_correct=bool(feature.get("hf_replay_correct")),
                )
                progress.update()
    if len(result) != expected_questions:
        raise RuntimeError(f"Question index count mismatch: {len(result)} != {expected_questions}")
    return result


def shard_output_paths(root: Path, dataset: str, split: str, shard_name: str) -> dict[str, Path]:
    base = root / "label_shards" / dataset / split / shard_name
    return {"root": base, "rows": base / "rows.jsonl", "complete": base / "COMPLETE.json"}


def shard_complete(paths: dict[str, Path], expected: int, args: argparse.Namespace, feature_size: int) -> bool:
    if not paths["rows"].is_file() or not paths["complete"].is_file():
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and int(marker.get("pair_count", -1)) == expected
        and int(marker.get("layer", -1)) == args.layer
        and marker.get("anchor") == args.anchor
        and math.isclose(float(marker.get("threshold", float("nan"))), args.threshold, abs_tol=1e-12)
        and int(marker.get("source_feature_size_bytes", -1)) == feature_size
    )


def process_document_shard(
    args: argparse.Namespace,
    dataset: str,
    feature_dir: Path,
    output: dict[str, Path],
    question_index: dict[str, QuestionRef],
    question_cache: QuestionTensorCache,
    reference: dict[str, dict[str, Any]],
    layer_index: int,
    anchor_index: int,
    progress: PipelineProgress,
) -> int:
    feature_rows = read_jsonl(feature_dir / "pairs.jsonl")
    refs: list[QuestionRef] = []
    identities: list[dict[str, Any]] = []
    for local, feature in enumerate(feature_rows):
        pair_id = str(feature.get("pair_id") or "")
        identity = reference.get(pair_id)
        ref = question_index.get(str(feature.get("sample_id") or ""))
        if identity is None or ref is None or int(feature.get("tensor_row", -1)) != local:
            raise RuntimeError(f"Document/reference alignment failure: {dataset}/{feature_dir.name}:{local}")
        if identity.get("dataset") != dataset or ref.dataset != dataset:
            raise RuntimeError(f"Dataset mismatch for {pair_id}")
        refs.append(ref)
        identities.append(identity)
    h0, directions, no_logits = question_cache.rows(refs)
    with safe_open(feature_dir / "features.safetensors", framework="pt", device="cpu") as handle:
        h_doc = handle.get_slice("anchor_hidden")[:, layer_index, anchor_index, :].float()
        doc_logits = handle.get_tensor("choice_logits").float()
    delta = h_doc - h0
    projection = torch.sum(delta * directions, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    cosine = projection / delta_norm.clamp_min(1e-12)
    gold = torch.tensor([ref.gold_index for ref in refs], dtype=torch.long)
    no_logprob = F.log_softmax(no_logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
    doc_logprob = F.log_softmax(doc_logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
    exact_delta = doc_logprob - no_logprob
    no_answers = torch.argmax(no_logits, dim=-1)
    doc_answers = torch.argmax(doc_logits, dim=-1)
    rows: list[dict[str, Any]] = []
    for local, (feature, ref, identity) in enumerate(zip(feature_rows, refs, identities)):
        score = float(projection[local].item())
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite hidden utility score: {feature['pair_id']}")
        label = hidden_label(score, args.threshold)
        no_correct = int(no_answers[local].item()) == ref.gold_index
        doc_correct = int(doc_answers[local].item()) == ref.gold_index
        rows.append(
            {
                "schema_version": 1,
                "policy": "anchored_pre_choice_gold_direction_hidden_utility_v1",
                **identity,
                "pair_id": feature["pair_id"],
                "projection_score": score,
                "hidden_label": label,
                "hidden_threshold": args.threshold,
                "layer": args.layer,
                "anchor": args.anchor,
                "delta_h_norm": float(delta_norm[local].item()),
                "delta_c_cosine": float(cosine[local].item()),
                "exact_gold_logprob_delta": float(exact_delta[local].item()),
                "no_doc_prediction": CHOICES[int(no_answers[local].item())],
                "with_doc_prediction": CHOICES[int(doc_answers[local].item())],
                "gold_answer": CHOICES[ref.gold_index],
                "no_doc_correct": no_correct,
                "with_doc_correct": doc_correct,
                "answer_transition": ("C" if no_correct else "W") + "->" + ("C" if doc_correct else "W"),
                "hidden_quality_pass": True,
                "source_feature_shard": feature_dir.name,
                "source_tensor_row": local,
            }
        )
    atomic_jsonl(output["rows"], rows)
    atomic_json(
        output["complete"],
        {
            "run_version": RUN_VERSION,
            "score_version": SCORE_VERSION,
            "completed_at": utc_now(),
            "dataset": dataset,
            "pair_count": len(rows),
            "label_counts": dict(Counter(row["hidden_label"] for row in rows)),
            "layer": args.layer,
            "anchor": args.anchor,
            "threshold": args.threshold,
            "source_feature_size_bytes": (feature_dir / "features.safetensors").stat().st_size,
        },
    )
    progress.update(len(rows))
    progress.set_detail(f"dataset={dataset} shard={feature_dir.name}")
    return len(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    no_manifest, doc_manifest, layer_index, anchor_index = validate_contracts(args)
    question_counts = {dataset: int(no_manifest["datasets"][dataset]) for dataset in args.datasets}
    pair_counts = {dataset: int(doc_manifest["datasets"][dataset]) for dataset in args.datasets}
    total_questions = sum(question_counts.values())
    total_pairs = sum(pair_counts.values())
    shard_plan: list[tuple[str, Path, dict[str, Path], int]] = []
    score_completed = 0
    for dataset in args.datasets:
        observed = 0
        roots = sorted(
            (args.document_feature_root / "with_document_features" / dataset / args.split / "shards").glob(
                "shard_*"
            )
        )
        if not roots:
            raise FileNotFoundError(f"No document hidden feature shards for {dataset}")
        for root in roots:
            marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
            count = int(marker["pair_count"])
            observed += count
            output = shard_output_paths(args.output_root, dataset, args.split, root.name)
            feature_size = (root / "features.safetensors").stat().st_size
            if args.resume and shard_complete(output, count, args, feature_size):
                score_completed += count
            shard_plan.append((dataset, root, output, count))
        if observed != pair_counts[dataset]:
            raise RuntimeError(f"Document feature count mismatch for {dataset}: {observed} != {pair_counts[dataset]}")

    progress = PipelineProgress(
        overall_total=total_questions + 3 * total_pairs,
        overall_initial=score_completed,
        desc="AnchoredHiddenOracleLabels",
        enabled=args.show_progress,
    )
    try:
        reference = load_reference_labels(args, total_pairs, progress)
        questions = build_question_index(args, total_questions, progress)
        cache = QuestionTensorCache(
            args.no_rag_feature_root,
            args.direction_root,
            args.split,
            layer_index,
            anchor_index,
            args.tensor_cache_shards,
        )
        progress.set_stage(
            "3/4 compute hD-h0 projection and three-class labels",
            total=total_pairs,
            initial=score_completed,
        )
        for dataset, feature_root, output, count in shard_plan:
            feature_size = (feature_root / "features.safetensors").stat().st_size
            if args.resume and shard_complete(output, count, args, feature_size):
                continue
            output["root"].mkdir(parents=True, exist_ok=True)
            written = process_document_shard(
                args,
                dataset,
                feature_root,
                output,
                questions,
                cache,
                reference,
                layer_index,
                anchor_index,
                progress,
            )
            if written != count:
                raise RuntimeError(f"Hidden-label shard count mismatch: {written} != {count}")

        progress.set_stage("4/4 merge labels and audit RAG2 agreement", total=total_pairs)
        args.output_root.mkdir(parents=True, exist_ok=True)
        combined_path = args.output_root / "hidden_oracle_labels.jsonl"
        combined_partial = combined_path.with_name(combined_path.name + ".partial")
        dataset_handles: dict[str, Any] = {}
        dataset_paths: dict[str, Path] = {}
        for dataset in args.datasets:
            path = args.output_root / dataset / args.split / "hidden_oracle_labels.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset_paths[dataset] = path
            dataset_handles[dataset] = path.with_name(path.name + ".partial").open(
                "w", encoding="utf-8", buffering=16 * 1024 * 1024
            )
        label_counts = {dataset: Counter() for dataset in args.datasets}
        cross_counts = {dataset: Counter() for dataset in args.datasets}
        seen: set[str] = set()
        with combined_partial.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as combined:
            for dataset, _feature_root, output, _count in shard_plan:
                for row in iter_jsonl(output["rows"]):
                    pair_id = str(row["pair_id"])
                    if pair_id in seen:
                        raise RuntimeError(f"Duplicate hidden label during merge: {pair_id}")
                    seen.add(pair_id)
                    serialized = json.dumps(row, ensure_ascii=False) + "\n"
                    combined.write(serialized)
                    dataset_handles[dataset].write(serialized)
                    label_counts[dataset][row["hidden_label"]] += 1
                    cross_counts[dataset][f"{row.get('pseudo_label')}|{row['hidden_label']}"] += 1
                    progress.update()
            combined.flush()
            os.fsync(combined.fileno())
        for dataset, handle in dataset_handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(dataset_paths[dataset].with_name(dataset_paths[dataset].name + ".partial"), dataset_paths[dataset])
        os.replace(combined_partial, combined_path)
        if len(seen) != total_pairs:
            raise RuntimeError(f"Merged hidden-label count mismatch: {len(seen)} != {total_pairs}")
        atomic_json(
            args.output_root / "manifest.json",
            {
                "type": RUN_VERSION,
                "created_at": utc_now(),
                "questions": total_questions,
                "pairs": total_pairs,
                "questions_by_dataset": question_counts,
                "pairs_by_dataset": pair_counts,
                "label_counts": {dataset: dict(label_counts[dataset]) for dataset in args.datasets},
                "rag2_hidden_cross_counts": {
                    dataset: dict(cross_counts[dataset]) for dataset in args.datasets
                },
                "threshold": args.threshold,
                "layer": args.layer,
                "anchor": args.anchor,
                "score_definition": "dot(hD-h0, unit negative gold-answer CE gradient at h0)",
                "oracle_action": "pass Helpful only; block Neutral and Harmful; zero Helpful falls back to no-RAG",
                "no_rag_feature_root": str(args.no_rag_feature_root.resolve()),
                "document_feature_root": str(args.document_feature_root.resolve()),
                "direction_root": str(args.direction_root.resolve()),
                "reference_rag2_labels_path": str(args.reference_rag2_labels_path.resolve()),
                "combined_labels_path": str(combined_path.resolve()),
            },
        )
        logging.info("Anchored external hidden oracle labels complete: %s", args.output_root)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

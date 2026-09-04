#!/usr/bin/env python3
"""Validate a train-time document mask against physical token deletion.

This is a bounded, no-training mechanism test.  On mixed Semantic Top-8
questions it measures each document's influence as the full-vocabulary JSD
between the frozen Llama Direct-Choice distribution with all documents and
the distribution after one document intervention.  It then compares:

* reference: physically delete the mapped document tokens;
* proxy: keep sequence length fixed, block the same document at every layer,
  and compact the remaining position IDs as if those tokens were deleted.

Gold answers are not used.  Semantic labels are used only to require mixed
Support/non-support questions and to report diagnostic subgroups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_direct_choice_document_mask_validity_v1"
INFLUENCE_VERSION = "full_vocabulary_jsd_from_top8_direct_choice_distribution_v1"
PHYSICAL_VERSION = "delete_exact_mapped_document_tokens_from_full_token_sequence_v1"
MASK_VERSION = "all_layer_exact_kv_block_with_compact_positions_v1"
SUPPORT_LABELS = frozenset({"direct_support", "supporting_evidence"})
NON_SUPPORT_LABELS = frozenset({"no_evidence", "misleading_evidence"})

DEFAULT_BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
DEFAULT_COHORT = (
    DEFAULT_BASE
    / "document_attribution_faithfulness_mvp_v1"
    / "medqa_train_rationale_answer_gradxinput_mixed256_non64_v1"
    / "cohort.jsonl"
)
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_OUTPUT = (
    DEFAULT_BASE
    / "direct_choice_document_mask_validity_v1"
    / "medqa_train_mixed256_question_first_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-file", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-questions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--spearman-threshold", type=float, default=0.80)
    parser.add_argument("--top1-threshold", type=float, default=0.70)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def model_identity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    tokenizer = path / "tokenizer_config.json"
    shards = sorted(path.glob("*.safetensors"))
    if not config.is_file() or not tokenizer.is_file() or not shards:
        raise FileNotFoundError(f"Incomplete local model: {path}")
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256_file(config),
        "tokenizer_config_sha256": sha256_file(tokenizer),
        "weight_files": [
            {"name": item.name, "size_bytes": item.stat().st_size}
            for item in shards
        ],
    }


def select_mixed_rows(path: Path, maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ValueError("--max-questions must be positive")
    rows = [row for row in iter_jsonl(path) if str(row.get("cohort")) == "mixed"]
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('sample_id', '')}".encode("utf-8")
        ).hexdigest()
    )
    selected = rows[:maximum]
    if len(selected) < maximum:
        raise RuntimeError(f"Requested {maximum} mixed questions but found {len(selected)}")
    seen: set[str] = set()
    for row in selected:
        sample_id = str(row.get("sample_id") or "")
        documents = list(row.get("documents") or [])
        labels = [str(value) for value in row.get("semantic_labels") or []]
        if not sample_id or sample_id in seen:
            raise RuntimeError(f"Missing or duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if len(documents) != 8 or len(labels) != 8:
            raise RuntimeError(f"Expected exactly eight documents and labels: {sample_id}")
        if not any(label in SUPPORT_LABELS for label in labels):
            raise RuntimeError(f"Mixed question has no Support document: {sample_id}")
        if not any(label in NON_SUPPORT_LABELS for label in labels):
            raise RuntimeError(f"Mixed question has no non-support document: {sample_id}")
        if any(label not in SUPPORT_LABELS | NON_SUPPORT_LABELS for label in labels):
            raise RuntimeError(f"Unsupported Semantic label in mixed question: {sample_id}")
        if [int(document.get("rerank_rank", -1)) for document in documents] != list(range(1, 9)):
            raise RuntimeError(f"Invalid rerank order: {sample_id}")
        if any(not str(document.get("text") or "").strip() for document in documents):
            raise RuntimeError(f"Empty document text: {sample_id}")
    return selected


def build_token_document_ids(sequence: dict[str, Any]) -> torch.Tensor:
    input_ids = list(sequence["input_ids"])
    mapping = torch.full((len(input_ids),), -1, dtype=torch.long)
    occupied: set[int] = set()
    for document_index, positions in enumerate(sequence["document_token_indices"]):
        if not positions:
            raise RuntimeError(f"Document {document_index + 1} has no mapped tokens")
        overlap = occupied.intersection(int(position) for position in positions)
        if overlap:
            raise RuntimeError(f"Document token spans overlap at {sorted(overlap)[:3]}")
        for position in positions:
            mapping[int(position)] = document_index
            occupied.add(int(position))
    return mapping


def build_physical_deletion_batch(
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Delete exactly the document tokens mapped for the hard-mask proxy."""

    if input_ids.ndim != 1 or token_document_ids.shape != input_ids.shape:
        raise ValueError("input_ids and token_document_ids must be aligned vectors")
    document_count = int(token_document_ids.max().item()) + 1
    if document_count <= 0:
        raise ValueError("No mapped documents")
    sequences: list[torch.Tensor] = []
    for document_index in range(document_count):
        kept = token_document_ids.ne(document_index)
        if int((~kept).sum().item()) <= 0:
            raise RuntimeError(f"Document {document_index + 1} has no removable tokens")
        sequences.append(input_ids[kept])
    maximum = max(int(sequence.numel()) for sequence in sequences)
    ids = torch.full((document_count, maximum), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros_like(ids)
    for row_index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        ids[row_index, -length:] = sequence
        attention[row_index, -length:] = 1
    positions = attention.cumsum(dim=1) - 1
    positions.masked_fill_(attention.eq(0), 0)
    return {"input_ids": ids, "attention_mask": attention, "position_ids": positions}


def build_compact_mask_batch(
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Block one document per row and compact all remaining position IDs."""

    if input_ids.ndim != 1 or token_document_ids.shape != input_ids.shape:
        raise ValueError("input_ids and token_document_ids must be aligned vectors")
    document_count = int(token_document_ids.max().item()) + 1
    blocked = torch.arange(document_count, dtype=torch.long)
    ids = input_ids.unsqueeze(0).expand(document_count, -1).clone()
    mapping = token_document_ids.unsqueeze(0).expand(document_count, -1).clone()
    blocked_keys = mapping.eq(blocked.unsqueeze(1))
    if bool((blocked_keys.sum(dim=1) <= 0).any()):
        raise RuntimeError("Every document must map to at least one blocked token")
    kept = ~blocked_keys
    positions = kept.long().cumsum(dim=1) - 1
    positions.clamp_min_(0)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": positions,
        "token_document_ids": mapping,
        "blocked_document_ids": blocked,
    }


def jsd_from_logits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Base-2 Jensen-Shannon divergence along the final axis."""

    left, right = torch.broadcast_tensors(left.float(), right.float())
    log_left = torch.log_softmax(left, dim=-1)
    log_right = torch.log_softmax(right, dim=-1)
    log_middle = torch.logaddexp(log_left, log_right) - math.log(2.0)
    divergence = 0.5 * (
        (log_left.exp() * (log_left - log_middle)).sum(dim=-1)
        + (log_right.exp() * (log_right - log_middle)).sum(dim=-1)
    ) / math.log(2.0)
    return divergence.clamp_min(0.0)


@torch.inference_mode()
def plain_logits(model: Any, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    logits = outputs.logits[:, -1].float().cpu()
    del outputs
    return logits


@torch.inference_mode()
def masked_logits(model: Any, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
        semantic_token_document_ids=batch["token_document_ids"].to(device),
        semantic_blocked_document_ids=batch["blocked_document_ids"].to(device),
        semantic_document_block_layer_start=0,
    )
    logits = outputs.logits[:, -1].float().cpu()
    del outputs
    return logits


def top_index(values: Sequence[float]) -> int:
    return int(np.asarray(values, dtype=np.float64).argmax())


def summarize(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    correlations = [spearman(row["physical_influence_jsd"], row["mask_influence_jsd"]) for row in rows]
    defined = [float(value) for value in correlations if value is not None and math.isfinite(value)]
    top1 = [
        float(top_index(row["physical_influence_jsd"]) == top_index(row["mask_influence_jsd"]))
        for row in rows
    ]
    equivalence = [value for row in rows for value in row["physical_vs_mask_jsd"]]
    choice_agreements = [value for row in rows for value in row["physical_vs_mask_choice_agreement"]]
    full_argmax_agreements = [value for row in rows for value in row["physical_vs_mask_full_argmax_agreement"]]
    physical = [value for row in rows for value in row["physical_influence_jsd"]]
    masked = [value for row in rows for value in row["mask_influence_jsd"]]
    low_signal = {
        threshold: float(np.mean([max(row["physical_influence_jsd"]) < threshold for row in rows]))
        for threshold in (1e-6, 1e-5, 1e-4, 1e-3)
    }
    mean_spearman = float(np.mean(defined)) if defined else None
    median_spearman = float(np.median(defined)) if defined else None
    top1_overlap = float(np.mean(top1)) if top1 else None
    pass_spearman = mean_spearman is not None and mean_spearman >= args.spearman_threshold
    pass_top1 = top1_overlap is not None and top1_overlap >= args.top1_threshold
    return {
        "questions": len(rows),
        "documents": len(physical),
        "primary": {
            "defined_spearman_questions": len(defined),
            "undefined_spearman_questions": len(rows) - len(defined),
            "mean_within_question_spearman": mean_spearman,
            "median_within_question_spearman": median_spearman,
            "top1_document_overlap": top1_overlap,
        },
        "intervention_equivalence": {
            "mean_full_vocabulary_jsd_physical_vs_mask": float(np.mean(equivalence)),
            "median_full_vocabulary_jsd_physical_vs_mask": float(np.median(equivalence)),
            "constrained_choice_agreement": float(np.mean(choice_agreements)),
            "full_vocabulary_argmax_agreement": float(np.mean(full_argmax_agreements)),
        },
        "signal": {
            "mean_physical_influence_jsd": float(np.mean(physical)),
            "mean_mask_influence_jsd": float(np.mean(masked)),
            "low_signal_question_fraction": {f"max_jsd_below_{threshold:g}": value for threshold, value in low_signal.items()},
        },
        "pre_registered_decision": {
            "mean_spearman_threshold": args.spearman_threshold,
            "top1_overlap_threshold": args.top1_threshold,
            "spearman_pass": pass_spearman,
            "top1_pass": pass_top1,
            "overall_pass": bool(pass_spearman and pass_top1),
        },
    }


def bootstrap_ci(
    rows: Sequence[dict[str, Any]], replicates: int, seed: int,
    progress: HierarchicalProgress,
) -> dict[str, list[float] | None]:
    if replicates <= 0 or len(rows) < 2:
        return {"mean_within_question_spearman": None, "top1_document_overlap": None}
    per_question_spearman = [
        spearman(row["physical_influence_jsd"], row["mask_influence_jsd"])
        for row in rows
    ]
    per_question_top1 = [
        float(top_index(row["physical_influence_jsd"]) == top_index(row["mask_influence_jsd"]))
        for row in rows
    ]
    rng = np.random.default_rng(seed)
    sampled_spearman: list[float] = []
    sampled_top1: list[float] = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(rows), size=len(rows))
        correlations = [
            float(per_question_spearman[int(index)])
            for index in indices
            if per_question_spearman[int(index)] is not None
        ]
        if correlations:
            sampled_spearman.append(float(np.mean(correlations)))
        sampled_top1.append(float(np.mean([per_question_top1[int(index)] for index in indices])))
        if (replicate + 1) % 25 == 0 or replicate + 1 == replicates:
            progress.set(replicate + 1)
    def interval(values: Sequence[float]) -> list[float] | None:
        if not values:
            return None
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "mean_within_question_spearman": interval(sampled_spearman),
        "top1_document_overlap": interval(sampled_top1),
    }


def render_report(summary: dict[str, Any]) -> str:
    primary = summary["primary"]
    equivalent = summary["intervention_equivalence"]
    decision = summary["pre_registered_decision"]
    confidence = summary.get("bootstrap_95pct_ci", {})
    def number(value: Any, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"
    def interval(value: Any) -> str:
        return "NA" if value is None else f"[{number(value[0])}, {number(value[1])}]"
    return "\n".join([
        "# Direct-Choice document-mask validity",
        "",
        f"- Questions: {summary['questions']}",
        f"- Documents: {summary['documents']}",
        "- Target: full next-token vocabulary distribution; gold answer is not used",
        "- Reference: physical deletion of exactly the mapped document tokens",
        "- Proxy: all-layer hard key/value mask with compact position IDs",
        "",
        "| Primary metric | Value | 95% bootstrap CI | Pass threshold |",
        "|---|---:|---:|---:|",
        f"| Mean within-question Spearman | {number(primary['mean_within_question_spearman'])} | "
        f"{interval(confidence.get('mean_within_question_spearman'))} | ≥ {number(decision['mean_spearman_threshold'], 2)} |",
        f"| Top-1 document overlap | {number(primary['top1_document_overlap'])} | "
        f"{interval(confidence.get('top1_document_overlap'))} | ≥ {number(decision['top1_overlap_threshold'], 2)} |",
        "",
        "| Secondary intervention-equivalence metric | Value |",
        "|---|---:|",
        f"| Mean full-vocabulary JSD, physical vs mask | {number(equivalent['mean_full_vocabulary_jsd_physical_vs_mask'], 6)} |",
        f"| Constrained A/B/C/D prediction agreement | {number(equivalent['constrained_choice_agreement'])} |",
        f"| Full-vocabulary argmax agreement | {number(equivalent['full_vocabulary_argmax_agreement'])} |",
        "",
        "## Decision",
        "",
        f"- Spearman criterion passed: {decision['spearman_pass']}",
        f"- Top-1 criterion passed: {decision['top1_pass']}",
        f"- **Overall passed: {decision['overall_pass']}**",
        "- Passing only establishes that this hard mask is a usable deletion proxy. It does not establish learnability or continuous contribution control.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if args.max_questions <= 0 or args.bootstrap_replicates < 0:
        raise ValueError("Invalid question or bootstrap count")
    if not 0 <= args.spearman_threshold <= 1 or not 0 <= args.top1_threshold <= 1:
        raise ValueError("Pass thresholds must be in [0, 1]")
    if not args.cohort_file.is_file() or not args.model.is_dir():
        raise FileNotFoundError(args.cohort_file if not args.cohort_file.is_file() else args.model)

    stages = [
        "preflight mixed Top-8 contract",
        "physical deletion vs all-layer hard mask",
        "bootstrap confidence intervals and report",
    ]
    progress = HierarchicalProgress(stages, [8.0, 1200.0, 8.0])
    try:
        progress.start(1, args.max_questions, "question")
        rows = select_mixed_rows(args.cohort_file, args.max_questions, args.seed)
        selected_ids = [str(row["sample_id"]) for row in rows]
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for exact document token spans")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        choice_token_ids(tokenizer, torch.device("cpu"))
        maximum_tokens = 0
        for row in rows:
            sequence = direct_sequence(tokenizer, row, list(row["documents"]))
            build_token_document_ids(sequence)
            maximum_tokens = max(maximum_tokens, len(sequence["input_ids"]))
            if len(sequence["input_ids"]) > args.max_input_tokens:
                raise RuntimeError(
                    f"Input exceeds --max-input-tokens: {row['sample_id']} "
                    f"tokens={len(sequence['input_ids'])} limit={args.max_input_tokens}"
                )
            progress.update()
        free_bytes = shutil.disk_usage(args.output_dir.parent if args.output_dir.parent.exists() else PROJECT_ROOT).free
        if free_bytes < 1024 ** 3:
            raise RuntimeError(f"Less than 1 GiB free disk space: {free_bytes / 1024**3:.2f} GiB")

        contract = {
            "run_version": RUN_VERSION,
            "created_at": utc_now(),
            "hypothesis": "all-layer hard document masking preserves physical-deletion document influence order",
            "primary_metrics": ["mean_within_question_spearman", "top1_document_overlap"],
            "pass_thresholds": {
                "mean_within_question_spearman": args.spearman_threshold,
                "top1_document_overlap": args.top1_threshold,
            },
            "cohort": {
                "path": str(args.cohort_file.resolve()),
                "sha256": sha256_file(args.cohort_file),
                "selection": "cohort=mixed; seeded sample_id hash order",
                "seed": args.seed,
                "selected_ids": selected_ids,
            },
            "model": model_identity(args.model),
            "prompt": "anchored_direct_choice_question_first_without_rationale",
            "influence_version": INFLUENCE_VERSION,
            "physical_intervention": PHYSICAL_VERSION,
            "mask_intervention": MASK_VERSION,
            "max_input_tokens": args.max_input_tokens,
            "dtype": args.dtype,
            "bootstrap_replicates": args.bootstrap_replicates,
            "code_commit": git_commit(),
        }
        fingerprint_payload = {key: value for key, value in contract.items() if key != "created_at"}
        contract["contract_fingerprint"] = canonical_hash(fingerprint_payload)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract_fingerprint") != contract["contract_fingerprint"]:
                raise RuntimeError("Resume contract mismatch; use a new OUTPUT_DIR")
        else:
            atomic_json(manifest_path, contract)
        progress.complete(
            f"selected={len(rows)} documents={len(rows)*8} max_tokens={maximum_tokens} "
            f"free_disk={free_bytes/1024**3:.1f}GiB "
            f"manifest={manifest_path}"
        )
        if args.preflight_only:
            progress.finish(f"preflight-only passed; output={args.output_dir}")
            return

        row_dir = args.output_dir / "question_rows"
        row_dir.mkdir(parents=True, exist_ok=True)
        cached: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row["sample_id"])
            row_path = row_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"
            if not args.resume or not row_path.is_file():
                continue
            value = json.loads(row_path.read_text(encoding="utf-8"))
            if value.get("contract_fingerprint") == contract["contract_fingerprint"] and value.get("sample_id") == sample_id:
                cached[sample_id] = value

        progress.start(2, len(rows), "question", initial=len(cached))
        if len(cached) < len(rows):
            attention_name = register_semantic_attention()
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            device = torch.device(args.device)
            print(
                f"[stage 2/{len(stages)} model load] model={args.model} device={device} dtype={args.dtype} "
                f"attention={attention_name}", flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                attn_implementation=attention_name,
                local_files_only=True,
                low_cpu_mem_usage=True,
            ).to(device).eval()
            choice_ids = choice_token_ids(tokenizer, torch.device("cpu"))
            for row in rows:
                sample_id = str(row["sample_id"])
                if sample_id in cached:
                    continue
                sequence = direct_sequence(tokenizer, row, list(row["documents"]))
                if len(sequence["input_ids"]) > args.max_input_tokens:
                    raise RuntimeError(
                        f"Input exceeds --max-input-tokens: {sample_id} "
                        f"tokens={len(sequence['input_ids'])} limit={args.max_input_tokens}"
                    )
                ids = torch.tensor(sequence["input_ids"], dtype=torch.long)
                mapping = build_token_document_ids(sequence)
                full_batch = {
                    "input_ids": ids.unsqueeze(0),
                    "attention_mask": torch.ones((1, ids.numel()), dtype=torch.long),
                    "position_ids": torch.arange(ids.numel(), dtype=torch.long).unsqueeze(0),
                }
                physical_batch = build_physical_deletion_batch(ids, mapping, int(tokenizer.pad_token_id))
                mask_batch = build_compact_mask_batch(ids, mapping)
                full = plain_logits(model, full_batch, device)[0]
                physical = plain_logits(model, physical_batch, device)
                masked = masked_logits(model, mask_batch, device)
                physical_influence = jsd_from_logits(full.unsqueeze(0), physical)
                mask_influence = jsd_from_logits(full.unsqueeze(0), masked)
                physical_vs_mask = jsd_from_logits(physical, masked)
                full_choice = full.index_select(0, choice_ids)
                physical_choice = physical.index_select(1, choice_ids)
                mask_choice = masked.index_select(1, choice_ids)
                result = {
                    "run_version": RUN_VERSION,
                    "contract_fingerprint": contract["contract_fingerprint"],
                    "sample_id": sample_id,
                    "semantic_labels": list(row["semantic_labels"]),
                    "document_pair_ids": [str(document.get("pair_id") or "") for document in row["documents"]],
                    "document_token_counts": list(sequence["document_token_counts"]),
                    "input_tokens": len(sequence["input_ids"]),
                    "full_constrained_choice": CHOICES[int(full_choice.argmax().item())],
                    "physical_influence_jsd": [float(value) for value in physical_influence.tolist()],
                    "mask_influence_jsd": [float(value) for value in mask_influence.tolist()],
                    "physical_vs_mask_jsd": [float(value) for value in physical_vs_mask.tolist()],
                    "physical_constrained_choices": [CHOICES[int(index)] for index in physical_choice.argmax(dim=1).tolist()],
                    "mask_constrained_choices": [CHOICES[int(index)] for index in mask_choice.argmax(dim=1).tolist()],
                    "physical_vs_mask_choice_agreement": [
                        float(left == right)
                        for left, right in zip(physical_choice.argmax(dim=1).tolist(), mask_choice.argmax(dim=1).tolist())
                    ],
                    "physical_vs_mask_full_argmax_agreement": [
                        float(left == right)
                        for left, right in zip(physical.argmax(dim=1).tolist(), masked.argmax(dim=1).tolist())
                    ],
                    "within_question_spearman": spearman(physical_influence.tolist(), mask_influence.tolist()),
                    "top1_document_agreement": float(
                        int(physical_influence.argmax().item()) == int(mask_influence.argmax().item())
                    ),
                }
                row_path = row_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"
                atomic_json(row_path, result)
                cached[sample_id] = result
                progress.update()
                del full, physical, masked, physical_influence, mask_influence, physical_vs_mask
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ordered = [cached[str(row["sample_id"])] for row in rows]
        atomic_jsonl(args.output_dir / "per_question.jsonl", ordered)
        progress.complete(
            f"questions={len(ordered)} interventions={len(ordered)*16} durable_rows={row_dir}"
        )

        progress.start(3, max(1, args.bootstrap_replicates), "replicate")
        summary = summarize(ordered, args)
        confidence = bootstrap_ci(ordered, args.bootstrap_replicates, args.seed, progress)
        summary.update({
            "run_version": RUN_VERSION,
            "contract_fingerprint": contract["contract_fingerprint"],
            "bootstrap_95pct_ci": confidence,
            "scope_limit": (
                "This validates only a hard document-exclusion proxy for the frozen Llama "
                "Direct-Choice next-token distribution. It does not validate continuous gates, "
                "gold correctness, causal reasoning, or held-out learnability."
            ),
        })
        atomic_json(args.output_dir / "summary.json", summary)
        atomic_text(args.output_dir / "report.md", render_report(summary))
        progress.complete(
            f"pass={summary['pre_registered_decision']['overall_pass']} "
            f"summary={args.output_dir/'summary.json'}"
        )
        progress.finish(
            f"overall_pass={summary['pre_registered_decision']['overall_pass']} output={args.output_dir}"
        )
    except Exception:
        print(
            f"[workflow FAILED] rerun the identical command to resume durable question rows; "
            f"output={args.output_dir}", flush=True,
        )
        raise


if __name__ == "__main__":
    main()

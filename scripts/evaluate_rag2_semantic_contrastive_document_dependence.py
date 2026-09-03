#!/usr/bin/env python3
"""Audit whether a semantic-contrastive LoRA uses question-matched evidence.

The held-out cohort contains questions with both a Direct Support document and
a No Evidence document.  The frozen base model and the trained adapter are
scored under five otherwise identical direct-choice inputs:

* no_rag: no document;
* real_support: the current question's highest-ranked Direct Support document;
* same_question_no_evidence: the current question's highest-ranked No Evidence document;
* shuffled_support: another question's Direct Support document, source matched;
* dummy_evidence: one fixed content-free control document.

Gold answers are used only after the forward pass.  The primary diagnostic is
the change from the frozen model to the adapter in the paired
real_support-minus-shuffled_support accuracy contrast.  This distinguishes
question-document use from generic task/domain learning.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import (  # noqa: E402
    PROMPT_POLICY_VERSION,
    sequence_for_prompt,
)
from medrag.core import BenchmarkSample  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_semantic_contrastive_document_dependence_v1"
CONDITIONS = (
    "no_rag",
    "real_support",
    "same_question_no_evidence",
    "shuffled_support",
    "dummy_evidence",
)
DEFAULT_PAIR_FILE = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "direct_semantic_mismatch_pilot_pairs_v1/medmcqa/test.jsonl"
)
DEFAULT_BASE_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_ADAPTER = (
    WORKSPACE_ROOT
    / "models/RAG2-Direct-Semantic-Contrastive-LoRA/medmcqa"
    / "medmcqa_pilot_direct_semantic_contrastive_v3/final_model"
)
DEFAULT_OUTPUT = DEFAULT_ADAPTER.parent / "document_dependence_eval_v1"
DEFAULT_DUMMY = (
    "This document contains no information relevant to the medical question "
    "or its answer choices."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR_FILE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-questions", type=int, default=2054)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--questions-per-shard", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy-evidence", default=DEFAULT_DUMMY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--expected-model-phase-seconds", type=float, default=480.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_identity(path: Path) -> dict[str, Any]:
    files = []
    for child in sorted(path.glob("*")):
        if child.is_file():
            stat = child.stat()
            files.append({"name": child.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"path": str(path.resolve()), "files": files}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def make_sample(row: dict[str, Any]) -> BenchmarkSample:
    return BenchmarkSample(
        id=str(row["sample_id"]), row_idx=int(row["row_idx"]),
        task=str(row["dataset"]), collection="unified", dataset=str(row["dataset"]),
        split=str(row["split"]), question=str(row["question"]), options=dict(row["options"]),
        answer=str(row["gold_answer"]), answers=[str(row["gold_answer"])], raw=dict(row),
    )


def canonical_question_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"], row["split"], int(row["row_idx"]), row["question"],
        tuple(sorted(dict(row["options"]).items())), row["gold_answer"],
        bool(row["frozen_no_rag_correct"]), row["frozen_no_rag_prediction"],
        tuple(float(x) for x in row["frozen_no_rag_probabilities"]),
    )


def select_highest_rank(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any] | None:
    eligible = [row for row in rows if str(row["semantic_label"]) == label]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (int(row["rerank_rank"]), str(row["pair_id"])))


def source_matched_derangement(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map every recipient to another question's support, preserving source.

    Within each source, a cyclic shift is selected to minimize log-length
    mismatch while prohibiting identical question or document IDs.
    """
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["direct"]["document_source"])].append(record)
    mapping: dict[str, dict[str, Any]] = {}
    for source, values in sorted(by_source.items()):
        ordered = sorted(
            values,
            key=lambda value: (len(str(value["direct"]["document_text"])), str(value["sample_id"])),
        )
        if len(ordered) < 2:
            raise RuntimeError(f"Cannot source-match shuffled support for singleton source: {source}")
        best: tuple[float, int] | None = None
        for offset in range(1, len(ordered)):
            cost = 0.0
            valid = True
            for index, recipient in enumerate(ordered):
                donor = ordered[(index + offset) % len(ordered)]
                if donor["sample_id"] == recipient["sample_id"]:
                    valid = False
                    break
                if donor["direct"]["document_stable_id"] == recipient["direct"]["document_stable_id"]:
                    valid = False
                    break
                left = max(1, len(str(recipient["direct"]["document_text"])))
                right = max(1, len(str(donor["direct"]["document_text"])))
                cost += abs(math.log(left) - math.log(right))
            if valid and (best is None or cost < best[0]):
                best = (cost, offset)
        if best is None:
            raise RuntimeError(f"No valid source-matched support derangement for source: {source}")
        offset = best[1]
        for index, recipient in enumerate(ordered):
            mapping[str(recipient["sample_id"])] = ordered[(index + offset) % len(ordered)]
    if len(mapping) != len(records):
        raise RuntimeError("Incomplete shuffled-support mapping")
    return mapping


def load_cohort(path: Path, expected: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_ids: set[str] = set()
    row_count = 0
    for row in iter_jsonl(path):
        row_count += 1
        pair_id = str(row["pair_id"])
        if pair_id in pair_ids:
            raise RuntimeError(f"Duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        grouped[str(row["sample_id"])].append(row)
    records = []
    for sample_id, rows in grouped.items():
        reference = canonical_question_key(rows[0])
        if any(canonical_question_key(row) != reference for row in rows[1:]):
            raise RuntimeError(f"Inconsistent question fields: {sample_id}")
        direct = select_highest_rank(rows, "direct_support")
        no_evidence = select_highest_rank(rows, "no_evidence")
        if direct is None or no_evidence is None:
            continue
        records.append({
            "sample_id": sample_id,
            "question": rows[0],
            "direct": direct,
            "no_evidence": no_evidence,
        })
    records.sort(key=lambda value: (int(value["question"]["row_idx"]), value["sample_id"]))
    if expected > 0 and len(records) != expected:
        raise RuntimeError(f"Cohort count mismatch: expected={expected} actual={len(records)}")
    shuffled = source_matched_derangement(records)
    stats = {
        "source_rows": row_count,
        "source_questions": len(grouped),
        "selected_questions": len(records),
        "no_rag_correct": sum(bool(value["question"]["frozen_no_rag_correct"]) for value in records),
        "no_rag_wrong": sum(not bool(value["question"]["frozen_no_rag_correct"]) for value in records),
        "direct_source": dict(Counter(str(value["direct"]["document_source"]) for value in records)),
        "mean_direct_chars": float(np.mean([len(str(value["direct"]["document_text"])) for value in records])),
        "mean_shuffled_chars": float(np.mean([
            len(str(shuffled[value["sample_id"]]["direct"]["document_text"])) for value in records
        ])),
    }
    return records, shuffled, stats


def condition_document(
    record: dict[str, Any], shuffled: dict[str, dict[str, Any]], condition: str, dummy: str
) -> tuple[str | None, dict[str, Any]]:
    if condition == "no_rag":
        return None, {}
    if condition == "real_support":
        row = record["direct"]
        return str(row["document_text"]), {"pair_id": row["pair_id"], "source": row["document_source"]}
    if condition == "same_question_no_evidence":
        row = record["no_evidence"]
        return str(row["document_text"]), {"pair_id": row["pair_id"], "source": row["document_source"]}
    if condition == "shuffled_support":
        donor = shuffled[record["sample_id"]]
        row = donor["direct"]
        return str(row["document_text"]), {
            "pair_id": row["pair_id"], "source": row["document_source"],
            "donor_sample_id": donor["sample_id"],
        }
    if condition == "dummy_evidence":
        return dummy, {"source": "fixed_dummy"}
    raise ValueError(condition)


def pad_sequences(sequences: Sequence[Sequence[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(map(len, sequences))
    ids = torch.full((len(sequences), maximum), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        ids[index, maximum - length:] = torch.tensor(sequence, dtype=torch.long)
        mask[index, maximum - length:] = 1
    positions = mask.cumsum(dim=-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def choice_token_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    values = []
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice is not a single token: {choice} -> {ids}")
        values.append(ids[0])
    return torch.tensor(values, dtype=torch.long, device=device)


def phase_shard_path(output: Path, model_kind: str, index: int) -> Path:
    return output / "score_shards" / model_kind / f"shard_{index:05d}.jsonl"


def load_valid_shard(path: Path, expected_keys: set[tuple[str, str]]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        rows = list(iter_jsonl(path))
    except Exception:
        return None
    keys = {(str(row["sample_id"]), str(row["condition"])) for row in rows}
    if keys != expected_keys or len(rows) != len(expected_keys):
        return None
    return rows


def progress_bar(
    *, total: int, initial: int, description: str, workflow_started: float,
    future_seconds: float,
) -> tqdm:
    return tqdm(total=total, initial=initial, desc=description, unit="prompt", dynamic_ncols=True)


@torch.inference_mode()
def score_model_phase(
    *, args: argparse.Namespace, model: Any, tokenizer: Any, selected_ids: torch.Tensor,
    records: Sequence[dict[str, Any]], shuffled: dict[str, dict[str, Any]], model_kind: str,
    adapter_enabled: bool, workflow_started: float, future_seconds: float,
) -> list[dict[str, Any]]:
    shards = [records[start:start + args.questions_per_shard] for start in range(0, len(records), args.questions_per_shard)]
    completed_prompts = 0
    cached: dict[int, list[dict[str, Any]]] = {}
    for shard_index, shard in enumerate(shards):
        expected = {(str(record["sample_id"]), condition) for record in shard for condition in CONDITIONS}
        rows = load_valid_shard(phase_shard_path(args.output_dir, model_kind, shard_index), expected)
        if rows is not None:
            cached[shard_index] = rows
            completed_prompts += len(rows)
    total_prompts = len(records) * len(CONDITIONS)
    bar = progress_bar(
        total=total_prompts, initial=completed_prompts,
        description=f"[{model_kind} direct-choice scoring]", workflow_started=workflow_started,
        future_seconds=future_seconds,
    )
    phase_started = time.time()
    phase_initial = completed_prompts
    context = contextlib.nullcontext() if adapter_enabled else model.disable_adapter()
    with context:
        for shard_index, shard in enumerate(shards):
            if shard_index in cached:
                continue
            encoded = []
            for record in shard:
                sample = make_sample(record["question"])
                for condition in CONDITIONS:
                    document, metadata = condition_document(record, shuffled, condition, args.dummy_evidence)
                    token_ids, prompt = sequence_for_prompt(tokenizer, sample, document)
                    if len(token_ids) > args.max_input_tokens:
                        raise RuntimeError(
                            f"Prompt exceeds max tokens: sample={record['sample_id']} condition={condition} "
                            f"tokens={len(token_ids)} max={args.max_input_tokens}"
                        )
                    encoded.append({
                        "record": record, "condition": condition, "metadata": metadata,
                        "token_ids": token_ids,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    })
            output_rows = []
            for start in range(0, len(encoded), args.batch_size):
                batch = encoded[start:start + args.batch_size]
                ids, mask, positions = pad_sequences([value["token_ids"] for value in batch], tokenizer.pad_token_id)
                outputs = model(
                    input_ids=ids.to(args.device, non_blocking=True),
                    attention_mask=mask.to(args.device, non_blocking=True),
                    position_ids=positions.to(args.device, non_blocking=True),
                    use_cache=False,
                    logits_to_keep=1,
                )
                logits = outputs.logits[:, -1].index_select(-1, selected_ids).float().cpu()
                probabilities = torch.softmax(logits, dim=-1)
                for value, row_logits, row_probs in zip(batch, logits, probabilities):
                    record = value["record"]
                    gold_index = CHOICES.index(str(record["question"]["gold_answer"]))
                    wrong = torch.cat((row_logits[:gold_index], row_logits[gold_index + 1:]))
                    prediction_index = int(row_logits.argmax().item())
                    output_rows.append({
                        "run_version": RUN_VERSION,
                        "model": model_kind,
                        "sample_id": record["sample_id"],
                        "row_idx": int(record["question"]["row_idx"]),
                        "condition": value["condition"],
                        "gold_answer": record["question"]["gold_answer"],
                        "frozen_no_rag_correct": bool(record["question"]["frozen_no_rag_correct"]),
                        "prediction": CHOICES[prediction_index],
                        "correct": prediction_index == gold_index,
                        "choice_logits": [float(x) for x in row_logits.tolist()],
                        "choice_probabilities": [float(x) for x in row_probs.tolist()],
                        "gold_margin": float(row_logits[gold_index].item() - wrong.max().item()),
                        "prompt_sha256": value["prompt_sha256"],
                        "document": value["metadata"],
                    })
                completed_prompts += len(batch)
                measured = max(1e-9, time.time() - phase_started)
                new_done = max(1, completed_prompts - phase_initial)
                rate = new_done / measured
                stage_eta = (total_prompts - completed_prompts) / max(rate, 1e-9)
                overall_eta = stage_eta + future_seconds
                bar.set_postfix_str(
                    f"elapsed={format_duration(time.time()-workflow_started)} rate={rate:.1f}/s "
                    f"stage_eta={format_duration(stage_eta)} overall_eta={format_duration(overall_eta)}",
                    refresh=False,
                )
                bar.update(len(batch))
            path = phase_shard_path(args.output_dir, model_kind, shard_index)
            atomic_jsonl(path, output_rows)
            cached[shard_index] = output_rows
    bar.close()
    return [row for index in range(len(shards)) for row in cached[index]]


def index_scores(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample_id, condition = str(row["sample_id"]), str(row["condition"])
        if condition in output[sample_id]:
            raise RuntimeError(f"Duplicate score: {sample_id}/{condition}")
        output[sample_id][condition] = row
    return dict(output)


def quantile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize(
    base_rows: Sequence[dict[str, Any]], adapter_rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]], replicates: int, seed: int,
) -> dict[str, Any]:
    indexed = {"base": index_scores(base_rows), "adapter": index_scores(adapter_rows)}
    sample_ids = [str(record["sample_id"]) for record in records]
    subgroups = {
        "all": sample_ids,
        "no_rag_correct": [str(r["sample_id"]) for r in records if r["question"]["frozen_no_rag_correct"]],
        "no_rag_wrong": [str(r["sample_id"]) for r in records if not r["question"]["frozen_no_rag_correct"]],
    }
    metrics: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for subgroup, ids in subgroups.items():
        metrics[subgroup] = {}
        for model_kind in ("base", "adapter"):
            metrics[subgroup][model_kind] = {}
            for condition in CONDITIONS:
                rows = [indexed[model_kind][sample_id][condition] for sample_id in ids]
                correct = np.asarray([float(row["correct"]) for row in rows])
                margins = np.asarray([float(row["gold_margin"]) for row in rows])
                metrics[subgroup][model_kind][condition] = {
                    "n": len(rows),
                    "accuracy": float(correct.mean()),
                    "mean_gold_margin": float(margins.mean()),
                    "median_gold_margin": float(np.median(margins)),
                }
        contrast_specs = {
            "adapter_real_minus_shuffled": ("adapter", "real_support", "adapter", "shuffled_support"),
            "base_real_minus_shuffled": ("base", "real_support", "base", "shuffled_support"),
            "adapter_real_minus_no_rag": ("adapter", "real_support", "adapter", "no_rag"),
            "adapter_real_minus_no_evidence": ("adapter", "real_support", "adapter", "same_question_no_evidence"),
            "adapter_real_minus_dummy": ("adapter", "real_support", "adapter", "dummy_evidence"),
            "adapter_no_rag_minus_base_no_rag": ("adapter", "no_rag", "base", "no_rag"),
        }
        contrasts[subgroup] = {}
        for name, (left_model, left_condition, right_model, right_condition) in contrast_specs.items():
            accuracy_delta = np.asarray([
                float(indexed[left_model][sample_id][left_condition]["correct"])
                - float(indexed[right_model][sample_id][right_condition]["correct"])
                for sample_id in ids
            ])
            margin_delta = np.asarray([
                float(indexed[left_model][sample_id][left_condition]["gold_margin"])
                - float(indexed[right_model][sample_id][right_condition]["gold_margin"])
                for sample_id in ids
            ])
            contrasts[subgroup][name] = {
                "n": len(ids),
                "accuracy_delta": float(accuracy_delta.mean()),
                "mean_gold_margin_delta": float(margin_delta.mean()),
            }
        interaction_accuracy = np.asarray([
            (float(indexed["adapter"][sample_id]["real_support"]["correct"])
             - float(indexed["adapter"][sample_id]["shuffled_support"]["correct"]))
            - (float(indexed["base"][sample_id]["real_support"]["correct"])
               - float(indexed["base"][sample_id]["shuffled_support"]["correct"]))
            for sample_id in ids
        ])
        interaction_margin = np.asarray([
            (float(indexed["adapter"][sample_id]["real_support"]["gold_margin"])
             - float(indexed["adapter"][sample_id]["shuffled_support"]["gold_margin"]))
            - (float(indexed["base"][sample_id]["real_support"]["gold_margin"])
               - float(indexed["base"][sample_id]["shuffled_support"]["gold_margin"]))
            for sample_id in ids
        ])
        bootstrap_accuracy = np.empty(replicates, dtype=np.float64)
        bootstrap_margin = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            sampled = rng.integers(0, len(ids), len(ids))
            bootstrap_accuracy[replicate] = interaction_accuracy[sampled].mean()
            bootstrap_margin[replicate] = interaction_margin[sampled].mean()
        contrasts[subgroup]["semantic_specificity_interaction"] = {
            "n": len(ids),
            "definition": "(adapter real-support - shuffled-support) - (base real-support - shuffled-support)",
            "accuracy_delta": float(interaction_accuracy.mean()),
            "accuracy_delta_95ci": quantile_interval(bootstrap_accuracy),
            "mean_gold_margin_delta": float(interaction_margin.mean()),
            "mean_gold_margin_delta_95ci": quantile_interval(bootstrap_margin),
        }
    primary = contrasts["all"]["semantic_specificity_interaction"]
    primary["pre_registered_success_rule"] = "accuracy interaction >= +0.02 and paired bootstrap 95% CI lower bound > 0"
    primary["passed"] = bool(primary["accuracy_delta"] >= 0.02 and primary["accuracy_delta_95ci"][0] > 0)
    return {"metrics": metrics, "paired_contrasts": contrasts, "primary_result": primary}


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Semantic-contrastive document-dependence audit",
        "",
        f"Cohort: `{summary['cohort']['selected_questions']}` held-out MedMCQA questions; "
        f"No-RAG correct/wrong: `{summary['cohort']['no_rag_correct']}`/`{summary['cohort']['no_rag_wrong']}`.",
        "",
        "## Accuracy by evidence condition",
        "",
        "| Subgroup | Model | No-RAG | Real Direct Support | Same-question No Evidence | Shuffled Support | Dummy Evidence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for subgroup in ("all", "no_rag_correct", "no_rag_wrong"):
        for model_kind in ("base", "adapter"):
            values = summary["evaluation"]["metrics"][subgroup][model_kind]
            lines.append(
                f"| {subgroup} | {model_kind} | {percent(values['no_rag']['accuracy'])} | "
                f"{percent(values['real_support']['accuracy'])} | "
                f"{percent(values['same_question_no_evidence']['accuracy'])} | "
                f"{percent(values['shuffled_support']['accuracy'])} | "
                f"{percent(values['dummy_evidence']['accuracy'])} |"
            )
    lines += [
        "",
        "## Paired diagnostic",
        "",
        "`Gold margin` is the gold-option logit minus the strongest wrong-option logit; positive means the gold option wins.",
        "",
        "| Subgroup | Contrast | Accuracy difference | Mean gold-margin difference |",
        "|---|---|---:|---:|",
    ]
    for subgroup in ("all", "no_rag_correct", "no_rag_wrong"):
        values = summary["evaluation"]["paired_contrasts"][subgroup]
        for name in (
            "adapter_real_minus_shuffled", "base_real_minus_shuffled",
            "adapter_real_minus_no_rag", "adapter_no_rag_minus_base_no_rag",
            "semantic_specificity_interaction",
        ):
            row = values[name]
            lines.append(
                f"| {subgroup} | {name} | {100.0 * row['accuracy_delta']:+.2f}%p | "
                f"{row['mean_gold_margin_delta']:+.4f} |"
            )
    primary = summary["evaluation"]["primary_result"]
    lines += [
        "",
        "## Pre-registered decision",
        "",
        f"Primary accuracy interaction: `{100.0 * primary['accuracy_delta']:+.2f}%p` "
        f"(paired bootstrap 95% CI `{100.0 * primary['accuracy_delta_95ci'][0]:+.2f}` to "
        f"`{100.0 * primary['accuracy_delta_95ci'][1]:+.2f}%p`).",
        "",
        f"Success criterion passed: **{primary['passed']}**.",
        "",
        "Interpretation: a positive interaction means the adapter gained more from the question-matched "
        "Support document than from an equally formatted Support document taken from another question, "
        "beyond the same contrast already present in the frozen base model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    required = [args.pair_file, args.base_model / "config.json", args.adapter / "adapter_config.json", args.adapter / "adapter_model.safetensors"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    print(
        f"[overall 1/4 | elapsed {format_duration(time.time()-started)} | "
        f"overall ETA {format_duration(2*args.expected_model_phase_seconds+90)}] "
        "preflight held-out cohort, evidence controls, prompts, and immutable contract",
        flush=True,
    )
    records, shuffled, cohort_stats = load_cohort(args.pair_file, args.expected_questions)
    contract = {
        "run_version": RUN_VERSION,
        "dataset": "medmcqa",
        "split": "test",
        "pair_file": {"path": str(args.pair_file.resolve()), "sha256": sha256_file(args.pair_file)},
        "base_model": directory_identity(args.base_model),
        "adapter": directory_identity(args.adapter),
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "conditions": list(CONDITIONS),
        "selection_policy": "lowest_rerank_rank_then_pair_id_per_semantic_class_v1",
        "shuffle_policy": "source_matched_minimum_log_length_cyclic_derangement_v1",
        "dummy_evidence": args.dummy_evidence,
        "expected_questions": args.expected_questions,
        "max_input_tokens": args.max_input_tokens,
        "choice_labels": list(CHOICES),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("Evaluation contract mismatch; use a new --output-dir")
    if not contract_path.is_file():
        atomic_json(contract_path, contract)
    atomic_json(args.output_dir / "cohort_summary.json", cohort_stats)
    cohort_rows = []
    for record in records:
        donor = shuffled[record["sample_id"]]
        cohort_rows.append({
            "sample_id": record["sample_id"],
            "row_idx": record["question"]["row_idx"],
            "gold_answer": record["question"]["gold_answer"],
            "frozen_no_rag_correct": record["question"]["frozen_no_rag_correct"],
            "direct_pair_id": record["direct"]["pair_id"],
            "no_evidence_pair_id": record["no_evidence"]["pair_id"],
            "shuffled_donor_sample_id": donor["sample_id"],
            "shuffled_pair_id": donor["direct"]["pair_id"],
        })
    atomic_jsonl(args.output_dir / "cohort.jsonl", cohort_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_seen = 0
    prompt_count = 0
    preflight = tqdm(total=len(records) * len(CONDITIONS), desc="[preflight prompts]", unit="prompt", dynamic_ncols=True)
    for record in records:
        sample = make_sample(record["question"])
        for condition in CONDITIONS:
            document, _ = condition_document(record, shuffled, condition, args.dummy_evidence)
            token_ids, _ = sequence_for_prompt(tokenizer, sample, document)
            max_seen = max(max_seen, len(token_ids))
            if len(token_ids) > args.max_input_tokens:
                raise RuntimeError(f"Overlength prompt: {record['sample_id']}/{condition} tokens={len(token_ids)}")
            prompt_count += 1
            preflight.update()
    preflight.close()
    logging.info("Preflight complete: cohort=%s prompts/model=%s max_tokens=%s stats=%s", len(records), prompt_count, max_seen, cohort_stats)
    if args.preflight_only:
        print(f"[overall 1/4 complete | elapsed {format_duration(time.time()-started)}] preflight-only requested", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model scoring")
    if args.attn_implementation == "sdpa":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    logging.info("Loading base model and LoRA adapter: base=%s adapter=%s", args.base_model, args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, attn_implementation=args.attn_implementation,
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(args.device)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    model.eval()
    selected_ids = choice_token_ids(tokenizer, torch.device(args.device))
    print(
        f"[overall 2/4 | elapsed {format_duration(time.time()-started)} | "
        f"overall ETA {format_duration(2*args.expected_model_phase_seconds+60)}] "
        "score frozen base Llama under five evidence conditions",
        flush=True,
    )
    base_rows = score_model_phase(
        args=args, model=model, tokenizer=tokenizer, selected_ids=selected_ids,
        records=records, shuffled=shuffled, model_kind="base", adapter_enabled=False,
        workflow_started=started, future_seconds=args.expected_model_phase_seconds + 60,
    )
    print(
        f"[overall 3/4 | elapsed {format_duration(time.time()-started)} | "
        f"overall ETA {format_duration(args.expected_model_phase_seconds+60)}] "
        "score trained semantic-contrastive adapter under the same conditions",
        flush=True,
    )
    adapter_rows = score_model_phase(
        args=args, model=model, tokenizer=tokenizer, selected_ids=selected_ids,
        records=records, shuffled=shuffled, model_kind="adapter", adapter_enabled=True,
        workflow_started=started, future_seconds=60,
    )
    print(
        f"[overall 4/4 | elapsed {format_duration(time.time()-started)} | overall ETA 00h01m00s] "
        "paired aggregation, bootstrap confidence intervals, and decision report",
        flush=True,
    )
    evaluation = summarize(base_rows, adapter_rows, records, args.bootstrap_replicates, args.seed)
    summary = {
        "run_version": RUN_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "elapsed_seconds": time.time() - started,
        "cohort": cohort_stats,
        "evaluation": evaluation,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    write_markdown(args.output_dir / "summary.md", summary)
    primary = evaluation["primary_result"]
    print(
        f"[overall 4/4 complete | elapsed {format_duration(time.time()-started)} | overall ETA 00h00m00s] "
        f"primary_interaction={100*primary['accuracy_delta']:+.2f}%p "
        f"ci95=[{100*primary['accuracy_delta_95ci'][0]:+.2f}, {100*primary['accuracy_delta_95ci'][1]:+.2f}]%p "
        f"passed={primary['passed']} report={args.output_dir/'summary.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

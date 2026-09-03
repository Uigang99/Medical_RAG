#!/usr/bin/env python3
"""Compare question-first and document-first direct-choice prompts on frozen Llama.

This is the pre-training feasibility gate for document-token-restricted LoRA.
It uses one fixed MedMCQA validation cohort and scores the same evidence under
both prompt orders.  Gold answers are never placed in model inputs.
"""

from __future__ import annotations

import argparse
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
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_rag2_direct_choice_train_outcomes import (  # noqa: E402
    DIRECT_CHOICE_FORMAT_INSTRUCTION,
    DIRECT_CHOICE_INSTRUCTION,
    PROMPT_POLICY_VERSION,
    render_chat_prompt,
    sequence_for_prompt,
)
from evaluate_rag2_semantic_contrastive_document_dependence import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    choice_token_ids,
    format_duration,
    load_cohort,
    make_sample,
    pad_sequences,
    source_matched_derangement,
)
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from medrag.rag2_mcq import format_question  # noqa: E402


RUN_VERSION = "rag2_document_first_prompt_order_validity_v1"
REPORT_ESTIMATE_SECONDS = 15.0
ORDERS = ("question_first", "document_first")
DOCUMENT_CONDITIONS = (
    "real_support",
    "same_question_no_evidence",
    "shuffled_support",
    "dummy_evidence",
    "top8_unfiltered",
)
DEFAULT_BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
DEFAULT_PAIR_FILE = DEFAULT_BASE / "direct_semantic_mismatch_pilot_pairs_v1/medmcqa/val.jsonl"
DEFAULT_CANDIDATES = DEFAULT_BASE / "candidates/source_balanced32_rerank8_v1/medmcqa/train/candidates_top8.jsonl"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_CANDIDATES.parent / "candidate_manifest.json"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_OUTPUT = DEFAULT_BASE / "document_first_prompt_order_validity_v1/medmcqa_val512"
DEFAULT_DUMMY = "This document contains no information relevant to the medical question or its answer choices."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR_FILE)
    parser.add_argument("--candidate-file", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-questions", type=int, default=512)
    parser.add_argument("--expected-candidate-rows", type=int, default=182818)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--questions-per-shard", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--top8-document-token-budget", type=int, default=1500)
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy-evidence", default=DEFAULT_DUMMY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--expected-candidate-seconds", type=float, default=30.0)
    parser.add_argument("--expected-scoring-seconds", type=float, default=240.0)
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


def file_identity(path: Path, *, hash_content: bool = False) -> dict[str, Any]:
    stat = path.stat()
    value = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        value["sha256"] = digest.hexdigest()
    return value


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def select_cohort(records: Sequence[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(records):
        selected = list(records)
    else:
        selected = random.Random(seed).sample(list(records), maximum)
    return sorted(selected, key=lambda value: (int(value["question"]["row_idx"]), value["sample_id"]))


def load_selected_candidates(
    path: Path, wanted: set[str], expected_rows: int, workflow_started: float, future_seconds: float,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    found: dict[str, list[dict[str, Any]]] = {}
    rows_seen = 0
    bar = tqdm(
        total=expected_rows,
        desc="[overall 2/4 | candidate join 1/2]",
        unit="question",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    stage_started = time.time()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows_seen += 1
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in wanted:
                documents = sorted(list(row.get("candidate_documents") or []), key=lambda value: int(value["rerank_rank"]))
                if len(documents) != 8 or [int(value["rerank_rank"]) for value in documents] != list(range(1, 9)):
                    raise RuntimeError(f"Invalid Top-8 candidate layout: {sample_id}")
                found[sample_id] = documents
            if rows_seen % 4096 == 0 or rows_seen == expected_rows:
                elapsed = max(1e-9, time.time() - stage_started)
                rate = rows_seen / elapsed
                stage_eta = max(0, expected_rows - rows_seen) / max(rate, 1e-9)
                bar.set_postfix_str(
                    f"found={len(found)}/{len(wanted)} rate={rate:.0f}/s stage_eta={format_duration(stage_eta)} "
                    f"overall_eta={format_duration(stage_eta+future_seconds)}",
                    refresh=False,
                )
                bar.update(rows_seen - bar.n)
    if rows_seen != expected_rows:
        raise RuntimeError(f"Candidate row count mismatch: expected={expected_rows} actual={rows_seen}")
    bar.close()
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"Missing selected candidates: count={len(missing)} first={missing[:5]}")
    return found, rows_seen


def build_document_first_user_prompt(sample: Any, evidence: str) -> str:
    value = str(evidence).strip()
    if not value:
        raise ValueError("Document-first prompt requires non-empty evidence")
    row = {"question": sample.question, "options": sample.options}
    return (
        f"{DIRECT_CHOICE_INSTRUCTION}\n"
        f"{DIRECT_CHOICE_FORMAT_INSTRUCTION}\n"
        f"Documents:\n{value}\n\n"
        f"Here is the question: {format_question(row)}"
    )


def sequence_for_order(tokenizer: Any, sample: Any, evidence: str | None, order: str) -> tuple[list[int], str]:
    if not evidence:
        # No-RAG must remain token-identical to the established direct-choice prompt.
        return sequence_for_prompt(tokenizer, sample, None)
    if order == "question_first":
        return sequence_for_prompt(tokenizer, sample, evidence)
    if order != "document_first":
        raise ValueError(order)
    prompt = render_chat_prompt(tokenizer, build_document_first_user_prompt(sample, evidence))
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    return token_ids, prompt


def equal_token_allocation(capacities: Sequence[int], budget: int) -> list[int]:
    allocations = [0] * len(capacities)
    remaining = max(0, int(budget))
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        for position, index in enumerate(active):
            grant = min(share, capacities[index] - allocations[index], remaining)
            allocations[index] += grant
            remaining -= grant
            if allocations[index] < capacities[index]:
                next_active.append(index)
            if remaining <= 0:
                next_active.extend(active[position + 1:])
                break
        active = next_active
    return allocations


def pack_top8(
    tokenizer: Any, sample: Any, documents: Sequence[dict[str, Any]], token_budget: int, max_input_tokens: int,
) -> tuple[str, dict[str, Any]]:
    raw_ids = [tokenizer.encode(" ".join(str(row.get("text") or "").split()), add_special_tokens=False) for row in documents]
    if any(not ids for ids in raw_ids):
        raise RuntimeError(f"Empty Top-8 document: {sample.id}")
    allocations = equal_token_allocation([len(ids) for ids in raw_ids], token_budget)

    def evidence_for(values: Sequence[int]) -> str:
        return "\n\n".join(
            tokenizer.decode(ids[:amount], skip_special_tokens=True).strip()
            for ids, amount in zip(raw_ids, values)
        )

    evidence = evidence_for(allocations)
    for _ in range(32):
        lengths = [len(sequence_for_order(tokenizer, sample, evidence, order)[0]) for order in ORDERS]
        overflow = max(lengths) - max_input_tokens
        if overflow <= 0:
            break
        adjustable = [index for index, amount in enumerate(allocations) if amount > 1]
        if not adjustable:
            raise RuntimeError(f"Cannot pack Top-8 within context: {sample.id}")
        remaining = overflow + 8
        for index in sorted(adjustable, key=lambda value: allocations[value], reverse=True):
            if remaining <= 0:
                break
            reduction = min(allocations[index] - 1, max(1, math.ceil(remaining / len(adjustable))))
            allocations[index] -= reduction
            remaining -= reduction
        evidence = evidence_for(allocations)
    else:
        raise RuntimeError(f"Top-8 packing did not converge: {sample.id}")
    lengths = {order: len(sequence_for_order(tokenizer, sample, evidence, order)[0]) for order in ORDERS}
    return evidence, {
        "raw_tokens": [len(ids) for ids in raw_ids],
        "kept_tokens": allocations,
        "truncated_documents": sum(amount < len(ids) for amount, ids in zip(allocations, raw_ids)),
        "prompt_tokens": lengths,
    }


def evidence_for_condition(
    record: dict[str, Any], shuffled: dict[str, dict[str, Any]], candidates: dict[str, list[dict[str, Any]]],
    condition: str, dummy: str, top8_evidence: dict[str, str],
) -> tuple[str, dict[str, Any]]:
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
            "pair_id": row["pair_id"], "source": row["document_source"], "donor_sample_id": donor["sample_id"],
        }
    if condition == "dummy_evidence":
        return dummy, {"source": "fixed_dummy"}
    if condition == "top8_unfiltered":
        docs = candidates[record["sample_id"]]
        return top8_evidence[record["sample_id"]], {
            "source": "top8_unfiltered", "stable_ids": [row["stable_id"] for row in docs],
        }
    raise ValueError(condition)


def score_tasks_for_record(
    tokenizer: Any, record: dict[str, Any], shuffled: dict[str, dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]], dummy: str, top8_evidence: dict[str, str],
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    sample = make_sample(record["question"])
    token_ids, prompt = sequence_for_order(tokenizer, sample, None, "question_first")
    expected_hash = str(record["question"]["frozen_no_rag_prompt_sha256"])
    actual_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"No-RAG prompt changed for {sample.id}: cached={expected_hash} actual={actual_hash}")
    values = [{
        "sample_id": record["sample_id"], "order": "shared", "condition": "no_rag",
        "token_ids": token_ids, "prompt_sha256": actual_hash, "document": {},
    }]
    for condition in DOCUMENT_CONDITIONS:
        evidence, metadata = evidence_for_condition(record, shuffled, candidates, condition, dummy, top8_evidence)
        for order in ORDERS:
            ids, rendered = sequence_for_order(tokenizer, sample, evidence, order)
            if len(ids) > max_input_tokens:
                raise RuntimeError(f"Overlength prompt: {sample.id}/{order}/{condition}={len(ids)}")
            values.append({
                "sample_id": record["sample_id"], "order": order, "condition": condition,
                "token_ids": ids, "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "document": metadata,
            })
    return values


def shard_path(output: Path, index: int) -> Path:
    return output / "score_shards" / f"shard_{index:05d}.jsonl"


def valid_shard(path: Path, expected: set[tuple[str, str, str]]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        rows = list(iter_jsonl(path))
    except Exception:
        return None
    keys = {(str(row["sample_id"]), str(row["order"]), str(row["condition"])) for row in rows}
    return rows if keys == expected and len(rows) == len(expected) else None


@torch.inference_mode()
def score(
    args: argparse.Namespace, model: Any, tokenizer: Any, selected_ids: torch.Tensor,
    records: Sequence[dict[str, Any]], shuffled: dict[str, dict[str, Any]], candidates: dict[str, list[dict[str, Any]]],
    top8_evidence: dict[str, str], workflow_started: float,
) -> list[dict[str, Any]]:
    shards = [records[start:start + args.questions_per_shard] for start in range(0, len(records), args.questions_per_shard)]
    cached: dict[int, list[dict[str, Any]]] = {}
    complete = 0
    for index, shard in enumerate(shards):
        expected = {
            (str(record["sample_id"]), order, condition)
            for record in shard
            for order, condition in (("shared", "no_rag"),) + tuple(
                (value, item) for item in DOCUMENT_CONDITIONS for value in ORDERS
            )
        }
        rows = valid_shard(shard_path(args.output_dir, index), expected)
        if rows is not None:
            cached[index] = rows
            complete += len(rows)
    total = len(records) * (1 + 2 * len(DOCUMENT_CONDITIONS))
    bar = tqdm(
        total=total,
        initial=complete,
        desc="[overall 3/4 | frozen-Llama scoring]",
        unit="prompt",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    stage_started = time.time()
    initial = complete
    for index, shard in enumerate(shards):
        if index in cached:
            continue
        source_by_id = {str(record["sample_id"]): record for record in shard}
        tasks = [
            task
            for record in shard
            for task in score_tasks_for_record(
                tokenizer, record, shuffled, candidates, args.dummy_evidence, top8_evidence, args.max_input_tokens
            )
        ]
        output = []
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start + args.batch_size]
            ids, mask, positions = pad_sequences([value["token_ids"] for value in batch], tokenizer.pad_token_id)
            result = model(
                input_ids=ids.to(args.device, non_blocking=True),
                attention_mask=mask.to(args.device, non_blocking=True),
                position_ids=positions.to(args.device, non_blocking=True),
                use_cache=False, logits_to_keep=1,
            )
            logits = result.logits[:, -1].index_select(-1, selected_ids).float().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            for task, row_logits, row_probs in zip(batch, logits, probabilities):
                source = source_by_id[str(task["sample_id"])]
                gold_index = CHOICES.index(str(source["question"]["gold_answer"]))
                wrong = torch.cat((row_logits[:gold_index], row_logits[gold_index + 1:]))
                prediction = int(row_logits.argmax())
                output.append({
                    "run_version": RUN_VERSION,
                    "sample_id": task["sample_id"], "row_idx": int(source["question"]["row_idx"]),
                    "order": task["order"], "condition": task["condition"],
                    "gold_answer": source["question"]["gold_answer"],
                    "prediction": CHOICES[prediction], "correct": prediction == gold_index,
                    "choice_logits": [float(value) for value in row_logits.tolist()],
                    "choice_probabilities": [float(value) for value in row_probs.tolist()],
                    "gold_margin": float(row_logits[gold_index] - wrong.max()),
                    "prompt_sha256": task["prompt_sha256"], "document": task["document"],
                })
            complete += len(batch)
            elapsed = max(1e-9, time.time() - stage_started)
            new = max(1, complete - initial)
            rate = new / elapsed
            stage_eta = (total - complete) / max(rate, 1e-9)
            bar.set_postfix_str(
                f"elapsed={format_duration(time.time()-workflow_started)} rate={rate:.1f}/s "
                f"stage_eta={format_duration(stage_eta)} "
                f"overall_eta={format_duration(stage_eta+REPORT_ESTIMATE_SECONDS)}",
                refresh=False,
            )
            bar.update(len(batch))
        atomic_jsonl(shard_path(args.output_dir, index), output)
        cached[index] = output
    bar.close()
    return [row for index in range(len(shards)) for row in cached[index]]


def bootstrap_interval(values: np.ndarray, replicates: int, rng: np.random.Generator) -> list[float]:
    sampled = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        positions = rng.integers(0, len(values), len(values))
        sampled[index] = values[positions].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def summarize(rows: Sequence[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    index = {(str(row["sample_id"]), str(row["order"]), str(row["condition"])): row for row in rows}
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    current_no_rag = {sample_id: bool(index[(sample_id, "shared", "no_rag")]["correct"]) for sample_id in sample_ids}
    groups = {
        "all": sample_ids,
        "no_rag_correct": [value for value in sample_ids if current_no_rag[value]],
        "no_rag_wrong": [value for value in sample_ids if not current_no_rag[value]],
    }
    rng = np.random.default_rng(seed)
    metrics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for group_name, ids in groups.items():
        metrics[group_name] = {
            "no_rag": {
                "n": len(ids),
                "accuracy": float(np.mean([index[(value, "shared", "no_rag")]["correct"] for value in ids])),
                "mean_gold_margin": float(np.mean([index[(value, "shared", "no_rag")]["gold_margin"] for value in ids])),
            }
        }
        for order in ORDERS:
            metrics[group_name][order] = {}
            for condition in DOCUMENT_CONDITIONS:
                selected = [index[(value, order, condition)] for value in ids]
                metrics[group_name][order][condition] = {
                    "n": len(ids),
                    "accuracy": float(np.mean([row["correct"] for row in selected])),
                    "mean_gold_margin": float(np.mean([row["gold_margin"] for row in selected])),
                }
        comparisons[group_name] = {}
        for order in ORDERS:
            accuracy = np.asarray([
                float(index[(value, order, "real_support")]["correct"])
                - float(index[(value, order, "shuffled_support")]["correct"])
                for value in ids
            ])
            margin = np.asarray([
                float(index[(value, order, "real_support")]["gold_margin"])
                - float(index[(value, order, "shuffled_support")]["gold_margin"])
                for value in ids
            ])
            comparisons[group_name][f"{order}_support_advantage"] = {
                "n": len(ids), "accuracy_delta": float(accuracy.mean()),
                "accuracy_delta_95ci": bootstrap_interval(accuracy, replicates, rng),
                "mean_gold_margin_delta": float(margin.mean()),
                "mean_gold_margin_delta_95ci": bootstrap_interval(margin, replicates, rng),
            }
        interaction_accuracy = np.asarray([
            (float(index[(value, "document_first", "real_support")]["correct"])
             - float(index[(value, "document_first", "shuffled_support")]["correct"]))
            - (float(index[(value, "question_first", "real_support")]["correct"])
               - float(index[(value, "question_first", "shuffled_support")]["correct"]))
            for value in ids
        ])
        interaction_margin = np.asarray([
            (float(index[(value, "document_first", "real_support")]["gold_margin"])
             - float(index[(value, "document_first", "shuffled_support")]["gold_margin"]))
            - (float(index[(value, "question_first", "real_support")]["gold_margin"])
               - float(index[(value, "question_first", "shuffled_support")]["gold_margin"]))
            for value in ids
        ])
        top8_accuracy = np.asarray([
            float(index[(value, "document_first", "top8_unfiltered")]["correct"])
            - float(index[(value, "question_first", "top8_unfiltered")]["correct"])
            for value in ids
        ])
        comparisons[group_name]["document_first_minus_question_first_support_advantage"] = {
            "n": len(ids), "accuracy_delta": float(interaction_accuracy.mean()),
            "accuracy_delta_95ci": bootstrap_interval(interaction_accuracy, replicates, rng),
            "mean_gold_margin_delta": float(interaction_margin.mean()),
            "mean_gold_margin_delta_95ci": bootstrap_interval(interaction_margin, replicates, rng),
        }
        comparisons[group_name]["document_first_minus_question_first_top8"] = {
            "n": len(ids), "accuracy_delta": float(top8_accuracy.mean()),
            "accuracy_delta_95ci": bootstrap_interval(top8_accuracy, replicates, rng),
        }
    old = comparisons["all"]["question_first_support_advantage"]
    new = comparisons["all"]["document_first_support_advantage"]
    top8 = comparisons["all"]["document_first_minus_question_first_top8"]
    passed = bool(
        new["mean_gold_margin_delta"] > 0
        and new["accuracy_delta"] >= old["accuracy_delta"] - 0.02
        and top8["accuracy_delta"] >= -0.02
    )
    return {
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "decision": {
            "passed": passed,
            "criteria": {
                "document_first_real_minus_shuffled_mean_gold_margin_positive": new["mean_gold_margin_delta"] > 0,
                "support_accuracy_advantage_degradation_at_most_0p02": new["accuracy_delta"] >= old["accuracy_delta"] - 0.02,
                "top8_accuracy_drop_at_most_0p02": top8["accuracy_delta"] >= -0.02,
            },
            "measured": {
                "question_first_support_accuracy_advantage": old["accuracy_delta"],
                "document_first_support_accuracy_advantage": new["accuracy_delta"],
                "support_advantage_change": new["accuracy_delta"] - old["accuracy_delta"],
                "document_first_support_mean_gold_margin_advantage": new["mean_gold_margin_delta"],
                "top8_accuracy_change": top8["accuracy_delta"],
            },
        },
    }


def pct(value: float) -> str:
    return f"{100*value:.2f}%"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    result = summary["evaluation"]
    lines = [
        "# Document-first prompt-order validity pilot", "",
        f"Cohort: `{summary['cohort']['selected_questions']}` MedMCQA validation questions; "
        f"current frozen No-RAG correct/wrong: `{summary['evaluation_counts']['no_rag_correct']}`/"
        f"`{summary['evaluation_counts']['no_rag_wrong']}`.", "",
        "Accuracy is the fraction of questions whose highest-logit A/B/C/D option is gold.", "",
        "| Group | Order | No-RAG | Real Support | Same-question No Evidence | Shuffled Support | Dummy | Top-8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", "no_rag_correct", "no_rag_wrong"):
        for order in ORDERS:
            values = result["metrics"][group]
            current = values[order]
            lines.append(
                f"| {group} | {order} | {pct(values['no_rag']['accuracy'])} | "
                f"{pct(current['real_support']['accuracy'])} | {pct(current['same_question_no_evidence']['accuracy'])} | "
                f"{pct(current['shuffled_support']['accuracy'])} | {pct(current['dummy_evidence']['accuracy'])} | "
                f"{pct(current['top8_unfiltered']['accuracy'])} |"
            )
    lines += ["", "## Support-specific effect", "",
              "Support advantage is Real Support accuracy minus Shuffled Support accuracy.", "",
              "| Group | Question-first | Document-first | Change | Document-first gold-margin advantage |",
              "|---|---:|---:|---:|---:|"]
    for group in ("all", "no_rag_correct", "no_rag_wrong"):
        values = result["paired_comparisons"][group]
        old = values["question_first_support_advantage"]
        new = values["document_first_support_advantage"]
        change = values["document_first_minus_question_first_support_advantage"]
        lines.append(
            f"| {group} | {100*old['accuracy_delta']:+.2f}%p | {100*new['accuracy_delta']:+.2f}%p | "
            f"{100*change['accuracy_delta']:+.2f}%p | {new['mean_gold_margin_delta']:+.4f} |"
        )
    decision = result["decision"]
    lines += ["", "## Pre-registered decision", "", f"Passed: **{decision['passed']}**", "", "```json",
              json.dumps(decision, ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.time()
    required = [args.pair_file, args.candidate_file, args.candidate_manifest, args.model / "config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    expected_total = args.expected_candidate_seconds + args.expected_scoring_seconds + REPORT_ESTIMATE_SECONDS
    print(
        f"[overall 1/4 | elapsed 00h00m00s | overall ETA {format_duration(expected_total)}] "
        "select fixed validation cohort and validate immutable contracts", flush=True,
    )
    all_records, _, source_stats = load_cohort(args.pair_file, 0)
    records = select_cohort(all_records, args.max_questions, args.seed)
    shuffled = source_matched_derangement(records)
    wanted = {str(value["sample_id"]) for value in records}
    contract = {
        "run_version": RUN_VERSION, "dataset": "medmcqa", "split": "val",
        "pair_file": file_identity(args.pair_file, hash_content=True),
        "candidate_file": file_identity(args.candidate_file),
        "candidate_manifest": file_identity(args.candidate_manifest, hash_content=True),
        "model": str(args.model.resolve()), "prompt_policy_reference": PROMPT_POLICY_VERSION,
        "orders": list(ORDERS), "conditions": ["no_rag", *DOCUMENT_CONDITIONS],
        "cohort_policy": "seeded_random_from_questions_with_direct_support_and_no_evidence_v1",
        "shuffle_policy": "source_matched_minimum_log_length_cyclic_derangement_v1",
        "max_questions": args.max_questions, "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "top8_document_token_budget": args.top8_document_token_budget,
        "dummy_evidence": args.dummy_evidence,
        "hypothesis": (
            "Moving documents before the question preserves evidence-specific direct-choice behavior while "
            "making document-token states causally unable to encode the later question."
        ),
        "primary_metric": "real_support_minus_source_matched_shuffled_support_accuracy",
        "pass_thresholds": {
            "document_first_real_minus_shuffled_mean_gold_margin": ">0",
            "support_accuracy_advantage_degradation": ">=-0.02",
            "top8_accuracy_change": ">=-0.02",
        },
        "scoring": {
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "choice_score": "next-token logits for A/B/C/D after exact Final answer: ( prefix",
            "bootstrap_replicates": args.bootstrap_replicates,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("Prompt-order pilot contract mismatch; use a new --output-dir")
    if not contract_path.is_file():
        atomic_json(contract_path, contract)
    atomic_jsonl(args.output_dir / "cohort.jsonl", [
        {"sample_id": value["sample_id"], "row_idx": value["question"]["row_idx"],
         "cached_no_rag_correct": value["question"]["frozen_no_rag_correct"],
         "direct_pair_id": value["direct"]["pair_id"], "no_evidence_pair_id": value["no_evidence"]["pair_id"]}
        for value in records
    ])
    print(
        f"[overall 2/4 | elapsed {format_duration(time.time()-started)} | overall ETA "
        f"{format_duration(args.expected_candidate_seconds+args.expected_scoring_seconds+REPORT_ESTIMATE_SECONDS)}] "
        "stream 5.5-GiB candidate file and join selected Top-8 documents", flush=True,
    )
    candidates, candidate_rows = load_selected_candidates(
        args.candidate_file, wanted, args.expected_candidate_rows, started, args.expected_scoring_seconds + 60
    )
    # Pair rows must point into the immutable Top-8 candidate set.
    for record in records:
        stable_ids = {str(value["stable_id"]) for value in candidates[record["sample_id"]]}
        for name in ("direct", "no_evidence"):
            if str(record[name]["document_stable_id"]) not in stable_ids:
                raise RuntimeError(f"Selected semantic document is absent from Top-8: {record['sample_id']}/{name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    top8_evidence: dict[str, str] = {}
    packing = {}
    preflight = tqdm(
        total=len(records),
        desc="[overall 2/4 | prompt preflight 2/2]",
        unit="question",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    max_tokens = 0
    for record in records:
        sample = make_sample(record["question"])
        evidence, details = pack_top8(
            tokenizer, sample, candidates[record["sample_id"]], args.top8_document_token_budget, args.max_input_tokens
        )
        top8_evidence[record["sample_id"]] = evidence
        packing[record["sample_id"]] = details
        tasks = score_tasks_for_record(
            tokenizer, record, shuffled, candidates, args.dummy_evidence, top8_evidence, args.max_input_tokens
        )
        max_tokens = max(max_tokens, max(len(value["token_ids"]) for value in tasks))
        preflight.update()
    preflight.close()
    atomic_json(args.output_dir / "packing_summary.json", {
        "questions": len(records), "maximum_prompt_tokens": max_tokens,
        "mean_top8_kept_tokens": float(np.mean([sum(value["kept_tokens"]) for value in packing.values()])),
        "mean_truncated_documents": float(np.mean([value["truncated_documents"] for value in packing.values()])),
        "per_question": packing,
    })
    cohort = {
        "eligible_questions": len(all_records), "selected_questions": len(records),
        "cached_no_rag_correct": sum(bool(value["question"]["frozen_no_rag_correct"]) for value in records),
        "cached_no_rag_wrong": sum(not bool(value["question"]["frozen_no_rag_correct"]) for value in records),
        "candidate_rows": candidate_rows, "source_cohort": source_stats,
    }
    atomic_json(args.output_dir / "cohort_summary.json", cohort)
    logging.info("Preflight complete: cohort=%s max_tokens=%s candidate_rows=%s", cohort, max_tokens, candidate_rows)
    if args.preflight_only:
        print(f"[overall 2/4 complete | elapsed {format_duration(time.time()-started)}] preflight-only requested", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-Llama scoring")
    if args.attn_implementation == "sdpa":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(
        f"[overall 3/4 | elapsed {format_duration(time.time()-started)} | overall ETA "
        f"{format_duration(args.expected_scoring_seconds+REPORT_ESTIMATE_SECONDS)}] load and score frozen Llama",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_implementation,
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    selected_ids = choice_token_ids(tokenizer, torch.device(args.device))
    rows = score(args, model, tokenizer, selected_ids, records, shuffled, candidates, top8_evidence, started)
    print(
        f"[overall 4/4 | elapsed {format_duration(time.time()-started)} | overall ETA "
        f"{format_duration(REPORT_ESTIMATE_SECONDS)}] "
        "paired aggregation and bootstrap decision", flush=True,
    )
    evaluation = summarize(rows, args.bootstrap_replicates, args.seed)
    current_index = {(row["sample_id"], row["order"], row["condition"]): row for row in rows}
    current_correct = sum(current_index[(value["sample_id"], "shared", "no_rag")]["correct"] for value in records)
    summary = {
        "run_version": RUN_VERSION, "completed_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(), "elapsed_seconds": time.time() - started,
        "cohort": cohort,
        "evaluation_counts": {"no_rag_correct": current_correct, "no_rag_wrong": len(records)-current_correct},
        "evaluation": evaluation,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    write_markdown(args.output_dir / "summary.md", summary)
    decision = evaluation["decision"]
    print(
        f"[overall 4/4 complete | elapsed {format_duration(time.time()-started)} | overall ETA 00h00m00s] "
        f"passed={decision['passed']} measured={json.dumps(decision['measured'], sort_keys=True)} "
        f"report={args.output_dir/'summary.md'}", flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate document-level attribution on a fixed Direct-Choice answer.

This no-training audit reuses the exact cohort from the earlier
rationale+answer Gradient×Input experiment, but removes the rationale from the
model prompt and target.  For each question it:

1. obtains the frozen model's constrained Direct-Choice answer from Top-8;
2. computes document Gradient×Input for that answer token;
3. physically removes every document and measures the same answer token's
   full-vocabulary log-probability change;
4. adds every document to No-RAG and measures the same score; and
5. removes the two/four most- versus least-influential documents together.

Gold answers and semantic labels never define the attribution.  They are used
only for diagnostic subgroups.  The primary comparison is within-question
relative document order, not a pooled cross-question scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
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
    PROMPT_POLICY_VERSION,
    build_anchored_direct_choice_user_prompt,
    make_sample,
    render_chat_prompt,
)
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_direct_choice_document_attribution_validity_v1"
SCORE_VERSION = "full_vocabulary_log_probability_of_top8_predicted_choice_token_v1"
GRADIENT_ATTRIBUTION = "document_signed_sum_gradient_times_input_on_direct_choice_logprob_v1"
REMOVAL_ATTRIBUTION = "top8_conditional_leave_one_document_out_logprob_delta_v1"
SINGLETON_ATTRIBUTION = "no_rag_conditional_single_document_addition_logprob_delta_v1"
SUPPORT_LABELS = frozenset({"direct_support", "supporting_evidence"})
NON_SUPPORT_LABELS = frozenset({"no_evidence", "misleading_evidence"})

DEFAULT_BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
DEFAULT_PRIOR = (
    DEFAULT_BASE
    / "document_attribution_faithfulness_mvp_v1"
    / "medqa_train_rationale_answer_gradxinput_mixed256_non64_v1"
)
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_OUTPUT = (
    DEFAULT_BASE
    / "direct_choice_document_attribution_validity_v1"
    / "medqa_train_same320_question_first_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-file", type=Path, default=DEFAULT_PRIOR / "cohort.jsonl")
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR / "report.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-questions", type=int, default=0, help="0 uses the complete frozen cohort")
    parser.add_argument("--context-batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        result["sha256"] = sha256_file(path)
    return result


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
        "weight_files": [{"name": item.name, "size_bytes": item.stat().st_size} for item in shards],
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc


class HierarchicalProgress:
    """Visible weighted overall progress plus active-stage rate and ETA."""

    def __init__(self, stages: Sequence[str], estimates: Sequence[float]) -> None:
        if not stages or len(stages) != len(estimates):
            raise ValueError("stages and estimates must have equal non-zero lengths")
        self.stages = list(stages)
        self.estimates = [max(0.1, float(value)) for value in estimates]
        self.started = time.time()
        self.stage_started = self.started
        self.stage_index = 0
        self.stage_total = 1
        self.stage_done = 0
        self.stage_initial = 0
        self.unit = "item"
        self.history: deque[tuple[float, int]] = deque()
        self.stage_bar: tqdm[Any] | None = None
        self.overall = tqdm(
            total=1000, desc="Overall", position=0, leave=True, dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar:10}| [initializing]",
        )

    def start(self, index: int, total: int, unit: str, *, initial: int = 0) -> None:
        if self.stage_bar is not None:
            self.stage_bar.close()
        self.stage_index = index
        self.stage_total = max(1, int(total))
        self.stage_done = min(max(0, int(initial)), self.stage_total)
        self.stage_initial = self.stage_done
        self.unit = unit
        self.stage_started = time.time()
        self.history.clear()
        self.history.append((self.stage_started, self.stage_done))
        self.stage_bar = tqdm(
            total=total, initial=initial, desc=f"Stage {index}/{len(self.stages)} - {self.stages[index-1]}",
            unit=unit, position=1, leave=False, dynamic_ncols=True,
        )
        self._render()

    def _rate(self, now: float) -> float | None:
        while len(self.history) > 2 and self.history[1][0] < now - 60.0:
            self.history.popleft()
        if len(self.history) < 2:
            return None
        seconds = self.history[-1][0] - self.history[0][0]
        count = self.history[-1][1] - self.history[0][1]
        return count / seconds if seconds > 0 and count > 0 else None

    def _render(self) -> None:
        now = time.time()
        rate = self._rate(now)
        fraction = self.stage_done / self.stage_total
        stage_eta = (self.stage_total - self.stage_done) / rate if rate else self.estimates[self.stage_index - 1] * (1 - fraction)
        future_eta = sum(self.estimates[self.stage_index:])
        total_weight = sum(self.estimates)
        completed_weight = sum(self.estimates[: self.stage_index - 1]) + self.estimates[self.stage_index - 1] * fraction
        self.overall.n = min(1000, round(1000 * completed_weight / total_weight))
        self.overall.bar_format = (
            "{desc}: {percentage:3.0f}%|{bar:10}| "
            f"[stage {self.stage_index}/{len(self.stages)}, elapsed {format_duration(now-self.started)}, "
            f"ETA {format_duration(stage_eta+future_eta)}]"
        )
        self.overall.refresh()
        if self.stage_bar is not None:
            rate_text = "calibrating" if rate is None else f"{rate:.2f} {self.unit}/s"
            self.stage_bar.set_postfix_str(f"{rate_text}, ETA {format_duration(stage_eta)}", refresh=False)

    def set(self, done: int) -> None:
        done = min(max(0, int(done)), self.stage_total)
        delta = done - self.stage_done
        self.stage_done = done
        self.history.append((time.time(), done))
        if self.stage_bar is not None and delta > 0:
            self.stage_bar.update(delta)
        self._render()

    def update(self, amount: int = 1) -> None:
        self.set(self.stage_done + amount)

    def complete(self, detail: str) -> None:
        self.set(self.stage_total)
        duration = time.time() - self.stage_started
        self.estimates[self.stage_index - 1] = duration
        if self.stage_bar is not None:
            self.stage_bar.close()
            self.stage_bar = None
        tqdm.write(f"[stage {self.stage_index}/{len(self.stages)} complete | duration {format_duration(duration)}] {detail}")

    def finish(self, detail: str) -> None:
        if self.stage_bar is not None:
            self.stage_bar.close()
        self.overall.n = 1000
        self.overall.bar_format = (
            "{desc}: {percentage:3.0f}%|{bar:10}| "
            f"[stage {len(self.stages)}/{len(self.stages)}, elapsed {format_duration(time.time()-self.started)}, ETA 00h00m00s]"
        )
        self.overall.refresh()
        self.overall.close()
        print(f"[workflow complete] {detail}", flush=True)


def load_cohort(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if maximum > 0:
        rows = rows[:maximum]
    if not rows:
        raise RuntimeError("Empty attribution cohort")
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise RuntimeError(f"Missing or duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        documents = list(row.get("documents") or [])
        labels = list(row.get("semantic_labels") or [])
        if len(documents) != 8 or len(labels) != 8:
            raise RuntimeError(f"Expected exactly eight documents and labels: {sample_id}")
        if [int(doc.get("rerank_rank", -1)) for doc in documents] != list(range(1, 9)):
            raise RuntimeError(f"Invalid rerank order: {sample_id}")
        if any(not str(doc.get("text") or "").strip() for doc in documents):
            raise RuntimeError(f"Empty document text: {sample_id}")
        if str(row.get("gold_answer")) not in CHOICES:
            raise RuntimeError(f"Invalid gold label: {sample_id}")
    return rows


def direct_prompt(tokenizer: Any, row: dict[str, Any], documents: Sequence[dict[str, Any]]) -> tuple[str, list[int]]:
    sample = make_sample({
        "row_idx": int(row["row_idx"]),
        "sample_id": str(row["sample_id"]),
        "dataset": str(row["sample_id"]).split(":", 1)[0],
        "split": str(row.get("analysis_split") or "train"),
        "question": str(row["question"]),
        "options": dict(row["options"]),
        "answer": str(row["gold_answer"]),
    })
    document_blob = "\n\n".join(str(item["text"]).strip() for item in documents)
    user_prompt = build_anchored_direct_choice_user_prompt(sample, document_blob or None)
    prompt = render_chat_prompt(tokenizer, user_prompt)
    return prompt, list(tokenizer.encode(prompt, add_special_tokens=False))


def direct_sequence(tokenizer: Any, row: dict[str, Any], documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prompt, input_ids = direct_prompt(tokenizer, row, documents)
    if not documents:
        return {"prompt": prompt, "input_ids": input_ids, "document_token_indices": [], "document_token_counts": []}
    document_blob = "\n\n".join(str(item["text"]).strip() for item in documents)
    block_start = prompt.rfind(document_blob)
    if block_start < 0:
        raise RuntimeError(f"Cannot locate direct-choice document block: {row['sample_id']}")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != input_ids:
        raise RuntimeError(f"Direct prompt token replay mismatch: {row['sample_id']}")
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    spans: list[tuple[int, int]] = []
    cursor = block_start
    for index, document in enumerate(documents):
        text = str(document["text"]).strip()
        if prompt[cursor: cursor + len(text)] != text:
            raise RuntimeError(f"Document character replay mismatch: {row['sample_id']}:{index+1}")
        spans.append((cursor, cursor + len(text)))
        cursor += len(text)
        if index + 1 < len(documents):
            if prompt[cursor: cursor + 2] != "\n\n":
                raise RuntimeError(f"Document separator mismatch: {row['sample_id']}:{index+1}")
            cursor += 2
    token_indices = [
        [position for position, (start, end) in enumerate(offsets) if end > left and start < right]
        for left, right in spans
    ]
    if any(not values for values in token_indices):
        raise RuntimeError(f"Document has no direct-prompt tokens: {row['sample_id']}")
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "document_token_indices": token_indices,
        "document_token_counts": [len(values) for values in token_indices],
    }


def choice_token_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    values: list[int] = []
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice {choice!r} is not one token after the fixed answer prefix: {ids}")
        values.append(int(ids[0]))
    return torch.tensor(values, dtype=torch.long, device=device)


def token_logprob_and_choice_logits(model: Any, hidden: torch.Tensor, target_id: torch.Tensor, choice_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.lm_head(hidden).float()
    target = logits.gather(-1, target_id.view(-1, 1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    choices = logits.index_select(-1, choice_ids)
    return target, choices


def full_gradient_attribution(
    model: Any, sequence: dict[str, Any], choice_ids: torch.Tensor, device: torch.device,
) -> tuple[str, int, float, list[float], list[float], list[float]]:
    ids = torch.tensor(sequence["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    embeddings = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    outputs = model.model(
        inputs_embeds=embeddings, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True,
    )
    hidden = outputs.last_hidden_state[:, -1]
    preliminary = model.lm_head(hidden).float().index_select(-1, choice_ids)
    predicted_index = int(preliminary[0].argmax().item())
    target_id = choice_ids[predicted_index].view(1)
    score, choices = token_logprob_and_choice_logits(model, hidden, target_id, choice_ids)
    gradient = torch.autograd.grad(score[0], embeddings, retain_graph=False, create_graph=False)[0]
    token_values = (gradient.float() * embeddings.float()).sum(dim=-1)[0]
    signed = [float(token_values[indices].sum().item()) for indices in sequence["document_token_indices"]]
    result = (
        CHOICES[predicted_index], int(target_id.item()), float(score.detach().item()),
        [float(x) for x in choices[0].detach().cpu().tolist()], signed, [abs(x) for x in signed],
    )
    del outputs, hidden, preliminary, score, choices, gradient, token_values, embeddings, ids
    return result


@torch.inference_mode()
def score_contexts(
    model: Any, tokenizer: Any, row: dict[str, Any], contexts: Sequence[Sequence[dict[str, Any]]],
    target_id: int, choice_ids: torch.Tensor, device: torch.device, batch_size: int, max_tokens: int,
) -> tuple[list[float], list[str], list[int]]:
    sequences = [direct_sequence(tokenizer, row, documents) for documents in contexts]
    lengths = [len(value["input_ids"]) for value in sequences]
    if max(lengths) > max_tokens:
        raise RuntimeError(
            f"Direct prompt exceeds --max-input-tokens: {row['sample_id']} max={max(lengths)} limit={max_tokens}"
        )
    scores: list[float] = []
    predictions: list[str] = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        maximum = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), maximum), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        for index, item in enumerate(batch):
            values = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
            ids[index, -len(values):] = values
            attention[index, -len(values):] = 1
        positions = attention.cumsum(dim=-1) - 1
        positions.masked_fill_(attention == 0, 0)
        outputs = model.model(
            input_ids=ids, attention_mask=attention, position_ids=positions,
            use_cache=False, return_dict=True,
        )
        target_ids = torch.full((len(batch),), int(target_id), dtype=torch.long, device=device)
        values, choices = token_logprob_and_choice_logits(model, outputs.last_hidden_state[:, -1], target_ids, choice_ids)
        scores.extend(float(x) for x in values.cpu().tolist())
        predictions.extend(CHOICES[int(index)] for index in choices.argmax(dim=-1).cpu().tolist())
        del outputs, ids, attention, positions, target_ids, values, choices
    return scores, predictions, lengths


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x, y = rankdata(left), rankdata(right)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pair_order_accuracy(left: Sequence[float], right: Sequence[float]) -> float | None:
    values: list[float] = []
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            ldiff, rdiff = left[i] - left[j], right[i] - right[j]
            if ldiff == 0 or rdiff == 0:
                values.append(0.5)
            else:
                values.append(float((ldiff > 0) == (rdiff > 0)))
    return float(np.mean(values)) if values else None


def pairwise_support_accuracy(row: dict[str, Any], field: str) -> float | None:
    values = row[field]
    support = [values[i] for i, label in enumerate(row["semantic_labels"]) if label in SUPPORT_LABELS]
    non = [values[i] for i, label in enumerate(row["semantic_labels"]) if label in NON_SUPPORT_LABELS]
    if not support or not non:
        return None
    pairs = [1.0 if a > b else 0.5 if a == b else 0.0 for a in support for b in non]
    return float(np.mean(pairs))


def auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def comparison_metrics(rows: Sequence[dict[str, Any]], left_field: str, right_field: str) -> dict[str, Any]:
    correlations = [spearman(row[left_field], row[right_field]) for row in rows]
    correlations = [value for value in correlations if value is not None]
    ordering = [pair_order_accuracy(row[left_field], row[right_field]) for row in rows]
    ordering = [value for value in ordering if value is not None]
    top1 = [float(np.argmax(row[left_field]) == np.argmax(row[right_field])) for row in rows]
    pooled_left = [value for row in rows for value in row[left_field]]
    pooled_right = [value for row in rows for value in row[right_field]]
    return {
        "questions": len(rows),
        "documents": len(pooled_left),
        "questions_with_defined_spearman": len(correlations),
        "mean_question_spearman": float(np.mean(correlations)) if correlations else None,
        "median_question_spearman": float(np.median(correlations)) if correlations else None,
        "mean_question_pair_order_accuracy": float(np.mean(ordering)) if ordering else None,
        "top1_overlap": float(np.mean(top1)) if top1 else None,
        "pooled_spearman_secondary": spearman(pooled_left, pooled_right),
    }


def metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gradient = comparison_metrics(rows, "gradient_attribution_abs", "loo_effect_abs")
    context = comparison_metrics(rows, "loo_effect_abs", "singleton_effect_abs")
    removal_flips = [value for row in rows for value in row["removal_answer_flips"]]
    loo = [value for row in rows for value in row["loo_effect_abs"]]
    support_loo = [pairwise_support_accuracy(row, "loo_effect_abs") for row in rows]
    support_singleton = [pairwise_support_accuracy(row, "singleton_effect_abs") for row in rows]
    block: dict[str, Any] = {
        "questions": len(rows),
        "documents": len(loo),
        "direct_choice_gradient_vs_exact_loo": gradient,
        "exact_loo_vs_no_rag_singleton_addition": context,
        "removal_answer_flip_rate": float(np.mean(removal_flips)) if removal_flips else None,
        "loo_effect_auc_for_removal_answer_flip": auc(removal_flips, loo),
        "mean_loo_effect_when_answer_flips": float(np.mean([x for x, y in zip(loo, removal_flips) if y])) if any(removal_flips) else None,
        "mean_loo_effect_when_answer_does_not_flip": float(np.mean([x for x, y in zip(loo, removal_flips) if not y])) if not all(removal_flips) else None,
        "support_over_non_support_loo_pair_accuracy": float(np.mean([x for x in support_loo if x is not None])) if any(x is not None for x in support_loo) else None,
        "support_over_non_support_singleton_pair_accuracy": float(np.mean([x for x in support_singleton if x is not None])) if any(x is not None for x in support_singleton) else None,
        "mean_absolute_loo_effect": float(np.mean(loo)) if loo else None,
        "low_signal_question_fraction_max_loo_below_0p01": float(np.mean([max(row["loo_effect_abs"]) < 0.01 for row in rows])),
        "low_signal_question_fraction_max_loo_below_0p05": float(np.mean([max(row["loo_effect_abs"]) < 0.05 for row in rows])),
    }
    for size in (2, 4):
        top = [row["coalition_deletion"][str(size)]["top_abs_effect"] for row in rows]
        bottom = [row["coalition_deletion"][str(size)]["bottom_abs_effect"] for row in rows]
        block[f"top{size}_vs_bottom{size}_coalition_win_rate"] = float(np.mean([
            1.0 if a > b else 0.5 if a == b else 0.0 for a, b in zip(top, bottom)
        ]))
        block[f"mean_top{size}_coalition_effect"] = float(np.mean(top))
        block[f"mean_bottom{size}_coalition_effect"] = float(np.mean(bottom))
    return block


def bootstrap(
    rows: Sequence[dict[str, Any]], replicates: int, seed: int,
    *, prior_by_id: dict[str, dict[str, Any]] | None = None,
    progress: HierarchicalProgress | None = None,
) -> dict[str, Any]:
    if len(rows) < 2 or replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "gradient_mean_question_spearman": [],
        "gradient_top1_overlap": [],
        "loo_singleton_mean_question_spearman": [],
        "loo_singleton_top1_overlap": [],
        "top2_coalition_win_rate": [],
        "top4_coalition_win_rate": [],
        "direct_minus_rationale_mean_question_spearman": [],
        "direct_minus_rationale_top1_overlap": [],
    }
    for _ in range(replicates):
        sampled = [rows[int(index)] for index in rng.integers(0, len(rows), size=len(rows))]
        summary = metric_summary(sampled)
        gradient = summary["direct_choice_gradient_vs_exact_loo"]
        context = summary["exact_loo_vs_no_rag_singleton_addition"]
        candidates = {
            "gradient_mean_question_spearman": gradient["mean_question_spearman"],
            "gradient_top1_overlap": gradient["top1_overlap"],
            "loo_singleton_mean_question_spearman": context["mean_question_spearman"],
            "loo_singleton_top1_overlap": context["top1_overlap"],
            "top2_coalition_win_rate": summary["top2_vs_bottom2_coalition_win_rate"],
            "top4_coalition_win_rate": summary["top4_vs_bottom4_coalition_win_rate"],
        }
        for key, value in candidates.items():
            if value is not None:
                values[key].append(float(value))
        if prior_by_id is not None:
            prior_sampled = [prior_by_id[str(row["sample_id"])] for row in sampled]
            prior_metrics = comparison_metrics(prior_sampled, "attribution_abs", "removal_effect_abs")
            if gradient["mean_question_spearman"] is not None and prior_metrics["mean_question_spearman"] is not None:
                values["direct_minus_rationale_mean_question_spearman"].append(
                    float(gradient["mean_question_spearman"] - prior_metrics["mean_question_spearman"])
                )
            if gradient["top1_overlap"] is not None and prior_metrics["top1_overlap"] is not None:
                values["direct_minus_rationale_top1_overlap"].append(
                    float(gradient["top1_overlap"] - prior_metrics["top1_overlap"])
                )
        if progress is not None and (_ + 1) % 10 == 0:
            progress.set(_ + 1)
    return {
        key: [float(x) for x in np.quantile(items, [0.025, 0.975])] if items else None
        for key, items in values.items()
    }


def prior_reference(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    all_metrics = report.get("summary", {}).get("metrics", {}).get("all")
    if not isinstance(all_metrics, dict):
        raise RuntimeError(f"Invalid prior rationale+answer report: {path}")
    return {
        "path": str(path.resolve()),
        "questions": all_metrics.get("questions"),
        "pooled_spearman_gradient_vs_removal": all_metrics.get("pooled_spearman_attribution_vs_removal"),
        "median_question_spearman_gradient_vs_removal": all_metrics.get("median_question_spearman"),
        "top1_overlap_gradient_vs_removal": all_metrics.get("top1_overlap"),
    }


def load_prior_rows(path: Path, sample_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    rows_dir = path.resolve().parent / "rows"
    if not rows_dir.is_dir():
        raise FileNotFoundError(f"Missing prior rationale+answer row directory: {rows_dir}")
    output: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24] + ".json"
        row_path_value = rows_dir / name
        if not row_path_value.is_file():
            raise FileNotFoundError(f"Missing prior row: {sample_id} -> {row_path_value}")
        row = json.loads(row_path_value.read_text(encoding="utf-8"))
        if str(row.get("sample_id")) != sample_id:
            raise RuntimeError(f"Prior row identity mismatch: {sample_id}")
        if len(row.get("attribution_abs") or []) != 8 or len(row.get("removal_effect_abs") or []) != 8:
            raise RuntimeError(f"Invalid prior attribution row: {sample_id}")
        output[sample_id] = row
    return output


def report_markdown(report: dict[str, Any]) -> str:
    all_metrics = report["summary"]["all"]
    gradient = all_metrics["direct_choice_gradient_vs_exact_loo"]
    context = all_metrics["exact_loo_vs_no_rag_singleton_addition"]
    prior = report.get("prior_rationale_answer_reference")
    paired_prior = report["summary"].get("prior_rationale_answer_recomputed_same_cohort")
    fmt = lambda value: "NA" if value is None else f"{value:.4f}"
    lines = [
        "# Direct-Choice document attribution validity",
        "",
        "Top-8에서 생성한 Direct-Choice 한 토큰만 고정하여 문서 간 상대 영향력을 평가했다. Gold answer와 Semantic label은 진단에만 사용했다.",
        "",
        "## Experiment 1: Gradient×Input versus exact document removal",
        "",
        "| Target | Questions | Mean within-question Spearman | Median within-question Spearman | Top-1 overlap | Pair-order accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Direct Choice | {gradient['questions']} | {fmt(gradient['mean_question_spearman'])} | {fmt(gradient['median_question_spearman'])} | {fmt(gradient['top1_overlap'])} | {fmt(gradient['mean_question_pair_order_accuracy'])} |",
    ]
    if paired_prior is not None:
        lines.append(
            f"| Prior rationale+answer | {paired_prior['questions']} | {fmt(paired_prior['mean_question_spearman'])} | "
            f"{fmt(paired_prior['median_question_spearman'])} | {fmt(paired_prior['top1_overlap'])} | "
            f"{fmt(paired_prior['mean_question_pair_order_accuracy'])} |"
        )
        lines.append("")
        lines.append(
            f"Paired Direct-minus-rationale difference: mean within-question Spearman "
            f"{fmt(report['summary']['paired_direct_minus_rationale']['mean_question_spearman'])}, "
            f"Top-1 overlap {fmt(report['summary']['paired_direct_minus_rationale']['top1_overlap'])}."
        )
    lines.extend([
        "",
        "## Experiment 2: scope of exact removal attribution",
        "",
        "| Comparison | Mean within-question Spearman | Top-1 overlap | Pair-order accuracy |",
        "|---|---:|---:|---:|",
        f"| Top-8 removal vs No-RAG singleton addition | {fmt(context['mean_question_spearman'])} | {fmt(context['top1_overlap'])} | {fmt(context['mean_question_pair_order_accuracy'])} |",
        "",
        f"- Top-2 influential-document coalition beats bottom-2: {fmt(all_metrics['top2_vs_bottom2_coalition_win_rate'])}",
        f"- Top-4 influential-document coalition beats bottom-4: {fmt(all_metrics['top4_vs_bottom4_coalition_win_rate'])}",
        f"- Removing one document changes the constrained answer in {fmt(all_metrics['removal_answer_flip_rate'])} of document interventions.",
        f"- Low-signal questions (largest removal effect <0.01): {fmt(all_metrics['low_signal_question_fraction_max_loo_below_0p01'])}",
        "",
        "Exact LOO is a valid conditional effect in the current Top-8 by definition. Agreement with singleton addition and coalition deletion determines whether it can also be treated as a stable relative document property.",
    ])
    return "\n".join(lines) + "\n"


def row_path(output: Path, sample_id: str) -> Path:
    name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return output / "rows" / f"{name}.json"


def valid_cached_row(path: Path, sample_id: str, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("run_version") != RUN_VERSION or row.get("sample_id") != sample_id or row.get("contract_fingerprint") != fingerprint:
        return None
    return row


def main() -> None:
    args = parse_args()
    if args.context_batch_size <= 0 or args.max_input_tokens <= 0 or args.bootstrap_replicates < 0:
        raise ValueError("Invalid batch, token, or bootstrap setting")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress = HierarchicalProgress(
        ["validate frozen cohort and Direct-Choice prompt contract", "score Direct-Choice gradients and causal interventions", "bootstrap metrics and write report"],
        [15.0, 900.0, 25.0],
    )
    try:
        progress.start(1, 3, "check")
        cohort = load_cohort(args.cohort_file, args.max_questions)
        progress.update()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("Fast tokenizer is required for exact document token spans")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        max_seen = 0
        for row in cohort:
            sequence = direct_sequence(tokenizer, row, row["documents"])
            max_seen = max(max_seen, len(sequence["input_ids"]))
        if max_seen > args.max_input_tokens:
            raise RuntimeError(f"Full Top-8 prompt exceeds max input tokens: observed={max_seen} limit={args.max_input_tokens}")
        progress.update()
        prior = prior_reference(args.prior_report)
        progress.update()
        progress.complete(f"questions={len(cohort)} max_full_prompt_tokens={max_seen} prior_reference={'yes' if prior else 'no'}")

        contract = {
            "run_version": RUN_VERSION,
            "purpose": "test_direct_choice_document_attribution_relative_influence_and_context_stability",
            "cohort": file_identity(args.cohort_file, content_hash=True),
            "cohort_questions": len(cohort),
            "same_cohort_as_prior_rationale_answer_test": True,
            "prior_report": file_identity(args.prior_report, content_hash=True) if args.prior_report.is_file() else None,
            "prior_complete": file_identity(args.prior_report.parent / "COMPLETE.json", content_hash=True)
            if (args.prior_report.parent / "COMPLETE.json").is_file() else None,
            "model": model_identity(args.model),
            "prompt_policy_version": PROMPT_POLICY_VERSION,
            "prompt_order": "question_options_then_documents",
            "response_target": "top8_frozen_model_constrained_direct_choice_without_rationale",
            "score_version": SCORE_VERSION,
            "gradient_attribution": GRADIENT_ATTRIBUTION,
            "removal_attribution": REMOVAL_ATTRIBUTION,
            "singleton_attribution": SINGLETON_ATTRIBUTION,
            "gold_answer_use": "diagnostic_correct_wrong_subgroups_only",
            "semantic_label_use": "diagnostic_support_non_support_subgroups_only",
            "context_batch_size": args.context_batch_size,
            "max_input_tokens": args.max_input_tokens,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "pre_registered_interpretation": {
                "gradient_proxy_pass": "mean within-question Spearman >=0.30 and Top-1 overlap >=0.25",
                "loo_context_stability_pass": "mean within-question Spearman >=0.30 and Top-1 overlap >=0.25",
                "loo_coalition_consistency_pass": "Top-vs-bottom coalition win rate >=0.60 for both sizes 2 and 4",
                "scope": "LOO remains a full-Top8 conditional effect even when portability checks fail",
            },
            "code_commit_before_run": git_commit(),
        }
        fingerprint = canonical_hash(contract)
        manifest_path = output / "experiment_manifest.json"
        manifest = {**contract, "contract_fingerprint": fingerprint, "created_at": utc_now()}
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract_fingerprint") != fingerprint:
                raise RuntimeError("Output contract mismatch; use a new versioned output directory")
        else:
            atomic_json(manifest_path, manifest)
        if args.preflight_only:
            progress.finish(f"preflight passed; manifest={manifest_path}")
            return

        cached: dict[str, dict[str, Any]] = {}
        for row in cohort:
            path = row_path(output, row["sample_id"])
            value = valid_cached_row(path, row["sample_id"], fingerprint) if args.resume else None
            if value is not None:
                cached[row["sample_id"]] = value
        progress.start(2, len(cohort), "question", initial=len(cached))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if args.attn_implementation == "sdpa":
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        device = torch.device(args.device)
        print(f"[model loading] {args.model} device={args.device} dtype={args.dtype}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=dtype,
            attn_implementation=args.attn_implementation, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        model.requires_grad_(False)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        choice_ids = choice_token_ids(tokenizer, device)

        for row in cohort:
            sample_id = str(row["sample_id"])
            if sample_id in cached:
                continue
            documents = list(row["documents"])
            full_sequence = direct_sequence(tokenizer, row, documents)
            prediction, target_id, full_score, full_choice_logits, grad_signed, grad_abs = full_gradient_attribution(
                model, full_sequence, choice_ids, device,
            )
            contexts: list[list[dict[str, Any]]] = [[]]
            contexts.extend([[document] for document in documents])
            contexts.extend([[doc for j, doc in enumerate(documents) if j != removed] for removed in range(8)])
            scores, predictions, lengths = score_contexts(
                model, tokenizer, row, contexts, target_id, choice_ids, device,
                args.context_batch_size, args.max_input_tokens,
            )
            no_rag_score = scores[0]
            singleton_scores = scores[1:9]
            removed_scores = scores[9:17]
            removed_predictions = predictions[9:17]
            loo_signed = [full_score - value for value in removed_scores]
            loo_abs = [abs(value) for value in loo_signed]
            singleton_signed = [value - no_rag_score for value in singleton_scores]
            singleton_abs = [abs(value) for value in singleton_signed]

            order = sorted(range(8), key=lambda index: (-loo_abs[index], index))
            coalition_contexts: list[list[dict[str, Any]]] = []
            coalition_keys: list[tuple[int, str]] = []
            for size in (2, 4):
                top, bottom = set(order[:size]), set(order[-size:])
                coalition_contexts.append([doc for index, doc in enumerate(documents) if index not in top])
                coalition_keys.append((size, "top"))
                coalition_contexts.append([doc for index, doc in enumerate(documents) if index not in bottom])
                coalition_keys.append((size, "bottom"))
            coalition_scores, coalition_predictions, coalition_lengths = score_contexts(
                model, tokenizer, row, coalition_contexts, target_id, choice_ids, device,
                args.context_batch_size, args.max_input_tokens,
            )
            coalition: dict[str, dict[str, Any]] = {"2": {}, "4": {}}
            for (size, kind), value, answer in zip(coalition_keys, coalition_scores, coalition_predictions):
                coalition[str(size)][f"{kind}_score"] = value
                coalition[str(size)][f"{kind}_abs_effect"] = abs(full_score - value)
                coalition[str(size)][f"{kind}_answer"] = answer

            result = {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "sample_id": sample_id,
                "cohort": str(row["cohort"]),
                "semantic_labels": list(row["semantic_labels"]),
                "gold_answer": str(row["gold_answer"]),
                "direct_choice_answer": prediction,
                "direct_choice_correct": prediction == str(row["gold_answer"]),
                "target_token_id": target_id,
                "full_choice_logits": full_choice_logits,
                "full_score": full_score,
                "no_rag_score": no_rag_score,
                "singleton_scores": singleton_scores,
                "removed_scores": removed_scores,
                "gradient_attribution_signed": grad_signed,
                "gradient_attribution_abs": grad_abs,
                "loo_effect_signed": loo_signed,
                "loo_effect_abs": loo_abs,
                "singleton_effect_signed": singleton_signed,
                "singleton_effect_abs": singleton_abs,
                "removal_answers": removed_predictions,
                "removal_answer_flips": [answer != prediction for answer in removed_predictions],
                "coalition_deletion": coalition,
                "document_pair_ids": [str(doc["pair_id"]) for doc in documents],
                "document_sources": [str(doc.get("source") or "unknown") for doc in documents],
                "document_rerank_ranks": [int(doc["rerank_rank"]) for doc in documents],
                "document_char_counts": [len(str(doc["text"])) for doc in documents],
                "document_token_counts": full_sequence["document_token_counts"],
                "prompt_tokens": len(full_sequence["input_ids"]),
                "intervention_prompt_tokens": lengths,
                "coalition_prompt_tokens": coalition_lengths,
            }
            atomic_json(row_path(output, sample_id), result)
            cached[sample_id] = result
            progress.update()
        progress.complete(f"durable_rows={len(cached)} rows_dir={output / 'rows'}")

        ordered_rows = [cached[str(row["sample_id"])] for row in cohort]
        progress.start(3, max(1, args.bootstrap_replicates), "replicate")
        groups = {
            "all": ordered_rows,
            "mixed": [row for row in ordered_rows if row["cohort"] == "mixed"],
            "all_non_support": [row for row in ordered_rows if row["cohort"] == "all_non_support"],
            "direct_answer_correct": [row for row in ordered_rows if row["direct_choice_correct"]],
            "direct_answer_wrong": [row for row in ordered_rows if not row["direct_choice_correct"]],
        }
        summaries = {name: metric_summary(values) for name, values in groups.items()}
        prior_by_id = load_prior_rows(args.prior_report, [str(row["sample_id"]) for row in ordered_rows])
        prior_rows = [prior_by_id[str(row["sample_id"])] for row in ordered_rows]
        prior_same_cohort = comparison_metrics(prior_rows, "attribution_abs", "removal_effect_abs")
        intervals = bootstrap(
            ordered_rows, args.bootstrap_replicates, args.seed,
            prior_by_id=prior_by_id, progress=progress,
        )
        progress.set(max(1, args.bootstrap_replicates))
        gradient = summaries["all"]["direct_choice_gradient_vs_exact_loo"]
        context = summaries["all"]["exact_loo_vs_no_rag_singleton_addition"]
        decisions = {
            "gradient_proxy_pass": bool(
                gradient["mean_question_spearman"] is not None
                and gradient["mean_question_spearman"] >= 0.30
                and gradient["top1_overlap"] >= 0.25
            ),
            "loo_context_stability_pass": bool(
                context["mean_question_spearman"] is not None
                and context["mean_question_spearman"] >= 0.30
                and context["top1_overlap"] >= 0.25
            ),
            "loo_coalition_consistency_pass": bool(
                summaries["all"]["top2_vs_bottom2_coalition_win_rate"] >= 0.60
                and summaries["all"]["top4_vs_bottom4_coalition_win_rate"] >= 0.60
            ),
        }
        report = {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "completed_at": utc_now(),
            "prior_rationale_answer_reference": prior,
            "summary": {
                **summaries,
                "cluster_bootstrap_95ci": intervals,
                "prior_rationale_answer_recomputed_same_cohort": prior_same_cohort,
                "paired_direct_minus_rationale": {
                    "mean_question_spearman": (
                        gradient["mean_question_spearman"] - prior_same_cohort["mean_question_spearman"]
                        if gradient["mean_question_spearman"] is not None and prior_same_cohort["mean_question_spearman"] is not None
                        else None
                    ),
                    "top1_overlap": (
                        gradient["top1_overlap"] - prior_same_cohort["top1_overlap"]
                        if gradient["top1_overlap"] is not None and prior_same_cohort["top1_overlap"] is not None
                        else None
                    ),
                },
                "random_baselines": {"spearman": 0.0, "top1_overlap": 0.125, "pair_order_accuracy": 0.5},
                "pre_registered_decisions": decisions,
            },
        }
        atomic_jsonl(output / "per_question.jsonl", ordered_rows)
        atomic_json(output / "report.json", report)
        (output / "report.md").write_text(report_markdown(report), encoding="utf-8")
        atomic_json(output / "COMPLETE.json", {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "completed_at": utc_now(),
            "questions": len(ordered_rows),
            "report_sha256": sha256_file(output / "report.json"),
        })
        progress.complete(f"decisions={decisions} report={output / 'report.md'}")
        progress.finish(f"questions={len(ordered_rows)} output={output}")
    except Exception:
        print(
            f"[workflow FAILED] stage={progress.stage_index}/{len(progress.stages)} "
            f"completed={progress.stage_done}/{progress.stage_total} output={output}; "
            "rerun the identical command to resume durable rows",
            file=sys.stderr, flush=True,
        )
        raise


if __name__ == "__main__":
    main()

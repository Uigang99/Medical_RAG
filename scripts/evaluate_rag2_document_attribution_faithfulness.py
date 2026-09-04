#!/usr/bin/env python3
"""Test whether document Gradient×Input reflects realized-response influence.

This is a bounded, no-training feasibility test.  For each cached Top-8
rationale+answer response Y, it compares one-forward document attribution
against the change in mean token log-likelihood of that same Y after physically
removing each document.  Gold answers are used only to balance and stratify the
diagnostic cohort, never to define the score or attribution target.
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
from collections import Counter, defaultdict, deque
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

from generate_rag2_anchored_document_traces import document_pair_id  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    PROMPT_VERSION,
    build_anchored_user_prompt,
    normalized_mcq_row,
    render_chat_prompt,
    sha256_text,
)


RUN_VERSION = "rag2_realized_response_document_attribution_faithfulness_v1"
ATTRIBUTION_METHOD = "document_signed_sum_gradient_times_input_then_absolute_v1"
REMOVAL_TARGET = "absolute_delta_mean_token_log_likelihood_same_cached_response_v1"
SUPPORT_LABELS = frozenset({"direct_support", "supporting_evidence"})
NON_SUPPORT_LABELS = frozenset({"no_evidence", "misleading_evidence"})
EXCLUDED_LABELS = frozenset({"indeterminate_or_mixed"})
FATAL_TRACE_FLAGS = frozenset({"rationale_length_exhausted", "empty_rationale"})

DEFAULT_BASE = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
DEFAULT_CANDIDATE_ROOT = DEFAULT_BASE / "candidates/source_balanced32_rerank8_v1"
DEFAULT_TRACE_ROOT = DEFAULT_BASE / "semantic_subset_rationale_traces_v1/rationale_traces"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "document_attribution_faithfulness_mvp_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), default="medqa")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--analysis-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixed-questions", type=int, default=256)
    parser.add_argument("--all-non-support-questions", type=int, default=64)
    parser.add_argument("--expected-documents", type=int, default=8)
    parser.add_argument("--removal-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-length", type=int, default=8192)
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
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_identity(path: Path, *, hash_content: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def model_identity(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = [root / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json") if (root / name).is_file()]
    paths.extend(sorted(root.glob("*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No model files: {root}")
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            **({"sha256": file_identity(path, hash_content=True)["sha256"]} if path.stat().st_size < 16 * 1024 * 1024 else {}),
        }
        for path in paths
    ]


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
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from error


class WorkflowProgress:
    """Stable two-line stage progress; overall ETA appears after GPU calibration."""

    def __init__(self, stages: Sequence[str], future_estimates: Sequence[float | None]) -> None:
        if len(stages) != len(future_estimates):
            raise ValueError("stages and estimates mismatch")
        self.stages = list(stages)
        self.estimates = list(future_estimates)
        self.started = time.time()
        self.stage_started = self.started
        self.stage_index = 0
        self.stage_total = 1
        self.stage_done = 0
        self.stage_unit = "item"
        self.history: deque[tuple[float, int]] = deque()
        self.overall = tqdm(total=len(stages), desc="Overall", position=0, leave=True, dynamic_ncols=True)
        self.active: tqdm[Any] | None = None

    def start(self, index: int, total: int, unit: str, *, initial: int = 0) -> None:
        if self.active is not None:
            self.active.close()
        self.stage_index = index
        self.stage_started = time.time()
        self.stage_total = max(1, int(total))
        self.stage_done = min(max(0, int(initial)), self.stage_total)
        self.stage_unit = unit
        self.history.clear()
        self.history.append((self.stage_started, self.stage_done))
        self.active = tqdm(
            total=total, initial=initial, position=1, leave=False, dynamic_ncols=True, unit=unit,
            desc=f"Stage {index}/{len(self.stages)} - {self.stages[index-1]}",
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
        stage_eta = (self.stage_total - self.stage_done) / rate if rate else None
        future = self.estimates[self.stage_index:]
        overall_eta = None if stage_eta is None or any(value is None for value in future) else stage_eta + sum(float(value) for value in future)
        rate_text = "calibrating" if rate is None else f"{rate:.2f} {self.stage_unit}/s"
        if self.active is not None:
            self.active.set_postfix_str(f"{rate_text}, ETA {format_duration(stage_eta)}", refresh=False)
        self.overall.set_postfix_str(
            f"stage {self.stage_index}/{len(self.stages)}, elapsed {format_duration(now-self.started)}, ETA {format_duration(overall_eta)}",
            refresh=False,
        )

    def set(self, completed: int) -> None:
        completed = min(max(0, int(completed)), self.stage_total)
        delta = completed - self.stage_done
        self.stage_done = completed
        self.history.append((time.time(), completed))
        if self.active is not None and delta > 0:
            self.active.update(delta)
        self._render()

    def update(self, count: int = 1) -> None:
        self.set(self.stage_done + count)

    def complete(self, detail: str = "") -> None:
        self.set(self.stage_total)
        duration = time.time() - self.stage_started
        self.estimates[self.stage_index - 1] = duration
        if self.active is not None:
            self.active.close()
            self.active = None
        self.overall.n = self.stage_index
        self.overall.refresh()
        suffix = f" | {detail}" if detail else ""
        tqdm.write(f"[stage {self.stage_index}/{len(self.stages)} complete | duration {format_duration(duration)}]{suffix}")

    def finish(self, detail: str) -> None:
        if self.active is not None:
            self.active.close()
        self.overall.n = len(self.stages)
        self.overall.set_postfix_str(
            f"stage {len(self.stages)}/{len(self.stages)}, elapsed {format_duration(time.time()-self.started)}, ETA 00h00m00s"
        )
        self.overall.close()
        print(f"[workflow complete] {detail}", flush=True)


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def trace_category(labels: Sequence[str]) -> str | None:
    values = set(labels)
    if values & EXCLUDED_LABELS or values - SUPPORT_LABELS - NON_SUPPORT_LABELS:
        return None
    has_support = bool(values & SUPPORT_LABELS)
    has_non_support = bool(values & NON_SUPPORT_LABELS)
    if has_support and has_non_support:
        return "mixed"
    if has_non_support and not has_support:
        return "all_non_support"
    return None


def select_balanced(
    records: Sequence[dict[str, Any]], category: str, count: int, seed: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    selected: list[dict[str, Any]] = []
    needs = {False: count // 2, True: count - count // 2}
    for correctness, need in needs.items():
        bucket = [row for row in records if row["cohort"] == category and bool(row["answer_correct"]) == correctness]
        bucket.sort(key=lambda row: stable_key(seed, row["sample_id"]))
        if len(bucket) < need:
            raise RuntimeError(
                f"Insufficient {category} answer_correct={correctness}: required={need} available={len(bucket)}"
            )
        selected.extend(bucket[:need])
    return sorted(selected, key=lambda row: stable_key(seed + 1, row["sample_id"]))


def compact_trace(row: dict[str, Any], expected_documents: int, analysis_split: str) -> dict[str, Any] | None:
    if "all_top8" not in set(row.get("policies") or []):
        return None
    if str(row.get("analysis_split")) != analysis_split:
        return None
    labels = [str(value) for value in row.get("semantic_labels") or []]
    documents = list(row.get("documents") or [])
    if len(labels) != expected_documents or len(documents) != expected_documents:
        return None
    category = trace_category(labels)
    if category is None:
        return None
    flags = set(row.get("quality_flags") or [])
    response = str(row.get("canonical_response") or "")
    if flags & FATAL_TRACE_FLAGS or not response.strip():
        return None
    ranks = [int(item.get("rerank_rank", -1)) for item in documents]
    if ranks != list(range(1, expected_documents + 1)):
        raise RuntimeError(f"Invalid trace document ranks: {row.get('sample_id')} {ranks}")
    return {
        "sample_id": str(row["sample_id"]),
        "row_idx": int(row["row_idx"]),
        "analysis_split": analysis_split,
        "cohort": category,
        "semantic_labels": labels,
        "trace_documents": documents,
        "canonical_response": response,
        "answer": str(row["answer"]),
        "gold_answer": str(row["gold_answer"]),
        "answer_correct": bool(row["answer_correct"]),
        "user_prompt_sha256": str(row["user_prompt_sha256"]),
        "quality_flags": sorted(flags),
    }


def scan_traces(args: argparse.Namespace, progress: WorkflowProgress) -> tuple[list[dict[str, Any]], int]:
    shard_paths = sorted((args.trace_root / "trace_shards" / args.dataset / args.source_split).glob("shard_*/subsets.jsonl"))
    if not shard_paths:
        raise FileNotFoundError("No semantic subset trace shards")
    progress.start(1, len(shard_paths), "shard")
    eligible: list[dict[str, Any]] = []
    all_top8 = 0
    seen: set[str] = set()
    for index, path in enumerate(shard_paths, 1):
        for raw in iter_jsonl(path):
            if "all_top8" in set(raw.get("policies") or []) and str(raw.get("analysis_split")) == args.analysis_split:
                all_top8 += 1
            row = compact_trace(raw, args.expected_documents, args.analysis_split)
            if row is not None:
                if row["sample_id"] in seen:
                    raise RuntimeError(f"Duplicate all_top8 trace: {row['sample_id']}")
                seen.add(row["sample_id"])
                eligible.append(row)
        progress.set(index)
    counts = Counter((row["cohort"], row["answer_correct"]) for row in eligible)
    progress.complete(f"all_top8={all_top8} eligible={len(eligible)} buckets={dict(counts)}")
    return eligible, all_top8


def normalize_candidate_documents(sample_id: str, raw: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    documents = sorted(list(raw.get("candidate_documents") or []), key=lambda item: int(item.get("rerank_rank", 10**9)))
    if len(documents) != expected:
        raise RuntimeError(f"Expected {expected} candidate documents: {sample_id}")
    result = []
    for fallback_rank, original in enumerate(documents, 1):
        document = dict(original)
        rank = int(document.get("rerank_rank") or fallback_rank)
        if rank != fallback_rank:
            raise RuntimeError(f"Non-contiguous candidate ranks: {sample_id}")
        text = str(document.get("text") or "").strip()
        if not text:
            raise RuntimeError(f"Empty candidate document: {sample_id}:{rank}")
        document["text"] = text
        document["pair_id"] = document_pair_id(sample_id, document, rank)
        result.append(document)
    return result


def join_candidates(
    args: argparse.Namespace, selected: Sequence[dict[str, Any]], progress: WorkflowProgress,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_path = args.candidate_root / args.dataset / args.source_split / "candidates_top8.jsonl"
    candidate_manifest_path = candidate_path.parent / "candidate_manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    expected_rows = int(candidate_manifest.get("selected_question_count", -1))
    if expected_rows <= 0 or int(candidate_manifest.get("top_k", -1)) != args.expected_documents:
        raise RuntimeError("Candidate manifest is not the expected Top-8 contract")
    wanted = {row["sample_id"]: row for row in selected}
    assembled: list[dict[str, Any]] = []
    progress.start(2, expected_rows, "question")
    rows_seen = 0
    for raw in iter_jsonl(candidate_path):
        rows_seen += 1
        sample_id = str(raw.get("sample_id"))
        if sample_id in wanted:
            trace = wanted[sample_id]
            question = normalized_mcq_row(raw)
            documents = normalize_candidate_documents(sample_id, raw, args.expected_documents)
            for rank, (candidate, expected) in enumerate(zip(documents, trace["trace_documents"]), 1):
                if candidate["pair_id"] != expected.get("pair_id"):
                    raise RuntimeError(f"Trace/candidate pair mismatch: {sample_id}:{rank}")
                if sha256_text(candidate["text"]) != expected.get("text_sha256"):
                    raise RuntimeError(f"Trace/candidate text mismatch: {sample_id}:{rank}")
            if question["answer"] != trace["gold_answer"]:
                raise RuntimeError(f"Gold answer mismatch: {sample_id}")
            document_blob = "\n\n".join(item["text"] for item in documents)
            user_prompt = build_anchored_user_prompt(question, document_blob)
            if sha256_text(user_prompt) != trace["user_prompt_sha256"]:
                raise RuntimeError(f"Cached prompt replay mismatch: {sample_id}")
            assembled.append(
                {
                    **{key: value for key, value in trace.items() if key != "trace_documents"},
                    "question": question["question"],
                    "options": question["options"],
                    "documents": documents,
                }
            )
        if rows_seen % 512 == 0 or rows_seen == expected_rows:
            progress.set(rows_seen)
    if rows_seen != expected_rows:
        raise RuntimeError(f"Candidate row count mismatch: expected={expected_rows} actual={rows_seen}")
    if len(assembled) != len(selected):
        missing = sorted(set(wanted) - {row["sample_id"] for row in assembled})
        raise RuntimeError(f"Missing candidate rows: {missing[:5]} (count={len(missing)})")
    assembled.sort(key=lambda row: stable_key(args.seed + 2, row["sample_id"]))
    progress.complete(f"selected={len(assembled)} candidate_rows={rows_seen}")
    return assembled, {"candidate": file_identity(candidate_path), "candidate_manifest": file_identity(candidate_manifest_path, hash_content=True)}


def response_sequence(tokenizer: Any, row: dict[str, Any], documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    document_blob = "\n\n".join(str(item["text"]).strip() for item in documents)
    user_prompt = build_anchored_user_prompt(row, document_blob)
    chat_prompt = render_chat_prompt(tokenizer, user_prompt)
    response = str(row["canonical_response"])
    prompt_ids = list(tokenizer.encode(chat_prompt, add_special_tokens=False))
    encoded = tokenizer(chat_prompt + response, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"Prompt is not a stable token prefix: {row['sample_id']}")
    response_ids = input_ids[len(prompt_ids):]
    if not response_ids:
        raise RuntimeError(f"Empty response token sequence: {row['sample_id']}")
    document_start = chat_prompt.rfind(document_blob)
    if document_start < 0:
        raise RuntimeError(f"Cannot locate document block: {row['sample_id']}")
    spans: list[tuple[int, int]] = []
    cursor = document_start
    for index, document in enumerate(documents):
        text = str(document["text"]).strip()
        if (chat_prompt + response)[cursor:cursor + len(text)] != text:
            raise RuntimeError(f"Document character replay mismatch: {row['sample_id']}:{index+1}")
        spans.append((cursor, cursor + len(text)))
        cursor += len(text)
        if index + 1 < len(documents):
            if (chat_prompt + response)[cursor:cursor + 2] != "\n\n":
                raise RuntimeError(f"Document separator mismatch: {row['sample_id']}:{index+1}")
            cursor += 2
    token_indices = [
        [position for position, (start, end) in enumerate(offsets) if end > span_start and start < span_end]
        for span_start, span_end in spans
    ]
    if any(not values for values in token_indices):
        raise RuntimeError(f"Document has no overlapping tokens: {row['sample_id']}")
    return {
        "input_ids": input_ids,
        "prompt_tokens": len(prompt_ids),
        "response_ids": response_ids,
        "document_token_indices": token_indices,
        "document_token_counts": [len(values) for values in token_indices],
        "total_tokens": len(input_ids),
    }


def score_from_hidden(model: Any, hidden: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    selected = hidden.index_select(1, positions)
    logits = model.lm_head(selected).float()
    target_logits = logits.gather(-1, targets.view(1, -1, 1).expand(logits.shape[0], -1, 1)).squeeze(-1)
    return (target_logits - torch.logsumexp(logits, dim=-1)).mean(dim=-1)


def full_score_and_attribution(model: Any, sequence: dict[str, Any], device: torch.device) -> tuple[float, list[float], list[float]]:
    ids = torch.tensor(sequence["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    prompt_tokens = int(sequence["prompt_tokens"])
    positions = torch.arange(prompt_tokens - 1, ids.shape[1] - 1, dtype=torch.long, device=device)
    targets = ids[0, prompt_tokens:]
    embeddings = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    output = model.model(
        inputs_embeds=embeddings,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        return_dict=True,
    )
    score = score_from_hidden(model, output.last_hidden_state, positions, targets)[0]
    gradient = torch.autograd.grad(score, embeddings, retain_graph=False, create_graph=False)[0]
    token_attribution = (gradient.float() * embeddings.float()).sum(dim=-1)[0]
    signed = [float(token_attribution[indices].sum().item()) for indices in sequence["document_token_indices"]]
    del output, gradient, embeddings, token_attribution
    return float(score.detach().item()), signed, [abs(value) for value in signed]


@torch.inference_mode()
def removed_scores(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    sequences = [
        response_sequence(tokenizer, row, [doc for j, doc in enumerate(row["documents"]) if j != removed])
        for removed in range(len(row["documents"]))
    ]
    expected_response = sequences[0]["response_ids"]
    if any(sequence["response_ids"] != expected_response for sequence in sequences[1:]):
        raise RuntimeError(f"Response tokenization changed across removals: {row['sample_id']}")
    results: list[float] = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        maximum = max(sequence["total_tokens"] for sequence in batch)
        input_ids = torch.full((len(batch), maximum), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        response_length = len(expected_response)
        predictor_positions: list[list[int]] = []
        for row_index, sequence in enumerate(batch):
            values = torch.tensor(sequence["input_ids"], dtype=torch.long, device=device)
            left = maximum - len(values)
            input_ids[row_index, left:] = values
            attention[row_index, left:] = 1
            positions = list(range(left + sequence["prompt_tokens"] - 1, maximum - 1))
            if len(positions) != response_length:
                raise RuntimeError("Removal predictor/response length mismatch")
            predictor_positions.append(positions)
        if any(values != predictor_positions[0] for values in predictor_positions[1:]):
            raise RuntimeError("Left padding did not align response predictor positions")
        position_ids = attention.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        positions = torch.tensor(predictor_positions[0], dtype=torch.long, device=device)
        targets = torch.tensor(expected_response, dtype=torch.long, device=device)
        results.extend(float(value) for value in score_from_hidden(model, output.last_hidden_state, positions, targets).cpu().tolist())
        del output, input_ids, attention, position_ids
    return results


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = rankdata(left)
    y = rankdata(right)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_support_accuracy(row: dict[str, Any], field: str) -> float | None:
    values = row[field]
    support = [values[index] for index, label in enumerate(row["semantic_labels"]) if label in SUPPORT_LABELS]
    non_support = [values[index] for index, label in enumerate(row["semantic_labels"]) if label in NON_SUPPORT_LABELS]
    if not support or not non_support:
        return None
    comparisons = [1.0 if left > right else 0.5 if left == right else 0.0 for left in support for right in non_support]
    return float(np.mean(comparisons))


def metric_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    attr = [value for row in rows for value in row["attribution_abs"]]
    removal = [value for row in rows for value in row["removal_effect_abs"]]
    per_question = [spearman(row["attribution_abs"], row["removal_effect_abs"]) for row in rows]
    per_question = [value for value in per_question if value is not None]
    top1 = [int(np.argmax(row["attribution_abs"]) == np.argmax(row["removal_effect_abs"])) for row in rows]
    support_attr = [pairwise_support_accuracy(row, "attribution_abs") for row in rows]
    support_removal = [pairwise_support_accuracy(row, "removal_effect_abs") for row in rows]
    return {
        "questions": len(rows),
        "documents": len(attr),
        "pooled_spearman_attribution_vs_removal": spearman(attr, removal),
        "median_question_spearman": float(np.median(per_question)) if per_question else None,
        "questions_with_defined_spearman": len(per_question),
        "top1_overlap": float(np.mean(top1)) if top1 else None,
        "support_over_non_support_attribution_pair_accuracy_macro": float(np.mean([x for x in support_attr if x is not None])) if any(x is not None for x in support_attr) else None,
        "support_over_non_support_removal_pair_accuracy_macro": float(np.mean([x for x in support_removal if x is not None])) if any(x is not None for x in support_removal) else None,
        "document_length_spearman_with_attribution": spearman(
            [value for row in rows for value in row["document_token_counts"]], attr,
        ),
        "rerank_rank_spearman_with_attribution": spearman(
            [value for row in rows for value in row["rerank_ranks"]], attr,
        ),
        "mean_full_response_log_likelihood": float(np.mean([row["full_score"] for row in rows])) if rows else None,
        "mean_absolute_removal_effect": float(np.mean(removal)) if removal else None,
    }


def bootstrap_ci(rows: Sequence[dict[str, Any]], replicates: int, seed: int) -> dict[str, list[float] | None]:
    if len(rows) < 2 or replicates <= 0:
        return {"pooled_spearman": None, "top1_overlap": None}
    rng = np.random.default_rng(seed)
    pooled: list[float] = []
    top1: list[float] = []
    for _ in range(replicates):
        sampled = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
        block = metric_block(sampled)
        if block["pooled_spearman_attribution_vs_removal"] is not None:
            pooled.append(block["pooled_spearman_attribution_vs_removal"])
        top1.append(block["top1_overlap"])
    interval = lambda values: [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None
    return {"pooled_spearman": interval(pooled), "top1_overlap": interval(top1)}


def summarize(rows: Sequence[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all": list(rows),
        "mixed": [row for row in rows if row["cohort"] == "mixed"],
        "all_non_support": [row for row in rows if row["cohort"] == "all_non_support"],
        "response_correct": [row for row in rows if row["answer_correct"]],
        "response_wrong": [row for row in rows if not row["answer_correct"]],
        "mixed_response_correct": [row for row in rows if row["cohort"] == "mixed" and row["answer_correct"]],
        "mixed_response_wrong": [row for row in rows if row["cohort"] == "mixed" and not row["answer_correct"]],
    }
    metrics = {name: metric_block(values) for name, values in groups.items()}
    metrics["all"]["cluster_bootstrap_95ci"] = bootstrap_ci(groups["all"], replicates, seed)
    primary = metrics["all"]
    correct = metrics["response_correct"]["pooled_spearman_attribution_vs_removal"]
    wrong = metrics["response_wrong"]["pooled_spearman_attribution_vs_removal"]
    pass_checks = {
        "pooled_spearman_at_least_0p30": primary["pooled_spearman_attribution_vs_removal"] is not None and primary["pooled_spearman_attribution_vs_removal"] >= 0.30,
        "top1_overlap_at_least_0p25": primary["top1_overlap"] is not None and primary["top1_overlap"] >= 0.25,
        "positive_in_correct_and_wrong_subgroups": correct is not None and wrong is not None and correct > 0 and wrong > 0,
    }
    return {
        "metrics": metrics,
        "baselines": {"random_expected_spearman": 0.0, "random_expected_top1_overlap": 0.125, "random_expected_pair_accuracy": 0.5},
        "pre_registered_success": {"checks": pass_checks, "passed": all(pass_checks.values())},
    }


def report_markdown(report: dict[str, Any]) -> str:
    metrics = report["summary"]["metrics"]
    lines = [
        "# Rationale+answer document attribution faithfulness MVP",
        "",
        "`Gradient×Input` 문서 attribution이 같은 응답을 teacher forcing할 때 문서 제거로 생기는 평균 token log-likelihood 변화와 일치하는지 평가한다.",
        "Gold answer는 표본 균형과 정답/오답 subgroup에만 사용되었다.",
        "",
        "| Cohort | Questions | Pooled Spearman | Median per-question Spearman | Top-1 overlap | Support > Non-support attribution |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "mixed", "all_non_support", "response_correct", "response_wrong"):
        item = metrics[name]
        fmt = lambda value: "NA" if value is None else f"{value:.4f}"
        lines.append(
            f"| {name} | {item['questions']} | {fmt(item['pooled_spearman_attribution_vs_removal'])} | "
            f"{fmt(item['median_question_spearman'])} | {fmt(item['top1_overlap'])} | "
            f"{fmt(item['support_over_non_support_attribution_pair_accuracy_macro'])} |"
        )
    success = report["summary"]["pre_registered_success"]
    lines.extend([
        "",
        f"**Pre-registered feasibility verdict:** {'PASS' if success['passed'] else 'FAIL'}",
        "",
        "- Random baselines: Spearman 0, Top-1 overlap 0.125, Support/Non-support pair accuracy 0.5.",
        "- This measures faithfulness to the cached realized rationale+answer, not the effect of free-regenerating a new answer after removal.",
    ])
    return "\n".join(lines) + "\n"


def row_output_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "rows" / f"{hashlib.sha256(sample_id.encode()).hexdigest()[:24]}.json"


def cached_row(path: Path, sample_id: str, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("run_version") == RUN_VERSION and row.get("sample_id") == sample_id and row.get("contract_fingerprint") == fingerprint:
        return row
    return None


def main() -> None:
    args = parse_args()
    if args.mixed_questions <= 0 or args.all_non_support_questions < 0:
        raise ValueError("Invalid cohort sizes")
    if args.removal_batch_size <= 0:
        raise ValueError("removal-batch-size must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = WorkflowProgress(
        ["scan cached Top-8 traces and select balanced cohort", "join exact candidate documents and freeze contract", "score full attribution and eight physical removals", "aggregate faithfulness metrics and decision"],
        [30.0, 15.0, None, 20.0],
    )

    try:
        generation_manifest_path = args.trace_root / "generation_manifest.json"
        generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        if generation_manifest.get("prompt_version") != PROMPT_VERSION or generation_manifest.get("direct_choice_generation") is not False:
            raise RuntimeError("Trace cache is not the canonical rationale+answer prompt contract")
        if str(Path(generation_manifest["model_name_or_path"]).resolve()) != str(args.model.resolve()):
            raise RuntimeError("Trace generator and attribution model differ")

        eligible, _ = scan_traces(args, progress)
        selected = select_balanced(eligible, "mixed", args.mixed_questions, args.seed)
        selected += select_balanced(eligible, "all_non_support", args.all_non_support_questions, args.seed + 1000)
        selected.sort(key=lambda row: stable_key(args.seed + 2000, row["sample_id"]))
        assembled, candidate_identity = join_candidates(args, selected, progress)

        contract = {
            "run_version": RUN_VERSION,
            "purpose": "test_one_forward_document_attribution_against_same_response_physical_removal",
            "dataset": args.dataset,
            "source_split": args.source_split,
            "analysis_split": args.analysis_split,
            "cohort": {
                "mixed_questions": args.mixed_questions,
                "all_non_support_questions": args.all_non_support_questions,
                "balanced_on_cached_response_correctness": True,
                "support_labels": sorted(SUPPORT_LABELS),
                "non_support_labels": sorted(NON_SUPPORT_LABELS),
                "excluded_labels": sorted(EXCLUDED_LABELS),
                "seed": args.seed,
            },
            "score": "mean_token_log_likelihood_of_entire_cached_canonical_rationale_and_answer",
            "attribution_method": ATTRIBUTION_METHOD,
            "removal_target": REMOVAL_TARGET,
            "gold_answer_use": "cohort_balance_and_diagnostic_subgroups_only",
            "model": str(args.model.resolve()),
            "model_identity": model_identity(args.model),
            "prompt_version": PROMPT_VERSION,
            "generation_manifest": file_identity(generation_manifest_path, hash_content=True),
            "candidate_inputs": candidate_identity,
            "expected_documents": args.expected_documents,
            "max_model_length": args.max_model_length,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "pre_registered_success_rule": {
                "pooled_spearman": ">=0.30",
                "top1_overlap": ">=0.25",
                "pooled_spearman_in_correct_and_wrong_subgroups": ">0",
            },
            "code_commit_before_run": git_commit(),
        }
        fingerprint = canonical_hash(contract)
        manifest = {**contract, "contract_fingerprint": fingerprint, "created_at": utc_now(), "question_count": len(assembled)}
        manifest_path = output_dir / "experiment_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract_fingerprint") != fingerprint:
                raise RuntimeError("Output contract mismatch; use a new versioned output directory")
        else:
            atomic_json(manifest_path, manifest)
        cohort_path = output_dir / "cohort.jsonl"
        if not cohort_path.is_file():
            atomic_jsonl(cohort_path, assembled)
        else:
            cached_ids = [row["sample_id"] for row in iter_jsonl(cohort_path)]
            if cached_ids != [row["sample_id"] for row in assembled]:
                raise RuntimeError("Cached cohort does not match selected contract")

        if args.preflight_only:
            progress.finish(f"preflight-only cohort={cohort_path} manifest={manifest_path}")
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for attribution scoring")
        if args.attn_implementation == "sdpa":
            # This host's cuDNN SDPA planner rejects some long, gradient-bearing
            # causal shapes.  Keep PyTorch's fused/efficient/math SDPA paths.
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("A fast tokenizer is required for exact document token spans")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(args.device)
        model.eval()
        model.requires_grad_(False)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        device = torch.device(args.device)

        cached: dict[str, dict[str, Any]] = {}
        for row in assembled:
            path = row_output_path(output_dir, row["sample_id"])
            value = cached_row(path, row["sample_id"], fingerprint) if args.resume else None
            if value is not None:
                cached[row["sample_id"]] = value
        progress.start(3, len(assembled), "question", initial=len(cached))
        max_tokens_seen = 0
        for index, row in enumerate(assembled, 1):
            if row["sample_id"] in cached:
                continue
            sequence = response_sequence(tokenizer, row, row["documents"])
            max_tokens_seen = max(max_tokens_seen, sequence["total_tokens"])
            if sequence["total_tokens"] > args.max_model_length:
                raise RuntimeError(
                    f"Context exceeds max-model-length: {row['sample_id']} tokens={sequence['total_tokens']} limit={args.max_model_length}"
                )
            full_score, attribution_signed, attribution_abs = full_score_and_attribution(model, sequence, device)
            removal_scores = removed_scores(model, tokenizer, row, device, args.removal_batch_size)
            removal_effect = [abs(full_score - score) for score in removal_scores]
            result = {
                "run_version": RUN_VERSION,
                "contract_fingerprint": fingerprint,
                "sample_id": row["sample_id"],
                "cohort": row["cohort"],
                "answer_correct": row["answer_correct"],
                "semantic_labels": row["semantic_labels"],
                "pair_ids": [item["pair_id"] for item in row["documents"]],
                "sources": [str(item.get("source") or "unknown") for item in row["documents"]],
                "rerank_ranks": [int(item["rerank_rank"]) for item in row["documents"]],
                "document_char_counts": [len(item["text"]) for item in row["documents"]],
                "document_token_counts": sequence["document_token_counts"],
                "input_tokens": sequence["total_tokens"],
                "response_tokens": len(sequence["response_ids"]),
                "response_sha256": sha256_text(row["canonical_response"]),
                "full_score": full_score,
                "removed_scores": removal_scores,
                "removal_effect_abs": removal_effect,
                "attribution_signed": attribution_signed,
                "attribution_abs": attribution_abs,
                "attribution_abs_per_document_token": [
                    value / count for value, count in zip(attribution_abs, sequence["document_token_counts"])
                ],
            }
            atomic_json(row_output_path(output_dir, row["sample_id"]), result)
            cached[row["sample_id"]] = result
            progress.update()
            if index % 8 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        progress.complete(f"questions={len(cached)} max_input_tokens={max_tokens_seen} rows={output_dir / 'rows'}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.start(4, len(assembled), "question")
        result_rows: list[dict[str, Any]] = []
        for index, row in enumerate(assembled, 1):
            value = cached_row(row_output_path(output_dir, row["sample_id"]), row["sample_id"], fingerprint)
            if value is None:
                raise RuntimeError(f"Missing completed result: {row['sample_id']}")
            result_rows.append(value)
            progress.set(index)
        summary = summarize(result_rows, args.bootstrap_replicates, args.seed)
        report = {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "completed_at": utc_now(),
            "summary": summary,
        }
        atomic_json(output_dir / "report.json", report)
        markdown = output_dir / "report.md"
        temporary = markdown.with_name(markdown.name + f".tmp.{os.getpid()}")
        temporary.write_text(report_markdown(report), encoding="utf-8")
        temporary.replace(markdown)
        atomic_json(output_dir / "COMPLETE.json", {
            "run_version": RUN_VERSION,
            "contract_fingerprint": fingerprint,
            "question_count": len(result_rows),
            "passed": summary["pre_registered_success"]["passed"],
            "report": str((output_dir / "report.json").resolve()),
            "completed_at": utc_now(),
        })
        progress.complete(f"passed={summary['pre_registered_success']['passed']} report={markdown}")
        progress.finish(f"result={markdown}")
    except Exception:
        print(
            f"[workflow FAILED] stage={progress.stage_index}/{len(progress.stages)} "
            f"completed={progress.stage_done}/{progress.stage_total} output={output_dir}; "
            "rerun the identical command to resume durable question rows",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()

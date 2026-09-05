#!/usr/bin/env python3
"""Evaluate multi-token PCED under the anchored Rationale+Answer contract.

PCED fusion is applied from the first rationale token through the last
rationale token. The shared generated rationale is then placed into each
expert's canonical answer prefix and the same fixed beta/prior are used for
the constrained A/B/C/D decision. This avoids the fixed-rationale confound of
combining document experts only after an independently generated rationale.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rag2_pced_direct_choice import (  # noqa: E402
    CHOICES,
    DATASETS,
    MMLU_DATASETS,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    condition_summary,
    iter_jsonl,
    minmax,
    model_identity,
    paired_comparison,
    sample_key,
    sha256_file,
)
from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path  # noqa: E402
from medrag.core import BenchmarkSample  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    END_REASONING_MARKER,
    assistant_decision_prefix,
    build_anchored_user_prompt,
    canonical_response,
    normalize_rationale,
    rationale_generation_prompt,
    render_chat_prompt,
)


RUN_VERSION = "rag2_rationale_answer_pced_dynamic_topk_v1"
GENERATION_RULE = "tokenwise_pced_full_vocab_rationale_then_same_beta_constrained_choice_v1"
DEFAULT_LLAMA = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_NO_RAG = (
    PROJECT_ROOT
    / "databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1/"
    "no_rag_rationales_all_mcq_test/no_rag"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--semantic-score-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=PROJECT_ROOT / "datasets/benchmark")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--no-rag-root", type=Path, default=DEFAULT_NO_RAG)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--prior-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-rationale-tokens", type=int, default=512)
    parser.add_argument("--answer-reserve-tokens", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def pair_key(sample: BenchmarkSample, document: dict[str, Any]) -> str:
    identifier = document.get("stable_id") or document.get("corpus_id") or document.get("db_id")
    rank = int(document.get("rerank_rank", -1))
    if not identifier or rank <= 0:
        raise RuntimeError(f"Cannot identify document pair: {sample.id}")
    return f"{sample.id}::{rank}::{identifier}"


def load_inputs(args: argparse.Namespace) -> tuple[list[BenchmarkSample], list[dict[str, Any]]]:
    samples: list[BenchmarkSample] = []
    for dataset in DATASETS:
        path = resolve_benchmark_path(args.benchmark_root, "mcq", args.collection, dataset, args.split)
        loaded = load_benchmark_samples(path, "mcq", args.collection, dataset, args.split)
        if any(not isinstance(row.options, dict) or set(row.options) != set(CHOICES) for row in loaded):
            raise RuntimeError(f"Rationale+Answer comparison requires exact A/B/C/D options: {dataset}")
        samples.extend(loaded)
    candidate_rows = list(iter_jsonl(args.candidate_cache))
    by_key = {str(row.get("key") or ""): row for row in candidate_rows}
    if len(by_key) != len(candidate_rows):
        raise RuntimeError("Candidate cache has duplicate or empty keys")
    aligned: list[dict[str, Any]] = []
    for sample in samples:
        row = by_key.get(sample_key(sample))
        if row is None:
            raise RuntimeError(f"Candidate cache is missing {sample_key(sample)}")
        documents = list(row.get("reranked_documents") or [])
        if len(documents) != args.top_k:
            raise RuntimeError(f"Expected exactly Top-{args.top_k}: {sample.id} has {len(documents)}")
        if [int(doc.get("rerank_rank", -1)) for doc in documents] != list(range(1, args.top_k + 1)):
            raise RuntimeError(f"Non-canonical rerank order: {sample.id}")
        aligned.append(row)
    if len(samples) != len(candidate_rows):
        raise RuntimeError(f"Benchmark/candidate count mismatch: {len(samples)} != {len(candidate_rows)}")
    if args.max_questions > 0:
        samples = samples[: args.max_questions]
        aligned = aligned[: args.max_questions]
    return samples, aligned


def load_semantic(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in iter_jsonl(path):
        key = str(row.get("pair_key") or "")
        probability = float(row.get("prob_support"))
        if not key or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid semantic probability row: {row}")
        values[key] = probability
    return values


def load_no_rag(args: argparse.Namespace, samples: Sequence[BenchmarkSample]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        path = args.no_rag_root / dataset / args.split / "no_rag_generations.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            identifier = str(row.get("sample_id") or "")
            if not identifier or identifier in values:
                raise RuntimeError(f"Duplicate/empty no-RAG sample id in {path}: {identifier}")
            answer = str(row.get("answer") or "").strip().upper()
            if answer not in CHOICES or not bool(row.get("valid", True)):
                raise RuntimeError(f"Invalid cached no-RAG result: {identifier}")
            values[identifier] = row
    missing = [sample.id for sample in samples if sample.id not in values]
    if missing:
        raise RuntimeError(f"No-RAG cache missing {len(missing)} selected samples; first={missing[0]}")
    return values


def build_contract(args: argparse.Namespace, samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
    candidate_manifest = args.candidate_cache.parent / "manifest.json"
    if not candidate_manifest.is_file():
        raise FileNotFoundError(candidate_manifest)
    value = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    expected = {
        "rows": 6545,
        "per_source_top_k": args.top_k,
        "candidate_pool_top_k": 4 * args.top_k,
        "rerank_top_k": args.top_k,
        "evaluation_top_k": args.top_k,
    }
    mismatch = {key: (wanted, value.get(key)) for key, wanted in expected.items() if value.get(key) != wanted}
    if mismatch:
        raise RuntimeError(f"Candidate projection contract mismatch: {mismatch}")
    return {
        "run_version": RUN_VERSION,
        "generation_rule": GENERATION_RULE,
        "scope": "MCQ evaluation; method itself is token-vocabulary based, final decision constrained to A/B/C/D",
        "datasets": list(DATASETS),
        "questions": len(samples),
        "top_k": args.top_k,
        "gamma": args.gamma,
        "dynamic_beta": "mean full-vocabulary JSD(expert, no-context) at first rationale token; fixed thereafter",
        "candidate_cache": {"path": str(args.candidate_cache.resolve()), "sha256": sha256_file(args.candidate_cache)},
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "semantic_score_cache": {"path": str(args.semantic_score_cache.resolve()), "sha256": sha256_file(args.semantic_score_cache)},
        "no_rag_root": str(args.no_rag_root.resolve()),
        "llama": model_identity(args.llama_model),
        "max_input_tokens": args.max_input_tokens,
        "max_rationale_tokens": args.max_rationale_tokens,
        "answer_reserve_tokens": args.answer_reserve_tokens,
        "decoding": "greedy; exact anchored rationale stop markers; constrained final choice",
        "test_tuning": "none; gamma=2.5 fixed from PCED and no threshold selected on test",
        "seed": args.seed,
        "code_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__)),
    }


def ensure_contract(output_dir: Path, contract: dict[str, Any]) -> str:
    def stable(value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        result.pop("code_commit", None)
        result.pop("script_sha256", None)
        return result
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if stable(previous) != stable(contract):
            raise RuntimeError(f"Experiment contract mismatch; use a new output directory: {path}")
    else:
        atomic_json(path, contract)
    return canonical_hash(stable(contract))


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


class StageProgress:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.names = ["preflight", "Base-RAG rationale+answer", "PCED rerank prior", "PCED semantic prior", "aggregate"]
        self.started = time.time()
        self.bar: tqdm[Any] | None = None
        self.stage_started = self.started
        self.index = 0
        self.done = 0
        self.total = 1
        self.initial = 0

    def start(self, index: int, total: int, unit: str, initial: int = 0) -> None:
        if self.bar is not None:
            self.bar.close()
        self.index, self.total, self.done = index, max(1, total), initial
        self.initial = initial
        self.stage_started = time.time()
        print(
            f"[overall stage {index}/5 | Top-{self.top_k} | elapsed {format_duration(time.time()-self.started)} | "
            f"overall ETA unknown until stage throughput is measured] {self.names[index-1]}", flush=True,
        )
        self.bar = tqdm(total=total, initial=initial, unit=unit, dynamic_ncols=True,
                        desc=f"Stage {index}/5 - Top-{self.top_k} {self.names[index-1]}")

    def update(self, amount: int = 1) -> None:
        self.done += amount
        if self.bar is not None:
            self.bar.update(amount)
            elapsed = max(time.time() - self.stage_started, 1e-6)
            active_done = max(self.done - self.initial, 0)
            rate = active_done / elapsed
            eta = (self.total - self.done) / rate if rate > 0 else None
            self.bar.set_postfix_str(
                f"elapsed={format_duration(time.time()-self.started)} rate={rate:.2f}/s stage_ETA={format_duration(eta)}",
                refresh=False,
            )

    def complete(self, detail: str) -> None:
        if self.done < self.total:
            self.update(self.total - self.done)
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        print(f"[stage {self.index}/5 complete | duration {format_duration(time.time()-self.stage_started)}] {detail}", flush=True)


class RationaleAnswerPcedGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if args.attn_implementation == "sdpa":
            # The installed PyTorch/cuDNN stack advertises SDPA shapes for
            # which cuDNN cannot build a plan. Keep the supported flash,
            # memory-efficient, and math kernels available.
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(args.llama_model, local_files_only=True, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.llama_model, local_files_only=True, low_cpu_mem_usage=True,
            dtype=dtype, attn_implementation=args.attn_implementation,
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        choice_ids = []
        for choice in CHOICES:
            encoded = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"Choice {choice} is not a single token: {encoded}")
            choice_ids.append(encoded[0])
        self.choice_ids = torch.tensor(choice_ids, dtype=torch.long, device=self.device)
        stop_texts = (
            END_REASONING_MARKER,
            "\n" + END_REASONING_MARKER,
            "\nFinal answer:",
            "Final answer:",
            "\nTherefore, the answer",
            "Therefore, the answer",
        )
        self.stop_sequences = [
            self.tokenizer.encode(text, add_special_tokens=False) for text in stop_texts
        ]

    def _prompt(self, sample: BenchmarkSample, document_text: str | None) -> str:
        return rationale_generation_prompt(self.tokenizer, sample.raw, document_text)

    def fit_documents(self, sample: BenchmarkSample, texts: Sequence[str]) -> tuple[str, dict[str, Any]]:
        # Match the established anchored multi-document evaluator: normalize
        # whitespace inside each chunk and keep blank lines only as document
        # boundaries.
        normalized = [" ".join(str(text).split()) for text in texts]
        original = [self.tokenizer.encode(text, add_special_tokens=False) for text in normalized]
        caps = [len(tokens) for tokens in original]
        maximum_prompt = self.args.max_input_tokens - self.args.max_rationale_tokens - self.args.answer_reserve_tokens
        if maximum_prompt <= 0:
            raise ValueError("Generation reserves leave no prompt-token budget")

        def build() -> tuple[str, int]:
            kept = [self.tokenizer.decode(
                        tokens[:cap], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).strip()
                    for tokens, cap in zip(original, caps, strict=True)]
            text = "\n\n".join(kept)
            return text, len(self.tokenizer.encode(self._prompt(sample, text), add_special_tokens=False))

        text, length = build()
        while length > maximum_prompt:
            active = [index for index, cap in enumerate(caps) if cap > 16]
            if not active:
                raise RuntimeError(f"Cannot fit Rationale+Answer prompt: {sample.id}")
            decrement = max(1, math.ceil((length - maximum_prompt) / len(active)))
            for index in active:
                caps[index] = max(16, caps[index] - decrement)
            text, length = build()
        return text, {
            "prompt_tokens": length,
            "original_document_tokens": int(sum(map(len, original))),
            "used_document_tokens": int(sum(caps)),
            "truncated": caps != [len(tokens) for tokens in original],
        }

    @torch.inference_mode()
    def _prefill(self, prompts: Sequence[str]) -> tuple[torch.Tensor, Any, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts), add_special_tokens=False, padding=True, return_tensors="pt"
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        if ids.shape[1] >= self.args.max_input_tokens:
            raise RuntimeError(f"Prefill reaches context limit: {ids.shape[1]} >= {self.args.max_input_tokens}")
        output = self.model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
        return output.logits[:, -1].float(), output.past_key_values, mask

    @torch.inference_mode()
    def _advance(self, token: int, cache: Any, mask: torch.Tensor) -> tuple[torch.Tensor, Any, torch.Tensor]:
        current = torch.full((mask.shape[0], 1), token, dtype=torch.long, device=self.device)
        next_mask = torch.cat([mask, torch.ones_like(current)], dim=1)
        output = self.model(
            input_ids=current, attention_mask=next_mask, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        return output.logits[:, -1].float(), output.past_key_values, next_mask

    @staticmethod
    def jsd_beta(logits: torch.Tensor) -> float:
        if logits.shape[0] < 2:
            raise RuntimeError("PCED requires one amateur and at least one expert")
        amateur_log = F.log_softmax(logits[0], dim=-1)
        expert_log = F.log_softmax(logits[1:], dim=-1)
        amateur = amateur_log.exp().unsqueeze(0).expand_as(expert_log)
        experts = expert_log.exp()
        mixture = 0.5 * (amateur + experts)
        log_mixture = mixture.clamp_min(1e-12).log()
        jsd = 0.5 * (
            (experts * (expert_log - log_mixture)).sum(-1)
            + (amateur * (amateur_log.unsqueeze(0) - log_mixture)).sum(-1)
        )
        return float(jsd.mean().item())

    def _stopped(self, generated: Sequence[int]) -> bool:
        if generated and self.tokenizer.eos_token_id is not None and generated[-1] == self.tokenizer.eos_token_id:
            return True
        return any(
            stop and len(generated) >= len(stop) and list(generated[-len(stop) :]) == stop
            for stop in self.stop_sequences
        )

    @torch.inference_mode()
    def _choice_logits(self, sample: BenchmarkSample, document_texts: Sequence[str | None], rationale: str) -> torch.Tensor:
        prompts = [
            render_chat_prompt(self.tokenizer, build_anchored_user_prompt(sample.raw, text))
            + assistant_decision_prefix(rationale)
            for text in document_texts
        ]
        logits, _, _ = self._prefill(prompts)
        return logits.index_select(-1, self.choice_ids)

    @torch.inference_mode()
    def generate_base(self, sample: BenchmarkSample, documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
        document_text, packing = self.fit_documents(sample, [str(doc["text"]) for doc in documents])
        logits, cache, mask = self._prefill([self._prompt(sample, document_text)])
        generated: list[int] = []
        finish_reason = "length"
        for _ in range(self.args.max_rationale_tokens):
            token = int(logits[0].argmax().item())
            generated.append(token)
            if self._stopped(generated):
                finish_reason = "stop"
                break
            logits, cache, mask = self._advance(token, cache, mask)
        raw = self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rationale, flags = normalize_rationale(raw)
        choice_logits = self._choice_logits(sample, [document_text], rationale)[0]
        answer = CHOICES[int(choice_logits.argmax().item())]
        return {
            "sample_id": sample.id,
            "dataset": sample.dataset,
            "gold_answer": sample.answer,
            "answer": answer,
            "correct": answer == sample.answer,
            "rationale": rationale,
            "canonical_response": canonical_response(rationale, answer, sample.options or {}),
            "rationale_tokens": len(generated),
            "finish_reason": finish_reason,
            "quality_flags": sorted(set(flags + (["rationale_length_exhausted"] if finish_reason == "length" else []))),
            "choice_logits": [float(value) for value in choice_logits.cpu().tolist()],
            "prompt_packing": packing,
        }

    @torch.inference_mode()
    def generate_pced(
        self, sample: BenchmarkSample, documents: Sequence[dict[str, Any]], prior: np.ndarray,
    ) -> dict[str, Any]:
        fitted: list[str] = []
        expert_packing: list[dict[str, Any]] = []
        for document in documents:
            text, metadata = self.fit_documents(sample, [str(document["text"])])
            fitted.append(text)
            expert_packing.append(metadata)
        prompts = [self._prompt(sample, None)] + [self._prompt(sample, text) for text in fitted]
        logits, cache, mask = self._prefill(prompts)
        beta = self.jsd_beta(logits)
        prior_tensor = torch.tensor(prior, dtype=torch.float32, device=self.device).clamp_min(self.args.prior_epsilon)
        generated: list[int] = []
        winners: list[int] = []
        finish_reason = "length"
        for _ in range(self.args.max_rationale_tokens):
            scores = (1.0 + beta) * logits[1:] - beta * logits[0].unsqueeze(0)
            scores = scores + self.args.gamma * prior_tensor.log().unsqueeze(1)
            flat = int(scores.reshape(-1).argmax().item())
            token = flat % scores.shape[1]
            winner = flat // scores.shape[1]
            generated.append(token)
            winners.append(winner)
            if self._stopped(generated):
                finish_reason = "stop"
                break
            logits, cache, mask = self._advance(token, cache, mask)
        raw = self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rationale, flags = normalize_rationale(raw)
        choice_logits = self._choice_logits(sample, [None] + fitted, rationale)
        choice_scores = (1.0 + beta) * choice_logits[1:] - beta * choice_logits[0].unsqueeze(0)
        choice_scores = choice_scores + self.args.gamma * prior_tensor.log().unsqueeze(1)
        flat = int(choice_scores.reshape(-1).argmax().item())
        answer_index = flat % len(CHOICES)
        final_expert = flat // len(CHOICES)
        answer = CHOICES[answer_index]
        winner_counts = Counter(winners)
        switches = sum(left != right for left, right in zip(winners, winners[1:]))
        return {
            "sample_id": sample.id,
            "dataset": sample.dataset,
            "gold_answer": sample.answer,
            "answer": answer,
            "correct": answer == sample.answer,
            "rationale": rationale,
            "canonical_response": canonical_response(rationale, answer, sample.options or {}),
            "rationale_tokens": len(generated),
            "finish_reason": finish_reason,
            "quality_flags": sorted(set(flags + (["rationale_length_exhausted"] if finish_reason == "length" else []))),
            "beta": beta,
            "prior": [float(value) for value in prior.tolist()],
            "rationale_expert_token_counts": {str(index + 1): int(count) for index, count in winner_counts.items()},
            "rationale_expert_switches": switches,
            "final_expert_rank": final_expert + 1,
            "choice_score_by_option": [float(value) for value in choice_scores.max(dim=0).values.cpu().tolist()],
            "expert_prompt_packing": expert_packing,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def valid_shard(path: Path, expected_ids: Sequence[str], contract_hash: str) -> bool:
    marker = path.with_suffix(".complete.json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        rows = list(iter_jsonl(path))
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("contract_hash") == contract_hash and [row.get("sample_id") for row in rows] == list(expected_ids)


def cached_condition(
    args: argparse.Namespace,
    condition: str,
    samples: Sequence[BenchmarkSample],
    candidates: Sequence[dict[str, Any]],
    semantic: dict[str, float],
    contract_hash: str,
    generator: RationaleAnswerPcedGenerator,
    progress: StageProgress,
    stage_index: int,
) -> list[dict[str, Any]]:
    root = args.output_dir / "generation_shards" / condition
    root.mkdir(parents=True, exist_ok=True)
    shards = [(start, min(len(samples), start + args.shard_size)) for start in range(0, len(samples), args.shard_size)]
    completed = 0
    for shard_index, (start, stop) in enumerate(shards):
        path = root / f"shard_{shard_index:05d}.jsonl"
        if valid_shard(path, [sample.id for sample in samples[start:stop]], contract_hash):
            completed += stop - start
    progress.start(stage_index, len(samples), "question", completed)
    for shard_index, (start, stop) in enumerate(shards):
        path = root / f"shard_{shard_index:05d}.jsonl"
        expected = [sample.id for sample in samples[start:stop]]
        if valid_shard(path, expected, contract_hash):
            continue
        output: list[dict[str, Any]] = []
        shard_started = time.time()
        for sample, row in zip(samples[start:stop], candidates[start:stop], strict=True):
            documents = list(row["reranked_documents"])
            if condition == "base_rag":
                result = generator.generate_base(sample, documents)
            else:
                if condition == "pced_rerank":
                    prior = minmax([float(doc["rerank_score"]) for doc in documents], args.prior_epsilon)
                elif condition == "pced_semantic":
                    prior = np.asarray(
                        [semantic[pair_key(sample, doc)] for doc in documents], dtype=np.float64
                    )
                    prior = np.clip(prior, args.prior_epsilon, 1.0 - args.prior_epsilon)
                else:
                    raise ValueError(condition)
                result = generator.generate_pced(sample, documents, prior)
            output.append(result)
            progress.update(1)
        atomic_jsonl(path, output)
        atomic_json(path.with_suffix(".complete.json"), {
            "contract_hash": contract_hash,
            "condition": condition,
            "shard": shard_index,
            "questions": len(output),
            "elapsed_seconds": time.time() - shard_started,
        })
    progress.complete(f"condition={condition} questions={len(samples)} shards={len(shards)} cache={root}")
    result: list[dict[str, Any]] = []
    for shard_index, (start, stop) in enumerate(shards):
        path = root / f"shard_{shard_index:05d}.jsonl"
        if not valid_shard(path, [sample.id for sample in samples[start:stop]], contract_hash):
            raise RuntimeError(f"Incomplete generation shard: {path}")
        result.extend(iter_jsonl(path))
    return result


def render_table(summary: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{100*value:.2f}"
    labels = {
        "no_rag": "No-RAG",
        "base_rag": "Base-RAG (concatenated)",
        "pced_rerank": "PCED (rerank-score prior)",
        "pced_semantic": "PCED (semantic-support prior)",
    }
    lines = [
        f"Rationale+Answer PCED evaluation (Top-{summary['top_k']})", "",
        "| Condition | N | MedMCQA | MedQA | MMLU pooled | Micro | Macro-8 | Macro-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["conditions"].items():
        lines.append(
            f"| {labels[condition]} | {metrics['questions']} | {pct(metrics['medmcqa_accuracy'])} | "
            f"{pct(metrics['medqa_accuracy'])} | {pct(metrics['mmlu_pooled_accuracy'])} | "
            f"{pct(metrics['micro_accuracy'])} | {pct(metrics['macro8_accuracy'])} | {pct(metrics['macro3_accuracy'])} |"
        )
    lines.extend(["", "Paired changes on the identical cohort:", ""])
    for label, value in summary["paired_comparisons"].items():
        interval = value["paired_bootstrap_95ci"]
        lines.append(
            f"- {label}: {100*value['accuracy_delta']:+.2f}%p "
            f"(95% CI {100*interval[0]:+.2f} to {100*interval[1]:+.2f}; "
            f"W→C={value['wrong_to_correct']}, C→W={value['correct_to_wrong']})"
        )
    return "\n".join(lines) + "\n"


def aggregate(
    args: argparse.Namespace,
    samples: Sequence[BenchmarkSample],
    no_rag: dict[str, dict[str, Any]],
    conditions: dict[str, Sequence[dict[str, Any]]],
    progress: StageProgress,
) -> None:
    progress.start(5, len(samples) + 3 * args.bootstrap_replicates, "unit")
    indexed = {name: {str(row["sample_id"]): row for row in rows} for name, rows in conditions.items()}
    results: list[dict[str, Any]] = []
    for sample in samples:
        predictions = {"no_rag": str(no_rag[sample.id]["answer"]).upper()}
        diagnostics: dict[str, Any] = {}
        for name, rows in indexed.items():
            row = rows[sample.id]
            predictions[name] = str(row["answer"])
            diagnostics[name] = {
                "rationale_tokens": row.get("rationale_tokens"),
                "finish_reason": row.get("finish_reason"),
                "quality_flags": row.get("quality_flags"),
                "beta": row.get("beta"),
                "final_expert_rank": row.get("final_expert_rank"),
            }
        results.append({
            "sample_id": sample.id, "dataset": sample.dataset, "row_idx": sample.row_idx,
            "gold_answer": sample.answer, "predictions": predictions, "diagnostics": diagnostics,
        })
        progress.update(1)
    order = ("no_rag", "base_rag", "pced_rerank", "pced_semantic")
    metrics = {name: condition_summary(results, name) for name in order}
    for name, value in metrics.items():
        value["mean_documents_available"] = 0.0 if name == "no_rag" else float(args.top_k)
    specs = (
        ("PCED rerank vs Base-RAG", "pced_rerank", "base_rag"),
        ("PCED semantic vs PCED rerank", "pced_semantic", "pced_rerank"),
        ("PCED semantic vs Base-RAG", "pced_semantic", "base_rag"),
    )
    comparisons: dict[str, Any] = {}
    for index, (label, condition, baseline) in enumerate(specs):
        comparisons[label] = paired_comparison(
            results, condition, baseline, args.bootstrap_replicates, args.seed + index
        )
        progress.update(args.bootstrap_replicates)
    generation_diagnostics: dict[str, Any] = {}
    for name, rows in conditions.items():
        exhausted = sum("rationale_length_exhausted" in (row.get("quality_flags") or []) for row in rows)
        truncated = sum(
            bool((row.get("prompt_packing") or {}).get("truncated"))
            or any(bool(item.get("truncated")) for item in (row.get("expert_prompt_packing") or []))
            for row in rows
        )
        generation_diagnostics[name] = {
            "questions": len(rows),
            "mean_rationale_tokens": float(np.mean([row.get("rationale_tokens", 0) for row in rows])),
            "rationale_length_exhausted": exhausted,
            "document_prompt_truncated": truncated,
            "mean_distinct_rationale_experts_used": (
                float(np.mean([
                    len(row.get("rationale_expert_token_counts") or {}) for row in rows
                ])) if name.startswith("pced_") else None
            ),
        }
    summary = {
        "run_version": RUN_VERSION,
        "top_k": args.top_k,
        "questions": len(samples),
        "conditions": metrics,
        "paired_comparisons": comparisons,
        "generation_diagnostics": generation_diagnostics,
    }
    atomic_jsonl(args.output_dir / "predictions.jsonl", results)
    atomic_json(args.output_dir / "summary.json", summary)
    table = render_table(summary)
    (args.output_dir / "summary_table_pretty.txt").write_text(table, encoding="utf-8")
    progress.complete(f"summary={args.output_dir/'summary.json'} predictions={args.output_dir/'predictions.jsonl'}")
    print(table, flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0 or args.top_k > 32:
        raise ValueError("top-k must be in [1, 32]")
    if args.max_input_tokens <= args.max_rationale_tokens + args.answer_reserve_tokens:
        raise ValueError("Invalid context/generation token budget")
    if args.shard_size <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("Shard size and bootstrap replicates must be positive")
    for path in (args.candidate_cache, args.semantic_score_cache, args.llama_model / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.attn_implementation == "sdpa":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    progress = StageProgress(args.top_k)
    try:
        progress.start(1, 5, "check")
        samples, candidates = load_inputs(args)
        progress.update()
        semantic = load_semantic(args.semantic_score_cache)
        expected_semantic = {
            pair_key(sample, document)
            for sample, row in zip(samples, candidates, strict=True)
            for document in row["reranked_documents"]
        }
        if not expected_semantic.issubset(semantic):
            missing = expected_semantic - set(semantic)
            raise RuntimeError(f"Semantic cache missing {len(missing)} pairs; first={next(iter(missing))}")
        progress.update()
        no_rag = load_no_rag(args, samples)
        progress.update()
        contract = build_contract(args, samples)
        contract_hash = ensure_contract(args.output_dir, contract)
        progress.update()
        tokenizer = AutoTokenizer.from_pretrained(args.llama_model, local_files_only=True, use_fast=True)
        sampled = min(64, len(samples))
        maximum_single = 0
        for sample, row in zip(samples[:sampled], candidates[:sampled], strict=True):
            for document in row["reranked_documents"]:
                maximum_single = max(maximum_single, len(tokenizer.encode(
                    rationale_generation_prompt(tokenizer, sample.raw, str(document["text"])), add_special_tokens=False
                )))
        del tokenizer
        if maximum_single + args.max_rationale_tokens + args.answer_reserve_tokens > args.max_input_tokens:
            print(
                f"[preflight notice] sampled single expert may require token truncation: max_prompt={maximum_single}",
                flush=True,
            )
        progress.update()
        progress.complete(
            f"questions={len(samples)} documents={len(samples)*args.top_k} semantic_pairs={len(expected_semantic)} "
            f"sampled_single_prompt_max={maximum_single} manifest={args.output_dir/'experiment_manifest.json'}"
        )
        if args.preflight_only:
            print("[preflight-only complete] no Llama generation was run", flush=True)
            return
        generator = RationaleAnswerPcedGenerator(args)
        try:
            outputs = {
                "base_rag": cached_condition(
                    args, "base_rag", samples, candidates, semantic, contract_hash, generator, progress, 2
                ),
                "pced_rerank": cached_condition(
                    args, "pced_rerank", samples, candidates, semantic, contract_hash, generator, progress, 3
                ),
                "pced_semantic": cached_condition(
                    args, "pced_semantic", samples, candidates, semantic, contract_hash, generator, progress, 4
                ),
            }
        finally:
            generator.close()
        aggregate(args, samples, no_rag, outputs, progress)
    except Exception as exc:
        print(
            f"[workflow FAILED | Top-{args.top_k} | active stage={progress.index}/5 | "
            f"completed={progress.done}/{progress.total}] {type(exc).__name__}: {exc}; "
            "rerun the identical command to resume from complete shards",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()

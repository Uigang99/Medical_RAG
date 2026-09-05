#!/usr/bin/env python3
"""Evaluate Direct-Choice PCED and a Semantic-prior PCED variant.

The target Llama, benchmark cohort, dynamic Top-k documents, prompt, constrained
A/B/C/D answer space, contrastive term, and gamma are shared by every PCED
condition.  The proposed condition changes only the document prior from the
reranker score to the trained Semantic-Support probability.

This is the one-decoding-step MCQ specialization of PCED.  It tests independent
document experts and retrieval-aware contrastive logit fusion, but it cannot
test expert switching across multiple generated tokens.
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

from cache_rag2_direct_choice_train_outcomes import (  # noqa: E402
    PROMPT_POLICY_VERSION,
    sequence_for_prompt,
)
from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path  # noqa: E402
from medrag.core import BenchmarkSample  # noqa: E402
from medrag.filtering.rag2_filter import Rag2FlanT5Filter  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_direct_choice_pced_semantic_prior_dynamic_topk_v3"
PCED_RULE_VERSION = "pced_eq2_eq3_dynamic_mean_adacad_jsd_first_token_gamma2p5_v1"
RERANK_PRIOR_VERSION = "question_minmax_reranker_only_v2"
SEMANTIC_PRIOR_VERSION = "binary_support_classifier_raw_probability_v1"
MATCHED_PRIOR_VERSION = "rerank_prior_values_ranked_by_semantic_probability_v2"

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
MMLU_DATASETS = tuple(name for name in DATASETS if name.startswith("mmlu_"))

DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "databases/run_cache/rag2_llama3_paper_exact_terminal_v1/"
    "all_mcq_source_balanced32_rationale_full_rerank32/candidates/"
    "07083d5bac341d9b/candidates.jsonl"
)
DEFAULT_LLAMA = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_MEDMCQA_SEMANTIC = (
    WORKSPACE_ROOT
    / "models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medmcqa/"
    "medmcqa_semantic_top8_binary_support_epoch5_len1280_fullpair/"
    "20260829_212146/final_model"
)
DEFAULT_MEDQA_SEMANTIC = (
    WORKSPACE_ROOT
    / "models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa/"
    "medqa_semantic_top8_binary_support_epoch8_len1280_fullpair/"
    "20260830_170945/final_model"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/rag2_pced_direct_choice_v2/all_mcq_source_balanced32_rerank8_rerank_prior"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--benchmark-root", type=Path, default=PROJECT_ROOT / "datasets/benchmark")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--medmcqa-semantic-model", type=Path, default=DEFAULT_MEDMCQA_SEMANTIC)
    parser.add_argument("--medqa-semantic-model", type=Path, default=DEFAULT_MEDQA_SEMANTIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--prior-epsilon", type=float, default=1e-4)
    parser.add_argument("--semantic-batch-size", type=int, default=128)
    parser.add_argument("--semantic-max-input-tokens", type=int, default=1280)
    parser.add_argument("--question-batch-size", type=int, default=16)
    parser.add_argument("--prompt-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--max-questions", type=int, default=0, help="0 evaluates all 6,545 test questions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


class WorkflowProgress:
    """Stage-weighted overall status plus an active-stage rolling ETA."""

    def __init__(self, names: Sequence[str], estimated_seconds: Sequence[float]) -> None:
        self.names = list(names)
        self.estimates = [float(value) for value in estimated_seconds]
        self.started = time.time()
        self.stage_started = self.started
        self.stage_index = 0
        self.stage_total = 1
        self.stage_done = 0
        self.stage_initial = 0
        self.bar: tqdm[Any] | None = None

    def start(self, index: int, total: int, unit: str, *, initial: int = 0) -> None:
        if self.bar is not None:
            self.bar.close()
        self.stage_index = index
        self.stage_total = max(1, int(total))
        self.stage_done = min(max(0, int(initial)), self.stage_total)
        self.stage_initial = self.stage_done
        self.stage_started = time.time()
        print(
            f"[overall stage {index}/{len(self.names)} | elapsed {format_duration(time.time()-self.started)}] "
            f"{self.names[index-1]}",
            flush=True,
        )
        self.bar = tqdm(
            total=total,
            initial=initial,
            desc=f"Stage {index}/{len(self.names)} - {self.names[index-1]}",
            unit=unit,
            dynamic_ncols=True,
        )
        self._status()

    def _status(self) -> None:
        now = time.time()
        completed_now = self.stage_done - self.stage_initial
        elapsed = max(now - self.stage_started, 1e-6)
        rate = max(0.0, completed_now / elapsed)
        fraction = self.stage_done / self.stage_total
        stage_eta = (self.stage_total - self.stage_done) / rate if rate > 0 else None
        future = sum(self.estimates[self.stage_index :])
        total_weight = sum(self.estimates)
        done_weight = sum(self.estimates[: self.stage_index - 1]) + self.estimates[self.stage_index - 1] * fraction
        overall_pct = 100.0 * done_weight / total_weight
        if self.bar is not None:
            eta = None if stage_eta is None else stage_eta + future
            self.bar.set_postfix_str(
                f"overall={overall_pct:.1f}% elapsed={format_duration(now-self.started)} "
                f"overall_ETA={format_duration(eta)}",
                refresh=False,
            )

    def update(self, amount: int) -> None:
        self.stage_done = min(self.stage_total, self.stage_done + int(amount))
        if self.bar is not None:
            self.bar.update(int(amount))
        self._status()

    def complete(self, detail: str) -> None:
        remaining = self.stage_total - self.stage_done
        if remaining > 0:
            self.update(remaining)
        duration = time.time() - self.stage_started
        self.estimates[self.stage_index - 1] = max(0.1, duration)
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        print(
            f"[stage {self.stage_index}/{len(self.names)} complete | duration {format_duration(duration)}] {detail}",
            flush=True,
        )

    def fail(self, detail: str) -> None:
        if self.bar is not None:
            self.bar.close()
        print(
            f"[workflow FAILED | stage {self.stage_index}/{len(self.names)} | "
            f"completed {self.stage_done}/{self.stage_total}] {detail}",
            flush=True,
        )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not config.is_file() or not tokenizer.is_file() or not weights:
        raise FileNotFoundError(f"Incomplete local model: {path}")
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256_file(config),
        "tokenizer_config_sha256": sha256_file(tokenizer),
        "weights": [{"name": item.name, "size": item.stat().st_size} for item in weights],
    }


def sample_key(sample: BenchmarkSample) -> str:
    return f"{sample.dataset}::{sample.split}::{sample.id}::{sample.row_idx}"


def load_inputs(args: argparse.Namespace) -> tuple[list[BenchmarkSample], list[dict[str, Any]]]:
    samples: list[BenchmarkSample] = []
    for dataset in DATASETS:
        path = resolve_benchmark_path(args.benchmark_root, "mcq", args.collection, dataset, args.split)
        loaded = load_benchmark_samples(path, "mcq", args.collection, dataset, args.split)
        if any(not isinstance(row.options, dict) or set(row.options) != set(CHOICES) for row in loaded):
            raise RuntimeError(f"Direct-Choice requires exact A/B/C/D options: {dataset}")
        samples.extend(loaded)
    candidate_rows = list(iter_jsonl(args.candidate_cache))
    candidate_by_key = {str(row.get("key") or ""): row for row in candidate_rows}
    if len(candidate_by_key) != len(candidate_rows):
        raise RuntimeError("Candidate cache contains missing or duplicate keys")
    aligned: list[dict[str, Any]] = []
    for sample in samples:
        key = sample_key(sample)
        row = candidate_by_key.get(key)
        if row is None:
            raise RuntimeError(f"Candidate cache is missing sample: {key}")
        documents = list(row.get("reranked_documents") or [])[: args.top_k]
        if len(documents) != args.top_k:
            raise RuntimeError(f"Expected Top-{args.top_k} documents: {key}")
        ranks = [int(document.get("rerank_rank", -1)) for document in documents]
        if ranks != list(range(1, args.top_k + 1)):
            raise RuntimeError(f"Non-canonical rerank prefix: {key} ranks={ranks}")
        if any(not str(document.get("text") or "").strip() for document in documents):
            raise RuntimeError(f"Empty candidate document: {key}")
        aligned.append(row)
    if len(samples) != len(candidate_rows):
        raise RuntimeError(
            f"Benchmark/candidate count mismatch: benchmark={len(samples)} candidates={len(candidate_rows)}"
        )
    if args.max_questions > 0:
        samples = samples[: args.max_questions]
        aligned = aligned[: args.max_questions]
    return samples, aligned


def pair_key(sample: BenchmarkSample, document: dict[str, Any]) -> str:
    stable_id = str(document.get("stable_id") or document.get("corpus_id") or document.get("db_id") or "")
    rank = int(document.get("rerank_rank", -1))
    if not stable_id or rank <= 0:
        raise RuntimeError(f"Cannot identify document pair: {sample.id}")
    return f"{sample.id}::{rank}::{stable_id}"


def contract(args: argparse.Namespace, samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
    manifest = args.candidate_cache.parent / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    candidate_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        "rows": 6545,
        "per_source_top_k": args.top_k,
        "candidate_pool_top_k": 4 * args.top_k,
        "rerank_top_k": args.top_k,
        "candidate_layout": "source_balanced",
    }
    mismatches = {
        key: {"expected": value, "actual": candidate_manifest.get(key)}
        for key, value in expected.items()
        if candidate_manifest.get(key, "source_balanced" if key == "candidate_layout" else None) != value
    }
    if mismatches:
        raise RuntimeError(f"Candidate manifest mismatch: {mismatches}")
    return {
        "run_version": RUN_VERSION,
        "pced_rule_version": PCED_RULE_VERSION,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "rerank_prior_version": RERANK_PRIOR_VERSION,
        "semantic_prior_version": SEMANTIC_PRIOR_VERSION,
        "matched_prior_version": MATCHED_PRIOR_VERSION,
        "scope": "one-step constrained Direct-Choice MCQ specialization; no multi-token expert-switch claim",
        "datasets": list(DATASETS),
        "split": args.split,
        "question_count": len(samples),
        "top_k": args.top_k,
        "candidate_projection": "dense Top-k per each of four corpora, rerank 4k, select final Top-k",
        "gamma": args.gamma,
        "dynamic_beta": (
            "mean over Top-k per-expert full-vocabulary JSD(expert, no-context) at the first/only token; "
            "shared across experts"
        ),
        "prior_epsilon": args.prior_epsilon,
        "candidate_cache": {
            "path": str(args.candidate_cache.resolve()),
            "size": args.candidate_cache.stat().st_size,
            "sha256": sha256_file(args.candidate_cache),
            "manifest_sha256": sha256_file(manifest),
        },
        "models": {
            "llama": model_identity(args.llama_model),
            "medmcqa_semantic": model_identity(args.medmcqa_semantic_model),
            "medqa_semantic": model_identity(args.medqa_semantic_model),
            "routing": "medqa uses medqa model; medmcqa and all MMLU subsets use medmcqa model",
        },
        "answer_space": list(CHOICES),
        "test_tuning": "none; paper gamma=2.5 and dynamic AdaCAD-style beta fixed before test evaluation",
        "seed": args.seed,
        "code_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__)),
    }


def ensure_contract(output_dir: Path, value: dict[str, Any]) -> str:
    # The commit is provenance, not a cache-meaning input.  Keeping it out of
    # the resumability hash lets an interrupted run survive unrelated commits;
    # every setting that can change scores remains part of the contract.
    def resumable_contract(contract: dict[str, Any]) -> dict[str, Any]:
        result = dict(contract)
        result.pop("code_commit", None)
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if resumable_contract(previous) != resumable_contract(value):
            raise RuntimeError(
                f"Experiment contract mismatch at {path}; use a new versioned --output-dir"
            )
    else:
        atomic_json(path, value)
    return canonical_hash(resumable_contract(value))


def semantic_scores_complete(path: Path, expected_keys: set[str], contract_hash: str) -> bool:
    marker = path.with_suffix(".complete.json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        rows = list(iter_jsonl(path))
        keys = {str(row.get("pair_key") or "") for row in rows}
    except Exception:
        return False
    return (
        metadata.get("contract_hash") == contract_hash
        and int(metadata.get("pairs", -1)) == len(expected_keys)
        and keys == expected_keys
        and len(rows) == len(expected_keys)
    )


def load_semantic_scores(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.is_file():
        return values
    for row in iter_jsonl(path):
        key = str(row.get("pair_key") or "")
        probability = float(row.get("prob_support"))
        if key and math.isfinite(probability) and 0.0 <= probability <= 1.0:
            values[key] = probability
    return values


def cache_semantic_scores(
    args: argparse.Namespace,
    samples: Sequence[BenchmarkSample],
    candidate_rows: Sequence[dict[str, Any]],
    contract_hash: str,
    progress: WorkflowProgress,
) -> dict[str, float]:
    output_path = args.output_dir / "semantic_support_probabilities.jsonl"
    expected_keys = {
        pair_key(sample, document)
        for sample, row in zip(samples, candidate_rows, strict=True)
        for document in list(row["reranked_documents"])[: args.top_k]
    }
    if semantic_scores_complete(output_path, expected_keys, contract_hash):
        progress.start(2, len(expected_keys), "pair", initial=len(expected_keys))
        progress.complete(f"reused complete cache: {output_path}")
        return load_semantic_scores(output_path)

    cached = load_semantic_scores(output_path)
    cached = {key: value for key, value in cached.items() if key in expected_keys}
    progress.start(2, len(expected_keys), "pair", initial=len(cached))
    routes = (
        ("medmcqa_and_mmlu", args.medmcqa_semantic_model, lambda name: name != "medqa"),
        ("medqa", args.medqa_semantic_model, lambda name: name == "medqa"),
    )
    append_mode = output_path.is_file() and bool(cached)
    for route_name, model_path, accepts in routes:
        missing: list[tuple[str, BenchmarkSample, str]] = []
        for sample, row in zip(samples, candidate_rows, strict=True):
            if not accepts(sample.dataset):
                continue
            for document in list(row["reranked_documents"])[: args.top_k]:
                key = pair_key(sample, document)
                if key not in cached:
                    missing.append((key, sample, str(document["text"]).strip()))
        if not missing:
            continue
        print(f"[semantic route {route_name}] remaining={len(missing)} model={model_path}", flush=True)
        scorer = Rag2FlanT5Filter(
            model_path=model_path,
            batch_size=args.semantic_batch_size,
            max_input_length=args.semantic_max_input_tokens,
            max_new_tokens=1,
            max_doc_chars=0,
            device=args.device,
            bf16=args.dtype == "bfloat16",
            scoring_method="special_token",
            input_format="official",
        )
        for start in range(0, len(missing), 4096):
            chunk = missing[start : start + 4096]
            scores = scorer.score_evidences(
                [item[1] for item in chunk],
                [item[2] for item in chunk],
                progress_callback=progress.update,
            )
            with output_path.open("a" if append_mode else "w", encoding="utf-8") as handle:
                for (key, sample, _), score in zip(chunk, scores, strict=True):
                    probability = float(score["prob_helpful"])
                    cached[key] = probability
                    handle.write(json.dumps({
                        "pair_key": key,
                        "sample_id": sample.id,
                        "dataset": sample.dataset,
                        "prob_support": probability,
                        "prediction": str(score["prediction"]),
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            append_mode = True
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if set(cached) != expected_keys:
        raise RuntimeError(f"Semantic cache incomplete: cached={len(cached)} expected={len(expected_keys)}")
    compact = [
        {"pair_key": key, "prob_support": cached[key]}
        for key in sorted(expected_keys)
    ]
    atomic_jsonl(output_path, compact)
    atomic_json(output_path.with_suffix(".complete.json"), {
        "contract_hash": contract_hash,
        "pairs": len(expected_keys),
        "completed_at": time.time(),
    })
    progress.complete(f"pairs={len(expected_keys)} cache={output_path}")
    return cached


def pad_left(
    sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), maximum), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        ids[index, -len(sequence) :] = values
        mask[index, -len(sequence) :] = 1
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids


class DirectChoicePcedScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llama_model, local_files_only=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            local_files_only=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        self.decoder = self.model.model
        ids = []
        for choice in CHOICES:
            encoded = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"Choice {choice} is not one token after the fixed prefix: {encoded}")
            ids.append(int(encoded[0]))
        self.choice_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        weight = self.model.get_output_embeddings().weight
        self.choice_weight = weight.index_select(0, self.choice_ids).detach()

    def _fit_documents(
        self, sample: BenchmarkSample, texts: Sequence[str], *, joined: bool,
    ) -> tuple[list[int], dict[str, Any]]:
        """Fit documents to the fixed context without changing their order.

        Equal token caps keep one long document from consuming the entire
        concatenated Base-RAG prompt. PCED experts call this with one document.
        """
        cleaned = [str(text).strip() for text in texts]
        original_tokens = [self.tokenizer.encode(text, add_special_tokens=False) for text in cleaned]
        separator = "\n\n" if joined else ""

        def build(caps: Sequence[int]) -> list[int]:
            kept = [
                self.tokenizer.decode(
                    tokens[:cap], skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
                for tokens, cap in zip(original_tokens, caps, strict=True)
            ]
            return sequence_for_prompt(self.tokenizer, sample, separator.join(kept))[0]

        caps = [len(tokens) for tokens in original_tokens]
        sequence = build(caps)
        while len(sequence) > self.args.max_input_tokens:
            active = [index for index, cap in enumerate(caps) if cap > 16]
            if not active:
                raise RuntimeError(
                    f"Cannot fit prompt within {self.args.max_input_tokens} tokens: {sample.id}"
                )
            excess = len(sequence) - self.args.max_input_tokens
            decrement = max(1, math.ceil(excess / len(active)))
            for index in active:
                caps[index] = max(16, caps[index] - decrement)
            sequence = build(caps)
        return sequence, {
            "input_tokens": len(sequence),
            "original_document_tokens": int(sum(map(len, original_tokens))),
            "used_document_tokens": int(sum(caps)),
            "truncated": caps != [len(tokens) for tokens in original_tokens],
        }

    def sequences(
        self, samples: Sequence[BenchmarkSample], rows: Sequence[dict[str, Any]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[dict[str, Any]]]:
        no_rag: list[list[int]] = []
        base: list[list[int]] = []
        experts: list[list[int]] = []
        packing: list[dict[str, Any]] = []
        for sample, row in zip(samples, rows, strict=True):
            documents = list(row["reranked_documents"])[: self.args.top_k]
            no_rag.append(sequence_for_prompt(self.tokenizer, sample, None)[0])
            base_sequence, base_packing = self._fit_documents(
                sample, [str(document["text"]) for document in documents], joined=True
            )
            base.append(base_sequence)
            expert_meta: list[dict[str, Any]] = []
            for document in documents:
                sequence, metadata = self._fit_documents(
                    sample, [str(document["text"])], joined=False
                )
                experts.append(sequence)
                expert_meta.append(metadata)
            packing.append({"base": base_packing, "experts": expert_meta})
        maximum = max(map(len, no_rag + base + experts))
        if maximum > self.args.max_input_tokens:
            raise RuntimeError(
                f"Direct-Choice prompt exceeds max tokens: observed={maximum} limit={self.args.max_input_tokens}"
            )
        return no_rag, base, experts, packing

    @torch.inference_mode()
    def hidden(self, sequences: Sequence[Sequence[int]]) -> torch.Tensor:
        result: list[torch.Tensor | None] = [None] * len(sequences)
        order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
        cursor = 0
        batch_size = min(self.args.prompt_batch_size, len(order))
        while cursor < len(order):
            indices = order[cursor : cursor + batch_size]
            batch = [sequences[index] for index in indices]
            try:
                ids, mask, positions = pad_left(
                    batch, int(self.tokenizer.pad_token_id), self.device
                )
                output = self.decoder(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -1]
                for index, hidden in zip(indices, output, strict=True):
                    result[index] = hidden.detach().cpu()
                cursor += len(indices)
                del ids, mask, positions, output
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[OOM recovery] Direct prompt batch reduced to {batch_size}; outputs unchanged", flush=True)
        if any(value is None for value in result):
            raise RuntimeError("Missing hidden state after Direct-Choice scoring")
        return torch.stack([value for value in result if value is not None]).to(self.device)

    @torch.inference_mode()
    def score_questions(
        self, samples: Sequence[BenchmarkSample], rows: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        no_sequences, base_sequences, expert_sequences, packing = self.sequences(samples, rows)
        no_hidden = self.hidden(no_sequences)
        base_hidden = self.hidden(base_sequences)
        expert_hidden = self.hidden(expert_sequences)
        no_full = self.model.lm_head(no_hidden).float()
        no_log_probs = F.log_softmax(no_full, dim=-1)
        no_probabilities = no_log_probs.exp()
        no_choices = no_full.index_select(-1, self.choice_ids)
        base_choices = F.linear(base_hidden, self.choice_weight).float()
        expert_choices: list[torch.Tensor] = []
        expert_jsd: list[torch.Tensor] = []
        for start in range(0, len(expert_hidden), self.args.prompt_batch_size):
            stop = min(len(expert_hidden), start + self.args.prompt_batch_size)
            hidden = expert_hidden[start:stop]
            logits = self.model.lm_head(hidden).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            owners = torch.arange(start, stop, device=self.device) // self.args.top_k
            q = no_probabilities.index_select(0, owners)
            log_q = no_log_probs.index_select(0, owners)
            mixture = 0.5 * (probabilities + q)
            log_mixture = mixture.clamp_min(1e-12).log()
            jsd = 0.5 * (
                (probabilities * (log_probs - log_mixture)).sum(-1)
                + (q * (log_q - log_mixture)).sum(-1)
            )
            expert_choices.append(logits.index_select(-1, self.choice_ids).cpu())
            expert_jsd.append(jsd.cpu())
            del hidden, logits, log_probs, probabilities, owners, q, log_q, mixture, log_mixture, jsd
        q_count = len(samples)
        choices = torch.cat(expert_choices).view(q_count, self.args.top_k, len(CHOICES))
        jsd = torch.cat(expert_jsd).view(q_count, self.args.top_k)
        beta = jsd.mean(-1)
        results: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            results.append({
                "sample_id": sample.id,
                "dataset": sample.dataset,
                "row_idx": sample.row_idx,
                "gold_answer": sample.answer,
                "no_rag_choice_logits": [float(x) for x in no_choices[index].cpu().tolist()],
                "base_rag_choice_logits": [float(x) for x in base_choices[index].cpu().tolist()],
                "expert_choice_logits": [[float(x) for x in row] for row in choices[index].tolist()],
                "expert_jsd": [float(x) for x in jsd[index].tolist()],
                "beta": float(beta[index]),
                "prompt_packing": packing[index],
            })
        del no_hidden, base_hidden, expert_hidden, no_full, no_log_probs, no_probabilities
        del no_choices, base_choices, choices, jsd, beta
        return results

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def valid_score_shard(path: Path, expected_ids: Sequence[str], contract_hash: str) -> bool:
    marker = path.with_suffix(".complete.json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        rows = list(iter_jsonl(path))
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        metadata.get("contract_hash") == contract_hash
        and [str(row.get("sample_id") or "") for row in rows] == list(expected_ids)
    )


def cache_llama_scores(
    args: argparse.Namespace,
    samples: Sequence[BenchmarkSample],
    candidate_rows: Sequence[dict[str, Any]],
    contract_hash: str,
    progress: WorkflowProgress,
) -> list[dict[str, Any]]:
    root = args.output_dir / "direct_choice_score_shards"
    root.mkdir(parents=True, exist_ok=True)
    shards = [
        (start, min(len(samples), start + args.shard_size))
        for start in range(0, len(samples), args.shard_size)
    ]
    completed_questions = 0
    for shard_index, (start, stop) in enumerate(shards):
        path = root / f"shard_{shard_index:05d}.jsonl"
        expected = [sample.id for sample in samples[start:stop]]
        if valid_score_shard(path, expected, contract_hash):
            completed_questions += stop - start
    progress.start(3, len(samples), "question", initial=completed_questions)
    if completed_questions == len(samples):
        progress.complete(f"reused {len(shards)} complete score shards under {root}")
    else:
        scorer = DirectChoicePcedScorer(args)
        try:
            for shard_index, (start, stop) in enumerate(shards):
                path = root / f"shard_{shard_index:05d}.jsonl"
                expected = [sample.id for sample in samples[start:stop]]
                if valid_score_shard(path, expected, contract_hash):
                    continue
                rows: list[dict[str, Any]] = []
                shard_started = time.time()
                for cursor in range(start, stop, args.question_batch_size):
                    end = min(stop, cursor + args.question_batch_size)
                    rows.extend(scorer.score_questions(samples[cursor:end], candidate_rows[cursor:end]))
                    progress.update(end - cursor)
                atomic_jsonl(path, rows)
                atomic_json(path.with_suffix(".complete.json"), {
                    "contract_hash": contract_hash,
                    "shard_index": shard_index,
                    "questions": len(rows),
                    "elapsed_seconds": time.time() - shard_started,
                })
        finally:
            scorer.close()
        progress.complete(f"questions={len(samples)} shards={len(shards)} cache={root}")
    output: list[dict[str, Any]] = []
    for shard_index, (start, stop) in enumerate(shards):
        path = root / f"shard_{shard_index:05d}.jsonl"
        expected = [sample.id for sample in samples[start:stop]]
        if not valid_score_shard(path, expected, contract_hash):
            raise RuntimeError(f"Incomplete Llama score shard: {path}")
        output.extend(iter_jsonl(path))
    return output


def minmax(values: Sequence[float], epsilon: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"Non-finite prior score: {array}")
    span = float(array.max() - array.min())
    if span <= 1e-12:
        return np.full_like(array, 0.5)
    normalized = (array - array.min()) / span
    return epsilon + normalized * (1.0 - 2.0 * epsilon)


def rerank_prior(documents: Sequence[dict[str, Any]], epsilon: float) -> np.ndarray:
    """Map within-question reranker scores to the bounded PCED prior range."""
    return minmax([float(document["rerank_score"]) for document in documents], epsilon)


def matched_semantic_prior(rerank: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Keep the rerank-prior value distribution and replace only its ordering."""
    result = np.empty_like(rerank)
    result[np.argsort(semantic, kind="stable")] = np.sort(rerank)
    return result


def pced_prediction(
    no_logits: np.ndarray,
    expert_logits: np.ndarray,
    beta: float,
    prior: np.ndarray,
    gamma: float,
) -> tuple[str, int, list[float]]:
    scores = (1.0 + beta) * expert_logits - beta * no_logits[None, :]
    scores = scores + gamma * np.log(prior[:, None])
    best_by_choice = scores.max(axis=0)
    choice_index = int(best_by_choice.argmax())
    expert_index = int(scores[:, choice_index].argmax())
    return CHOICES[choice_index], expert_index, [float(value) for value in best_by_choice]


def condition_summary(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    per_dataset = {
        dataset: (
            sum(row["predictions"][condition] == row["gold_answer"] for row in grouped[dataset])
            / len(grouped[dataset])
            if grouped[dataset]
            else None
        )
        for dataset in DATASETS
    }
    mmlu_rows = [row for dataset in MMLU_DATASETS for row in grouped[dataset]]
    mmlu_accuracy = (
        sum(row["predictions"][condition] == row["gold_answer"] for row in mmlu_rows)
        / len(mmlu_rows)
        if mmlu_rows
        else None
    )
    present_dataset_scores = [value for value in per_dataset.values() if value is not None]
    major_group_scores = [
        value for value in (per_dataset["medmcqa"], per_dataset["medqa"], mmlu_accuracy)
        if value is not None
    ]
    return {
        "questions": len(rows),
        "correct": sum(row["predictions"][condition] == row["gold_answer"] for row in rows),
        "per_dataset_questions": {dataset: len(grouped[dataset]) for dataset in DATASETS},
        "per_dataset_accuracy": per_dataset,
        "medmcqa_accuracy": per_dataset["medmcqa"],
        "medqa_accuracy": per_dataset["medqa"],
        "mmlu_pooled_accuracy": mmlu_accuracy,
        "micro_accuracy": sum(row["predictions"][condition] == row["gold_answer"] for row in rows) / len(rows),
        "macro8_accuracy": float(np.mean(present_dataset_scores)),
        "macro3_accuracy": float(np.mean(major_group_scores)),
    }


def paired_comparison(
    rows: Sequence[dict[str, Any]], condition: str, baseline: str, replicates: int, seed: int,
) -> dict[str, Any]:
    proposed = np.asarray([row["predictions"][condition] == row["gold_answer"] for row in rows], dtype=np.float64)
    reference = np.asarray([row["predictions"][baseline] == row["gold_answer"] for row in rows], dtype=np.float64)
    delta = proposed - reference
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(rows), size=len(rows))
        bootstrap[index] = delta[selected].mean()
    return {
        "baseline": baseline,
        "accuracy_delta": float(delta.mean()),
        "paired_bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "wrong_to_correct": int(np.sum((reference == 0) & (proposed == 1))),
        "correct_to_wrong": int(np.sum((reference == 1) & (proposed == 0))),
        "net_correct": int(proposed.sum() - reference.sum()),
    }


def render_table(summary: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "—" if value is None else f"{100*value:.2f}"

    names = {
        "no_rag": "No-RAG",
        "base_rag": "Base-RAG (concatenated)",
        "pced_rerank": "PCED (rerank-score prior)",
        "pced_semantic": "PCED (semantic-support prior)",
        "pced_semantic_matched": "PCED (semantic rank, matched scale; diagnostic)",
    }
    lines = [
        f"Direct-Choice PCED evaluation (Top-{summary['top_k']})",
        "",
        "| Condition | N | MedMCQA | MedQA | MMLU pooled | Micro | Macro-8 | Macro-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["conditions"].items():
        lines.append(
            f"| {names[condition]} | {metrics['questions']} | "
            f"{percent(metrics['medmcqa_accuracy'])} | {percent(metrics['medqa_accuracy'])} | "
            f"{percent(metrics['mmlu_pooled_accuracy'])} | {percent(metrics['micro_accuracy'])} | "
            f"{percent(metrics['macro8_accuracy'])} | {percent(metrics['macro3_accuracy'])} |"
        )
    lines.extend([
        "",
        "| Condition | Anatomy | Clinical knowledge | College biology | College medicine | Medical genetics | Professional medicine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for condition, metrics in summary["conditions"].items():
        per_dataset = metrics["per_dataset_accuracy"]
        lines.append(
            f"| {names[condition]} | {percent(per_dataset['mmlu_anatomy'])} | "
            f"{percent(per_dataset['mmlu_clinical_knowledge'])} | "
            f"{percent(per_dataset['mmlu_college_biology'])} | "
            f"{percent(per_dataset['mmlu_college_medicine'])} | "
            f"{percent(per_dataset['mmlu_medical_genetics'])} | "
            f"{percent(per_dataset['mmlu_professional_medicine'])} |"
        )
    lines.extend(["", "Paired changes on the identical cohort:", ""])
    for name, comparison in summary["paired_comparisons"].items():
        interval = comparison["paired_bootstrap_95ci"]
        lines.append(
            f"- {name}: {100*comparison['accuracy_delta']:+.2f}%p "
            f"(95% CI {100*interval[0]:+.2f} to {100*interval[1]:+.2f}; "
            f"W→C={comparison['wrong_to_correct']}, C→W={comparison['correct_to_wrong']})"
        )
    lines.extend([
        "",
        "Scope: one constrained A/B/C/D decoding step. This does not evaluate multi-token expert switching.",
        "",
    ])
    return "\n".join(lines)


def aggregate(
    args: argparse.Namespace,
    samples: Sequence[BenchmarkSample],
    candidate_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    semantic_scores: dict[str, float],
    progress: WorkflowProgress,
) -> dict[str, Any]:
    score_by_id = {str(row["sample_id"]): row for row in score_rows}
    progress.start(4, len(samples) + args.bootstrap_replicates * 4, "unit")
    results: list[dict[str, Any]] = []
    selected_ranks: dict[str, list[int]] = defaultdict(list)
    selected_sources: dict[str, Counter[str]] = defaultdict(Counter)
    selected_semantic: dict[str, list[float]] = defaultdict(list)
    for sample, candidate in zip(samples, candidate_rows, strict=True):
        raw = score_by_id.get(sample.id)
        if raw is None:
            raise RuntimeError(f"Missing Llama scores: {sample.id}")
        documents = list(candidate["reranked_documents"])[: args.top_k]
        rerank = rerank_prior(documents, args.prior_epsilon)
        semantic = np.asarray(
            [semantic_scores[pair_key(sample, document)] for document in documents], dtype=np.float64
        )
        semantic = np.clip(semantic, args.prior_epsilon, 1.0 - args.prior_epsilon)
        matched = matched_semantic_prior(rerank, semantic)
        no_logits = np.asarray(raw["no_rag_choice_logits"], dtype=np.float64)
        base_logits = np.asarray(raw["base_rag_choice_logits"], dtype=np.float64)
        expert_logits = np.asarray(raw["expert_choice_logits"], dtype=np.float64)
        predictions = {
            "no_rag": CHOICES[int(no_logits.argmax())],
            "base_rag": CHOICES[int(base_logits.argmax())],
        }
        expert_indices: dict[str, int] = {}
        best_scores: dict[str, list[float]] = {}
        for condition, prior in (
            ("pced_rerank", rerank),
            ("pced_semantic", semantic),
            ("pced_semantic_matched", matched),
        ):
            prediction, expert_index, values = pced_prediction(
                no_logits, expert_logits, float(raw["beta"]), prior, args.gamma
            )
            predictions[condition] = prediction
            expert_indices[condition] = expert_index
            best_scores[condition] = values
            selected_ranks[condition].append(expert_index + 1)
            selected_sources[condition][str(documents[expert_index]["source"])] += 1
            selected_semantic[condition].append(float(semantic[expert_index]))
        results.append({
            "sample_id": sample.id,
            "dataset": sample.dataset,
            "row_idx": sample.row_idx,
            "gold_answer": sample.answer,
            "predictions": predictions,
            "beta": float(raw["beta"]),
            "expert_jsd": raw["expert_jsd"],
            "rerank_prior": rerank.tolist(),
            "semantic_support_probability": semantic.tolist(),
            "semantic_matched_prior": matched.tolist(),
            "selected_expert_index": expert_indices,
            "selected_expert_rank": {key: value + 1 for key, value in expert_indices.items()},
            "selected_expert_id": {
                key: str(documents[value].get("stable_id") or documents[value].get("corpus_id") or "")
                for key, value in expert_indices.items()
            },
            "pced_best_score_by_choice": best_scores,
            "no_rag_choice_logits": raw["no_rag_choice_logits"],
            "base_rag_choice_logits": raw["base_rag_choice_logits"],
            "expert_choice_logits": raw["expert_choice_logits"],
        })
        progress.update(1)
    conditions = (
        "no_rag", "base_rag", "pced_rerank", "pced_semantic", "pced_semantic_matched"
    )
    condition_metrics = {condition: condition_summary(results, condition) for condition in conditions}
    for condition, metrics in condition_metrics.items():
        metrics["mean_documents_available"] = 0.0 if condition == "no_rag" else float(args.top_k)
    comparisons: dict[str, Any] = {}
    specifications = (
        ("PCED rerank vs No-RAG", "pced_rerank", "no_rag"),
        ("PCED rerank vs Base-RAG", "pced_rerank", "base_rag"),
        ("PCED semantic vs PCED rerank", "pced_semantic", "pced_rerank"),
        ("PCED semantic vs Base-RAG", "pced_semantic", "base_rag"),
    )
    for index, (label, condition, baseline) in enumerate(specifications):
        comparisons[label] = paired_comparison(
            results, condition, baseline, args.bootstrap_replicates, args.seed + index
        )
        progress.update(args.bootstrap_replicates)
    expert_diagnostics = {
        condition: {
            "mean_selected_rerank_rank": float(np.mean(selected_ranks[condition])),
            "selected_source_counts": dict(selected_sources[condition]),
            "mean_selected_semantic_support_probability": float(np.mean(selected_semantic[condition])),
        }
        for condition in ("pced_rerank", "pced_semantic", "pced_semantic_matched")
    }
    summary = {
        "run_version": RUN_VERSION,
        "top_k": args.top_k,
        "questions": len(results),
        "conditions": condition_metrics,
        "paired_comparisons": comparisons,
        "expert_selection_diagnostics": expert_diagnostics,
        "pre_registered_success": {
            "pced_reproduction": {
                "rule": "PCED rerank minus Base-RAG micro accuracy >= +1.0%p and paired 95% CI lower bound > 0",
                "passed": (
                    comparisons["PCED rerank vs Base-RAG"]["accuracy_delta"] >= 0.01
                    and comparisons["PCED rerank vs Base-RAG"]["paired_bootstrap_95ci"][0] > 0
                ),
            },
            "semantic_prior": {
                "rule": "PCED semantic minus PCED rerank micro accuracy >= +0.5%p and paired 95% CI lower bound > 0",
                "passed": (
                    comparisons["PCED semantic vs PCED rerank"]["accuracy_delta"] >= 0.005
                    and comparisons["PCED semantic vs PCED rerank"]["paired_bootstrap_95ci"][0] > 0
                ),
            },
        },
    }
    atomic_jsonl(args.output_dir / "predictions.jsonl", results)
    atomic_json(args.output_dir / "summary.json", summary)
    table = render_table(summary)
    (args.output_dir / "summary_table_pretty.txt").write_text(table, encoding="utf-8")
    progress.complete(
        f"predictions={args.output_dir/'predictions.jsonl'} summary={args.output_dir/'summary.json'}"
    )
    print(table, flush=True)
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0 or args.question_batch_size <= 0 or args.prompt_batch_size <= 0:
        raise ValueError("Top-k and batch sizes must be positive")
    if args.shard_size <= 0 or args.max_input_tokens <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("Shard size, max input tokens, and bootstrap replicates must be positive")
    if not (0.0 < args.prior_epsilon < 0.5) or args.gamma < 0:
        raise ValueError("Invalid prior epsilon or gamma")
    for path in (
        args.candidate_cache,
        args.llama_model / "config.json",
        args.medmcqa_semantic_model / "config.json",
        args.medqa_semantic_model / "config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    validate_args(args)
    progress = WorkflowProgress(
        [
            "preflight benchmark/candidate/model contracts",
            f"score Top-{args.top_k} Semantic-Support probabilities",
            f"score no-RAG, concatenated Top-{args.top_k}, and {args.top_k} independent Llama experts",
            "apply both PCED priors, bootstrap paired metrics, and write report",
        ],
        [45.0, 240.0, 1500.0, 25.0],
    )
    try:
        progress.start(1, 3, "check")
        samples, candidates = load_inputs(args)
        progress.update(1)
        manifest = contract(args, samples)
        progress.update(1)
        contract_hash = ensure_contract(args.output_dir, manifest)
        tokenizer = AutoTokenizer.from_pretrained(args.llama_model, local_files_only=True, use_fast=True)
        audit_count = min(128, len(samples))
        maximum = 0
        sampled_overlength_base = 0
        for sample, row in zip(samples[:audit_count], candidates[:audit_count], strict=True):
            docs = list(row["reranked_documents"])[: args.top_k]
            no_length = len(sequence_for_prompt(tokenizer, sample, None)[0])
            expert_lengths = [
                len(sequence_for_prompt(tokenizer, sample, str(document["text"]).strip())[0])
                for document in docs
            ]
            base_length = len(sequence_for_prompt(
                tokenizer, sample, "\n\n".join(str(document["text"]).strip() for document in docs)
            )[0])
            sampled_overlength_base += int(base_length > args.max_input_tokens)
            maximum = max(maximum, no_length, *expert_lengths)
        del tokenizer
        if maximum > args.max_input_tokens:
            raise RuntimeError(f"Sample no-RAG/single-expert prompt exceeds max input tokens: {maximum}")
        progress.update(1)
        progress.complete(
            f"questions={len(samples)} documents={len(samples)*args.top_k} "
            f"sampled_single_expert_max_tokens={maximum} "
            f"sampled_base_prompts_requiring_equal_token_truncation={sampled_overlength_base}/{audit_count} "
            f"manifest={args.output_dir/'experiment_manifest.json'}"
        )
        if args.preflight_only:
            print("[preflight-only complete] no model scoring was run", flush=True)
            return
        semantic = cache_semantic_scores(args, samples, candidates, contract_hash, progress)
        if args.attn_implementation == "sdpa":
            # PyTorch may select cuDNN SDPA for these variable-length left-padded
            # batches, but that backend has no execution plan for some shapes on
            # the installed stack. Flash/efficient/math SDPA are supported here.
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print(
                "[SDPA policy] cuDNN disabled; Flash/Efficient/Math enabled",
                flush=True,
            )
        llama = cache_llama_scores(args, samples, candidates, contract_hash, progress)
        aggregate(args, samples, candidates, llama, semantic, progress)
    except Exception as exc:
        progress.fail(
            f"{type(exc).__name__}: {exc}; rerun the identical command to reuse durable completed caches"
        )
        raise


if __name__ == "__main__":
    main()

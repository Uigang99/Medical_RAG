#!/usr/bin/env python3
"""Cache exact direct-choice outcomes for every train question and Top-8 document.

For each immutable reranked candidate row this script evaluates the frozen
target Llama at the *same* direct-choice decision point:

  * no-RAG:       ``Q + options -> Final answer:``
  * single-doc:   ``Q + options + document D_i -> Final answer:``

The model is not asked to generate a rationale.  The answer is the greedy
maximum among the four permitted next-token choices A/B/C/D.  Raw vocabulary
logits for those four tokens and their conditional four-way probabilities are
stored losslessly in sharded safetensors files; JSONL rows retain only
reproducibility metadata and derived features.  Gold-answer fields are used
only after scoring to materialize analysis features and never enter a prompt.

Outputs are question-sharded, atomically committed, contract-keyed, and safe
to resume.  The cache is intentionally reusable by later behavioral-label,
subset, calibration, and semantic/behavioral-gap analyses.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_document_traces import (  # noqa: E402
    normalized_candidate_rows,
)
from generate_rag2_anchored_layer_pilot import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)
from medrag.core import BenchmarkSample  # noqa: E402
from medrag.filtering.rag2_preanswer_text_hidden import (  # noqa: E402
    CHOICES,
    FINAL_ANSWER_PREFILL,
    PREANSWER_PROMPT_VERSION,
    build_preanswer_user_prompt,
)


RUN_VERSION = "rag2_direct_choice_single_document_outcomes_v1"
PROMPT_POLICY_VERSION = "rag2_fixed_direct_choice_context_v1"
SCORE_POLICY_VERSION = "exact_four_choice_next_token_logits_v1"
FEATURE_POLICY_VERSION = "direct_choice_outcome_features_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "candidates/source_balanced32_rerank8_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    / "direct_choice_single_document_outcomes_source_balanced32_rerank8_v1"
)
SUPPORTED_DATASETS = ("medmcqa", "medqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-file", default="candidates_top8.jsonl")
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--expected-per-source-top-k", type=int, default=8)
    parser.add_argument("--expected-candidate-pool-top-k", type=int, default=32)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--questions-per-shard", type=int, default=1024)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=128,
        help="Initial exact-forward batch size. CUDA OOM retries halve it without changing outputs.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument(
        "--questions-for-prompt-sample",
        type=int,
        default=32,
        help="Per-dataset prompt sample checked by --dry-run before the complete preflight audit.",
    )
    parser.add_argument(
        "--estimated-json-bytes-per-pair",
        type=int,
        default=1250,
        help="Conservative metadata size used only for disk preflight.",
    )
    parser.add_argument("--disk-reserve-gib", type=float, default=20.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
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


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(str(value).encode("utf-8"))


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
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        value["sha256"] = sha256_file(path)
    return value


def model_identity(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    files = []
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        path = model_path / name
        if path.is_file():
            files.append({"name": name, **file_identity(path, content_hash=True)})
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        weights = sorted(model_path.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No model weights under {model_path}")
    files.extend({"name": path.name, **file_identity(path)} for path in weights)
    return {"path": str(model_path.resolve()), "files": files}


def candidate_paths(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    root = args.candidate_root / dataset / args.split
    return root / args.candidate_file, root / "candidate_manifest.json"


def candidate_contract(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    path, manifest_path = candidate_paths(args, dataset)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing candidate inputs for {dataset}: {path} / {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": dataset,
        "split": args.split,
        "candidate_layout": "source_balanced",
        "per_source_top_k": args.expected_per_source_top_k,
        "candidate_pool_top_k": args.expected_candidate_pool_top_k,
        "top_k": args.docs_per_question,
    }
    mismatches = {key: {"expected": value, "actual": manifest.get(key)} for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Candidate contract mismatch for {dataset}: {mismatches}")
    questions = int(manifest.get("selected_question_count", -1))
    if questions <= 0:
        raise RuntimeError(f"Invalid selected_question_count for {dataset}: {questions}")
    return {
        "dataset": dataset,
        "candidate_path": file_identity(path),
        "candidate_manifest_path": file_identity(manifest_path, content_hash=True),
        "selected_question_count": questions,
        "selected_pair_count": questions * int(args.docs_per_question),
        "source_layout": {
            "candidate_layout": manifest["candidate_layout"],
            "per_source_top_k": int(manifest["per_source_top_k"]),
            "candidate_pool_top_k": int(manifest["candidate_pool_top_k"]),
            "rerank_top_k": int(manifest["top_k"]),
            "sources": list(manifest.get("sources") or []),
        },
    }


def immutable_contract(args: argparse.Namespace, candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "score_policy_version": SCORE_POLICY_VERSION,
        "feature_policy_version": FEATURE_POLICY_VERSION,
        "preanswer_prompt_version": PREANSWER_PROMPT_VERSION,
        "final_answer_prefill": FINAL_ANSWER_PREFILL,
        "choice_labels": list(CHOICES),
        "choice_tokenization": "leading_space_single_token_after_final_answer_colon_v1",
        "datasets": list(args.datasets),
        "split": args.split,
        "docs_per_question": int(args.docs_per_question),
        "questions_per_shard": int(args.questions_per_shard),
        "max_input_tokens": int(args.max_input_tokens),
        "model": model_identity(args.model_name_or_path),
        "candidate_contracts": candidates,
        "code_sha256": sha256_file(Path(__file__)),
    }


def contract_digest(contract: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def ensure_contract(output_root: Path, contract: dict[str, Any]) -> str:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "run_contract.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                "Direct-choice cache contract mismatch. Use a new --output-root; completed cache contents will not be overwritten."
            )
    else:
        atomic_write_json(path, contract)
    return contract_digest(contract)


def make_sample(row: dict[str, Any]) -> BenchmarkSample:
    return BenchmarkSample(
        row_idx=int(row["row_idx"]),
        id=str(row["sample_id"]),
        task=str(row["dataset"]),
        collection="unified",
        dataset=str(row["dataset"]),
        split=str(row["split"]),
        question=str(row["question"]),
        options=dict(row["options"]),
        answer=str(row["answer"]),
        answers=[str(row["answer"])],
        raw=dict(row),
    )


def render_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(rendered) + FINAL_ANSWER_PREFILL


def sequence_for_prompt(tokenizer: Any, sample: BenchmarkSample, document_text: str | None) -> tuple[list[int], str]:
    user_prompt = build_preanswer_user_prompt(sample, document_text)
    prompt = render_chat_prompt(tokenizer, user_prompt)
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    marker_ids = list(tokenizer.encode(FINAL_ANSWER_PREFILL, add_special_tokens=False))
    if token_ids[-len(marker_ids) :] != marker_ids:
        raise RuntimeError("Rendered prompt does not end in the fixed Final answer: marker")
    return token_ids, prompt


def document_text(document: dict[str, Any]) -> str:
    value = " ".join(str(document.get("text") or document.get("title") or "").split())
    if not value:
        raise ValueError("Encountered an empty reranked document")
    return value


def document_metadata(document: dict[str, Any], text: str) -> dict[str, Any]:
    return {
        "pair_id": str(document["pair_id"]),
        "source": str(document.get("source") or ""),
        "stable_id": str(
            document.get("stable_id")
            or document.get("corpus_id")
            or document.get("chunk_id")
            or document.get("db_id")
            or f"{document.get('source')}:{document.get('local_id')}"
        ),
        "rerank_rank": int(document["rerank_rank"]),
        "rerank_score": document.get("rerank_score"),
        "retrieval_rank": document.get("retrieval_rank"),
        "retrieval_score": document.get("retrieval_score"),
        "document_text_sha256": sha256_text(text),
        "document_char_count": len(text),
    }


def shard_paths(root: Path, dataset: str, split: str, index: int) -> dict[str, Path]:
    directory = root / "outcome_shards" / dataset / split / f"shard_{index:05d}"
    return {
        "root": directory,
        "questions": directory / "questions.jsonl",
        "pairs": directory / "pairs.jsonl",
        "scores": directory / "scores.safetensors",
        "complete": directory / "COMPLETE.json",
    }


def valid_complete(paths: dict[str, Path], *, contract_sha256: str, question_count: int, pair_count: int) -> bool:
    if not all(paths[name].is_file() for name in ("questions", "pairs", "scores", "complete")):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
        with safe_open(paths["scores"], framework="pt", device="cpu") as handle:
            required = {
                "no_rag_choice_logits": (question_count, 4),
                "no_rag_choice_probabilities": (question_count, 4),
                "single_document_choice_logits": (pair_count, 4),
                "single_document_choice_probabilities": (pair_count, 4),
            }
            if set(handle.keys()) != set(required):
                return False
            for key, shape in required.items():
                if tuple(handle.get_tensor(key).shape) != shape:
                    return False
    except Exception:
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and marker.get("contract_sha256") == contract_sha256
        and int(marker.get("question_count", -1)) == question_count
        and int(marker.get("pair_count", -1)) == pair_count
        and int(marker.get("questions_size_bytes", -1)) == paths["questions"].stat().st_size
        and int(marker.get("pairs_size_bytes", -1)) == paths["pairs"].stat().st_size
        and int(marker.get("scores_size_bytes", -1)) == paths["scores"].stat().st_size
    )


def pad_left(sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        input_ids[index, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, -len(sequence) :] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


class ExactDirectChoiceScorer:
    """Score A/B/C/D from one frozen Llama forward without vocabulary-logit waste."""

    def __init__(self, args: argparse.Namespace) -> None:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
        self.dtype = dtype_map[args.dtype]
        self.max_input_tokens = int(args.max_input_tokens)
        self.batch_size = max(1, int(args.prompt_batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.choice_token_ids = {
            choice: self._leading_space_choice_token_id(choice) for choice in CHOICES
        }
        self.choice_ids = torch.tensor([self.choice_token_ids[choice] for choice in CHOICES], dtype=torch.long)
        logging.info(
            "Loading frozen direct-choice scorer: model=%s device=%s dtype=%s attention=%s choice_token_ids=%s",
            args.model_name_or_path,
            self.device,
            args.dtype,
            args.attn_implementation,
            self.choice_token_ids,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(args.model_name_or_path),
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation=args.attn_implementation,
            )
        except (ImportError, ValueError) as exc:
            if args.attn_implementation != "flash_attention_2":
                raise
            logging.warning("flash_attention_2 unavailable (%s); retrying with SDPA", exc)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(args.model_name_or_path),
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation="sdpa",
            )
        self.model.requires_grad_(False)
        self.model.eval().to(self.device)
        self.decoder = getattr(self.model, "model", None)
        if self.decoder is None:
            raise RuntimeError("Expected a Llama-style causal LM exposing .model decoder")
        self.output_embeddings = self.model.get_output_embeddings()
        if self.output_embeddings is None:
            raise RuntimeError("Causal LM has no output embedding head")

    def _leading_space_choice_token_id(self, choice: str) -> int:
        ids = self.tokenizer.encode(f" {choice}", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice continuation {choice!r} is not exactly one leading-space token: {ids}")
        return int(ids[0])

    def score(self, sequences: Sequence[Sequence[int]]) -> np.ndarray:
        if not sequences:
            return np.empty((0, len(CHOICES)), dtype=np.float32)
        for sequence in sequences:
            if len(sequence) > self.max_input_tokens:
                raise ValueError(
                    f"Direct-choice prompt exceeds --max-input-tokens={self.max_input_tokens}: {len(sequence)}"
                )
        results = np.empty((len(sequences), len(CHOICES)), dtype=np.float32)
        order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
        cursor = 0
        active_batch = self.batch_size
        while cursor < len(order):
            batch_indices = order[cursor : cursor + active_batch]
            batch_sequences = [sequences[index] for index in batch_indices]
            input_ids = attention_mask = position_ids = None
            outputs = last_hidden = four_logits = None
            try:
                input_ids, attention_mask, position_ids = pad_left(
                    batch_sequences, int(self.tokenizer.pad_token_id), self.device
                )
                with torch.inference_mode():
                    outputs = self.decoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_hidden = outputs.last_hidden_state[:, -1, :]
                    four_logits = self.output_embeddings(last_hidden).index_select(
                        dim=-1, index=self.choice_ids.to(last_hidden.device)
                    )
                values = four_logits.float().cpu().numpy()
                for row_index, value in zip(batch_indices, values):
                    results[row_index] = value
                cursor += len(batch_indices)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or active_batch <= 1:
                    raise
                torch.cuda.empty_cache()
                next_batch = max(1, active_batch // 2)
                logging.warning(
                    "CUDA OOM while scoring direct choices; retrying current batch with prompt_batch_size=%d -> %d",
                    active_batch,
                    next_batch,
                )
                active_batch = next_batch
            finally:
                if input_ids is not None:
                    del input_ids
                if attention_mask is not None:
                    del attention_mask
                if position_ids is not None:
                    del position_ids
                if outputs is not None:
                    del outputs
                if last_hidden is not None:
                    del last_hidden
                if four_logits is not None:
                    del four_logits
        return results

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def choice_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64, copy=False))
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def summary_features(logits: np.ndarray, gold_index: int) -> dict[str, Any]:
    probabilities = choice_probabilities(logits[np.newaxis, :])[0]
    prediction_index = int(np.argmax(logits))
    ordered = np.argsort(logits)[::-1]
    wrong = [index for index in range(len(CHOICES)) if index != gold_index]
    gold_margin = float(logits[gold_index] - np.max(logits[wrong]))
    top1_margin = float(logits[ordered[0]] - logits[ordered[1]])
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    gold_rank = int(np.where(ordered == gold_index)[0][0]) + 1
    return {
        "prediction": CHOICES[prediction_index],
        "prediction_index": prediction_index,
        "answer_correct": bool(prediction_index == gold_index),
        "choice_entropy": entropy,
        "top1_probability": float(probabilities[prediction_index]),
        "top1_margin": top1_margin,
        "gold_probability": float(probabilities[gold_index]),
        "gold_margin": gold_margin,
        "gold_rank": gold_rank,
    }


def transition_label(no_rag_correct: bool, document_correct: bool) -> str:
    return f"{'C' if no_rag_correct else 'W'}2{'C' if document_correct else 'W'}"


def context_audit(
    args: argparse.Namespace,
    tokenizer: Any,
    candidates: dict[str, dict[str, Any]],
    *,
    full: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    total_start = time.monotonic()
    logging.info(
        "[overall 1/2 | elapsed 00h00m00s | overall ETA unknown until direct-forward throughput is measured] "
        "preflight candidate integrity, prompt lengths, and disk capacity"
    )
    for dataset in args.datasets:
        path = Path(candidates[dataset]["candidate_path"]["path"])
        planned_questions = int(candidates[dataset]["selected_question_count"])
        limit = planned_questions if full else min(planned_questions, max(1, int(args.questions_for_prompt_sample)))
        progress = tqdm(total=limit, desc=f"DirectChoicePreflight:{dataset}", unit="question", dynamic_ncols=True)
        observed_questions = 0
        observed_pairs = 0
        seen_samples: set[str] = set()
        max_tokens = 0
        max_record: dict[str, Any] | None = None
        no_rag_tokens: list[int] = []
        single_doc_tokens: list[int] = []
        for row in normalized_candidate_rows(path, dataset, args.split, args.docs_per_question):
            if observed_questions >= limit:
                break
            sample = make_sample(row)
            if sample.id in seen_samples:
                raise RuntimeError(f"Duplicate sample_id in {dataset}: {sample.id}")
            seen_samples.add(sample.id)
            no_ids, _ = sequence_for_prompt(tokenizer, sample, None)
            lengths = [("no_rag", None, len(no_ids))]
            for document in row["documents"]:
                text = document_text(document)
                ids, _ = sequence_for_prompt(tokenizer, sample, text)
                lengths.append(("single_document", int(document["rerank_rank"]), len(ids)))
            for kind, rank, length in lengths:
                if length > int(args.max_input_tokens):
                    raise RuntimeError(
                        f"Prompt exceeds max_input_tokens={args.max_input_tokens}: dataset={dataset} "
                        f"sample={sample.id} condition={kind} rank={rank} tokens={length}"
                    )
                if kind == "no_rag":
                    no_rag_tokens.append(length)
                else:
                    single_doc_tokens.append(length)
                if length > max_tokens:
                    max_tokens = length
                    max_record = {"sample_id": sample.id, "condition": kind, "rerank_rank": rank, "tokens": length}
            observed_questions += 1
            observed_pairs += len(row["documents"])
            progress.update(1)
            if observed_questions % 32 == 0 or observed_questions == limit:
                elapsed = max(1e-9, time.monotonic() - total_start)
                rate = observed_questions / elapsed
                remaining = max(0, limit - observed_questions)
                progress.set_postfix_str(
                    f"pairs={observed_pairs} max_tokens={max_tokens} rate={rate:.1f}q/s ETA={format_duration(remaining / rate)}",
                    refresh=False,
                )
        progress.close()
        if full and observed_questions != planned_questions:
            raise RuntimeError(f"Candidate coverage mismatch for {dataset}: {observed_questions} != {planned_questions}")
        if observed_pairs != observed_questions * args.docs_per_question:
            raise RuntimeError(f"Candidate pair count mismatch in {dataset}: {observed_pairs}")
        results[dataset] = {
            "questions_checked": observed_questions,
            "pairs_checked": observed_pairs,
            "full_audit": bool(full),
            "max_prompt_tokens": max_tokens,
            "max_prompt": max_record,
            "no_rag_prompt_token_summary": numeric_summary(no_rag_tokens),
            "single_document_prompt_token_summary": numeric_summary(single_doc_tokens),
        }
    return results


def numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def disk_preflight(args: argparse.Namespace, candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_questions = sum(int(value["selected_question_count"]) for value in candidates.values())
    total_pairs = sum(int(value["selected_pair_count"]) for value in candidates.values())
    tensor_bytes = (total_questions + total_pairs) * 4 * 2 * np.dtype(np.float32).itemsize
    metadata_bytes = total_questions * 700 + total_pairs * int(args.estimated_json_bytes_per_pair)
    projected = tensor_bytes + metadata_bytes
    usage = shutil.disk_usage(args.output_root.parent)
    reserve = int(float(args.disk_reserve_gib) * 1024**3)
    if usage.free < projected + reserve:
        raise RuntimeError(
            f"Insufficient disk for direct-choice cache: free={usage.free / 1024**3:.2f}GiB, "
            f"projected={projected / 1024**3:.2f}GiB, reserve={args.disk_reserve_gib:.2f}GiB"
        )
    return {
        "total_questions": total_questions,
        "total_single_document_pairs": total_pairs,
        "total_direct_choice_prompts": total_questions + total_pairs,
        "estimated_tensor_gib": tensor_bytes / 1024**3,
        "estimated_metadata_gib": metadata_bytes / 1024**3,
        "estimated_total_gib": projected / 1024**3,
        "free_gib_before_run": usage.free / 1024**3,
        "reserve_gib": float(args.disk_reserve_gib),
    }


def stream_chunks(values: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def score_question_shard(
    scorer: ExactDirectChoiceScorer,
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor], dict[str, int]]:
    sequences: list[list[int]] = []
    specs: list[dict[str, Any]] = []
    question_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    for question_index, row in enumerate(rows):
        sample = make_sample(row)
        gold_index = CHOICES.index(str(sample.answer))
        no_ids, no_prompt = sequence_for_prompt(scorer.tokenizer, sample, None)
        sequences.append(no_ids)
        specs.append(
            {
                "condition": "no_rag",
                "question_index": question_index,
                "sample": sample,
                "gold_index": gold_index,
                "prompt_sha256": sha256_text(no_prompt),
                "prompt_tokens": len(no_ids),
                "document": None,
            }
        )
        for document in row["documents"]:
            text = document_text(document)
            ids, prompt = sequence_for_prompt(scorer.tokenizer, sample, text)
            sequences.append(ids)
            specs.append(
                {
                    "condition": "single_document",
                    "question_index": question_index,
                    "sample": sample,
                    "gold_index": gold_index,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_tokens": len(ids),
                    "document": document_metadata(document, text),
                }
            )

    logits = scorer.score(sequences)
    probabilities = choice_probabilities(logits)
    no_logits = np.empty((len(rows), len(CHOICES)), dtype=np.float32)
    no_probabilities = np.empty((len(rows), len(CHOICES)), dtype=np.float32)
    pair_logits = np.empty((len(rows) * 8, len(CHOICES)), dtype=np.float32)
    pair_probabilities = np.empty((len(rows) * 8, len(CHOICES)), dtype=np.float32)
    question_state: dict[int, tuple[dict[str, Any], np.ndarray, np.ndarray, int]] = {}
    pair_offset = 0
    for spec, current_logits, current_probabilities in zip(specs, logits, probabilities):
        sample: BenchmarkSample = spec["sample"]
        gold_index = int(spec["gold_index"])
        if spec["condition"] == "no_rag":
            q_index = int(spec["question_index"])
            features = summary_features(current_logits, gold_index)
            no_logits[q_index] = current_logits
            no_probabilities[q_index] = current_probabilities
            record = {
                "run_version": RUN_VERSION,
                "dataset": sample.dataset,
                "split": sample.split,
                "sample_id": sample.id,
                "row_idx": sample.row_idx,
                "tensor_row": q_index,
                "gold_answer": sample.answer,
                "prompt_sha256": spec["prompt_sha256"],
                "prompt_token_count": int(spec["prompt_tokens"]),
                "choice_tensor_key": "no_rag_choice_logits",
                "probability_tensor_key": "no_rag_choice_probabilities",
                **features,
            }
            question_records.append(record)
            question_state[q_index] = (record, current_logits, current_probabilities, gold_index)
            continue

        q_index = int(spec["question_index"])
        question_record, baseline_logits, baseline_probabilities, gold_index = question_state[q_index]
        features = summary_features(current_logits, gold_index)
        baseline_features = summary_features(baseline_logits, gold_index)
        pair_logits[pair_offset] = current_logits
        pair_probabilities[pair_offset] = current_probabilities
        record = {
            "run_version": RUN_VERSION,
            "dataset": sample.dataset,
            "split": sample.split,
            "sample_id": sample.id,
            "row_idx": sample.row_idx,
            "question_tensor_row": q_index,
            "tensor_row": pair_offset,
            "gold_answer": sample.answer,
            "prompt_sha256": spec["prompt_sha256"],
            "prompt_token_count": int(spec["prompt_tokens"]),
            "choice_tensor_key": "single_document_choice_logits",
            "probability_tensor_key": "single_document_choice_probabilities",
            "no_rag_prediction": question_record["prediction"],
            "no_rag_answer_correct": bool(question_record["answer_correct"]),
            "no_rag_gold_probability": float(baseline_features["gold_probability"]),
            "no_rag_gold_margin": float(baseline_features["gold_margin"]),
            "delta_gold_probability": float(features["gold_probability"] - baseline_features["gold_probability"]),
            "delta_gold_margin": float(features["gold_margin"] - baseline_features["gold_margin"]),
            "delta_choice_logits": [float(value) for value in (current_logits - baseline_logits)],
            "delta_choice_probabilities": [float(value) for value in (current_probabilities - baseline_probabilities)],
            "prediction_changed_from_no_rag": bool(features["prediction"] != question_record["prediction"]),
            "correctness_transition": transition_label(bool(question_record["answer_correct"]), bool(features["answer_correct"])),
            "document": spec["document"],
            **features,
        }
        pair_records.append(record)
        pair_offset += 1
    if len(question_records) != len(rows) or len(pair_records) != len(rows) * 8:
        raise RuntimeError("Scored shard cardinality mismatch")
    tensors = {
        "no_rag_choice_logits": torch.from_numpy(no_logits),
        "no_rag_choice_probabilities": torch.from_numpy(no_probabilities),
        "single_document_choice_logits": torch.from_numpy(pair_logits),
        "single_document_choice_probabilities": torch.from_numpy(pair_probabilities),
    }
    transitions = Counter(record["correctness_transition"] for record in pair_records)
    return question_records, pair_records, tensors, dict(transitions)


def atomic_write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    save_safetensors(tensors, str(temporary), metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.questions_per_shard <= 0 or args.prompt_batch_size <= 0:
        raise ValueError("--questions-per-shard and --prompt-batch-size must be positive")
    if args.docs_per_question != 8:
        raise ValueError("This train cache is deliberately fixed to the stored rerank Top-8 candidates")
    if args.max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be positive")

    candidates = {dataset: candidate_contract(args, dataset) for dataset in args.datasets}
    disk = disk_preflight(args, candidates)
    contract = immutable_contract(args, candidates)
    contract_sha256 = contract_digest(contract)

    # A dry-run does not create a cache directory or write a manifest. It validates
    # every immutable input contract plus a representative rendered prompt sample.
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
    )
    if args.dry_run:
        audit = context_audit(args, tokenizer, candidates, full=False)
        logging.info(
            "Dry-run passed: questions=%d single_document_pairs=%d prompts=%d projected_output=%.2fGiB free=%.2fGiB sample_audit=%s",
            disk["total_questions"],
            disk["total_single_document_pairs"],
            disk["total_direct_choice_prompts"],
            disk["estimated_total_gib"],
            disk["free_gib_before_run"],
            audit,
        )
        return

    ensure_contract(args.output_root, contract)
    audit_path = args.output_root / "preflight_context_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("contract_sha256") != contract_sha256:
            raise RuntimeError("Existing preflight audit belongs to a different contract")
        logging.info("Reusing complete direct-choice prompt audit: %s", audit_path)
    else:
        audit = {
            "run_version": RUN_VERSION,
            "contract_sha256": contract_sha256,
            "created_at": utc_now(),
            "disk_preflight": disk,
            "datasets": context_audit(args, tokenizer, candidates, full=True),
        }
        atomic_write_json(audit_path, audit)
        logging.info("Direct-choice preflight complete: %s", audit_path)
    if args.preflight_only:
        logging.info("Preflight-only requested; no Llama forward was run.")
        return

    total_questions = int(disk["total_questions"])
    total_pairs = int(disk["total_single_document_pairs"])
    completed_questions = 0
    completed_pairs = 0
    for dataset in args.datasets:
        count = int(candidates[dataset]["selected_question_count"])
        for shard_index in range(math.ceil(count / args.questions_per_shard)):
            expected_questions = min(args.questions_per_shard, count - shard_index * args.questions_per_shard)
            expected_pairs = expected_questions * args.docs_per_question
            paths = shard_paths(args.output_root, dataset, args.split, shard_index)
            if args.resume and valid_complete(
                paths,
                contract_sha256=contract_sha256,
                question_count=expected_questions,
                pair_count=expected_pairs,
            ):
                completed_questions += expected_questions
                completed_pairs += expected_pairs
    if completed_questions > total_questions or completed_pairs > total_pairs:
        raise RuntimeError("Completed shard count exceeds immutable candidate contract")

    logging.info(
        "[overall 2/2 | elapsed 00h00m00s | overall ETA calibrates after first scored shard] "
        "exact direct-choice scoring: cached_questions=%d/%d cached_single_document_pairs=%d/%d",
        completed_questions,
        total_questions,
        completed_pairs,
        total_pairs,
    )
    scorer = None
    started = time.monotonic()
    total_transitions: Counter[str] = Counter()
    try:
        if completed_questions < total_questions:
            scorer = ExactDirectChoiceScorer(args)
        for dataset in args.datasets:
            path = Path(candidates[dataset]["candidate_path"]["path"])
            planned_questions = int(candidates[dataset]["selected_question_count"])
            total_prompts = planned_questions * (1 + args.docs_per_question)
            done_prompts = 0
            progress = tqdm(
                total=total_prompts,
                initial=0,
                desc=f"DirectChoiceScore:{dataset}",
                unit="prompt",
                dynamic_ncols=True,
            )
            observed_questions = 0
            shard_total = math.ceil(planned_questions / args.questions_per_shard)
            for shard_index, rows in enumerate(
                stream_chunks(
                    normalized_candidate_rows(path, dataset, args.split, args.docs_per_question),
                    args.questions_per_shard,
                )
            ):
                expected_questions = len(rows)
                expected_pairs = expected_questions * args.docs_per_question
                paths = shard_paths(args.output_root, dataset, args.split, shard_index)
                if args.resume and valid_complete(
                    paths,
                    contract_sha256=contract_sha256,
                    question_count=expected_questions,
                    pair_count=expected_pairs,
                ):
                    progress.update(expected_questions + expected_pairs)
                    done_prompts += expected_questions + expected_pairs
                    observed_questions += expected_questions
                    continue
                if scorer is None:
                    raise RuntimeError("An incomplete shard requires the frozen direct-choice scorer")
                shard_started = time.monotonic()
                questions, pairs, tensors, transitions = score_question_shard(scorer, rows)
                paths["root"].mkdir(parents=True, exist_ok=True)
                atomic_write_jsonl(paths["questions"], questions)
                atomic_write_jsonl(paths["pairs"], pairs)
                atomic_write_safetensors(
                    paths["scores"],
                    tensors,
                    {
                        "run_version": RUN_VERSION,
                        "contract_sha256": contract_sha256,
                        "choice_labels": ",".join(CHOICES),
                        "probability_definition": "softmax_over_A_B_C_D_only",
                    },
                )
                marker = {
                    "run_version": RUN_VERSION,
                    "contract_sha256": contract_sha256,
                    "completed_at": utc_now(),
                    "dataset": dataset,
                    "split": args.split,
                    "shard_index": shard_index,
                    "question_count": len(questions),
                    "pair_count": len(pairs),
                    "correctness_transitions": transitions,
                    "questions_size_bytes": paths["questions"].stat().st_size,
                    "pairs_size_bytes": paths["pairs"].stat().st_size,
                    "scores_size_bytes": paths["scores"].stat().st_size,
                    "shard_elapsed_seconds": time.monotonic() - shard_started,
                }
                atomic_write_json(paths["complete"], marker)
                total_transitions.update(transitions)
                delta = len(questions) + len(pairs)
                progress.update(delta)
                done_prompts += delta
                observed_questions += len(rows)
                elapsed = max(1e-9, time.monotonic() - started)
                prompt_rate = (completed_questions + completed_pairs + done_prompts) / elapsed
                remaining_prompts = (total_questions + total_pairs) - (completed_questions + completed_pairs + done_prompts)
                progress.set_postfix_str(
                    f"shard={shard_index + 1}/{shard_total} q={observed_questions}/{planned_questions} "
                    f"rate={prompt_rate:.1f}prompt/s overall_ETA={format_duration(remaining_prompts / prompt_rate)}",
                    refresh=False,
                )
            progress.close()
            if observed_questions != planned_questions:
                raise RuntimeError(f"Scoring coverage mismatch for {dataset}: {observed_questions} != {planned_questions}")

        markers = []
        for dataset in args.datasets:
            markers.extend(
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((args.output_root / "outcome_shards" / dataset / args.split).glob("shard_*/COMPLETE.json"))
            )
        if len(markers) != sum(math.ceil(int(candidates[dataset]["selected_question_count"]) / args.questions_per_shard) for dataset in args.datasets):
            raise RuntimeError("Final shard marker coverage mismatch")
        total_transitions = Counter()
        for marker in markers:
            total_transitions.update(marker.get("correctness_transitions") or {})
        manifest = {
            "run_version": RUN_VERSION,
            "contract_sha256": contract_sha256,
            "completed_at": utc_now(),
            "output_root": str(args.output_root.resolve()),
            "datasets": {dataset: int(candidates[dataset]["selected_question_count"]) for dataset in args.datasets},
            "total_questions": total_questions,
            "total_single_document_pairs": total_pairs,
            "total_direct_choice_prompts": total_questions + total_pairs,
            "choice_token_ids": scorer.choice_token_ids if scorer is not None else {
                choice: int(tokenizer.encode(f" {choice}", add_special_tokens=False)[0]) for choice in CHOICES
            },
            "stored_features": {
                "safetensors": [
                    "raw logits for A/B/C/D",
                    "conditional softmax probabilities over A/B/C/D",
                ],
                "no_rag_jsonl": [
                    "argmax prediction/correctness",
                    "gold probability, gold margin, gold rank",
                    "top-1 probability, top-1 margin, entropy",
                    "prompt token count and prompt hash",
                ],
                "single_document_jsonl": [
                    "all no-RAG features",
                    "gold probability/margin deltas",
                    "four-choice logit/probability deltas",
                    "prediction change and C2C/C2W/W2C/W2W transition",
                    "source/retrieval/rerank metadata and document text hash",
                ],
            },
            "correctness_transitions": dict(total_transitions),
            "preflight_context_audit": str(audit_path.resolve()),
            "run_contract": str((args.output_root / "run_contract.json").resolve()),
            "next_stage": "derive direct-choice behavioral labels or semantic-behavioral subgroup analyses from this immutable cache",
        }
        atomic_write_json(args.output_root / "outcome_manifest.json", manifest)
        logging.info(
            "Direct-choice outcome cache complete: questions=%d pairs=%d transitions=%s output=%s",
            total_questions,
            total_pairs,
            dict(total_transitions),
            args.output_root,
        )
    finally:
        if scorer is not None:
            scorer.close()


if __name__ == "__main__":
    main()

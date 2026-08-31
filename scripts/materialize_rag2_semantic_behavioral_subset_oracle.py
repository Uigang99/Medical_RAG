#!/usr/bin/env python3
"""Find exact gold-margin-optimal subsets inside semantic Top-k candidates.

This is an Oracle materializer, not a deployable filter.  It uses the gold MCQ
answer only to score every subset with the frozen target Llama's exact
direct-choice logits.  Two selections are produced from the same scored
subsets:

* the best subset containing only ``direct_support`` documents;
* the best subset from the configured semantic candidate labels (by default
  ``direct_support`` plus ``supporting_evidence``).

The empty subset is always included, so the Oracle can fall back to the
target model's internal knowledge when every candidate decreases the margin.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_preanswer_text_hidden import (  # noqa: E402
    CHOICES,
    FINAL_ANSWER_PREFILL,
    PREANSWER_PROMPT_VERSION,
    build_preanswer_user_prompt,
)
from medrag.progress import PipelineProgress  # noqa: E402


RUN_VERSION = "rag2_semantic_behavioral_exact_subset_oracle_v1"
SELECTION_POLICIES = (
    "behavioral_best_direct",
    "behavioral_best_semantic_candidates",
)
SEMANTIC_LABELS = {
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
}
DEFAULT_DATASETS = (
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-file", default="candidates_topk_union.jsonl")
    parser.add_argument("--semantic-labels-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--candidate-semantic-labels",
        nargs="+",
        default=["direct_support", "supporting_evidence"],
    )
    parser.add_argument("--questions-per-shard", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=65_536)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-questions-per-dataset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all joins and exact subset counts without loading the target Llama.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    files = []
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        candidate = resolved / name
        if candidate.is_file():
            files.append(path_identity(candidate))
    for candidate in sorted(resolved.glob("*.safetensors")):
        files.append(path_identity(candidate))
    return {"path": str(resolved), "files": files}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def document_stable_id(document: dict[str, Any]) -> str:
    value = document.get("stable_id") or document.get("corpus_id") or document.get("chunk_id")
    if not value:
        value = document.get("db_id") or f"{document.get('source')}:{document.get('local_id')}"
    return str(value)


def normalized_gold(row: dict[str, Any]) -> str:
    value = str(row.get("answer") or "").strip().upper()
    if value not in CHOICES:
        answers = row.get("answers") or []
        value = next((str(item).strip().upper() for item in answers if str(item).strip().upper() in CHOICES), "")
    if value not in CHOICES:
        raise ValueError(f"Invalid gold answer for {row.get('key')}: {value!r}")
    return value


def load_semantic_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_key = str(row.get("sample_key") or "")
        stable_id = str(row.get("doc_stable_id") or "")
        label = str(row.get("semantic_label") or "")
        if not sample_key or not stable_id or label not in SEMANTIC_LABELS:
            raise ValueError(f"Invalid semantic decision identity/label: {row}")
        key = (sample_key, stable_id)
        if key in index:
            raise ValueError(f"Duplicate semantic decision: {key}")
        index[key] = row
    return index


def load_questions(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic = load_semantic_index(args.semantic_labels_path)
    candidate_labels = set(args.candidate_semantic_labels)
    if "direct_support" not in candidate_labels:
        raise ValueError("--candidate-semantic-labels must include direct_support")
    unknown = candidate_labels - SEMANTIC_LABELS
    if unknown:
        raise ValueError(f"Unknown candidate semantic labels: {sorted(unknown)}")
    questions: list[dict[str, Any]] = []
    source_identities: list[dict[str, Any]] = []
    for dataset in args.datasets:
        path = args.candidate_root / dataset / args.split / args.candidate_file
        if not path.is_file():
            raise FileNotFoundError(path)
        source_identities.append(path_identity(path))
        dataset_rows = list(iter_jsonl(path))
        if args.max_questions_per_dataset > 0:
            dataset_rows = dataset_rows[: args.max_questions_per_dataset]
        for row in dataset_rows:
            if str(row.get("dataset")) != dataset:
                raise ValueError(f"Candidate dataset mismatch in {path}: {row.get('dataset')} != {dataset}")
            sample_key = str(row.get("key") or "")
            sample_id = str(row.get("sample_id") or "")
            selected = list((row.get("selected_document_ids_by_top_k") or {}).get(str(args.top_k)) or [])
            if len(selected) != args.top_k or len(set(selected)) != args.top_k:
                raise ValueError(
                    f"Expected {args.top_k} unique exact-condition documents for {sample_key}, got {len(selected)}"
                )
            documents = {document_stable_id(doc): doc for doc in row.get("candidate_documents") or []}
            if len(documents) != len(row.get("candidate_documents") or []):
                raise ValueError(f"Duplicate document identity in candidate union: {sample_key}")
            top_documents = []
            for rank, stable_id in enumerate(selected, start=1):
                document = documents.get(stable_id)
                if document is None:
                    raise KeyError(f"Top-{args.top_k} document absent from union for {sample_key}: {stable_id}")
                decision = semantic.get((sample_key, stable_id))
                if decision is None:
                    raise KeyError(f"Missing semantic label for {(sample_key, stable_id)}")
                text = clean_text(document.get("text"))
                if not text:
                    raise ValueError(f"Empty document body for {(sample_key, stable_id)}")
                top_documents.append(
                    {
                        "doc_rank": rank,
                        "doc_stable_id": stable_id,
                        "source": document.get("source"),
                        "text": text,
                        "semantic_label": decision["semantic_label"],
                    }
                )
            broad = [doc for doc in top_documents if doc["semantic_label"] in candidate_labels]
            direct = [doc for doc in broad if doc["semantic_label"] == "direct_support"]
            options = row.get("options") or {}
            if set(options) != set(CHOICES):
                raise ValueError(f"Expected A/B/C/D options for {sample_key}: {sorted(options)}")
            questions.append(
                {
                    "dataset": dataset,
                    "split": args.split,
                    "row_idx": int(row.get("row_idx", -1)),
                    "sample_id": sample_id,
                    "sample_key": sample_key,
                    "question": clean_text(row.get("question")),
                    "options": {choice: clean_text(options[choice]) for choice in CHOICES},
                    "gold_answer": normalized_gold(row),
                    "top_k_documents": top_documents,
                    "semantic_candidates": broad,
                    "direct_candidate_ids": [doc["doc_stable_id"] for doc in direct],
                    "subset_count": 1 << len(broad),
                }
            )
    expected_semantic_pairs = sum(len(row["top_k_documents"]) for row in questions)
    logging.info(
        "Preflight complete: questions=%s Top-%s docs=%s semantic candidates=%s exact subsets=%s",
        len(questions),
        args.top_k,
        expected_semantic_pairs,
        sum(len(row["semantic_candidates"]) for row in questions),
        sum(row["subset_count"] for row in questions),
    )
    return questions, source_identities


@dataclass(frozen=True)
class ShardSpec:
    dataset: str
    index: int
    questions: list[dict[str, Any]]
    subset_count: int
    root: Path


def shard_plan(args: argparse.Namespace, questions: list[dict[str, Any]]) -> list[ShardSpec]:
    plan: list[ShardSpec] = []
    for dataset in args.datasets:
        rows = [row for row in questions if row["dataset"] == dataset]
        for start in range(0, len(rows), args.questions_per_shard):
            batch = rows[start : start + args.questions_per_shard]
            index = start // args.questions_per_shard
            root = args.output_root / "score_shards" / dataset / args.split / f"shard_{index:05d}"
            plan.append(
                ShardSpec(
                    dataset=dataset,
                    index=index,
                    questions=batch,
                    subset_count=sum(row["subset_count"] for row in batch),
                    root=root,
                )
            )
    return plan


def render_direct_choice_ids(
    tokenizer: Any,
    marker_ids: Sequence[int],
    question: dict[str, Any],
    selected_documents: Sequence[dict[str, Any]],
) -> list[int]:
    context = "\n\n".join(doc["text"] for doc in selected_documents) or None
    sample = SimpleNamespace(
        id=question["sample_id"],
        question=question["question"],
        options=question["options"],
    )
    user_prompt = build_preanswer_user_prompt(sample, context)
    messages = [{"role": "user", "content": user_prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = re.sub(
        r"(?is)(<\|im_start\|>assistant\s*)<think>\s*$",
        r"\1",
        str(rendered),
    )
    values = list(tokenizer.encode(rendered, add_special_tokens=False)) + list(marker_ids)
    if values[-len(marker_ids) :] != list(marker_ids):
        raise RuntimeError("Final-answer marker is not the direct-choice prompt suffix")
    return values


def subset_document_ids(candidates: Sequence[dict[str, Any]], mask: int) -> list[str]:
    return [
        str(document["doc_stable_id"])
        for index, document in enumerate(candidates)
        if mask & (1 << index)
    ]


def choose_best_subset(
    subsets: Sequence[dict[str, Any]],
    eligible_document_ids: set[str],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for row in subsets:
        selected_ids = list(row["selected_document_ids"])
        if any(stable_id not in eligible_document_ids for stable_id in selected_ids):
            continue
        if best is None:
            best = row
            continue
        margin = float(row["gold_margin"])
        best_margin = float(best["gold_margin"])
        if margin > best_margin + tolerance:
            best = row
            continue
        if math.isclose(margin, best_margin, rel_tol=0.0, abs_tol=tolerance):
            candidate_key = (len(selected_ids), tuple(row["selected_document_ranks"]), int(row["mask"]))
            best_key = (
                len(best["selected_document_ids"]),
                tuple(best["selected_document_ranks"]),
                int(best["mask"]),
            )
            if candidate_key < best_key:
                best = row
    if best is None:
        raise RuntimeError("No eligible subset; the empty subset should always be eligible")
    return best


class DirectChoiceScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if args.attn_implementation == "sdpa" and torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.marker_ids = self.tokenizer.encode(FINAL_ANSWER_PREFILL, add_special_tokens=False)
        if not self.marker_ids:
            raise RuntimeError("Final answer marker tokenized to an empty sequence")
        self.choice_token_ids = []
        for choice in CHOICES:
            token_ids = self.tokenizer.encode(" " + choice, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"Choice {choice} is not one leading-space token: {token_ids}")
            self.choice_token_ids.append(int(token_ids[0]))
        logging.info("Loading frozen target Llama for exact subset scoring: %s", args.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path),
            dtype=dtype_map[args.dtype],
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        self.model.requires_grad_(False)
        self.model.eval().to(self.device)
        self.choice_ids = torch.tensor(self.choice_token_ids, dtype=torch.long, device=self.device)
        logging.info(
            "Subset scorer ready: device=%s choice_token_ids=%s prompt=%s",
            self.device,
            dict(zip(CHOICES, self.choice_token_ids)),
            PREANSWER_PROMPT_VERSION,
        )

    def _forward(self, records: Sequence[dict[str, Any]], progress: PipelineProgress) -> None:
        if not records:
            return
        max_length = max(len(row["input_ids"]) for row in records)
        input_ids = torch.full(
            (len(records), max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(records):
            values = row["input_ids"]
            input_ids[row_index, -len(values) :] = torch.tensor(values, dtype=torch.long, device=self.device)
            attention_mask[row_index, -len(values) :] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        try:
            with torch.inference_mode():
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                final_state = outputs.last_hidden_state[:, -1, :]
                logits = self.model.lm_head(final_state).index_select(-1, self.choice_ids).float()
                probabilities = F.softmax(logits, dim=-1)
        except torch.OutOfMemoryError:
            del input_ids, attention_mask, position_ids
            gc.collect()
            torch.cuda.empty_cache()
            if len(records) <= 1:
                raise
            midpoint = len(records) // 2
            logging.warning(
                "Subset-score OOM at batch=%s max_tokens=%s; retrying %s + %s",
                len(records),
                max_length,
                midpoint,
                len(records) - midpoint,
            )
            self._forward(records[:midpoint], progress)
            self._forward(records[midpoint:], progress)
            return
        logits_cpu = logits.cpu()
        probabilities_cpu = probabilities.cpu()
        for index, row in enumerate(records):
            gold_index = CHOICES.index(row["gold_answer"])
            wrong = [value for value in range(len(CHOICES)) if value != gold_index]
            row["choice_logits"] = [float(value) for value in logits_cpu[index].tolist()]
            row["choice_probabilities"] = [float(value) for value in probabilities_cpu[index].tolist()]
            row["gold_margin"] = float(
                logits_cpu[index, gold_index] - logits_cpu[index, wrong].max()
            )
            row["gold_probability"] = float(probabilities_cpu[index, gold_index])
            row["prediction"] = CHOICES[int(torch.argmax(logits_cpu[index]).item())]
            del row["input_ids"]
        progress.update(len(records))
        del outputs, final_state, logits, probabilities, logits_cpu, probabilities_cpu

    def score_records(self, records: list[dict[str, Any]], progress: PipelineProgress) -> None:
        records.sort(key=lambda row: len(row["input_ids"]))
        start = 0
        while start < len(records):
            end = start
            longest = len(records[start]["input_ids"])
            while end < len(records) and end - start < self.args.max_batch_size:
                proposed_longest = max(longest, len(records[end]["input_ids"]))
                if end > start and proposed_longest * (end - start + 1) > self.args.max_batch_tokens:
                    break
                longest = proposed_longest
                end += 1
            if end == start:
                end += 1
            progress.set_detail(f"batch={end-start} padded_tokens={longest*(end-start)}")
            self._forward(records[start:end], progress)
            start = end

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_shard(
    args: argparse.Namespace,
    scorer: DirectChoiceScorer,
    shard: ShardSpec,
    contract_hash: str,
    progress: PipelineProgress,
) -> None:
    records: list[dict[str, Any]] = []
    question_meta: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for question_index, question in enumerate(shard.questions):
        candidates = question["semantic_candidates"]
        subsets = []
        for mask in range(1 << len(candidates)):
            selected_documents = [
                document for index, document in enumerate(candidates) if mask & (1 << index)
            ]
            sequence = render_direct_choice_ids(
                scorer.tokenizer,
                scorer.marker_ids,
                question,
                selected_documents,
            )
            if len(sequence) > args.max_input_tokens:
                raise ValueError(
                    f"Direct-choice subset exceeds {args.max_input_tokens} tokens: "
                    f"sample={question['sample_key']} mask={mask} tokens={len(sequence)}"
                )
            record = {
                "question_index": question_index,
                "mask": mask,
                "selected_document_ids": [doc["doc_stable_id"] for doc in selected_documents],
                "selected_document_ranks": [doc["doc_rank"] for doc in selected_documents],
                "gold_answer": question["gold_answer"],
                "input_tokens": len(sequence),
                "input_ids": sequence,
            }
            subsets.append(record)
            records.append(record)
        question_meta.append((question, subsets))
    scorer.score_records(records, progress)
    output_rows = []
    candidate_labels = set(args.candidate_semantic_labels)
    for question, subsets in question_meta:
        direct_ids = set(question["direct_candidate_ids"])
        broad_ids = {
            doc["doc_stable_id"]
            for doc in question["semantic_candidates"]
            if doc["semantic_label"] in candidate_labels
        }
        best_direct = choose_best_subset(subsets, direct_ids)
        best_broad = choose_best_subset(subsets, broad_ids)
        by_mask = sorted(subsets, key=lambda row: int(row["mask"]))
        output_rows.append(
            {
                "run_version": RUN_VERSION,
                "dataset": question["dataset"],
                "split": question["split"],
                "row_idx": question["row_idx"],
                "sample_id": question["sample_id"],
                "sample_key": question["sample_key"],
                "gold_answer": question["gold_answer"],
                "top_k": args.top_k,
                "top_k_documents": question["top_k_documents"],
                "candidate_semantic_labels": list(args.candidate_semantic_labels),
                "semantic_candidates": question["semantic_candidates"],
                "subsets": [
                    {key: value for key, value in row.items() if key not in {"question_index", "gold_answer"}}
                    for row in by_mask
                ],
                "optima": {
                    "behavioral_best_direct": {
                        "selected_document_ids": best_direct["selected_document_ids"],
                        "selected_document_ranks": best_direct["selected_document_ranks"],
                        "subset_size": len(best_direct["selected_document_ids"]),
                        "gold_margin": best_direct["gold_margin"],
                        "gold_probability": best_direct["gold_probability"],
                        "prediction": best_direct["prediction"],
                    },
                    "behavioral_best_semantic_candidates": {
                        "selected_document_ids": best_broad["selected_document_ids"],
                        "selected_document_ranks": best_broad["selected_document_ranks"],
                        "subset_size": len(best_broad["selected_document_ids"]),
                        "gold_margin": best_broad["gold_margin"],
                        "gold_probability": best_broad["gold_probability"],
                        "prediction": best_broad["prediction"],
                    },
                },
            }
        )
    atomic_jsonl(shard.root / "questions.jsonl", output_rows)
    atomic_json(
        shard.root / "COMPLETE.json",
        {
            "status": "complete",
            "run_version": RUN_VERSION,
            "contract_fingerprint": contract_hash,
            "dataset": shard.dataset,
            "shard_index": shard.index,
            "questions": len(output_rows),
            "subsets": shard.subset_count,
            "created_at": utc_now(),
        },
    )


def shard_is_complete(shard: ShardSpec, contract_hash: str) -> bool:
    complete_path = shard.root / "COMPLETE.json"
    rows_path = shard.root / "questions.jsonl"
    if not complete_path.is_file() or not rows_path.is_file():
        return False
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("contract_fingerprint") != contract_hash:
        raise RuntimeError(
            f"Completed shard contract mismatch: {complete_path}; use a new output root"
        )
    return (
        complete.get("status") == "complete"
        and int(complete.get("questions", -1)) == len(shard.questions)
        and int(complete.get("subsets", -1)) == shard.subset_count
    )


def question_result_rows(plan: Sequence[ShardSpec]) -> Iterator[dict[str, Any]]:
    for shard in plan:
        yield from iter_jsonl(shard.root / "questions.jsonl")


def materialize_outputs(
    args: argparse.Namespace,
    plan: Sequence[ShardSpec],
    progress: PipelineProgress,
    contract_hash: str,
) -> dict[str, Any]:
    label_paths = {
        policy: args.output_root / f"{policy}_labels.jsonl"
        for policy in SELECTION_POLICIES
    }
    temporary_paths = {policy: path.with_name(path.name + ".partial") for policy, path in label_paths.items()}
    optima_path = args.output_root / "question_optima.jsonl"
    optima_temporary = optima_path.with_name(optima_path.name + ".partial")
    for path in [*temporary_paths.values(), optima_temporary]:
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {
        policy: temporary_paths[policy].open("w", encoding="utf-8")
        for policy in SELECTION_POLICIES
    }
    optima_handle = optima_temporary.open("w", encoding="utf-8")
    counts = {policy: Counter() for policy in SELECTION_POLICIES}
    question_count = 0
    label_row_counts = Counter()
    broad_beats_direct = 0
    try:
        for row in question_result_rows(plan):
            question_count += 1
            optima_handle.write(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "dataset",
                            "split",
                            "row_idx",
                            "sample_id",
                            "sample_key",
                            "gold_answer",
                            "top_k",
                            "candidate_semantic_labels",
                            "semantic_candidates",
                            "optima",
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            direct_margin = float(row["optima"]["behavioral_best_direct"]["gold_margin"])
            broad_margin = float(
                row["optima"]["behavioral_best_semantic_candidates"]["gold_margin"]
            )
            broad_beats_direct += int(broad_margin > direct_margin + 1e-8)
            for policy in SELECTION_POLICIES:
                optimum = row["optima"][policy]
                selected_ids = set(optimum["selected_document_ids"])
                counts[policy]["questions"] += 1
                counts[policy]["selected_documents"] += len(selected_ids)
                counts[policy]["empty_subsets"] += int(not selected_ids)
                counts[policy]["positive_margin"] += int(float(optimum["gold_margin"]) > 0.0)
                if policy == "behavioral_best_semantic_candidates":
                    selected_semantics = {
                        doc["semantic_label"]
                        for doc in row["semantic_candidates"]
                        if doc["doc_stable_id"] in selected_ids
                    }
                    counts[policy]["questions_with_supporting_selected"] += int(
                        "supporting_evidence" in selected_semantics
                    )
                for document in row["top_k_documents"]:
                    selected = document["doc_stable_id"] in selected_ids
                    label_row = {
                        "schema_version": RUN_VERSION,
                        "sample_key": row["sample_key"],
                        "dataset": row["dataset"],
                        "sample_id": row["sample_id"],
                        "row_idx": row["row_idx"],
                        "doc_rank": document["doc_rank"],
                        "doc_stable_id": document["doc_stable_id"],
                        "source": document["source"],
                        "semantic_label": document["semantic_label"],
                        "candidate_semantic_labels": row["candidate_semantic_labels"],
                        "selection_policy": policy,
                        "selected": selected,
                        "pseudo_label": "Helpful" if selected else "Not Helpful",
                        "selected_subset_document_ids": optimum["selected_document_ids"],
                        "selected_subset_size": optimum["subset_size"],
                        "selected_subset_gold_margin": optimum["gold_margin"],
                        "selected_subset_gold_probability": optimum["gold_probability"],
                        "selected_subset_prediction": optimum["prediction"],
                        "selector_prompt_version": PREANSWER_PROMPT_VERSION,
                        "gold_used_for_selection": True,
                    }
                    handles[policy].write(json.dumps(label_row, ensure_ascii=False) + "\n")
                    label_row_counts[policy] += 1
            progress.update(1)
    finally:
        optima_handle.close()
        for handle in handles.values():
            handle.close()
    for policy, temporary in temporary_paths.items():
        os.replace(temporary, label_paths[policy])
    os.replace(optima_temporary, optima_path)
    summary = {
        "run_version": RUN_VERSION,
        "status": "complete",
        "created_at": utc_now(),
        "contract_fingerprint": contract_hash,
        "questions": question_count,
        "top_k": args.top_k,
        "candidate_semantic_labels": list(args.candidate_semantic_labels),
        "subset_scores": sum(shard.subset_count for shard in plan),
        "broad_margin_strictly_exceeds_direct_questions": broad_beats_direct,
        "broad_margin_strictly_exceeds_direct_rate": broad_beats_direct / max(1, question_count),
        "policies": {},
        "label_paths": {policy: str(path.resolve()) for policy, path in label_paths.items()},
        "question_optima_path": str(optima_path.resolve()),
    }
    for policy in SELECTION_POLICIES:
        counter = counts[policy]
        summary["policies"][policy] = {
            "questions": counter["questions"],
            "label_rows": label_row_counts[policy],
            "mean_selected_documents": counter["selected_documents"] / max(1, counter["questions"]),
            "empty_subset_rate": counter["empty_subsets"] / max(1, counter["questions"]),
            "positive_margin_rate": counter["positive_margin"] / max(1, counter["questions"]),
            "questions_with_supporting_selected_rate": (
                counter["questions_with_supporting_selected"] / max(1, counter["questions"])
                if policy == "behavioral_best_semantic_candidates"
                else None
            ),
        }
    atomic_json(args.output_root / "summary.json", summary)
    lines = [
        "# Exact semantic-candidate behavioral subset Oracle materialization",
        "",
        f"- Questions: {question_count:,}",
        f"- Exact subset scores: {summary['subset_scores']:,}",
        f"- Candidate semantic labels: {', '.join(args.candidate_semantic_labels)}",
        f"- Broad optimum margin > Direct optimum margin: {broad_beats_direct:,}/{question_count:,} "
        f"({summary['broad_margin_strictly_exceeds_direct_rate'] * 100:.2f}%)",
        "",
        "| Policy | Avg selected docs | Empty subset | Positive gold margin | Supporting selected |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in SELECTION_POLICIES:
        values = summary["policies"][policy]
        supporting = values["questions_with_supporting_selected_rate"]
        lines.append(
            f"| {policy} | {values['mean_selected_documents']:.3f} | "
            f"{values['empty_subset_rate'] * 100:.2f}% | "
            f"{values['positive_margin_rate'] * 100:.2f}% | "
            f"{'-' if supporting is None else f'{supporting * 100:.2f}%'} |"
        )
    atomic_text(args.output_root / "summary_table_pretty.txt", "\n".join(lines) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.top_k <= 0 or args.top_k > 8:
        raise ValueError("Exact subset materialization is intentionally limited to 1 <= Top-k <= 8")
    if args.questions_per_shard <= 0 or args.max_batch_size <= 0 or args.max_batch_tokens <= 0:
        raise ValueError("Shard and batch sizes must be positive")
    for path in (args.semantic_labels_path, args.model_name_or_path, args.candidate_root):
        if not path.exists():
            raise FileNotFoundError(path)
    questions, candidate_sources = load_questions(args)
    if not questions:
        raise RuntimeError("No questions selected")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_complete",
                    "questions": len(questions),
                    "top_k_documents": sum(len(row["top_k_documents"]) for row in questions),
                    "semantic_candidates": sum(len(row["semantic_candidates"]) for row in questions),
                    "exact_subsets": sum(row["subset_count"] for row in questions),
                    "candidate_semantic_labels": list(args.candidate_semantic_labels),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    contract = {
        "run_version": RUN_VERSION,
        "candidate_sources": candidate_sources,
        "semantic_labels": path_identity(args.semantic_labels_path),
        "model": model_identity(args.model_name_or_path),
        "datasets": list(args.datasets),
        "split": args.split,
        "top_k": args.top_k,
        "candidate_semantic_labels": list(args.candidate_semantic_labels),
        "questions_per_shard": args.questions_per_shard,
        "max_batch_size": args.max_batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_input_tokens": args.max_input_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "attention_backend_policy": (
            "disable_cudnn_sdpa_keep_flash_efficient_math_v1"
            if args.attn_implementation == "sdpa"
            else "eager"
        ),
        "selector_prompt_version": PREANSWER_PROMPT_VERSION,
        "choice_order": list(CHOICES),
        "score": "gold logit minus strongest wrong-option logit",
        "empty_subset_included": True,
        "tie_break": "higher margin, then fewer documents, then earlier rerank ranks",
        "max_questions_per_dataset": args.max_questions_per_dataset,
    }
    contract_hash = fingerprint(contract)
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Output-root contract mismatch; use a new output root")
    else:
        atomic_json(contract_path, {**contract, "contract_fingerprint": contract_hash})
    complete_manifest = args.output_root / "manifest.json"
    if args.resume and complete_manifest.is_file():
        manifest = json.loads(complete_manifest.read_text(encoding="utf-8"))
        if manifest.get("contract_fingerprint") != contract_hash:
            raise RuntimeError("Completed manifest contract mismatch; use a new output root")
        if manifest.get("status") == "complete":
            print((args.output_root / "summary_table_pretty.txt").read_text(encoding="utf-8"))
            logging.info("Completed exact subset Oracle is unchanged: %s", args.output_root)
            return
    plan = shard_plan(args, questions)
    complete = [shard for shard in plan if args.resume and shard_is_complete(shard, contract_hash)]
    completed_subsets = sum(shard.subset_count for shard in complete)
    total_subsets = sum(shard.subset_count for shard in plan)
    progress = PipelineProgress(
        overall_total=total_subsets + len(questions),
        overall_initial=completed_subsets,
        desc="SemanticBehaviorSubsetOracle",
    )
    scorer: DirectChoiceScorer | None = None
    try:
        progress.set_stage(
            "1/2 exact direct-choice scoring for every semantic-candidate subset",
            total=total_subsets,
            initial=completed_subsets,
        )
        remaining = [shard for shard in plan if shard not in complete]
        if remaining:
            scorer = DirectChoiceScorer(args)
            for shard in remaining:
                progress.set_detail(
                    f"dataset={shard.dataset} shard={shard.index} questions={len(shard.questions)}"
                )
                score_shard(args, scorer, shard, contract_hash, progress)
            scorer.close()
            scorer = None
        progress.set_stage(
            "2/2 materialize Best-Direct and Best-Semantic-Candidate memberships",
            total=len(questions),
        )
        summary = materialize_outputs(args, plan, progress, contract_hash)
        atomic_json(
            complete_manifest,
            {
                "status": "complete",
                "run_version": RUN_VERSION,
                "contract_fingerprint": contract_hash,
                "created_at": utc_now(),
                "questions": len(questions),
                "subsets": total_subsets,
                "summary": str((args.output_root / "summary.json").resolve()),
            },
        )
    finally:
        if scorer is not None:
            scorer.close()
        progress.close()
    print((args.output_root / "summary_table_pretty.txt").read_text(encoding="utf-8"))
    logging.info("Exact subset Oracle materialization complete: %s", args.output_root)


if __name__ == "__main__":
    main()

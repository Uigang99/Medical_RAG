#!/usr/bin/env python3
"""Extract pre-answer hidden states and gold-answer directions for a RAG2 pilot.

The model sees a fixed MCQ prompt ending in the assistant prefill
``Final answer:``.  The hidden state at the final colon is therefore the state
immediately before the single answer token.  For every selected question we
compute the no-document state h0 and the loss-decreasing gold-answer direction

    c = - d[-log p(gold | h0)] / d h0.

For every question-document pair we independently compute hD.  The script
uses constrained A/B/C/D next-token decoding, stores human-readable metadata
as JSONL, and stores full vectors in safetensors shards.  Completed shards are
atomic and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    WORKSPACE_ROOT
    / "Medical_RAG/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2"
    / "candidates/quality_selected_source_balanced40_rerank32_v1"
)
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
FORMAT_VERSION = "rag2_preanswer_hidden_gold_direction_pilot_v1"
PROMPT_VERSION = "rag2_fixed_direct_choice_context_v1"
CHOICES = ("A", "B", "C", "D")
FINAL_ANSWER_PREFILL = "Final answer:"
SAMPLE_ID_MARKER = b'"sample_id": "'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample question-document pairs and extract no-document/document "
            "pre-answer states plus the no-document gold-answer gradient direction."
        )
    )
    parser.add_argument(
        "--medmcqa-candidates-path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "medmcqa/train/candidates_top32.jsonl",
    )
    parser.add_argument(
        "--medqa-candidates-path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "medqa/train/candidates_top32.jsonl",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=10_000)
    parser.add_argument("--docs-per-question", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["16", "24", "28", "final"],
        help="Llama hidden-state indices after the named block, plus optional final normalized state.",
    )
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--questions-per-shard", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="eager"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
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


def atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def extract_sample_id(line: bytes, row_index: int) -> str:
    start = line.find(SAMPLE_ID_MARKER)
    if start < 0:
        return f"row:{row_index}"
    start += len(SAMPLE_ID_MARKER)
    end = line.find(b'"', start)
    if end < 0:
        return f"row:{row_index}"
    return line[start:end].decode("utf-8", errors="replace")


def stable_selection_key(dataset: str, sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{dataset}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def candidate_offsets(path: Path, dataset: str, needed: int, seed: int) -> list[tuple[int, int, str]]:
    """Return a deterministic, approximately uniform hash sample without parsing a huge JSONL."""
    reserve = max(64, math.ceil(needed * 0.10))
    keep = needed + reserve
    heap: list[tuple[int, int, int, str]] = []
    total_bytes = path.stat().st_size
    with path.open("rb") as handle, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"sample-plan:{dataset}",
        dynamic_ncols=True,
    ) as progress:
        row_index = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            progress.update(len(line))
            sample_id = extract_sample_id(line, row_index)
            key = stable_selection_key(dataset, sample_id, seed)
            item = (-key, -row_index, offset, sample_id)
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            row_index += 1
    selected = [(-neg_key, -neg_row, offset, sample_id) for neg_key, neg_row, offset, sample_id in heap]
    selected.sort(key=lambda value: (value[0], value[1]))
    return [(row_index, offset, sample_id) for _, row_index, offset, sample_id in selected]


def normalize_gold(row: dict[str, Any]) -> str | None:
    answer = row.get("answer")
    if answer is None and isinstance(row.get("answers"), list) and row["answers"]:
        answer = row["answers"][0]
    if isinstance(answer, int) and 0 <= answer < 4:
        return CHOICES[answer]
    answer = str(answer or "").strip().upper()
    return answer if answer in CHOICES else None


def normalize_options(row: dict[str, Any]) -> dict[str, str] | None:
    raw = row.get("options")
    if isinstance(raw, dict):
        options = {choice: str(raw.get(choice) or "").strip() for choice in CHOICES}
    elif isinstance(raw, list) and len(raw) >= 4:
        options = {choice: str(raw[index] or "").strip() for index, choice in enumerate(CHOICES)}
    else:
        return None
    return options if all(options.values()) else None


def document_sort_key(document: dict[str, Any], fallback: int) -> tuple[int, int]:
    rank = document.get("rerank_rank")
    try:
        rank_value = int(rank)
    except (TypeError, ValueError):
        rank_value = fallback
    return rank_value, fallback


def normalize_document(document: dict[str, Any], rank: int) -> dict[str, Any] | None:
    text = str(document.get("text") or "").strip()
    if not text:
        return None
    stable_id = str(document.get("stable_id") or document.get("id") or f"rank:{rank}")
    return {
        "stable_id": stable_id,
        "source": str(document.get("source") or "unknown"),
        "title": str(document.get("title") or ""),
        "text": text,
        "rerank_rank": int(document.get("rerank_rank") or rank),
        "rerank_score": document.get("rerank_score", document.get("score")),
    }


def load_rows_at_offsets(
    path: Path,
    dataset: str,
    offsets: Sequence[tuple[int, int, str]],
    question_limit: int,
    docs_per_question: int,
    pair_limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining_pairs = pair_limit
    with path.open("rb") as handle:
        for row_index, offset, expected_sample_id in offsets:
            if len(selected) >= question_limit or remaining_pairs <= 0:
                break
            handle.seek(offset)
            row = json.loads(handle.readline())
            sample_id = str(row.get("sample_id") or f"{dataset}:row:{row_index}")
            if expected_sample_id != sample_id and not expected_sample_id.startswith("row:"):
                raise RuntimeError(f"Offset contract mismatch: expected={expected_sample_id} actual={sample_id}")
            gold = normalize_gold(row)
            options = normalize_options(row)
            question = str(row.get("question") or "").strip()
            raw_documents = row.get("candidate_documents")
            if not gold or not options or not question or not isinstance(raw_documents, list):
                continue
            ordered = sorted(
                enumerate(raw_documents, start=1),
                key=lambda item: document_sort_key(item[1], item[0]),
            )
            documents: list[dict[str, Any]] = []
            for fallback_rank, raw_document in ordered:
                document = normalize_document(raw_document, fallback_rank)
                if document is not None:
                    documents.append(document)
                if len(documents) >= min(docs_per_question, remaining_pairs):
                    break
            if not documents:
                continue
            selected.append(
                {
                    "dataset": dataset,
                    "source_split": str(row.get("split") or "train"),
                    "source_row_index": row_index,
                    "sample_id": sample_id,
                    "question": question,
                    "options": options,
                    "gold_answer": gold,
                    "documents": documents,
                }
            )
            remaining_pairs -= len(documents)
    if remaining_pairs != 0:
        raise RuntimeError(
            f"Could not form requested {pair_limit} pairs for {dataset}; missing={remaining_pairs}"
        )
    return selected


def split_pair_quotas(total: int, datasets: Sequence[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(datasets))
    return {dataset: base + (index < remainder) for index, dataset in enumerate(datasets)}


def build_selection_plan(args: argparse.Namespace, plan_path: Path) -> list[dict[str, Any]]:
    paths = {
        "medmcqa": args.medmcqa_candidates_path,
        "medqa": args.medqa_candidates_path,
    }
    quotas = split_pair_quotas(args.max_pairs, list(paths))
    plan: list[dict[str, Any]] = []
    for dataset, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        pair_quota = quotas[dataset]
        question_limit = math.ceil(pair_quota / args.docs_per_question)
        offsets = candidate_offsets(path, dataset, question_limit, args.selection_seed)
        rows = load_rows_at_offsets(
            path,
            dataset,
            offsets,
            question_limit,
            args.docs_per_question,
            pair_quota,
        )
        plan.extend(rows)
    for question_index, row in enumerate(plan):
        row["question_index"] = question_index
        for document_index, document in enumerate(row["documents"]):
            document["document_index"] = document_index
            # Keep the canonical pair contract used by the candidate builder
            # and Codex semantic annotations.  The stable ID already embeds
            # its corpus/source namespace, so adding ``source`` again would
            # make otherwise identical pairs impossible to join by pair_id.
            document["pair_id"] = f"{row['sample_id']}::{document['rerank_rank']}::{document['stable_id']}"
    atomic_write_jsonl(plan_path, plan)
    return plan


def load_or_create_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan_path = args.output_dir / "selection_plan.jsonl"
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError(f"Selection plan exists and --no-resume was requested: {plan_path}")
        plan = list(iter_jsonl(plan_path))
    else:
        plan = build_selection_plan(args, plan_path)
    pair_count = sum(len(row["documents"]) for row in plan)
    if pair_count != args.max_pairs:
        raise RuntimeError(f"Selection plan has {pair_count} pairs, expected {args.max_pairs}")
    return plan


def parse_layer_specs(specs: Sequence[str], num_hidden_layers: int) -> tuple[list[str], list[int]]:
    names: list[str] = []
    indices: list[int] = []
    for raw in specs:
        value = str(raw).strip().lower()
        if value == "final":
            name, index = "final", -1
        else:
            try:
                block = int(value)
            except ValueError as error:
                raise ValueError(f"Invalid layer: {raw}") from error
            if block < 1 or block >= num_hidden_layers:
                raise ValueError(
                    f"Intermediate layer must be in [1,{num_hidden_layers - 1}], got {block}. "
                    "Use 'final' for the final normalized representation."
                )
            name, index = f"layer_{block}", block
        if name in names:
            raise ValueError(f"Duplicate layer: {raw}")
        names.append(name)
        indices.append(index)
    return names, indices


def build_user_prompt(question: str, options: dict[str, str], document_text: str | None) -> str:
    options_text = "\n".join(f"{choice}. {options[choice]}" for choice in CHOICES)
    context = document_text.strip() if document_text and document_text.strip() else "None"
    return (
        "Select the single best answer to the following medical multiple-choice question.\n"
        "Output exactly one uppercase option letter from the given options.\n"
        "Do not provide an explanation or any additional text.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Context:\n{context}"
    )


def render_input_ids(tokenizer: Any, user_prompt: str, marker_ids: Sequence[int]) -> list[int]:
    # Recent Transformers returns a BatchEncoding rather than a bare list when
    # ``tokenize=True``.  Rendering first keeps this contract stable across
    # tokenizer backend versions and preserves the special chat tokens.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return list(ids) + list(marker_ids)


def pad_token_sequences(
    sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, -length:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, -length:] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


@dataclass
class QuestionFeatures:
    h0: torch.Tensor
    c_unit: torch.Tensor
    c_norm: torch.Tensor
    choice_logits: torch.Tensor
    choice_probs: torch.Tensor
    gold_choice_logprob: torch.Tensor


@dataclass
class DocumentFeatures:
    hD: torch.Tensor
    choice_logits: torch.Tensor
    choice_probs: torch.Tensor


class FeatureExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(args.device)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map[args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.marker_ids = self.tokenizer.encode(FINAL_ANSWER_PREFILL, add_special_tokens=False)
        if not self.marker_ids:
            raise RuntimeError("Final-answer prefill tokenized to an empty sequence")
        self.choice_token_ids: list[int] = []
        for choice in CHOICES:
            ids = self.tokenizer.encode(" " + choice, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Choice {choice!r} is not one token: {ids}")
            self.choice_token_ids.append(ids[0])
        logging.info("Loading model on %s: %s", self.device, args.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path),
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.layer_names, self.layer_indices = parse_layer_specs(
            args.layers, int(self.model.config.num_hidden_layers)
        )
        self.hidden_size = int(self.model.config.hidden_size)
        self.choice_ids_tensor = torch.tensor(
            self.choice_token_ids, dtype=torch.long, device=self.device
        )
        logging.info(
            "Model ready: layers=%s hidden=%d choice_token_ids=%s marker_ids=%s",
            self.layer_names,
            self.hidden_size,
            dict(zip(CHOICES, self.choice_token_ids)),
            self.marker_ids,
        )

    def encode_questions(
        self, rows: Sequence[dict[str, Any]], document_texts: Sequence[str | None]
    ) -> tuple[list[list[int]], list[str]]:
        sequences: list[list[int]] = []
        prompts: list[str] = []
        for row, document_text in zip(rows, document_texts):
            user_prompt = build_user_prompt(row["question"], row["options"], document_text)
            ids = render_input_ids(self.tokenizer, user_prompt, self.marker_ids)
            if len(ids) > self.args.max_input_tokens:
                raise ValueError(
                    f"Prompt exceeds --max-input-tokens for {row['sample_id']}: "
                    f"{len(ids)} > {self.args.max_input_tokens}"
                )
            if ids[-len(self.marker_ids) :] != self.marker_ids:
                raise RuntimeError("Final-answer marker is not the input suffix")
            sequences.append(ids)
            prompts.append(user_prompt)
        return sequences, prompts

    def _base_forward(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        return self.model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def no_document_features(
        self, sequences: Sequence[Sequence[int]], gold_indices: Sequence[int]
    ) -> QuestionFeatures:
        input_ids, attention_mask, position_ids = pad_token_sequences(
            sequences, self.tokenizer.pad_token_id, self.device
        )
        inputs_embeds = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        with torch.enable_grad():
            outputs = self._base_forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            selected_states = [outputs.hidden_states[index] for index in self.layer_indices]
            selected_last = torch.stack([state[:, -1, :] for state in selected_states], dim=1)
            final_last = outputs.last_hidden_state[:, -1, :]
            logits = self.model.lm_head(final_last)
            choice_logits = logits.index_select(dim=-1, index=self.choice_ids_tensor).float()
            gold = torch.tensor(gold_indices, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(choice_logits, gold, reduction="sum")
            gradients = torch.autograd.grad(loss, selected_states, retain_graph=False)
            c_raw = torch.stack([-gradient[:, -1, :].float() for gradient in gradients], dim=1)
        c_norm = torch.linalg.vector_norm(c_raw, dim=-1)
        c_unit = c_raw / c_norm.clamp_min(1e-12).unsqueeze(-1)
        choice_probs = F.softmax(choice_logits, dim=-1)
        gold_choice_logprob = F.log_softmax(choice_logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
        return QuestionFeatures(
            h0=selected_last.detach().float().cpu(),
            c_unit=c_unit.detach().cpu(),
            c_norm=c_norm.detach().cpu(),
            choice_logits=choice_logits.detach().cpu(),
            choice_probs=choice_probs.detach().cpu(),
            gold_choice_logprob=gold_choice_logprob.detach().cpu(),
        )

    def document_features(self, sequences: Sequence[Sequence[int]]) -> DocumentFeatures:
        input_ids, attention_mask, position_ids = pad_token_sequences(
            sequences, self.tokenizer.pad_token_id, self.device
        )
        with torch.inference_mode():
            outputs = self._base_forward(
                input_ids=input_ids,
                inputs_embeds=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            selected_last = torch.stack(
                [outputs.hidden_states[index][:, -1, :] for index in self.layer_indices], dim=1
            )
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
            choice_logits = logits.index_select(dim=-1, index=self.choice_ids_tensor).float()
            choice_probs = F.softmax(choice_logits, dim=-1)
        return DocumentFeatures(
            hD=selected_last.float().cpu(),
            choice_logits=choice_logits.cpu(),
            choice_probs=choice_probs.cpu(),
        )


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def choice_index(answer: str) -> int:
    return CHOICES.index(answer)


def predicted_choice(probabilities: torch.Tensor) -> str:
    return CHOICES[int(torch.argmax(probabilities).item())]


def float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.tolist()]


def shard_paths(output_dir: Path, shard_index: int) -> dict[str, Path]:
    root = output_dir / "shards" / f"shard_{shard_index:05d}"
    return {
        "root": root,
        "questions_meta": root / "questions.jsonl",
        "pairs_meta": root / "pairs.jsonl",
        "questions_tensor": root / "question_features.safetensors",
        "pairs_tensor": root / "pair_features.safetensors",
        "complete": root / "COMPLETE.json",
    }


def complete_shard_valid(paths: dict[str, Path], expected_questions: int, expected_pairs: int) -> bool:
    if not paths["complete"].is_file():
        return False
    required = ("questions_meta", "pairs_meta", "questions_tensor", "pairs_tensor")
    if any(not paths[name].is_file() for name in required):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("format_version") == FORMAT_VERSION
        and marker.get("question_count") == expected_questions
        and marker.get("pair_count") == expected_pairs
    )


def process_shard(
    args: argparse.Namespace,
    extractor: FeatureExtractor,
    shard_index: int,
    rows: Sequence[dict[str, Any]],
) -> None:
    paths = shard_paths(args.output_dir, shard_index)
    paths["root"].mkdir(parents=True, exist_ok=True)
    question_h0: list[torch.Tensor] = []
    question_c_unit: list[torch.Tensor] = []
    question_c_norm: list[torch.Tensor] = []
    question_logits: list[torch.Tensor] = []
    question_probs: list[torch.Tensor] = []
    question_metadata: list[dict[str, Any]] = []
    question_runtime: list[dict[str, Any]] = []

    for batch_rows in chunks(rows, args.question_batch_size):
        sequences, prompts = extractor.encode_questions(batch_rows, [None] * len(batch_rows))
        gold_indices = [choice_index(row["gold_answer"]) for row in batch_rows]
        features = extractor.no_document_features(sequences, gold_indices)
        for local_index, (row, sequence, prompt) in enumerate(zip(batch_rows, sequences, prompts)):
            h0 = features.h0[local_index]
            c_unit = features.c_unit[local_index]
            c_norm = features.c_norm[local_index]
            logits = features.choice_logits[local_index]
            probabilities = features.choice_probs[local_index]
            prediction = predicted_choice(probabilities)
            shard_question_index = len(question_h0)
            question_h0.append(h0.half())
            question_c_unit.append(c_unit.half())
            question_c_norm.append(c_norm)
            question_logits.append(logits)
            question_probs.append(probabilities)
            question_runtime.append(
                {
                    "h0": h0,
                    "c_unit": c_unit,
                    "gold_choice_logprob": features.gold_choice_logprob[local_index],
                    "prediction": prediction,
                    "shard_question_index": shard_question_index,
                }
            )
            question_metadata.append(
                {
                    "format_version": FORMAT_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "shard_index": shard_index,
                    "shard_question_index": shard_question_index,
                    "global_question_index": row["question_index"],
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "source_split": row["source_split"],
                    "source_row_index": row["source_row_index"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold_answer": row["gold_answer"],
                    "no_document_answer": prediction,
                    "no_document_correct": prediction == row["gold_answer"],
                    "no_document_visible_output": f"{FINAL_ANSWER_PREFILL} {prediction}",
                    "no_document_choice_logits": float_list(logits),
                    "no_document_choice_probabilities": float_list(probabilities),
                    "no_document_gold_choice_logprob": float(
                        features.gold_choice_logprob[local_index].item()
                    ),
                    "input_token_count": len(sequence),
                    "user_prompt_sha256": sha256_text(prompt),
                    "feature_file": "question_features.safetensors",
                    "feature_tensor_row": shard_question_index,
                }
            )

    pair_hD: list[torch.Tensor] = []
    pair_logits: list[torch.Tensor] = []
    pair_probs: list[torch.Tensor] = []
    pair_utility: list[torch.Tensor] = []
    pair_delta_norm: list[torch.Tensor] = []
    pair_cosine: list[torch.Tensor] = []
    pair_gold_logprob_delta: list[torch.Tensor] = []
    pair_question_rows: list[int] = []
    pair_metadata: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for local_question_index, row in enumerate(rows):
        for document in row["documents"]:
            pending.append((local_question_index, row, document))

    for batch_items in chunks(pending, args.document_batch_size):
        batch_rows = [item[1] for item in batch_items]
        document_texts = [item[2]["text"] for item in batch_items]
        sequences, prompts = extractor.encode_questions(batch_rows, document_texts)
        features = extractor.document_features(sequences)
        for local_index, ((question_local, row, document), sequence, prompt) in enumerate(
            zip(batch_items, sequences, prompts)
        ):
            runtime = question_runtime[question_local]
            hD = features.hD[local_index]
            logits = features.choice_logits[local_index]
            probabilities = features.choice_probs[local_index]
            delta = hD - runtime["h0"]
            delta_norm = torch.linalg.vector_norm(delta, dim=-1)
            utility = torch.sum(delta * runtime["c_unit"], dim=-1)
            cosine = utility / delta_norm.clamp_min(1e-12)
            gold_index = choice_index(row["gold_answer"])
            gold_logprob = F.log_softmax(logits, dim=-1)[gold_index]
            gold_delta = gold_logprob - runtime["gold_choice_logprob"]
            prediction = predicted_choice(probabilities)
            no_doc_prediction = runtime["prediction"]
            transition = (
                ("C" if no_doc_prediction == row["gold_answer"] else "W")
                + "->"
                + ("C" if prediction == row["gold_answer"] else "W")
            )
            pair_index = len(pair_hD)
            pair_hD.append(hD.half())
            pair_logits.append(logits)
            pair_probs.append(probabilities)
            pair_utility.append(utility)
            pair_delta_norm.append(delta_norm)
            pair_cosine.append(cosine)
            pair_gold_logprob_delta.append(gold_delta)
            pair_question_rows.append(runtime["shard_question_index"])
            pair_metadata.append(
                {
                    "format_version": FORMAT_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "shard_index": shard_index,
                    "shard_pair_index": pair_index,
                    "shard_question_index": runtime["shard_question_index"],
                    "global_question_index": row["question_index"],
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "pair_id": document["pair_id"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold_answer": row["gold_answer"],
                    "document": document,
                    "no_document_answer": no_doc_prediction,
                    "with_document_answer": prediction,
                    "with_document_correct": prediction == row["gold_answer"],
                    "answer_transition": transition,
                    "with_document_visible_output": f"{FINAL_ANSWER_PREFILL} {prediction}",
                    "with_document_choice_logits": float_list(logits),
                    "with_document_choice_probabilities": float_list(probabilities),
                    "gold_choice_logprob_delta": float(gold_delta.item()),
                    "utility_projection_by_layer": {
                        name: float(utility[index].item())
                        for index, name in enumerate(extractor.layer_names)
                    },
                    "delta_h_norm_by_layer": {
                        name: float(delta_norm[index].item())
                        for index, name in enumerate(extractor.layer_names)
                    },
                    "delta_c_cosine_by_layer": {
                        name: float(cosine[index].item())
                        for index, name in enumerate(extractor.layer_names)
                    },
                    "input_token_count": len(sequence),
                    "user_prompt_sha256": sha256_text(prompt),
                    "feature_file": "pair_features.safetensors",
                    "feature_tensor_row": pair_index,
                    "h0_and_c_question_tensor_row": runtime["shard_question_index"],
                }
            )

    question_tensors = {
        "h0": torch.stack(question_h0),
        "c_unit": torch.stack(question_c_unit),
        "c_norm": torch.stack(question_c_norm).float(),
        "choice_logits": torch.stack(question_logits).float(),
        "choice_probabilities": torch.stack(question_probs).float(),
    }
    pair_tensors = {
        "hD": torch.stack(pair_hD),
        "choice_logits": torch.stack(pair_logits).float(),
        "choice_probabilities": torch.stack(pair_probs).float(),
        "utility_projection": torch.stack(pair_utility).float(),
        "delta_h_norm": torch.stack(pair_delta_norm).float(),
        "delta_c_cosine": torch.stack(pair_cosine).float(),
        "gold_choice_logprob_delta": torch.stack(pair_gold_logprob_delta).float(),
        "question_tensor_row": torch.tensor(pair_question_rows, dtype=torch.int64),
    }
    tensor_metadata = {
        "format_version": FORMAT_VERSION,
        "layer_order": canonical_json(extractor.layer_names),
        "choice_order": canonical_json(CHOICES),
        "vector_note": "delta_h is reconstructed as hD - h0; c_raw is reconstructed as c_unit * c_norm",
    }
    atomic_save_safetensors(paths["questions_tensor"], question_tensors, tensor_metadata)
    atomic_save_safetensors(paths["pairs_tensor"], pair_tensors, tensor_metadata)
    atomic_write_jsonl(paths["questions_meta"], question_metadata)
    atomic_write_jsonl(paths["pairs_meta"], pair_metadata)
    atomic_write_json(
        paths["complete"],
        {
            "format_version": FORMAT_VERSION,
            "completed_at": utc_now(),
            "shard_index": shard_index,
            "question_count": len(question_metadata),
            "pair_count": len(pair_metadata),
        },
    )


def expected_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "candidate_paths": {
            "medmcqa": str(args.medmcqa_candidates_path.resolve()),
            "medqa": str(args.medqa_candidates_path.resolve()),
        },
        "max_pairs": args.max_pairs,
        "docs_per_question": args.docs_per_question,
        "selection_seed": args.selection_seed,
        "requested_layers": [str(value) for value in args.layers],
        "max_input_tokens": args.max_input_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "questions_per_shard": args.questions_per_shard,
        "prompt_contract": {
            "assistant_prefill": FINAL_ANSWER_PREFILL,
            "no_document_context": "None",
            "document_context": "raw document text only",
            "decoding": "one constrained argmax token over A/B/C/D",
            "direction": "c = negative gradient of choice-normalized gold NLL with respect to h0",
        },
    }


def validate_or_write_manifest(args: argparse.Namespace) -> None:
    path = args.output_dir / "run_manifest.json"
    expected = expected_manifest(args)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        actual_contract = {key: actual.get(key) for key in expected}
        if actual_contract != expected:
            raise RuntimeError(
                "Existing output manifest is incompatible with this invocation. "
                "Use a new --output-dir."
            )
        if not args.resume:
            raise FileExistsError(f"Output already exists and --no-resume was requested: {args.output_dir}")
        return
    manifest = dict(expected)
    manifest.update(
        {
            "created_at": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "storage_contract": {
                "question_features": "h0, c_unit, c_norm, A-D logits/probabilities",
                "pair_features": "hD, scalar delta/c metrics, A-D logits/probabilities",
                "reconstruction": "delta_h = hD - h0; c_raw = c_unit * c_norm",
                "vector_dtype": "float16",
                "scalar_dtype": "float32",
            },
        }
    )
    atomic_write_json(path, manifest)


def consolidate_outputs(args: argparse.Namespace, shard_count: int) -> dict[str, Any]:
    question_output = args.output_dir / "questions.jsonl"
    pair_output = args.output_dir / "pairs.jsonl"
    question_temp = question_output.with_name(question_output.name + ".partial")
    pair_temp = pair_output.with_name(pair_output.name + ".partial")
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    projection_sums: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    question_count = 0
    pair_count = 0
    with question_temp.open("w", encoding="utf-8") as question_handle, pair_temp.open(
        "w", encoding="utf-8"
    ) as pair_handle:
        for shard_index in range(shard_count):
            paths = shard_paths(args.output_dir, shard_index)
            for row in iter_jsonl(paths["questions_meta"]):
                question_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                question_count += 1
                dataset = row["dataset"]
                counters[dataset]["questions"] += 1
                counters[dataset]["no_doc_correct"] += int(row["no_document_correct"])
            for row in iter_jsonl(paths["pairs_meta"]):
                pair_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                pair_count += 1
                dataset = row["dataset"]
                counters[dataset]["pairs"] += 1
                counters[dataset]["with_doc_correct"] += int(row["with_document_correct"])
                counters[dataset][f"transition:{row['answer_transition']}"] += 1
                for layer, value in row["utility_projection_by_layer"].items():
                    projection_sums[dataset][layer] += float(value)
        question_handle.flush()
        pair_handle.flush()
        os.fsync(question_handle.fileno())
        os.fsync(pair_handle.fileno())
    os.replace(question_temp, question_output)
    os.replace(pair_temp, pair_output)

    datasets: dict[str, Any] = {}
    for dataset, counter in counters.items():
        questions = counter["questions"]
        pairs = counter["pairs"]
        datasets[dataset] = {
            "questions": questions,
            "pairs": pairs,
            "no_document_correct": counter["no_doc_correct"],
            "no_document_accuracy": counter["no_doc_correct"] / questions if questions else None,
            "with_document_correct": counter["with_doc_correct"],
            "with_document_pair_accuracy": counter["with_doc_correct"] / pairs if pairs else None,
            "transitions": {
                transition: counter[f"transition:{transition}"]
                for transition in ("C->C", "C->W", "W->C", "W->W")
            },
            "mean_utility_projection_by_layer": {
                layer: total / pairs for layer, total in projection_sums[dataset].items()
            },
        }
    summary = {
        "format_version": FORMAT_VERSION,
        "completed_at": utc_now(),
        "questions": question_count,
        "pairs": pair_count,
        "datasets": datasets,
        "question_metadata": str(question_output),
        "pair_metadata": str(pair_output),
        "vector_shards": str(args.output_dir / "shards"),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> None:
    if args.max_pairs < 2:
        raise ValueError("--max-pairs must be at least 2")
    if args.docs_per_question < 1:
        raise ValueError("--docs-per-question must be positive")
    if args.question_batch_size < 1 or args.document_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.questions_per_shard < 1:
        raise ValueError("--questions-per-shard must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_or_write_manifest(args)
    plan = load_or_create_plan(args)
    shards = list(chunks(plan, args.questions_per_shard))
    pending: list[tuple[int, Sequence[dict[str, Any]]]] = []
    completed_pairs = 0
    for shard_index, shard_rows in enumerate(shards):
        expected_pairs = sum(len(row["documents"]) for row in shard_rows)
        paths = shard_paths(args.output_dir, shard_index)
        if args.resume and complete_shard_valid(paths, len(shard_rows), expected_pairs):
            completed_pairs += expected_pairs
        else:
            pending.append((shard_index, shard_rows))
    logging.info(
        "Pilot contract: questions=%d pairs=%d shards=%d completed_pairs=%d pending_shards=%d",
        len(plan),
        args.max_pairs,
        len(shards),
        completed_pairs,
        len(pending),
    )
    extractor: FeatureExtractor | None = None
    if pending:
        extractor = FeatureExtractor(args)
        progress = tqdm(
            total=args.max_pairs,
            initial=completed_pairs,
            unit="pair",
            desc="PreAnswerHidden",
            dynamic_ncols=True,
        )
        for shard_index, shard_rows in pending:
            pair_count = sum(len(row["documents"]) for row in shard_rows)
            process_shard(args, extractor, shard_index, shard_rows)
            progress.update(pair_count)
            progress.set_postfix(shard=shard_index)
        progress.close()
    summary = consolidate_outputs(args, len(shards))
    logging.info("Complete: %s", json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()

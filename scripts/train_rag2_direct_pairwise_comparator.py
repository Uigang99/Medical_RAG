#!/usr/bin/env python3
"""Diagnose behavioral-utility ranking with direct document-pair comparison.

This experiment intentionally isolates the learning formulation while keeping
the Flan-T5-large base checkpoint fixed.  It removes the previous independent
scalar scorer, mean pooling, and NULL-document objective.  For each question,
two evidence documents are shown in one prompt and the full encoder-decoder
predicts whether Candidate A or Candidate B has the larger cached utility.

Both candidate orders are always trained and evaluated.  Loss is first
averaged over the two orders of a semantic pair, then over pairs in a question,
and finally over questions.  A question-level offset therefore cannot solve
the task, and questions with many eligible pairs cannot dominate the loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import socket
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import format_options  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from scripts.train_rag2_margin_regressor import (  # noqa: E402
    PREPARED_VERSION,
    atomic_json,
    decode_json,
    limit_questions,
    load_contract,
    load_splits,
    read_jsonl,
)


TRAINER_VERSION = "rag2_direct_pairwise_comparator_v1"
PAIR_PROMPT_VERSION = "question_options_a0_two_documents_predict_ab_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=base / "gold_margin_regression_v1/prepared"
    )
    parser.add_argument(
        "--model-name-or-path", type=Path, default=WORKSPACE_ROOT / "models/Flan-T5-large"
    )
    parser.add_argument(
        "--no-rag-generation-root",
        type=Path,
        default=base / "train_no_rag_anchored_features_v1/no_rag",
    )
    # Required by the shared prepared-data contract validator.  This direct
    # comparator always uses the deployment-available cached No-RAG answer.
    parser.set_defaults(input_mode="text_no_rag_answer")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/RAG2-DirectPairComparator-FlanT5-large",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-train-epochs", type=int, required=True)
    parser.add_argument("--train-questions-per-batch", type=int, default=2)
    parser.add_argument("--eval-questions-per-batch", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--document-pair-min-utility-gap", type=float, default=0.1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--minimum-document-tokens", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-eval-questions", type=int, default=None)
    parser.add_argument("--evaluate-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--checkpoint-split", choices=("validation", "train"), default="validation"
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--stop-train-question-macro-accuracy",
        type=float,
        default=None,
        help="Optional memorization-test success threshold.",
    )
    parser.add_argument("--trace-shard-cache-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class NoRAGAnswers:
    """Deployment-available No-RAG predictions, indexed with pipeline progress."""

    def __init__(
        self,
        root: Path,
        dataset: str,
        split: str,
        progress: PipelineProgress,
    ) -> None:
        directory = root / dataset / split
        manifest_path = directory / "manifest.json"
        generations_path = directory / "no_rag_generations.jsonl"
        if not manifest_path.is_file() or not generations_path.is_file():
            raise FileNotFoundError(f"Incomplete No-RAG cache: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("dataset")) != dataset or str(manifest.get("split")) != split:
            raise RuntimeError(f"No-RAG cache contract mismatch: {manifest_path}")
        total = int(manifest.get("rows") or 0)
        progress.set_stage("index cached No-RAG answers", total=total)
        answers: dict[str, str] = {}
        with generations_path.open("rb", buffering=64 * 1024 * 1024) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = decode_json(line)
                sample_id = str(row.get("sample_id") or "")
                choice = str(row.get("answer") or "").strip().upper()
                answer_text = " ".join(str(row.get("answer_text") or "").split())
                if not sample_id or not choice:
                    raise RuntimeError(f"Invalid No-RAG answer row: {generations_path}")
                rendered = f"({choice}) {answer_text}" if answer_text else f"({choice})"
                previous = answers.setdefault(sample_id, rendered)
                if previous != rendered:
                    raise RuntimeError(f"Conflicting No-RAG answers for {sample_id}")
                progress.update(1)
        if total and len(answers) != total:
            raise RuntimeError(f"No-RAG index mismatch: expected={total} actual={len(answers)}")
        self.answers = answers

    def answer_for(self, sample_id: str) -> str:
        try:
            return self.answers[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"Missing cached No-RAG answer for {sample_id}") from error


class TraceStore:
    """LRU reader for question, options, and evidence from anchored traces."""

    def __init__(self, trace_root: Path, dataset: str, split: str, capacity: int) -> None:
        self.trace_root = trace_root
        self.dataset = dataset
        self.split = split
        self.capacity = max(1, int(capacity))
        self.cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def _rows(self, shard: str) -> list[dict[str, Any]]:
        rows = self.cache.pop(shard, None)
        if rows is not None:
            self.cache[shard] = rows
            return rows
        path = self.trace_root / "trace_shards" / self.dataset / self.split / shard / "pairs.jsonl"
        rows = read_jsonl(path)
        self.cache[shard] = rows
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return rows

    def trace(self, pointer: dict[str, Any]) -> dict[str, Any]:
        row = self._rows(str(pointer["trace_shard"]))[int(pointer["trace_pair_row"])]
        if str(row.get("pair_id")) != str(pointer["pair_id"]):
            raise RuntimeError(f"Trace pointer mismatch: {pointer['pair_id']}")
        if not str(row.get("document_text_used") or "").strip():
            raise RuntimeError(f"Empty document text: {pointer['pair_id']}")
        return row


class DirectPairDataset(TorchDataset):
    """Question-local pairs with one canonical higher-utility winner."""

    def __init__(
        self,
        flat: Any,
        min_gap: float,
        split_name: str,
        progress: PipelineProgress,
    ) -> None:
        self.flat = flat
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, sample_id in enumerate(flat["sample_id"]):
            grouped[str(sample_id)].append(index)
        self.pairs: list[tuple[int, int, float, str, bool]] = []
        self.question_to_pairs: dict[str, list[int]] = defaultdict(list)
        progress.set_stage(f"construct {split_name} document pairs", total=len(grouped))
        no_pair_questions = 0
        for sample_id, indices in grouped.items():
            start = len(self.pairs)
            for left, right in combinations(indices, 2):
                left_u = float(flat[left]["utility_target"])
                right_u = float(flat[right]["utility_target"])
                gap = abs(left_u - right_u)
                if gap < float(min_gap):
                    continue
                winner, loser = (left, right) if left_u > right_u else (right, left)
                correct = bool(flat[winner]["no_rag_correct_audit_only"])
                if correct != bool(flat[loser]["no_rag_correct_audit_only"]):
                    raise RuntimeError(f"Inconsistent No-RAG audit state in {sample_id}")
                pair_index = len(self.pairs)
                self.pairs.append((winner, loser, gap, sample_id, correct))
                self.question_to_pairs[sample_id].append(pair_index)
            if len(self.pairs) == start:
                no_pair_questions += 1
            progress.update(1)
        self.question_to_pairs = dict(self.question_to_pairs)
        self.question_count = len(self.question_to_pairs)
        self.no_pair_questions = no_pair_questions

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        winner, loser, gap, sample_id, correct = self.pairs[index]
        return {
            "winner": self.flat[winner],
            "loser": self.flat[loser],
            "gap": gap,
            "sample_id": sample_id,
            "no_rag_correct": correct,
        }


class QuestionBatchSampler(Sampler[list[int]]):
    """Sample whole questions so question-macro loss is exact."""

    def __init__(
        self,
        dataset: DirectPairDataset,
        questions_per_batch: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.questions_per_batch = int(questions_per_batch)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.questions = list(dataset.question_to_pairs)

    def __len__(self) -> int:
        return math.ceil(len(self.questions) / self.questions_per_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        questions = list(self.questions)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(questions)
        for start in range(0, len(questions), self.questions_per_batch):
            yield [
                pair_index
                for sample_id in questions[start : start + self.questions_per_batch]
                for pair_index in self.dataset.question_to_pairs[sample_id]
            ]


class SymmetricPairPacker:
    """Tokenize both documents with equal truncation opportunity."""

    def __init__(self, tokenizer: Any, max_tokens: int, minimum_document_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = int(max_tokens)
        self.minimum_document_tokens = int(minimum_document_tokens)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def pack(
        self,
        *,
        question: str,
        options: str,
        no_rag_answer: str,
        document_a: str,
        document_b: str,
    ) -> list[int]:
        prefix = self._encode(
            "Compare which evidence candidate is more useful for answering the question.\n\n"
            "Question: "
        )
        option_marker = self._encode("\n")
        answer_marker = self._encode(
            "\n\nInitial answer generated without retrieved evidence: "
        )
        candidate_a_marker = self._encode("\n\nCandidate A:\n")
        middle = self._encode("\n\nCandidate B:\n")
        ending = self._encode("\n\nAnswer:")
        question_tokens = self._encode(" ".join(str(question).split()))
        option_tokens = self._encode(str(options).strip())
        answer_tokens = self._encode(" ".join(str(no_rag_answer).split()))
        eos = [int(self.tokenizer.eos_token_id)]
        structural = (
            prefix
            + option_marker
            + answer_marker
            + candidate_a_marker
            + middle
            + ending
        )
        context_budget = (
            self.max_tokens
            - len(structural)
            - len(eos)
            - 2 * self.minimum_document_tokens
        )
        if context_budget < 32:
            raise RuntimeError(
                f"Pair prompt has no usable question budget: structural={len(structural)} "
                f"max={self.max_tokens}"
            )
        # Keep the target model's short No-RAG answer, then split the remaining
        # context budget evenly between question and options with redistribution.
        answer_take = min(len(answer_tokens), 32, max(1, context_budget // 5))
        q_o_budget = context_budget - answer_take
        question_take = min(len(question_tokens), q_o_budget // 2)
        option_take = min(len(option_tokens), q_o_budget // 2)
        spare_context = q_o_budget - question_take - option_take
        if spare_context and question_take < len(question_tokens):
            extra = min(spare_context, len(question_tokens) - question_take)
            question_take += extra
            spare_context -= extra
        if spare_context and option_take < len(option_tokens):
            option_take += min(spare_context, len(option_tokens) - option_take)
        header = (
            prefix
            + question_tokens[:question_take]
            + option_marker
            + option_tokens[:option_take]
            + answer_marker
            + answer_tokens[:answer_take]
            + candidate_a_marker
        )
        available = self.max_tokens - len(header) - len(middle) - len(ending) - len(eos)
        if available < 2 * self.minimum_document_tokens:
            raise AssertionError("Context allocation violated the minimum document budget")
        a_tokens = self._encode(" ".join(str(document_a).split()))
        b_tokens = self._encode(" ".join(str(document_b).split()))
        half = available // 2
        a_take = min(len(a_tokens), half)
        b_take = min(len(b_tokens), half)
        spare = available - a_take - b_take
        if spare and a_take < len(a_tokens):
            extra = min(spare, len(a_tokens) - a_take)
            a_take += extra
            spare -= extra
        if spare and b_take < len(b_tokens):
            b_take += min(spare, len(b_tokens) - b_take)
        ids = (
            header
            + a_tokens[:a_take]
            + middle
            + b_tokens[:b_take]
            + ending
            + eos
        )
        if len(ids) > self.max_tokens:
            raise AssertionError(f"Symmetric packing overflow: {len(ids)}>{self.max_tokens}")
        return ids


class DirectPairCollator:
    def __init__(
        self,
        tokenizer: Any,
        store: TraceStore,
        answers: NoRAGAnswers,
        max_tokens: int,
        minimum_document_tokens: int,
        bf16: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.store = store
        self.answers = answers
        self.packer = SymmetricPairPacker(tokenizer, max_tokens, minimum_document_tokens)
        self.bf16 = bool(bf16)

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        orientation_labels: list[int] = []
        sample_ids: list[str] = []
        gaps: list[float] = []
        no_rag_correct: list[bool] = []
        winner_ranks: list[int] = []
        loser_ranks: list[int] = []
        for row in rows:
            winner_trace = self.store.trace(row["winner"])
            loser_trace = self.store.trace(row["loser"])
            if str(winner_trace["question"]) != str(loser_trace["question"]):
                raise RuntimeError(f"Question mismatch inside pair: {row['sample_id']}")
            winner_options = format_options(winner_trace.get("options") or {})
            loser_options = format_options(loser_trace.get("options") or {})
            if winner_options != loser_options:
                raise RuntimeError(f"Option mismatch inside pair: {row['sample_id']}")
            common = {
                "question": str(winner_trace["question"]),
                "options": winner_options,
                "no_rag_answer": self.answers.answer_for(row["sample_id"]),
            }
            winner_doc = str(winner_trace["document_text_used"])
            loser_doc = str(loser_trace["document_text_used"])
            # Canonical order: the higher-utility document is Candidate A.
            input_ids.append(
                self.packer.pack(document_a=winner_doc, document_b=loser_doc, **common)
            )
            orientation_labels.append(0)
            # Swapped order: the same higher-utility document is Candidate B.
            input_ids.append(
                self.packer.pack(document_a=loser_doc, document_b=winner_doc, **common)
            )
            orientation_labels.append(1)
            sample_ids.append(str(row["sample_id"]))
            gaps.append(float(row["gap"]))
            no_rag_correct.append(bool(row["no_rag_correct"]))
            winner_ranks.append(int(row["winner"]["doc_rank"]))
            loser_ranks.append(int(row["loser"]["doc_rank"]))

        max_length = max(len(ids) for ids in input_ids)
        if self.bf16:
            max_length = min(self.packer.max_tokens, int(math.ceil(max_length / 8) * 8))
        padded = []
        masks = []
        for ids in input_ids:
            padding = max_length - len(ids)
            padded.append(ids + [int(self.tokenizer.pad_token_id)] * padding)
            masks.append([1] * len(ids) + [0] * padding)
        sample_to_index = {sample: index for index, sample in enumerate(dict.fromkeys(sample_ids))}
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "orientation_labels": torch.tensor(orientation_labels, dtype=torch.long),
            "question_index": torch.tensor(
                [sample_to_index[sample] for sample in sample_ids], dtype=torch.long
            ),
            "sample_ids": sample_ids,
            "gaps": np.asarray(gaps, dtype=np.float32),
            "no_rag_correct": np.asarray(no_rag_correct, dtype=np.bool_),
            "winner_ranks": winner_ranks,
            "loser_ranks": loser_ranks,
        }


def make_loader(
    dataset: DirectPairDataset,
    collator: DirectPairCollator,
    questions_per_batch: int,
    seed: int,
    shuffle: bool,
) -> tuple[DataLoader, QuestionBatchSampler]:
    sampler = QuestionBatchSampler(dataset, questions_per_batch, seed, shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    return loader, sampler


def choice_logits(
    model: Any,
    batch: dict[str, Any],
    device: torch.device,
    choice_token_ids: Sequence[int],
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    decoder_start = int(model.config.decoder_start_token_id)
    decoder_input_ids = torch.full(
        (input_ids.shape[0], 1), decoder_start, dtype=torch.long, device=device
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    return outputs.logits[:, 0, list(choice_token_ids)]


def question_macro_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    question_index: torch.Tensor,
) -> torch.Tensor:
    orientation_loss = F.cross_entropy(logits, labels, reduction="none")
    if orientation_loss.numel() % 2:
        raise RuntimeError("Every semantic pair must have exactly two orientations")
    pair_loss = orientation_loss.view(-1, 2).mean(dim=1)
    question_losses = [pair_loss[question_index == group].mean() for group in torch.unique(question_index)]
    return torch.stack(question_losses).mean()


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def compute_metrics(records: list[dict[str, Any]], mean_loss: float) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No comparison records were evaluated")
    correct = np.asarray([row["correct"] for row in records], dtype=np.float64)
    p_winner = np.asarray([row["p_winner"] for row in records], dtype=np.float64)
    swap_error = np.asarray([row["swap_error"] for row in records], dtype=np.float64)
    orientation_labels = np.asarray(
        [label for _ in records for label in (1, 0)], dtype=np.int64
    )
    orientation_scores = np.asarray(
        [score for row in records for score in (row["p_a_canonical"], row["p_a_swapped"])],
        dtype=np.float64,
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[row["sample_id"]].append(float(row["correct"]))
    result: dict[str, Any] = {
        "pairs": len(records),
        "questions": len(grouped),
        "loss": float(mean_loss),
        "pair_accuracy": float(correct.mean()),
        "question_macro_accuracy": float(np.mean([np.mean(values) for values in grouped.values()])),
        "orientation_auc": _safe_auc(orientation_labels, orientation_scores),
        "mean_winner_probability": float(p_winner.mean()),
        "swap_consistency_mae": float(swap_error.mean()),
    }
    for state, name in ((True, "no_rag_correct"), (False, "no_rag_wrong")):
        selected = [row for row in records if bool(row["no_rag_correct"]) is state]
        selected_grouped: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            selected_grouped[row["sample_id"]].append(float(row["correct"]))
        result[name] = {
            "pairs": len(selected),
            "questions": len(selected_grouped),
            "pair_accuracy": float(np.mean([row["correct"] for row in selected]))
            if selected
            else float("nan"),
            "question_macro_accuracy": float(
                np.mean([np.mean(values) for values in selected_grouped.values()])
            )
            if selected_grouped
            else float("nan"),
        }
    bands = ((0.1, 0.2), (0.2, 0.5), (0.5, float("inf")))
    result["by_utility_gap"] = {}
    for lower, upper in bands:
        selected = [row for row in records if lower <= float(row["gap"]) < upper]
        key = f"[{lower:.1f},{'inf' if math.isinf(upper) else f'{upper:.1f}'})"
        result["by_utility_gap"][key] = {
            "pairs": len(selected),
            "accuracy": float(np.mean([row["correct"] for row in selected]))
            if selected
            else float("nan"),
        }
    return result


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    choice_token_ids: Sequence[int],
    progress: PipelineProgress,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    progress.set_stage(stage, total=len(loader.dataset))
    records: list[dict[str, Any]] = []
    weighted_loss = 0.0
    pair_count = 0
    for batch in loader:
        logits = choice_logits(model, batch, device, choice_token_ids)
        labels = batch["orientation_labels"].to(device, non_blocking=True)
        question_index = batch["question_index"].to(device, non_blocking=True)
        loss = question_macro_loss(logits, labels, question_index)
        probabilities = logits.float().softmax(dim=-1)[:, 0].detach().cpu().numpy().reshape(-1, 2)
        count = len(batch["sample_ids"])
        weighted_loss += float(loss.item()) * count
        pair_count += count
        for index, sample_id in enumerate(batch["sample_ids"]):
            p_a_canonical = float(probabilities[index, 0])
            p_a_swapped = float(probabilities[index, 1])
            p_winner = 0.5 * (p_a_canonical + (1.0 - p_a_swapped))
            records.append(
                {
                    "sample_id": sample_id,
                    "winner_rank": int(batch["winner_ranks"][index]),
                    "loser_rank": int(batch["loser_ranks"][index]),
                    "gap": float(batch["gaps"][index]),
                    "no_rag_correct": bool(batch["no_rag_correct"][index]),
                    "p_a_canonical": p_a_canonical,
                    "p_a_swapped": p_a_swapped,
                    "p_winner": p_winner,
                    "correct": bool(p_winner > 0.5),
                    "swap_error": abs(p_a_canonical + p_a_swapped - 1.0),
                }
            )
        progress.update(count)
    return compute_metrics(records, weighted_loss / max(1, pair_count)), records


def save_model(model: Any, tokenizer: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    if path.exists():
        shutil.rmtree(path)
    os.replace(temporary, path)


def save_last_checkpoint(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    trainer_state: dict[str, Any],
) -> Path:
    checkpoint = output_dir / "last_checkpoint"
    save_model(model, tokenizer, checkpoint / "model")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer_state": trainer_state,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        checkpoint / "training_state.pt",
    )
    atomic_json(checkpoint / "trainer_state.json", trainer_state)
    return checkpoint


def load_training_state(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    return dict(payload["trainer_state"])


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def extend_progress(progress: PipelineProgress, additional: int) -> None:
    progress.overall_total += max(0, int(additional))
    if progress._pbar is not None:
        progress._pbar.total = progress.overall_total
        progress._pbar.refresh()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_train_epochs < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Epoch and accumulation values must be positive")
    if not 0 < args.document_pair_min_utility_gap < 2:
        raise ValueError("document-pair-min-utility-gap must be in (0,2)")
    if args.max_input_tokens < 128:
        raise ValueError("max-input-tokens is too small for a two-document prompt")
    if args.checkpoint_split == "train" and not args.evaluate_train:
        raise ValueError("checkpoint-split=train requires --evaluate-train")

    set_seed(args.seed)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    manifest = load_contract(args)
    flat_splits = load_splits(args)
    flat_splits["train"] = limit_questions(flat_splits["train"], args.max_train_questions, args.seed)
    flat_splits["validation"] = limit_questions(
        flat_splits["validation"], args.max_eval_questions, args.seed + 1
    )
    flat_splits["test"] = limit_questions(
        flat_splits["test"], args.max_eval_questions, args.seed + 2
    )
    question_counts = {
        name: len(set(str(value) for value in split["sample_id"]))
        for name, split in flat_splits.items()
    }
    no_rag_manifest = json.loads(
        (
            args.no_rag_generation_root
            / args.dataset
            / "train"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    initial_total = int(no_rag_manifest.get("rows") or 0) + sum(question_counts.values())
    progress = PipelineProgress(
        overall_total=initial_total,
        desc=f"DirectPair:{args.dataset}",
        enabled=args.show_progress,
    )
    answers = NoRAGAnswers(args.no_rag_generation_root, args.dataset, "train", progress)
    pair_splits = {
        name: DirectPairDataset(
            flat_splits[name], args.document_pair_min_utility_gap, name, progress
        )
        for name in ("train", "validation", "test")
    }
    pair_summary = {
        name: {
            "questions_with_pairs": split.question_count,
            "questions_without_decisive_pair": split.no_pair_questions,
            "semantic_pairs": len(split),
            "oriented_examples": 2 * len(split),
        }
        for name, split in pair_splits.items()
    }
    logging.info("Direct-pair data ready: %s", json.dumps(pair_summary, ensure_ascii=False))
    if not len(pair_splits["train"]) or not len(pair_splits["validation"]):
        raise RuntimeError("No decisive train or validation pairs")

    run_dir = args.output_dir
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.output_root / args.dataset / args.run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "trainer_version": TRAINER_VERSION,
        "prompt_version": PAIR_PROMPT_VERSION,
        "dataset": args.dataset,
        "model_name_or_path": str(args.model_name_or_path),
        "prepared_manifest": manifest,
        "pair_summary": pair_summary,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "host": socket.gethostname(),
        "created_at": datetime.now().isoformat(),
        "input_contract": {
            "included": ["question", "options", "cached No-RAG predicted answer", "document A", "document B"],
            "forbidden": ["gold answer", "utility values", "No-RAG correctness", "NULL document"],
            "prediction": "A or B has larger behavioral utility",
            "pair_order": "both orientations for every semantic pair",
        },
    }
    atomic_json(run_dir / "manifest.json", run_manifest)
    if args.dry_run:
        progress.close()
        logging.info("Dry run complete: %s", run_dir)
        return

    tokenizer_source = (
        args.resume_from_checkpoint / "model"
        if args.resume_from_checkpoint is not None
        else args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    choice_ids = [
        tokenizer.encode("A", add_special_tokens=False),
        tokenizer.encode("B", add_special_tokens=False),
    ]
    if any(len(value) != 1 for value in choice_ids) or choice_ids[0] == choice_ids[1]:
        raise RuntimeError(f"A/B must be distinct single tokens: {choice_ids}")
    choice_token_ids = [value[0] for value in choice_ids]
    model_source = tokenizer_source
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    logging.info("Loading full Flan-T5 encoder-decoder from %s", model_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_source,
        local_files_only=True,
        dtype=dtype,
    )
    if args.dropout is not None:
        model.config.dropout_rate = float(args.dropout)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = float(args.dropout)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    device = torch.device("cuda:0")
    model.to(device)

    trace_root = Path(manifest["trace_root"])
    store = TraceStore(
        trace_root, args.dataset, "train", args.trace_shard_cache_size
    )
    collator = DirectPairCollator(
        tokenizer,
        store,
        answers,
        args.max_input_tokens,
        args.minimum_document_tokens,
        args.bf16,
    )
    train_loader, train_sampler = make_loader(
        pair_splits["train"], collator, args.train_questions_per_batch, args.seed, True
    )
    train_eval_loader, _ = make_loader(
        pair_splits["train"], collator, args.eval_questions_per_batch, args.seed, False
    )
    validation_loader, _ = make_loader(
        pair_splits["validation"], collator, args.eval_questions_per_batch, args.seed + 1, False
    )
    test_loader, _ = make_loader(
        pair_splits["test"], collator, args.eval_questions_per_batch, args.seed + 2, False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_update_steps = update_steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_update_steps * args.warmup_ratio),
        num_training_steps=total_update_steps,
    )
    state = {
        "completed_epoch": 0,
        "best_epoch": 0,
        "best_score": float("-inf"),
        "patience": 0,
        "global_step": 0,
        "history": [],
    }
    if args.resume_from_checkpoint is not None:
        state = load_training_state(args.resume_from_checkpoint, optimizer, scheduler)
        logging.info("Resumed at completed epoch %s", state["completed_epoch"])

    remaining_epochs = max(0, args.num_train_epochs - int(state["completed_epoch"]))
    per_epoch = len(pair_splits["train"]) + len(pair_splits["validation"])
    if args.evaluate_train:
        per_epoch += len(pair_splits["train"])
    final_work = len(pair_splits["validation"]) + len(pair_splits["test"])
    extend_progress(progress, remaining_epochs * per_epoch + final_work)

    initial_metrics = None
    if int(state["completed_epoch"]) == 0:
        extend_progress(progress, len(pair_splits["validation"]))
        initial_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            choice_token_ids,
            progress,
            "untrained validation baseline",
        )
        atomic_json(run_dir / "initial_validation_metrics.json", initial_metrics)
        logging.info(
            "Untrained validation: pair_acc=%.4f question_macro=%.4f swap_mae=%.4f",
            initial_metrics["pair_accuracy"],
            initial_metrics["question_macro_accuracy"],
            initial_metrics["swap_consistency_mae"],
        )

    stop_requested = False
    for epoch in range(int(state["completed_epoch"]) + 1, args.num_train_epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        progress.set_stage(f"train epoch {epoch}/{args.num_train_epochs}", total=len(pair_splits["train"]))
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_pairs = 0
        for batch_index, batch in enumerate(train_loader, 1):
            logits = choice_logits(model, batch, device, choice_token_ids)
            labels = batch["orientation_labels"].to(device, non_blocking=True)
            question_index = batch["question_index"].to(device, non_blocking=True)
            loss = question_macro_loss(logits, labels, question_index)
            (loss / args.gradient_accumulation_steps).backward()
            count = len(batch["sample_ids"])
            epoch_loss += float(loss.item()) * count
            epoch_pairs += count
            should_step = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["global_step"] += 1
            if batch_index % args.logging_steps == 0:
                progress.set_detail(
                    f"loss={epoch_loss/max(1,epoch_pairs):.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
            progress.update(count)

        train_metrics = None
        if args.evaluate_train:
            train_metrics, _ = evaluate(
                model,
                train_eval_loader,
                device,
                choice_token_ids,
                progress,
                f"train-set memorization epoch {epoch}",
            )
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            choice_token_ids,
            progress,
            f"validation epoch {epoch}",
        )
        selection_metrics = train_metrics if args.checkpoint_split == "train" else validation_metrics
        score = float(selection_metrics["question_macro_accuracy"])
        improved = score > float(state["best_score"]) + args.minimum_improvement
        if improved:
            state["best_score"] = score
            state["best_epoch"] = epoch
            state["patience"] = 0
            extend_progress(progress, 1)
            progress.set_stage(f"save best model epoch {epoch}", total=1)
            save_model(model, tokenizer, run_dir / "best_model")
            progress.update(1)
        else:
            state["patience"] += 1
        epoch_record = {
            "epoch": epoch,
            "train_optimization_loss": epoch_loss / max(1, epoch_pairs),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "selection_split": args.checkpoint_split,
            "selection_score": score,
            "improved": improved,
        }
        state["history"].append(epoch_record)
        state["completed_epoch"] = epoch
        atomic_json(run_dir / "history.json", state["history"])
        extend_progress(progress, 1)
        progress.set_stage(f"save resumable checkpoint epoch {epoch}", total=1)
        save_last_checkpoint(run_dir, model, tokenizer, optimizer, scheduler, state)
        progress.update(1)
        logging.info(
            "Epoch %d: val_pair_acc=%.4f val_qmacro=%.4f correct/wrong=%.4f/%.4f%s",
            epoch,
            validation_metrics["pair_accuracy"],
            validation_metrics["question_macro_accuracy"],
            validation_metrics["no_rag_correct"]["pair_accuracy"],
            validation_metrics["no_rag_wrong"]["pair_accuracy"],
            (
                f" train_qmacro={train_metrics['question_macro_accuracy']:.4f}"
                if train_metrics is not None
                else ""
            ),
        )
        if (
            train_metrics is not None
            and args.stop_train_question_macro_accuracy is not None
            and train_metrics["question_macro_accuracy"]
            >= args.stop_train_question_macro_accuracy
        ):
            logging.info("Memorization success threshold reached; stopping early")
            stop_requested = True
        elif state["patience"] >= args.early_stopping_patience:
            logging.info("Early stopping after %d non-improving epoch(s)", state["patience"])
            stop_requested = True
        if stop_requested:
            remaining = args.num_train_epochs - epoch
            progress.overall_total -= remaining * per_epoch
            if progress._pbar is not None:
                progress._pbar.total = progress.overall_total
                progress._pbar.refresh()
            break

    best_path = run_dir / "best_model"
    if not best_path.is_dir():
        raise RuntimeError("No best model was saved")
    extend_progress(progress, 1)
    progress.set_stage("load best model", total=1)
    del model
    torch.cuda.empty_cache()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        best_path, local_files_only=True, dtype=dtype
    ).to(device)
    model.config.use_cache = False
    progress.update(1)
    final_validation, validation_records = evaluate(
        model,
        validation_loader,
        device,
        choice_token_ids,
        progress,
        "final best-checkpoint validation",
    )
    final_test, test_records = evaluate(
        model,
        test_loader,
        device,
        choice_token_ids,
        progress,
        "final held-out test",
    )
    summary = {
        "trainer_version": TRAINER_VERSION,
        "best_epoch": state["best_epoch"],
        "best_selection_score": state["best_score"],
        "selection_split": args.checkpoint_split,
        "initial_validation": initial_metrics,
        "final_validation": final_validation,
        "final_test": final_test,
        "pair_summary": pair_summary,
        "diagnostic_interpretation": {
            "memorization_success": (
                None
                if args.stop_train_question_macro_accuracy is None
                else bool(
                    any(
                        row.get("train_metrics")
                        and row["train_metrics"]["question_macro_accuracy"]
                        >= args.stop_train_question_macro_accuracy
                        for row in state["history"]
                    )
                )
            ),
            "random_pair_accuracy": 0.5,
            "null_objective_used": False,
            "independent_scalar_score_used": False,
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    write_jsonl(run_dir / "validation_pair_predictions.jsonl", validation_records)
    write_jsonl(run_dir / "test_pair_predictions.jsonl", test_records)
    progress.close()
    logging.info(
        "Direct pair comparison complete: best_epoch=%s test_pair_acc=%.4f "
        "test_qmacro=%.4f output=%s",
        state["best_epoch"],
        final_test["pair_accuracy"],
        final_test["question_macro_accuracy"],
        run_dir,
    )


if __name__ == "__main__":
    main()

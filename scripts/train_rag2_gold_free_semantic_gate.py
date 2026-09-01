#!/usr/bin/env python3
"""Train a gold-free, set-conditioned semantic document gate.

The frozen Llama is never updated.  The controller is trained with two
within-question response preferences:

* full Top-8 context should prefer the frozen valid-only response to No-RAG;
* invalid-only context should prefer the frozen No-RAG response to valid-only.

Gold answers are neither loaded nor used by this trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from medrag.generation.learned_semantic_attention import (  # noqa: E402
    document_bias_to_token_bias,
    freeze_module_for_controller_training,
    semantic_ordering_hinge_loss,
)
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.generation.semantic_context_gate import SemanticContextGate  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import build_anchored_user_prompt, render_chat_prompt  # noqa: E402
from medrag.training.rag2_semantic_attention_data import RAG2SemanticAttentionDataset  # noqa: E402
from train_rag2_semantic_attention_controller import (  # noqa: E402
    list_feature_shards,
    load_feature_shard,
    model_bundle_identity,
)


RUN_VERSION = "rag2_gold_free_set_semantic_gate_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("overfit", "pilot"), required=True)
    parser.add_argument("--train-questions", type=int, default=256)
    parser.add_argument("--val-questions", type=int, default=128)
    parser.add_argument("--test-questions", type=int, default=128)
    parser.add_argument("--overfit-questions", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prior-strength", type=float, default=0.25)
    parser.add_argument("--max-suppression-factor", type=float, default=20.0)
    parser.add_argument("--semantic-layer-start", type=int, default=16)
    parser.add_argument("--preference-beta", type=float, default=5.0)
    parser.add_argument("--ordering-loss-weight", type=float, default=0.1)
    parser.add_argument("--ordering-margin", type=float, default=0.1)
    parser.add_argument("--anchor-loss-weight", type=float, default=1e-3)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_teachers(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        rows: list[dict[str, Any]] = []
        for path in sorted((root / "teacher_shards" / split).glob("shard_*/questions.jsonl")):
            rows.extend(iter_jsonl(path))
        if not rows or len({row["sample_id"] for row in rows}) != len(rows):
            raise RuntimeError(f"Invalid or duplicate teacher rows for {split}")
        result[split] = rows
    return result


def load_features(
    feature_dir: Path,
    manifest: dict[str, Any],
    wanted: dict[str, set[str]],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    hidden = int(manifest["feature_hidden_size"])
    output: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for split in ("train", "val", "test"):
        split_rows: dict[str, dict[str, torch.Tensor]] = {}
        paths = list_feature_shards(feature_dir, split, manifest)
        for path in paths:
            payload = load_feature_shard(
                path, dataset=str(manifest["dataset"]), split=split,
                fingerprint=str(manifest["contract_fingerprint"]), hidden_size=hidden,
            )
            for index, sample_id in enumerate(payload["sample_ids"]):
                sample_id = str(sample_id)
                if sample_id not in wanted[split]:
                    continue
                split_rows[sample_id] = {
                    "features": payload["semantic_features"][index].float(),
                    "margins": payload["semantic_margins"][index].float(),
                    "targets": payload["semantic_targets"][index].long(),
                    "masks": payload["semantic_masks"][index].bool(),
                }
        missing = wanted[split] - set(split_rows)
        if missing:
            raise RuntimeError(f"Prepared features miss {len(missing)} {split} teachers")
        output[split] = split_rows
    return output


def load_questions(index_path: Path, wanted: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        dataset = RAG2SemanticAttentionDataset(index_path, split)
        for index in range(len(dataset)):
            question = dataset[index]
            if question.sample_id in wanted[split]:
                output[question.sample_id] = question
        dataset.close()
    missing = set.union(*wanted.values()) - set(output)
    if missing:
        raise RuntimeError(f"Semantic index misses {len(missing)} selected questions")
    return output


def prompt_row(question: Any) -> dict[str, Any]:
    # No answer field is required for prompt construction.
    return {"question": question.question, "options": dict(question.options)}


def document_spans(user_prompt: str, documents: list[tuple[int, str]]) -> list[tuple[int, int, int]]:
    marker = "Documents:\n"
    cursor = user_prompt.find(marker)
    if cursor < 0:
        if documents:
            raise RuntimeError("Documents marker is missing")
        return []
    cursor += len(marker)
    spans: list[tuple[int, int, int]] = []
    for original_index, text in documents:
        start = user_prompt.find(text, cursor)
        if start < 0:
            raise RuntimeError("Unable to align document text")
        end = start + len(text)
        spans.append((original_index, start, end))
        cursor = end
    return spans


def encode_condition(
    tokenizer: Any,
    question: Any,
    documents: list[tuple[int, str]],
    response: str,
    max_tokens: int,
) -> dict[str, torch.Tensor | int]:
    evidence = "\n\n".join(text for _, text in documents)
    user_prompt = build_anchored_user_prompt(prompt_row(question), evidence or None)
    chat_prompt = render_chat_prompt(tokenizer, user_prompt)
    full_text = chat_prompt + str(response)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    response_start = len(tokenizer(chat_prompt, add_special_tokens=False)["input_ids"])
    response_tokens = int(full_ids.numel()) - response_start
    if response_tokens <= 1 or int(full_ids.numel()) > max_tokens:
        raise RuntimeError(
            f"Invalid encoded length for {question.sample_id}: total={full_ids.numel()} "
            f"response={response_tokens} max={max_tokens}"
        )
    # Omit the last response token.  The final R model positions then predict
    # exactly all R response tokens, including the first assistant token.
    model_ids = full_ids[:-1]
    token_documents = torch.full((model_ids.numel(),), -1, dtype=torch.long)
    user_start = full_text.find(user_prompt)
    if user_start < 0:
        raise RuntimeError("Rendered chat prompt does not contain user prompt")
    for original_index, start, end in document_spans(user_prompt, documents):
        absolute_start, absolute_end = user_start + start, user_start + end
        indices = [
            index for index, (left, right) in enumerate(offsets[: model_ids.numel()])
            if right > absolute_start and left < absolute_end
        ]
        if not indices:
            raise RuntimeError("Document span has no token overlap")
        token_documents[indices] = original_index
    query_mask = torch.zeros(model_ids.numel(), dtype=torch.float32)
    query_mask[max(0, response_start - 1) :] = 1.0
    targets = full_ids[response_start:]
    return {
        "input_ids": model_ids,
        "token_document_ids": token_documents,
        "query_mask": query_mask,
        "targets": targets,
        "response_tokens": response_tokens,
    }


def mean_response_logp(
    model: Any,
    encoded: dict[str, torch.Tensor | int],
    document_bias: torch.Tensor,
    layer_start: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids = encoded["input_ids"].to(device).unsqueeze(0)  # type: ignore[union-attr]
    token_documents = encoded["token_document_ids"].to(device).unsqueeze(0)  # type: ignore[union-attr]
    query_mask = encoded["query_mask"].to(device).unsqueeze(0)  # type: ignore[union-attr]
    targets = encoded["targets"].to(device)  # type: ignore[union-attr]
    response_tokens = int(encoded["response_tokens"])
    token_bias = document_bias_to_token_bias(document_bias, token_documents)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        logits_to_keep=response_tokens,
        semantic_token_bias=token_bias,
        semantic_query_mask=query_mask,
        semantic_layer_start=layer_start,
    )
    logits = outputs.logits[0].float()
    if logits.shape[0] != targets.numel():
        raise RuntimeError(f"Response-logit alignment mismatch: {logits.shape[0]} != {targets.numel()}")
    token_logp = F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return token_logp.mean()


def prepare_examples(
    teachers: dict[str, list[dict[str, Any]]],
    questions: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, rows in teachers.items():
        examples: list[dict[str, Any]] = []
        for row in rows:
            question = questions[row["sample_id"]]
            full_documents = [
                (index, " ".join(document.text.split()))
                for index, document in enumerate(question.documents)
            ]
            invalid_documents = [
                item for item in full_documents
                if question.documents[item[0]].semantic_support_target == 0
            ]
            valid_response = str(row["valid_only_response"])
            no_rag_response = str(row["no_rag_response"])
            examples.append({
                "sample_id": row["sample_id"],
                "full_valid": encode_condition(tokenizer, question, full_documents, valid_response, max_tokens),
                "full_norag": encode_condition(tokenizer, question, full_documents, no_rag_response, max_tokens),
                "invalid_norag": encode_condition(tokenizer, question, invalid_documents, no_rag_response, max_tokens),
                "invalid_valid": encode_condition(tokenizer, question, invalid_documents, valid_response, max_tokens),
            })
        output[split] = examples
    return output


def preference_loss(positive: torch.Tensor, negative: torch.Tensor, beta: float) -> torch.Tensor:
    return -F.logsigmoid(float(beta) * (positive - negative))


def evaluate(
    split: str,
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, torch.Tensor]],
    controller: SemanticContextGate,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
    progress: PipelineProgress,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controller.eval()
    state: dict[str, Any] = defaultdict(float)
    details: list[dict[str, Any]] = []
    progress.set_stage(stage, total=len(rows))
    for index, example in enumerate(rows):
        sample_id = example["sample_id"]
        values = features[sample_id]
        document_mask = values["masks"].to(device).unsqueeze(0)
        with torch.no_grad():
            output = controller(
                values["features"].to(device).unsqueeze(0),
                values["margins"].to(device).unsqueeze(0),
                document_mask,
            )
            learned_bias = output.document_bias
            zero_bias = torch.zeros_like(learned_bias)
            policy_results: dict[str, dict[str, float]] = {}
            for policy, bias in (("zero", zero_bias), ("learned", learned_bias)):
                full_pos = mean_response_logp(model, example["full_valid"], bias, args.semantic_layer_start, device)
                full_neg = mean_response_logp(model, example["full_norag"], bias, args.semantic_layer_start, device)
                invalid_pos = mean_response_logp(model, example["invalid_norag"], bias, args.semantic_layer_start, device)
                invalid_neg = mean_response_logp(model, example["invalid_valid"], bias, args.semantic_layer_start, device)
                full_margin = float((full_pos - full_neg).item())
                invalid_margin = float((invalid_pos - invalid_neg).item())
                policy_results[policy] = {"full_margin": full_margin, "invalid_margin": invalid_margin}
                state[f"{policy}_full_correct"] += float(full_margin > 0)
                state[f"{policy}_invalid_correct"] += float(invalid_margin > 0)
                state[f"{policy}_full_margin"] += full_margin
                state[f"{policy}_invalid_margin"] += invalid_margin
            targets = values["targets"].to(device)
            support = targets == 1
            nonsupport = targets == 0
            gates = output.keep_gate[0]
            valid_gate = float(gates[support].mean().item())
            invalid_gate = float(gates[nonsupport].mean().item())
            state["valid_gate"] += valid_gate
            state["invalid_gate"] += invalid_gate
            details.append({
                "sample_id": sample_id,
                "zero": policy_results["zero"],
                "learned": policy_results["learned"],
                "valid_gate": valid_gate,
                "invalid_gate": invalid_gate,
                "document_gates": [float(value) for value in gates.cpu().tolist()],
            })
        progress.update(1)
        progress.set_detail(f"split={split} item={index + 1}/{len(rows)}")
    count = len(rows)
    metrics = {"questions": count}
    for policy in ("zero", "learned"):
        full = state[f"{policy}_full_correct"] / count
        invalid = state[f"{policy}_invalid_correct"] / count
        metrics[policy] = {
            "full_prefers_valid_response_accuracy": full,
            "invalid_prefers_norag_response_accuracy": invalid,
            "macro_preference_accuracy": 0.5 * (full + invalid),
            "mean_full_preference_margin": state[f"{policy}_full_margin"] / count,
            "mean_invalid_preference_margin": state[f"{policy}_invalid_margin"] / count,
        }
    metrics["mean_valid_gate"] = state["valid_gate"] / count
    metrics["mean_invalid_gate"] = state["invalid_gate"] / count
    metrics["valid_minus_invalid_gate"] = metrics["mean_valid_gate"] - metrics["mean_invalid_gate"]
    metrics["macro_delta_vs_zero"] = (
        metrics["learned"]["macro_preference_accuracy"]
        - metrics["zero"]["macro_preference_accuracy"]
    )
    return metrics, details


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.epochs <= 0 or args.gradient_accumulation <= 0 or args.preference_beta <= 0:
        raise ValueError("Epoch, accumulation, and preference beta must be positive")
    for path in (args.feature_dir, args.index_path, args.teacher_dir, args.llm_model):
        if not path.exists():
            raise FileNotFoundError(path)
    feature_manifest = json.loads((args.feature_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    teacher_manifest = json.loads((args.teacher_dir / "teacher_manifest.json").read_text(encoding="utf-8"))
    if feature_manifest.get("dataset") != args.dataset or teacher_manifest.get("dataset") != args.dataset:
        raise RuntimeError("Dataset contract mismatch")
    if teacher_manifest.get("feature_contract") != feature_manifest.get("contract_fingerprint"):
        raise RuntimeError("Teacher/prepared-feature contract mismatch")
    if feature_manifest.get("llm_model_bundle") != model_bundle_identity(args.llm_model):
        raise RuntimeError("Prepared features use a different target Llama")
    teachers = load_teachers(args.teacher_dir)
    limits = {"train": args.train_questions, "val": args.val_questions, "test": args.test_questions}
    if args.mode == "overfit":
        limits = {"train": args.overfit_questions, "val": args.overfit_questions, "test": args.overfit_questions}
        source = teachers["train"][: args.overfit_questions]
        teachers = {"train": source, "val": source, "test": source}
    else:
        teachers = {split: rows[: limits[split]] for split, rows in teachers.items()}
    if any(len(teachers[split]) != limits[split] for split in limits):
        raise RuntimeError(f"Teacher count is smaller than requested: {limits}")
    wanted = {split: {row["sample_id"] for row in rows} for split, rows in teachers.items()}
    # Overfit reuses train IDs under all logical phases; feature lookup must do the same.
    feature_wanted = wanted if args.mode == "pilot" else {"train": wanted["train"], "val": set(), "test": set()}
    features_by_split = load_features(args.feature_dir, feature_manifest, feature_wanted)
    if args.mode == "overfit":
        train_features = features_by_split["train"]
        features_by_split = {"train": train_features, "val": train_features, "test": train_features}
        question_wanted = {"train": wanted["train"], "val": set(), "test": set()}
    else:
        question_wanted = wanted
    questions = load_questions(args.index_path, question_wanted)

    run_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "mode": args.mode,
        "gold_used_for_training": False,
        "target": "within-question response preference, not answer correctness",
        "feature_contract": feature_manifest["contract_fingerprint"],
        "teacher_contract": teacher_manifest["contract_fingerprint"],
        "llm_model": str(args.llm_model.resolve()),
        "split_questions": limits,
        "epochs": args.epochs,
        "patience": args.patience,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "architecture": {
            "hidden_size": args.hidden_size,
            "set_layers": args.set_layers,
            "set_heads": args.set_heads,
            "dropout": args.dropout,
            "prior_strength": args.prior_strength,
            "max_suppression_factor": args.max_suppression_factor,
        },
        "loss": {
            "preference_beta": args.preference_beta,
            "ordering_weight": args.ordering_loss_weight,
            "ordering_margin": args.ordering_margin,
            "anchor_weight": args.anchor_loss_weight,
        },
        "semantic_layer_start": args.semantic_layer_start,
        "seed": args.seed,
        "selection_rule": "all labels determinate and both valid/invalid present",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "experiment_manifest.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise RuntimeError("Training resume contract mismatch; use a new output directory")
    else:
        atomic_json(contract_path, run_contract)
    logging.info("Gold-free gate plan: mode=%s splits=%s output=%s", args.mode, limits, args.output_dir)
    if args.plan_only:
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logging.info("Preflight: tokenizing four response conditions per selected question")
    encoded_examples = prepare_examples(teachers, questions, tokenizer, args.max_input_tokens)
    max_length = max(
        int(value["input_ids"].numel())
        for rows in encoded_examples.values() for row in rows
        for key, value in row.items() if key != "sample_id"
    )
    logging.info("Preflight complete: maximum model input tokens=%d/%d", max_length, args.max_input_tokens)

    attention_name = register_semantic_attention()
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=attention_name,
    ).to(device)
    freeze_module_for_controller_training(model)
    controller = SemanticContextGate(
        input_dim=int(feature_manifest["feature_hidden_size"]),
        hidden_dim=args.hidden_size,
        heads=args.set_heads,
        layers=args.set_layers,
        dropout=args.dropout,
        prior_strength=args.prior_strength,
        max_suppression_factor=args.max_suppression_factor,
    ).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "latest_checkpoint.pt"
    start_epoch, best_metric, bad_epochs, history = 0, float("-inf"), 0, []
    if checkpoint_path.is_file() and args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["run_contract"] != run_contract:
            raise RuntimeError("Checkpoint contract mismatch")
        controller.load_state_dict(checkpoint["controller"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint["history"])
        logging.info("Resuming at epoch %d", start_epoch + 1)

    train_count, val_count, test_count = (len(encoded_examples[s]) for s in ("train", "val", "test"))
    total = args.epochs * (train_count + val_count) + test_count
    progress = PipelineProgress(total, start_epoch * (train_count + val_count), desc=f"GoldFreeGate:{args.dataset}:{args.mode}")
    try:
        for epoch in range(start_epoch, args.epochs):
            controller.train()
            order = list(range(train_count))
            random.Random(args.seed + epoch).shuffle(order)
            progress.set_stage(f"1/3 train epoch {epoch + 1}/{args.epochs}", total=train_count)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for position, index in enumerate(order, start=1):
                example = encoded_examples["train"][index]
                values = features_by_split["train"][example["sample_id"]]
                features = values["features"].to(device).unsqueeze(0)
                margins = values["margins"].to(device).unsqueeze(0)
                masks = values["masks"].to(device).unsqueeze(0)
                targets = values["targets"].to(device).unsqueeze(0)
                # Backpropagate the two context preferences separately.  This
                # keeps at most two frozen-Llama response graphs resident and
                # avoids retaining four long rationale graphs at once.
                full_output = controller(features, margins, masks)
                full_pos = mean_response_logp(model, example["full_valid"], full_output.document_bias, args.semantic_layer_start, device)
                full_neg = mean_response_logp(model, example["full_norag"], full_output.document_bias, args.semantic_layer_start, device)
                full_loss = 0.5 * preference_loss(full_pos, full_neg, args.preference_beta)
                (full_loss / args.gradient_accumulation).backward()

                invalid_output = controller(features, margins, masks)
                invalid_pos = mean_response_logp(model, example["invalid_norag"], invalid_output.document_bias, args.semantic_layer_start, device)
                invalid_neg = mean_response_logp(model, example["invalid_valid"], invalid_output.document_bias, args.semantic_layer_start, device)
                invalid_loss = 0.5 * preference_loss(invalid_pos, invalid_neg, args.preference_beta)
                (invalid_loss / args.gradient_accumulation).backward()

                regularized_output = controller(features, margins, masks)
                ordering = semantic_ordering_hinge_loss(
                    regularized_output.document_bias, targets, masks, margin=args.ordering_margin,
                )
                anchor = regularized_output.residual.square()[masks].mean()
                regularization = (
                    args.ordering_loss_weight * ordering
                    + args.anchor_loss_weight * anchor
                )
                (regularization / args.gradient_accumulation).backward()
                loss = full_loss + invalid_loss + regularization
                if position % args.gradient_accumulation == 0 or position == train_count:
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += float(loss.detach().item())
                progress.update(1)
                progress.set_detail(
                    f"item={position}/{train_count} loss={float(loss.detach().item()):.3f}"
                )
            validation, _ = evaluate(
                "val", encoded_examples["val"], features_by_split["val"], controller,
                model, args, device, progress, f"2/3 validation epoch {epoch + 1}/{args.epochs}",
            )
            metric = float(validation["learned"]["macro_preference_accuracy"])
            row = {"epoch": epoch + 1, "train_loss": loss_sum / train_count, "validation": validation}
            history.append(row)
            atomic_json(args.output_dir / "history.json", history)
            if metric > best_metric:
                best_metric, bad_epochs = metric, 0
                atomic_torch(args.output_dir / "best_controller.pt", {
                    "run_version": RUN_VERSION, "run_contract": run_contract,
                    "epoch": epoch + 1, "controller": controller.state_dict(),
                    "validation": validation,
                })
            else:
                bad_epochs += 1
            atomic_torch(checkpoint_path, {
                "run_version": RUN_VERSION, "run_contract": run_contract,
                "epoch": epoch + 1, "controller": controller.state_dict(),
                "optimizer": optimizer.state_dict(), "best_metric": best_metric,
                "bad_epochs": bad_epochs, "history": history,
            })
            logging.info(
                "Epoch %d: val_macro=%.4f zero=%.4f delta=%+.4f gate_gap=%+.4f",
                epoch + 1, metric, validation["zero"]["macro_preference_accuracy"],
                validation["macro_delta_vs_zero"], validation["valid_minus_invalid_gate"],
            )
            if args.patience and bad_epochs >= args.patience:
                logging.info("Early stopping after epoch %d", epoch + 1)
                break

        best = torch.load(args.output_dir / "best_controller.pt", map_location="cpu", weights_only=False)
        controller.load_state_dict(best["controller"])
        test, details = evaluate(
            "test", encoded_examples["test"], features_by_split["test"], controller,
            model, args, device, progress, "3/3 final test with validation-selected checkpoint",
        )
        if args.mode == "overfit":
            passed = (
                test["learned"]["macro_preference_accuracy"] >= 0.80
                and test["valid_minus_invalid_gate"] >= 0.10
            )
            criterion = "same-16-question macro preference accuracy >=0.80 and gate gap >=0.10"
        else:
            passed = (
                test["macro_delta_vs_zero"] >= 0.10
                and test["learned"]["full_prefers_valid_response_accuracy"] >= 0.60
                and test["learned"]["invalid_prefers_norag_response_accuracy"] >= 0.60
                and test["valid_minus_invalid_gate"] >= 0.10
            )
            criterion = "test macro +0.10 over no-gate, both preferences >=0.60, gate gap >=0.10"
        summary = {
            "run_version": RUN_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "gold_used_for_training": False,
            "best_epoch": int(best["epoch"]),
            "best_validation": best["validation"],
            "test": test,
            "success_criterion": criterion,
            "passed": passed,
            "scope_limit": "fixed Top-8 MedQA pilot; gate is access strength, not causal attribution",
        }
        atomic_json(args.output_dir / "summary.json", summary)
        write_jsonl(args.output_dir / "test_predictions.jsonl", details)
        atomic_torch(args.output_dir / "final_controller.pt", {
            "run_version": RUN_VERSION, "run_contract": run_contract,
            "best_epoch": int(best["epoch"]), "controller": controller.state_dict(),
            "test": test,
        })
        logging.info("Training complete: passed=%s test=%s output=%s", passed, json.dumps(test), args.output_dir)
    except BaseException:
        logging.exception(
            "Stopped: last durable checkpoint=%s; rerunning the same command safely resumes at the last epoch",
            checkpoint_path,
        )
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

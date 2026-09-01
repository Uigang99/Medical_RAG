#!/usr/bin/env python3
"""Evaluate free rationale+answer generation for zero and learned semantic gates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rag2_semantic_attention_mvp import (  # noqa: E402
    build_semantic_tensors,
    greedy_generate_with_semantic_attention,
)
from medrag.generation.learned_semantic_attention import freeze_module_for_controller_training  # noqa: E402
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.generation.semantic_context_gate import SemanticContextGate  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    END_REASONING_MARKER,
    FINAL_ANSWER_PREFIX,
    RATIONALE_HEADER,
    build_anchored_user_prompt,
    canonical_response,
    normalize_rationale,
    render_chat_prompt,
)
from train_rag2_gold_free_semantic_gate import (  # noqa: E402
    load_features,
    load_questions,
    load_teachers,
)


RUN_VERSION = "rag2_gold_free_semantic_gate_free_generation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path, default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-questions", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def generate(
    question: Any,
    biases: list[float],
    tokenizer: Any,
    model: Any,
    choice_ids: dict[str, int],
    layer_start: int,
    max_new_tokens: int,
    max_model_length: int,
) -> tuple[str, str, list[str], int]:
    texts = [" ".join(document.text.split()) for document in question.documents]
    evidence = "\n\n".join(texts)
    row = {"question": question.question, "options": dict(question.options)}
    user_prompt = build_anchored_user_prompt(row, evidence)
    chat_prompt = render_chat_prompt(tokenizer, user_prompt)
    rationale_prompt = chat_prompt + RATIONALE_HEADER
    input_ids, token_bias, query_mask = build_semantic_tensors(
        tokenizer, rationale_prompt, user_prompt, texts, biases,
        assistant_start=len(chat_prompt), reserved_length=max_new_tokens,
        device=model.device,
    )
    if int(input_ids.shape[1]) + max_new_tokens > max_model_length:
        raise RuntimeError(f"Generation budget exceeded for {question.sample_id}")
    stop_texts = [END_REASONING_MARKER, "\nFinal answer:", "\nTherefore, the answer"]
    stop_ids = [tokenizer.encode(text, add_special_tokens=False) for text in stop_texts]
    generated = greedy_generate_with_semantic_attention(
        model, input_ids, token_bias, query_mask, layer_start,
        max_new_tokens, stop_ids, tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    rationale, flags = normalize_rationale(raw)
    decision_prompt = (
        chat_prompt + RATIONALE_HEADER + rationale + "\n" + END_REASONING_MARKER
        + "\n" + FINAL_ANSWER_PREFIX
    )
    decision_ids, decision_bias, decision_query = build_semantic_tensors(
        tokenizer, decision_prompt, user_prompt, texts, biases,
        assistant_start=len(chat_prompt), reserved_length=0, device=model.device,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=decision_ids, attention_mask=torch.ones_like(decision_ids),
            use_cache=False, semantic_token_bias=decision_bias,
            semantic_query_mask=decision_query, semantic_layer_start=layer_start,
        )
    answer = max(CHOICES, key=lambda choice: float(outputs.logits[0, -1, choice_ids[choice]].item()))
    return answer, canonical_response(rationale, answer, dict(question.options)), flags, int(generated.shape[1])


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    for path in (args.feature_dir, args.index_path, args.teacher_dir, args.controller, args.llm_model):
        if not path.exists():
            raise FileNotFoundError(path)
    feature_manifest = json.loads((args.feature_dir / "preparation_manifest.json").read_text())
    teachers = load_teachers(args.teacher_dir)["test"][: args.test_questions]
    wanted_ids = {row["sample_id"] for row in teachers}
    wanted = {"train": set(), "val": set(), "test": wanted_ids}
    features = load_features(args.feature_dir, feature_manifest, wanted)["test"]
    questions = load_questions(args.index_path, wanted)
    checkpoint = torch.load(args.controller, map_location="cpu", weights_only=False)
    train_contract = checkpoint["run_contract"]
    if train_contract.get("gold_used_for_training") is not False:
        raise RuntimeError("Controller is not the gold-free experiment")
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "test_questions": args.test_questions,
        "sample_ids": [row["sample_id"] for row in teachers],
        "controller": str(args.controller.resolve()),
        "controller_best_epoch": checkpoint["best_epoch"],
        "llm_model": str(args.llm_model.resolve()),
        "generation": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "policies": ["zero_gate", "learned_gate"],
        "gold_used_for_policy": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "experiment_manifest.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("Evaluation resume contract mismatch")
    else:
        atomic_json(contract_path, contract)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
    attention_name = register_semantic_attention()
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=attention_name,
    ).to(device)
    freeze_module_for_controller_training(model)
    choice_ids = {}
    for choice in CHOICES:
        token_ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"Choice token mismatch: {choice}={token_ids}")
        choice_ids[choice] = token_ids[0]
    architecture = train_contract["architecture"]
    controller = SemanticContextGate(
        input_dim=int(feature_manifest["feature_hidden_size"]),
        hidden_dim=int(architecture["hidden_size"]), heads=int(architecture["set_heads"]),
        layers=int(architecture["set_layers"]), dropout=float(architecture["dropout"]),
        prior_strength=float(architecture["prior_strength"]),
        max_suppression_factor=float(architecture["max_suppression_factor"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()

    outputs = {policy: args.output_dir / f"{policy}.jsonl" for policy in contract["policies"]}
    cached = {policy: existing(path) if args.resume else {} for policy, path in outputs.items()}
    total = args.test_questions * 2
    completed = sum(len(rows) for rows in cached.values())
    progress = PipelineProgress(
        overall_total=total,
        overall_initial=completed,
        desc=f"GoldFreeGateGeneration:{args.dataset}",
    )
    try:
        for policy_index, policy in enumerate(contract["policies"], start=1):
            progress.set_stage(
                f"{policy_index}/2 free rationale+answer generation policy={policy}",
                total=args.test_questions, initial=len(cached[policy]),
            )
            for index, teacher in enumerate(teachers, start=1):
                sample_id = teacher["sample_id"]
                if sample_id in cached[policy]:
                    continue
                question = questions[sample_id]
                values = features[sample_id]
                with torch.no_grad():
                    gate = controller(
                        values["features"].to(device).unsqueeze(0),
                        values["margins"].to(device).unsqueeze(0),
                        values["masks"].to(device).unsqueeze(0),
                    )
                biases = (
                    [0.0] * 8 if policy == "zero_gate"
                    else [float(value) for value in gate.document_bias[0].cpu().tolist()]
                )
                answer, response, flags, rationale_tokens = generate(
                    question, biases, tokenizer, model, choice_ids,
                    int(train_contract["semantic_layer_start"]), args.max_new_tokens,
                    args.max_model_length,
                )
                row = {
                    "run_version": RUN_VERSION, "sample_id": sample_id, "policy": policy,
                    "prediction": answer, "correct": answer in question.gold_answers,
                    "canonical_response": response, "quality_flags": flags,
                    "rationale_tokens": rationale_tokens, "document_biases": biases,
                }
                append_jsonl(outputs[policy], row)
                cached[policy][sample_id] = row
                progress.update(1)
                progress.set_detail(f"policy={policy} item={index}/{args.test_questions}")
    finally:
        progress.close()

    zero = cached["zero_gate"]
    learned = cached["learned_gate"]
    zero_correct = sum(bool(row["correct"]) for row in zero.values())
    learned_correct = sum(bool(row["correct"]) for row in learned.values())
    wrong_to_correct = sum(not zero[key]["correct"] and learned[key]["correct"] for key in zero)
    correct_to_wrong = sum(zero[key]["correct"] and not learned[key]["correct"] for key in zero)
    summary = {
        "run_version": RUN_VERSION, "completed_at": datetime.now(timezone.utc).isoformat(),
        "questions": args.test_questions,
        "zero_gate": {"correct": zero_correct, "accuracy": zero_correct / args.test_questions},
        "learned_gate": {"correct": learned_correct, "accuracy": learned_correct / args.test_questions},
        "absolute_accuracy_change": (learned_correct - zero_correct) / args.test_questions,
        "wrong_to_correct": wrong_to_correct, "correct_to_wrong": correct_to_wrong,
        "net_answer_gain": wrong_to_correct - correct_to_wrong,
        "interpretation_limit": "accuracy is an evaluation-only consequence; it was absent from training",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    logging.info("Free-generation evaluation complete: %s", json.dumps(summary))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate zero-bias, frozen-prior, and learned semantic attention policies.

All three conditions use the same Hugging Face Llama replay, tokenization,
cached rationale, and final-choice query.  This isolates the learned
controller from both the fixed semantic prior and vLLM-versus-HF backend
differences present in the training script's diagnostic cached reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.generation.learned_semantic_attention import (  # noqa: E402
    SemanticResidualAttentionController,
    freeze_module_for_controller_training,
)
from medrag.generation.semantic_attention import register_semantic_attention  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402
from scripts.train_rag2_semantic_attention_controller import (  # noqa: E402
    batch_controller_inputs,
    collate_prefix_batch,
    count_questions,
    final_choice_logits,
    list_feature_shards,
    load_feature_shard,
    model_bundle_identity,
)


RUN_VERSION = "rag2_semantic_attention_controller_ablation_v1"
POLICIES = ("zero_bias", "semantic_prior", "learned_controller")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("val", "test"),
    )
    parser.add_argument(
        "--llm-model",
        type=Path,
        default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct",
    )
    parser.add_argument("--question-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_controller(run_contract: dict[str, Any]) -> SemanticResidualAttentionController:
    return SemanticResidualAttentionController(
        input_dim=int(run_contract["feature_hidden_size"]),
        hidden_dim=int(run_contract["controller_hidden_size"]),
        dropout=float(run_contract["controller_dropout"]),
        temperature=float(run_contract["semantic_temperature"]),
        max_suppression_bias=math.log(float(run_contract["max_suppression_factor"])),
        prior_strength=float(run_contract["prior_strength"]),
        boundary_epsilon=float(run_contract["boundary_epsilon"]),
    )


def result_path(output_dir: Path, split: str, policy: str) -> Path:
    return output_dir / "condition_results" / f"{split}__{policy}.json"


def valid_cached_result(path: Path, fingerprint: str, expected: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("run_version") == RUN_VERSION
        and value.get("contract_fingerprint") == fingerprint
        and len(value.get("predictions") or []) == expected
        and len(value.get("gold_options") or []) == expected
        and len(value.get("no_rag_correct") or []) == expected
    )


def summarize_condition(
    predictions: list[int],
    gold: list[int],
    no_rag_correct: list[bool],
    ce_sum: float,
    bias_sum: float,
    gate_sum: float,
    document_count: int,
) -> dict[str, Any]:
    count = len(gold)
    correct = [prediction == target for prediction, target in zip(predictions, gold, strict=True)]
    groups: dict[str, dict[str, Any]] = {}
    for flag, name in ((False, "no_rag_wrong"), (True, "no_rag_correct")):
        indices = [index for index, value in enumerate(no_rag_correct) if value is flag]
        group_correct = sum(correct[index] for index in indices)
        groups[name] = {
            "questions": len(indices),
            "correct": group_correct,
            "accuracy": group_correct / max(1, len(indices)),
        }
    return {
        "questions": count,
        "correct": sum(correct),
        "accuracy": sum(correct) / max(1, count),
        "answer_ce": ce_sum / max(1, count),
        "mean_document_bias": bias_sum / max(1, document_count),
        "mean_keep_gate": gate_sum / max(1, document_count),
        "no_rag_groups": groups,
    }


def compare_conditions(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    if base["gold_options"] != target["gold_options"]:
        raise RuntimeError("Ablation conditions have different gold-answer order")
    gold = base["gold_options"]
    base_correct = [p == g for p, g in zip(base["predictions"], gold, strict=True)]
    target_correct = [p == g for p, g in zip(target["predictions"], gold, strict=True)]
    wrong_to_correct = sum((not before) and after for before, after in zip(base_correct, target_correct))
    correct_to_wrong = sum(before and (not after) for before, after in zip(base_correct, target_correct))
    total = len(gold)
    return {
        "base": base["policy"],
        "target": target["policy"],
        "accuracy_delta": (sum(target_correct) - sum(base_correct)) / max(1, total),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "net_answer_gain": wrong_to_correct - correct_to_wrong,
        "changed_predictions": sum(
            left != right
            for left, right in zip(base["predictions"], target["predictions"], strict=True)
        ),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.question_batch_size <= 0:
        raise ValueError("--question-batch-size must be positive")
    for path in (args.feature_dir, args.controller_path, args.llm_model):
        if not path.exists():
            raise FileNotFoundError(path)
    feature_manifest = json.loads(
        (args.feature_dir / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(args.controller_path, map_location="cpu", weights_only=False)
    run_contract = checkpoint.get("run_contract") or {}
    if (
        checkpoint.get("run_version") != "rag2_semantic_final_choice_attention_controller_v1"
        or run_contract.get("dataset") != args.dataset
        or feature_manifest.get("dataset") != args.dataset
        or run_contract.get("feature_contract_fingerprint")
        != feature_manifest.get("contract_fingerprint")
    ):
        raise RuntimeError("Controller, dataset, and prepared-feature contracts do not match")
    llm_identity = model_bundle_identity(args.llm_model)
    if run_contract.get("llm_model_bundle") != llm_identity:
        raise RuntimeError("Controller was trained against a different Llama bundle")
    split_paths = {
        split: list_feature_shards(args.feature_dir, split, feature_manifest)
        for split in args.splits
    }
    split_counts = {split: count_questions(paths) for split, paths in split_paths.items()}
    contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "feature_contract_fingerprint": feature_manifest["contract_fingerprint"],
        "controller_sha256": sha256_file(args.controller_path),
        "controller_path": str(args.controller_path.resolve()),
        "llm_model_bundle": llm_identity,
        "splits": list(args.splits),
        "split_questions": split_counts,
        "policies": list(POLICIES),
        "question_batch_size": args.question_batch_size,
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract = {**contract, "contract_fingerprint": fingerprint}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "ablation_contract.json"
    if contract_path.is_file() and args.resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Ablation resume contract mismatch; use a new output directory")
    elif not args.resume and any(args.output_dir.iterdir()):
        raise RuntimeError("--no-resume requires a new or empty output directory")
    else:
        atomic_write_json(contract_path, contract)

    total = sum(split_counts.values()) * len(POLICIES)
    completed = sum(
        split_counts[split]
        for split in args.splits
        for policy in POLICIES
        if args.resume
        and valid_cached_result(result_path(args.output_dir, split, policy), fingerprint, split_counts[split])
    )
    logging.info(
        "Controller ablation plan: splits=%s policies=%s cached=%d/%d",
        split_counts,
        POLICIES,
        completed,
        total,
    )
    progress = PipelineProgress(
        overall_total=total,
        overall_initial=completed,
        desc=f"SemanticGateAblation:{args.dataset}",
    )
    device = torch.device(args.device)
    learned: SemanticResidualAttentionController | None = None
    prior: SemanticResidualAttentionController | None = None
    model: Any | None = None
    tokenizer: Any | None = None
    choice_token_ids: torch.Tensor | None = None
    attention_name: str | None = None
    try:
        if completed < total:
            learned = make_controller(run_contract).to(device)
            learned.load_state_dict(checkpoint["controller"])
            learned.eval()
            prior = make_controller(run_contract).to(device)
            prior.eval()  # Its zero-initialized final layer gives exactly zero residual.
            tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            choice_ids = []
            for label in CHOICES:
                encoded = tokenizer.encode(label, add_special_tokens=False)
                if len(encoded) != 1:
                    raise RuntimeError(f"Choice {label} is not a single token: {encoded}")
                choice_ids.append(int(encoded[0]))
            choice_token_ids = torch.tensor(choice_ids, dtype=torch.long, device=device)
            attention_name = register_semantic_attention()
            logging.info("Loading frozen Llama for backend-matched ablation: %s", args.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device)
            freeze_module_for_controller_training(model)

        results: dict[str, dict[str, Any]] = {}
        hidden_size = int(feature_manifest["feature_hidden_size"])
        for split in args.splits:
            for policy in POLICIES:
                path = result_path(args.output_dir, split, policy)
                if args.resume and valid_cached_result(path, fingerprint, split_counts[split]):
                    results[f"{split}:{policy}"] = json.loads(path.read_text(encoding="utf-8"))
                    continue
                if any(value is None for value in (learned, prior, model, tokenizer, choice_token_ids, attention_name)):
                    raise RuntimeError("Ablation models were not initialized")
                progress.set_stage(
                    f"{split}: {policy} with identical HF final-choice replay",
                    total=split_counts[split],
                )
                predictions: list[int] = []
                gold_options: list[int] = []
                no_rag_flags: list[bool] = []
                sample_ids: list[str] = []
                ce_sum = bias_sum = gate_sum = 0.0
                document_count = 0
                residual_sum = residual_square_sum = residual_abs_sum = 0.0
                residual_count = 0
                with torch.no_grad():
                    for shard_number, shard_path in enumerate(split_paths[split], start=1):
                        payload = load_feature_shard(
                            shard_path,
                            dataset=args.dataset,
                            split=split,
                            fingerprint=feature_manifest["contract_fingerprint"],
                            hidden_size=hidden_size,
                        )
                        for start in range(0, len(payload["sample_ids"]), args.question_batch_size):
                            indices = list(
                                range(start, min(start + args.question_batch_size, len(payload["sample_ids"])))
                            )
                            values = batch_controller_inputs(payload, indices, device)
                            if policy == "zero_bias":
                                document_bias = torch.zeros_like(values["margins"])
                                residual = torch.zeros_like(document_bias)
                            else:
                                controller = prior if policy == "semantic_prior" else learned
                                output = controller(values["features"], values["margins"])
                                document_bias = output.document_bias
                                residual = output.residual
                            prefix = collate_prefix_batch(
                                payload,
                                indices,
                                pad_token_id=int(tokenizer.pad_token_id),
                                device=device,
                            )
                            logits = final_choice_logits(
                                model,
                                attention_name,
                                prefix,
                                document_bias,
                                choice_token_ids,
                                int(run_contract["semantic_layer_start"]),
                            )
                            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                            gold_options.extend(values["gold"].cpu().tolist())
                            no_rag_flags.extend(values["no_rag_correct"].cpu().tolist())
                            sample_ids.extend(payload["sample_ids"][index] for index in indices)
                            ce_sum += float(
                                F.cross_entropy(logits, values["gold"], reduction="sum").item()
                            )
                            bias_sum += float(document_bias.sum().item())
                            gate_sum += float(document_bias.exp().sum().item())
                            document_count += int(document_bias.numel())
                            residual_sum += float(residual.sum().item())
                            residual_square_sum += float(residual.square().sum().item())
                            residual_abs_sum += float(residual.abs().sum().item())
                            residual_count += int(residual.numel())
                            progress.update(len(indices))
                            progress.set_detail(
                                f"split={split} policy={policy} shard={shard_number}/{len(split_paths[split])}"
                            )
                result = {
                    "run_version": RUN_VERSION,
                    "contract_fingerprint": fingerprint,
                    "split": split,
                    "policy": policy,
                    "sample_ids": sample_ids,
                    "predictions": predictions,
                    "gold_options": gold_options,
                    "no_rag_correct": no_rag_flags,
                    "metrics": summarize_condition(
                        predictions,
                        gold_options,
                        no_rag_flags,
                        ce_sum,
                        bias_sum,
                        gate_sum,
                        document_count,
                    ),
                    "residual": {
                        "mean": residual_sum / max(1, residual_count),
                        "mean_abs": residual_abs_sum / max(1, residual_count),
                        "rms": math.sqrt(residual_square_sum / max(1, residual_count)),
                    },
                }
                atomic_write_json(path, result)
                results[f"{split}:{policy}"] = result

        summary: dict[str, Any] = {
            "run_version": RUN_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "contract_fingerprint": fingerprint,
            "conditions": {},
            "comparisons": {},
        }
        for split in args.splits:
            split_results = {policy: results[f"{split}:{policy}"] for policy in POLICIES}
            summary["conditions"][split] = {
                policy: {
                    "metrics": split_results[policy]["metrics"],
                    "residual": split_results[policy]["residual"],
                }
                for policy in POLICIES
            }
            summary["comparisons"][split] = {
                "prior_vs_zero": compare_conditions(
                    split_results["zero_bias"], split_results["semantic_prior"]
                ),
                "learned_vs_zero": compare_conditions(
                    split_results["zero_bias"], split_results["learned_controller"]
                ),
                "learned_vs_prior": compare_conditions(
                    split_results["semantic_prior"], split_results["learned_controller"]
                ),
            }
        atomic_write_json(args.output_dir / "summary.json", summary)
        logging.info("Backend-matched controller ablation complete: %s", args.output_dir)
    finally:
        progress.close()


if __name__ == "__main__":
    main()

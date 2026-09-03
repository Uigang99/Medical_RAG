#!/usr/bin/env python3
"""Held-out generalization pilot for document-token-restricted Llama LoRA."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rag2_document_first_prompt_order import HierarchicalProgress  # noqa: E402
from medrag.training.document_path_lora import DocumentPathAdapter  # noqa: E402
from train_rag2_document_path_overfit import (  # noqa: E402
    BASE,
    EncodedDataset,
    SmallestRecords,
    atomic_json,
    atomic_jsonl,
    atomic_torch,
    collate,
    evaluate,
    fidelity_audit,
    file_identity,
    fingerprint,
    forward_choice_logits,
    git_commit,
    losses,
    no_rag_logit_error,
    prepare_record,
    selected_choice_head,
    sha256_file,
    stable_priority,
    utc_now,
)


RUN_VERSION = "rag2_document_path_generalization_pilot_v1"
DATA_VERSION = "rag2_document_path_generalization_4k_1k_1k_v1"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa",), default="medmcqa")
    parser.add_argument("--data-root", type=Path, default=BASE)
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/RAG2-Document-Path-LoRA",
    )
    parser.add_argument("--run-name", default="medmcqa_document_first_generalization_4k_v1")
    parser.add_argument("--train-questions", type=int, default=4000)
    parser.add_argument("--val-questions", type=int, default=1000)
    parser.add_argument("--test-questions", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--preference-margin", type=float, default=0.5)
    parser.add_argument("--positive-baseline-margin", type=float, default=0.25)
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument("--positive-baseline-weight", type=float, default=0.5)
    parser.add_argument("--negative-invariance-weight", type=float, default=1.0)
    parser.add_argument("--swap-invariance-weight", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--base-logit-tolerance", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager",), default="eager")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def select_split(
    args: argparse.Namespace,
    split: str,
    source_path: Path,
    source_count: int,
    requested: int,
    output_path: Path,
    progress: HierarchicalProgress,
    progress_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if requested % 2:
        raise ValueError(f"{split} question count must be even")
    reservoirs = {
        "no_rag_correct": SmallestRecords(requested // 2),
        "no_rag_wrong": SmallestRecords(requested // 2),
    }
    available: Counter[str] = Counter()
    observed_ids: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for observed, line in enumerate(handle, 1):
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in observed_ids:
                raise RuntimeError(f"Duplicate {split} question: {sample_id}")
            observed_ids.add(sample_id)
            _stratum, prepared = prepare_record(row)
            group = "no_rag_correct" if prepared["frozen_no_rag"]["answer_correct"] else "no_rag_wrong"
            available[group] += 1
            reservoirs[group].add(
                stable_priority(args.seed, f"{split}\0{sample_id}"), sample_id, prepared
            )
            if observed % 32 == 0 or observed == source_count:
                progress.set_absolute(progress_offset + observed)
    if len(observed_ids) != source_count:
        raise RuntimeError(
            f"{split} source count mismatch: expected={source_count} actual={len(observed_ids)}"
        )
    selected: list[dict[str, Any]] = []
    for group, reservoir in reservoirs.items():
        values = reservoir.rows()
        if len(values) != requested // 2:
            raise RuntimeError(
                f"Insufficient {split}/{group}: requested={requested//2} available={available[group]}"
            )
        selected.extend(values)
    random.Random(args.seed + SPLITS.index(split)).shuffle(selected)
    atomic_jsonl(output_path, selected)
    summary = {
        "source_questions": source_count,
        "selected_questions": len(selected),
        "available_no_rag_groups": dict(available),
        "selected_no_rag_groups": dict(
            Counter(
                "no_rag_correct" if row["frozen_no_rag"]["answer_correct"] else "no_rag_wrong"
                for row in selected
            )
        ),
        "selected_strata": dict(Counter(row["stratum"] for row in selected)),
        "sha256": sha256_file(output_path),
    }
    return selected, summary


def materialize_splits(
    args: argparse.Namespace,
    source_root: Path,
    source_manifest: dict[str, Any],
    output_root: Path,
    progress: HierarchicalProgress,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    requested = {
        "train": args.train_questions,
        "val": args.val_questions,
        "test": args.test_questions,
    }
    manifest_path = output_root / "manifest.json"
    output_paths = {split: output_root / f"{split}.jsonl" for split in SPLITS}
    expected_contract = fingerprint(
        {
            "data_version": DATA_VERSION,
            "source_contract_sha256": source_manifest["contract_sha256"],
            "requested": requested,
            "selection": "No-RAG correct/wrong 50:50; natural alignment within each group",
            "seed": args.seed,
        }
    )
    source_counts = {
        split: int(source_manifest["split_summary"][split]["questions"])
        for split in SPLITS
    }
    total_source = sum(source_counts.values())
    if args.resume and manifest_path.is_file() and all(path.is_file() for path in output_paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("selection_contract_sha256") == expected_contract
            and all(
                manifest["splits"][split]["sha256"] == sha256_file(output_paths[split])
                for split in SPLITS
            )
        ):
            progress.set_initial(total_source)
            return {
                split: [json.loads(line) for line in output_paths[split].open("r", encoding="utf-8")]
                for split in SPLITS
            }, manifest

    rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    offset = 0
    selected_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        rows[split], summaries[split] = select_split(
            args,
            split,
            source_root / "training_dataset" / f"{split}.jsonl",
            source_counts[split],
            requested[split],
            output_paths[split],
            progress,
            offset,
        )
        selected_ids[split] = {str(row["sample_id"]) for row in rows[split]}
        offset += source_counts[split]
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = selected_ids[left] & selected_ids[right]
            if overlap:
                raise RuntimeError(f"Question leakage {left}/{right}: {sorted(overlap)[:3]}")
    manifest = {
        "data_version": DATA_VERSION,
        "created_at": utc_now(),
        "selection_contract_sha256": expected_contract,
        "source_contract_sha256": source_manifest["contract_sha256"],
        "selection": "No-RAG correct/wrong 50:50; natural alignment within each group",
        "question_overlap_across_splits": 0,
        "splits": summaries,
    }
    atomic_json(manifest_path, manifest)
    return rows, manifest


def bootstrap_delta(
    baseline: Sequence[dict[str, Any]],
    trained: Sequence[dict[str, Any]],
    path: tuple[str, ...],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if [row["sample_id"] for row in baseline] != [row["sample_id"] for row in trained]:
        raise RuntimeError("Paired bootstrap row order mismatch")

    def read(row: dict[str, Any]) -> float:
        value: Any = row
        for key in path:
            value = value[key]
        return float(value)

    delta = np.asarray([read(t) - read(b) for b, t in zip(baseline, trained)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        positions = rng.integers(0, len(delta), len(delta))
        means[index] = delta[positions].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not torch.cuda.is_available() and not args.prepare_only:
        raise RuntimeError("CUDA is required for the generalization pilot")
    if min(args.train_questions, args.val_questions, args.test_questions) <= 0:
        raise ValueError("All split sizes must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_root = args.data_root / args.dataset
    source_manifest_path = source_root / "training_dataset/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    prepared_root = source_root / "document_path_generalization_4k_1k_1k_v1"
    output_dir = args.output_root / args.dataset / args.run_name
    contract = {
        "run_version": RUN_VERSION,
        "purpose": "held-out document-path generalization pilot",
        "dataset": args.dataset,
        "source_manifest": file_identity(source_manifest_path, content_hash=True),
        "split_questions": {
            "train": args.train_questions,
            "val": args.val_questions,
            "test": args.test_questions,
        },
        "selection": "No-RAG correct/wrong 50:50; natural alignment within group",
        "test_used_for_checkpoint_selection": False,
        "prompt_order": "documents_then_question_options",
        "gold_and_semantic_labels_in_prompt": False,
        "trainable_path": "document-token positions of all Llama K/V projections only",
        "model_config": file_identity(args.model_name_or_path / "config.json", content_hash=True),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "loss": {
            "preference_margin": args.preference_margin,
            "positive_baseline_margin": args.positive_baseline_margin,
            "preference_weight": args.preference_weight,
            "positive_baseline_weight": args.positive_baseline_weight,
            "negative_invariance_weight": args.negative_invariance_weight,
            "swap_invariance_weight": args.swap_invariance_weight,
        },
        "primary_metric": "test both_preferences delta versus frozen baseline",
        "pass_threshold": {
            "both_preferences_delta": 0.05,
            "positive_accuracy_min_delta": 0.0,
            "negative_kl_max_increase": 0.0,
            "swap_kl_max_increase": 0.0,
        },
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_input_tokens": args.max_input_tokens,
        "base_logit_tolerance": args.base_logit_tolerance,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    contract_hash = fingerprint(contract)
    contract_path = output_dir / "run_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract_hash:
            raise RuntimeError("Generalization run contract mismatch; use a new --run-name")
    else:
        atomic_json(
            contract_path,
            {
                "contract_sha256": contract_hash,
                "created_at": utc_now(),
                "code_commit": git_commit(),
                "code_sha256": sha256_file(Path(__file__)),
                "contract": contract,
            },
        )

    source_counts = {
        split: int(source_manifest["split_summary"][split]["questions"])
        for split in SPLITS
    }
    selected_total = args.train_questions + args.val_questions + args.test_questions
    stage_names = (
        "audit source splits and materialize disjoint pilot cohorts",
        "tokenize train validation and test conditions",
        "load frozen Llama and verify adapter fidelity",
        "measure frozen validation baseline",
        "train adapters and select checkpoint on validation",
        "evaluate frozen and selected models on held-out test",
        "write paired report and completion manifest",
    )
    stage_estimates = (30.0, 45.0, 20.0, 120.0, 3300.0, 300.0, 10.0)
    progress = HierarchicalProgress(stage_names, stage_estimates)
    progress.log(
        f"[workflow plan] dataset={args.dataset} train={args.train_questions} "
        f"val={args.val_questions} test={args.test_questions} epochs={args.epochs} "
        f"batch={args.batch_size} output={output_dir}"
    )
    try:
        progress.start_stage(1, sum(source_counts.values()), "question")
        split_rows, prepared_manifest = materialize_splits(
            args, source_root, source_manifest, prepared_root, progress
        )
        selected_counts = {split: len(rows) for split, rows in split_rows.items()}
        progress.complete_stage(f"selected={selected_counts} data={prepared_root}")
        if args.prepare_only:
            progress.finish("prepare-only: no model loaded")
            return

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        progress.start_stage(2, selected_total, "question")
        datasets: dict[str, EncodedDataset] = {}
        offset = 0
        for split in SPLITS:
            datasets[split] = EncodedDataset(
                split_rows[split], tokenizer, args.max_input_tokens, progress,
                progress_offset=offset,
            )
            offset += len(split_rows[split])
        progress.complete_stage(
            "max_tokens=" + json.dumps({split: data.max_tokens for split, data in datasets.items()})
        )

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
        logging.info("Loading frozen target Llama: %s", args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, low_cpu_mem_usage=True,
            dtype=dtype, attn_implementation=args.attn_implementation,
        ).to(args.device)
        model._document_path_tokenizer = tokenizer
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        adapter = DocumentPathAdapter(
            model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
        )
        zero_state = adapter.adapter_state_dict()
        choice_weight, choice_bias = selected_choice_head(model)
        device = torch.device(args.device)
        collator = lambda values: collate(values, int(tokenizer.pad_token_id))

        progress.start_stage(3, min(32, len(datasets["train"])) * 2, "question-condition-set")
        fidelity = fidelity_audit(
            model, adapter, datasets["train"], tokenizer, choice_weight, choice_bias, args, progress
        )
        progress.complete_stage(f"fidelity={json.dumps(fidelity, ensure_ascii=False)}")

        validation_loader = DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False, collate_fn=collator
        )
        progress.start_stage(4, args.val_questions, "question")
        frozen_validation = evaluate(
            model, adapter, validation_loader, choice_weight, choice_bias, args, progress
        )
        progress.complete_stage(
            f"both={frozen_validation['both_preferences']:.4f} "
            f"positive_acc={frozen_validation['positive_correct']:.4f}"
        )

        parameters = list(adapter.trainable_parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        checkpoint_path = output_dir / "checkpoint.pt"
        best_path = output_dir / "best_adapter.pt"
        history: list[dict[str, Any]] = []
        start_epoch = 1
        best_epoch = 0
        best_score = -float("inf")
        if args.resume and checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("contract_sha256") != contract_hash:
                raise RuntimeError("Generalization checkpoint contract mismatch")
            adapter.load_adapter_state_dict(checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(args.device)
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint["best_epoch"])
            best_score = float(checkpoint["best_score"])
            logging.info("Resuming at epoch %d; best_epoch=%d", start_epoch, best_epoch)

        passes_per_epoch = args.train_questions + args.val_questions
        progress.start_stage(5, args.epochs * passes_per_epoch, "question-pass")
        progress.set_initial((start_epoch - 1) * passes_per_epoch)
        for epoch in range(start_epoch, args.epochs + 1):
            train_loader = DataLoader(
                datasets["train"], batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(args.seed + epoch),
                collate_fn=collator,
            )
            model.train()
            train_sums: Counter[str] = Counter()
            trained = 0
            progress.log(f"[stage 5/7 | train epoch {epoch}/{args.epochs}] starting")
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = forward_choice_logits(
                    model, adapter, batch, choice_weight, choice_bias, device
                )
                current = losses(logits, batch["rows"], args, device)
                current["loss"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                count = len(batch["rows"])
                trained += count
                for key, value in current.items():
                    train_sums[key] += float(value.detach()) * count
                progress.update(count)
            validation = evaluate(
                model, adapter, validation_loader, choice_weight, choice_bias, args, progress
            )
            score = float(validation["both_preferences"])
            epoch_row = {
                "epoch": epoch,
                "train": {key: value / trained for key, value in train_sums.items()},
                "validation": validation,
                "selection_score": score,
            }
            history.append(epoch_row)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                atomic_torch(
                    best_path,
                    {"contract_sha256": contract_hash, "epoch": epoch, "adapter": adapter.adapter_state_dict()},
                )
            progress.log(
                f"[epoch {epoch}/{args.epochs}] loss={epoch_row['train']['loss']:.4f} "
                f"val_both={validation['both_preferences']:.4f} "
                f"baseline={frozen_validation['both_preferences']:.4f} "
                f"delta={validation['both_preferences']-frozen_validation['both_preferences']:+.4f} "
                f"positive_acc={validation['positive_correct']:.4f} best_epoch={best_epoch}"
            )
            atomic_torch(
                checkpoint_path,
                {
                    "contract_sha256": contract_hash,
                    "epoch": epoch,
                    "adapter": adapter.adapter_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
            )
        progress.complete_stage(f"epochs={len(history)} best_epoch={best_epoch} best_score={best_score:.4f}")
        if not best_path.is_file():
            raise RuntimeError("No validation-selected adapter was saved")

        test_loader = DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collator
        )
        progress.start_stage(6, args.test_questions * 2 + min(32, args.test_questions), "question")
        frozen_test_rows: list[dict[str, Any]] = []
        adapter.load_adapter_state_dict(zero_state)
        frozen_test = evaluate(
            model, adapter, test_loader, choice_weight, choice_bias, args, progress,
            prediction_rows=frozen_test_rows,
        )
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best.get("contract_sha256") != contract_hash:
            raise RuntimeError("Best adapter contract mismatch")
        adapter.load_adapter_state_dict(best["adapter"])
        trained_test_rows: list[dict[str, Any]] = []
        trained_test = evaluate(
            model, adapter, test_loader, choice_weight, choice_bias, args, progress,
            prediction_rows=trained_test_rows,
        )
        final_no_rag_error = no_rag_logit_error(
            model, adapter, datasets["test"].values[: min(32, args.test_questions)],
            tokenizer, choice_weight, choice_bias, args, progress,
        )
        atomic_jsonl(output_dir / "test_frozen_predictions.jsonl", frozen_test_rows)
        atomic_jsonl(output_dir / "test_trained_predictions.jsonl", trained_test_rows)
        progress.complete_stage(
            f"baseline_both={frozen_test['both_preferences']:.4f} "
            f"trained_both={trained_test['both_preferences']:.4f}"
        )

        progress.start_stage(7, 1, "report")
        scalar_deltas = {
            key: float(trained_test[key] - frozen_test[key])
            for key in (
                "positive_gt_negative", "positive_gt_swap", "both_preferences",
                "positive_correct", "negative_correct", "swap_correct",
                "negative_kl", "swap_kl", "positive_c2w", "negative_c2w",
            )
        }
        paired_intervals = {
            "both_preferences": bootstrap_delta(
                frozen_test_rows, trained_test_rows, ("relations", "both_preferences"),
                args.bootstrap_replicates, args.seed,
            ),
            "positive_accuracy": bootstrap_delta(
                frozen_test_rows, trained_test_rows, ("positive", "correct"),
                args.bootstrap_replicates, args.seed + 1,
            ),
            "negative_kl": bootstrap_delta(
                frozen_test_rows, trained_test_rows, ("negative", "kl_from_no_rag"),
                args.bootstrap_replicates, args.seed + 2,
            ),
            "swap_kl": bootstrap_delta(
                frozen_test_rows, trained_test_rows, ("swap", "kl_from_no_rag"),
                args.bootstrap_replicates, args.seed + 3,
            ),
        }
        thresholds = contract["pass_threshold"]
        passed = bool(
            scalar_deltas["both_preferences"] >= thresholds["both_preferences_delta"]
            and scalar_deltas["positive_correct"] >= thresholds["positive_accuracy_min_delta"]
            and scalar_deltas["negative_kl"] <= thresholds["negative_kl_max_increase"]
            and scalar_deltas["swap_kl"] <= thresholds["swap_kl_max_increase"]
            and final_no_rag_error <= args.base_logit_tolerance
            and adapter.audit()["max_non_document_delta"] == 0.0
        )
        summary = {
            "run_version": RUN_VERSION,
            "completed_at": utc_now(),
            "contract_sha256": contract_hash,
            "passed": passed,
            "interpretation_scope": "held-out mechanism pilot; not final benchmark accuracy",
            "prepared_manifest": prepared_manifest,
            "frozen_base_fidelity": fidelity,
            "frozen_validation": frozen_validation,
            "best_epoch": best_epoch,
            "history": history,
            "test_questions": args.test_questions,
            "frozen_test": frozen_test,
            "trained_test": trained_test,
            "test_deltas": scalar_deltas,
            "paired_bootstrap_95ci": paired_intervals,
            "final_no_rag_max_logit_error": final_no_rag_error,
            "final_adapter_audit": adapter.audit(),
            "pass_threshold": thresholds,
            "next_action": (
                "scale to the preregistered full internal train split"
                if passed
                else "stop scaling and analyze held-out failure modes"
            ),
        }
        atomic_json(output_dir / "summary.json", summary)
        progress.update(1)
        progress.complete_stage(f"passed={passed} summary={output_dir/'summary.json'}")
        progress.finish(f"passed={passed} next={summary['next_action']}")
    except Exception:
        progress.log(
            f"[workflow FAILED] stage={progress.stage_index}/{progress.stage_count} "
            f"completed={progress.stage_done}/{progress.stage_total} "
            f"prepared={prepared_root} checkpoint={output_dir/'checkpoint.pt'}; "
            "rerun the identical command to resume"
        )
        raise


if __name__ == "__main__":
    main()

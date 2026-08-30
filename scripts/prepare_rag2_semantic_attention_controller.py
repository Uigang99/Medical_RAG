#!/usr/bin/env python3
"""Prepare validated, sharded inputs for semantic attention controller training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.filtering.rag2_official import build_official_filter_input, format_options  # noqa: E402
from medrag.filtering.semantic_features import (  # noqa: E402
    POOLING_VERSION,
    FrozenSemanticEvidenceEncoder,
)
from medrag.io_utils import iter_jsonl  # noqa: E402
from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    FINAL_ANSWER_PREFIX,
    GENERATION_POLICY_VERSION,
    PROMPT_VERSION,
    TRACE_VERSION,
    assistant_decision_prefix,
    build_anchored_user_prompt,
    render_chat_prompt,
)
from medrag.training.rag2_semantic_attention_data import (  # noqa: E402
    RAG2SemanticAttentionDataset,
    SemanticAttentionDataSources,
    build_semantic_attention_index,
    make_semantic_attention_build_plan,
)


RUN_VERSION = "rag2_semantic_attention_training_features_v1"
RATIONALE_RUN_VERSION = "rag2_top8_unbiased_rationale_cache_v1"
ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
)
DEFAULT_CANDIDATE_ROOT = ARTIFACT_ROOT / "candidates/source_balanced32_rerank8_v1"
DEFAULT_SPLIT_ROOT = ARTIFACT_ROOT / "filter_training_inputs_rag2_paper_reproduction_three_class_v1"
DEFAULT_NO_RAG_ROOT = ARTIFACT_ROOT / "train_no_rag_anchored_features_v1/no_rag"
DEFAULT_LABEL_ROOT = Path(
    "/home/user/codex_rag2_outputs/"
    "codex_evidence_utility_labels_three_anchor_top8_terra_medium_v1_incremental/"
    "terra_medium"
)
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_FILTERS = {
    "medqa": (
        WORKSPACE_ROOT
        / "models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa"
        / "medqa_semantic_top8_binary_support_epoch8_len1280_fullpair"
        / "20260830_170945/final_model"
    ),
    "medmcqa": (
        WORKSPACE_ROOT
        / "models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medmcqa"
        / "medmcqa_semantic_top8_binary_support_epoch5_len1280_fullpair"
        / "20260829_212146/final_model"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--semantic-label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--no-rag-root", type=Path, default=DEFAULT_NO_RAG_ROOT)
    parser.add_argument("--rationale-cache", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, default=None)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--expected-documents", type=int, default=8)
    parser.add_argument("--questions-per-shard", type=int, default=128)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--semantic-max-input-length", type=int, default=2048)
    parser.add_argument("--llm-max-model-length", type=int, default=8192)
    parser.add_argument("--max-questions-per-split", type=int, default=0)
    parser.add_argument(
        "--allow-partial-rationale-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, hash_small: bool = True) -> dict[str, Any]:
    stat = path.stat()
    value = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_small and stat.st_size < 16 * 1024 * 1024:
        value["sha256"] = sha256_file(path)
    return value


def model_bundle_identity(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / name
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        if (root / name).is_file()
    ]
    paths.extend(sorted(root.glob("*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No immutable model artifacts under {root}")
    return [file_identity(path) for path in paths]


def load_rationale_cache(root: Path, dataset: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete rationale cache has no generation manifest: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_contract = {
        "run_version": RATIONALE_RUN_VERSION,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "dataset": dataset,
        "split": "train",
        "docs_per_question": 8,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_contract.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Rationale-cache contract mismatch: {mismatches}")
    total = int(manifest["question_count"])
    per_shard = int(manifest["questions_per_shard"])
    shard_count = math.ceil(total / per_shard)
    expected_paths = [
        root / "rationale_shards" / f"shard_{index:05d}" / "questions.jsonl"
        for index in range(shard_count)
    ]
    actual_paths = sorted((root / "rationale_shards").glob("shard_*/questions.jsonl"))
    if actual_paths != expected_paths:
        raise RuntimeError(
            f"Rationale shard set differs from manifest: expected={len(expected_paths)} "
            f"actual={len(actual_paths)}"
        )
    rows: dict[str, dict[str, Any]] = {}
    fingerprint = str(manifest["contract_fingerprint"])
    for shard_index, path in enumerate(
        tqdm(
            expected_paths,
            total=shard_count,
            desc="stage=load validated rationale cache",
            unit="shard",
            dynamic_ncols=True,
            leave=False,
        )
    ):
        marker = path.with_name("COMPLETE.json")
        if not marker.is_file():
            raise RuntimeError(f"Rationale shard has no completion marker: {path}")
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        expected_questions = min(per_shard, total - shard_index * per_shard)
        if (
            marker_value.get("run_version") != RATIONALE_RUN_VERSION
            or marker_value.get("contract_fingerprint") != fingerprint
            or int(marker_value.get("question_count", -1)) != expected_questions
            or int(marker_value.get("rows_size_bytes", -1)) != path.stat().st_size
            or marker_value.get("rows_sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"Rationale completion marker mismatch: {marker}")
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in rows:
                raise ValueError(f"Missing or duplicate rationale sample_id: {sample_id!r}")
            if (
                row.get("run_version") != RATIONALE_RUN_VERSION
                or row.get("dataset") != dataset
                or row.get("split") != "train"
                or len(row.get("document_pair_ids") or []) != 8
                or str(row.get("answer") or "").strip().upper() not in "ABCD"
                or not str(row.get("rationale") or "").strip()
            ):
                raise ValueError(f"Invalid rationale-cache row contract: {sample_id}")
            rows[sample_id] = row
    if len(rows) != total:
        raise RuntimeError(f"Rationale cache coverage mismatch: {len(rows)} != {total}")
    return rows, {
        "manifest": file_identity(manifest_path),
        "contract_fingerprint": fingerprint,
        "questions": total,
        "shards": shard_count,
    }


def document_character_spans(user_prompt: str, document_texts: list[str]) -> list[tuple[int, int]]:
    marker = "Documents:\n"
    cursor = user_prompt.find(marker)
    if cursor < 0:
        raise RuntimeError("Anchored prompt has no Documents marker")
    cursor += len(marker)
    spans: list[tuple[int, int]] = []
    for text in document_texts:
        start = user_prompt.find(text, cursor)
        if start < 0:
            raise RuntimeError("Unable to align a Top-8 document to the anchored prompt")
        end = start + len(text)
        spans.append((start, end))
        cursor = end
    return spans


def encode_final_decision(
    tokenizer: Any,
    question: Any,
    rationale: str,
    max_model_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    document_texts = [" ".join(document.text.split()) for document in question.documents]
    row = {
        "question": question.question,
        "options": dict(question.options),
        "answer": question.gold_answers[0],
    }
    evidence = "\n\n".join(document_texts)
    user_prompt = build_anchored_user_prompt(row, evidence)
    chat_prompt = render_chat_prompt(tokenizer, user_prompt)
    full_prompt = chat_prompt + assistant_decision_prefix(rationale)
    if not full_prompt.endswith(FINAL_ANSWER_PREFIX):
        raise RuntimeError("Decision prompt does not end at the fixed choice anchor")
    encoded = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.int32)
    if input_ids.numel() > max_model_length:
        raise RuntimeError(
            f"Decision prompt exceeds Llama budget for {question.sample_id}: "
            f"{input_ids.numel()} > {max_model_length}"
        )
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    user_start = full_prompt.find(user_prompt)
    if user_start < 0:
        raise RuntimeError("Rendered chat prompt does not retain the anchored user prompt")
    token_document_ids = torch.full((len(offsets),), -1, dtype=torch.int8)
    for document_index, (start, end) in enumerate(
        document_character_spans(user_prompt, document_texts)
    ):
        absolute_start = user_start + start
        absolute_end = user_start + end
        matches = [
            token_index
            for token_index, (left, right) in enumerate(offsets)
            if right > absolute_start and left < absolute_end
        ]
        if not matches:
            raise RuntimeError(f"Document {document_index} has no aligned Llama tokens")
        token_document_ids[matches] = document_index
    final_text = tokenizer.decode([int(input_ids[-1])], skip_special_tokens=False)
    if "(" not in final_text:
        raise RuntimeError(
            f"Final choice anchor is not the last token for {question.sample_id}: {final_text!r}"
        )
    return input_ids, token_document_ids


def shard_paths(output_dir: Path, split: str, index: int) -> tuple[Path, Path]:
    root = output_dir / "feature_shards" / split / f"shard_{index:05d}"
    return root / "features.pt", root / "COMPLETE.json"


def valid_shard(data_path: Path, marker_path: Path, expected: int, fingerprint: str) -> bool:
    if not data_path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and marker.get("contract_fingerprint") == fingerprint
        and int(marker.get("question_count", -1)) == expected
        and int(marker.get("data_size_bytes", -1)) == data_path.stat().st_size
        and marker.get("data_sha256") == sha256_file(data_path)
    )


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.expected_documents != 8:
        raise ValueError("The current semantic annotations formally support exactly Top-8")
    if args.questions_per_shard <= 0 or args.semantic_batch_size <= 0:
        raise ValueError("Shard and batch sizes must be positive")
    semantic_model = args.semantic_model or DEFAULT_FILTERS[args.dataset]
    paths = {
        "candidates": args.candidate_root / args.dataset / "train/candidates_top8.jsonl",
        "labels": args.semantic_label_root / args.dataset / "codex_semantic_labels.jsonl",
        "splits": args.split_root / args.dataset / "sample_ids",
        "no_rag": args.no_rag_root / args.dataset / "train/no_rag_generations.jsonl",
    }
    for path in [*paths.values(), args.rationale_cache, semantic_model, args.llm_model]:
        if not path.exists():
            raise FileNotFoundError(path)

    data_sources = SemanticAttentionDataSources(
        dataset=args.dataset,
        candidates_path=paths["candidates"],
        semantic_labels_path=paths["labels"],
        split_ids_root=paths["splits"],
        no_rag_path=paths["no_rag"],
        expected_documents=args.expected_documents,
    )
    build_plan = make_semantic_attention_build_plan(data_sources)
    if args.dry_run:
        logging.info(
            "Semantic-attention preparation dry run: dataset=%s expected_questions=%s "
            "expected_pairs=%s split_ids=%s",
            args.dataset,
            build_plan.expected_questions,
            build_plan.expected_semantic_rows,
            dict(build_plan.split_id_counts),
        )
        return
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("--no-resume requires an empty or new feature output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index_path or (args.output_dir / "grouped_questions.sqlite")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_result = build_semantic_attention_index(
        data_sources,
        index_path,
        resume=args.resume,
        show_progress=True,
    )
    rationales, rationale_identity = load_rationale_cache(args.rationale_cache, args.dataset)
    source_contract = {
        "run_version": RUN_VERSION,
        "dataset": args.dataset,
        "grouped_source_fingerprint": index_result.manifest["source_fingerprint"],
        "rationale_cache": rationale_identity,
        "semantic_model_bundle": model_bundle_identity(semantic_model),
        "semantic_pooling_version": POOLING_VERSION,
        "semantic_max_input_length": args.semantic_max_input_length,
        "semantic_extraction_dtype": args.dtype,
        "llm_model_bundle": model_bundle_identity(args.llm_model),
        "llm_max_model_length": args.llm_max_model_length,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "expected_documents": args.expected_documents,
        "questions_per_shard": args.questions_per_shard,
        "max_questions_per_split": args.max_questions_per_split,
    }
    fingerprint = hashlib.sha256(
        json.dumps(source_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract_path = args.output_dir / "preparation_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != {**source_contract, "contract_fingerprint": fingerprint}:
            raise RuntimeError("Prepared-feature resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(contract_path, {**source_contract, "contract_fingerprint": fingerprint})

    datasets = {split: RAG2SemanticAttentionDataset(index_result.index_path, split) for split in ("train", "val", "test")}
    selected: dict[str, list[int]] = {}
    missing_rationales: list[str] = []
    for split, dataset in datasets.items():
        indices: list[int] = []
        iterator = tqdm(
            range(len(dataset)),
            total=len(dataset),
            desc=f"stage=select prepared questions ({split})",
            unit="question",
            dynamic_ncols=True,
            leave=False,
        )
        for index in iterator:
            sample_id = dataset[index].sample_id
            if sample_id not in rationales:
                missing_rationales.append(sample_id)
                continue
            indices.append(index)
            if args.max_questions_per_split > 0 and len(indices) >= args.max_questions_per_split:
                break
        selected[split] = indices
    if missing_rationales and not args.allow_partial_rationale_cache:
        raise RuntimeError(
            f"Unbiased rationale cache misses {len(missing_rationales)} indexed questions: "
            f"{missing_rationales[:5]}"
        )
    total_questions = sum(len(indices) for indices in selected.values())
    total_units = total_questions * (args.expected_documents + 1)
    completed_units = 0
    for split, indices in selected.items():
        expected_shard_count = math.ceil(len(indices) / args.questions_per_shard)
        expected_roots = {
            args.output_dir / "feature_shards" / split / f"shard_{index:05d}"
            for index in range(expected_shard_count)
        }
        actual_roots = set((args.output_dir / "feature_shards" / split).glob("shard_*"))
        extras = actual_roots - expected_roots
        if extras:
            raise RuntimeError(
                f"Unexpected stale {split} feature shards: {sorted(map(str, extras))[:5]}"
            )
        for shard_index, shard_indices in enumerate(chunks(indices, args.questions_per_shard)):
            data_path, marker_path = shard_paths(args.output_dir, split, shard_index)
            if args.resume and valid_shard(data_path, marker_path, len(shard_indices), fingerprint):
                completed_units += len(shard_indices) * (args.expected_documents + 1)
    logging.info(
        "Semantic-attention preparation: dataset=%s selected=%s cached_units=%d/%d",
        args.dataset,
        {split: len(indices) for split, indices in selected.items()},
        completed_units,
        total_units,
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    extractor = None
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Llama document-span alignment requires a fast tokenizer")
    progress = PipelineProgress(
        overall_total=total_units,
        overall_initial=completed_units,
        desc=f"PrepareSemanticGate:{args.dataset}",
    )
    split_summary: dict[str, int] = {}
    try:
        for split_index, (split, indices) in enumerate(selected.items(), start=1):
            progress.set_stage(
                f"{split_index}/3 semantic features + final-choice token map ({split})",
                total=len(indices) * (args.expected_documents + 1),
                initial=sum(
                    len(shard_indices) * (args.expected_documents + 1)
                    for shard_index, shard_indices in enumerate(chunks(indices, args.questions_per_shard))
                    if args.resume
                    and valid_shard(
                        *shard_paths(args.output_dir, split, shard_index),
                        len(shard_indices),
                        fingerprint,
                    )
                ),
            )
            written = 0
            for shard_index, shard_indices in enumerate(chunks(indices, args.questions_per_shard)):
                data_path, marker_path = shard_paths(args.output_dir, split, shard_index)
                if args.resume and valid_shard(data_path, marker_path, len(shard_indices), fingerprint):
                    written += len(shard_indices)
                    continue
                questions = [datasets[split][index] for index in shard_indices]
                prompts = [
                    build_official_filter_input(
                        question=question.question,
                        options=format_options(question.options),
                        evidence=" ".join(document.text.split()),
                    )
                    for question in questions
                    for document in question.documents
                ]
                if extractor is None:
                    extractor = FrozenSemanticEvidenceEncoder(
                        semantic_model,
                        device=args.device,
                        dtype=dtype,
                        max_input_length=args.semantic_max_input_length,
                        batch_size=args.semantic_batch_size,
                    )
                encoded = extractor.encode_prompts(prompts, progress_callback=progress.update)
                expected_pairs = len(questions) * args.expected_documents
                if encoded.features.shape != (expected_pairs, extractor.hidden_size):
                    raise RuntimeError("Frozen semantic feature shape mismatch")
                input_ids: list[torch.Tensor] = []
                token_document_ids: list[torch.Tensor] = []
                semantic_targets: list[list[int]] = []
                semantic_masks: list[list[bool]] = []
                sample_ids: list[str] = []
                pair_ids: list[list[str]] = []
                gold_options: list[int] = []
                baseline_options: list[int] = []
                no_rag_correct: list[bool] = []
                semantic_class_ids: list[list[int]] = []
                for question in questions:
                    rationale_row = rationales[question.sample_id]
                    expected_ids = [document.pair_id for document in question.documents]
                    if list(rationale_row.get("document_pair_ids") or []) != expected_ids:
                        raise RuntimeError(f"Rationale/candidate document mismatch: {question.sample_id}")
                    ids, mapping = encode_final_decision(
                        tokenizer,
                        question,
                        str(rationale_row["rationale"]),
                        args.llm_max_model_length,
                    )
                    input_ids.append(ids)
                    token_document_ids.append(mapping)
                    sample_ids.append(question.sample_id)
                    pair_ids.append(expected_ids)
                    semantic_targets.append(
                        [
                            -1 if document.semantic_support_target is None else int(document.semantic_support_target)
                            for document in question.documents
                        ]
                    )
                    semantic_masks.append([document.semantic_loss_mask for document in question.documents])
                    semantic_class_ids.append(
                        [
                            -1 if document.semantic_class_id is None else int(document.semantic_class_id)
                            for document in question.documents
                        ]
                    )
                    gold_options.append("ABCD".index(question.gold_answers[0]))
                    baseline_answer = str(rationale_row.get("answer") or "").strip().upper()
                    if baseline_answer not in "ABCD":
                        raise RuntimeError(f"Invalid cached baseline answer for {question.sample_id}")
                    baseline_options.append("ABCD".index(baseline_answer))
                    expected_correct = baseline_answer in question.gold_answers
                    if bool(rationale_row.get("answer_correct")) != expected_correct:
                        raise RuntimeError(
                            f"Cached Top-8 answer correctness mismatch for {question.sample_id}"
                        )
                    if question.no_rag.answer_correct is None:
                        raise RuntimeError(f"No-RAG correctness missing for {question.sample_id}")
                    no_rag_correct.append(bool(question.no_rag.answer_correct))
                    progress.update(1)
                payload = {
                    "run_version": RUN_VERSION,
                    "contract_fingerprint": fingerprint,
                    "dataset": args.dataset,
                    "split": split,
                    "sample_ids": sample_ids,
                    "pair_ids": pair_ids,
                    "semantic_features": encoded.features.reshape(
                        len(questions), args.expected_documents, -1
                    ),
                    "semantic_margins": encoded.margins.reshape(
                        len(questions), args.expected_documents
                    ),
                    "semantic_logits": encoded.logits.reshape(
                        len(questions), args.expected_documents, 2
                    ),
                    "semantic_targets": torch.tensor(semantic_targets, dtype=torch.int8),
                    "semantic_masks": torch.tensor(semantic_masks, dtype=torch.bool),
                    "semantic_class_ids": torch.tensor(semantic_class_ids, dtype=torch.int8),
                    "gold_options": torch.tensor(gold_options, dtype=torch.int8),
                    "baseline_options": torch.tensor(baseline_options, dtype=torch.int8),
                    "no_rag_correct": torch.tensor(no_rag_correct, dtype=torch.bool),
                    "input_ids": input_ids,
                    "token_document_ids": token_document_ids,
                }
                atomic_torch_save(data_path, payload)
                atomic_write_json(
                    marker_path,
                    {
                        "run_version": RUN_VERSION,
                        "contract_fingerprint": fingerprint,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "question_count": len(questions),
                        "document_count": expected_pairs,
                        "data_size_bytes": data_path.stat().st_size,
                        "data_sha256": sha256_file(data_path),
                        "sample_ids_sha256": hashlib.sha256(
                            "\n".join(sample_ids).encode()
                        ).hexdigest(),
                    },
                )
                written += len(questions)
                progress.set_detail(f"split={split} shard={shard_index + 1}")
            split_summary[split] = written
        atomic_write_json(
            args.output_dir / "preparation_manifest.json",
            {
                **source_contract,
                "contract_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "split_questions": split_summary,
                "feature_hidden_size": int(extractor.hidden_size if extractor is not None else 1024),
            },
        )
        logging.info("Semantic-attention training features complete: %s", args.output_dir)
    finally:
        progress.close()
        for dataset in datasets.values():
            dataset.close()
        if extractor is not None:
            extractor.close()
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare unbiased fixed-rationale conditional-removal attribution teachers.

The source cache supplies an unbiased Top-K rationale replay prompt and exact
document token spans.  This script freezes the target Llama, physically removes
each document (and all documents), and stores:

* base-2 JSD conditional-removal targets from the final-choice distribution;
* target-Llama document-span means at selected layers;
* a global pre-rationale prefix state at those same layers.

The cached rationale is held fixed across interventions.  Consequently these
targets describe final-choice sensitivity conditional on that reasoning path;
they are not full free-regeneration reasoning attribution.  No learned
semantic gate or attention bias is applied while producing the teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import CHOICES  # noqa: E402


RUN_VERSION = "rag2_target_llm_conditional_removal_teacher_v1"
SOURCE_RUN_VERSION = "rag2_semantic_attention_training_features_v1"
TEACHER_MODE = "unbiased_fixed_rationale_final_choice_conditional_removal_v1"
DEFAULT_LLM = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("medmcqa", "medqa"), required=True)
    parser.add_argument("--source-feature-dir", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=(20, 28))
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--max-questions-per-split", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--expected-document-count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(encoded.encode("utf-8"))


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


def model_bundle_identity(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / name
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        if (root / name).is_file()
    ]
    paths.extend(sorted(root.glob("*.safetensors")))
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            **({"sha256": sha256_file(path)} if path.stat().st_size < 16 * 1024 * 1024 else {}),
        }
        for path in paths
    ]


def source_shards(feature_dir: Path, split: str) -> list[Path]:
    paths = sorted((feature_dir / "feature_shards" / split).glob("shard_*/features.pt"))
    if not paths:
        raise FileNotFoundError(f"No source feature shards for {split}: {feature_dir}")
    return paths


def load_source_shard(
    path: Path,
    *,
    dataset: str,
    split: str,
    fingerprint: str,
    expected_documents: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("run_version") != SOURCE_RUN_VERSION
        or payload.get("dataset") != dataset
        or payload.get("split") != split
        or payload.get("contract_fingerprint") != fingerprint
    ):
        raise ValueError(f"Unsupported source shard: {path}")
    required = (
        "sample_ids",
        "pair_ids",
        "input_ids",
        "token_document_ids",
        "assistant_query_starts",
    )
    if any(name not in payload for name in required):
        raise ValueError(f"Source shard misses required prompt fields: {path}")
    count = len(payload["sample_ids"])
    if any(len(payload[name]) != count for name in required):
        raise ValueError(f"Source shard row count mismatch: {path}")
    for row, pair_ids in enumerate(payload["pair_ids"]):
        if len(pair_ids) != expected_documents:
            raise ValueError(
                f"Source row does not contain {expected_documents} documents: {path}:{row}"
            )
    return payload


def select_ids(
    feature_dir: Path,
    splits: list[str],
    *,
    dataset: str,
    fingerprint: str,
    expected_documents: int,
    maximum: int,
    seed: int,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for split_offset, split in enumerate(splits):
        values: list[str] = []
        for path in source_shards(feature_dir, split):
            payload = load_source_shard(
                path,
                dataset=dataset,
                split=split,
                fingerprint=fingerprint,
                expected_documents=expected_documents,
            )
            values.extend(str(value) for value in payload["sample_ids"])
        if len(values) != len(set(values)):
            raise RuntimeError(f"Duplicate sample IDs in source split {split}")
        if maximum > 0 and len(values) > maximum:
            values = random.Random(seed + split_offset).sample(values, maximum)
        selected[split] = sorted(values)
    return selected


def output_row_path(output_dir: Path, split: str, sample_id: str) -> Path:
    digest = sha256_bytes(sample_id.encode("utf-8"))[:24]
    return output_dir / "rows" / split / f"{digest}.pt"


def valid_cached_row(path: Path, *, sample_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("run_version") == RUN_VERSION
        and payload.get("sample_id") == sample_id
        and payload.get("contract_fingerprint") == fingerprint
    )


def jensen_shannon_divergence(reference: torch.Tensor, alternative: torch.Tensor) -> float:
    reference = reference.double().clamp_min(1e-30)
    alternative = alternative.double().clamp_min(1e-30)
    midpoint = 0.5 * (reference + alternative)
    divergence = 0.5 * (
        torch.sum(reference * torch.log2(reference / midpoint))
        + torch.sum(alternative * torch.log2(alternative / midpoint))
    )
    return float(divergence.clamp(min=0.0, max=1.0).item())


def build_conditional_removal_batch(
    input_ids: torch.Tensor,
    token_document_ids: torch.Tensor,
    *,
    document_count: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Return full, repeat, empty, then one removal per document."""

    ids = input_ids.long().cpu()
    mapping = token_document_ids.long().cpu()
    if ids.ndim != 1 or mapping.shape != ids.shape:
        raise ValueError("input_ids and token_document_ids must be aligned vectors")
    variants: list[tuple[torch.Tensor, torch.Tensor]] = [(ids, mapping), (ids, mapping)]
    keep_empty = mapping.lt(0)
    variants.append((ids[keep_empty], mapping[keep_empty]))
    for document_index in range(document_count):
        keep = mapping.ne(document_index)
        if int((~keep).sum().item()) <= 0:
            raise RuntimeError(f"Document slot {document_index} has no mapped tokens")
        variants.append((ids[keep], mapping[keep]))
    maximum = max(int(row_ids.numel()) for row_ids, _ in variants)
    padded_ids = torch.full((len(variants), maximum), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros_like(padded_ids)
    padded_mapping = torch.full_like(padded_ids, -1)
    for row, (row_ids, row_mapping) in enumerate(variants):
        length = int(row_ids.numel())
        left = maximum - length
        padded_ids[row, left:] = row_ids
        attention_mask[row, left:] = 1
        padded_mapping[row, left:] = row_mapping
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_document_ids": padded_mapping,
    }


class SelectedLayerCapture:
    """Capture only the full-context row from selected decoder layers."""

    def __init__(self, model: Any, layers: list[int]) -> None:
        decoder_layers = model.model.layers
        if any(index < 0 or index >= len(decoder_layers) for index in layers):
            raise ValueError(f"Selected layer outside model range 0..{len(decoder_layers) - 1}")
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [
            decoder_layers[index].register_forward_hook(self._hook(index)) for index in layers
        ]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.values[index] = hidden[0].detach().to(device="cpu", dtype=torch.float32)

        return capture

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def extract_one(
    *,
    payload: dict[str, Any],
    row_index: int,
    model: Any,
    capture: SelectedLayerCapture,
    selected_layers: list[int],
    choice_token_ids: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    document_count: int,
    dataset: str,
    split: str,
    fingerprint: str,
) -> dict[str, Any]:
    input_ids = payload["input_ids"][row_index]
    mapping = payload["token_document_ids"][row_index]
    assistant_start = int(payload["assistant_query_starts"][row_index])
    if not 1 <= assistant_start < int(input_ids.numel()):
        raise ValueError("assistant_query_start is outside the replay prompt")
    batch = build_conditional_removal_batch(
        input_ids,
        mapping,
        document_count=document_count,
        pad_token_id=pad_token_id,
    )
    capture.clear()
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1].float().index_select(1, choice_token_ids).cpu()
    del outputs
    if set(capture.values) != set(selected_layers):
        raise RuntimeError("Not all requested target-Llama layers were captured")
    prefix_mapping = mapping[:assistant_start]
    document_features: list[torch.Tensor] = []
    document_lengths: list[int] = []
    for document_index in range(document_count):
        token_mask = prefix_mapping.eq(document_index)
        token_count = int(token_mask.sum().item())
        if token_count <= 0:
            raise RuntimeError(f"Document {document_index} has no pre-rationale tokens")
        document_lengths.append(token_count)
        document_features.append(
            torch.stack(
                [capture.values[layer][:assistant_start][token_mask].mean(dim=0) for layer in selected_layers],
                dim=0,
            )
        )
    global_features = torch.stack(
        [capture.values[layer][assistant_start - 1] for layer in selected_layers],
        dim=0,
    )
    probabilities = torch.softmax(logits, dim=-1)
    full = probabilities[0]
    repeated = probabilities[1]
    empty = probabilities[2]
    loo = probabilities[3:]
    loo_jsd = [jensen_shannon_divergence(full, row) for row in loo]
    sample_id = str(payload["sample_ids"][row_index])
    return {
        "run_version": RUN_VERSION,
        "contract_fingerprint": fingerprint,
        "teacher_mode": TEACHER_MODE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "dataset": dataset,
        "split": split,
        "pair_ids": [str(value) for value in payload["pair_ids"][row_index]],
        "document_count": document_count,
        "selected_layers": selected_layers,
        "document_features": torch.stack(document_features).to(dtype=torch.bfloat16),
        "global_features": global_features.to(dtype=torch.bfloat16),
        "document_lengths": torch.tensor(document_lengths, dtype=torch.int32),
        "rerank_positions": torch.arange(document_count, dtype=torch.int32),
        "loo_jsd": torch.tensor(loo_jsd, dtype=torch.float32),
        "total_loo_jsd": float(sum(loo_jsd)),
        "set_shift_jsd": jensen_shannon_divergence(full, empty),
        "repeat_noise_jsd": jensen_shannon_divergence(full, repeated),
        "full_choice_probabilities": full.to(torch.float32),
        "empty_choice_probabilities": empty.to(torch.float32),
        "loo_choice_probabilities": loo.to(torch.float32),
        "target_feature_dtype": str(dtype).removeprefix("torch."),
        "fixed_rationale": True,
        "unbiased_target_llm": True,
    }


def iter_selected_rows(
    feature_dir: Path,
    split: str,
    selected: set[str],
    *,
    dataset: str,
    fingerprint: str,
    expected_documents: int,
) -> Iterable[tuple[dict[str, Any], int, str, int, int]]:
    paths = source_shards(feature_dir, split)
    for shard_index, path in enumerate(paths, start=1):
        payload = load_source_shard(
            path,
            dataset=dataset,
            split=split,
            fingerprint=fingerprint,
            expected_documents=expected_documents,
        )
        for row_index, sample_id_value in enumerate(payload["sample_ids"]):
            sample_id = str(sample_id_value)
            if sample_id in selected:
                yield payload, row_index, sample_id, shard_index, len(paths)


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_feature_dir.is_dir():
        raise FileNotFoundError(args.source_feature_dir)
    if not args.llm_model.is_dir():
        raise FileNotFoundError(args.llm_model)
    if args.max_questions_per_split < 0 or args.expected_document_count <= 0:
        raise ValueError("sample maximum must be non-negative and document count positive")
    if not args.layers or len(args.layers) != len(set(args.layers)):
        raise ValueError("--layers must contain unique layer indices")
    if any(split not in {"train", "val", "test"} for split in args.splits):
        raise ValueError("--splits only supports train, val, test")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    source_manifest_path = args.source_feature_dir / "preparation_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset") != args.dataset:
        raise ValueError("Source feature dataset differs from --dataset")
    if int(source_manifest.get("expected_documents", -1)) != args.expected_document_count:
        raise ValueError(
            "The rationale/source cache was not generated for the requested K. "
            "Generate a K-specific rationale cache instead of slicing another K's teacher."
        )
    source_fingerprint = str(source_manifest["contract_fingerprint"])
    splits = list(dict.fromkeys(args.splits))
    selected = select_ids(
        args.source_feature_dir,
        splits,
        dataset=args.dataset,
        fingerprint=source_fingerprint,
        expected_documents=args.expected_document_count,
        maximum=args.max_questions_per_split,
        seed=args.sample_seed,
    )
    model_identity = model_bundle_identity(args.llm_model)
    run_contract = {
        "run_version": RUN_VERSION,
        "teacher_mode": TEACHER_MODE,
        "dataset": args.dataset,
        "source_feature_dir": str(args.source_feature_dir.resolve()),
        "source_contract_fingerprint": source_fingerprint,
        "llm_model": str(args.llm_model.resolve()),
        "llm_model_bundle": model_identity,
        "layers": list(args.layers),
        "splits": splits,
        "selected_sample_ids": selected,
        "sample_seed": args.sample_seed,
        "document_count": args.expected_document_count,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "attention_bias": "none",
        "fixed_rationale": True,
    }
    fingerprint = canonical_hash(run_contract)
    run_contract["contract_fingerprint"] = fingerprint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError("Attribution-teacher resume contract mismatch; use a new output directory")
    else:
        atomic_write_json(contract_path, run_contract)

    total = sum(len(values) for values in selected.values())
    cached_by_split: dict[str, set[str]] = {}
    for split, ids in selected.items():
        cached_by_split[split] = {
            sample_id
            for sample_id in ids
            if valid_cached_row(
                output_row_path(args.output_dir, split, sample_id),
                sample_id=sample_id,
                fingerprint=fingerprint,
            )
        }
    cached = sum(len(values) for values in cached_by_split.values())
    logging.info(
        "Target-LLM attribution preparation plan: split_counts=%s cached=%d remaining=%d "
        "K=%d layers=%s interventions=%d teacher=%s",
        {name: len(values) for name, values in selected.items()},
        cached,
        total - cached,
        args.expected_document_count,
        args.layers,
        (total - cached) * (args.expected_document_count + 3),
        TEACHER_MODE,
    )
    if args.plan_only:
        return

    progress = PipelineProgress(
        overall_total=2 * total,
        overall_initial=cached,
        desc=f"TargetLLMAttributionPrepare:{args.dataset}",
    )
    capture: SelectedLayerCapture | None = None
    try:
        if cached < total:
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model, local_files_only=True, use_fast=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            choice_ids: list[int] = []
            for label in CHOICES:
                token_ids = tokenizer.encode(label, add_special_tokens=False)
                if len(token_ids) != 1:
                    raise RuntimeError(f"Choice label {label} is not one token: {token_ids}")
                choice_ids.append(int(token_ids[0]))
            device = torch.device(args.device)
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            logging.info("Loading unbiased frozen target Llama on %s: %s", device, args.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                local_files_only=True,
                dtype=dtype,
                attn_implementation=args.attn_implementation,
            ).to(device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            capture = SelectedLayerCapture(model, list(args.layers))
            choice_token_ids = torch.tensor(choice_ids, dtype=torch.long, device=device)
            progress.set_stage(
                "1/2 unbiased Llama span features + conditional-removal teacher",
                total=total,
                initial=cached,
            )
            for split in splits:
                pending = set(selected[split]) - cached_by_split[split]
                for payload, row_index, sample_id, shard_index, shard_count in iter_selected_rows(
                    args.source_feature_dir,
                    split,
                    pending,
                    dataset=args.dataset,
                    fingerprint=source_fingerprint,
                    expected_documents=args.expected_document_count,
                ):
                    row = extract_one(
                        payload=payload,
                        row_index=row_index,
                        model=model,
                        capture=capture,
                        selected_layers=list(args.layers),
                        choice_token_ids=choice_token_ids,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=device,
                        dtype=dtype,
                        document_count=args.expected_document_count,
                        dataset=args.dataset,
                        split=split,
                        fingerprint=fingerprint,
                    )
                    atomic_torch_save(output_row_path(args.output_dir, split, sample_id), row)
                    cached_by_split[split].add(sample_id)
                    progress.update(1)
                    progress.set_detail(
                        f"split={split} shard={shard_index}/{shard_count} sample={sample_id}"
                    )
            missing = {
                split: sorted(set(selected[split]) - cached_by_split[split])
                for split in splits
                if set(selected[split]) - cached_by_split[split]
            }
            if missing:
                raise RuntimeError(f"Selected source rows were not found: {missing}")
            capture.close()
            capture = None
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            progress.set_stage(
                "1/2 unbiased Llama span features + conditional-removal teacher",
                total=total,
                initial=total,
            )

        progress.set_stage("2/2 validate cached rows and publish manifest", total=total)
        signal_totals: list[float] = []
        set_shifts: list[float] = []
        repeat_noise: list[float] = []
        for split in splits:
            for sample_id in selected[split]:
                path = output_row_path(args.output_dir, split, sample_id)
                if not valid_cached_row(path, sample_id=sample_id, fingerprint=fingerprint):
                    raise RuntimeError(f"Invalid prepared attribution row: {path}")
                row = torch.load(path, map_location="cpu", weights_only=False)
                signal_totals.append(float(row["total_loo_jsd"]))
                set_shifts.append(float(row["set_shift_jsd"]))
                repeat_noise.append(float(row["repeat_noise_jsd"]))
                progress.update(1)
        manifest = {
            **run_contract,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "split_questions": {name: len(values) for name, values in selected.items()},
            "target_hidden_size": int(row["document_features"].shape[-1]),
            "selected_layer_count": len(args.layers),
            "summary": {
                "mean_total_loo_jsd": sum(signal_totals) / max(1, len(signal_totals)),
                "mean_set_shift_jsd": sum(set_shifts) / max(1, len(set_shifts)),
                "max_repeat_noise_jsd": max(repeat_noise, default=0.0),
            },
        }
        atomic_write_json(args.output_dir / "preparation_manifest.json", manifest)
        logging.info("Target-LLM attribution teachers complete: %s", args.output_dir)
    finally:
        if capture is not None:
            capture.close()
        progress.close()


if __name__ == "__main__":
    main()

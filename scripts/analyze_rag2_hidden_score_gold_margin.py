#!/usr/bin/env python3
"""Validate hidden utility scores against exact direct-choice behavior.

This script reuses the external-test Top-32 candidate cache, cached layer-28
``h0``/``hD`` tensors, and materialized hidden projection scores.  The old
cache intentionally did not store next-choice logits, so exact behavioral
margins cannot be recovered from ``h0``/``hD`` alone.  In ``materialize``
mode, this script computes and caches only the missing A/B/C/D logits:

* no-document logits and the gold-direction gradient norm once per question;
* with-document logits once per cached question-document pair.

It then compares:

* ``projection_score = (hD-h0) dot c_hat``;
* ``linearized_gold_logprob_delta = projection_score * ||c||``;
* exact four-choice gold log-probability change; and
* exact gold-vs-best-alternative margin change.

All expensive outputs are sharded, atomic, and resumable.  Retrieval,
embedding, reranking, rationale generation, and h0/hD extraction are never
repeated.
"""

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
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extract_rag2_preanswer_hidden_pilot import (  # noqa: E402
    CHOICES,
    PROMPT_VERSION,
    FeatureExtractor,
    choice_index,
    pad_token_sequences,
)
from materialize_rag2_external_hidden_oracle_labels import (  # noqa: E402
    FEATURE_CACHE_TYPE,
    FEATURE_CACHE_VERSION,
    stable_id,
)


LOGIT_CACHE_TYPE = "rag2_preanswer_exact_choice_logits"
LOGIT_CACHE_VERSION = "rag2_preanswer_exact_choice_logits_v1"
ANALYSIS_VERSION = "rag2_hidden_score_gold_margin_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "materialize", "analyze"], default="all")
    parser.add_argument("--candidates-path", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--hidden-labels-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--choice-logit-cache-dir",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/exact_choice_logits_cache.",
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--question-batch-size", type=int, default=32)
    parser.add_argument("--document-batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every cache contract and report missing logit shards without loading the LLM.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_save_tensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        str(temporary),
        metadata=metadata,
    )
    os.replace(temporary, path)


def sha256_file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def normalize_gold(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is None and isinstance(row.get("answers"), list) and row["answers"]:
        answer = row["answers"][0]
    if isinstance(answer, int) and 0 <= answer < 4:
        answer = CHOICES[answer]
    value = str(answer or "").strip().upper()
    if value not in CHOICES:
        raise ValueError(f"Invalid gold answer for {row.get('key')}: {answer!r}")
    return value


def load_manifest(path: Path, expected_type: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if expected_type is not None and value.get("type") != expected_type:
        raise RuntimeError(
            f"Manifest type mismatch: {path} has {value.get('type')!r}, expected {expected_type!r}"
        )
    return value


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    candidates = list(rows(args.candidates_path))
    feature_manifest = load_manifest(args.feature_cache_dir / "manifest.json", FEATURE_CACHE_TYPE)
    settings = feature_manifest.get("settings") or {}
    pair_count = sum(len(row.get("candidate_documents") or []) for row in candidates)
    expected = {
        "version": FEATURE_CACHE_VERSION,
        "questions": len(candidates),
        "documents": pair_count,
        "prompt_version": PROMPT_VERSION,
        "hidden_layer": int(args.layer),
        "hidden_max_input_tokens": int(args.max_input_tokens),
        "hidden_dtype": args.dtype,
        "hidden_attn_implementation": args.attn_implementation,
        "model_path": str(args.model_name_or_path.resolve()),
    }
    actual = {
        "version": feature_manifest.get("version"),
        "questions": feature_manifest.get("questions"),
        "documents": feature_manifest.get("documents"),
        "prompt_version": settings.get("prompt_version"),
        "hidden_layer": settings.get("hidden_layer"),
        "hidden_max_input_tokens": settings.get("hidden_max_input_tokens"),
        "hidden_dtype": settings.get("hidden_dtype"),
        "hidden_attn_implementation": settings.get("hidden_attn_implementation"),
        "model_path": str(Path((settings.get("state_model") or {}).get("path", "")).resolve()),
    }
    mismatches = [
        f"{name}: cache={actual.get(name)!r} requested={value!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise RuntimeError("Feature-cache contract mismatch:\n- " + "\n- ".join(mismatches))

    hidden_manifest_path = args.hidden_labels_path.parent / "manifest.json"
    hidden_manifest = load_manifest(hidden_manifest_path, "rag2_external_hidden_oracle_labels")
    if int(hidden_manifest.get("pairs", -1)) != pair_count:
        raise RuntimeError(
            f"Hidden-label pair count mismatch: {hidden_manifest.get('pairs')} != {pair_count}"
        )
    if int(hidden_manifest.get("questions", -1)) != len(candidates):
        raise RuntimeError("Hidden-label question count mismatch")
    if int(hidden_manifest.get("layer", -1)) != int(args.layer):
        raise RuntimeError("Hidden-label layer mismatch")
    if Path(hidden_manifest.get("feature_cache_dir", "")).resolve() != args.feature_cache_dir.resolve():
        raise RuntimeError("Hidden labels were not computed from the requested h0/hD cache")
    actual_hidden_rows = sum(1 for _ in rows(args.hidden_labels_path))
    if actual_hidden_rows != pair_count:
        raise RuntimeError(
            f"Hidden-label file is incomplete: rows={actual_hidden_rows} expected={pair_count}"
        )

    model_identity = settings.get("state_model") or {}
    model_root = args.model_name_or_path.resolve()
    for entry in [*(model_identity.get("files") or []), *(model_identity.get("weight_shards") or [])]:
        path = model_root / str(entry.get("name") or "")
        if not path.is_file():
            raise FileNotFoundError(f"Cached-feature model file is missing: {path}")
        stat = path.stat()
        if stat.st_size != int(entry.get("size", -1)) or stat.st_mtime_ns != int(
            entry.get("mtime_ns", -1)
        ):
            raise RuntimeError(f"Model file changed after h0/hD caching: {path}")

    shard_paths = sorted((args.feature_cache_dir / "shards").glob("*.json"))
    if len(shard_paths) != int(feature_manifest.get("shards", -1)):
        raise RuntimeError(
            f"Feature shard count mismatch: files={len(shard_paths)} manifest={feature_manifest.get('shards')}"
        )
    return candidates, shard_paths, feature_manifest


def cache_settings(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model_name_or_path.resolve()
    model_files = []
    names = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    names.extend(path.name for path in sorted(model_path.glob("*.safetensors")))
    for name in names:
        path = model_path / name
        if path.is_file():
            model_files.append(sha256_file_identity(path))
    return {
        "version": LOGIT_CACHE_VERSION,
        "candidates": sha256_file_identity(args.candidates_path),
        "feature_cache_dir": str(args.feature_cache_dir.resolve()),
        "hidden_labels": sha256_file_identity(args.hidden_labels_path),
        "model_path": str(model_path),
        "model_files": model_files,
        "prompt_version": PROMPT_VERSION,
        "layer": int(args.layer),
        "max_input_tokens": int(args.max_input_tokens),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }


def settings_fingerprint(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expected_shard_metadata(
    source_metadata: dict[str, Any], fingerprint: str
) -> dict[str, Any]:
    return {
        "type": LOGIT_CACHE_TYPE,
        "version": LOGIT_CACHE_VERSION,
        "settings_fingerprint": fingerprint,
        "name": source_metadata["name"],
        "route": source_metadata["route"],
        "sample_indices": source_metadata["sample_indices"],
        "sample_keys": source_metadata["sample_keys"],
        "document_keys": source_metadata["document_keys"],
        "questions": int(source_metadata["questions"]),
        "documents": int(source_metadata["documents"]),
    }


def logit_shard_valid(cache_dir: Path, expected: dict[str, Any]) -> bool:
    metadata_path = cache_dir / "shards" / f"{expected['name']}.json"
    tensor_path = metadata_path.with_suffix(".safetensors")
    if not metadata_path.is_file() or not tensor_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    try:
        tensors = load_file(str(tensor_path), device="cpu")
    except Exception:
        return False
    expected_shapes = {
        "no_document_choice_logits": (expected["questions"], 4),
        "with_document_choice_logits": (expected["documents"], 4),
        "gold_direction_norm": (expected["questions"],),
        "document_offsets": (expected["questions"] + 1,),
    }
    return set(tensors) == set(expected_shapes) and all(
        tuple(tensors[name].shape) == shape for name, shape in expected_shapes.items()
    )


def atomic_write_logit_shard(
    cache_dir: Path,
    expected: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> None:
    shard_dir = cache_dir / "shards"
    tensor_path = shard_dir / f"{expected['name']}.safetensors"
    metadata_path = shard_dir / f"{expected['name']}.json"
    atomic_save_tensors(
        tensor_path,
        tensors,
        {
            "type": LOGIT_CACHE_TYPE,
            "version": LOGIT_CACHE_VERSION,
            "settings_fingerprint": str(expected["settings_fingerprint"]),
        },
    )
    metadata = dict(expected)
    metadata.update(
        {
            "created_at": utc_now(),
            "tensor_path": tensor_path.name,
            "stored_tensors": sorted(tensors),
        }
    )
    atomic_write_json(metadata_path, metadata)


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def exact_choice_logits(extractor: FeatureExtractor, sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    input_ids, attention_mask, position_ids = pad_token_sequences(
        sequences, extractor.tokenizer.pad_token_id, extractor.device
    )
    with torch.inference_mode():
        outputs = extractor.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = extractor.model.lm_head(outputs.last_hidden_state[:, -1, :])
        choice_logits = logits.index_select(dim=-1, index=extractor.choice_ids_tensor).float().cpu()
    return choice_logits


def adaptive_exact_logits(
    extractor: FeatureExtractor,
    sequences: Sequence[Sequence[int]],
    *,
    description: str,
) -> torch.Tensor:
    oom = False
    try:
        return exact_choice_logits(extractor, sequences)
    except BaseException as error:
        if not is_oom(error):
            raise
        # Leave the except block before retrying.  A live exception traceback
        # retains the failed forward's CUDA tensors; recursively retrying from
        # inside this block therefore makes each smaller batch inherit all
        # previous failed allocations.
        oom = True
    if not oom:  # pragma: no cover - defensive; the try either returns or sets oom
        raise RuntimeError("unreachable adaptive exact-logit state")
    extractor.model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    if len(sequences) <= 1:
        raise RuntimeError(
            f"CUDA OOM for one exact-logit prompt: {description} tokens={len(sequences[0])}"
        ) from None
    logging.warning(
        "%s OOM for batch=%s max_tokens=%s; retrying as two temporary micro-batches",
        description,
        len(sequences),
        max(map(len, sequences)),
    )
    midpoint = len(sequences) // 2
    left = adaptive_exact_logits(extractor, sequences[:midpoint], description=description)
    right = adaptive_exact_logits(extractor, sequences[midpoint:], description=description)
    return torch.cat([left, right], dim=0)


def adaptive_question_features(
    extractor: FeatureExtractor,
    sequences: Sequence[Sequence[int]],
    gold_indices: Sequence[int],
    *,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    oom = False
    try:
        features = extractor.no_document_features(sequences, gold_indices)
        logits = features.choice_logits.float()
        c_norm = features.c_norm[:, 0].float()
        del features
        return logits, c_norm
    except BaseException as error:
        if not is_oom(error):
            raise
        # Retry only after the exception traceback (and its partial autograd
        # graph) has gone out of scope.  This is essential for a long outlier
        # prompt that OOMs an otherwise safe batch.
        oom = True
    if not oom:  # pragma: no cover - defensive
        raise RuntimeError("unreachable adaptive question-feature state")
    extractor.model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    if len(sequences) <= 1:
        raise RuntimeError(
            f"CUDA OOM for one no-document prompt: {description} tokens={len(sequences[0])}"
        ) from None
    logging.warning(
        "%s OOM for batch=%s max_tokens=%s; retrying as two temporary micro-batches",
        description,
        len(sequences),
        max(map(len, sequences)),
    )
    midpoint = len(sequences) // 2
    left = adaptive_question_features(
        extractor, sequences[:midpoint], gold_indices[:midpoint], description=description
    )
    right = adaptive_question_features(
        extractor, sequences[midpoint:], gold_indices[midpoint:], description=description
    )
    return torch.cat([left[0], right[0]], dim=0), torch.cat([left[1], right[1]], dim=0)


def candidate_documents(row: dict[str, Any]) -> list[dict[str, Any]]:
    documents = list(row.get("candidate_documents") or [])
    if not documents:
        raise ValueError(f"No candidate documents for {row.get('key')}")
    return documents


def validate_shard_candidates(
    source_metadata: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    indices = [int(value) for value in source_metadata["sample_indices"]]
    shard_rows = [candidates[index] for index in indices]
    if [str(row.get("key")) for row in shard_rows] != list(source_metadata["sample_keys"]):
        raise RuntimeError(f"Sample order mismatch for {source_metadata['name']}")
    for position, row in enumerate(shard_rows):
        documents = candidate_documents(row)
        actual = [[str(doc.get("db_id") or ""), int(doc.get("local_id", -1))] for doc in documents]
        if actual != source_metadata["document_keys"][position]:
            raise RuntimeError(
                f"Document identity/order mismatch for {source_metadata['name']}:{row.get('key')}"
            )
    return shard_rows


def materialize_logits(
    args: argparse.Namespace,
    candidates: Sequence[dict[str, Any]],
    feature_shards: Sequence[Path],
    feature_manifest: dict[str, Any],
    cache_dir: Path,
) -> None:
    settings = cache_settings(args)
    fingerprint = settings_fingerprint(settings)
    manifest = {
        "type": LOGIT_CACHE_TYPE,
        "version": LOGIT_CACHE_VERSION,
        "created_at": utc_now(),
        "settings_fingerprint": fingerprint,
        "settings": settings,
        "questions": int(feature_manifest["questions"]),
        "documents": int(feature_manifest["documents"]),
        "shards": len(feature_shards),
        "stored_tensors": [
            "no_document_choice_logits",
            "with_document_choice_logits",
            "gold_direction_norm",
            "document_offsets",
        ],
    }
    completed = 0
    missing: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for source_path in feature_shards:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        expected = expected_shard_metadata(source, fingerprint)
        if args.resume and logit_shard_valid(cache_dir, expected):
            completed += 1
        else:
            missing.append((source_path, source, expected))
    logging.info(
        "Exact-choice logit cache: completed=%s missing=%s total=%s",
        completed,
        len(missing),
        len(feature_shards),
    )
    if args.dry_run:
        return
    if not missing:
        atomic_write_json(cache_dir / "manifest.json", manifest)
        return

    extractor_args = SimpleNamespace(
        device=args.device,
        dtype=args.dtype,
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=False,
        attn_implementation=args.attn_implementation,
        layers=[str(args.layer)],
        max_input_tokens=args.max_input_tokens,
    )
    extractor = FeatureExtractor(extractor_args)
    try:
        progress = tqdm(
            missing,
            desc="ExactChoiceLogits",
            unit="shard",
            dynamic_ncols=True,
        )
        for _, source, expected in progress:
            shard_rows = validate_shard_candidates(source, candidates)
            no_doc_logits: list[torch.Tensor] = []
            c_norms: list[torch.Tensor] = []
            for start in range(0, len(shard_rows), args.question_batch_size):
                batch = shard_rows[start : start + args.question_batch_size]
                sequences, _ = extractor.encode_questions(batch, [None] * len(batch))
                gold = [choice_index(normalize_gold(row)) for row in batch]
                logits, c_norm = adaptive_question_features(
                    extractor,
                    sequences,
                    gold,
                    description=f"no-document:{source['name']}:{start}",
                )
                no_doc_logits.append(logits)
                c_norms.append(c_norm)

            flat_rows: list[dict[str, Any]] = []
            flat_documents: list[dict[str, Any]] = []
            offsets = [0]
            for row in shard_rows:
                documents = candidate_documents(row)
                flat_rows.extend([row] * len(documents))
                flat_documents.extend(documents)
                offsets.append(len(flat_documents))
            doc_logits: list[torch.Tensor] = []
            for start in range(0, len(flat_rows), args.document_batch_size):
                batch_rows = flat_rows[start : start + args.document_batch_size]
                texts = [
                    str(doc.get("text") or "").strip()
                    for doc in flat_documents[start : start + args.document_batch_size]
                ]
                if any(not value for value in texts):
                    raise ValueError(f"Empty document text in {source['name']} at pair offset {start}")
                sequences, _ = extractor.encode_questions(batch_rows, texts)
                doc_logits.append(
                    adaptive_exact_logits(
                        extractor,
                        sequences,
                        description=f"with-document:{source['name']}:{start}",
                    )
                )
            tensors = {
                "no_document_choice_logits": torch.cat(no_doc_logits, dim=0),
                "with_document_choice_logits": torch.cat(doc_logits, dim=0),
                "gold_direction_norm": torch.cat(c_norms, dim=0),
                "document_offsets": torch.tensor(offsets, dtype=torch.int64),
            }
            atomic_write_logit_shard(cache_dir, expected, tensors)
            completed += 1
            progress.set_postfix_str(f"cached={completed}/{len(feature_shards)}")
            del tensors, no_doc_logits, doc_logits, c_norms
            gc.collect()
    finally:
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_write_json(cache_dir / "manifest.json", manifest)


def log_softmax_4(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits)
    shifted = logits - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def choice_behavior(logits: np.ndarray, gold_index: int) -> dict[str, Any]:
    if logits.shape != (4,):
        raise ValueError(f"Expected four choice logits, got {logits.shape}")
    others = np.delete(logits, gold_index)
    prediction_index = int(np.argmax(logits))
    log_probs = log_softmax_4(logits)
    return {
        "prediction": CHOICES[prediction_index],
        "correct": prediction_index == gold_index,
        "gold_logprob": float(log_probs[gold_index]),
        "gold_margin": float(logits[gold_index] - np.max(others)),
    }


def hidden_rows_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows(path):
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError("Hidden-label row without pair_id")
        if pair_id in output:
            raise RuntimeError(f"Duplicate hidden-label pair_id: {pair_id}")
        output[pair_id] = row
    return output


def build_pair_frame(
    args: argparse.Namespace,
    candidates: Sequence[dict[str, Any]],
    feature_shards: Sequence[Path],
    cache_dir: Path,
) -> pd.DataFrame:
    hidden = hidden_rows_by_pair(args.hidden_labels_path)
    records: list[dict[str, Any]] = []
    for source_path in tqdm(feature_shards, desc="JoinExactMargins", unit="shard", dynamic_ncols=True):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        shard_rows = validate_shard_candidates(source, candidates)
        logit_path = cache_dir / "shards" / f"{source['name']}.safetensors"
        if not logit_path.is_file():
            raise FileNotFoundError(
                f"Missing exact-logit shard {logit_path}; run with --mode materialize or --mode all"
            )
        logits = load_file(str(logit_path), device="cpu")
        hidden_tensors = load_file(str(source_path.with_suffix(".safetensors")), device="cpu")
        offsets = logits["document_offsets"].to(torch.int64).tolist()
        hidden_offsets = hidden_tensors["document_offsets"].to(torch.int64).tolist()
        if offsets != hidden_offsets:
            raise RuntimeError(f"Document offsets differ in {source['name']}")
        h0 = hidden_tensors["h0"].float()
        hD = hidden_tensors["hD"].float()
        no_doc_logits = logits["no_document_choice_logits"].float().numpy()
        doc_logits = logits["with_document_choice_logits"].float().numpy()
        c_norms = logits["gold_direction_norm"].float().numpy()
        for question_position, row in enumerate(shard_rows):
            gold = normalize_gold(row)
            gold_index = choice_index(gold)
            base = choice_behavior(no_doc_logits[question_position], gold_index)
            documents = candidate_documents(row)
            begin, end = offsets[question_position], offsets[question_position + 1]
            if end - begin != len(documents):
                raise RuntimeError(f"Document count mismatch for {row.get('key')}")
            delta_norms = torch.linalg.vector_norm(
                hD[begin:end] - h0[question_position].unsqueeze(0), dim=-1
            ).numpy()
            for document_position, document in enumerate(documents):
                flat_position = begin + document_position
                pair_id = f"{row['sample_id']}::{document_position + 1}::{stable_id(document)}"
                hidden_row = hidden.pop(pair_id, None)
                if hidden_row is None:
                    raise RuntimeError(f"Missing hidden score for {pair_id}")
                score = float(hidden_row["projection_score"])
                doc = choice_behavior(doc_logits[flat_position], gold_index)
                transition = ("C" if base["correct"] else "W") + "->" + (
                    "C" if doc["correct"] else "W"
                )
                records.append(
                    {
                        "dataset": str(row.get("dataset")),
                        "sample_key": str(row.get("key")),
                        "sample_id": str(row.get("sample_id")),
                        "pair_id": pair_id,
                        "doc_rank": document_position + 1,
                        "source": str(document.get("source") or "unknown"),
                        "gold_answer": gold,
                        "no_document_prediction": base["prediction"],
                        "with_document_prediction": doc["prediction"],
                        "no_document_correct": bool(base["correct"]),
                        "with_document_correct": bool(doc["correct"]),
                        "answer_transition": transition,
                        "projection_score": score,
                        "gold_direction_norm": float(c_norms[question_position]),
                        "linearized_gold_logprob_delta": score * float(c_norms[question_position]),
                        "delta_h_norm": float(delta_norms[document_position]),
                        "delta_c_cosine": score / max(float(delta_norms[document_position]), 1e-12),
                        "no_document_gold_logprob": base["gold_logprob"],
                        "with_document_gold_logprob": doc["gold_logprob"],
                        "actual_gold_logprob_delta": doc["gold_logprob"] - base["gold_logprob"],
                        "no_document_gold_margin": base["gold_margin"],
                        "with_document_gold_margin": doc["gold_margin"],
                        "actual_gold_margin_delta": doc["gold_margin"] - base["gold_margin"],
                    }
                )
        del logits, hidden_tensors, h0, hD
    if hidden:
        examples = list(hidden)[:5]
        raise RuntimeError(f"Unused hidden-label pairs remain: {len(hidden)} examples={examples}")
    return pd.DataFrame.from_records(records)


def safe_pearson(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right, method="pearson")
    return None if pd.isna(value) else float(value)


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right, method="spearman")
    return None if pd.isna(value) else float(value)


def binary_metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, Any]:
    pred = predicted.astype(bool).to_numpy()
    gold = actual.astype(bool).to_numpy()
    tp = int(np.sum(pred & gold))
    fp = int(np.sum(pred & ~gold))
    tn = int(np.sum(~pred & ~gold))
    fn = int(np.sum(~pred & gold))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "n": len(pred),
        "positive_rate_predicted": float(np.mean(pred)) if len(pred) else None,
        "positive_rate_actual": float(np.mean(gold)) if len(gold) else None,
        "accuracy": float(np.mean(pred == gold)) if len(pred) else None,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def regression_metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, Any]:
    x = predicted.to_numpy(dtype=np.float64)
    y = actual.to_numpy(dtype=np.float64)
    if len(x) < 2 or np.var(x) <= 0:
        return {"slope": None, "intercept": None, "r_squared": None, "mae": None, "rmse": None}
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    residual = y - fitted
    total = y - np.mean(y)
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) else None
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
    }


def group_summary(frame: pd.DataFrame, thresholds: Sequence[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pairs": int(len(frame)),
        "questions": int(frame["sample_key"].nunique()),
        "projection_score_mean": float(frame["projection_score"].mean()),
        "linearized_gold_logprob_delta_mean": float(frame["linearized_gold_logprob_delta"].mean()),
        "actual_gold_logprob_delta_mean": float(frame["actual_gold_logprob_delta"].mean()),
        "actual_gold_margin_delta_mean": float(frame["actual_gold_margin_delta"].mean()),
        "correlation": {
            "projection_vs_gold_logprob_delta": {
                "pearson": safe_pearson(frame["projection_score"], frame["actual_gold_logprob_delta"]),
                "spearman": safe_spearman(frame["projection_score"], frame["actual_gold_logprob_delta"]),
            },
            "linearized_vs_gold_logprob_delta": {
                "pearson": safe_pearson(
                    frame["linearized_gold_logprob_delta"], frame["actual_gold_logprob_delta"]
                ),
                "spearman": safe_spearman(
                    frame["linearized_gold_logprob_delta"], frame["actual_gold_logprob_delta"]
                ),
            },
            "projection_vs_gold_margin_delta": {
                "pearson": safe_pearson(frame["projection_score"], frame["actual_gold_margin_delta"]),
                "spearman": safe_spearman(frame["projection_score"], frame["actual_gold_margin_delta"]),
            },
            "linearized_vs_gold_margin_delta": {
                "pearson": safe_pearson(
                    frame["linearized_gold_logprob_delta"], frame["actual_gold_margin_delta"]
                ),
                "spearman": safe_spearman(
                    frame["linearized_gold_logprob_delta"], frame["actual_gold_margin_delta"]
                ),
            },
        },
        "linear_approximation": regression_metrics(
            frame["linearized_gold_logprob_delta"], frame["actual_gold_logprob_delta"]
        ),
        "thresholds": {},
        "answer_transitions": {
            key: int(value) for key, value in frame["answer_transition"].value_counts().items()
        },
    }
    for threshold in thresholds:
        name = format(threshold, ".8g")
        predicted = frame["projection_score"] > threshold
        summary["thresholds"][name] = {
            "vs_gold_logprob_improvement": binary_metrics(
                predicted, frame["actual_gold_logprob_delta"] > 0
            ),
            "vs_gold_margin_improvement": binary_metrics(
                predicted, frame["actual_gold_margin_delta"] > 0
            ),
        }
    return summary


def quantile_table(
    frame: pd.DataFrame,
    value_column: str,
    bins: int,
) -> pd.DataFrame:
    ranked = frame[value_column].rank(method="first")
    labels = pd.qcut(ranked, q=min(bins, len(frame)), labels=False, duplicates="drop")
    working = frame.assign(quantile_bin=labels.astype(int) + 1)
    table = (
        working.groupby("quantile_bin", observed=True)
        .agg(
            pairs=("pair_id", "size"),
            value_min=(value_column, "min"),
            value_max=(value_column, "max"),
            value_mean=(value_column, "mean"),
            projection_score_mean=("projection_score", "mean"),
            delta_h_norm_mean=("delta_h_norm", "mean"),
            actual_gold_logprob_delta_mean=("actual_gold_logprob_delta", "mean"),
            actual_gold_margin_delta_mean=("actual_gold_margin_delta", "mean"),
            with_document_accuracy=("with_document_correct", "mean"),
        )
        .reset_index()
    )
    return table


def subgroup_table(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {name: value for name, value in zip(group_columns, keys)}
        record.update(
            {
                "pairs": len(group),
                "questions": group["sample_key"].nunique(),
                "projection_score_mean": group["projection_score"].mean(),
                "actual_gold_logprob_delta_mean": group["actual_gold_logprob_delta"].mean(),
                "actual_gold_margin_delta_mean": group["actual_gold_margin_delta"].mean(),
                "pearson_projection_logprob": safe_pearson(
                    group["projection_score"], group["actual_gold_logprob_delta"]
                ),
                "spearman_projection_logprob": safe_spearman(
                    group["projection_score"], group["actual_gold_logprob_delta"]
                ),
                "pearson_projection_margin": safe_pearson(
                    group["projection_score"], group["actual_gold_margin_delta"]
                ),
                "spearman_projection_margin": safe_spearman(
                    group["projection_score"], group["actual_gold_margin_delta"]
                ),
                "sign_agreement_logprob": (
                    (group["projection_score"] > 0)
                    == (group["actual_gold_logprob_delta"] > 0)
                ).mean(),
                "sign_agreement_margin": (
                    (group["projection_score"] > 0)
                    == (group["actual_gold_margin_delta"] > 0)
                ).mean(),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def write_plots(frame: pd.DataFrame, output_dir: Path, bins: int) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping plots")
        return []
    paths: list[str] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sample = frame.sample(n=min(60_000, len(frame)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hexbin(
        sample["projection_score"],
        sample["actual_gold_logprob_delta"],
        gridsize=60,
        mincnt=1,
        bins="log",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Hidden projection score", ylabel="Actual gold log-probability delta")
    axes[1].hexbin(
        sample["projection_score"],
        sample["actual_gold_margin_delta"],
        gridsize=60,
        mincnt=1,
        bins="log",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Hidden projection score", ylabel="Actual gold margin delta")
    fig.tight_layout()
    path = plot_dir / "hidden_score_vs_actual_change.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    calibration = quantile_table(frame, "projection_score", bins)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        calibration["projection_score_mean"],
        calibration["actual_gold_logprob_delta_mean"],
        marker="o",
        label="Gold log-probability delta",
    )
    axis.plot(
        calibration["projection_score_mean"],
        calibration["actual_gold_margin_delta_mean"],
        marker="s",
        label="Gold margin delta",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Mean hidden score in quantile", ylabel="Mean exact behavioral change")
    axis.legend()
    fig.tight_layout()
    path = plot_dir / "hidden_score_quantile_calibration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def percent(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.2f}%"


def number(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def markdown_summary(summary: dict[str, Any], subgroup: pd.DataFrame) -> str:
    overall = summary["overall"]
    corr = overall["correlation"]
    lines = [
        "# Hidden score vs exact gold-margin validation",
        "",
        f"- Pairs: **{overall['pairs']:,}**",
        f"- Questions: **{overall['questions']:,}**",
        "- Prompt: fixed direct-choice pre-answer state (`Final answer:`)",
        "- `projection_score`: `(hD-h0) · c_hat`",
        "- `linearized_gold_logprob_delta`: `projection_score × ||c||`",
        "- Actual margin: `gold choice logit - max(other choice logits)`",
        "",
        "## Overall correlation",
        "",
        "| Comparison | Pearson | Spearman |",
        "|---|---:|---:|",
    ]
    labels = {
        "projection_vs_gold_logprob_delta": "Projection vs gold log-prob delta",
        "linearized_vs_gold_logprob_delta": "Gradient-scaled projection vs gold log-prob delta",
        "projection_vs_gold_margin_delta": "Projection vs gold margin delta",
        "linearized_vs_gold_margin_delta": "Gradient-scaled projection vs gold margin delta",
    }
    for key, label in labels.items():
        values = corr[key]
        lines.append(
            f"| {label} | {number(values['pearson'])} | {number(values['spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Threshold agreement",
            "",
            "The rows below treat an exact positive change as the behavioral target.",
            "",
            "| Hidden threshold | Target | Precision | Recall | Specificity | F1 | Agreement | Predicted positive | Actual positive |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold, targets in overall["thresholds"].items():
        for target_name, label in (
            ("vs_gold_logprob_improvement", "Gold log-prob improves"),
            ("vs_gold_margin_improvement", "Gold margin improves"),
        ):
            values = targets[target_name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        threshold,
                        label,
                        percent(values["precision"]),
                        percent(values["recall"]),
                        percent(values["specificity"]),
                        percent(values["f1"]),
                        percent(values["accuracy"]),
                        percent(values["positive_rate_predicted"]),
                        percent(values["positive_rate_actual"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Dataset / no-RAG correctness",
            "",
            "| Dataset | No-RAG correct | Pairs | Projection↔logP Pearson | Projection↔margin Pearson | logP sign agreement | margin sign agreement |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in subgroup.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {bool(row.no_document_correct)} | {int(row.pairs):,} | "
            f"{number(row.pearson_projection_logprob)} | {number(row.pearson_projection_margin)} | "
            f"{percent(row.sign_agreement_logprob)} | {percent(row.sign_agreement_margin)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The gradient-scaled projection is a **local first-order prediction of four-choice gold log-probability change**, not an exact causal effect.",
            "- Gold margin additionally depends on which competing option is strongest, so its agreement can be lower even when gold log-probability agreement is high.",
            "- Large `||hD-h0||` values test the approximation far from its linearization point; inspect `by_delta_norm_quantile.csv` before choosing a new label threshold.",
            "- This analysis uses the same fixed direct-choice prompt that produced h0/hD. It should not be mixed with free-rationale generation margins.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    args: argparse.Namespace,
    candidates: Sequence[dict[str, Any]],
    feature_shards: Sequence[Path],
    cache_dir: Path,
) -> None:
    cache_manifest = load_manifest(cache_dir / "manifest.json", LOGIT_CACHE_TYPE)
    expected_fingerprint = settings_fingerprint(cache_settings(args))
    if cache_manifest.get("settings_fingerprint") != expected_fingerprint:
        raise RuntimeError("Exact-logit cache settings do not match this analysis invocation")
    frame = build_pair_frame(args, candidates, feature_shards, cache_dir)
    if len(frame) != sum(len(row.get("candidate_documents") or []) for row in candidates):
        raise RuntimeError("Joined pair frame is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "pair_level_validation.parquet"
    temporary = pair_path.with_name(pair_path.name + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, pair_path)

    summary: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": utc_now(),
        "inputs": {
            "candidates_path": str(args.candidates_path.resolve()),
            "feature_cache_dir": str(args.feature_cache_dir.resolve()),
            "hidden_labels_path": str(args.hidden_labels_path.resolve()),
            "choice_logit_cache_dir": str(cache_dir.resolve()),
            "model_name_or_path": str(args.model_name_or_path.resolve()),
            "layer": args.layer,
            "thresholds": args.thresholds,
        },
        "overall": group_summary(frame, args.thresholds),
        "by_dataset": {
            str(dataset): group_summary(group, args.thresholds)
            for dataset, group in frame.groupby("dataset", observed=True)
        },
        "by_no_document_correct": {
            str(bool(correct)): group_summary(group, args.thresholds)
            for correct, group in frame.groupby("no_document_correct", observed=True)
        },
        "by_answer_transition": {
            str(transition): group_summary(group, args.thresholds)
            for transition, group in frame.groupby("answer_transition", observed=True)
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)

    score_bins = quantile_table(frame, "projection_score", args.quantile_bins)
    score_bins.to_csv(args.output_dir / "by_projection_score_quantile.csv", index=False)
    norm_bins = quantile_table(frame, "delta_h_norm", args.quantile_bins)
    norm_bins.to_csv(args.output_dir / "by_delta_norm_quantile.csv", index=False)

    dataset_correct = subgroup_table(frame, ["dataset", "no_document_correct"])
    dataset_correct.to_csv(args.output_dir / "by_dataset_no_rag_correctness.csv", index=False)
    dataset_source = subgroup_table(frame, ["dataset", "source"])
    dataset_source.to_csv(args.output_dir / "by_dataset_source.csv", index=False)
    transition = subgroup_table(frame, ["dataset", "answer_transition"])
    transition.to_csv(args.output_dir / "by_dataset_answer_transition.csv", index=False)

    rank_edges = [0, 1, 2, 4, 8, 16, 32]
    frame["rerank_bucket"] = pd.cut(
        frame["doc_rank"], bins=rank_edges, labels=["1", "2", "3-4", "5-8", "9-16", "17-32"]
    )
    rank_table = subgroup_table(frame, ["dataset", "rerank_bucket"])
    rank_table.to_csv(args.output_dir / "by_dataset_rerank_bucket.csv", index=False)

    plot_paths = write_plots(frame, args.output_dir, args.quantile_bins)
    summary["artifacts"] = {
        "pair_level": str(pair_path),
        "plots": plot_paths,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_text(args.output_dir / "summary.md", markdown_summary(summary, dataset_correct))
    logging.info(
        "Analysis complete: pairs=%s questions=%s summary=%s",
        len(frame),
        frame["sample_key"].nunique(),
        args.output_dir / "summary.md",
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.choice_logit_cache_dir = (
        args.choice_logit_cache_dir or args.output_dir / "exact_choice_logits_cache"
    )
    candidates, feature_shards, feature_manifest = validate_inputs(args)
    logging.info(
        "Input contract valid: questions=%s pairs=%s feature_shards=%s",
        len(candidates),
        sum(len(row.get("candidate_documents") or []) for row in candidates),
        len(feature_shards),
    )
    if args.mode in {"all", "materialize"}:
        materialize_logits(
            args,
            candidates,
            feature_shards,
            feature_manifest,
            args.choice_logit_cache_dir,
        )
    if args.dry_run:
        return
    if args.mode in {"all", "analyze"}:
        run_analysis(args, candidates, feature_shards, args.choice_logit_cache_dir)


if __name__ == "__main__":
    main()

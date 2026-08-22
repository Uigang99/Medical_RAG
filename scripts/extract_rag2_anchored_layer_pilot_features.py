#!/usr/bin/env python3
"""Extract all-block hidden states at three fixed anchors for a RAG2 pilot.

The input rationale text and constrained answer were already generated.  This
script teacher-forces the exact canonical prefix through ``Final answer: (``
and captures every decoder block at:

* ``pre_rationale``: the final token of ``Rationale:``;
* ``post_rationale``: the final token of ``### END OF REASONING ###``;
* ``pre_choice``: the final token of ``Final answer: (``.

For no-document traces, one backward pass captures the loss-decreasing gold
answer direction at all block/anchor combinations.  For each document trace,
``hD`` and exact four-choice logits are saved.  Outputs are sharded, atomic,
and resumable.
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
from typing import Any, Iterable, Iterator, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.rag2_anchored_trace import (  # noqa: E402
    ANCHOR_NAMES,
    CHOICES,
    PROMPT_VERSION,
    TRACE_VERSION,
    encode_to_pre_choice,
    normalized_mcq_row,
)


DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
FEATURE_VERSION = "rag2_three_anchor_all_block_features_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-batch-size", type=int, default=2)
    parser.add_argument("--document-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def atomic_save_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        str(temporary),
        metadata=metadata,
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from error
    return rows


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_identity(path: Path) -> dict[str, Any]:
    names = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    names.extend(item.name for item in sorted(path.glob("*.safetensors")))
    return {"path": str(path.resolve()), "files": [file_identity(path / name) for name in names if (path / name).is_file()]}


def pad_encodings(
    encodings: Sequence[Any],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(value.input_ids) for value in encodings)
    batch_size = len(encodings)
    input_ids = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    anchor_indices = torch.zeros(
        (batch_size, len(ANCHOR_NAMES)), dtype=torch.long, device=device
    )
    last_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    for row_index, value in enumerate(encodings):
        length = len(value.input_ids)
        input_ids[row_index, :length] = torch.tensor(value.input_ids, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
        last_indices[row_index] = length - 1
        anchor_indices[row_index] = torch.tensor(
            [value.anchor_indices[name] for name in ANCHOR_NAMES],
            dtype=torch.long,
            device=device,
        )
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids, anchor_indices, last_indices


class AllBlockAnchorExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map[args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path),
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("A fast tokenizer is required for exact anchor offset mapping")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has neither pad nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("Loading model on %s: %s", self.device, args.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path),
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.blocks = list(self.model.model.layers)
        self.layer_names = [f"block_{index:02d}" for index in range(len(self.blocks))]
        self.hidden_size = int(self.model.config.hidden_size)
        self.choice_token_ids: dict[str, int] = {}
        for label in CHOICES:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Choice {label!r} is not one token: {ids}")
            self.choice_token_ids[label] = int(ids[0])
        self.choice_ids = torch.tensor(
            [self.choice_token_ids[label] for label in CHOICES],
            dtype=torch.long,
            device=self.device,
        )
        self._active_anchor_indices: torch.Tensor | None = None
        self._active_states: dict[int, torch.Tensor] = {}
        self._active_directions: dict[int, torch.Tensor] = {}
        self._capture_gradients = False
        self._handles = [
            block.register_forward_hook(self._make_hook(layer_index))
            for layer_index, block in enumerate(self.blocks)
        ]
        logging.info(
            "Model ready: blocks=%d anchors=%s hidden=%d choices=%s",
            len(self.blocks),
            list(ANCHOR_NAMES),
            self.hidden_size,
            self.choice_token_ids,
        )

    def _make_hook(self, layer_index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            indices = self._active_anchor_indices
            if indices is None:
                return
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self._active_states[layer_index] = (
                hidden[batch_indices, indices].detach()
            )
            if self._capture_gradients:
                saved_indices = indices.detach().clone()

                def gradient_hook(gradient: torch.Tensor) -> None:
                    rows = torch.arange(gradient.shape[0], device=gradient.device)[:, None]
                    self._active_directions[layer_index] = (
                        -gradient[rows, saved_indices].detach()
                    )

                hidden.register_hook(gradient_hook)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def encode_rows(self, rows: Sequence[dict[str, Any]]) -> list[Any]:
        encodings = []
        for row in rows:
            normalized = normalized_mcq_row(
                {
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["gold_answer"],
                }
            )
            document_text = row.get("document_text_used") if row.get("kind") == "with_document" else None
            encoding = encode_to_pre_choice(
                self.tokenizer,
                normalized,
                document_text,
                str(row.get("rationale") or ""),
            )
            if len(encoding.input_ids) > self.args.max_input_tokens:
                raise ValueError(
                    f"Anchored prefix exceeds --max-input-tokens for {row.get('pair_id') or row['sample_id']}: "
                    f"{len(encoding.input_ids)} > {self.args.max_input_tokens}"
                )
            if encoding.prompt_sha256 != row.get("user_prompt_sha256"):
                raise RuntimeError(
                    f"Prompt contract mismatch for {row.get('pair_id') or row['sample_id']}: "
                    f"trace={row.get('user_prompt_sha256')} replay={encoding.prompt_sha256}"
                )
            encodings.append(encoding)
        return encodings

    def _forward(
        self,
        encodings: Sequence[Any],
        *,
        gold_indices: Sequence[int] | None,
    ) -> dict[str, torch.Tensor]:
        input_ids, attention_mask, position_ids, anchor_indices, last_indices = pad_encodings(
            encodings, self.tokenizer.pad_token_id, self.device
        )
        self._active_anchor_indices = anchor_indices
        self._active_states = {}
        self._active_directions = {}
        self._capture_gradients = gold_indices is not None
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        try:
            if gold_indices is not None:
                inputs_embeds = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                with torch.enable_grad():
                    outputs = self.model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    decision_hidden = outputs.last_hidden_state[batch_indices, last_indices]
                    logits = self.model.lm_head(decision_hidden)
                    choice_logits = logits.index_select(-1, self.choice_ids).float()
                    gold = torch.tensor(gold_indices, dtype=torch.long, device=self.device)
                    loss = F.cross_entropy(choice_logits, gold, reduction="sum")
                    loss.backward()
            else:
                with torch.inference_mode():
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    decision_hidden = outputs.last_hidden_state[batch_indices, last_indices]
                    logits = self.model.lm_head(decision_hidden)
                    choice_logits = logits.index_select(-1, self.choice_ids).float()
            missing_states = sorted(set(range(len(self.blocks))) - set(self._active_states))
            if missing_states:
                raise RuntimeError(f"Forward hooks missed block states: {missing_states}")
            states = torch.stack(
                [self._active_states[index] for index in range(len(self.blocks))], dim=1
            ).float().cpu()
            result = {
                "states": states,
                "choice_logits": choice_logits.detach().float().cpu(),
                "choice_probabilities": F.softmax(choice_logits.detach().float(), dim=-1).cpu(),
            }
            if gold_indices is not None:
                missing_gradients = sorted(set(range(len(self.blocks))) - set(self._active_directions))
                if missing_gradients:
                    raise RuntimeError(f"Backward hooks missed block gradients: {missing_gradients}")
                directions = torch.stack(
                    [self._active_directions[index] for index in range(len(self.blocks))], dim=1
                ).float().cpu()
                direction_norm = torch.linalg.vector_norm(directions, dim=-1)
                result["c_unit"] = directions / direction_norm.clamp_min(1e-12).unsqueeze(-1)
                result["c_norm"] = direction_norm
                gold = torch.tensor(gold_indices, dtype=torch.long)
                result["gold_choice_logprob"] = F.log_softmax(
                    result["choice_logits"], dim=-1
                ).gather(1, gold[:, None]).squeeze(1)
            return result
        finally:
            self._active_anchor_indices = None
            self._capture_gradients = False
            self._active_states = {}
            self._active_directions = {}


def feature_paths(output_dir: Path, shard_name: str) -> dict[str, Path]:
    root = output_dir / "feature_shards" / shard_name
    return {
        "root": root,
        "questions_meta": root / "questions.jsonl",
        "pairs_meta": root / "pairs.jsonl",
        "questions_tensor": root / "question_features.safetensors",
        "pairs_tensor": root / "pair_features.safetensors",
        "complete": root / "COMPLETE.json",
    }


def complete_valid(paths: dict[str, Path], question_count: int, pair_count: int) -> bool:
    required = [
        paths["questions_meta"],
        paths["pairs_meta"],
        paths["questions_tensor"],
        paths["pairs_tensor"],
        paths["complete"],
    ]
    if any(not path.is_file() for path in required):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("feature_version") == FEATURE_VERSION
        and marker.get("question_count") == question_count
        and marker.get("pair_count") == pair_count
    )


def choice_index(label: str) -> int:
    return CHOICES.index(str(label).upper())


def predicted_choice(probabilities: torch.Tensor) -> str:
    return CHOICES[int(torch.argmax(probabilities).item())]


def process_trace_shard(
    args: argparse.Namespace,
    extractor: AllBlockAnchorExtractor,
    trace_root: Path,
    output_paths: dict[str, Path],
) -> None:
    questions = read_jsonl(trace_root / "questions.jsonl")
    pairs = read_jsonl(trace_root / "pairs.jsonl")
    question_features: dict[str, dict[str, Any]] = {}
    question_states: list[torch.Tensor] = []
    question_c_unit: list[torch.Tensor] = []
    question_c_norm: list[torch.Tensor] = []
    question_logits: list[torch.Tensor] = []
    question_probabilities: list[torch.Tensor] = []
    question_gold_logprob: list[torch.Tensor] = []
    question_meta: list[dict[str, Any]] = []

    for batch_rows in chunks(questions, args.question_batch_size):
        encodings = extractor.encode_rows(batch_rows)
        features = extractor._forward(
            encodings,
            gold_indices=[choice_index(row["gold_answer"]) for row in batch_rows],
        )
        for local_index, (row, encoding) in enumerate(zip(batch_rows, encodings)):
            tensor_row = len(question_states)
            states = features["states"][local_index]
            c_unit = features["c_unit"][local_index]
            c_norm = features["c_norm"][local_index]
            logits = features["choice_logits"][local_index]
            probabilities = features["choice_probabilities"][local_index]
            prediction = predicted_choice(probabilities)
            question_states.append(states.half())
            question_c_unit.append(c_unit.half())
            question_c_norm.append(c_norm)
            question_logits.append(logits)
            question_probabilities.append(probabilities)
            question_gold_logprob.append(features["gold_choice_logprob"][local_index])
            question_features[row["sample_id"]] = {
                "tensor_row": tensor_row,
                "h0": states,
                "c_unit": c_unit,
                "c_norm": c_norm,
                "gold_logprob": features["gold_choice_logprob"][local_index],
                "prediction": prediction,
            }
            question_meta.append(
                {
                    "feature_version": FEATURE_VERSION,
                    "trace_version": row["trace_version"],
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "gold_answer": row["gold_answer"],
                    "generated_answer": row["answer"],
                    "hf_replay_answer": prediction,
                    "generated_hf_answer_match": prediction == row["answer"],
                    "no_document_correct": prediction == row["gold_answer"],
                    "valid_for_layer_analysis": bool(row.get("valid_for_layer_analysis")),
                    "quality_flags": row.get("quality_flags") or [],
                    "input_token_count": len(encoding.input_ids),
                    "anchor_indices": encoding.anchor_indices,
                    "anchor_token_ids": encoding.anchor_token_ids,
                    "anchor_token_text": encoding.anchor_token_text,
                    "tensor_row": tensor_row,
                }
            )

    pair_states: list[torch.Tensor] = []
    pair_logits: list[torch.Tensor] = []
    pair_probabilities: list[torch.Tensor] = []
    pair_projection: list[torch.Tensor] = []
    pair_cosine: list[torch.Tensor] = []
    pair_delta_norm: list[torch.Tensor] = []
    pair_c_norm: list[torch.Tensor] = []
    pair_gold_delta: list[torch.Tensor] = []
    pair_question_rows: list[int] = []
    pair_meta: list[dict[str, Any]] = []

    for batch_rows in chunks(pairs, args.document_batch_size):
        encodings = extractor.encode_rows(batch_rows)
        features = extractor._forward(encodings, gold_indices=None)
        for local_index, (row, encoding) in enumerate(zip(batch_rows, encodings)):
            question = question_features.get(row["sample_id"])
            if question is None:
                raise RuntimeError(f"No no-document trace for pair {row['pair_id']}")
            hD = features["states"][local_index]
            logits = features["choice_logits"][local_index]
            probabilities = features["choice_probabilities"][local_index]
            delta = hD - question["h0"]
            delta_norm = torch.linalg.vector_norm(delta, dim=-1)
            projection = torch.sum(delta * question["c_unit"], dim=-1)
            cosine = projection / delta_norm.clamp_min(1e-12)
            gold_index = choice_index(row["gold_answer"])
            gold_logprob = F.log_softmax(logits, dim=-1)[gold_index]
            gold_delta = gold_logprob - question["gold_logprob"]
            prediction = predicted_choice(probabilities)
            no_document_prediction = question["prediction"]
            transition = (
                ("C" if no_document_prediction == row["gold_answer"] else "W")
                + "->"
                + ("C" if prediction == row["gold_answer"] else "W")
            )
            tensor_row = len(pair_states)
            pair_states.append(hD.half())
            pair_logits.append(logits)
            pair_probabilities.append(probabilities)
            pair_projection.append(projection)
            pair_cosine.append(cosine)
            pair_delta_norm.append(delta_norm)
            pair_c_norm.append(question["c_norm"])
            pair_gold_delta.append(gold_delta)
            pair_question_rows.append(question["tensor_row"])
            pair_meta.append(
                {
                    "feature_version": FEATURE_VERSION,
                    "trace_version": row["trace_version"],
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "document": row["document"],
                    "gold_answer": row["gold_answer"],
                    "generated_answer": row["answer"],
                    "hf_replay_answer": prediction,
                    "generated_hf_answer_match": prediction == row["answer"],
                    "no_document_answer": no_document_prediction,
                    "no_document_correct": no_document_prediction == row["gold_answer"],
                    "with_document_correct": prediction == row["gold_answer"],
                    "answer_transition": transition,
                    "gold_choice_logprob_delta": float(gold_delta.item()),
                    "valid_for_layer_analysis": bool(
                        row.get("valid_for_layer_analysis")
                        and question_meta[question["tensor_row"]]["valid_for_layer_analysis"]
                    ),
                    "quality_flags": row.get("quality_flags") or [],
                    "input_token_count": len(encoding.input_ids),
                    "anchor_indices": encoding.anchor_indices,
                    "anchor_token_ids": encoding.anchor_token_ids,
                    "anchor_token_text": encoding.anchor_token_text,
                    "tensor_row": tensor_row,
                    "question_tensor_row": question["tensor_row"],
                }
            )

    tensor_metadata = {
        "feature_version": FEATURE_VERSION,
        "trace_version": TRACE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "layer_order": json.dumps(extractor.layer_names),
        "anchor_order": json.dumps(list(ANCHOR_NAMES)),
        "choice_order": json.dumps(list(CHOICES)),
        "vector_layout": "[row, decoder_block, anchor, hidden]",
    }
    atomic_save_safetensors(
        output_paths["questions_tensor"],
        {
            "h0": torch.stack(question_states),
            "c_unit": torch.stack(question_c_unit),
            "c_norm": torch.stack(question_c_norm).float(),
            "choice_logits": torch.stack(question_logits).float(),
            "choice_probabilities": torch.stack(question_probabilities).float(),
            "gold_choice_logprob": torch.stack(question_gold_logprob).float(),
        },
        tensor_metadata,
    )
    atomic_save_safetensors(
        output_paths["pairs_tensor"],
        {
            "hD": torch.stack(pair_states),
            "utility_projection": torch.stack(pair_projection).float(),
            "delta_c_cosine": torch.stack(pair_cosine).float(),
            "delta_h_norm": torch.stack(pair_delta_norm).float(),
            "c_norm": torch.stack(pair_c_norm).float(),
            "choice_logits": torch.stack(pair_logits).float(),
            "choice_probabilities": torch.stack(pair_probabilities).float(),
            "gold_choice_logprob_delta": torch.stack(pair_gold_delta).float(),
            "question_tensor_row": torch.tensor(pair_question_rows, dtype=torch.int64),
        },
        tensor_metadata,
    )
    atomic_write_jsonl(output_paths["questions_meta"], question_meta)
    atomic_write_jsonl(output_paths["pairs_meta"], pair_meta)
    atomic_write_json(
        output_paths["complete"],
        {
            "feature_version": FEATURE_VERSION,
            "completed_at": utc_now(),
            "question_count": len(question_meta),
            "pair_count": len(pair_meta),
            "valid_question_count": sum(row["valid_for_layer_analysis"] for row in question_meta),
            "valid_pair_count": sum(row["valid_for_layer_analysis"] for row in pair_meta),
        },
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.question_batch_size <= 0 or args.document_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    generation_manifest_path = args.trace_dir / "generation_manifest.json"
    if not generation_manifest_path.is_file():
        raise FileNotFoundError(generation_manifest_path)
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if generation_manifest.get("trace_version") != TRACE_VERSION:
        raise RuntimeError("Trace version mismatch")
    trace_shards = sorted((args.trace_dir / "trace_shards").glob("shard_*"))
    if len(trace_shards) != int(generation_manifest.get("shards", -1)):
        raise RuntimeError(
            f"Trace shard count mismatch: files={len(trace_shards)} manifest={generation_manifest.get('shards')}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = AllBlockAnchorExtractor(args)
    try:
        progress = tqdm(
            total=int(generation_manifest["pairs"]),
            desc="ThreeAnchorFeatures",
            unit="pair",
            dynamic_ncols=True,
        )
        for trace_root in trace_shards:
            trace_marker = json.loads((trace_root / "COMPLETE.json").read_text(encoding="utf-8"))
            expected_questions = int(trace_marker["question_count"])
            expected_pairs = int(trace_marker["pair_count"])
            output_paths = feature_paths(args.output_dir, trace_root.name)
            if args.resume and complete_valid(output_paths, expected_questions, expected_pairs):
                progress.update(expected_pairs)
                continue
            output_paths["root"].mkdir(parents=True, exist_ok=True)
            process_trace_shard(args, extractor, trace_root, output_paths)
            progress.update(expected_pairs)
            progress.set_postfix(shard=trace_root.name)
        progress.close()
        manifest = {
            "feature_version": FEATURE_VERSION,
            "trace_version": TRACE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "created_at": utc_now(),
            "trace_dir": str(args.trace_dir.resolve()),
            "generation_manifest": file_identity(generation_manifest_path),
            "model": model_identity(args.model_name_or_path),
            "questions": int(generation_manifest["questions"]),
            "pairs": int(generation_manifest["pairs"]),
            "shards": len(trace_shards),
            "layer_order": extractor.layer_names,
            "anchor_order": list(ANCHOR_NAMES),
            "choice_token_ids": extractor.choice_token_ids,
            "hidden_size": extractor.hidden_size,
            "storage_dtype": "float16",
            "compute_dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "max_input_tokens": args.max_input_tokens,
        }
        atomic_write_json(args.output_dir / "feature_manifest.json", manifest)
        logging.info("All-block three-anchor extraction complete: %s", args.output_dir)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()

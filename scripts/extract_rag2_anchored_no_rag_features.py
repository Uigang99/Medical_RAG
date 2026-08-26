#!/usr/bin/env python3
"""Extract reusable no-RAG hidden features from anchored training traces.

Selected decoder-block states are saved at all three stable anchors:
``pre_rationale``, ``post_rationale``, and ``pre_choice``.  Four-choice logits
and probabilities at the final decision position are retained as diagnostics.
No backward pass is used in this stage.  Outputs are sharded, atomic, and
resumable; CUDA OOM retries split only the affected batch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    ANCHOR_NAMES,
    CHOICES,
    PROMPT_VERSION,
    TRACE_VERSION,
    encode_to_pre_choice,
    normalized_mcq_row,
)

RUN_VERSION = "rag2_anchored_no_rag_selected_layer_features_v1"
GENERATION_RUN_VERSION = "rag2_anchored_no_rag_train_generation_v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
MCQ_DATASETS = (
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", choices=MCQ_DATASETS, default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 12, 20, 28, 31], help="Zero-based decoder block indices.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
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


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(temporary), metadata=metadata)
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
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error
    return rows


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def feature_paths(root: Path, dataset: str, split: str, shard_name: str) -> dict[str, Path]:
    base = root / "no_rag_features" / dataset / split / "shards" / shard_name
    return {"root": base, "meta": base / "questions.jsonl", "tensor": base / "features.safetensors", "complete": base / "COMPLETE.json"}


def complete_valid(paths: dict[str, Path], expected: int, layers: Sequence[int]) -> bool:
    if any(not paths[key].is_file() for key in ("meta", "tensor", "complete")):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and int(marker.get("question_count", -1)) == expected
        and marker.get("layers") == list(layers)
    )


def pad_encodings(encodings: Sequence[Any], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    max_length = max(len(item.input_ids) for item in encodings)
    batch = len(encodings)
    input_ids = torch.full((batch, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    anchor_indices = torch.zeros((batch, len(ANCHOR_NAMES)), dtype=torch.long, device=device)
    last_indices = torch.zeros(batch, dtype=torch.long, device=device)
    for index, item in enumerate(encodings):
        length = len(item.input_ids)
        input_ids[index, :length] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        anchor_indices[index] = torch.tensor([item.anchor_indices[name] for name in ANCHOR_NAMES], dtype=torch.long, device=device)
        last_indices[index] = length - 1
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids, anchor_indices, last_indices


class SelectedLayerExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("Fast tokenizer required for anchor offsets")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("Loading hidden-state model on %s: %s", self.device, args.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path), dtype=dtype, low_cpu_mem_usage=True,
            local_files_only=True, trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)
        blocks = list(self.model.model.layers)
        if len(set(args.layers)) != len(args.layers) or any(index < 0 or index >= len(blocks) for index in args.layers):
            raise ValueError(f"Invalid --layers {args.layers}; model has {len(blocks)} blocks")
        self.layers = list(args.layers)
        self.layer_names = [f"block_{index:02d}" for index in self.layers]
        self.hidden_size = int(self.model.config.hidden_size)
        self.choice_token_ids = {}
        for choice in CHOICES:
            token_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"Choice {choice} is not one token: {token_ids}")
            self.choice_token_ids[choice] = int(token_ids[0])
        self.choice_ids = torch.tensor([self.choice_token_ids[choice] for choice in CHOICES], dtype=torch.long, device=self.device)
        self._anchor_indices: torch.Tensor | None = None
        self._states: dict[int, torch.Tensor] = {}
        self._handles = [blocks[index].register_forward_hook(self._hook(index)) for index in self.layers]

    def _hook(self, layer_index: int) -> Any:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if self._anchor_indices is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self._states[layer_index] = hidden[rows, self._anchor_indices].detach()
        return capture

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def encode(self, row: dict[str, Any]) -> Any:
        normalized = normalized_mcq_row({"question": row["question"], "options": row["options"], "answer": row["gold_answer"]})
        encoding = encode_to_pre_choice(self.tokenizer, normalized, None, str((row.get("parsed") or {}).get("rationale") or ""))
        if len(encoding.input_ids) > self.args.max_input_tokens:
            raise ValueError(f"Input exceeds max tokens for {row['sample_id']}: {len(encoding.input_ids)} > {self.args.max_input_tokens}")
        if encoding.prompt_sha256 != row.get("user_prompt_sha256"):
            raise RuntimeError(f"Prompt hash mismatch for {row['sample_id']}")
        return encoding

    def extract(self, rows: Sequence[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], list[Any]]:
        encodings = [self.encode(row) for row in rows]
        input_ids, attention_mask, position_ids, anchor_indices, last_indices = pad_encodings(encodings, self.tokenizer.pad_token_id, self.device)
        self._anchor_indices = anchor_indices
        self._states = {}
        batch_indices = torch.arange(len(rows), device=self.device)
        try:
            with torch.inference_mode():
                outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                decision_hidden = outputs.last_hidden_state[batch_indices, last_indices]
                logits = self.model.lm_head(decision_hidden).index_select(-1, self.choice_ids).float()
            if set(self._states) != set(self.layers):
                raise RuntimeError(f"Missing hooked layers: {sorted(set(self.layers) - set(self._states))}")
            hidden = torch.stack([self._states[index] for index in self.layers], dim=1).float().cpu()
            return {"anchor_hidden": hidden, "choice_logits": logits.cpu(), "choice_probabilities": F.softmax(logits, dim=-1).cpu()}, encodings
        finally:
            self._anchor_indices = None
            self._states = {}


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def adaptive_extract(extractor: SelectedLayerExtractor, rows: Sequence[dict[str, Any]]) -> list[tuple[dict[str, torch.Tensor], list[Any], Sequence[dict[str, Any]]]]:
    try:
        features, encodings = extractor.extract(rows)
        return [(features, encodings, rows)]
    except RuntimeError as error:
        if not is_cuda_oom(error) or len(rows) <= 1:
            raise
        logging.warning("Feature OOM for batch=%d; retrying this batch as %d + %d", len(rows), len(rows) // 2, len(rows) - len(rows) // 2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        middle = len(rows) // 2
        return adaptive_extract(extractor, rows[:middle]) + adaptive_extract(extractor, rows[middle:])


def process_shard(args: argparse.Namespace, extractor: SelectedLayerExtractor, trace_path: Path, paths: dict[str, Path]) -> int:
    rows = read_jsonl(trace_path)
    hidden_rows: list[torch.Tensor] = []
    logits_rows: list[torch.Tensor] = []
    probability_rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for batch in chunks(rows, args.batch_size):
        for features, encodings, actual_rows in adaptive_extract(extractor, batch):
            for index, (row, encoding) in enumerate(zip(actual_rows, encodings)):
                probabilities = features["choice_probabilities"][index]
                hf_answer = CHOICES[int(torch.argmax(probabilities).item())]
                tensor_row = len(hidden_rows)
                hidden_rows.append(features["anchor_hidden"][index].half())
                logits_rows.append(features["choice_logits"][index])
                probability_rows.append(probabilities)
                metadata.append({
                    "run_version": RUN_VERSION,
                    "trace_version": row["trace_version"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "row_idx": int(row["row_idx"]),
                    "gold_answer": row["gold_answer"],
                    "generated_answer": row["answer"],
                    "generated_answer_correct": bool(row["answer_correct"]),
                    "hf_replay_answer": hf_answer,
                    "hf_replay_correct": hf_answer == row["gold_answer"],
                    "generated_hf_answer_match": hf_answer == row["answer"],
                    "input_token_count": len(encoding.input_ids),
                    "anchor_indices": encoding.anchor_indices,
                    "anchor_token_ids": encoding.anchor_token_ids,
                    "anchor_token_text": encoding.anchor_token_text,
                    "tensor_row": tensor_row,
                })
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_save_safetensors(
        paths["tensor"],
        {
            "anchor_hidden": torch.stack(hidden_rows),
            "choice_logits": torch.stack(logits_rows).float(),
            "choice_probabilities": torch.stack(probability_rows).float(),
        },
        {
            "run_version": RUN_VERSION,
            "trace_version": TRACE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "layer_order": json.dumps(extractor.layer_names),
            "anchor_order": json.dumps(list(ANCHOR_NAMES)),
            "choice_order": json.dumps(list(CHOICES)),
            "anchor_hidden_layout": "[row, selected_decoder_block, anchor, hidden]",
        },
    )
    atomic_write_jsonl(paths["meta"], metadata)
    atomic_write_json(paths["complete"], {"run_version": RUN_VERSION, "completed_at": utc_now(), "question_count": len(metadata), "layers": args.layers})
    return len(metadata)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    manifest_path = args.trace_root / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    generation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generation_manifest.get("run_version") != GENERATION_RUN_VERSION:
        raise RuntimeError(f"Generation version mismatch: {generation_manifest.get('run_version')}")
    counts = {dataset: int(generation_manifest["datasets"][dataset]) for dataset in args.datasets}
    total = sum(counts.values())
    trace_shards: list[tuple[str, Path, int]] = []
    completed = 0
    for dataset in args.datasets:
        roots = sorted((args.trace_root / "trace_shards" / dataset / args.split).glob("shard_*"))
        observed = 0
        for root in roots:
            marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
            expected = int(marker["question_count"])
            observed += expected
            paths = feature_paths(args.output_root, dataset, args.split, root.name)
            trace_shards.append((dataset, root, expected))
            if args.resume and complete_valid(paths, expected, args.layers):
                completed += expected
        if observed != counts[dataset]:
            raise RuntimeError(
                f"Trace shard coverage mismatch for {dataset}: {observed} != {counts[dataset]}"
            )
    extractor = SelectedLayerExtractor(args)
    progress = PipelineProgress(overall_total=3 * total, overall_initial=2 * total + completed, desc="AnchoredNoRAG")
    progress.set_stage("3/3 no-RAG hidden features", total=total, initial=completed)
    newly_written = 0
    try:
        for dataset, trace_root, expected in trace_shards:
            paths = feature_paths(args.output_root, dataset, args.split, trace_root.name)
            if args.resume and complete_valid(paths, expected, args.layers):
                continue
            written = process_shard(args, extractor, trace_root / "questions.jsonl", paths)
            if written != expected:
                raise RuntimeError(f"Feature count mismatch in {trace_root}: {written} != {expected}")
            newly_written += written
            progress.update(written)
        atomic_write_json(
            args.output_root / "feature_manifest.json",
            {
                "run_version": RUN_VERSION,
                "trace_version": TRACE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "created_at": utc_now(),
                "trace_root": str(args.trace_root.resolve()),
                "model_name_or_path": str(args.model_name_or_path.resolve()),
                "datasets": counts,
                "total_questions": total,
                "layers": args.layers,
                "layer_order": extractor.layer_names,
                "anchor_order": list(ANCHOR_NAMES),
                "choice_token_ids": extractor.choice_token_ids,
                "hidden_size": extractor.hidden_size,
                "storage_dtype": "float16",
                "compute_dtype": args.dtype,
                "max_input_tokens": args.max_input_tokens,
                "newly_written": newly_written,
                "feature_semantics": {
                    "pre_rationale": "state at the final token of Rationale: before free reasoning",
                    "post_rationale": "state at the final token of the fixed end-of-reasoning marker",
                    "pre_choice": "state at the opening parenthesis immediately before A/B/C/D",
                },
            },
        )
        logging.info("No-RAG feature extraction complete: %s", args.output_root)
    finally:
        progress.close()
        extractor.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract one gold-answer direction per anchored no-RAG question.

The expensive with-document hidden states already exist.  This script replays
only the no-RAG anchored response prefix, differentiates the four-choice gold
loss at ``block_28/pre_choice``, and stores a unit loss-decreasing direction.
Outputs are aligned to the existing no-RAG feature shards, atomic, and fully
resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.progress import PipelineProgress  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    PROMPT_VERSION,
    TRACE_VERSION,
    encode_to_pre_choice,
    normalized_mcq_row,
)


RUN_VERSION = "rag2_anchored_gold_direction_selected_anchor_v1"
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
    base = PROJECT_ROOT / "datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-rag-root",
        type=Path,
        default=base / "train_no_rag_anchored_features_v1",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base / "hidden_utility_extreme_curriculum_v1/gold_directions",
    )
    parser.add_argument("--datasets", nargs="+", choices=MCQ_DATASETS, default=["medmcqa", "medqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--anchor", choices=("pre_rationale", "post_rationale", "pre_choice"), default="pre_choice")
    # Must match the cached h0 extraction batch so eager-attention padding and
    # bfloat16 reduction geometry are replayed exactly.
    parser.add_argument("--question-batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--state-max-abs-tolerance", type=float, default=0.25)
    parser.add_argument("--state-min-cosine", type=float, default=0.999)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL: {path}:{line_number}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        str(temporary),
        metadata=metadata,
    )
    os.replace(temporary, path)


def shard_paths(root: Path, dataset: str, split: str, shard_name: str) -> dict[str, Path]:
    base = root / dataset / split / "shards" / shard_name
    return {
        "root": base,
        "directions": base / "directions.safetensors",
        "questions": base / "questions.jsonl",
        "complete": base / "COMPLETE.json",
    }


def complete_valid(
    paths: dict[str, Path], count: int, layer: int, anchor: str, replay_batch_size: int
) -> bool:
    if any(not paths[name].is_file() for name in ("directions", "questions", "complete")):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("run_version") == RUN_VERSION
        and int(marker.get("question_count", -1)) == count
        and int(marker.get("layer", -1)) == layer
        and marker.get("anchor") == anchor
        and int(marker.get("replay_batch_size", -1)) == replay_batch_size
    )


def pad_encodings(encodings: Sequence[Any], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    max_length = max(len(value.input_ids) for value in encodings)
    batch_size = len(encodings)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    anchor_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    last_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    for index, value in enumerate(encodings):
        length = len(value.input_ids)
        input_ids[index, :length] = torch.tensor(value.input_ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        anchor_indices[index] = int(value.anchor_indices["pre_choice"])
        last_indices[index] = length - 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids, anchor_indices, last_indices


class DirectionExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_name_or_path), local_files_only=True, use_fast=True, trust_remote_code=True
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("A fast tokenizer is required for anchor validation")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("Loading direction model on %s: %s", self.device, args.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name_or_path),
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)
        blocks = list(self.model.model.layers)
        if not 0 <= args.layer < len(blocks):
            raise ValueError(f"--layer must be in [0,{len(blocks)-1}]")
        self.block = blocks[args.layer]
        self.hidden_size = int(self.model.config.hidden_size)
        self.choice_token_ids = {}
        for label in CHOICES:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Choice {label} is not one token: {ids}")
            self.choice_token_ids[label] = int(ids[0])
        self.choice_ids = torch.tensor(
            [self.choice_token_ids[label] for label in CHOICES], dtype=torch.long, device=self.device
        )
        self._anchor_indices: torch.Tensor | None = None
        self._replacement_h0: torch.Tensor | None = None
        self._captured_state: torch.Tensor | None = None
        self._captured_leaf: torch.Tensor | None = None
        self._handle = self.block.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if self._anchor_indices is None or self._replacement_h0 is None:
            return
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        self._captured_state = hidden[rows, self._anchor_indices].detach()
        # The cached h0 is the state used by every downstream score.  Inject it
        # at the selected anchor and detach the prefix so c is evaluated at
        # exactly that state while only blocks after --layer retain a graph.
        leaf = hidden.detach().clone()
        leaf[rows, self._anchor_indices] = self._replacement_h0.to(
            device=hidden.device, dtype=hidden.dtype
        )
        leaf.requires_grad_(True)
        self._captured_leaf = leaf
        if isinstance(output, tuple):
            return (leaf, *output[1:])
        return leaf

    def close(self) -> None:
        self._handle.remove()
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_rows(self, rows: Sequence[dict[str, Any]]) -> list[Any]:
        encodings = []
        for row in rows:
            normalized = normalized_mcq_row(
                {"question": row["question"], "options": row["options"], "answer": row["gold_answer"]}
            )
            rationale = str(row.get("model_raw_rationale") or (row.get("parsed") or {}).get("rationale") or "")
            encoding = encode_to_pre_choice(self.tokenizer, normalized, None, rationale)
            if len(encoding.input_ids) > self.args.max_input_tokens:
                raise ValueError(
                    f"Prefix too long for {row['sample_id']}: {len(encoding.input_ids)} > "
                    f"{self.args.max_input_tokens}"
                )
            if encoding.prompt_sha256 != row.get("user_prompt_sha256"):
                raise RuntimeError(f"Prompt hash mismatch for {row['sample_id']}")
            encodings.append(encoding)
        return encodings

    def forward(
        self, rows: Sequence[dict[str, Any]], cached_h0: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        encodings = self.encode_rows(rows)
        input_ids, attention_mask, position_ids, anchors, last = pad_encodings(
            encodings, self.tokenizer.pad_token_id, self.device
        )
        self._anchor_indices = anchors
        self._replacement_h0 = cached_h0.to(self.device)
        self._captured_state = None
        self._captured_leaf = None
        batch_rows = torch.arange(input_ids.shape[0], device=self.device)
        try:
            with torch.enable_grad():
                output = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                final_hidden = output.last_hidden_state[batch_rows, last]
                choice_logits = self.model.lm_head(final_hidden).index_select(-1, self.choice_ids).float()
                gold = torch.tensor(
                    [CHOICES.index(str(row["gold_answer"]).upper()) for row in rows],
                    dtype=torch.long,
                    device=self.device,
                )
                F.cross_entropy(choice_logits, gold, reduction="sum").backward()
            if (
                self._captured_state is None
                or self._captured_leaf is None
                or self._captured_leaf.grad is None
            ):
                raise RuntimeError("Selected block hook did not capture state and gradient")
            directions = -self._captured_leaf.grad[batch_rows, anchors].detach().float()
            norms = torch.linalg.vector_norm(directions, dim=-1)
            probabilities = F.softmax(choice_logits.detach().float(), dim=-1)
            gold_logprob = F.log_softmax(choice_logits.detach().float(), dim=-1).gather(
                1, gold[:, None]
            ).squeeze(1)
            return {
                "h0_replay": self._captured_state.float().cpu(),
                "c_unit": (directions / norms.clamp_min(1e-12).unsqueeze(-1)).cpu(),
                "c_norm": norms.cpu(),
                "choice_logits": choice_logits.detach().float().cpu(),
                "choice_probabilities": probabilities.cpu(),
                "gold_choice_logprob": gold_logprob.cpu(),
            }
        finally:
            self._anchor_indices = None
            self._replacement_h0 = None
            self._captured_state = None
            self._captured_leaf = None

    def adaptive_forward(
        self, rows: Sequence[dict[str, Any]], cached_h0: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        oom = False
        try:
            return self.forward(rows, cached_h0)
        except torch.cuda.OutOfMemoryError:
            if len(rows) <= 1:
                raise
            oom = True
        if not oom:  # pragma: no cover - return/exception above is exhaustive
            raise AssertionError("unreachable")
        # Retry only after leaving the except block; otherwise the exception
        # traceback retains the failed CUDA graph throughout recursion.
        logging.warning("Direction OOM for batch=%d; retrying as two half-batches", len(rows))
        gc.collect()
        torch.cuda.empty_cache()
        middle = len(rows) // 2
        left = self.adaptive_forward(rows[:middle], cached_h0[:middle])
        right = self.adaptive_forward(rows[middle:], cached_h0[middle:])
        return {key: torch.cat((left[key], right[key]), dim=0) for key in left}


def validate_contract(args: argparse.Namespace) -> tuple[dict[str, Any], int, int]:
    manifest_path = args.no_rag_root / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("trace_version") != TRACE_VERSION or manifest.get("prompt_version") != PROMPT_VERSION:
        raise RuntimeError("Anchored no-RAG feature contract mismatch")
    layer_order = [int(value) for value in manifest.get("layers") or []]
    anchor_order = [str(value) for value in manifest.get("anchor_order") or []]
    if args.layer not in layer_order or args.anchor not in anchor_order:
        raise RuntimeError(f"Requested layer/anchor absent: layers={layer_order} anchors={anchor_order}")
    if args.anchor != "pre_choice":
        raise ValueError("Gold answer direction extraction is intentionally restricted to pre_choice")
    return manifest, layer_order.index(args.layer), anchor_order.index(args.anchor)


def process_shard(
    args: argparse.Namespace,
    extractor: DirectionExtractor,
    dataset: str,
    trace_dir: Path,
    feature_dir: Path,
    output: dict[str, Path],
    layer_index: int,
    anchor_index: int,
    progress: PipelineProgress,
) -> int:
    trace_rows = read_jsonl(trace_dir / "questions.jsonl")
    feature_rows = read_jsonl(feature_dir / "questions.jsonl")
    if len(trace_rows) != len(feature_rows):
        raise RuntimeError(f"Trace/feature count mismatch for {trace_dir}")
    with safe_open(feature_dir / "features.safetensors", framework="pt", device="cpu") as handle:
        cached_h0 = handle.get_slice("anchor_hidden")[:, layer_index, anchor_index, :].float()
    for index, (trace, feature) in enumerate(zip(trace_rows, feature_rows)):
        if trace["sample_id"] != feature["sample_id"] or int(feature["tensor_row"]) != index:
            raise RuntimeError(f"Question alignment mismatch in {trace_dir} row={index}")

    c_units: list[torch.Tensor] = []
    c_norms: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    gold_logprobs: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    offset = 0
    for batch in chunks(trace_rows, args.question_batch_size):
        expected = cached_h0[offset : offset + len(batch)]
        result = extractor.adaptive_forward(batch, expected)
        replay = result["h0_replay"]
        max_abs = torch.amax(torch.abs(expected - replay), dim=-1)
        cosine = F.cosine_similarity(expected.float(), replay.float(), dim=-1)
        # Prompt hashes are already exact.  A low replay cosine now acts only
        # as a gross contract guard because the cached state itself is injected
        # before computing c; ordinary bf16 replay drift is audit metadata.
        gross_invalid = cosine < 0.98
        if bool(gross_invalid.any()):
            local = int(torch.nonzero(gross_invalid, as_tuple=False)[0].item())
            raise RuntimeError(
                f"Gross cached h0 replay mismatch for {batch[local]['sample_id']}: "
                f"max_abs={max_abs[local].item():.6f} cosine={cosine[local].item():.8f}"
            )
        for local, row in enumerate(batch):
            prediction = CHOICES[int(torch.argmax(result["choice_probabilities"][local]).item())]
            metadata.append(
                {
                    "run_version": RUN_VERSION,
                    "dataset": dataset,
                    "split": args.split,
                    "sample_id": row["sample_id"],
                    "tensor_row": offset + local,
                    "gold_answer": row["gold_answer"],
                    "replay_answer": prediction,
                    "no_rag_correct": prediction == row["gold_answer"],
                    "h0_replay_max_abs": float(max_abs[local].item()),
                    "h0_replay_cosine": float(cosine[local].item()),
                    "h0_injected_for_direction": True,
                    "h0_replay_within_nominal_tolerance": bool(
                        max_abs[local] <= args.state_max_abs_tolerance
                        or cosine[local] >= args.state_min_cosine
                    ),
                }
            )
        c_units.append(result["c_unit"].half())
        c_norms.append(result["c_norm"])
        logits.append(result["choice_logits"])
        probabilities.append(result["choice_probabilities"])
        gold_logprobs.append(result["gold_choice_logprob"])
        offset += len(batch)
        progress.update(len(batch))
        progress.set_detail(f"dataset={dataset} shard={trace_dir.name}")

    atomic_safetensors(
        output["directions"],
        {
            "c_unit": torch.cat(c_units, dim=0),
            "c_norm": torch.cat(c_norms, dim=0).float(),
            "choice_logits": torch.cat(logits, dim=0).float(),
            "choice_probabilities": torch.cat(probabilities, dim=0).float(),
            "gold_choice_logprob": torch.cat(gold_logprobs, dim=0).float(),
        },
        {
            "run_version": RUN_VERSION,
            "dataset": dataset,
            "split": args.split,
            "layer": str(args.layer),
            "anchor": args.anchor,
            "vector_semantics": "unit negative gradient of four-choice gold CE at cached h0 anchor",
            "gradient_graph": "cached h0 intervention at selected block; backward through suffix only",
        },
    )
    atomic_jsonl(output["questions"], metadata)
    atomic_json(
        output["complete"],
        {
            "run_version": RUN_VERSION,
            "completed_at": utc_now(),
            "dataset": dataset,
            "split": args.split,
            "question_count": len(metadata),
            "layer": args.layer,
            "anchor": args.anchor,
            "replay_batch_size": args.question_batch_size,
            "max_h0_replay_difference": max(row["h0_replay_max_abs"] for row in metadata),
            "minimum_h0_replay_cosine": min(row["h0_replay_cosine"] for row in metadata),
        },
    )
    return len(metadata)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.question_batch_size <= 0 or args.max_input_tokens <= 0:
        raise ValueError("Batch size and token limit must be positive")
    manifest, layer_index, anchor_index = validate_contract(args)
    # Training artifacts historically stored trace_shards and no_rag_features
    # below one root.  External-test artifacts intentionally keep immutable
    # generation traces in the run cache and derived tensors in a separate
    # dataset root.  The feature manifest is the authoritative link between
    # those locations.
    trace_root_value = manifest.get("trace_root")
    trace_root = (
        Path(str(trace_root_value)).resolve()
        if trace_root_value
        else args.no_rag_root.resolve()
    )
    if not (trace_root / "generation_manifest.json").is_file():
        raise FileNotFoundError(
            f"No-RAG trace root from feature manifest is invalid: {trace_root}"
        )
    shard_plan: list[tuple[str, Path, Path, dict[str, Path], int]] = []
    total = 0
    completed = 0
    for dataset in args.datasets:
        trace_dirs = sorted((trace_root / "trace_shards" / dataset / args.split).glob("shard_*"))
        if not trace_dirs:
            raise FileNotFoundError(f"No no-RAG trace shards for {dataset}")
        for trace_dir in trace_dirs:
            feature_dir = args.no_rag_root / "no_rag_features" / dataset / args.split / "shards" / trace_dir.name
            marker = json.loads((trace_dir / "COMPLETE.json").read_text(encoding="utf-8"))
            count = int(marker["question_count"])
            output = shard_paths(args.output_root, dataset, args.split, trace_dir.name)
            total += count
            if args.resume and complete_valid(
                output, count, args.layer, args.anchor, args.question_batch_size
            ):
                completed += count
            shard_plan.append((dataset, trace_dir, feature_dir, output, count))
    logging.info(
        "Gold direction plan: total=%d completed=%d remaining=%d layer=%d anchor=%s",
        total,
        completed,
        total - completed,
        args.layer,
        args.anchor,
    )
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = PipelineProgress(
        overall_total=total,
        overall_initial=completed,
        desc="AnchoredGoldDirection",
        enabled=args.show_progress,
    )
    progress.set_stage("1/1 gold direction extraction", total=total, initial=completed)
    extractor = DirectionExtractor(args)
    try:
        newly_written = 0
        for dataset, trace_dir, feature_dir, output, count in shard_plan:
            if args.resume and complete_valid(
                output, count, args.layer, args.anchor, args.question_batch_size
            ):
                continue
            output["root"].mkdir(parents=True, exist_ok=True)
            newly_written += process_shard(
                args,
                extractor,
                dataset,
                trace_dir,
                feature_dir,
                output,
                layer_index,
                anchor_index,
                progress,
            )
        atomic_json(
            args.output_root / "direction_manifest.json",
            {
                "run_version": RUN_VERSION,
                "created_at": utc_now(),
                "no_rag_root": str(args.no_rag_root.resolve()),
                "trace_root": str(trace_root),
                "model_name_or_path": str(args.model_name_or_path.resolve()),
                "datasets": {dataset: int(manifest["datasets"][dataset]) for dataset in args.datasets},
                "total_questions": total,
                "newly_written": newly_written,
                "layer": args.layer,
                "anchor": args.anchor,
                "replay_batch_size": args.question_batch_size,
                "hidden_size": extractor.hidden_size,
                "choice_token_ids": extractor.choice_token_ids,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "model_input_contract": {
                    "gold_answer_used_only_for": "offline c extraction",
                    "forbidden_filter_inputs": ["gold_answer", "c", "projection_score", "answer_transition"],
                },
            },
        )
        logging.info("Gold direction extraction complete: %s", args.output_root)
    finally:
        extractor.close()
        progress.close()


if __name__ == "__main__":
    main()

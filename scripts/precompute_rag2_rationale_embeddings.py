from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.io_utils import read_json, write_json
from medrag.progress import StageProgress
from medrag.rag2_mcq import parse_mcq_output, parse_paper_exact_mcq_output


DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "MedCPT-Query-Encoder"
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "datasets" / "benchmark"
DEFAULT_NO_RAG_ROOT = (
    PROJECT_ROOT / "datasets" / "filtering" / "rag2" / "llama3_8b_paper_answer_format_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "databases"
    / "query_embeddings"
    / "medcpt_query_encoder"
    / "rag2_llama3_8b_paper_answer_format_v2"
)
QUERY_ENCODING_PROTOCOL_VERSION = "rag2_released_medcpt_query_cls_truncate512_v1"
RETRIEVAL_QUERY_CANONICALIZATION_VERSION = "rationale_only_plus_single_canonical_answer_v3"
PAPER_EXACT_RETRIEVAL_QUERY_FIELD = "parsed.rationale_query(raw_visible_response)"
PAPER_EXACT_RETRIEVAL_QUERY_CANONICALIZATION_VERSION = "raw_visible_response_no_rewrite_v1"
PAPER_EXACT_RETRIEVAL_QUERY_POLICY = "complete_visible_response_including_expressed_answer_no_rewrite_v1"
MCQ_EVAL_DATASETS = [
    "medmcqa",
    "medqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]
QUALITY_POLICIES = ("technical", "conservative")
SEMANTIC_RISK_PATTERNS = {
    "question_flawed_or_ambiguous": re.compile(
        r"\b(?:question (?:is|seems) (?:flawed|ambiguous)|ambiguous question|"
        r"no correct (?:option|answer)|all (?:the )?options are correct)\b",
        re.IGNORECASE,
    ),
    "forced_choice_after_refusal": re.compile(
        r"\b(?:if i had to choose|if forced to choose|although (?:the )?question .*?(?:flawed|ambiguous))\b",
        re.IGNORECASE,
    ),
    "insufficient_information": re.compile(
        r"\b(?:cannot determine|can't determine|not enough information|insufficient information|"
        r"unable to determine)\b",
        re.IGNORECASE,
    ),
}
TRAILING_ANSWER_LEAD_IN = re.compile(
    r"(?is)(?:\s*(?:therefore\s*,?\s+)?(?:the\s+)?(?:final\s+)?answer\s+is\s*:?[\s]*)+$"
)
ANSWER_CONCLUSION_LEAD_IN = re.compile(
    r"(?is)\btherefore\s*,?\s+the\s+answer\s+is\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed RAG2 no-RAG rationale queries with the MedCPT query encoder.")
    parser.add_argument("--dataset", choices=MCQ_EVAL_DATASETS, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--no-rag-root", type=Path, default=DEFAULT_NO_RAG_ROOT)
    parser.add_argument("--no-rag-path", type=Path, default=None)
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=None,
        help=(
            "Optional usable_rows.jsonl produced by audit_rag2_no_rag_quality_selection.py. "
            "When supplied, embed exactly those row_idx values (including deterministic recovered answers) "
            "instead of reapplying the generic quality policy."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help=(
            "MedCPT query length. The released RAG2 retriever uses truncation=True and max_length=512; "
            "this script records every truncation in metadata and the manifest."
        ),
    )
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--invalid-row-policy",
        choices=["exclude", "error"],
        default="exclude",
        help="Exclude malformed/truncated no-RAG rows with an audit file, or stop before embedding.",
    )
    parser.add_argument(
        "--quality-policy",
        choices=QUALITY_POLICIES,
        default="conservative",
        help=(
            "technical excludes malformed/PPL-invalid rows; conservative also excludes parsed responses that "
            "explicitly call the question ambiguous, insufficient, or make a forced choice."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL: {path}:{line_no}") from exc


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def load_latest_rows(path: Path, expected: int) -> tuple[list[dict[str, Any]], int]:
    latest: dict[int, dict[str, Any]] = {}
    duplicate_rows = 0
    for _, row in iter_jsonl(path):
        row_idx = int(row.get("row_idx", -1))
        if row_idx < 0:
            raise ValueError(f"Missing valid row_idx in {path}: sample_id={row.get('sample_id')}")
        duplicate_rows += int(row_idx in latest)
        latest[row_idx] = row
    if not latest:
        raise RuntimeError(f"No no-RAG rows found: {path}")
    missing = [row_idx for row_idx in range(expected) if row_idx not in latest]
    extras = sorted(row_idx for row_idx in latest if row_idx >= expected)
    if missing or extras:
        raise RuntimeError(
            f"No-RAG output is incomplete or misaligned: expected={expected} latest={len(latest)} "
            f"missing={len(missing)} first_missing={missing[:10]} extras={extras[:10]}"
        )
    return [latest[row_idx] for row_idx in range(expected)], duplicate_rows


def load_external_selection(
    path: Path,
    *,
    dataset: str,
    split: str,
    expected_rows: int,
) -> dict[int, dict[str, Any]]:
    """Load the strict no-RAG quality selection without consulting gold labels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing --selection-path: {path}")
    selected: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed selection JSONL: {path}:{line_no}") from exc
            row_idx = int(item.get("row_idx", -1))
            if row_idx < 0 or row_idx >= expected_rows:
                raise ValueError(
                    f"Selection row_idx out of range at {path}:{line_no}: {row_idx} not in [0,{expected_rows})"
                )
            if str(item.get("dataset")) != dataset or str(item.get("split")) != split:
                raise ValueError(
                    f"Selection dataset/split mismatch at {path}:{line_no}: "
                    f"{item.get('dataset')}/{item.get('split')} != {dataset}/{split}"
                )
            answer = str(item.get("selected_no_rag_answer") or "").upper()
            if not answer:
                raise ValueError(f"Selection is missing selected_no_rag_answer at {path}:{line_no}")
            if row_idx in selected:
                raise ValueError(f"Duplicate row_idx={row_idx} in selection file: {path}")
            selected[row_idx] = item
    if not selected:
        raise ValueError(f"--selection-path contains no usable rows: {path}")
    return selected


def load_no_rag_protocol(source_path: Path, dataset: str, split: str) -> dict[str, Any]:
    manifest_path = source_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing no-RAG artifact manifest: {manifest_path}. "
            "The embedding stage requires the generation protocol recorded beside the JSONL artifact."
        )
    manifest = read_json(manifest_path)
    expected = {
        "type": "rag2_no_rag_rationale_artifact",
        "dataset": dataset,
        "split": split,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    required = ["prompt_version", "ppl_scope_version", "generation_policy_version"]
    missing = [key for key in required if not str(manifest.get(key) or "").strip()]
    if mismatches or missing:
        raise ValueError(
            f"Incompatible no-RAG artifact manifest: mismatches={mismatches} missing={missing}"
        )
    return {**manifest, "manifest_path": str(manifest_path)}


def retrieval_query_contract(protocol: dict[str, Any]) -> dict[str, str]:
    """Describe the query text used by this cache without conflating two protocols."""
    if protocol.get("prompt_profile") == "paper_compatible_three_anchor":
        return {
            "query_field": "retrieval_query",
            "canonicalization_version": "anchored_rationale_plus_fixed_terminal_answer_v1",
            "query_semantics": (
                "complete generated rationale plus the fixed terminal answer; the original MCQ "
                "question and options are excluded"
            ),
        }
    if protocol.get("prompt_profile") == "paper_exact":
        return {
            "query_field": PAPER_EXACT_RETRIEVAL_QUERY_FIELD,
            "canonicalization_version": PAPER_EXACT_RETRIEVAL_QUERY_CANONICALIZATION_VERSION,
            "query_semantics": (
                "complete visible no-RAG model response, including the model's expressed final option, "
                "with no canonical answer rewrite; the original MCQ question and options are excluded"
            ),
        }
    return {
        "query_field": "reparsed(no_rag_generation).rationale_query",
        "canonicalization_version": RETRIEVAL_QUERY_CANONICALIZATION_VERSION,
        "query_semantics": (
            "complete no-RAG model response including the final 'Therefore, the answer is ...' conclusion; "
            "the original MCQ question and options are excluded"
        ),
    }


def resolve_rationale_query(
    row: dict[str, Any], protocol: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Rebuild the canonical retrieval query from the visible model response.

    Older completed artifacts may predate the current canonicalizer, leaving a
    duplicated or non-parenthesized conclusion in ``parsed.rationale_query``.
    Re-parsing is deterministic and preserves the generated rationale while
    making the retrieval query use one canonical answer conclusion.
    """
    generated = str(row.get("no_rag_generation") or row.get("model_raw_generation") or "")
    if not generated:
        return None, None
    options = row.get("options") or {}
    if protocol.get("prompt_profile") == "paper_compatible_three_anchor":
        stored_query = str(
            row.get("retrieval_query")
            or ((row.get("parsed") or {}).get("rationale_query") or "")
        ).strip()
        answer = str(
            ((row.get("parsed") or {}).get("final_answer") or "")
        ).strip().upper()
        return stored_query or None, answer or None
    if protocol.get("prompt_profile") == "paper_exact":
        # The released RAG2 prompt does not specify a response marker.  Its
        # retrieval query is consequently the unmodified visible response,
        # rather than a post-hoc canonical 'Therefore...' reconstruction.
        stored_query = str(((row.get("parsed") or {}).get("rationale_query") or "")).strip()
        raw_visible_response = stored_query or generated.strip()
        parsed = parse_paper_exact_mcq_output(raw_visible_response, options)
        return raw_visible_response or None, parsed.final_answer
    reparsed = parse_mcq_output(generated, options)
    final_answer = reparsed.final_answer
    option_text = str(options.get(str(final_answer or "").upper()) or "").strip()
    rationale_only = str(reparsed.rationale_only or "").strip()
    if final_answer and option_text and rationale_only:
        prefix = " ".join(rationale_only.split())
        # A few greedy generations state an answer once in the rationale and
        # then repeat it in the required final sentence. Retrieval needs only
        # the reasoning before that first conclusion plus one canonical end.
        prefix = ANSWER_CONCLUSION_LEAD_IN.split(prefix, maxsplit=1)[0].strip()
        prefix = TRAILING_ANSWER_LEAD_IN.sub("", prefix).strip()
        conclusion = f"Therefore, the answer is ({final_answer}) {option_text}."
        return f"{prefix} {conclusion}".strip(), final_answer
    return reparsed.rationale_query, final_answer


def semantic_quality_reasons(row: dict[str, Any]) -> list[str]:
    rationale = str(((row.get("parsed") or {}).get("rationale_only") or ""))
    return [
        f"semantic_risk:{name}"
        for name, pattern in SEMANTIC_RISK_PATTERNS.items()
        if pattern.search(rationale)
    ]


def row_validation_reasons(
    row: dict[str, Any],
    dataset: str,
    split: str,
    protocol: dict[str, Any],
    quality_policy: str,
) -> list[str]:
    parsed = row.get("parsed") or {}
    rationale_stats = ((row.get("generation_stats") or {}).get("rationale") or {})
    reasons: list[str] = []
    if str(row.get("dataset")) != dataset or str(row.get("split")) != split:
        reasons.append("dataset_or_split_mismatch")
    if row.get("prompt_version") != protocol["prompt_version"]:
        reasons.append("prompt_version_mismatch")
    if row.get("ppl_scope_version") != protocol["ppl_scope_version"]:
        reasons.append("ppl_scope_version_mismatch")
    if row.get("generation_policy_version") != protocol["generation_policy_version"]:
        reasons.append("generation_policy_version_mismatch")
    rationale_query, reparsed_answer = resolve_rationale_query(row, protocol)
    rationale_query = " ".join(str(rationale_query or "").split())
    if not rationale_query:
        reasons.append("missing_rationale_query")
    if not parsed.get("final_answer"):
        reasons.append("missing_answer")
    elif reparsed_answer != parsed.get("final_answer"):
        reasons.append("reparsed_answer_mismatch")
    if parsed.get("parse_errors"):
        reasons.append("parse_errors")
    if not rationale_stats.get("token_count") or rationale_stats.get("ppl") is None:
        reasons.append("missing_rationale_ppl")
    if row.get("truncated_by_max_tokens") or row.get("finish_reason") == "length":
        reasons.append("max_tokens_exhausted")
    if quality_policy == "conservative" and not reasons:
        reasons.extend(semantic_quality_reasons(row))
    return reasons


def partition_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    split: str,
    protocol: dict[str, Any],
    quality_policy: str,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[str]]]]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[tuple[dict[str, Any], list[str]]] = []
    for row in rows:
        reasons = row_validation_reasons(row, dataset, split, protocol, quality_policy)
        if reasons:
            invalid_rows.append((row, reasons))
        else:
            valid_rows.append(row)
    return valid_rows, invalid_rows


def autocast_context(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def embed_batch(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    untruncated = tokenizer(
        texts,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    original_lengths = np.asarray(
        [len(input_ids) for input_ids in untruncated["input_ids"]], dtype="int32"
    )
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded_lengths = encoded["attention_mask"].sum(dim=1).cpu().numpy().astype("int32", copy=False)
    truncated = original_lengths > max_length
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    with torch.inference_mode(), autocast_context(device):
        output = model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
    vectors = np.ascontiguousarray(embeddings.float().cpu().numpy(), dtype="float32")
    return vectors, original_lengths, encoded_lengths, truncated


def cache_is_current(
    output_dir: Path,
    source_path: Path,
    rows: int,
    dimension: int,
    max_length: int,
    protocol: dict[str, Any],
    quality_policy: str,
    selection_path: Path | None = None,
) -> bool:
    manifest_path = output_dir / "manifest.json"
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"
    if not manifest_path.exists() or not embeddings_path.exists() or not metadata_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
        embeddings = np.load(embeddings_path, mmap_mode="r")
    except Exception:
        return False
    stat = source_path.stat()
    query_contract = retrieval_query_contract(protocol)
    selection_manifest = manifest.get("external_quality_selection") or {}
    return (
        tuple(embeddings.shape) == (rows, dimension)
        and int(manifest.get("rows", -1)) == rows
        and int(manifest.get("dimension", -1)) == dimension
        and manifest.get("prompt_version") == protocol["prompt_version"]
        and manifest.get("ppl_scope_version") == protocol["ppl_scope_version"]
        and manifest.get("generation_policy_version") == protocol["generation_policy_version"]
        and manifest.get("query_encoding_protocol_version") == QUERY_ENCODING_PROTOCOL_VERSION
        and manifest.get("query_field") == query_contract["query_field"]
        and manifest.get("retrieval_query_canonicalization_version")
        == query_contract["canonicalization_version"]
        and manifest.get("quality_policy") == quality_policy
        and (
            (selection_path is None and not selection_manifest)
            or (
                selection_path is not None
                and selection_manifest.get("path") == str(selection_path)
                and int(selection_manifest.get("rows", -1)) == rows
            )
        )
        and int((manifest.get("model") or {}).get("max_length", -1)) == max_length
        and int(manifest.get("source_size_bytes", -1)) == stat.st_size
        and int(manifest.get("source_mtime_ns", -1)) == stat.st_mtime_ns
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("--batch-size and --max-length must be positive.")

    source_path = args.no_rag_path or (
        args.no_rag_root / "no_rag" / args.dataset / args.split / "no_rag_generations.jsonl"
    )
    benchmark_path = args.benchmark_root / "mcq" / args.collection / args.dataset / f"{args.split}.jsonl"
    output_dir = args.output_dir or (args.output_root / args.dataset / args.split)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing no-RAG artifact: {source_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark split: {benchmark_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing MedCPT query encoder: {args.model_path}")

    protocol = load_no_rag_protocol(source_path, args.dataset, args.split)
    expected_rows = count_jsonl(benchmark_path)
    all_rows, duplicate_rows = load_latest_rows(source_path, expected_rows)
    selection_by_row_idx: dict[int, dict[str, Any]] | None = None
    if args.selection_path is not None:
        args.selection_path = args.selection_path.resolve()
        selection_by_row_idx = load_external_selection(
            args.selection_path,
            dataset=args.dataset,
            split=args.split,
            expected_rows=expected_rows,
        )
        rows = [all_rows[row_idx] for row_idx in sorted(selection_by_row_idx)]
        for row in rows:
            selected = selection_by_row_idx[int(row["row_idx"])]
            if str(selected.get("sample_id") or "") != str(row.get("sample_id") or ""):
                raise ValueError(
                    "Selection/sample alignment mismatch: "
                    f"row_idx={row.get('row_idx')} selection={selected.get('sample_id')} "
                    f"source={row.get('sample_id')}"
                )
        invalid_rows: list[tuple[dict[str, Any], list[str]]] = []
        invalid_reason_counts: Counter[str] = Counter(
            {"external_quality_selection": expected_rows - len(rows)}
        )
    else:
        rows, invalid_rows = partition_rows(
            all_rows, args.dataset, args.split, protocol, args.quality_policy
        )
        invalid_reason_counts = Counter(reason for _, reasons in invalid_rows for reason in reasons)
        if invalid_rows and args.invalid_row_policy == "error":
            preview = [
                f"row_idx={row.get('row_idx')} sample_id={row.get('sample_id')} reasons={','.join(reasons)}"
                for row, reasons in invalid_rows[:20]
            ]
            raise RuntimeError(
                f"No-RAG artifact contains {len(invalid_rows)} invalid rows. "
                "Use --invalid-row-policy exclude to omit them without regenerating.\n" + "\n".join(preview)
            )
    if not rows:
        raise RuntimeError("No valid rationale queries remain after quality filtering.")
    logging.info(
        "Rationale query audit: benchmark=%s selected=%s excluded=%s selection=%s reasons=%s superseded_duplicates=%s",
        expected_rows,
        len(rows),
        expected_rows - len(rows),
        str(args.selection_path) if args.selection_path is not None else "generic_quality_policy",
        dict(invalid_reason_counts),
        duplicate_rows,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.warning("CUDA is not visible; rationale embeddings will run on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": args.attn_implementation,
    }
    if model_dtype is not None:
        model_kwargs["dtype"] = model_dtype
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()
    dimension = int(model.config.hidden_size)
    model_max_length = int(getattr(model.config, "max_position_embeddings", args.max_length))
    if args.max_length > model_max_length:
        raise ValueError(
            f"--max-length={args.max_length} exceeds the model position limit ({model_max_length})."
        )

    if (
        cache_is_current(
            output_dir,
            source_path,
            len(rows),
            dimension,
            args.max_length,
            protocol,
            args.quality_policy,
            args.selection_path,
        )
        and not args.overwrite
    ):
        logging.info("Current rationale embedding cache already exists: %s", output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_tmp = output_dir / "embeddings.npy.tmp"
    metadata_tmp = output_dir / "metadata.jsonl.tmp"
    excluded_tmp = output_dir / "excluded_rows.jsonl.tmp"
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"
    excluded_path = output_dir / "excluded_rows.jsonl"
    embeddings = np.lib.format.open_memmap(
        embeddings_tmp,
        mode="w+",
        dtype="float32",
        shape=(len(rows), dimension),
    )

    progress = StageProgress(total=len(rows), desc=f"EmbedRationale:{args.dataset}", enabled=True)
    original_query_token_lengths: list[int] = []
    encoded_query_token_lengths: list[int] = []
    truncated_queries = 0
    try:
        with metadata_tmp.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as metadata_out:
            for start in range(0, len(rows), args.batch_size):
                end = min(start + args.batch_size, len(rows))
                batch = rows[start:end]
                texts = [str(resolve_rationale_query(row, protocol)[0]) for row in batch]
                vectors, original_lengths, encoded_lengths, truncated = embed_batch(
                    texts, tokenizer, model, device, args.max_length
                )
                if selection_by_row_idx is not None and bool(truncated.any()):
                    offending = [
                        int(batch[index]["row_idx"])
                        for index, value in enumerate(truncated.tolist())
                        if value
                    ]
                    raise RuntimeError(
                        "External quality selection contained a query exceeding the configured MedCPT limit: "
                        f"row_idx={offending[:10]} max_length={args.max_length}"
                    )
                embeddings[start:end] = vectors
                original_query_token_lengths.extend(int(value) for value in original_lengths.tolist())
                encoded_query_token_lengths.extend(int(value) for value in encoded_lengths.tolist())
                truncated_queries += int(truncated.sum())
                for local_index, (cache_index, row) in enumerate(zip(range(start, end), batch)):
                    parsed = row.get("parsed") or {}
                    query_text, _ = resolve_rationale_query(row, protocol)
                    rationale_stats = ((row.get("generation_stats") or {}).get("rationale") or {})
                    selection_item = (
                        selection_by_row_idx.get(int(row["row_idx"]))
                        if selection_by_row_idx is not None
                        else None
                    )
                    selected_answer = (
                        selection_item.get("selected_no_rag_answer")
                        if selection_item is not None
                        else parsed.get("final_answer")
                    )
                    metadata_out.write(
                        json.dumps(
                            {
                                "cache_index": cache_index,
                                "row_idx": int(row["row_idx"]),
                                "sample_id": row.get("sample_id"),
                                "dataset": args.dataset,
                                "split": args.split,
                                "query_text": query_text,
                                "query_token_count": int(encoded_lengths[local_index]),
                                "query_token_count_original": int(original_lengths[local_index]),
                                "query_token_count_encoded": int(encoded_lengths[local_index]),
                                "query_truncated": bool(truncated[local_index]),
                                "no_rag_prediction": selected_answer,
                                "no_rag_correct": parsed.get("final_answer_correct"),
                                "no_rag_answer_source": (
                                    selection_item.get("answer_source")
                                    if selection_item is not None
                                    else "stored_parser"
                                ),
                                "no_rag_rationale_ppl": rationale_stats.get("ppl"),
                                "no_rag_generation_prompt_variant": row.get(
                                    "generation_prompt_variant", "standard"
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                progress.update(end - start)
    finally:
        progress.close()

    embeddings.flush()
    del embeddings
    embeddings_tmp.replace(embeddings_path)
    metadata_tmp.replace(metadata_path)
    if selection_by_row_idx is None:
        with excluded_tmp.open("w", encoding="utf-8") as excluded_out:
            for row, reasons in invalid_rows:
                excluded_out.write(
                    json.dumps(
                        {
                            "row_idx": row.get("row_idx"),
                            "sample_id": row.get("sample_id"),
                            "dataset": row.get("dataset"),
                            "split": row.get("split"),
                            "reasons": reasons,
                            "finish_reason": row.get("finish_reason"),
                            "parse_errors": (row.get("parsed") or {}).get("parse_errors") or [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        excluded_tmp.replace(excluded_path)
    stat = source_path.stat()
    write_json(
        output_dir / "manifest.json",
        {
            "type": "rag2_rationale_query_embedding_cache",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "split": args.split,
            "rows": len(rows),
            "benchmark_rows": expected_rows,
            "excluded_rows": expected_rows - len(rows),
            "excluded_reason_counts": dict(invalid_reason_counts),
            "invalid_row_policy": (
                "external_quality_selection" if selection_by_row_idx is not None else args.invalid_row_policy
            ),
            "quality_policy": args.quality_policy,
            "external_quality_selection": (
                {
                    "path": str(args.selection_path),
                    "rows": len(rows),
                    "contract": (
                        "Rows and recovered answer metadata are sourced from "
                        "audit_rag2_no_rag_quality_selection.py; raw rationale query text is unchanged."
                    ),
                }
                if selection_by_row_idx is not None
                else None
            ),
            "semantic_risk_patterns": list(SEMANTIC_RISK_PATTERNS),
            "row_alignment": "cache_index_is_contiguous; row_idx_preserves_sparse_original_benchmark_index",
            "dimension": dimension,
            "dtype": "float32",
            "embedding_path": "embeddings.npy",
            "metadata_path": "metadata.jsonl",
            "excluded_rows_path": (
                str(args.selection_path.parent / "excluded_rows.jsonl")
                if selection_by_row_idx is not None and args.selection_path is not None
                else "excluded_rows.jsonl"
            ),
            "source_path": str(source_path),
            "benchmark_path": str(benchmark_path),
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "source_manifest_path": protocol["manifest_path"],
            "prompt_profile": protocol.get("prompt_profile"),
            "prompt_version": protocol["prompt_version"],
            "ppl_scope_version": protocol["ppl_scope_version"],
            "generation_policy_version": protocol["generation_policy_version"],
            "query_field": retrieval_query_contract(protocol)["query_field"],
            "retrieval_query_canonicalization_version": retrieval_query_contract(protocol)[
                "canonicalization_version"
            ],
            "query_includes_answer_conclusion": True,
            "query_semantics": retrieval_query_contract(protocol)["query_semantics"],
            "query_encoding_protocol_version": QUERY_ENCODING_PROTOCOL_VERSION,
            "query_encoding_protocol": (
                "released RAG2 query_encode.py: MedCPT-Query-Encoder, truncation=True, max_length=512, "
                "last_hidden_state[:, 0, :]"
            ),
            "query_token_lengths": {
                "original_min": min(original_query_token_lengths),
                "original_max": max(original_query_token_lengths),
                "original_mean": sum(original_query_token_lengths) / len(original_query_token_lengths),
                "encoded_min": min(encoded_query_token_lengths),
                "encoded_max": max(encoded_query_token_lengths),
                "encoded_mean": sum(encoded_query_token_lengths) / len(encoded_query_token_lengths),
                "truncated": truncated_queries,
                "truncated_rate": truncated_queries / len(rows),
            },
            "model": {
                "name": "MedCPT-Query-Encoder",
                "path": str(args.model_path),
                "embedding_field": "last_hidden_state[:, 0, :]",
                "max_length": args.max_length,
                "max_position_embeddings": model_max_length,
                "normalize": False,
            },
        },
    )
    logging.info("Rationale embedding cache complete: %s", output_dir)


if __name__ == "__main__":
    main()

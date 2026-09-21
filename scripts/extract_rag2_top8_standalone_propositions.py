#!/usr/bin/env python3
"""Run the previously validated Standalone-v2 proposition prompt on test Top-8 documents.

The input is the immutable 44,911-chunk selection already materialized by the
biomedical-KG MVP.  The model never receives a question, answer choice, gold
answer, semantic label, behavioral score, or annotated evidence sentence.

The prompt, chat rendering, model, and decoding settings reproduce the Qwen
Standalone-v2 pilot contract recovered from its original execution log.  This
stage only creates source-grounded proposition candidates.  It does not
perform entity linking, relation extraction, graph construction, or QA
evaluation.  Raw generations are written as atomic resumable shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The installed server does not provide the external compiler used by the
# FlashInfer sampler.  Native vLLM sampling is deterministic and sufficient.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


RUN_VERSION = "rag2_test_top8_qwen_standalone_v2_exact_v1"
PROMPT_VERSION = "qwen_standalone_v2_exact_20260917"
SCHEMA_VERSION = 1
EXPECTED_DOCUMENTS = 44_911
EXPECTED_BY_SOURCE = {
    "cpg": 10_144,
    "pmc": 14_239,
    "pubmed": 13_279,
    "textbooks": 7_249,
}
EXPECTED_SELECTION_SHA256 = "8cfaa4081d81bb0e2e2fb70cfa4d57066121de56bf9089873a7a0e6f97762e24"
SOURCES = tuple(EXPECTED_BY_SOURCE)

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "datasets/kg/rag2_test_top8_entity_evidence_graph_v1"
DEFAULT_INPUT = DEFAULT_INPUT_ROOT / "selected_chunks.jsonl"
DEFAULT_INPUT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.json"
DEFAULT_MODEL = WORKSPACE_ROOT / "models/Qwen3.5-9B"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/kg/rag2_test_top8_qwen_standalone_v2_exact_v1"

SPACE_RE = re.compile(r"\s+")
DEICTIC_RE = re.compile(
    r"\b(?:this|these|those|it|they|them|their|the study|our study|we|here)\b",
    flags=re.IGNORECASE,
)
SYSTEM_PROMPT = """Transform a medical or biomedical source passage into a complete set of clean, atomic, standalone propositions.

The proposition is a semantic normalization, not a sentence copied from the source. Remove rhetorical wording, repetition, headings, citations, and unnecessary modifiers while preserving the source meaning.

REQUIREMENTS
1. Each proposition must express one independently truth-evaluable relation.
2. Use only information explicitly supported by the supplied passage. Never add medical knowledge or guess a missing subject from a sentence cut off at a chunk boundary.
3. Every proposition must be understandable by itself. Repeat the concrete entity name or descriptive noun phrase in every proposition.
4. Do not use anaphoric pronouns or vague references in a proposition. Prohibited forms include: it, its, they, them, their, he, him, his, she, her, hers, this, that, these, those, the former, the latter, and phrases such as "the above findings" or "these patients".
5. When the passage does not provide a proper name, construct a source-supported descriptive noun phrase. Example: use "the HBsAg-positive, anti-HBc-negative infection reported in Senegal," not "this infection."
6. Preserve negation, uncertainty, association rather than causation, population, condition, comparison, quantity, temporality, and study- or case-specific scope.
7. Preserve the epistemic source of a claim when relevant. For example, retain "the study suggests," "the guideline recommends," or "the case report describes" instead of converting the statement into a universal fact.
8. Recommendations are propositions about what the source recommends. Express a recommendation as "The guideline recommends ..." when the passage is a guideline, rather than presenting the recommendation as a universal biological fact.
9. Split independent claims into separate propositions. Avoid duplicates and paraphrases of the same proposition.
10. Extract all substantive propositions supported by the passage. The supplied passage contains at least one proposition. A truncated fragment with an unresolved subject must be skipped, but other valid content in the passage must still be extracted.
11. For auditing only, attach the shortest exact contiguous source span supporting each normalized proposition. The evidence quote may contain pronouns; the normalized proposition may not.

Return JSON only, in exactly this shape:
{"propositions":[{"claim":"clean standalone proposition","evidence":"exact quote from passage"}]}"""

# SHA-256 recorded explicitly so an accidental prompt edit fails preflight.
EXPECTED_SYSTEM_PROMPT_SHA256 = "78deee68b61543e89f8cea2db23b4d9c6b9e6b6cdbd742e4fcc504b212dc4e51"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--manual-audit-size", type=int, default=300)
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=0,
        help="Development-only prefix limit. Zero enforces the complete 44,911-document contract.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "calibrating"
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TwoLineProgress:
    """Pinned two-line TTY footer, or sparse two-line redirected snapshots."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self.estimates: list[float | None] = [None] * len(names)
        self.actual: list[float | None] = [None] * len(names)
        self.started = time.time()
        self.stage_started = self.started
        self.index = 0
        self.total = 1
        self.done = 0
        self.initial = 0
        self.unit = "item"
        self.last_render = 0.0
        self.tty = sys.stderr.isatty()
        self.overall_bar: Any | None = None
        self.stage_bar: Any | None = None
        if self.tty:
            from tqdm.auto import tqdm

            self.tqdm = tqdm
            self.overall_bar = tqdm(total=1000, position=0, leave=True, dynamic_ncols=True)

    def plan(self, details: Sequence[str]) -> None:
        self.log("[workflow plan] " + " | ".join(f"{i+1}/{len(details)} {x}" for i, x in enumerate(details)))

    def log(self, message: str) -> None:
        if self.tty and hasattr(self, "tqdm"):
            self.tqdm.write(message)
        else:
            print(message, flush=True)

    def start(self, index: int, total: int, *, initial: int = 0, unit: str = "item", estimate: float | None = None) -> None:
        if self.stage_bar is not None:
            self.stage_bar.close()
        self.index = index
        self.total = max(1, int(total))
        self.done = min(max(0, int(initial)), self.total)
        self.initial = self.done
        self.unit = unit
        self.stage_started = time.time()
        self.estimates[index - 1] = estimate
        if self.tty:
            self.stage_bar = self.tqdm(
                total=self.total,
                initial=self.done,
                position=1,
                leave=False,
                dynamic_ncols=True,
                unit=unit,
                desc=f"Stage {index}/{len(self.names)} - {self.names[index-1]}",
            )
        self.render(force=True)

    def _rate_eta(self) -> tuple[float | None, float | None]:
        elapsed = max(time.time() - self.stage_started, 1e-9)
        newly_done = max(0, self.done - self.initial)
        rate = newly_done / elapsed if newly_done else None
        if self.done >= self.total:
            return rate, 0.0
        if rate:
            return rate, (self.total - self.done) / rate
        estimate = self.estimates[self.index - 1]
        if estimate is None:
            return None, None
        return None, estimate * (1.0 - self.done / self.total)

    def _overall(self, stage_eta: float | None) -> tuple[float | None, float | None]:
        if stage_eta is None or any(value is None for value in self.estimates):
            return None, None
        weights = [float(value) for value in self.estimates if value is not None]
        completed = sum(weights[: self.index - 1])
        completed += weights[self.index - 1] * self.done / self.total
        future = sum(weights[self.index :])
        return completed / max(sum(weights), 1e-9), stage_eta + future

    def render(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render < (1.0 if self.tty else 30.0):
            return
        self.last_render = now
        rate, stage_eta = self._rate_eta()
        fraction, overall_eta = self._overall(stage_eta)
        elapsed = now - self.started
        pct = "unknown" if fraction is None else f"{100*fraction:.0f}%"
        overall_line = (
            f"Overall: {pct} [stage {self.index}/{len(self.names)}, "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(overall_eta)}]"
        )
        stage_pct = 100 * self.done / self.total
        rate_text = "calibrating" if rate is None else f"{rate:,.2f} {self.unit}/s"
        stage_line = (
            f"Stage {self.index}/{len(self.names)} - {self.names[self.index-1]}: "
            f"{stage_pct:.0f}% {self.done}/{self.total} "
            f"[{rate_text}, ETA {format_duration(stage_eta)}]"
        )
        if self.tty:
            assert self.overall_bar is not None
            self.overall_bar.n = 0 if fraction is None else int(min(1000, 1000 * fraction))
            self.overall_bar.bar_format = (
                overall_line if fraction is None else
                "Overall: {percentage:3.0f}%|{bar:10}| "
                f"[stage {self.index}/{len(self.names)}, elapsed {format_duration(elapsed)}, "
                f"ETA {format_duration(overall_eta)}]"
            )
            self.overall_bar.refresh()
            if self.stage_bar is not None:
                self.stage_bar.n = self.done
                self.stage_bar.bar_format = (
                    "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} "
                    f"[{rate_text}, ETA {format_duration(stage_eta)}]"
                )
                self.stage_bar.refresh()
        else:
            print(overall_line, flush=True)
            print(stage_line, flush=True)

    def update(self, amount: int = 1) -> None:
        self.done = min(self.total, self.done + int(amount))
        if self.stage_bar is not None:
            self.stage_bar.n = self.done
        # The first durable generation shard provides the only compatible
        # throughput measurement for the expensive stage.
        if self.index == 3 and self.done - self.initial >= 32:
            elapsed = max(1e-9, time.time() - self.stage_started)
            rate = (self.done - self.initial) / elapsed
            self.estimates[2] = self.total / max(rate, 1e-9)
            if self.estimates[3] is None:
                self.estimates[3] = max(10.0, 0.003 * self.total)
        self.render()

    def complete(self, detail: str) -> None:
        self.done = self.total
        self.render(force=True)
        duration = time.time() - self.stage_started
        self.actual[self.index - 1] = duration
        self.estimates[self.index - 1] = duration
        self.log(f"[stage {self.index}/{len(self.names)} complete | duration {format_duration(duration)}] {detail}")

    def close(self, *, force_complete: bool = False) -> None:
        if self.stage_bar is not None:
            self.stage_bar.close()
        if self.overall_bar is not None:
            if force_complete:
                self.overall_bar.n = self.overall_bar.total
            self.overall_bar.refresh()
            self.overall_bar.close()


def proposition_schema() -> dict[str, Any]:
    """Document the exact v2 output contract; this is not guided decoding."""
    return {
        "type": "object",
        "properties": {
            "propositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "minLength": 1},
                        "evidence": {"type": "string", "minLength": 1},
                    },
                    "required": ["claim", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["propositions"],
        "additionalProperties": False,
    }


def document_prompt(document: Mapping[str, Any]) -> str:
    # Exact Standalone-v2 user message: no source, title, question, label, or answer.
    return "SOURCE PASSAGE:\n" + str(document["text"])


def normalize_proposition(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip()).casefold()


def parse_model_output(raw_output: str, source_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    value_text = raw_output.strip()
    if value_text.startswith("```"):
        value_text = value_text.split("\n", 1)[1]
        value_text = value_text.rsplit("```", 1)[0]
    try:
        value = json.loads(value_text)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(value, dict):
        return None, ["output_not_object"]
    propositions = value.get("propositions")
    if not isinstance(propositions, list):
        return None, errors + ["propositions_not_array"]
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(propositions):
        prefix = f"proposition_{index}"
        if not isinstance(item, dict):
            errors.append(f"{prefix}_not_object")
            continue
        claim = str(item.get("claim") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not claim:
            errors.append(f"{prefix}_empty_claim")
            continue
        if not evidence:
            errors.append(f"{prefix}_empty_evidence")
            continue
        exact_evidence = evidence in source_text
        if not exact_evidence:
            errors.append(f"{prefix}_evidence_not_exact")
        normalized = normalize_proposition(claim)
        if normalized in seen:
            errors.append(f"{prefix}_duplicate")
            continue
        seen.add(normalized)
        parsed.append(
            {
                "claim": claim,
                "evidence": evidence,
                "evidence_exact_match": exact_evidence,
                "decontextualization_flag": bool(DEICTIC_RE.search(claim)),
            }
        )
    if not parsed:
        errors.append("empty_propositions")
    return {"propositions": parsed}, errors


def load_documents(path: Path, manifest_path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing fixed input or manifest: {path}, {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract") or {}
    input_summary = manifest.get("input_summary") or {}
    if contract.get("selection_sha256") != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Input KG selection hash does not match the fixed test Top-8 contract")
    if input_summary.get("unique_documents") != EXPECTED_DOCUMENTS:
        raise RuntimeError(f"Input manifest document count changed: {input_summary.get('unique_documents')}")
    documents = list(iter_jsonl(path))
    stable_ids = [str(row.get("stable_id") or "") for row in documents]
    if len(documents) != EXPECTED_DOCUMENTS or len(set(stable_ids)) != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_DOCUMENTS:,} unique documents; observed rows={len(documents):,} "
            f"unique_ids={len(set(stable_ids)):,}"
        )
    counts = dict(sorted(Counter(str(row.get("source")) for row in documents).items()))
    if counts != EXPECTED_BY_SOURCE:
        raise RuntimeError(f"Source counts changed: observed={counts} expected={EXPECTED_BY_SOURCE}")
    for row in documents:
        text = str(row.get("text") or "")
        if not text or sha256_text(text) != row.get("text_sha256"):
            raise RuntimeError(f"Document text contract failed: {row.get('stable_id')}")
    selection_hash = sha256_text("\n".join(stable_ids))
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise RuntimeError(f"Selected chunk ordering/identity changed: {selection_hash}")
    if limit > 0:
        documents = documents[:limit]
    return documents, manifest


def validate_prompt_contract() -> None:
    actual = sha256_text(SYSTEM_PROMPT)
    if actual != EXPECTED_SYSTEM_PROMPT_SHA256:
        raise RuntimeError(f"Standalone-v2 prompt changed: expected={EXPECTED_SYSTEM_PROMPT_SHA256} actual={actual}")
    parsed, errors = parse_model_output(
        '{"propositions":[{"claim":"Aspirin reduces fever.","evidence":"Aspirin reduces fever."}]}',
        "Aspirin reduces fever.",
    )
    if parsed is None or errors:
        raise RuntimeError(f"Standalone-v2 parser self-check failed: {errors}")


def model_identity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    if not path.is_dir() or not config.is_file():
        raise FileNotFoundError(f"Missing local Qwen model/config: {path}")
    return {
        "path": str(path.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_contract(args: argparse.Namespace, input_manifest: Mapping[str, Any], documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "json_schema": proposition_schema(),
        "input": file_identity(args.input),
        "input_manifest": file_identity(args.input_manifest),
        "input_contract_sha256": input_manifest.get("contract_sha256"),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "selected_documents": len(documents),
        "limit_documents": args.limit_documents,
        "model": model_identity(args.model),
        "generation": {
            "temperature": 0.0,
            "max_output_tokens": args.max_output_tokens,
            "max_model_len": args.max_model_len,
            "generation_batch_size": args.generation_batch_size,
            "shard_size": args.shard_size,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "thinking": False,
            "structured_decoding": False,
        },
        "uses_questions_answers_labels_or_behavior": False,
    }


def existing_cache(shard_dir: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not shard_dir.exists():
        return cached
    for path in sorted(shard_dir.glob("batch_*.jsonl")):
        try:
            rows = list(iter_jsonl(path))
        except (ValueError, OSError):
            continue
        ids = [str(row.get("stable_id") or "") for row in rows]
        if not rows or len(ids) != len(set(ids)) or any(value not in expected_ids for value in ids):
            continue
        if any(not str(row.get("raw_output") or "").strip() for row in rows):
            continue
        cached.update(zip(ids, rows))
    return cached


def init_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        disable_log_stats=True,
        runner="generate",
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        seed=args.seed,
        additional_config={"gdn_prefill_backend": "triton"},
    )
    return tokenizer, llm


def chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate(
    args: argparse.Namespace,
    documents: Sequence[dict[str, Any]],
    cached: dict[str, dict[str, Any]],
    progress: TwoLineProgress,
    tokenizer: Any,
    llm: Any,
) -> dict[str, dict[str, Any]]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_output_tokens,
    )
    shard_dir = args.output_dir / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    initial = len(cached)
    progress.start(3, len(documents), initial=initial, unit="document", estimate=None)
    for start in range(0, len(documents), args.shard_size):
        shard_documents = list(documents[start : start + args.shard_size])
        missing = [row for row in shard_documents if row["stable_id"] not in cached]
        if not missing:
            continue
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(missing), args.generation_batch_size):
            batch = missing[offset : offset + args.generation_batch_size]
            prompts: list[str] = []
            for document in batch:
                prompts.append(chat_prompt(tokenizer, document_prompt(document)))
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for document, output in zip(batch, outputs):
                raw = output.outputs[0].text.strip() if output.outputs else ""
                rows.append(
                    {
                        "stable_id": document["stable_id"],
                        "source": document["source"],
                        "text_sha256": document["text_sha256"],
                        "raw_output": raw,
                    }
                )
        path = shard_dir / f"batch_{start:07d}_{start + len(shard_documents):07d}.jsonl"
        atomic_jsonl(path, rows)
        for row in rows:
            cached[row["stable_id"]] = row
        progress.update(len(rows))
    missing_ids = [row["stable_id"] for row in documents if row["stable_id"] not in cached]
    if missing_ids:
        raise RuntimeError(f"Generation missing {len(missing_ids)} documents; first={missing_ids[:3]}")
    progress.complete(f"generated_or_reused={len(cached):,} shards={shard_dir}")
    return cached


def aggregate(
    args: argparse.Namespace,
    documents: Sequence[dict[str, Any]],
    cached: Mapping[str, Mapping[str, Any]],
    progress: TwoLineProgress,
) -> dict[str, Any]:
    progress.start(4, len(documents), unit="document", estimate=max(10.0, 0.003 * len(documents)))
    document_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    parse_status: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    proposition_counts: Counter[str] = Counter()
    source_documents: Counter[str] = Counter()
    source_propositions: Counter[str] = Counter()
    decontextualization_flags = 0
    exact_evidence_spans = 0
    for document in documents:
        stable_id = document["stable_id"]
        raw = str(cached[stable_id].get("raw_output") or "")
        parsed, errors = parse_model_output(raw, str(document["text"]))
        status = "parse_failure" if parsed is None else "parsed"
        parse_status[status] += 1
        source_documents[document["source"]] += 1
        for error in errors:
            error_counts[error] += 1
        propositions = [] if parsed is None else list(parsed["propositions"])
        proposition_counts[document["source"]] += len(propositions)
        source_propositions[document["source"]] += len(propositions)
        document_rows.append(
            {
                "stable_id": stable_id,
                "chunk_id": document.get("chunk_id"),
                "doc_id": document.get("doc_id"),
                "source": document["source"],
                "title": document.get("title"),
                "text_sha256": document["text_sha256"],
                "parse_status": status,
                "parse_errors": errors,
                "raw_output": raw,
                "propositions": propositions,
            }
        )
        for local_index, proposition in enumerate(propositions, start=1):
            decontextualization_flags += int(proposition["decontextualization_flag"])
            exact_evidence_spans += int(proposition["evidence_exact_match"])
            flat_rows.append(
                {
                    "proposition_id": f"{stable_id}::P{local_index}",
                    "stable_id": stable_id,
                    "chunk_id": document.get("chunk_id"),
                    "source": document["source"],
                    "text_sha256": document["text_sha256"],
                    **proposition,
                }
            )
        progress.update(1)

    atomic_jsonl(args.output_dir / "document_propositions.jsonl", document_rows)
    atomic_jsonl(args.output_dir / "propositions.jsonl", flat_rows)

    rng = random.Random(args.seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in document_rows:
        by_source[row["source"]].append(row)
    audit: list[dict[str, Any]] = []
    per_source = max(1, args.manual_audit_size // len(SOURCES))
    for source in SOURCES:
        rows = by_source[source]
        flagged = [row for row in rows if row["parse_errors"] or row["parse_status"] != "parsed"]
        ordinary = [row for row in rows if row not in flagged]
        chosen = rng.sample(flagged, min(len(flagged), per_source // 3))
        remaining = per_source - len(chosen)
        chosen.extend(rng.sample(ordinary, min(len(ordinary), remaining)))
        for row in chosen:
            document = next(item for item in documents if item["stable_id"] == row["stable_id"])
            audit.append({**row, "source_text": document["text"]})
    rng.shuffle(audit)
    atomic_jsonl(args.output_dir / "manual_audit_sample.jsonl", audit[: args.manual_audit_size])

    total_documents = len(document_rows)
    parsed_documents = total_documents - parse_status["parse_failure"]
    summary = {
        "run_version": RUN_VERSION,
        "documents": total_documents,
        "parsed_documents": parsed_documents,
        "structured_parse_rate": parsed_documents / max(1, total_documents),
        "parse_status": dict(sorted(parse_status.items())),
        "parse_error_counts": dict(error_counts.most_common()),
        "propositions": len(flat_rows),
        "mean_propositions_per_document": len(flat_rows) / max(1, total_documents),
        "exact_evidence_spans": exact_evidence_spans,
        "exact_evidence_rate": exact_evidence_spans / max(1, len(flat_rows)),
        "decontextualization_flagged_propositions": decontextualization_flags,
        "by_source": {
            source: {
                "documents": source_documents[source],
                "propositions": source_propositions[source],
                "mean_propositions_per_document": source_propositions[source] / max(1, source_documents[source]),
            }
            for source in SOURCES
        },
        "manual_audit": {
            "rows": min(len(audit), args.manual_audit_size),
            "path": str((args.output_dir / "manual_audit_sample.jsonl").resolve()),
            "note": "Structural validation cannot establish proposition entailment or medical correctness.",
        },
        "scope": {
            "uses_questions_answers_labels_or_behavior": False,
            "entity_linking_performed": False,
            "relation_extraction_performed": False,
            "qa_evaluation_performed": False,
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    progress.complete(
        f"documents={total_documents:,} propositions={len(flat_rows):,} "
        f"parse_rate={100*summary['structured_parse_rate']:.2f}% output={args.output_dir}"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.generation_batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("Batch and shard sizes must be positive")
    if args.generation_batch_size > args.shard_size:
        raise ValueError("--generation-batch-size cannot exceed --shard-size")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0,1)")
    if args.limit_documents < 0:
        raise ValueError("--limit-documents must be non-negative")

    progress = TwoLineProgress(
        [
            "validate fixed Top-8 chunk contract",
            "validate exact Standalone-v2 prompt and load Qwen",
            "generate with exact Standalone-v2 contract",
            "parse claim/evidence outputs and write audit artifacts",
        ]
    )
    progress.plan(
        [
            "44,911 document-only chunks; no question/answer/labels",
            "Qwen3.5-9B + recovered v2 prompt; temperature=0; no guided decoding",
            "atomic resumable shards; ETA calibrated from first shard",
            "document/flat proposition caches + 300-row manual audit",
        ]
    )
    try:
        progress.start(1, EXPECTED_DOCUMENTS, unit="document", estimate=5.0)
        documents, input_manifest = load_documents(
            args.input,
            args.input_manifest,
            args.limit_documents,
        )
        progress.done = EXPECTED_DOCUMENTS
        progress.complete(
            f"selected={len(documents):,} complete_contract={args.limit_documents == 0} input={args.input}"
        )

        progress.start(2, 2 if args.preflight_only else 3, unit="check", estimate=30.0)
        validate_prompt_contract()
        progress.update(1)
        identity = model_identity(args.model)
        progress.update(1)
        tokenizer = None
        llm = None
        if not args.preflight_only:
            tokenizer, llm = init_llm(args)
            progress.update(1)
        progress.complete(f"prompt=exact-v2 parser=valid model={identity['path']}")

        contract = build_contract(args, input_manifest, documents)
        contract_hash = sha256_text(canonical_json(contract))
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract_sha256") != contract_hash:
                raise RuntimeError(
                    f"Proposition cache contract mismatch at {manifest_path}; use a new versioned output directory"
                )
        manifest = {
            "status": "prepared",
            "contract_sha256": contract_hash,
            "contract": contract,
            "hypothesis": (
                "A document-only Qwen transformation can recover standalone, source-indexed proposition "
                "candidates from the exact RAG2 test Top-8 chunk union."
            ),
            "scope_limit": (
                "This stage does not validate entailment, build graph edges, use test labels, or measure QA gains."
            ),
            "reproducibility": {
                "git_commit": git_commit(),
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
            },
        }
        if args.preflight_only:
            progress.log(
                f"[preflight complete] documents={len(documents):,} contract={contract_hash} "
                f"output={args.output_dir}"
            )
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)

        complete_path = args.output_dir / "COMPLETE.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if complete.get("contract_sha256") == contract_hash:
                progress.log(f"[already complete] output={args.output_dir}")
                return

        expected_ids = {row["stable_id"] for row in documents}
        cached = existing_cache(args.output_dir / "generation_shards", expected_ids)
        assert tokenizer is not None and llm is not None
        cached = generate(args, documents, cached, progress, tokenizer, llm)
        summary = aggregate(args, documents, cached, progress)
        manifest["status"] = "complete"
        manifest["summary_path"] = str((args.output_dir / "summary.json").resolve())
        atomic_json(manifest_path, manifest)
        atomic_json(
            complete_path,
            {
                "contract_sha256": contract_hash,
                "documents": len(documents),
                "propositions": summary["propositions"],
            },
        )
        progress.close(force_complete=True)
        print(
            f"[workflow complete] documents={len(documents):,} propositions={summary['propositions']:,} "
            f"output={args.output_dir}",
            flush=True,
        )
    except BaseException as exc:
        progress.log(
            f"[workflow FAILED] stage={progress.index}/{len(progress.names)} completed={progress.done}/{progress.total} "
            f"error={type(exc).__name__}: {exc}; rerun the identical command to resume durable shards"
        )
        progress.close()
        raise


if __name__ == "__main__":
    main()
